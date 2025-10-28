"""Main training script for RF-based Vision Token Compression"""
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
import random
import numpy as np

from vision_token_compression.models import (
    CLIPVisionEncoder,
    TokenCompressor,
    RFDiscriminator
)
from vision_token_compression.losses import RFWGANGPLoss, RFCosineSimilarityLoss
from vision_token_compression.data import create_imagenet_dataloaders
from vision_token_compression.rf_trainer import RFTokenCompressionTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(cuda_device: int) -> torch.device:
    """Setup CUDA device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        device = torch.device(f'cuda:{cuda_device}')
        print(f"Using CUDA device: {cuda_device}")
        print(f"Device name: {torch.cuda.get_device_name(cuda_device)}")
        print(f"Device memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    return device


def create_models(cfg: DictConfig, device: torch.device):
    """Create all models for RF-based training"""
    print("\n" + "=" * 80)
    print("Creating RF-based models...")
    print("=" * 80)

    # CLIP encoder
    print(f"\n1. CLIP Vision Encoder: {cfg.model.clip.model_name}")
    clip_encoder = CLIPVisionEncoder(
        model_name=cfg.model.clip.model_name,
        freeze=cfg.model.clip.freeze
    )

    grid_size = clip_encoder.get_grid_size()
    hidden_dim = clip_encoder.get_hidden_size()
    output_grid_size = cfg.model.compressor.output_grid_size
    rf_size = grid_size // output_grid_size

    print(f"   - Input grid size: {grid_size}×{grid_size} ({grid_size**2} tokens)")
    print(f"   - Hidden dimension: {hidden_dim}")
    print(f"   - Output grid size: {output_grid_size}×{output_grid_size} ({output_grid_size**2} tokens)")
    print(f"   - Compression ratio: {(grid_size**2) / (output_grid_size**2):.2f}x")
    print(f"   - RF size: {rf_size}×{rf_size} ({rf_size**2} tokens per RF)")

    # Token compressor
    print(f"\n2. Token Compressor")
    compressor = TokenCompressor(
        input_grid_size=grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim,
        num_layers=cfg.model.compressor.num_layers,
        use_residual=cfg.model.compressor.use_residual,
        use_layer_norm=cfg.model.compressor.use_layer_norm
    )
    print(f"   - Parameters: {sum(p.numel() for p in compressor.parameters()) / 1e6:.2f}M")

    # RF Discriminator
    print(f"\n3. RF Discriminator (token-level)")
    rf_discriminator = RFDiscriminator(
        hidden_dim=hidden_dim,
        num_layers=cfg.model.rf_discriminator.num_layers,
        mlp_ratio=cfg.model.rf_discriminator.mlp_ratio,
        dropout=cfg.model.rf_discriminator.dropout
    )
    print(f"   - Compares: 1 compressed token vs 1 sampled RF token")
    print(f"   - Applied to: {output_grid_size**2} token pairs per batch")
    print(f"   - Parameters: {sum(p.numel() for p in rf_discriminator.parameters()) / 1e6:.2f}M")

    # Move to device
    clip_encoder = clip_encoder.to(device)
    compressor = compressor.to(device)
    rf_discriminator = rf_discriminator.to(device)

    total_params = (
        sum(p.numel() for p in compressor.parameters()) +
        sum(p.numel() for p in rf_discriminator.parameters())
    )
    print(f"\nTotal trainable parameters: {total_params / 1e6:.2f}M")

    return clip_encoder, compressor, rf_discriminator


@hydra.main(version_base=None, config_path="vision_token_compression/configs", config_name="rf_config")
def main(cfg: DictConfig):
    """Main training function for RF-based system"""

    print("\n" + "=" * 80)
    print("RF-BASED VISION TOKEN COMPRESSION TRAINING")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.experiment.seed)
    print(f"\nRandom seed set to: {cfg.experiment.seed}")

    # Setup device
    device = setup_device(cfg.hardware.cuda_device)

    # Initialize wandb
    if cfg.experiment.use_wandb:
        wandb.init(
            project=cfg.experiment.wandb_project,
            entity=cfg.experiment.wandb_entity,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"\nWandB initialized: {cfg.experiment.wandb_project}/{cfg.experiment.name}")

    # Create models
    clip_encoder, compressor, rf_discriminator = create_models(cfg, device)

    # Create losses
    print("\n" + "=" * 80)
    print("Creating RF-based loss functions...")
    print("=" * 80)

    rf_wgan_loss = RFWGANGPLoss(lambda_gp=cfg.loss.rf_wgan.lambda_gp)
    rf_cosine_loss = RFCosineSimilarityLoss(
        temperature=cfg.loss.rf_cosine_similarity.temperature
    )

    print(f"\n1. RF WGAN-GP Loss")
    print(f"   - Gradient penalty weight: {cfg.loss.rf_wgan.lambda_gp}")
    print(f"   - Loss weight: {cfg.loss.weights.wgan}")
    print(f"   - Comparison: 36 compressed vs 36 sampled RF tokens (1:1)")

    print(f"\n2. RF Cosine Similarity Loss")
    print(f"   - Temperature: {cfg.loss.rf_cosine_similarity.temperature}")
    print(f"   - Loss weight: {cfg.loss.weights.cosine}")
    print(f"   - Goal: Maximize similarity between compressed tokens and their RF")
    print(f"   - Comparison: Each compressed token vs all {cfg.grid.rf_size}×{cfg.grid.rf_size} tokens in its RF")

    # Create data loaders
    print("\n" + "=" * 80)
    print("Creating data loaders...")
    print("=" * 80)

    train_loader, val_loader = create_imagenet_dataloaders(
        root=cfg.data.imagenet_root,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.clip.image_size,
        pin_memory=cfg.data.pin_memory,
        use_subset=cfg.data.use_subset,
        subset_size=cfg.data.subset_size
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Total training samples: {len(train_loader) * cfg.training.batch_size}")

    # Create trainer
    print("\n" + "=" * 80)
    print("Creating RF-based trainer...")
    print("=" * 80)

    trainer = RFTokenCompressionTrainer(
        clip_encoder=clip_encoder,
        compressor=compressor,
        rf_discriminator=rf_discriminator,
        rf_wgan_loss=rf_wgan_loss,
        rf_cosine_loss=rf_cosine_loss,
        device=device,
        compressed_grid_size=cfg.grid.compressed_grid_size,
        original_grid_size=cfg.grid.original_grid_size,
        learning_rate=cfg.training.learning_rate,
        discriminator_lr=cfg.training.discriminator_lr,
        cosine_weight=cfg.loss.weights.cosine,
        wgan_weight=cfg.loss.weights.wgan,
        n_critic=cfg.training.n_critic,
        use_wandb=cfg.experiment.use_wandb
    )

    print(f"\nLearning rate (Generator): {cfg.training.learning_rate}")
    print(f"Learning rate (Discriminator): {cfg.training.discriminator_lr}")
    print(f"Critic iterations: {cfg.training.n_critic}")
    print(f"Training approach: RF-based (1:1 token comparison)")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting RF-based training...")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{cfg.training.epochs}")
        print(f"{'=' * 80}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        print(f"\nTrain Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Validate
        if epoch % cfg.validation.frequency == 0:
            val_metrics = trainer.validate(
                val_loader,
                epoch,
                save_visualizations=cfg.validation.save_rf_visualizations
            )

            print(f"\nValidation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save best model (use cosine loss - minimize it, which maximizes similarity)
            if val_metrics['val_cosine_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_cosine_loss']
                trainer.save_checkpoint(
                    save_dir=cfg.checkpoint.save_dir,
                    epoch=epoch,
                    is_best=True
                )

        # Save checkpoint
        if epoch % cfg.checkpoint.save_frequency == 0:
            trainer.save_checkpoint(
                save_dir=cfg.checkpoint.save_dir,
                epoch=epoch,
                is_best=False
            )

    print("\n" + "=" * 80)
    print("RF-based training completed!")
    print("=" * 80)

    if cfg.experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
