"""
RF-based Vision Token Compression Training
Simplified, GPU-optimized training with detailed logging
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

from vision_token_compression.models import (
    CLIPVisionEncoder,
    TokenCompressor
)
from vision_token_compression.losses import SinkhornOTLoss, RFCosineSimilarityLoss
from vision_token_compression.data import create_imagenet_dataloaders


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RFTokenCompressionTrainer:
    """Simplified GPU-optimized trainer for RF-based token compression"""

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        steps_per_epoch: int = None
    ):
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.use_wandb = cfg.experiment.use_wandb

        # Build models
        print("\n" + "="*80)
        print("Building Models...")
        print("="*80)

        # CLIP encoder (frozen)
        self.clip_encoder = CLIPVisionEncoder(
            model_name=cfg.model.clip.model_name,
            freeze=cfg.model.clip.freeze
        ).to(device)
        self.clip_encoder.eval()

        # Apply torch.compile if enabled
        if cfg.hardware.compile:
            print(f"\n✓ Applying torch.compile to CLIP encoder...")
            self.clip_encoder = torch.compile(self.clip_encoder)

        hidden_dim = self.clip_encoder.get_hidden_size()

        # Token compressor
        self.compressor = TokenCompressor(
            input_grid_size=cfg.compression.input_grid_size,
            output_grid_size=cfg.compression.output_grid_size,
            hidden_dim=hidden_dim
        ).to(device)

        # Apply torch.compile if enabled
        if cfg.hardware.compile:
            print(f"\n✓ Applying torch.compile to compressor...")
            self.compressor = torch.compile(self.compressor)

        # Print model info
        rf_size = self.compressor.get_receptive_field_size()
        input_size = cfg.compression.input_grid_size
        output_size = cfg.compression.output_grid_size

        print(f"\n✓ CLIP Encoder: {cfg.model.clip.model_name}")
        print(f"  - Grid: {input_size}×{input_size} → {output_size}×{output_size}")
        print(f"  - Compression: {(input_size**2) / (output_size**2):.1f}x")
        print(f"  - Hidden dim: {hidden_dim}")

        print(f"\n✓ Token Compressor [{self.compressor.config_name}]")
        print(f"  - Architecture: Single Conv (no bottleneck)")
        print(f"  - Theoretical RF: {rf_size}×{rf_size}")
        print(f"  - Params: {sum(p.numel() for p in self.compressor.parameters())/1e6:.2f}M")

        total_params = sum(p.numel() for p in self.compressor.parameters())
        print(f"\n✓ Total trainable params: {total_params/1e6:.2f}M")

        # Build losses
        self.ot_loss = SinkhornOTLoss(
            epsilon=cfg.loss.sinkhorn_ot.epsilon,
            max_iter=cfg.loss.sinkhorn_ot.max_iter,
            threshold=cfg.loss.sinkhorn_ot.threshold,
            normalize=cfg.loss.sinkhorn_ot.normalize
        )
        # Pass actual convolution parameters to RF cosine loss
        self.cosine_loss = RFCosineSimilarityLoss(
            kernel_size=self.compressor.kernel_size,
            stride=self.compressor.stride,
            padding=self.compressor.padding
        )

        self.ot_weight = cfg.loss.weights.sinkhorn_ot
        self.cosine_weight = cfg.loss.weights.cosine

        # Optimizer (AdamW for better regularization)
        self.optimizer = torch.optim.AdamW(
            self.compressor.parameters(),
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.training.optimizer.betas),
            weight_decay=cfg.training.optimizer.weight_decay,
            eps=cfg.training.optimizer.eps
        )

        # Learning rate scheduler
        self.use_scheduler = cfg.training.scheduler.use
        if self.use_scheduler:
            self.scheduler = self._build_scheduler(cfg)
        else:
            self.scheduler = None

        # Mixed precision training
        self.use_amp = cfg.hardware.mixed_precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Grid sizes for loss computation
        self.compressed_grid_size = cfg.compression.output_grid_size
        self.original_grid_size = cfg.compression.input_grid_size

    def _build_scheduler(self, cfg: DictConfig):
        """Build learning rate scheduler with warmup (supports epoch or step-based warmup)"""
        if cfg.training.scheduler.type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            min_lr = cfg.training.scheduler.min_lr
            base_lr = cfg.training.learning_rate
            total_epochs = cfg.training.epochs

            # Determine warmup unit (epoch or step)
            warmup_unit = cfg.training.scheduler.get('warmup_unit', 'epoch')

            if warmup_unit == 'step':
                # Step-based warmup
                if self.steps_per_epoch is None:
                    raise ValueError("steps_per_epoch must be provided for step-based warmup")

                warmup_steps = cfg.training.scheduler.warmup_steps
                total_steps = self.steps_per_epoch * total_epochs

                # Warmup scheduler: linear warmup (step-based)
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1e-3,  # Start from 0.001 * base_lr
                    end_factor=1.0,     # End at base_lr
                    total_iters=warmup_steps
                )

                # Cosine annealing scheduler (step-based)
                cosine_steps = total_steps - warmup_steps
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_steps,
                    eta_min=min_lr
                )

                # Combine warmup + cosine
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )

                # Store scheduler type for step() logic
                scheduler.warmup_unit = 'step'

            else:
                # Epoch-based warmup (original behavior)
                warmup_epochs = cfg.training.scheduler.warmup_epochs

                # Warmup scheduler: linear warmup (epoch-based)
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1e-3,  # Start from 0.001 * base_lr
                    end_factor=1.0,     # End at base_lr
                    total_iters=warmup_epochs
                )

                # Cosine annealing scheduler (epoch-based)
                cosine_epochs = total_epochs - warmup_epochs
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_epochs,
                    eta_min=min_lr
                )

                # Combine warmup + cosine
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )

                # Store scheduler type for step() logic
                scheduler.warmup_unit = 'epoch'

            return scheduler
        else:
            raise ValueError(f"Unsupported scheduler type: {cfg.training.scheduler.type}")

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.compressor.train()
        self.clip_encoder.eval()

        # Accumulators
        metrics = {
            'ot_loss': 0.0,
            'cosine_loss': 0.0,
            'total_loss': 0.0,
            'mean_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0
        }

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.training.epochs}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)

            # Extract CLIP tokens (no gradient)
            with torch.no_grad():
                original_tokens = self.clip_encoder(images)

            # ============================================
            # Train Compressor
            # ============================================
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                # Compress tokens
                compressed_tokens = self.compressor(original_tokens)

                # Sinkhorn OT loss
                ot_loss, ot_info = self.ot_loss(
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens
                )

                # Cosine similarity loss (compute stats only when logging)
                should_log = (self.global_step % self.cfg.logging.log_frequency == 0)
                cosine_loss, cosine_info = self.cosine_loss(
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens,
                    compressed_grid_size=(self.compressed_grid_size, self.compressed_grid_size),
                    original_grid_size=(self.original_grid_size, self.original_grid_size),
                    compute_stats=should_log
                )

                # Combined loss
                total_loss = self.ot_weight * ot_loss + self.cosine_weight * cosine_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ============================================
            # Update metrics
            # ============================================
            metrics['ot_loss'] += ot_info['ot_loss']
            metrics['cosine_loss'] += cosine_info['cosine_sim_loss']
            metrics['total_loss'] += total_loss.item()
            metrics['mean_similarity'] += cosine_info['mean_similarity']
            metrics['min_similarity'] += cosine_info['min_similarity']
            metrics['max_similarity'] += cosine_info['max_similarity']

            # ============================================
            # Logging
            # ============================================
            if self.use_wandb and should_log:
                log_dict = {
                    # Sinkhorn OT loss
                    'train/ot_loss': ot_info['ot_loss'],
                    'train/ot_distance': ot_info['ot_distance'],
                    'train/sinkhorn_iterations': ot_info['sinkhorn_iterations'],

                    # Cosine similarity loss
                    'train/cosine_loss': cosine_info['cosine_sim_loss'],
                    'train/total_loss': total_loss.item(),

                    # Cosine similarity statistics
                    'train/mean_similarity': cosine_info['mean_similarity'],
                    'train/min_similarity': cosine_info['min_similarity'],
                    'train/max_similarity': cosine_info['max_similarity'],
                    'train/std_similarity': cosine_info['std_similarity'],

                    # Learning rate
                    'train/lr': self.optimizer.param_groups[0]['lr'],

                    # Training progress
                    'global_step': self.global_step,
                    'epoch': epoch
                }

                wandb.log(log_dict, step=self.global_step)

            # Update progress bar
            pbar.set_postfix({
                'OT': f"{ot_info['ot_loss']:.4f}",
                'Cos': f"{cosine_info['cosine_sim_loss']:.3f}",
                'Sim': f"{cosine_info['mean_similarity']:.3f}",
                'Total': f"{total_loss.item():.3f}"
            })

            # Step scheduler if using step-based warmup
            if self.use_scheduler and hasattr(self.scheduler, 'warmup_unit') and self.scheduler.warmup_unit == 'step':
                self.scheduler.step()

            self.global_step += 1

        # Average metrics
        num_batches = len(train_loader)
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.compressor.eval()

        metrics = {
            'val_cosine_loss': 0.0,
            'val_mean_similarity': 0.0,
            'val_min_similarity': 0.0,
            'val_max_similarity': 0.0,
            'val_std_similarity': 0.0
        }

        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        for images, _ in pbar:
            images = images.to(self.device, non_blocking=True)

            # Extract and compress
            original_tokens = self.clip_encoder(images)
            compressed_tokens = self.compressor(original_tokens)

            # Compute cosine similarity
            cosine_loss, cosine_info = self.cosine_loss(
                compressed_tokens,
                original_tokens,
                (self.compressed_grid_size, self.compressed_grid_size),
                (self.original_grid_size, self.original_grid_size)
            )

            metrics['val_cosine_loss'] += cosine_info['cosine_sim_loss']
            metrics['val_mean_similarity'] += cosine_info['mean_similarity']
            metrics['val_min_similarity'] += cosine_info['min_similarity']
            metrics['val_max_similarity'] += cosine_info['max_similarity']
            metrics['val_std_similarity'] += cosine_info['std_similarity']

            pbar.set_postfix({
                'Sim': f"{cosine_info['mean_similarity']:.3f}",
                'Loss': f"{cosine_info['cosine_sim_loss']:.3f}"
            })

        # Average metrics
        num_batches = len(val_loader)
        for key in metrics:
            metrics[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/cosine_loss': metrics['val_cosine_loss'],
                'val/mean_similarity': metrics['val_mean_similarity'],
                'val/min_similarity': metrics['val_min_similarity'],
                'val/max_similarity': metrics['val_max_similarity'],
                'val/std_similarity': metrics['val_std_similarity'],
                'epoch': epoch
            }, step=self.global_step)

        return metrics

    def save_checkpoint(self, save_dir: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'compressor_state_dict': self.compressor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }

        # Save regular checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch})")

        # Save latest
        latest_path = save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)


@hydra.main(version_base=None, config_path="vision_token_compression/configs", config_name="rf_config")
def main(cfg: DictConfig):
    """Main training function"""

    print("\n" + "="*80)
    print("RF-BASED VISION TOKEN COMPRESSION")
    print("="*80)
    print(f"\nExperiment: {cfg.experiment.name}")
    print(f"Seed: {cfg.experiment.seed}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cfg.hardware.cuda_device}')
        # Initialize CUDA context early to avoid warnings
        torch.cuda.set_device(device)
        torch.cuda.init()  # Explicitly initialize CUDA
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\n⚠️  Using CPU")

    # Initialize wandb
    if cfg.experiment.use_wandb:
        wandb.init(
            project=cfg.experiment.wandb_project,
            entity=cfg.experiment.wandb_entity,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=['rf-compression', 'clip-vit']
        )
        print(f"\n✓ WandB initialized: {cfg.experiment.wandb_project}/{cfg.experiment.name}")

    # Create data loaders
    print("\n" + "="*80)
    print("Loading Data...")
    print("="*80)

    train_loader, val_loader = create_imagenet_dataloaders(
        root=cfg.data.imagenet_root,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.clip.image_size,
        pin_memory=cfg.data.pin_memory,
        use_subset=cfg.data.use_subset,
        subset_size=cfg.data.subset_size
    )

    print(f"\n✓ Train: {len(train_loader)} batches")
    print(f"✓ Val: {len(val_loader)} batches")
    print(f"✓ Batch size: {cfg.training.batch_size}")

    # Create trainer
    trainer = RFTokenCompressionTrainer(cfg, device, steps_per_epoch=len(train_loader))

    # Training loop
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)

    save_dir = Path(cfg.checkpoint.save_dir)

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{cfg.training.epochs}")
        print(f"{'='*80}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Step scheduler (only for epoch-based warmup; step-based is called in train_epoch)
        if trainer.use_scheduler and hasattr(trainer.scheduler, 'warmup_unit') and trainer.scheduler.warmup_unit == 'epoch':
            trainer.scheduler.step()

        # Print current learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"\n✓ Learning Rate: {current_lr:.2e}")

        print(f"\n✓ Train Metrics:")
        print(f"  OT Loss: {train_metrics['ot_loss']:.4f} | Cosine: {train_metrics['cosine_loss']:.4f}")
        print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Similarity: {train_metrics['mean_similarity']:.4f} (min: {train_metrics['min_similarity']:.4f}, max: {train_metrics['max_similarity']:.4f})")

        # Validate
        if epoch % cfg.validation.frequency == 0:
            val_metrics = trainer.validate(val_loader, epoch)

            print(f"\n✓ Validation Metrics:")
            print(f"  Cosine Loss: {val_metrics['val_cosine_loss']:.4f}")
            print(f"  Similarity: {val_metrics['val_mean_similarity']:.4f} ± {val_metrics['val_std_similarity']:.4f}")

            # Save best model
            is_best = val_metrics['val_cosine_loss'] < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_metrics['val_cosine_loss']

            trainer.save_checkpoint(save_dir, epoch, is_best=is_best)

        # Save checkpoint periodically
        if epoch % cfg.checkpoint.save_frequency == 0:
            trainer.save_checkpoint(save_dir, epoch, is_best=False)

    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")

    if cfg.experiment.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
