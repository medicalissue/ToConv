"""Training Loop for Vision Token Compression"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Optional, Dict
import os

from .models import CLIPVisionEncoder, TokenCompressor, Discriminator, AutoEncoderDecoder
from .losses import WGANGPLoss, compute_gradient_penalty, AutoEncoderLoss


class TokenCompressionTrainer:
    """
    Trainer for vision token compression system.
    Combines WGAN-GP and AutoEncoder losses.
    """

    def __init__(
        self,
        clip_encoder: CLIPVisionEncoder,
        compressor: TokenCompressor,
        discriminator: Discriminator,
        ae_decoder: AutoEncoderDecoder,
        wgan_loss: WGANGPLoss,
        ae_loss: AutoEncoderLoss,
        device: torch.device,
        learning_rate: float = 1e-4,
        discriminator_lr: float = 1e-4,
        ae_weight: float = 1.0,
        wgan_weight: float = 1.0,
        n_critic: int = 5,
        use_wandb: bool = True
    ):
        """
        Args:
            clip_encoder: Frozen CLIP vision encoder
            compressor: Token compressor network
            discriminator: WGAN-GP discriminator
            ae_decoder: AutoEncoder decoder
            wgan_loss: WGAN-GP loss function
            ae_loss: AutoEncoder loss function
            device: Device to train on
            learning_rate: Learning rate for compressor and decoder
            discriminator_lr: Learning rate for discriminator
            ae_weight: Weight for autoencoder loss
            wgan_weight: Weight for WGAN-GP loss
            n_critic: Number of discriminator updates per generator update
            use_wandb: Whether to log to Weights & Biases
        """
        self.clip_encoder = clip_encoder.to(device)
        self.compressor = compressor.to(device)
        self.discriminator = discriminator.to(device)
        self.ae_decoder = ae_decoder.to(device)

        self.wgan_loss = wgan_loss
        self.ae_loss = ae_loss

        self.device = device
        self.ae_weight = ae_weight
        self.wgan_weight = wgan_weight
        self.n_critic = n_critic
        self.use_wandb = use_wandb

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(compressor.parameters()) + list(ae_decoder.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.999)
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.compressor.train()
        self.discriminator.train()
        self.ae_decoder.train()
        self.clip_encoder.eval()  # Always keep CLIP frozen

        metrics = {
            'disc_loss': 0.0,
            'gen_loss': 0.0,
            'ae_loss': 0.0,
            'total_loss': 0.0,
            'wasserstein_distance': 0.0,
            'gradient_penalty': 0.0
        }

        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Extract CLIP tokens (no gradients)
            with torch.no_grad():
                original_tokens = self.clip_encoder(images)

            # --- Train Discriminator ---
            for _ in range(self.n_critic):
                self.optimizer_D.zero_grad()

                # Compress tokens
                with torch.no_grad():
                    compressed_tokens = self.compressor(original_tokens)

                # Discriminator scores
                real_scores = self.discriminator(original_tokens)
                fake_scores = self.discriminator(compressed_tokens)

                # Gradient penalty
                gp = compute_gradient_penalty(
                    self.discriminator,
                    original_tokens,
                    compressed_tokens,
                    self.device
                )

                # Discriminator loss
                disc_loss, disc_info = self.wgan_loss.discriminator_loss(
                    real_scores, fake_scores, gp
                )

                disc_loss.backward()
                self.optimizer_D.step()

            # --- Train Generator (Compressor + Decoder) ---
            self.optimizer_G.zero_grad()

            # Compress tokens
            compressed_tokens = self.compressor(original_tokens)

            # WGAN-GP generator loss
            fake_scores = self.discriminator(compressed_tokens)
            gen_loss, gen_info = self.wgan_loss.generator_loss(fake_scores)

            # AutoEncoder reconstruction loss
            reconstructed_tokens = self.ae_decoder(compressed_tokens)
            ae_loss, ae_info = self.ae_loss(reconstructed_tokens, original_tokens)

            # Total generator loss
            total_loss = (
                self.wgan_weight * gen_loss +
                self.ae_weight * ae_loss
            )

            total_loss.backward()
            self.optimizer_G.step()

            # Update metrics
            metrics['disc_loss'] += disc_info['disc_loss']
            metrics['gen_loss'] += gen_info['gen_loss']
            metrics['ae_loss'] += ae_info['ae_loss']
            metrics['total_loss'] += total_loss.item()
            metrics['wasserstein_distance'] += disc_info['wasserstein_distance']
            metrics['gradient_penalty'] += disc_info['gradient_penalty']

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/disc_loss': disc_info['disc_loss'],
                    'train/gen_loss': gen_info['gen_loss'],
                    'train/ae_loss': ae_info['ae_loss'],
                    'train/total_loss': total_loss.item(),
                    'train/wasserstein_distance': disc_info['wasserstein_distance'],
                    'train/gradient_penalty': disc_info['gradient_penalty'],
                    'train/cosine_similarity': ae_info.get('cosine_similarity', 0.0),
                    'global_step': self.global_step
                })

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'D_loss': f"{disc_info['disc_loss']:.4f}",
                'G_loss': f"{gen_info['gen_loss']:.4f}",
                'AE_loss': f"{ae_info['ae_loss']:.4f}",
                'W_dist': f"{disc_info['wasserstein_distance']:.4f}"
            })

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.compressor.eval()
        self.discriminator.eval()
        self.ae_decoder.eval()

        metrics = {
            'val_ae_loss': 0.0,
            'val_cosine_similarity': 0.0
        }

        num_batches = len(val_loader)
        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        for images, _ in pbar:
            images = images.to(self.device)

            # Extract CLIP tokens
            original_tokens = self.clip_encoder(images)

            # Compress and reconstruct
            compressed_tokens = self.compressor(original_tokens)
            reconstructed_tokens = self.ae_decoder(compressed_tokens)

            # Compute reconstruction loss
            ae_loss, ae_info = self.ae_loss(reconstructed_tokens, original_tokens)

            metrics['val_ae_loss'] += ae_info['ae_loss']
            metrics['val_cosine_similarity'] += ae_info.get('cosine_similarity', 0.0)

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/ae_loss': metrics['val_ae_loss'],
                'val/cosine_similarity': metrics['val_cosine_similarity'],
                'epoch': epoch
            })

        return metrics

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'compressor_state_dict': self.compressor.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'ae_decoder_state_dict': self.ae_decoder.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }

        # Save regular checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save latest
        latest_path = save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.ae_decoder.load_state_dict(checkpoint['ae_decoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


if __name__ == "__main__":
    # Test the trainer
    print("Testing TokenCompressionTrainer...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create models
    clip_encoder = CLIPVisionEncoder(
        model_name="openai/clip-vit-large-patch14-336",
        freeze=True
    )

    grid_size = clip_encoder.get_grid_size()
    hidden_dim = clip_encoder.get_hidden_size()
    output_grid_size = 6

    compressor = TokenCompressor(
        input_grid_size=grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim
    )

    discriminator = Discriminator(
        num_tokens=output_grid_size ** 2,
        hidden_dim=hidden_dim
    )

    ae_decoder = AutoEncoderDecoder(
        compressed_grid_size=output_grid_size,
        original_grid_size=grid_size,
        hidden_dim=hidden_dim
    )

    # Create losses
    wgan_loss = WGANGPLoss(lambda_gp=10.0)
    ae_loss = AutoEncoderLoss(loss_type='hybrid')

    # Create trainer
    trainer = TokenCompressionTrainer(
        clip_encoder=clip_encoder,
        compressor=compressor,
        discriminator=discriminator,
        ae_decoder=ae_decoder,
        wgan_loss=wgan_loss,
        ae_loss=ae_loss,
        device=device,
        use_wandb=False
    )

    print("✓ Trainer created successfully!")
