"""RF-based Training Loop for Vision Token Compression"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Optional, Dict
import os

from .models import CLIPVisionEncoder, TokenCompressor
from .models.rf_discriminator import RFDiscriminator
from .losses.rf_wgan_gp import RFWGANGPLoss
from .losses.rf_cosine_similarity_loss import RFCosineSimilarityLoss


class RFTokenCompressionTrainer:
    """
    Trainer for RF-based vision token compression system.
    Combines RF-aware WGAN-GP and Cosine Similarity losses.
    """

    def __init__(
        self,
        clip_encoder: CLIPVisionEncoder,
        compressor: TokenCompressor,
        rf_discriminator: RFDiscriminator,
        rf_wgan_loss: RFWGANGPLoss,
        rf_cosine_loss: RFCosineSimilarityLoss,
        device: torch.device,
        compressed_grid_size: int = 6,
        original_grid_size: int = 24,
        learning_rate: float = 1e-4,
        discriminator_lr: float = 1e-4,
        cosine_weight: float = 1.0,
        wgan_weight: float = 1.0,
        n_critic: int = 5,
        use_wandb: bool = True
    ):
        """
        Args:
            clip_encoder: Frozen CLIP vision encoder
            compressor: Token compressor network
            rf_discriminator: RF-aware discriminator
            rf_wgan_loss: RF WGAN-GP loss function
            rf_cosine_loss: RF Cosine Similarity loss function
            device: Device to train on
            compressed_grid_size: 6 (6x6 grid)
            original_grid_size: 24 (24x24 grid)
            learning_rate: Learning rate for compressor
            discriminator_lr: Learning rate for discriminator
            cosine_weight: Weight for cosine similarity loss
            wgan_weight: Weight for WGAN-GP loss
            n_critic: Number of discriminator updates per generator update
            use_wandb: Whether to log to Weights & Biases
        """
        self.clip_encoder = clip_encoder.to(device)
        self.compressor = compressor.to(device)
        self.rf_discriminator = rf_discriminator.to(device)

        self.rf_wgan_loss = rf_wgan_loss
        self.rf_cosine_loss = rf_cosine_loss

        self.device = device
        inferred_compressed_grid = getattr(self.compressor, "output_grid_size", None)
        if inferred_compressed_grid is not None and inferred_compressed_grid > 0:
            if compressed_grid_size != inferred_compressed_grid:
                print(
                    "[RFTokenCompressionTrainer] Warning: "
                    f"compressed_grid_size={compressed_grid_size} does not match "
                    f"compressor output ({inferred_compressed_grid}). Using "
                    f"{inferred_compressed_grid}."
                )
            compressed_grid_size = inferred_compressed_grid

        if hasattr(self.clip_encoder, "get_grid_size"):
            inferred_original_grid = self.clip_encoder.get_grid_size()
            if original_grid_size != inferred_original_grid:
                print(
                    "[RFTokenCompressionTrainer] Warning: "
                    f"original_grid_size={original_grid_size} does not match "
                    f"encoder output ({inferred_original_grid}). Using "
                    f"{inferred_original_grid}."
                )
            original_grid_size = inferred_original_grid

        self.compressed_grid_size = compressed_grid_size
        self.original_grid_size = original_grid_size
        self.cosine_weight = cosine_weight
        self.wgan_weight = wgan_weight
        self.n_critic = n_critic
        self.use_wandb = use_wandb

        # Optimizers (only compressor for generator)
        self.optimizer_G = torch.optim.Adam(
            compressor.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            rf_discriminator.parameters(),
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
        Train for one epoch with RF-aware losses.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.compressor.train()
        self.rf_discriminator.train()
        self.clip_encoder.eval()  # Always keep CLIP frozen

        metrics = {
            'disc_loss': 0.0,
            'gen_loss': 0.0,
            'cosine_loss': 0.0,
            'total_loss': 0.0,
            'wasserstein_distance': 0.0,
            'gradient_penalty': 0.0,
            'mean_similarity': 0.0
        }

        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Extract CLIP tokens (no gradients)
            with torch.no_grad():
                original_tokens = self.clip_encoder(images)  # (B, 576, 1024)

            # --- Train RF Discriminator ---
            for _ in range(self.n_critic):
                self.optimizer_D.zero_grad()

                # Compress tokens
                with torch.no_grad():
                    compressed_tokens = self.compressor(original_tokens)  # (B, 36, 1024)

                # RF-aware discriminator loss
                disc_loss, disc_info = self.rf_wgan_loss.discriminator_loss(
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens,
                    discriminator=self.rf_discriminator,
                    compressed_grid_size=self.compressed_grid_size,
                    original_grid_size=self.original_grid_size,
                    device=self.device
                )

                disc_loss.backward()
                self.optimizer_D.step()

            # --- Train Generator (Compressor) ---
            self.optimizer_G.zero_grad()

            # Compress tokens
            compressed_tokens = self.compressor(original_tokens)  # (B, 36, 1024)

            # RF-aware generator loss
            gen_loss, gen_info = self.rf_wgan_loss.generator_loss(
                compressed_tokens=compressed_tokens,
                discriminator=self.rf_discriminator
            )

            # Cosine similarity loss (maximize similarity with RF tokens)
            cosine_loss, cosine_info = self.rf_cosine_loss(
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                compressed_grid_size=(self.compressed_grid_size, self.compressed_grid_size),
                original_grid_size=(self.original_grid_size, self.original_grid_size)
            )

            # Total generator loss
            total_loss = (
                self.wgan_weight * gen_loss +
                self.cosine_weight * cosine_loss
            )

            total_loss.backward()
            self.optimizer_G.step()

            # Update metrics
            metrics['disc_loss'] += disc_info['disc_loss']
            metrics['gen_loss'] += gen_info['gen_loss']
            metrics['cosine_loss'] += cosine_info['cosine_sim_loss']
            metrics['total_loss'] += total_loss.item()
            metrics['wasserstein_distance'] += disc_info['wasserstein_distance']
            metrics['gradient_penalty'] += disc_info['gradient_penalty']
            metrics['mean_similarity'] += cosine_info['mean_similarity']

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/disc_loss': disc_info['disc_loss'],
                    'train/gen_loss': gen_info['gen_loss'],
                    'train/cosine_loss': cosine_info['cosine_sim_loss'],
                    'train/total_loss': total_loss.item(),
                    'train/wasserstein_distance': disc_info['wasserstein_distance'],
                    'train/gradient_penalty': disc_info['gradient_penalty'],
                    'train/mean_similarity': cosine_info['mean_similarity'],
                    'train/min_similarity': cosine_info['min_similarity'],
                    'train/max_similarity': cosine_info['max_similarity'],
                    'train/std_similarity': cosine_info['std_similarity'],
                    'global_step': self.global_step
                })

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'D_loss': f"{disc_info['disc_loss']:.4f}",
                'G_loss': f"{gen_info['gen_loss']:.4f}",
                'Cos_loss': f"{cosine_info['cosine_sim_loss']:.4f}",
                'Sim': f"{cosine_info['mean_similarity']:.4f}"
            })

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        save_visualizations: bool = True
    ) -> Dict[str, float]:
        """
        Validate the model with RF-aware metrics.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            save_visualizations: Whether to save RF visualizations

        Returns:
            Dictionary of validation metrics
        """
        self.compressor.eval()
        self.rf_discriminator.eval()

        metrics = {
            'val_cosine_loss': 0.0,
            'val_mean_similarity': 0.0,
            'val_min_similarity': 0.0,
            'val_max_similarity': 0.0
        }

        num_batches = len(val_loader)
        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Extract and compress
            original_tokens = self.clip_encoder(images)
            compressed_tokens = self.compressor(original_tokens)

            # Compute cosine similarity loss
            cosine_loss, cosine_info = self.rf_cosine_loss(
                compressed_tokens,
                original_tokens,
                (self.compressed_grid_size, self.compressed_grid_size),
                (self.original_grid_size, self.original_grid_size)
            )

            # Accumulate metrics
            metrics['val_cosine_loss'] += cosine_info['cosine_sim_loss']
            metrics['val_mean_similarity'] += cosine_info['mean_similarity']
            metrics['val_min_similarity'] += cosine_info['min_similarity']
            metrics['val_max_similarity'] += cosine_info['max_similarity']

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/cosine_loss': metrics['val_cosine_loss'],
                'val/mean_similarity': metrics['val_mean_similarity'],
                'val/min_similarity': metrics['val_min_similarity'],
                'val/max_similarity': metrics['val_max_similarity'],
                'epoch': epoch
            })

        return metrics

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'compressor_state_dict': self.compressor.state_dict(),
            'rf_discriminator_state_dict': self.rf_discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'compressed_grid_size': self.compressed_grid_size,
            'original_grid_size': self.original_grid_size
        }

        # Save regular checkpoint
        checkpoint_path = save_dir / f"rf_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = save_dir / "rf_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save latest
        latest_path = save_dir / "rf_latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
        self.rf_discriminator.load_state_dict(checkpoint['rf_discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
