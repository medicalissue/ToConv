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
from .models.rf_autoencoder import RFAutoEncoderDecoder
from .losses.rf_wgan_gp import RFWGANGPLoss
from .losses.rf_autoencoder_loss import RFAutoEncoderLoss
from .utils.rf_utils import compute_rf_statistics, create_rf_heatmap


class RFTokenCompressionTrainer:
    """
    Trainer for RF-based vision token compression system.
    Combines RF-aware WGAN-GP and AutoEncoder losses.
    """

    def __init__(
        self,
        clip_encoder: CLIPVisionEncoder,
        compressor: TokenCompressor,
        rf_discriminator: RFDiscriminator,
        rf_ae_decoder: RFAutoEncoderDecoder,
        rf_wgan_loss: RFWGANGPLoss,
        rf_ae_loss: RFAutoEncoderLoss,
        device: torch.device,
        compressed_grid_size: int = 6,
        original_grid_size: int = 24,
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
            rf_discriminator: RF-aware discriminator
            rf_ae_decoder: RF-aware AutoEncoder decoder
            rf_wgan_loss: RF WGAN-GP loss function
            rf_ae_loss: RF AutoEncoder loss function
            device: Device to train on
            compressed_grid_size: 6 (6x6 grid)
            original_grid_size: 24 (24x24 grid)
            learning_rate: Learning rate for compressor and decoder
            discriminator_lr: Learning rate for discriminator
            ae_weight: Weight for autoencoder loss
            wgan_weight: Weight for WGAN-GP loss
            n_critic: Number of discriminator updates per generator update
            use_wandb: Whether to log to Weights & Biases
        """
        self.clip_encoder = clip_encoder.to(device)
        self.compressor = compressor.to(device)
        self.rf_discriminator = rf_discriminator.to(device)
        self.rf_ae_decoder = rf_ae_decoder.to(device)

        self.rf_wgan_loss = rf_wgan_loss
        self.rf_ae_loss = rf_ae_loss

        self.device = device
        self.compressed_grid_size = compressed_grid_size
        self.original_grid_size = original_grid_size
        self.ae_weight = ae_weight
        self.wgan_weight = wgan_weight
        self.n_critic = n_critic
        self.use_wandb = use_wandb

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(compressor.parameters()) + list(rf_ae_decoder.parameters()),
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
        self.rf_ae_decoder.train()
        self.clip_encoder.eval()  # Always keep CLIP frozen

        metrics = {
            'disc_loss': 0.0,
            'gen_loss': 0.0,
            'ae_loss': 0.0,
            'total_loss': 0.0,
            'wasserstein_distance': 0.0,
            'gradient_penalty': 0.0,
            'rf_similarity_mean': 0.0
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

            # --- Train Generator (Compressor + RF Decoder) ---
            self.optimizer_G.zero_grad()

            # Compress tokens
            compressed_tokens = self.compressor(original_tokens)  # (B, 36, 1024)

            # RF-aware generator loss
            gen_loss, gen_info = self.rf_wgan_loss.generator_loss(
                compressed_tokens=compressed_tokens,
                discriminator=self.rf_discriminator
            )

            # RF-aware reconstruction loss
            reconstructed_rfs = self.rf_ae_decoder(compressed_tokens)  # (B, 36, 16, 1024)
            ae_loss, ae_info = self.rf_ae_loss(
                reconstructed_rfs=reconstructed_rfs,
                original_tokens=original_tokens,
                compressed_grid_size=self.compressed_grid_size,
                original_grid_size=self.original_grid_size
            )

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
            metrics['rf_similarity_mean'] += ae_info.get('rf_similarity_mean', 0.0)

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/disc_loss': disc_info['disc_loss'],
                    'train/gen_loss': gen_info['gen_loss'],
                    'train/ae_loss': ae_info['ae_loss'],
                    'train/total_loss': total_loss.item(),
                    'train/wasserstein_distance': disc_info['wasserstein_distance'],
                    'train/gradient_penalty': disc_info['gradient_penalty'],
                    'train/rf_similarity_mean': ae_info.get('rf_similarity_mean', 0.0),
                    'train/rf_similarity_min': ae_info.get('rf_similarity_min', 0.0),
                    'train/rf_similarity_max': ae_info.get('rf_similarity_max', 0.0),
                    'global_step': self.global_step
                })

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'D_loss': f"{disc_info['disc_loss']:.4f}",
                'G_loss': f"{gen_info['gen_loss']:.4f}",
                'AE_loss': f"{ae_info['ae_loss']:.4f}",
                'RF_sim': f"{ae_info.get('rf_similarity_mean', 0.0):.4f}"
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
        self.rf_ae_decoder.eval()

        metrics = {
            'val_ae_loss': 0.0,
            'val_rf_similarity': 0.0,
            'val_excellent_rfs': 0.0,
            'val_good_rfs': 0.0,
            'val_fair_rfs': 0.0,
            'val_poor_rfs': 0.0
        }

        num_batches = len(val_loader)
        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        all_rf_similarities = []

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Extract and compress
            original_tokens = self.clip_encoder(images)
            compressed_tokens = self.compressor(original_tokens)

            # Reconstruct RFs
            reconstructed_rfs = self.rf_ae_decoder(compressed_tokens)

            # Compute RF reconstruction loss
            ae_loss, ae_info = self.rf_ae_loss(
                reconstructed_rfs,
                original_tokens,
                self.compressed_grid_size,
                self.original_grid_size
            )

            # Compute detailed RF statistics
            rf_stats = compute_rf_statistics(
                reconstructed_rfs,
                original_tokens,
                self.compressed_grid_size,
                self.original_grid_size
            )

            # Accumulate metrics
            metrics['val_ae_loss'] += ae_info['ae_loss']
            metrics['val_rf_similarity'] += rf_stats['rf_similarity_mean']
            metrics['val_excellent_rfs'] += rf_stats['excellent_rfs']
            metrics['val_good_rfs'] += rf_stats['good_rfs']
            metrics['val_fair_rfs'] += rf_stats['fair_rfs']
            metrics['val_poor_rfs'] += rf_stats['poor_rfs']

            all_rf_similarities.extend(rf_stats['per_rf_similarities'])

            # Save visualizations for first batch
            if save_visualizations and batch_idx == 0:
                self._save_rf_visualizations(
                    original_tokens,
                    compressed_tokens,
                    reconstructed_rfs,
                    epoch,
                    rf_stats
                )

        # Average metrics
        for key in metrics:
            if key.startswith('val_excellent') or key.startswith('val_good') or \
               key.startswith('val_fair') or key.startswith('val_poor'):
                metrics[key] /= num_batches  # These are counts, average them
            else:
                metrics[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/ae_loss': metrics['val_ae_loss'],
                'val/rf_similarity': metrics['val_rf_similarity'],
                'val/excellent_rfs': metrics['val_excellent_rfs'],
                'val/good_rfs': metrics['val_good_rfs'],
                'val/fair_rfs': metrics['val_fair_rfs'],
                'val/poor_rfs': metrics['val_poor_rfs'],
                'epoch': epoch
            })

        return metrics

    def _save_rf_visualizations(
        self,
        original_tokens: torch.Tensor,
        compressed_tokens: torch.Tensor,
        reconstructed_rfs: torch.Tensor,
        epoch: int,
        rf_stats: dict
    ):
        """Save RF reconstruction visualizations."""
        from .utils.rf_utils import visualize_rf_reconstruction, create_rf_heatmap

        vis_dir = Path("./visualizations")
        vis_dir.mkdir(exist_ok=True)

        # Create heatmap
        heatmap_path = vis_dir / f"rf_heatmap_epoch_{epoch}.png"
        create_rf_heatmap(
            rf_stats['per_rf_similarities'],
            self.compressed_grid_size,
            save_path=str(heatmap_path),
            title=f"RF Reconstruction Quality - Epoch {epoch}"
        )

        # Visualize specific RFs (best, worst, median)
        similarities = rf_stats['per_rf_similarities']
        best_idx = similarities.index(max(similarities))
        worst_idx = similarities.index(min(similarities))
        median_idx = sorted(range(len(similarities)), key=lambda i: similarities[i])[len(similarities)//2]

        for label, idx in [('best', best_idx), ('worst', worst_idx), ('median', median_idx)]:
            viz_path = vis_dir / f"rf_{label}_epoch_{epoch}_idx_{idx}.png"
            visualize_rf_reconstruction(
                original_tokens,
                compressed_tokens,
                reconstructed_rfs,
                compressed_idx=idx,
                save_path=str(viz_path)
            )

        if self.use_wandb:
            # Log images to wandb
            wandb.log({
                f"val/rf_heatmap": wandb.Image(str(heatmap_path)),
                f"val/rf_best": wandb.Image(str(vis_dir / f"rf_best_epoch_{epoch}_idx_{best_idx}.png")),
                f"val/rf_worst": wandb.Image(str(vis_dir / f"rf_worst_epoch_{epoch}_idx_{worst_idx}.png")),
                "epoch": epoch
            })

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
            'rf_ae_decoder_state_dict': self.rf_ae_decoder.state_dict(),
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
        self.rf_ae_decoder.load_state_dict(checkpoint['rf_ae_decoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
