"""AutoEncoder Loss for Token Reconstruction"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoderLoss(nn.Module):
    """
    AutoEncoder loss that measures reconstruction quality.

    The compressed tokens should be able to reconstruct the original tokens
    in their receptive field.
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
        normalize: bool = True
    ):
        """
        Args:
            loss_type: Type of reconstruction loss ('mse', 'cosine', or 'hybrid')
            reduction: How to reduce the loss ('mean' or 'sum')
            normalize: Whether to normalize tokens before computing loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize = normalize

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute reconstruction loss.

        Args:
            reconstructed: Reconstructed tokens (batch_size, num_tokens, hidden_dim)
            target: Target (original) tokens (batch_size, num_tokens, hidden_dim)

        Returns:
            loss: Reconstruction loss
            info: Dictionary with loss components
        """
        # Optionally normalize tokens
        if self.normalize:
            reconstructed = F.normalize(reconstructed, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        # Compute loss based on type
        if self.loss_type == 'mse':
            loss = self._mse_loss(reconstructed, target)
            info = {'ae_loss': loss.item(), 'loss_type': 'mse'}

        elif self.loss_type == 'cosine':
            loss = self._cosine_loss(reconstructed, target)
            info = {'ae_loss': loss.item(), 'loss_type': 'cosine'}

        elif self.loss_type == 'hybrid':
            mse_loss = self._mse_loss(reconstructed, target)
            cosine_loss = self._cosine_loss(reconstructed, target)
            loss = 0.5 * mse_loss + 0.5 * cosine_loss
            info = {
                'ae_loss': loss.item(),
                'mse_loss': mse_loss.item(),
                'cosine_loss': cosine_loss.item(),
                'loss_type': 'hybrid'
            }

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Add similarity metric
        with torch.no_grad():
            similarity = F.cosine_similarity(
                reconstructed.flatten(0, 1),
                target.flatten(0, 1),
                dim=-1
            ).mean()
            info['cosine_similarity'] = similarity.item()

        return loss, info

    def _mse_loss(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Mean Squared Error loss."""
        loss = F.mse_loss(reconstructed, target, reduction=self.reduction)
        return loss

    def _cosine_loss(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity loss (1 - cosine_similarity)."""
        # Flatten to (batch_size * num_tokens, hidden_dim)
        reconstructed_flat = reconstructed.flatten(0, 1)
        target_flat = target.flatten(0, 1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(reconstructed_flat, target_flat, dim=-1)

        # Convert to loss (1 - similarity)
        loss = 1 - cosine_sim

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class PerceptualReconstructionLoss(nn.Module):
    """
    Advanced reconstruction loss that considers perceptual similarity.
    Uses both token-level and spatial-level reconstruction.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        spatial_weight: float = 0.5,
        grid_size: int = 24
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            cosine_weight: Weight for cosine loss
            spatial_weight: Weight for spatial smoothness
            grid_size: Size of token grid for spatial operations
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.spatial_weight = spatial_weight
        self.grid_size = grid_size

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute perceptual reconstruction loss.

        Args:
            reconstructed: (batch_size, num_tokens, hidden_dim)
            target: (batch_size, num_tokens, hidden_dim)

        Returns:
            loss: Total loss
            info: Dictionary with loss components
        """
        # MSE loss
        mse_loss = F.mse_loss(reconstructed, target)

        # Cosine loss
        reconstructed_flat = reconstructed.flatten(0, 1)
        target_flat = target.flatten(0, 1)
        cosine_sim = F.cosine_similarity(reconstructed_flat, target_flat, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()

        # Spatial smoothness loss
        spatial_loss = self._compute_spatial_loss(reconstructed, target)

        # Total loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.cosine_weight * cosine_loss +
            self.spatial_weight * spatial_loss
        )

        info = {
            'ae_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'cosine_similarity': cosine_sim.mean().item()
        }

        return total_loss, info

    def _compute_spatial_loss(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial smoothness loss.
        Encourages locally similar tokens to remain similar.
        """
        batch_size, num_tokens, hidden_dim = reconstructed.shape

        # Reshape to 2D grid
        # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
        recon_2d = reconstructed.reshape(
            batch_size, self.grid_size, self.grid_size, hidden_dim
        ).permute(0, 3, 1, 2)

        target_2d = target.reshape(
            batch_size, self.grid_size, self.grid_size, hidden_dim
        ).permute(0, 3, 1, 2)

        # Compute differences with neighbors (horizontal and vertical)
        recon_diff_h = recon_2d[:, :, :, 1:] - recon_2d[:, :, :, :-1]
        recon_diff_v = recon_2d[:, :, 1:, :] - recon_2d[:, :, :-1, :]

        target_diff_h = target_2d[:, :, :, 1:] - target_2d[:, :, :, :-1]
        target_diff_v = target_2d[:, :, 1:, :] - target_2d[:, :, :-1, :]

        # Loss: differences should be preserved
        loss_h = F.mse_loss(recon_diff_h, target_diff_h)
        loss_v = F.mse_loss(recon_diff_v, target_diff_v)

        spatial_loss = (loss_h + loss_v) / 2

        return spatial_loss


if __name__ == "__main__":
    # Test AutoEncoder loss
    print("Testing AutoEncoder Loss...")

    batch_size = 4
    num_tokens = 576  # 24x24
    hidden_dim = 1024

    reconstructed = torch.randn(batch_size, num_tokens, hidden_dim)
    target = torch.randn(batch_size, num_tokens, hidden_dim)

    # Test MSE loss
    print("\n1. Testing MSE loss...")
    ae_loss_mse = AutoEncoderLoss(loss_type='mse')
    loss, info = ae_loss_mse(reconstructed, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # Test Cosine loss
    print("\n2. Testing Cosine loss...")
    ae_loss_cosine = AutoEncoderLoss(loss_type='cosine')
    loss, info = ae_loss_cosine(reconstructed, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # Test Hybrid loss
    print("\n3. Testing Hybrid loss...")
    ae_loss_hybrid = AutoEncoderLoss(loss_type='hybrid')
    loss, info = ae_loss_hybrid(reconstructed, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # Test Perceptual loss
    print("\n4. Testing Perceptual Reconstruction loss...")
    perceptual_loss = PerceptualReconstructionLoss(grid_size=24)
    loss, info = perceptual_loss(reconstructed, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # Test with identical tensors (should give perfect reconstruction)
    print("\n5. Testing with identical tensors...")
    loss, info = ae_loss_mse(target, target)
    print(f"Loss (should be ~0): {loss.item():.6f}")
    print(f"Cosine similarity (should be 1.0): {info['cosine_similarity']:.6f}")

    print("\n✓ AutoEncoder Loss test passed!")
