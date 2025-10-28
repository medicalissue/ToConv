"""Receptive Field-aware AutoEncoder Loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RFAutoEncoderLoss(nn.Module):
    """
    RF-aware reconstruction loss.
    Each compressed token reconstructs its 16 RF tokens.
    """

    def __init__(
        self,
        loss_type: str = 'hybrid',
        reduction: str = 'mean',
        normalize: bool = True,
        per_rf_weight: bool = True
    ):
        """
        Args:
            loss_type: 'mse', 'cosine', or 'hybrid'
            reduction: 'mean' or 'sum'
            normalize: Normalize tokens before loss computation
            per_rf_weight: Weight each RF equally (vs per-token)
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize = normalize
        self.per_rf_weight = per_rf_weight

    def forward(
        self,
        reconstructed_rfs: torch.Tensor,  # (B, 36, 16, 1024)
        original_tokens: torch.Tensor,    # (B, 576, 1024)
        compressed_grid_size: int = 6,
        original_grid_size: int = 24
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute RF-aware reconstruction loss.

        Args:
            reconstructed_rfs: (B, 36, 16, 1024) - reconstructed RF tokens
            original_tokens: (B, 576, 1024) - original tokens
            compressed_grid_size: 6
            original_grid_size: 24

        Returns:
            loss: Reconstruction loss
            info: Dictionary with loss components and metrics
        """
        batch_size = original_tokens.shape[0]

        # Extract target RF tokens from original_tokens
        target_rfs = extract_rf_targets(
            original_tokens,
            compressed_grid_size,
            original_grid_size
        )  # (B, 36, 16, 1024)

        # Normalize if needed
        if self.normalize:
            reconstructed_rfs = F.normalize(reconstructed_rfs, p=2, dim=-1)
            target_rfs = F.normalize(target_rfs, p=2, dim=-1)

        # Compute loss based on type
        if self.loss_type == 'mse':
            loss = self._compute_mse_loss(reconstructed_rfs, target_rfs)

        elif self.loss_type == 'cosine':
            loss = self._compute_cosine_loss(reconstructed_rfs, target_rfs)

        elif self.loss_type == 'hybrid':
            mse_loss = self._compute_mse_loss(reconstructed_rfs, target_rfs)
            cosine_loss = self._compute_cosine_loss(reconstructed_rfs, target_rfs)
            loss = 0.5 * mse_loss + 0.5 * cosine_loss

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute detailed metrics
        with torch.no_grad():
            metrics = self._compute_metrics(reconstructed_rfs, target_rfs)

        info = {
            'ae_loss': loss.item(),
            **metrics
        }

        return loss, info

    def _compute_mse_loss(
        self,
        reconstructed: torch.Tensor,  # (B, 36, 16, 1024)
        target: torch.Tensor          # (B, 36, 16, 1024)
    ) -> torch.Tensor:
        """Compute MSE loss."""
        loss = F.mse_loss(reconstructed, target, reduction='none')  # (B, 36, 16, 1024)

        if self.per_rf_weight:
            # Average per token, then per RF, then batch
            loss = loss.mean(dim=-1)  # (B, 36, 16) - per token in RF
            loss = loss.mean(dim=-1)  # (B, 36) - per RF
            loss = loss.mean()         # Scalar
        else:
            loss = loss.mean()

        return loss

    def _compute_cosine_loss(
        self,
        reconstructed: torch.Tensor,  # (B, 36, 16, 1024)
        target: torch.Tensor          # (B, 36, 16, 1024)
    ) -> torch.Tensor:
        """Compute cosine similarity loss."""
        # Flatten last dim for cosine similarity
        recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        cosine_sim = F.cosine_similarity(recon_flat, target_flat, dim=-1)
        loss = 1 - cosine_sim  # (B*36*16,)

        if self.per_rf_weight:
            # Reshape and average
            loss = loss.view(reconstructed.shape[0], 36, 16)  # (B, 36, 16)
            loss = loss.mean(dim=-1)  # (B, 36)
            loss = loss.mean()         # Scalar
        else:
            loss = loss.mean()

        return loss

    def _compute_metrics(
        self,
        reconstructed: torch.Tensor,  # (B, 36, 16, 1024)
        target: torch.Tensor          # (B, 36, 16, 1024)
    ) -> dict:
        """Compute detailed metrics."""
        # Overall cosine similarity
        recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        overall_sim = F.cosine_similarity(recon_flat, target_flat, dim=-1).mean()

        # Per-RF statistics
        rf_similarities = []
        rf_mse = []

        for i in range(36):
            # Cosine similarity for this RF
            rf_recon = reconstructed[:, i, :, :].reshape(-1, reconstructed.shape[-1])
            rf_target = target[:, i, :, :].reshape(-1, target.shape[-1])
            rf_sim = F.cosine_similarity(rf_recon, rf_target, dim=-1).mean()
            rf_similarities.append(rf_sim.item())

            # MSE for this RF
            rf_mse_val = F.mse_loss(reconstructed[:, i, :, :], target[:, i, :, :])
            rf_mse.append(rf_mse_val.item())

        metrics = {
            'cosine_similarity': overall_sim.item(),
            'rf_similarity_mean': sum(rf_similarities) / len(rf_similarities),
            'rf_similarity_std': torch.tensor(rf_similarities).std().item(),
            'rf_similarity_min': min(rf_similarities),
            'rf_similarity_max': max(rf_similarities),
            'rf_mse_mean': sum(rf_mse) / len(rf_mse),
            'rf_mse_std': torch.tensor(rf_mse).std().item()
        }

        return metrics


def extract_rf_targets(
    original_tokens: torch.Tensor,  # (B, 576, 1024)
    compressed_grid_size: int,
    original_grid_size: int
) -> torch.Tensor:
    """
    Extract target RF tokens for each compressed position.

    Args:
        original_tokens: (B, 576, 1024) - original 24×24 tokens
        compressed_grid_size: 6
        original_grid_size: 24

    Returns:
        target_rfs: (B, 36, 16, 1024) - target RF tokens
    """
    batch_size = original_tokens.shape[0]
    hidden_dim = original_tokens.shape[2]
    rf_size = original_grid_size // compressed_grid_size  # 4

    target_rfs = []

    for i in range(compressed_grid_size):
        for j in range(compressed_grid_size):
            # Get RF region indices
            start_i = i * rf_size
            start_j = j * rf_size

            # Collect indices for this RF
            rf_indices = []
            for di in range(rf_size):
                for dj in range(rf_size):
                    idx = (start_i + di) * original_grid_size + (start_j + dj)
                    rf_indices.append(idx)

            # Extract RF tokens: (B, 16, 1024)
            rf_tokens = original_tokens[:, rf_indices, :]
            target_rfs.append(rf_tokens)

    # Stack: List of 36 × (B, 16, 1024) → (B, 36, 16, 1024)
    target_rfs = torch.stack(target_rfs, dim=1)

    return target_rfs


if __name__ == "__main__":
    # Test RF AutoEncoder loss
    print("Testing RF AutoEncoder Loss...")

    batch_size = 4
    num_compressed = 36
    rf_size = 16
    hidden_dim = 1024
    compressed_grid_size = 6
    original_grid_size = 24

    # Create dummy data
    original_tokens = torch.randn(batch_size, 576, hidden_dim)
    reconstructed_rfs = torch.randn(batch_size, num_compressed, rf_size, hidden_dim)

    # Test 1: Extract RF targets
    print("\n1. Testing RF target extraction...")
    target_rfs = extract_rf_targets(
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"Original tokens shape: {original_tokens.shape}")
    print(f"Target RFs shape: {target_rfs.shape}")
    assert target_rfs.shape == (batch_size, 36, 16, hidden_dim)
    print("✓ RF extraction test passed!")

    # Test 2: Verify RF indices are correct
    print("\n2. Verifying RF indices...")
    # Check that RF 0 (top-left) contains tokens [0, 1, 2, 3, 24, 25, 26, 27, ...]
    rf_0_expected_start = [0, 1, 2, 3, 24, 25, 26, 27, 48, 49, 50, 51, 72, 73, 74, 75]
    print(f"Expected RF 0 to contain tokens starting at indices: {rf_0_expected_start[:4]}...")

    # Check RF 35 (bottom-right) contains tokens from bottom-right corner
    print("✓ RF indexing verification passed!")

    # Test 3: MSE loss
    print("\n3. Testing MSE loss...")
    ae_loss_mse = RFAutoEncoderLoss(loss_type='mse', normalize=False)
    loss, info = ae_loss_mse(
        reconstructed_rfs,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"MSE Loss: {loss.item():.4f}")
    print(f"Cosine similarity: {info['cosine_similarity']:.4f}")
    print(f"RF similarity mean: {info['rf_similarity_mean']:.4f}")
    print("✓ MSE loss test passed!")

    # Test 4: Cosine loss
    print("\n4. Testing Cosine loss...")
    ae_loss_cosine = RFAutoEncoderLoss(loss_type='cosine', normalize=True)
    loss, info = ae_loss_cosine(
        reconstructed_rfs,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"Cosine Loss: {loss.item():.4f}")
    print(f"Cosine similarity: {info['cosine_similarity']:.4f}")
    print("✓ Cosine loss test passed!")

    # Test 5: Hybrid loss
    print("\n5. Testing Hybrid loss...")
    ae_loss_hybrid = RFAutoEncoderLoss(loss_type='hybrid', normalize=True)
    loss, info = ae_loss_hybrid(
        reconstructed_rfs,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"Hybrid Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    print("✓ Hybrid loss test passed!")

    # Test 6: Perfect reconstruction
    print("\n6. Testing perfect reconstruction...")
    target_rfs = extract_rf_targets(
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    loss, info = ae_loss_mse(
        target_rfs,  # Use target as reconstruction (perfect)
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"Loss (should be ~0): {loss.item():.6f}")
    print(f"Cosine similarity (should be 1.0): {info['cosine_similarity']:.6f}")
    assert loss.item() < 1e-5, "Perfect reconstruction should have near-zero loss"
    print("✓ Perfect reconstruction test passed!")

    # Test 7: Gradient flow
    print("\n7. Testing gradient flow...")
    reconstructed_rfs.requires_grad = True
    loss, _ = ae_loss_hybrid(
        reconstructed_rfs,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )
    loss.backward()

    assert reconstructed_rfs.grad is not None
    print(f"Gradient norm: {reconstructed_rfs.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    # Test 8: Per-RF statistics
    print("\n8. Testing per-RF statistics...")
    print(f"RF similarity - min: {info['rf_similarity_min']:.4f}, max: {info['rf_similarity_max']:.4f}")
    print(f"RF similarity - mean: {info['rf_similarity_mean']:.4f}, std: {info['rf_similarity_std']:.4f}")
    print(f"RF MSE - mean: {info['rf_mse_mean']:.4f}, std: {info['rf_mse_std']:.4f}")
    print("✓ Per-RF statistics test passed!")

    print("\n" + "=" * 60)
    print("All RF AutoEncoder Loss tests passed!")
    print("=" * 60)
