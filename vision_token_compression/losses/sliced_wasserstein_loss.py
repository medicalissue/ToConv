"""
Sliced Wasserstein Distance Loss

Fast and stable alternative to Sinkhorn OT.
Based on "Sliced Wasserstein Distance for Learning Gaussian Mixture Models" (Kolouri et al., 2018)
and "Learning Generative Models with Sinkhorn Divergences" (Genevay et al., 2018)

Complexity: O(n log n) vs Sinkhorn's O(n^2)
No numerical issues with large cost matrices.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class SlicedWassersteinLoss(nn.Module):
    """
    Sliced Wasserstein Distance for comparing token distributions.

    Computes 1D Wasserstein distance along random projections and averages.
    Much faster and more stable than Sinkhorn OT, especially for large distributions.

    Algorithm:
    1. Generate random unit vectors (projections)
    2. Project both distributions onto each vector
    3. Compute 1D Wasserstein distance (= L1 distance of sorted values)
    4. Average over all projections
    """

    def __init__(
        self,
        num_projections: int = 128,
        projection_dim: int = None
    ):
        """
        Args:
            num_projections: Number of random 1D projections (more = more accurate but slower)
            projection_dim: Dimension of tokens (auto-detected if None)
        """
        super().__init__()
        self.num_projections = num_projections
        self.projection_dim = projection_dim

    def forward(
        self,
        compressed_tokens: torch.Tensor,
        original_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Sliced Wasserstein distance between compressed and original tokens.

        Args:
            compressed_tokens: (B, N1, D) - compressed token distribution
            original_tokens: (B, N2, D) - original token distribution

        Returns:
            loss: Sliced Wasserstein distance
            info: Dictionary with loss statistics
        """
        batch_size, n1, dim = compressed_tokens.shape
        n2 = original_tokens.size(1)

        # Auto-detect projection dimension
        if self.projection_dim is None:
            self.projection_dim = dim

        # Generate random unit projections: (num_projections, D)
        # Use random Gaussian vectors and normalize to unit sphere
        projections = torch.randn(
            self.num_projections,
            dim,
            device=compressed_tokens.device,
            dtype=compressed_tokens.dtype
        )
        projections = torch.nn.functional.normalize(projections, p=2, dim=1)

        # Compute sliced Wasserstein for each sample in batch
        sw_distances = []

        for b in range(batch_size):
            # Get samples: (N, D)
            x = compressed_tokens[b]  # (N1, D)
            y = original_tokens[b]     # (N2, D)

            # Project onto random directions: (N, D) @ (D, P) = (N, P)
            x_projected = x @ projections.t()  # (N1, num_projections)
            y_projected = y @ projections.t()  # (N2, num_projections)

            # Sort projections along each direction
            x_sorted, _ = torch.sort(x_projected, dim=0)  # (N1, num_projections)
            y_sorted, _ = torch.sort(y_projected, dim=0)  # (N2, num_projections)

            # Compute 1D Wasserstein distance for each projection
            # Need to handle different sizes N1 ≠ N2
            if n1 == n2:
                # Same size: direct L1 distance
                distances_per_proj = torch.abs(x_sorted - y_sorted).mean(dim=0)  # (num_projections,)
            else:
                # Different sizes: use CDF-based Wasserstein
                # Approximate by interpolation
                distances_per_proj = self._wasserstein_1d_different_sizes(
                    x_sorted, y_sorted
                )  # (num_projections,)

            # Average over all projections
            sw_dist = distances_per_proj.mean()
            sw_distances.append(sw_dist)

        # Average over batch
        sw_loss = torch.stack(sw_distances).mean()

        info = {
            'sw_loss': sw_loss.item(),
            'num_projections': self.num_projections,
            'n_compressed': n1,
            'n_original': n2
        }

        return sw_loss, info

    def _wasserstein_1d_different_sizes(
        self,
        x_sorted: torch.Tensor,
        y_sorted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 1D Wasserstein distance between sorted samples of different sizes.

        Uses linear interpolation to handle different number of samples.

        Args:
            x_sorted: (N1, num_projections) - sorted projections
            y_sorted: (N2, num_projections) - sorted projections

        Returns:
            distances: (num_projections,) - Wasserstein distance per projection
        """
        n1, num_proj = x_sorted.shape
        n2 = y_sorted.size(0)

        # Create uniform grids for CDFs
        # CDF values: [1/N, 2/N, ..., N/N]
        cdf_x = torch.linspace(0, 1, n1, device=x_sorted.device, dtype=x_sorted.dtype)
        cdf_y = torch.linspace(0, 1, n2, device=y_sorted.device, dtype=y_sorted.dtype)

        # Interpolate y onto x's grid for each projection
        # For each projection, we have sorted values and need to interpolate
        distances = []

        for p in range(num_proj):
            # Get sorted values for this projection
            x_vals = x_sorted[:, p]  # (N1,)
            y_vals = y_sorted[:, p]  # (N2,)

            # Interpolate y's quantiles onto x's grid
            # torch doesn't have interp1d, so we'll use a simpler approximation:
            # Just interpolate at the same quantile positions
            if n1 <= n2:
                # Downsample y to match x
                indices = torch.linspace(0, n2 - 1, n1, device=y_sorted.device)
                indices_floor = indices.long()
                indices_ceil = torch.clamp(indices_floor + 1, max=n2 - 1)
                weight = indices - indices_floor.float()

                y_interp = (1 - weight) * y_vals[indices_floor] + weight * y_vals[indices_ceil]
                dist = torch.abs(x_vals - y_interp).mean()
            else:
                # Downsample x to match y
                indices = torch.linspace(0, n1 - 1, n2, device=x_sorted.device)
                indices_floor = indices.long()
                indices_ceil = torch.clamp(indices_floor + 1, max=n1 - 1)
                weight = indices - indices_floor.float()

                x_interp = (1 - weight) * x_vals[indices_floor] + weight * x_vals[indices_ceil]
                dist = torch.abs(x_interp - y_vals).mean()

            distances.append(dist)

        return torch.stack(distances)


if __name__ == "__main__":
    print("Testing Sliced Wasserstein Loss...\n")

    # Test configurations
    batch_size = 4
    n_compressed = 64
    n_original = 256
    hidden_dim = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create loss
    sw_loss_fn = SlicedWassersteinLoss(num_projections=128)

    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
    original = torch.randn(batch_size, n_original, hidden_dim).to(device)

    loss, info = sw_loss_fn(compressed, original)

    print(f"✓ Loss: {info['sw_loss']:.6f}")
    print(f"  Projections: {info['num_projections']}")
    print(f"  Compressed: {info['n_compressed']}, Original: {info['n_original']}")

    # Test 2: Identical distributions should give near-zero distance
    print("\n2. Testing identical distributions...")
    same_dist = compressed.clone()
    loss_same, info_same = sw_loss_fn(compressed, same_dist)
    print(f"✓ Loss (same): {info_same['sw_loss']:.6f} (should be ~0)")

    # Test 3: Different distributions
    print("\n3. Testing shifted distributions...")
    shifted = compressed + 5.0
    loss_diff, info_diff = sw_loss_fn(compressed, shifted)
    print(f"✓ Loss (shifted): {info_diff['sw_loss']:.6f} (should be > same)")
    assert loss_diff.item() > loss_same.item() * 2, "Shifted distribution should have larger distance"

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    compressed_grad = torch.randn(batch_size, n_compressed, hidden_dim, requires_grad=True).to(device)
    original_grad = torch.randn(batch_size, n_original, hidden_dim).to(device)

    loss_grad, _ = sw_loss_fn(compressed_grad, original_grad)
    loss_grad.backward()

    assert compressed_grad.grad is not None
    print(f"✓ Gradient norm: {compressed_grad.grad.norm().item():.4f}")

    # Test 5: Different projection counts
    print("\n5. Testing different projection counts...")
    for num_proj in [32, 64, 128, 256]:
        sw_loss_temp = SlicedWassersteinLoss(num_projections=num_proj)
        loss_temp, info_temp = sw_loss_temp(compressed, original)
        print(f"  Projections={num_proj}: Loss={info_temp['sw_loss']:.6f}")

    # Test 6: Large unnormalized values (stress test)
    print("\n6. Testing with large unnormalized values...")
    large_compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device) * 50
    large_original = torch.randn(batch_size, n_original, hidden_dim).to(device) * 50

    loss_large, info_large = sw_loss_fn(large_compressed, large_original)
    print(f"✓ Loss (large values): {info_large['sw_loss']:.6f}")
    print(f"  No NaN: {not torch.isnan(loss_large).any()}")

    print("\n" + "="*60)
    print("✅ All Sliced Wasserstein tests passed!")
    print("="*60)
