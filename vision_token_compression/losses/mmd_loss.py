"""Maximum Mean Discrepancy (MMD) Loss for Token Comparison"""
import torch
import torch.nn as nn
from typing import Tuple, Dict


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss using RBF (Gaussian) kernel.

    Measures the distance between the distributions of compressed and original tokens.
    """

    def __init__(
        self,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float = None
    ):
        """
        Args:
            kernel_mul: Multiplier for bandwidth in multi-scale kernel
            kernel_num: Number of kernels to use in multi-scale kernel
            fix_sigma: If provided, use this fixed bandwidth. Otherwise, compute adaptively
        """
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float = None
    ) -> torch.Tensor:
        """
        Compute multi-scale Gaussian kernel matrix.

        Args:
            source: (n, d) - source samples
            target: (m, d) - target samples
            kernel_mul: Multiplier for bandwidth
            kernel_num: Number of kernels
            fix_sigma: Fixed bandwidth (if None, compute adaptively)

        Returns:
            kernel_val: ((n+m), (n+m)) - kernel matrix
        """
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        # Compute pairwise squared distances
        # (n+m, d) @ (d, n+m) -> (n+m, n+m)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        # L2 distance squared
        L2_distance_squared = ((total0 - total1) ** 2).sum(2)

        # Compute bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # Use median heuristic
            bandwidth = torch.median(L2_distance_squared[L2_distance_squared > 0])
            if bandwidth == 0:
                bandwidth = 1.0

        # Multi-scale bandwidths
        bandwidth_list = [bandwidth / (kernel_mul ** i) for i in range(kernel_num)]

        # Compute kernel matrix as sum of Gaussian kernels
        kernel_val = torch.zeros_like(L2_distance_squared)
        for bandwidth in bandwidth_list:
            kernel_val += torch.exp(-L2_distance_squared / (2 * bandwidth))

        return kernel_val

    def forward(
        self,
        compressed_tokens: torch.Tensor,
        original_tokens: torch.Tensor,
        compressed_grid_size: Tuple[int, int],
        original_grid_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MMD loss between compressed and original tokens.

        Args:
            compressed_tokens: (B, k², hidden_dim) - compressed tokens
            original_tokens: (B, H*W, hidden_dim) - original tokens
            compressed_grid_size: (k, k) - size of compressed grid
            original_grid_size: (H, W) - size of original grid

        Returns:
            loss: MMD loss value
            info: Dictionary with loss statistics
        """
        batch_size = compressed_tokens.size(0)

        # Flatten to (B*N, D) for batch-wise computation
        compressed_flat = compressed_tokens.reshape(-1, compressed_tokens.size(-1))
        original_flat = original_tokens.reshape(-1, original_tokens.size(-1))

        # Compute kernel matrix
        kernel_matrix = self.gaussian_kernel(
            compressed_flat,
            original_flat,
            self.kernel_mul,
            self.kernel_num,
            self.fix_sigma
        )

        # Split kernel matrix into blocks
        n_compressed = compressed_flat.size(0)
        n_original = original_flat.size(0)

        # K_XX: compressed vs compressed
        K_XX = kernel_matrix[:n_compressed, :n_compressed]

        # K_YY: original vs original
        K_YY = kernel_matrix[n_compressed:, n_compressed:]

        # K_XY: compressed vs original
        K_XY = kernel_matrix[:n_compressed, n_compressed:]

        # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        # Remove diagonal for unbiased estimate
        n_x = K_XX.size(0)
        n_y = K_YY.size(0)

        # Unbiased estimate: exclude diagonal
        K_XX_mean = (K_XX.sum() - K_XX.diagonal().sum()) / (n_x * (n_x - 1))
        K_YY_mean = (K_YY.sum() - K_YY.diagonal().sum()) / (n_y * (n_y - 1))
        K_XY_mean = K_XY.mean()

        mmd_squared = K_XX_mean + K_YY_mean - 2 * K_XY_mean

        # Ensure non-negative (can be slightly negative due to numerical errors)
        mmd_squared = torch.clamp(mmd_squared, min=0.0)
        mmd_loss = torch.sqrt(mmd_squared)

        info = {
            'mmd_loss': mmd_loss.item(),
            'mmd_squared': mmd_squared.item(),
            'K_XX_mean': K_XX_mean.item(),
            'K_YY_mean': K_YY_mean.item(),
            'K_XY_mean': K_XY_mean.item()
        }

        return mmd_loss, info


if __name__ == "__main__":
    # Test MMD loss
    print("Testing MMD Loss...")

    batch_size = 4
    compressed_grid_size = (6, 6)
    original_grid_size = (24, 24)
    hidden_dim = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dummy data
    original_tokens = torch.randn(batch_size, 576, hidden_dim).to(device)
    compressed_tokens = torch.randn(batch_size, 36, hidden_dim).to(device)

    # Create MMD loss
    mmd_loss = MMDLoss(kernel_mul=2.0, kernel_num=5).to(device)

    # Test 1: Basic forward pass
    print("1. Testing basic forward pass...")
    loss, info = mmd_loss(
        compressed_tokens,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    print(f"MMD loss: {loss.item():.6f}")
    print(f"MMD squared: {info['mmd_squared']:.6f}")
    print(f"K_XX mean: {info['K_XX_mean']:.6f}")
    print(f"K_YY mean: {info['K_YY_mean']:.6f}")
    print(f"K_XY mean: {info['K_XY_mean']:.6f}")
    print("✓ Basic forward test passed!")

    # Test 2: Same distribution should give small MMD
    print("\n2. Testing identical distributions...")
    same_tokens = compressed_tokens.clone()
    loss_same, info_same = mmd_loss(
        compressed_tokens,
        same_tokens,
        compressed_grid_size,
        compressed_grid_size
    )
    print(f"MMD loss (same dist): {loss_same.item():.6f}")
    assert loss_same.item() < 0.1, "Same distribution should have near-zero MMD"
    print("✓ Identical distribution test passed!")

    # Test 3: Different distributions should give larger MMD
    print("\n3. Testing different distributions...")
    different_tokens = compressed_tokens + 5.0  # Shift distribution
    loss_diff, info_diff = mmd_loss(
        compressed_tokens,
        different_tokens,
        compressed_grid_size,
        compressed_grid_size
    )
    print(f"MMD loss (shifted dist): {loss_diff.item():.6f}")
    assert loss_diff.item() > loss_same.item(), "Different distributions should have larger MMD"
    print("✓ Different distribution test passed!")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    compressed_tokens_grad = torch.randn(batch_size, 36, hidden_dim, requires_grad=True).to(device)
    original_tokens_grad = torch.randn(batch_size, 576, hidden_dim).to(device)

    loss_grad, _ = mmd_loss(
        compressed_tokens_grad,
        original_tokens_grad,
        compressed_grid_size,
        original_grid_size
    )
    loss_grad.backward()

    assert compressed_tokens_grad.grad is not None
    print(f"Gradient norm: {compressed_tokens_grad.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    # Test 5: Fixed sigma vs adaptive
    print("\n5. Testing fixed vs adaptive sigma...")
    mmd_fixed = MMDLoss(kernel_mul=2.0, kernel_num=5, fix_sigma=1.0).to(device)
    loss_fixed, info_fixed = mmd_fixed(
        compressed_tokens,
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )
    print(f"MMD loss (fixed sigma): {loss_fixed.item():.6f}")
    print(f"MMD loss (adaptive sigma): {loss.item():.6f}")
    print("✓ Fixed sigma test passed!")

    print("\n" + "=" * 60)
    print("All MMD Loss tests passed!")
    print("=" * 60)
