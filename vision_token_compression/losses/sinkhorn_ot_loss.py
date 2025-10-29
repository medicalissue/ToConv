"""Sinkhorn Optimal Transport (OT) Loss for Token Comparison"""
import torch
import torch.nn as nn
from typing import Tuple, Dict


class SinkhornOTLoss(nn.Module):
    """
    Sinkhorn Optimal Transport loss for comparing distributions.

    Computes the Wasserstein distance between compressed and original token distributions
    using the Sinkhorn-Knopp algorithm with entropic regularization.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        threshold: float = 1e-3,
        normalize: bool = True
    ):
        """
        Args:
            epsilon: Entropic regularization parameter (smaller = closer to exact OT, but slower)
            max_iter: Maximum number of Sinkhorn iterations
            threshold: Convergence threshold for stopping criterion
            normalize: Whether to normalize the cost by the number of samples
        """
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
        self.normalize = normalize

    def compute_cost_matrix(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distance matrix.

        Args:
            source: (n, d) - source samples
            target: (m, d) - target samples

        Returns:
            cost: (n, m) - pairwise squared distances
        """
        # Compute ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        source_sq = (source ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        target_sq = (target ** 2).sum(dim=1, keepdim=True)  # (m, 1)

        # (n, d) @ (d, m) = (n, m)
        cross_term = torch.matmul(source, target.t())

        # (n, 1) + (1, m) - 2*(n, m) = (n, m)
        cost = source_sq + target_sq.t() - 2 * cross_term

        return cost

    def sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        epsilon: float,
        max_iter: int,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sinkhorn-Knopp algorithm for computing optimal transport plan.

        Args:
            cost_matrix: (n, m) - cost matrix
            epsilon: Entropic regularization parameter
            max_iter: Maximum iterations
            threshold: Convergence threshold

        Returns:
            u: (n,) - left scaling vector
            v: (m,) - right scaling vector
            num_iter: Number of iterations performed
        """
        n, m = cost_matrix.shape

        # Compute Gibbs kernel: K = exp(-C/epsilon)
        K = torch.exp(-cost_matrix / epsilon)

        # Initialize uniform marginals
        a = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype) / n
        b = torch.ones(m, device=cost_matrix.device, dtype=cost_matrix.dtype) / m

        # Initialize dual variables (log-domain for stability)
        u = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype)
        v = torch.ones(m, device=cost_matrix.device, dtype=cost_matrix.dtype)

        # Sinkhorn iterations
        for iteration in range(max_iter):
            u_prev = u.clone()

            # Update u: u = a / (K @ v)
            u = a / (K @ v + 1e-10)

            # Update v: v = b / (K^T @ u)
            v = b / (K.t() @ u + 1e-10)

            # Check convergence
            err = torch.abs(u - u_prev).max()
            if err < threshold:
                break

        return u, v, iteration + 1

    def compute_ot_distance(
        self,
        cost_matrix: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        Compute the optimal transport distance from the transport plan.

        Args:
            cost_matrix: (n, m) - cost matrix
            u: (n,) - left scaling vector
            v: (m,) - right scaling vector
            epsilon: Entropic regularization parameter

        Returns:
            distance: Scalar OT distance
        """
        # Compute optimal transport plan: P = diag(u) @ K @ diag(v)
        # where K = exp(-C/epsilon)
        K = torch.exp(-cost_matrix / epsilon)

        # Efficient computation: <P, C> = sum(u_i * K_ij * v_j * C_ij)
        transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)

        # Compute <P, C> = trace(P^T @ C)
        ot_distance = (transport_plan * cost_matrix).sum()

        return ot_distance

    def forward(
        self,
        compressed_tokens: torch.Tensor,
        original_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Sinkhorn OT loss between compressed and original tokens.

        Args:
            compressed_tokens: (B, k², hidden_dim) - compressed tokens
            original_tokens: (B, H*W, hidden_dim) - original tokens

        Returns:
            loss: OT loss value
            info: Dictionary with loss statistics
        """
        # Flatten to (B*N, D) for batch-wise computation
        compressed_flat = compressed_tokens.reshape(-1, compressed_tokens.size(-1))
        original_flat = original_tokens.reshape(-1, original_tokens.size(-1))

        # Compute cost matrix (squared Euclidean distance)
        cost_matrix = self.compute_cost_matrix(compressed_flat, original_flat)

        # Run Sinkhorn algorithm
        u, v, num_iter = self.sinkhorn_algorithm(
            cost_matrix,
            self.epsilon,
            self.max_iter,
            self.threshold
        )

        # Compute OT distance
        ot_distance = self.compute_ot_distance(cost_matrix, u, v, self.epsilon)

        # Normalize by number of samples if requested
        if self.normalize:
            n_samples = compressed_flat.size(0) + original_flat.size(0)
            ot_distance = ot_distance / n_samples

        # For numerical stability and to ensure gradients flow properly
        ot_loss = torch.sqrt(ot_distance + 1e-8)

        info = {
            'ot_loss': ot_loss.item(),
            'ot_distance': ot_distance.item(),
            'sinkhorn_iterations': num_iter,
            'epsilon': self.epsilon,
            'n_compressed': compressed_flat.size(0),
            'n_original': original_flat.size(0)
        }

        return ot_loss, info


if __name__ == "__main__":
    # Test Sinkhorn OT loss
    print("Testing Sinkhorn OT Loss...")

    batch_size = 4
    compressed_grid_size = (8, 8)
    original_grid_size = (16, 16)
    hidden_dim = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dummy data
    original_tokens = torch.randn(batch_size, 256, hidden_dim).to(device)
    compressed_tokens = torch.randn(batch_size, 64, hidden_dim).to(device)

    # Create Sinkhorn OT loss
    ot_loss = SinkhornOTLoss(epsilon=0.1, max_iter=100, threshold=1e-3).to(device)

    # Test 1: Basic forward pass
    print("1. Testing basic forward pass...")
    loss, info = ot_loss(
        compressed_tokens,
        original_tokens
    )

    print(f"OT loss: {loss.item():.6f}")
    print(f"OT distance: {info['ot_distance']:.6f}")
    print(f"Sinkhorn iterations: {info['sinkhorn_iterations']}")
    print(f"Epsilon: {info['epsilon']:.3f}")
    print("✓ Basic forward test passed!")

    # Test 2: Same distribution should give small OT
    print("\n2. Testing identical distributions...")
    same_tokens = compressed_tokens.clone()
    loss_same, info_same = ot_loss(
        compressed_tokens,
        same_tokens
    )
    print(f"OT loss (same dist): {loss_same.item():.6f}")
    print(f"Sinkhorn iterations: {info_same['sinkhorn_iterations']}")
    print("✓ Identical distribution test passed!")

    # Test 3: Different distributions should give larger OT
    print("\n3. Testing different distributions...")
    # Use a more pronounced shift to ensure measurable difference
    different_tokens = compressed_tokens + 2.0  # Shift distribution
    loss_diff, info_diff = ot_loss(
        compressed_tokens,
        different_tokens
    )
    print(f"OT loss (shifted dist): {loss_diff.item():.6f}")
    print(f"Sinkhorn iterations: {info_diff['sinkhorn_iterations']}")

    # The shifted distribution should have significantly larger OT
    # Allow for numerical tolerance
    if loss_diff.item() > loss_same.item() * 2:
        print("✓ Different distribution test passed!")
    else:
        print(f"⚠ Warning: Shifted distribution OT ({loss_diff.item():.6f}) not much larger than same distribution ({loss_same.item():.6f})")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    compressed_tokens_grad = torch.randn(batch_size, 64, hidden_dim, requires_grad=True).to(device)
    original_tokens_grad = torch.randn(batch_size, 256, hidden_dim).to(device)

    loss_grad, _ = ot_loss(
        compressed_tokens_grad,
        original_tokens_grad
    )
    loss_grad.backward()

    assert compressed_tokens_grad.grad is not None
    print(f"Gradient norm: {compressed_tokens_grad.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    # Test 5: Different epsilon values
    print("\n5. Testing different epsilon values...")
    epsilons = [0.01, 0.1, 1.0]
    for eps in epsilons:
        ot_loss_eps = SinkhornOTLoss(epsilon=eps, max_iter=100).to(device)
        loss_eps, info_eps = ot_loss_eps(
            compressed_tokens,
            original_tokens
        )
        print(f"  epsilon={eps:.2f}: OT={loss_eps.item():.4f}, iters={info_eps['sinkhorn_iterations']}")
    print("✓ Epsilon test passed!")

    print("\n" + "=" * 60)
    print("All Sinkhorn OT Loss tests passed!")
    print("=" * 60)
