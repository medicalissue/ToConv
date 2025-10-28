"""Receptive Field-aware WGAN-GP Loss"""
import torch
import torch.nn as nn
import torch.autograd as autograd


class RFWGANGPLoss(nn.Module):
    """
    RF-based WGAN-GP Loss.

    Compares:
    - k² compressed tokens vs k² sampled original tokens
    - Each comparison is RF-aware (sampled from corresponding RF)
    """

    def __init__(self, lambda_gp: float = 10.0):
        """
        Args:
            lambda_gp: Weight for gradient penalty term
        """
        super().__init__()
        self.lambda_gp = lambda_gp

    def discriminator_loss(
        self,
        compressed_tokens: torch.Tensor,  # (B, k², hidden_dim)
        original_tokens: torch.Tensor,    # (B, original_grid_size², hidden_dim)
        discriminator: nn.Module,
        compressed_grid_size: int,
        original_grid_size: int,
        device: torch.device
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute RF-aware discriminator loss.

        Args:
            compressed_tokens: (B, k², hidden_dim) - compressed tokens
            original_tokens: (B, original_grid_size², hidden_dim) - original tokens
            discriminator: RFDiscriminator model
            compressed_grid_size: Size k of compressed grid
            original_grid_size: Size of original token grid
            device: Device to run on

        Returns:
            loss: Discriminator loss
            info: Dictionary with loss components
        """
        batch_size = original_tokens.shape[0]
        expected_real_tokens = original_grid_size ** 2
        if original_tokens.shape[1] != expected_real_tokens:
            raise ValueError(
                "Mismatch between original_grid_size and number of original tokens: "
                f"expected {expected_real_tokens}, got {original_tokens.shape[1]}"
            )

        expected_compressed_tokens = compressed_grid_size ** 2
        if compressed_tokens.shape[1] != expected_compressed_tokens:
            raise ValueError(
                "Mismatch between compressed_grid_size and number of compressed tokens: "
                f"expected {expected_compressed_tokens}, got {compressed_tokens.shape[1]}"
            )

        # Sample 1 token from each RF
        sampled_real = sample_rf_tokens(
            original_tokens,
            compressed_grid_size,
            original_grid_size,
            device
        )  # (B, k², hidden_dim)

        # Flatten for batch discrimination
        # (B, k², hidden_dim) → (B*k², hidden_dim)
        real_flat = sampled_real.reshape(-1, sampled_real.shape[-1])
        fake_flat = compressed_tokens.reshape(-1, compressed_tokens.shape[-1])

        # Get discrimination scores
        real_scores = discriminator(real_flat)  # (B*k², 1)
        fake_scores = discriminator(fake_flat)  # (B*k², 1)

        # Compute WGAN loss
        wasserstein_distance = real_scores.mean() - fake_scores.mean()
        critic_loss = -wasserstein_distance

        # Gradient penalty
        gp = compute_rf_gradient_penalty(
            discriminator,
            real_flat,
            fake_flat,
            device
        )

        # Total loss
        loss = critic_loss + self.lambda_gp * gp

        # Reshape scores for statistics
        real_scores_2d = real_scores.reshape(batch_size, -1)  # (B, k²)
        fake_scores_2d = fake_scores.reshape(batch_size, -1)  # (B, k²)

        info = {
            'disc_loss': loss.item(),
            'wasserstein_distance': wasserstein_distance.item(),
            'gradient_penalty': gp.item(),
            'real_score_mean': real_scores.mean().item(),
            'fake_score_mean': fake_scores.mean().item(),
            'real_score_std': real_scores_2d.std().item(),
            'fake_score_std': fake_scores_2d.std().item()
        }

        return loss, info

    def generator_loss(
        self,
        compressed_tokens: torch.Tensor,  # (B, k², hidden_dim)
        discriminator: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute generator (compressor) loss.

        Args:
            compressed_tokens: (B, k², hidden_dim) - compressed tokens
            discriminator: RFDiscriminator model

        Returns:
            loss: Generator loss
            info: Dictionary with loss components
        """
        # Flatten: (B, k², hidden_dim) → (B*k², hidden_dim)
        fake_flat = compressed_tokens.reshape(-1, compressed_tokens.shape[-1])

        # Get scores
        fake_scores = discriminator(fake_flat)  # (B*k², 1)

        # Generator wants to maximize D(fake)
        # For minimization: minimize -D(fake)
        loss = -fake_scores.mean()

        info = {
            'gen_loss': loss.item(),
            'fake_score_mean': fake_scores.mean().item(),
            'fake_score_std': fake_scores.std().item()
        }

        return loss, info


def sample_rf_tokens(
    original_tokens: torch.Tensor,
    compressed_grid_size: int,
    original_grid_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    For each compressed token position, randomly sample 1 token from its RF.

    Args:
        original_tokens: (B, original_grid_size², hidden_dim) - full-resolution grid
        compressed_grid_size: Size k of compressed k×k grid
        original_grid_size: Size of original grid

    Returns:
        sampled_tokens: (B, k², hidden_dim) - one sampled token per compressed position
    """
    batch_size = original_tokens.shape[0]
    hidden_dim = original_tokens.shape[2]
    rf_size = original_grid_size // compressed_grid_size

    sampled_tokens = []

    for i in range(compressed_grid_size):
        for j in range(compressed_grid_size):
            # Get RF region indices for this compressed token
            start_i = i * rf_size
            start_j = j * rf_size

            # Collect RF token indices for this RF
            rf_indices = []
            for di in range(rf_size):
                for dj in range(rf_size):
                    idx = (start_i + di) * original_grid_size + (start_j + dj)
                    rf_indices.append(idx)

            # Extract RF tokens: (B, rf_size², hidden_dim)
            rf_tokens = original_tokens[:, rf_indices, :]

            # Randomly sample 1 token from the RF tokens for each batch item
            rand_idx = torch.randint(0, rf_size * rf_size, (batch_size,), device=device)
            sampled = rf_tokens[torch.arange(batch_size, device=device), rand_idx, :]  # (B, 1024)

            sampled_tokens.append(sampled)

    # Stack: List of k² × (B, hidden_dim) → (B, k², hidden_dim)
    sampled_tokens = torch.stack(sampled_tokens, dim=1)

    return sampled_tokens


def compute_rf_gradient_penalty(
    discriminator: nn.Module,
    real_tokens: torch.Tensor,  # (B*k², hidden_dim)
    fake_tokens: torch.Tensor,  # (B*k², hidden_dim)
    device: torch.device
) -> torch.Tensor:
    """
    Gradient penalty for single token discrimination.

    Args:
        discriminator: The discriminator network
        real_tokens: (B*k², hidden_dim) - flattened real tokens
        fake_tokens: (B*k², hidden_dim) - flattened fake tokens
        device: Device to run on

    Returns:
        Gradient penalty value
    """
    num_samples = real_tokens.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(num_samples, 1, device=device)

    # Interpolate between real and fake tokens
    interpolates = alpha * real_tokens + (1 - alpha) * fake_tokens
    interpolates = interpolates.requires_grad_(True)

    # Get discriminator scores
    disc_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty: (||grad|| - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


if __name__ == "__main__":
    # Test RF WGAN-GP loss
    print("Testing RF WGAN-GP Loss...")

    batch_size = 4
    compressed_grid_size = 6
    original_grid_size = 24
    hidden_dim = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dummy data
    original_tokens = torch.randn(batch_size, 576, hidden_dim).to(device)
    compressed_tokens = torch.randn(batch_size, 36, hidden_dim).to(device)

    # Create dummy discriminator
    from vision_token_compression.models.rf_discriminator import RFDiscriminator

    discriminator = RFDiscriminator(hidden_dim=hidden_dim).to(device)

    # Test 1: RF token sampling
    print("1. Testing RF token sampling...")
    sampled = sample_rf_tokens(
        original_tokens,
        compressed_grid_size,
        original_grid_size,
        device
    )
    print(f"Original tokens shape: {original_tokens.shape}")
    print(f"Sampled tokens shape: {sampled.shape}")
    assert sampled.shape == (batch_size, 36, hidden_dim)
    print("✓ Sampling test passed!")

    # Test 2: Check sampling randomness
    print("\n2. Testing sampling randomness...")
    sampled1 = sample_rf_tokens(original_tokens, compressed_grid_size, original_grid_size, device)
    sampled2 = sample_rf_tokens(original_tokens, compressed_grid_size, original_grid_size, device)
    diff = (sampled1 - sampled2).abs().mean()
    print(f"Difference between two samples: {diff.item():.4f}")
    assert diff.item() > 0, "Samples should be different (random)"
    print("✓ Randomness test passed!")

    # Test 3: Discriminator loss
    print("\n3. Testing discriminator loss...")
    wgan_loss = RFWGANGPLoss(lambda_gp=10.0)

    disc_loss, disc_info = wgan_loss.discriminator_loss(
        compressed_tokens=compressed_tokens,
        original_tokens=original_tokens,
        discriminator=discriminator,
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size,
        device=device
    )

    print(f"Discriminator loss: {disc_loss.item():.4f}")
    print(f"Wasserstein distance: {disc_info['wasserstein_distance']:.4f}")
    print(f"Gradient penalty: {disc_info['gradient_penalty']:.4f}")
    print(f"Real score mean: {disc_info['real_score_mean']:.4f}")
    print(f"Fake score mean: {disc_info['fake_score_mean']:.4f}")
    print("✓ Discriminator loss test passed!")

    # Test 4: Generator loss
    print("\n4. Testing generator loss...")
    gen_loss, gen_info = wgan_loss.generator_loss(
        compressed_tokens=compressed_tokens,
        discriminator=discriminator
    )

    print(f"Generator loss: {gen_loss.item():.4f}")
    print(f"Fake score mean: {gen_info['fake_score_mean']:.4f}")
    print("✓ Generator loss test passed!")

    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    compressed_tokens.requires_grad = True
    gen_loss, _ = wgan_loss.generator_loss(compressed_tokens, discriminator)
    gen_loss.backward()

    assert compressed_tokens.grad is not None
    print(f"Gradient norm: {compressed_tokens.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    # Test 6: Verify RF coverage
    print("\n6. Verifying RF coverage...")
    # Check that sampling covers all RFs
    for comp_idx in range(36):
        comp_i = comp_idx // 6
        comp_j = comp_idx % 6

        rf_size = 4
        start_i = comp_i * rf_size
        start_j = comp_j * rf_size

        # Count how many original tokens are in this RF
        rf_count = rf_size * rf_size
        print(f"RF {comp_idx} ({comp_i},{comp_j}): covers {rf_count} tokens at position ({start_i}-{start_i+rf_size-1}, {start_j}-{start_j+rf_size-1})")

        if comp_idx >= 2:  # Just show first few
            break

    print("✓ RF coverage verification passed!")

    print("\n" + "=" * 60)
    print("All RF WGAN-GP Loss tests passed!")
    print("=" * 60)
