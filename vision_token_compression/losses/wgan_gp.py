"""WGAN-GP Loss with Gradient Penalty"""
import torch
import torch.nn as nn
import torch.autograd as autograd


class WGANGPLoss(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty loss.

    The discriminator tries to maximize the distance between real and fake scores.
    The generator tries to minimize this distance.
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
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
        gradient_penalty: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.

        Args:
            real_scores: Scores for real tokens (batch_size, 1)
            fake_scores: Scores for fake/compressed tokens (batch_size, 1)
            gradient_penalty: Gradient penalty term

        Returns:
            loss: Discriminator loss
            info: Dictionary with loss components
        """
        # Wasserstein distance: maximize D(real) - D(fake)
        # For minimization: minimize -D(real) + D(fake)
        wasserstein_distance = real_scores.mean() - fake_scores.mean()
        critic_loss = -wasserstein_distance

        # Total loss with gradient penalty
        loss = critic_loss + self.lambda_gp * gradient_penalty

        info = {
            'disc_loss': loss.item(),
            'wasserstein_distance': wasserstein_distance.item(),
            'gradient_penalty': gradient_penalty.item(),
            'real_score': real_scores.mean().item(),
            'fake_score': fake_scores.mean().item()
        }

        return loss, info

    def generator_loss(
        self,
        fake_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute generator (compressor) loss.

        Args:
            fake_scores: Scores for compressed tokens (batch_size, 1)

        Returns:
            loss: Generator loss
            info: Dictionary with loss components
        """
        # Generator wants to maximize D(fake)
        # For minimization: minimize -D(fake)
        loss = -fake_scores.mean()

        info = {
            'gen_loss': loss.item(),
            'fake_score': fake_scores.mean().item()
        }

        return loss, info


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_tokens: torch.Tensor,
    fake_tokens: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    The gradient penalty encourages the discriminator to have gradients
    with norm close to 1 everywhere.

    Args:
        discriminator: The discriminator network
        real_tokens: Real token sequences (batch_size, num_tokens, hidden_dim)
        fake_tokens: Fake token sequences (batch_size, num_tokens, hidden_dim)
        device: Device to run computation on

    Returns:
        Gradient penalty value
    """
    batch_size = real_tokens.size(0)

    # Random interpolation coefficient for each sample in batch
    # Shape: (batch_size, 1, 1) for broadcasting
    alpha = torch.rand(batch_size, 1, 1, device=device)

    # Interpolate between real and fake tokens
    # interpolates = alpha * real + (1 - alpha) * fake
    interpolates = alpha * real_tokens + (1 - alpha) * fake_tokens
    interpolates = interpolates.requires_grad_(True)

    # Get discriminator scores for interpolated tokens
    disc_interpolates = discriminator(interpolates)

    # Compute gradients of scores w.r.t. interpolated tokens
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients: (batch_size, num_tokens, hidden_dim) -> (batch_size, -1)
    gradients = gradients.reshape(batch_size, -1)

    # Compute gradient norm for each sample
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient penalty: (||grad|| - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def compute_gradient_penalty_efficient(
    discriminator: nn.Module,
    real_tokens: torch.Tensor,
    fake_tokens: torch.Tensor,
    device: torch.device,
    num_interpolates: int = 1
) -> torch.Tensor:
    """
    Memory-efficient version of gradient penalty computation.
    Computes gradient penalty on a subset of interpolations.

    Args:
        discriminator: The discriminator network
        real_tokens: Real token sequences
        fake_tokens: Fake token sequences
        device: Device to run computation on
        num_interpolates: Number of interpolations to use (for memory efficiency)

    Returns:
        Gradient penalty value
    """
    batch_size = real_tokens.size(0)

    # Sample subset if needed for memory efficiency
    if num_interpolates < batch_size:
        indices = torch.randperm(batch_size)[:num_interpolates]
        real_subset = real_tokens[indices]
        fake_subset = fake_tokens[indices]
    else:
        real_subset = real_tokens
        fake_subset = fake_tokens

    return compute_gradient_penalty(discriminator, real_subset, fake_subset, device)


class SpectralNormDiscriminatorLoss(nn.Module):
    """
    Alternative: Discriminator loss with spectral normalization instead of GP.
    This is more memory efficient but may be less stable.
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Discriminator loss without gradient penalty."""
        wasserstein_distance = real_scores.mean() - fake_scores.mean()
        loss = -wasserstein_distance

        info = {
            'disc_loss': loss.item(),
            'wasserstein_distance': wasserstein_distance.item(),
            'real_score': real_scores.mean().item(),
            'fake_score': fake_scores.mean().item()
        }

        return loss, info

    def generator_loss(
        self,
        fake_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Generator loss."""
        loss = -fake_scores.mean()

        info = {
            'gen_loss': loss.item(),
            'fake_score': fake_scores.mean().item()
        }

        return loss, info


if __name__ == "__main__":
    # Test WGAN-GP loss
    print("Testing WGAN-GP Loss...")

    batch_size = 4
    num_tokens = 36
    hidden_dim = 1024

    # Create dummy discriminator
    from vision_token_compression.models.discriminator import Discriminator

    discriminator = Discriminator(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim
    )

    # Create dummy data
    real_tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    fake_tokens = torch.randn(batch_size, num_tokens, hidden_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = discriminator.to(device)
    real_tokens = real_tokens.to(device)
    fake_tokens = fake_tokens.to(device)

    # Get scores
    real_scores = discriminator(real_tokens)
    fake_scores = discriminator(fake_tokens)

    print(f"Real scores shape: {real_scores.shape}")
    print(f"Fake scores shape: {fake_scores.shape}")

    # Compute gradient penalty
    gp = compute_gradient_penalty(discriminator, real_tokens, fake_tokens, device)
    print(f"Gradient penalty: {gp.item():.4f}")

    # Compute losses
    wgan_loss = WGANGPLoss(lambda_gp=10.0)

    disc_loss, disc_info = wgan_loss.discriminator_loss(real_scores, fake_scores, gp)
    gen_loss, gen_info = wgan_loss.generator_loss(fake_scores)

    print(f"\nDiscriminator loss: {disc_loss.item():.4f}")
    print(f"Discriminator info: {disc_info}")

    print(f"\nGenerator loss: {gen_loss.item():.4f}")
    print(f"Generator info: {gen_info}")

    print("\n✓ WGAN-GP Loss test passed!")
