"""WGAN-GP Discriminator for Token Distribution Matching"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator for WGAN-GP that distinguishes between
    original tokens and compressed tokens.

    Uses a simple MLP architecture to process token sequences.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_layers: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            num_tokens: Number of tokens in the sequence
            hidden_dim: Hidden dimension of tokens
            num_layers: Number of transformer/MLP layers
            mlp_ratio: Ratio for MLP hidden dimension expansion
            dropout: Dropout rate
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # Token-wise processing layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                DiscriminatorBlock(
                    hidden_dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
            )

        # Global pooling and final classification
        self.norm = nn.LayerNorm(hidden_dim)

        # Final MLP for discrimination
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between real and compressed tokens.

        Args:
            tokens: Token sequence of shape (batch_size, num_tokens, hidden_dim)

        Returns:
            Discrimination scores of shape (batch_size, 1)
            Higher values indicate "real" tokens
        """
        x = tokens

        # Process through layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        # Global average pooling across tokens
        # (batch_size, num_tokens, hidden_dim) -> (batch_size, hidden_dim)
        x = x.mean(dim=1)

        # Final discrimination score
        score = self.head(x)

        return score


class DiscriminatorBlock(nn.Module):
    """
    Single discriminator block with MLP and residual connection.
    """

    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_tokens, hidden_dim)

        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        # Residual connection
        x = x + self.mlp(self.norm1(x))
        return x


class MultiScaleDiscriminator(nn.Module):
    """
    Optional: Multi-scale discriminator that processes tokens at different granularities.
    This can help capture both local and global token statistics.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_scales: int = 3,
        num_layers_per_scale: int = 2
    ):
        """
        Args:
            num_tokens: Number of tokens in the sequence
            hidden_dim: Hidden dimension of tokens
            num_scales: Number of different scales to process
            num_layers_per_scale: Number of layers per scale
        """
        super().__init__()

        self.discriminators = nn.ModuleList([
            Discriminator(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                num_layers=num_layers_per_scale,
                dropout=0.1
            )
            for _ in range(num_scales)
        ])

        # Different pooling strategies for different scales
        self.pooling_layers = nn.ModuleList()
        for i in range(num_scales):
            if i == 0:
                # No pooling for first scale (full resolution)
                self.pooling_layers.append(nn.Identity())
            else:
                # Average pooling for other scales
                pool_size = 2 ** i
                self.pooling_layers.append(
                    nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
                )

    def forward(self, tokens: torch.Tensor) -> list:
        """
        Multi-scale discrimination.

        Args:
            tokens: (batch_size, num_tokens, hidden_dim)

        Returns:
            List of discrimination scores from different scales
        """
        scores = []

        for disc, pool in zip(self.discriminators, self.pooling_layers):
            # Pool tokens if needed
            # (B, N, C) -> (B, C, N) -> pool -> (B, C, N') -> (B, N', C)
            x = tokens.transpose(1, 2)
            x = pool(x)
            x = x.transpose(1, 2)

            # Discriminate
            score = disc(x)
            scores.append(score)

        return scores


if __name__ == "__main__":
    # Test the discriminator
    batch_size = 4
    num_tokens = 36  # e.g., 6x6 compressed tokens
    hidden_dim = 1024

    discriminator = Discriminator(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        num_layers=3
    )

    # Test forward pass
    dummy_tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    scores = discriminator(dummy_tokens)

    print(f"Input shape: {dummy_tokens.shape}")
    print(f"Output scores shape: {scores.shape}")
    assert scores.shape == (batch_size, 1)
    print(f"Sample scores: {scores.squeeze().detach()}")

    # Test multi-scale discriminator
    print("\nTesting multi-scale discriminator...")
    multi_disc = MultiScaleDiscriminator(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        num_scales=2
    )

    multi_scores = multi_disc(dummy_tokens)
    print(f"Number of scales: {len(multi_scores)}")
    for i, score in enumerate(multi_scores):
        print(f"Scale {i} scores shape: {score.shape}")

    print("✓ Discriminator test passed!")
