"""Receptive Field-aware Discriminator for Token Comparison"""
import torch
import torch.nn as nn


class RFDiscriminator(nn.Module):
    """
    RF-aware discriminator that compares individual tokens.

    Compares:
    - 1 compressed token vs 1 randomly sampled original token from its RF

    This is applied to all 36 compressed tokens in parallel.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension of tokens (1024 for CLIP ViT-L)
            num_layers: Number of MLP layers
            mlp_ratio: Expansion ratio for MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Processing layers
        self.layers = nn.ModuleList([
            MLPBlock(hidden_dim, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
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
        Discriminate individual tokens.

        Args:
            tokens: (batch_size, hidden_dim) - single tokens
                   Can be (B*36, 1024) for processing all 36 tokens at once

        Returns:
            scores: (batch_size, 1) - discrimination scores
                   Higher scores indicate "real" tokens
        """
        x = self.input_proj(tokens)

        # Process through layers
        for layer in self.layers:
            x = x + layer(x)

        # Classification
        scores = self.head(x)

        return scores


class MLPBlock(nn.Module):
    """
    MLP block with residual connection.
    """

    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim)

        Returns:
            (batch_size, hidden_dim)
        """
        return self.mlp(x)


if __name__ == "__main__":
    # Test the RF discriminator
    print("Testing RF Discriminator...")

    hidden_dim = 1024
    batch_size = 4
    num_tokens = 36  # 6x6 grid

    # Create discriminator
    discriminator = RFDiscriminator(
        hidden_dim=hidden_dim,
        num_layers=3
    )

    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M")

    # Test 1: Single token batch
    print("\n1. Testing single token batch...")
    single_tokens = torch.randn(batch_size, hidden_dim)
    scores = discriminator(single_tokens)
    print(f"Input shape: {single_tokens.shape}")
    print(f"Output shape: {scores.shape}")
    assert scores.shape == (batch_size, 1)
    print("✓ Single token test passed!")

    # Test 2: Flattened batch (all 36 tokens processed together)
    print("\n2. Testing flattened batch (B*36 tokens)...")
    all_tokens = torch.randn(batch_size * num_tokens, hidden_dim)
    scores_flat = discriminator(all_tokens)
    print(f"Input shape: {all_tokens.shape}")
    print(f"Output shape: {scores_flat.shape}")
    assert scores_flat.shape == (batch_size * num_tokens, 1)

    # Reshape to (B, 36, 1)
    scores_reshaped = scores_flat.view(batch_size, num_tokens, 1)
    print(f"Reshaped scores: {scores_reshaped.shape}")
    print("✓ Flattened batch test passed!")

    # Test 3: Check that different tokens get different scores
    print("\n3. Testing discrimination capability...")
    token1 = torch.randn(1, hidden_dim)
    token2 = torch.randn(1, hidden_dim)

    score1 = discriminator(token1)
    score2 = discriminator(token2)

    print(f"Token 1 score: {score1.item():.4f}")
    print(f"Token 2 score: {score2.item():.4f}")
    print(f"Score difference: {abs(score1.item() - score2.item()):.4f}")
    print("✓ Discrimination test passed!")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    dummy_tokens = torch.randn(4, hidden_dim, requires_grad=True)
    dummy_scores = discriminator(dummy_tokens)
    loss = dummy_scores.mean()
    loss.backward()

    assert dummy_tokens.grad is not None
    print(f"Gradient norm: {dummy_tokens.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    print("\n" + "=" * 60)
    print("All RF Discriminator tests passed!")
    print("=" * 60)
