"""Receptive Field-aware AutoEncoder Decoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RFAutoEncoderDecoder(nn.Module):
    """
    RF-aware decoder: each compressed token reconstructs its 4×4 RF.

    Input: (B, 36, 1024) - 36 compressed tokens
    Output: (B, 36, 16, 1024) - 36 RFs, each with 16 reconstructed tokens
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        rf_size: int = 4,
        num_layers: int = 3,
        use_conv: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Token dimension (1024 for CLIP ViT-L)
            rf_size: Receptive field size (4 for 4×4)
            num_layers: Number of decoder layers
            use_conv: Use 2D conv upsampling vs MLP expansion
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rf_size = rf_size
        self.num_rf_tokens = rf_size * rf_size  # 16
        self.use_conv = use_conv

        if use_conv:
            # 2D Conv upsampling: 1×1 → 4×4
            self.decoder = self._build_conv_decoder(hidden_dim, dropout)
        else:
            # MLP-based expansion
            self.decoder = self._build_mlp_decoder(hidden_dim, dropout)

    def _build_conv_decoder(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Build 2D convolutional decoder."""
        return nn.Sequential(
            # Input: (B*36, C, 1, 1)
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            # Now: (B*36, C, 2, 2)

            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            # Now: (B*36, C, 4, 4)

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

    def _build_mlp_decoder(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Build MLP-based decoder."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            nn.LayerNorm(hidden_dim * 16),
            nn.GELU()
        )

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct RF tokens from compressed tokens.

        Args:
            compressed_tokens: (B, 36, 1024) - compressed tokens

        Returns:
            reconstructed_rfs: (B, 36, 16, 1024)
                - For each of 36 compressed tokens
                - Reconstruct its 16 RF tokens
        """
        batch_size, num_compressed, hidden_dim = compressed_tokens.shape

        if self.use_conv:
            # Reshape for 2D conv: (B, 36, C) → (B*36, C, 1, 1)
            x = compressed_tokens.view(-1, hidden_dim, 1, 1)

            # Upsample: (B*36, C, 1, 1) → (B*36, C, 4, 4)
            x = self.decoder(x)

            # Reshape: (B*36, C, 4, 4) → (B*36, 4, 4, C) → (B, 36, 16, C)
            x = x.permute(0, 2, 3, 1)  # (B*36, 4, 4, C)
            x = x.reshape(batch_size, num_compressed, self.num_rf_tokens, hidden_dim)

        else:
            # MLP: (B, 36, C) → (B, 36, C*16)
            x = self.decoder(compressed_tokens)

            # Reshape: (B, 36, C*16) → (B, 36, 16, C)
            x = x.view(batch_size, num_compressed, self.num_rf_tokens, hidden_dim)

        return x


class RFAutoEncoderDecoderWithAttention(nn.Module):
    """
    Alternative RF decoder using attention mechanism.

    Each compressed token attends to learned queries for its RF tokens.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        rf_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Token dimension
            rf_size: Receptive field size
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rf_size = rf_size
        self.num_rf_tokens = rf_size * rf_size

        # Learnable queries for each RF position
        # Shape: (16, 1024) - one query per RF token
        self.rf_queries = nn.Parameter(
            torch.randn(self.num_rf_tokens, hidden_dim) * 0.02
        )

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_tokens: (B, 36, 1024)

        Returns:
            reconstructed_rfs: (B, 36, 16, 1024)
        """
        batch_size, num_compressed, hidden_dim = compressed_tokens.shape

        # Expand queries for each compressed token
        # (16, C) → (B, 36, 16, C)
        queries = self.rf_queries.unsqueeze(0).unsqueeze(0)  # (1, 1, 16, C)
        queries = queries.expand(batch_size, num_compressed, -1, -1)

        # Reshape for processing: (B, 36, 16, C) → (B*36, 16, C)
        queries_flat = queries.reshape(-1, self.num_rf_tokens, hidden_dim)

        # Expand compressed tokens: (B, 36, C) → (B*36, 1, C)
        keys_values = compressed_tokens.view(-1, 1, hidden_dim)

        # Apply attention layers
        x = queries_flat
        for layer in self.layers:
            x = layer(x, keys_values)

        x = self.norm(x)

        # Reshape: (B*36, 16, C) → (B, 36, 16, C)
        x = x.view(batch_size, num_compressed, self.num_rf_tokens, hidden_dim)

        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for RF token reconstruction."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (batch_size, 16, hidden_dim)
            keys_values: (batch_size, 1, hidden_dim)

        Returns:
            (batch_size, 16, hidden_dim)
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(
            self.norm1(queries),
            keys_values,
            keys_values
        )
        queries = queries + attn_out

        # MLP
        queries = queries + self.mlp(self.norm2(queries))

        return queries


if __name__ == "__main__":
    # Test the RF AutoEncoder decoder
    print("Testing RF AutoEncoder Decoder...")

    batch_size = 4
    num_compressed = 36  # 6x6 grid
    hidden_dim = 1024
    rf_size = 4

    # Test 1: Conv-based decoder
    print("\n1. Testing Conv-based decoder...")
    conv_decoder = RFAutoEncoderDecoder(
        hidden_dim=hidden_dim,
        rf_size=rf_size,
        num_layers=3,
        use_conv=True
    )

    compressed = torch.randn(batch_size, num_compressed, hidden_dim)
    reconstructed = conv_decoder(compressed)

    print(f"Input shape: {compressed.shape}")
    print(f"Output shape: {reconstructed.shape}")
    assert reconstructed.shape == (batch_size, num_compressed, 16, hidden_dim)
    print(f"Decoder parameters: {sum(p.numel() for p in conv_decoder.parameters()) / 1e6:.2f}M")
    print("✓ Conv decoder test passed!")

    # Test 2: MLP-based decoder
    print("\n2. Testing MLP-based decoder...")
    mlp_decoder = RFAutoEncoderDecoder(
        hidden_dim=hidden_dim,
        rf_size=rf_size,
        num_layers=3,
        use_conv=False
    )

    reconstructed_mlp = mlp_decoder(compressed)
    print(f"Output shape: {reconstructed_mlp.shape}")
    assert reconstructed_mlp.shape == (batch_size, num_compressed, 16, hidden_dim)
    print(f"Decoder parameters: {sum(p.numel() for p in mlp_decoder.parameters()) / 1e6:.2f}M")
    print("✓ MLP decoder test passed!")

    # Test 3: Attention-based decoder
    print("\n3. Testing Attention-based decoder...")
    attn_decoder = RFAutoEncoderDecoderWithAttention(
        hidden_dim=hidden_dim,
        rf_size=rf_size,
        num_layers=2
    )

    reconstructed_attn = attn_decoder(compressed)
    print(f"Output shape: {reconstructed_attn.shape}")
    assert reconstructed_attn.shape == (batch_size, num_compressed, 16, hidden_dim)
    print(f"Decoder parameters: {sum(p.numel() for p in attn_decoder.parameters()) / 1e6:.2f}M")
    print("✓ Attention decoder test passed!")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    dummy_compressed = torch.randn(2, 36, 1024, requires_grad=True)
    dummy_reconstructed = conv_decoder(dummy_compressed)
    loss = dummy_reconstructed.mean()
    loss.backward()

    assert dummy_compressed.grad is not None
    print(f"Gradient norm: {dummy_compressed.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    # Test 5: Check RF structure
    print("\n5. Checking RF structure...")
    single_token = compressed[0:1, 0:1, :]  # (1, 1, 1024)
    single_rf = conv_decoder(single_token)  # (1, 1, 16, 1024)
    print(f"Single compressed token: {single_token.shape}")
    print(f"Reconstructed RF: {single_rf.shape}")
    print(f"RF has {single_rf.shape[2]} tokens (expected 16)")
    assert single_rf.shape[2] == 16
    print("✓ RF structure test passed!")

    print("\n" + "=" * 60)
    print("All RF AutoEncoder Decoder tests passed!")
    print("=" * 60)
