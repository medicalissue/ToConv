"""AutoEncoder Decoder for Token Reconstruction"""
import torch
import torch.nn as nn
import math


class AutoEncoderDecoder(nn.Module):
    """
    Decoder that reconstructs original tokens from compressed tokens.
    Each compressed token should predict tokens in its receptive field.
    """

    def __init__(
        self,
        compressed_grid_size: int,
        original_grid_size: int,
        hidden_dim: int,
        num_layers: int = 3,
        use_attention: bool = False
    ):
        """
        Args:
            compressed_grid_size: Size of compressed token grid (k for kxk)
            original_grid_size: Size of original token grid
            hidden_dim: Hidden dimension of tokens
            num_layers: Number of decoder layers
            use_attention: Whether to use attention for decoding
        """
        super().__init__()

        self.compressed_grid_size = compressed_grid_size
        self.original_grid_size = original_grid_size
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Calculate receptive field size
        self.rf_size = original_grid_size // compressed_grid_size

        if use_attention:
            # Attention-based decoder
            self.decoder = AttentionDecoder(
                compressed_grid_size=compressed_grid_size,
                original_grid_size=original_grid_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        else:
            # Convolution-based decoder (upsampling)
            self.decoder = ConvDecoder(
                compressed_grid_size=compressed_grid_size,
                original_grid_size=original_grid_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original tokens from compressed tokens.

        Args:
            compressed_tokens: (batch_size, k^2, hidden_dim)

        Returns:
            Reconstructed tokens: (batch_size, original_grid_size^2, hidden_dim)
        """
        return self.decoder(compressed_tokens)


class ConvDecoder(nn.Module):
    """
    Convolutional transpose-based decoder for upsampling.
    """

    def __init__(
        self,
        compressed_grid_size: int,
        original_grid_size: int,
        hidden_dim: int,
        num_layers: int = 3
    ):
        super().__init__()

        self.compressed_grid_size = compressed_grid_size
        self.original_grid_size = original_grid_size
        self.hidden_dim = hidden_dim

        # Calculate total upsampling ratio
        self.upsampling_ratio = original_grid_size / compressed_grid_size

        # Build upsampling layers
        self.layers = nn.ModuleList()
        current_size = compressed_grid_size

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: reach exact target size
                target_size = original_grid_size
            else:
                # Intermediate layers: gradual upsampling
                target_size = int(current_size * (self.upsampling_ratio ** (1 / num_layers)))

            # Calculate stride and kernel size for upsampling
            stride = max(1, target_size // current_size)
            kernel_size = stride * 2  # Typically 2x stride for smooth upsampling

            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            padding = kernel_size // 2
            output_padding = (target_size - current_size * stride)

            layer_modules = []

            # Transposed convolution for upsampling
            layer_modules.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=max(0, output_padding),
                    bias=False
                )
            )

            # Normalization
            layer_modules.append(nn.GroupNorm(32, hidden_dim))

            # Activation (except last layer)
            if i < num_layers - 1:
                layer_modules.append(nn.GELU())

            self.layers.append(nn.Sequential(*layer_modules))
            current_size = (current_size - 1) * stride - 2 * padding + kernel_size + output_padding

        # Adaptive interpolation to ensure exact size
        self.adaptive_upsample = nn.Upsample(
            size=(original_grid_size, original_grid_size),
            mode='bilinear',
            align_corners=False
        )

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_tokens: (batch_size, k^2, hidden_dim)

        Returns:
            (batch_size, original_grid_size^2, hidden_dim)
        """
        batch_size = compressed_tokens.shape[0]

        # Reshape to 2D: (B, N, C) -> (B, C, H, W)
        x = compressed_tokens.view(
            batch_size,
            self.compressed_grid_size,
            self.compressed_grid_size,
            self.hidden_dim
        ).permute(0, 3, 1, 2)

        # Upsample through layers
        for layer in self.layers:
            x = layer(x)

        # Ensure exact output size
        x = self.adaptive_upsample(x)

        # Reshape back to tokens: (B, C, H, W) -> (B, N, C)
        reconstructed = x.permute(0, 2, 3, 1).reshape(
            batch_size,
            self.original_grid_size ** 2,
            self.hidden_dim
        )

        return reconstructed


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder that uses cross-attention to reconstruct tokens.
    """

    def __init__(
        self,
        compressed_grid_size: int,
        original_grid_size: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 8
    ):
        super().__init__()

        self.compressed_grid_size = compressed_grid_size
        self.original_grid_size = original_grid_size
        self.hidden_dim = hidden_dim

        # Learnable queries for each position in the original grid
        self.position_queries = nn.Parameter(
            torch.randn(1, original_grid_size ** 2, hidden_dim) * 0.02
        )

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_tokens: (batch_size, k^2, hidden_dim)

        Returns:
            (batch_size, original_grid_size^2, hidden_dim)
        """
        batch_size = compressed_tokens.shape[0]

        # Expand position queries for batch
        queries = self.position_queries.expand(batch_size, -1, -1)

        # Apply cross-attention layers
        x = queries
        for layer in self.layers:
            x = layer(x, compressed_tokens)

        x = self.norm(x)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for attending compressed tokens.
    """

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

    def forward(
        self,
        queries: torch.Tensor,
        keys_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch_size, num_queries, hidden_dim)
            keys_values: (batch_size, num_kv, hidden_dim)

        Returns:
            (batch_size, num_queries, hidden_dim)
        """
        # Cross-attention with residual
        attn_out, _ = self.cross_attn(
            self.norm1(queries),
            keys_values,
            keys_values
        )
        queries = queries + attn_out

        # MLP with residual
        queries = queries + self.mlp(self.norm2(queries))

        return queries


if __name__ == "__main__":
    # Test the decoder
    batch_size = 4
    compressed_grid_size = 6
    original_grid_size = 24
    hidden_dim = 1024

    # Test Conv Decoder
    print("Testing Conv Decoder...")
    conv_decoder = ConvDecoder(
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size,
        hidden_dim=hidden_dim,
        num_layers=3
    )

    compressed_tokens = torch.randn(batch_size, compressed_grid_size ** 2, hidden_dim)
    reconstructed = conv_decoder(compressed_tokens)

    print(f"Input shape: {compressed_tokens.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    assert reconstructed.shape == (batch_size, original_grid_size ** 2, hidden_dim)
    print("✓ Conv Decoder test passed!")

    # Test Attention Decoder
    print("\nTesting Attention Decoder...")
    attn_decoder = AttentionDecoder(
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size,
        hidden_dim=hidden_dim,
        num_layers=2
    )

    reconstructed_attn = attn_decoder(compressed_tokens)
    print(f"Reconstructed shape (attention): {reconstructed_attn.shape}")
    assert reconstructed_attn.shape == (batch_size, original_grid_size ** 2, hidden_dim)
    print("✓ Attention Decoder test passed!")

    # Test full AutoEncoder
    print("\nTesting AutoEncoder Decoder...")
    ae_decoder = AutoEncoderDecoder(
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size,
        hidden_dim=hidden_dim,
        use_attention=False
    )

    final_recon = ae_decoder(compressed_tokens)
    print(f"Final reconstruction shape: {final_recon.shape}")
    print("✓ All AutoEncoder tests passed!")
