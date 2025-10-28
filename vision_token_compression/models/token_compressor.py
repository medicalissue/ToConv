"""Simple 1-Layer Convolution-based Token Compressor"""
import torch
import torch.nn as nn
import math


class TokenCompressor(nn.Module):
    """
    Simple 1-layer convolution-based token compressor.

    Uses a single stride-2 convolution followed by adaptive pooling
    to compress tokens to target grid size.
    """

    def __init__(
        self,
        input_grid_size: int,
        output_grid_size: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 512
    ):
        """
        Args:
            input_grid_size: Size of input token grid (e.g., 24 for 24x24)
            output_grid_size: Size of output token grid (e.g., 6 for 6x6)
            hidden_dim: Token dimension (e.g., 1024 for CLIP ViT-L)
            bottleneck_dim: Hidden dimension for compression (configurable)
        """
        super().__init__()

        self.input_grid_size = input_grid_size
        self.output_grid_size = output_grid_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Compression ratio
        self.compression_ratio = (input_grid_size ** 2) / (output_grid_size ** 2)

        # Single stride-2 convolution for initial compression
        self.compress_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                bottleneck_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(32, bottleneck_dim),
            nn.GELU()
        )

        # Adaptive pooling to exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_grid_size, output_grid_size))

        # Project back to original dimension
        self.project = nn.Conv2d(
            bottleneck_dim,
            hidden_dim,
            kernel_size=1,
            bias=True
        )

        # Learnable position embeddings for output tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, hidden_dim, output_grid_size, output_grid_size) * 0.02
        )

        # Store layer info for RF calculation
        self.layer_info = [{
            'kernel_size': 3,
            'stride': 2,
            'padding': 1
        }]

        # Calculate receptive field
        self.receptive_field_size = self._calculate_receptive_field()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress tokens.

        Args:
            tokens: Input tokens of shape (batch_size, num_patches, hidden_dim)
                   where num_patches = input_grid_size^2

        Returns:
            Compressed tokens of shape (batch_size, output_grid_size^2, hidden_dim)
        """
        batch_size = tokens.shape[0]

        # Reshape to 2D grid: (B, N, C) -> (B, C, H, W)
        tokens_2d = self._to_2d(tokens)

        # Compress
        x = self.compress_conv(tokens_2d)  # (B, bottleneck_dim, H/2, W/2)

        # Adaptive pool to exact size
        x = self.adaptive_pool(x)  # (B, bottleneck_dim, output_h, output_w)

        # Project back to original dimension
        x = self.project(x)  # (B, hidden_dim, output_h, output_w)

        # Add positional embeddings
        x = x + self.pos_embed

        # Reshape back to token format: (B, C, H, W) -> (B, N, C)
        compressed_tokens = self._to_1d(x)

        return compressed_tokens

    def _to_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token sequence to 2D grid.

        Args:
            tokens: (batch_size, num_patches, hidden_dim)

        Returns:
            2D grid: (batch_size, hidden_dim, grid_size, grid_size)
        """
        batch_size, num_patches, hidden_dim = tokens.shape
        grid_size = int(math.sqrt(num_patches))

        assert grid_size == self.input_grid_size, \
            f"Expected {self.input_grid_size}^2 patches, got {num_patches}"

        # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
        tokens_2d = tokens.reshape(batch_size, grid_size, grid_size, hidden_dim)
        tokens_2d = tokens_2d.permute(0, 3, 1, 2)

        return tokens_2d

    def _to_1d(self, tokens_2d: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D grid back to token sequence.

        Args:
            tokens_2d: (batch_size, hidden_dim, grid_size, grid_size)

        Returns:
            Token sequence: (batch_size, num_patches, hidden_dim)
        """
        batch_size, hidden_dim, grid_h, grid_w = tokens_2d.shape

        # (B, C, H, W) -> (B, H, W, C) -> (B, N, C)
        tokens = tokens_2d.permute(0, 2, 3, 1)
        tokens = tokens.reshape(batch_size, grid_h * grid_w, hidden_dim)

        return tokens

    def _calculate_receptive_field(self) -> int:
        """
        Calculate the theoretical receptive field size.

        For single stride-2, kernel-3 layer:
        RF = 1 + (kernel_size - 1) * 1 = 1 + 2 = 3

        Returns:
            Receptive field size (height/width in original grid)
        """
        rf = 1
        jump = 1

        for layer_info in reversed(self.layer_info):
            kernel_size = layer_info['kernel_size']
            stride = layer_info['stride']
            rf = rf + (kernel_size - 1) * jump
            jump = jump * stride

        return rf

    def get_receptive_field_size(self) -> int:
        """
        Get the receptive field size of the compressor.

        Returns:
            Receptive field size (each output token sees an RF×RF region in input)
        """
        return self.receptive_field_size

    def get_effective_receptive_field_mapping(self) -> torch.Tensor:
        """
        Compute which input tokens correspond to each output token's effective RF.

        Due to adaptive pooling, each output position corresponds to a
        grid-aligned region in the input.

        Returns:
            Mapping tensor: (output_grid_size, output_grid_size, rf_size, rf_size)
            containing input token indices for each output position
        """
        # Calculate stride in input grid
        stride = self.input_grid_size / self.output_grid_size

        # RF size in grid units
        rf_size = int(stride)

        # Create coordinate grids
        output_i, output_j = torch.meshgrid(
            torch.arange(self.output_grid_size),
            torch.arange(self.output_grid_size),
            indexing='ij'
        )

        di, dj = torch.meshgrid(
            torch.arange(rf_size),
            torch.arange(rf_size),
            indexing='ij'
        )

        # Compute starting positions for each output token
        start_i = (output_i.float() * stride).long().unsqueeze(-1).unsqueeze(-1)
        start_j = (output_j.float() * stride).long().unsqueeze(-1).unsqueeze(-1)

        # Expand relative positions
        di_expanded = di.unsqueeze(0).unsqueeze(0)
        dj_expanded = dj.unsqueeze(0).unsqueeze(0)

        # Compute absolute input positions
        input_i = start_i + di_expanded
        input_j = start_j + dj_expanded

        # Clamp to bounds
        input_i = torch.clamp(input_i, 0, self.input_grid_size - 1)
        input_j = torch.clamp(input_j, 0, self.input_grid_size - 1)

        # Convert to linear indices
        mapping = input_i * self.input_grid_size + input_j

        return mapping.long()


if __name__ == "__main__":
    # Test the simplified compressor
    print("Testing simplified Token Compressor...")

    batch_size = 4
    input_grid_size = 24
    output_grid_size = 6
    hidden_dim = 1024
    bottleneck_dim = 512

    compressor = TokenCompressor(
        input_grid_size=input_grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim
    )

    # Test forward pass
    num_patches = input_grid_size ** 2
    dummy_tokens = torch.randn(batch_size, num_patches, hidden_dim)
    compressed = compressor(dummy_tokens)

    print(f"Input shape: {dummy_tokens.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Compression ratio: {input_grid_size**2 / output_grid_size**2:.1f}x")
    print(f"Bottleneck dim: {bottleneck_dim}")
    assert compressed.shape == (batch_size, output_grid_size ** 2, hidden_dim)

    # Test receptive field
    theoretical_rf = compressor.get_receptive_field_size()
    print(f"Theoretical receptive field: {theoretical_rf}×{theoretical_rf}")

    rf_mapping = compressor.get_effective_receptive_field_mapping()
    print(f"Effective RF mapping shape: {rf_mapping.shape}")
    print(f"Effective RF size per output: {rf_mapping.shape[2]}×{rf_mapping.shape[3]}")

    # Count parameters
    total_params = sum(p.numel() for p in compressor.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    print("✓ Simplified Token Compressor test passed!")
