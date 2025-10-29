"""Theoretical RF-based Token Compressor: Single Convolution, No Bottleneck, No Pooling"""
import torch
import torch.nn as nn
import math
from .partial_conv import PartialConv2d


class TokenCompressor(nn.Module):
    """
    Pure convolution-based token compressor using Theoretical Receptive Field
    with Partial Convolution for proper padding handling.

    Uses Partial Convolution instead of zero padding to avoid artifacts at boundaries.
    NO adaptive pooling, NO bottleneck - single convolution achieves exact output size.

    Supports 4 compression scenarios:
    1. 24×24 → 12×12 (stride=2, kernel=3, padding=1, RF=3)
    2. 24×24 → 8×8   (stride=3, kernel=3, padding=0, RF=3)
    3. 16×16 → 12×12 (stride=1, kernel=5, padding=0, RF=5)
    4. 16×16 → 8×8   (stride=2, kernel=3, padding=1, RF=3)
    """

    def __init__(
        self,
        input_grid_size: int,
        output_grid_size: int,
        hidden_dim: int = 1024
    ):
        """
        Args:
            input_grid_size: Size of input token grid (16 or 24)
            output_grid_size: Size of output token grid (8 or 12)
            hidden_dim: Token dimension (1024 for CLIP ViT-L)
        """
        super().__init__()

        self.input_grid_size = input_grid_size
        self.output_grid_size = output_grid_size
        self.hidden_dim = hidden_dim

        # Validate supported configurations
        valid_configs = [(24, 12), (24, 8), (16, 12), (16, 8)]
        if (input_grid_size, output_grid_size) not in valid_configs:
            raise ValueError(
                f"Unsupported configuration: {input_grid_size}×{input_grid_size} → {output_grid_size}×{output_grid_size}\n"
                f"Supported: 24→12, 24→8, 16→12, 16→8"
            )

        # Configuration name for checkpoint differentiation
        self.config_name = f"{input_grid_size}to{output_grid_size}"

        # Determine kernel, stride, padding for each scenario
        if input_grid_size == 24 and output_grid_size == 12:
            # 24 → 12: floor((24 - 3 + 2) / 2) + 1 = 12
            kernel_size, stride, padding = 3, 2, 1

        elif input_grid_size == 24 and output_grid_size == 8:
            # 24 → 8: floor((24 - 3 + 0) / 3) + 1 = 8
            kernel_size, stride, padding = 3, 3, 0

        elif input_grid_size == 16 and output_grid_size == 12:
            # 16 → 12: (16 - 5 + 0) / 1 + 1 = 12
            kernel_size, stride, padding = 5, 1, 0

        elif input_grid_size == 16 and output_grid_size == 8:
            # 16 → 8: floor((16 - 3 + 2) / 2) + 1 = 8
            kernel_size, stride, padding = 3, 2, 1

        # Store for theoretical RF calculation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Single convolution layer with Partial Convolution: hidden_dim → hidden_dim directly
        self.compress_conv = PartialConv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = nn.GroupNorm(32, hidden_dim)
        self.activation = nn.GELU()

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, hidden_dim, output_grid_size, output_grid_size) * 0.02
        )

        # Calculate theoretical receptive field
        self.receptive_field_size = self._calculate_receptive_field()

        # Verify output size
        test_size = (input_grid_size + 2 * padding - kernel_size) // stride + 1
        if test_size != output_grid_size:
            raise ValueError(
                f"Kernel/stride calculation error: {input_grid_size}→{test_size} (expected {output_grid_size})\n"
                f"kernel={kernel_size}, stride={stride}, padding={padding}"
            )

        print(f"TokenCompressor [{self.config_name}]: {input_grid_size}×{input_grid_size} → {output_grid_size}×{output_grid_size}")
        print(f"  Architecture: PartialConv(k={kernel_size}, s={stride}, p={padding})")
        print(f"  Theoretical RF: {self.receptive_field_size}×{self.receptive_field_size}")


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress tokens with single partial convolution.

        Args:
            tokens: (batch_size, num_patches, hidden_dim)

        Returns:
            Compressed tokens: (batch_size, output_grid_size^2, hidden_dim)
        """
        batch_size = tokens.shape[0]

        # Reshape to 2D grid: (B, N, C) → (B, C, H, W)
        tokens_2d = self._to_2d(tokens)

        # Create mask: all original tokens are valid (1), padding regions will be handled by PartialConv
        # Mask shape: (B, 1, H, W)
        mask = torch.ones(batch_size, 1, self.input_grid_size, self.input_grid_size,
                         device=tokens_2d.device, dtype=tokens_2d.dtype)

        # Partial convolution compression (handles padding properly)
        x, output_mask = self.compress_conv(tokens_2d, mask)

        # Apply normalization and activation
        x = self.norm(x)
        x = self.activation(x)

        # Verify output size (no adaptive pooling!)
        assert x.shape[2] == self.output_grid_size and x.shape[3] == self.output_grid_size, \
            f"Output size mismatch: got {x.shape[2]}×{x.shape[3]}, expected {self.output_grid_size}×{self.output_grid_size}"

        # Add positional embeddings
        x = x + self.pos_embed

        # Reshape back to token format: (B, C, H, W) → (B, N, C)
        compressed_tokens = self._to_1d(x)

        return compressed_tokens

    def _to_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token sequence to 2D grid"""
        batch_size, num_patches, hidden_dim = tokens.shape
        grid_size = int(math.sqrt(num_patches))

        assert grid_size == self.input_grid_size, \
            f"Expected {self.input_grid_size}^2={self.input_grid_size**2} patches, got {num_patches}"

        # (B, N, C) → (B, H, W, C) → (B, C, H, W)
        tokens_2d = tokens.reshape(batch_size, grid_size, grid_size, hidden_dim)
        tokens_2d = tokens_2d.permute(0, 3, 1, 2)

        return tokens_2d

    def _to_1d(self, tokens_2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D grid back to token sequence"""
        batch_size, hidden_dim, grid_h, grid_w = tokens_2d.shape

        # (B, C, H, W) → (B, H, W, C) → (B, N, C)
        tokens = tokens_2d.permute(0, 2, 3, 1)
        tokens = tokens.reshape(batch_size, grid_h * grid_w, hidden_dim)

        return tokens

    def _calculate_receptive_field(self) -> int:
        """
        Calculate theoretical receptive field for single convolution layer.

        Formula: RF = 1 + (kernel_size - 1) * jump
        For single layer: jump = 1
        Therefore: RF = kernel_size
        """
        return self.kernel_size

    def get_receptive_field_size(self) -> int:
        """Get theoretical receptive field size"""
        return self.receptive_field_size


if __name__ == "__main__":
    print("Testing TokenCompressor for all 4 scenarios...\n")

    scenarios = [
        (24, 12, "24×24 → 12×12"),
        (24, 8, "24×24 → 8×8"),
        (16, 12, "16×16 → 12×12"),
        (16, 8, "16×16 → 8×8"),
    ]

    for input_size, output_size, desc in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print('='*60)

        try:
            compressor = TokenCompressor(
                input_grid_size=input_size,
                output_grid_size=output_size,
                hidden_dim=1024
            )

            # Test forward pass
            batch_size = 4
            num_patches = input_size ** 2
            dummy_tokens = torch.randn(batch_size, num_patches, 1024)

            compressed = compressor(dummy_tokens)

            print(f"\n✓ Forward pass successful")
            print(f"  Input: {dummy_tokens.shape}")
            print(f"  Output: {compressed.shape}")
            print(f"  Expected: ({batch_size}, {output_size**2}, 1024)")

            assert compressed.shape == (batch_size, output_size ** 2, 1024), "Shape mismatch!"

            params = sum(p.numel() for p in compressor.parameters())
            print(f"  Parameters: {params/1e6:.2f}M")

        except Exception as e:
            print(f"\n✗ Failed: {e}")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print('='*60)
