"""2D Convolution-based Token Compressor"""
import torch
import torch.nn as nn
import math


class TokenCompressor(nn.Module):
    """
    2D Convolution-based token compressor that reduces spatial resolution
    while preserving local token relationships.

    Converts a grid of tokens from (grid_size x grid_size) to (k x k).
    """

    def __init__(
        self,
        input_grid_size: int,
        output_grid_size: int,
        hidden_dim: int,
        num_layers: int = 3,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        """
        Args:
            input_grid_size: Size of input token grid (e.g., 24 for 24x24)
            output_grid_size: Size of output token grid (k for kxk)
            hidden_dim: Hidden dimension of tokens (e.g., 1024 for CLIP ViT-L)
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_grid_size = input_grid_size
        self.output_grid_size = output_grid_size
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Calculate total compression ratio
        self.compression_ratio = input_grid_size / output_grid_size

        # Build the compression network
        self.layers = nn.ModuleList()

        # Calculate per-layer downsampling
        # We'll use strided convolutions to gradually reduce spatial dimensions
        current_size = input_grid_size

        for i in range(num_layers):
            # Calculate the stride for this layer
            if i == num_layers - 1:
                # Last layer: ensure we reach exact output size
                stride = current_size // output_grid_size
                kernel_size = stride + (stride % 2)  # Ensure odd kernel size
            else:
                # Intermediate layers: gradual downsampling
                target_size = int(current_size / (self.compression_ratio ** (1 / num_layers)))
                stride = max(1, current_size // target_size)
                kernel_size = stride + (stride % 2)

            # Ensure kernel size is at least 3 and odd
            kernel_size = max(3, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            padding = kernel_size // 2

            layer_modules = []

            # Convolutional layer
            layer_modules.append(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not use_layer_norm
                )
            )

            # Layer normalization (applied per spatial position)
            if use_layer_norm:
                layer_modules.append(nn.GroupNorm(32, hidden_dim))

            # Activation
            if i < num_layers - 1:  # No activation on last layer
                layer_modules.append(nn.GELU())

            self.layers.append(nn.Sequential(*layer_modules))

            # Update current size
            current_size = (current_size + 2 * padding - kernel_size) // stride + 1

        # Adaptive pooling to ensure exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_grid_size, output_grid_size))

        # Optional: learnable position embeddings for output tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, hidden_dim, output_grid_size, output_grid_size) * 0.02
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress tokens using 2D convolutions.

        Args:
            tokens: Input tokens of shape (batch_size, num_patches, hidden_dim)
                   where num_patches = input_grid_size^2

        Returns:
            Compressed tokens of shape (batch_size, k^2, hidden_dim)
            where k = output_grid_size
        """
        batch_size = tokens.shape[0]

        # Reshape tokens to 2D grid: (B, N, C) -> (B, C, H, W)
        tokens_2d = self._to_2d(tokens)

        # Apply convolutional layers
        x = tokens_2d
        for layer in self.layers:
            if self.use_residual and x.shape == layer(x).shape:
                x = x + layer(x)
            else:
                x = layer(x)

        # Ensure exact output size with adaptive pooling
        x = self.adaptive_pool(x)

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
        tokens_2d = tokens.view(batch_size, grid_size, grid_size, hidden_dim)
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

    def get_receptive_field_mapping(self) -> torch.Tensor:
        """
        Compute which input tokens correspond to each output token.

        Returns:
            Mapping tensor for receptive field computation
        """
        # Each output token has a receptive field in the input
        # This is useful for the autoencoder loss
        rf_size = self.input_grid_size // self.output_grid_size

        # Create mapping: (k, k, rf_size, rf_size)
        # For each output position (i, j), get input positions
        mapping = torch.zeros(
            self.output_grid_size,
            self.output_grid_size,
            rf_size,
            rf_size,
            dtype=torch.long
        )

        for i in range(self.output_grid_size):
            for j in range(self.output_grid_size):
                # Input region corresponding to output position (i, j)
                start_i = i * rf_size
                start_j = j * rf_size

                for di in range(rf_size):
                    for dj in range(rf_size):
                        input_i = start_i + di
                        input_j = start_j + dj
                        input_idx = input_i * self.input_grid_size + input_j
                        mapping[i, j, di, dj] = input_idx

        return mapping


if __name__ == "__main__":
    # Test the compressor
    batch_size = 4
    input_grid_size = 24  # 24x24 grid for ViT-L/14@336px
    output_grid_size = 6  # Compress to 6x6 grid
    hidden_dim = 1024  # CLIP ViT-L hidden dimension

    compressor = TokenCompressor(
        input_grid_size=input_grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim,
        num_layers=3
    )

    # Test forward pass
    num_patches = input_grid_size ** 2
    dummy_tokens = torch.randn(batch_size, num_patches, hidden_dim)
    compressed = compressor(dummy_tokens)

    print(f"Input shape: {dummy_tokens.shape}")
    print(f"Output shape: {compressed.shape}")
    assert compressed.shape == (batch_size, output_grid_size ** 2, hidden_dim)

    # Test receptive field mapping
    rf_mapping = compressor.get_receptive_field_mapping()
    print(f"Receptive field mapping shape: {rf_mapping.shape}")

    print("✓ Token Compressor test passed!")
