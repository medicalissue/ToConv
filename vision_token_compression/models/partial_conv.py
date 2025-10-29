"""
Partial Convolution Layer

Based on "Image Inpainting for Irregular Holes Using Partial Convolutions"
(Liu et al., ECCV 2018)

Handles convolution with masked inputs, where padding regions are masked out
to avoid zero-padding artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """
    Partial Convolution layer that properly handles masked inputs.

    Unlike standard convolution with zero padding, partial convolution:
    1. Takes both input and mask
    2. Only considers valid (non-padded) pixels in convolution
    3. Normalizes output based on number of valid pixels
    4. Updates mask for next layer

    Formula:
        output = W^T * (X ⊙ M) * sum(1) / (sum(M) + eps)
        output_mask = 1 if sum(M) > 0 else 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to input (but will be masked)
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: If True, adds a learnable bias
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Standard convolution for features
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False  # We'll add bias after normalization
        )

        # Bias (applied after mask normalization)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Register mask kernel (for computing sum(M))
        # This is a fixed kernel of all ones for counting valid pixels
        self.register_buffer(
            'mask_kernel',
            torch.ones(1, 1, kernel_size, kernel_size)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with partial convolution.

        Args:
            x: Input tensor (B, C, H, W)
            mask: Binary mask tensor (B, 1, H, W) where 1=valid, 0=padded
                  If None, assumes all pixels are valid

        Returns:
            output: Convolved output (B, C', H', W')
            output_mask: Updated mask (B, 1, H', W')
        """
        if mask is None:
            # If no mask provided, all pixels are valid
            mask = torch.ones(x.size(0), 1, x.size(2), x.size(3),
                             device=x.device, dtype=x.dtype)

        # Apply mask to input: X ⊙ M
        masked_input = x * mask

        # Compute convolution on masked input: W^T * (X ⊙ M)
        output = self.conv(masked_input)

        # Compute sum of valid pixels in each receptive field: sum(M)
        # We need to expand mask to match number of input channels for grouped conv
        if self.groups == 1:
            # Standard convolution: use single-channel mask
            mask_sum = F.conv2d(
                mask,
                self.mask_kernel,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation
            )
        else:
            # Grouped convolution: repeat mask for each group
            mask_expanded = mask.repeat(1, self.in_channels, 1, 1)
            mask_kernel_expanded = self.mask_kernel.repeat(self.in_channels, 1, 1, 1)
            mask_sum = F.conv2d(
                mask_expanded,
                mask_kernel_expanded,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.in_channels
            )
            # Average across input channels
            mask_sum = mask_sum.view(mask.size(0), self.in_channels, -1).mean(dim=1, keepdim=True)
            mask_sum = mask_sum.view(mask.size(0), 1, output.size(2), output.size(3))

        # Normalization factor: sum(1) / sum(M)
        # sum(1) = kernel_size^2 for a full kernel
        kernel_numel = self.kernel_size * self.kernel_size
        eps = 1e-8

        # Normalize: multiply by kernel_size^2 and divide by sum(M)
        # This ensures that areas with full kernel coverage have the same scale
        # as standard convolution
        normalization = kernel_numel / (mask_sum + eps)
        output = output * normalization

        # Add bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        # Update mask: output_mask = 1 if sum(M) > 0, else 0
        # This indicates which output pixels have at least one valid input pixel
        output_mask = (mask_sum > 0).float()

        return output, output_mask


if __name__ == "__main__":
    print("Testing PartialConv2d...")

    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    pconv = PartialConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
    x = torch.randn(2, 3, 8, 8)
    mask = torch.ones(2, 1, 8, 8)

    output, output_mask = pconv(x, mask)
    print(f"Input: {x.shape}, Mask: {mask.shape}")
    print(f"Output: {output.shape}, Output mask: {output_mask.shape}")
    print("✓ Basic forward test passed!")

    # Test 2: Partial masking (e.g., padding regions masked)
    print("\n2. Testing with partial masking...")
    mask_partial = torch.ones(2, 1, 8, 8)
    # Mask out border (simulating padding)
    mask_partial[:, :, 0, :] = 0
    mask_partial[:, :, -1, :] = 0
    mask_partial[:, :, :, 0] = 0
    mask_partial[:, :, :, -1] = 0

    output_partial, output_mask_partial = pconv(x, mask_partial)
    print(f"Input mask valid pixels: {mask_partial.sum().item()}/{mask_partial.numel()}")
    print(f"Output mask valid pixels: {output_mask_partial.sum().item()}/{output_mask_partial.numel()}")
    print("✓ Partial masking test passed!")

    # Test 3: With stride and no padding
    print("\n3. Testing with stride=3, padding=0...")
    pconv_stride = PartialConv2d(1024, 1024, kernel_size=3, stride=3, padding=0, bias=False)
    x_24 = torch.randn(2, 1024, 24, 24)
    mask_24 = torch.ones(2, 1, 24, 24)

    output_8, output_mask_8 = pconv_stride(x_24, mask_24)
    print(f"Input: {x_24.shape} -> Output: {output_8.shape}")
    print(f"Output mask all valid: {(output_mask_8 == 1).all().item()}")
    print("✓ Stride test passed!")

    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    x_grad = torch.randn(2, 3, 8, 8, requires_grad=True)
    mask_grad = torch.ones(2, 1, 8, 8)
    output_grad, _ = pconv(x_grad, mask_grad)
    loss = output_grad.sum()
    loss.backward()

    assert x_grad.grad is not None
    print(f"Gradient norm: {x_grad.grad.norm().item():.4f}")
    print("✓ Gradient flow test passed!")

    print("\n" + "="*60)
    print("All PartialConv2d tests passed!")
    print("="*60)
