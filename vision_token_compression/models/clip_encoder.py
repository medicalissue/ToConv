"""CLIP Vision Encoder Wrapper"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class CLIPVisionEncoder(nn.Module):
    """
    CLIP Vision Transformer wrapper that extracts vision tokens.
    Supports ViT-L/14 at 224px and 336px resolutions.
    The model is frozen during training.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14-336", freeze: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze the model parameters
        """
        super().__init__()

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        # Freeze the model if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # Get model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        # Calculate grid size (number of patches per dimension)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract vision tokens from images.

        Args:
            pixel_values: Preprocessed images of shape (batch_size, 3, H, W)

        Returns:
            Vision tokens of shape (batch_size, num_patches, hidden_size)
            Note: CLS token is excluded, only patch tokens are returned
        """
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            outputs = self.model(pixel_values, output_hidden_states=False)

            # Get the last hidden states
            # Shape: (batch_size, num_patches + 1, hidden_size)
            # The +1 is for the CLS token at position 0
            last_hidden_state = outputs.last_hidden_state

            # Remove CLS token, keep only patch tokens
            # Shape: (batch_size, num_patches, hidden_size)
            patch_tokens = last_hidden_state[:, 1:, :]

        return patch_tokens

    def _is_frozen(self) -> bool:
        """Check if model parameters are frozen"""
        return not any(p.requires_grad for p in self.model.parameters())

    def preprocess(self, images):
        """
        Preprocess PIL images or numpy arrays for the model.

        Args:
            images: PIL Image(s) or numpy array(s)

        Returns:
            Preprocessed tensor ready for the model
        """
        return self.processor(images=images, return_tensors="pt")

    def get_grid_size(self) -> int:
        """Get the grid size (patches per dimension)"""
        return self.grid_size

    def get_hidden_size(self) -> int:
        """Get the hidden dimension size"""
        return self.hidden_size


if __name__ == "__main__":
    # Test the encoder
    encoder = CLIPVisionEncoder(model_name="openai/clip-vit-large-patch14-336")
    print(f"Grid size: {encoder.get_grid_size()}x{encoder.get_grid_size()}")
    print(f"Number of patches: {encoder.num_patches}")
    print(f"Hidden size: {encoder.get_hidden_size()}")

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 336, 336)
    tokens = encoder(dummy_input)
    print(f"Output shape: {tokens.shape}")
    assert tokens.shape == (batch_size, encoder.num_patches, encoder.hidden_size)
    print("✓ CLIP Vision Encoder test passed!")
