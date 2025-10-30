"""Test script for CLIPVisionTower with ToConv compression"""
import torch
import sys
import os

# Add LLaVA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA'))

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower


class Args:
    def __init__(self, use_compression=False, input_size=24, output_size=12):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'
        self.use_token_compression = use_compression
        self.compression_input_size = input_size
        self.compression_output_size = output_size


def test_without_compression():
    print("=" * 60)
    print("Test 1: CLIPVisionTower WITHOUT compression")
    print("=" * 60)

    args = Args(use_compression=False)
    tower = CLIPVisionTower('openai/clip-vit-large-patch14', args)

    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    dummy_images = torch.randn(2, 3, 224, 224)

    print(f"\nInput shape: {dummy_images.shape}")

    # Forward pass
    features = tower(dummy_images)

    print(f"Output shape: {features.shape}")
    print(f"Expected patches: {tower.num_patches} ({tower.num_patches_per_side}x{tower.num_patches_per_side})")
    print(f"Hidden size: {tower.hidden_size}")
    print("\n✓ Test passed!\n")


def test_with_compression_24_to_12():
    print("=" * 60)
    print("Test 2: CLIPVisionTower WITH compression (24x24 → 12x12)")
    print("=" * 60)

    args = Args(use_compression=True, input_size=24, output_size=12)
    tower = CLIPVisionTower('openai/clip-vit-large-patch14-336', args)

    # Create dummy input
    dummy_images = torch.randn(2, 3, 336, 336)

    print(f"\nInput shape: {dummy_images.shape}")

    # Forward pass
    features = tower(dummy_images)

    print(f"Output shape: {features.shape}")
    print(f"Expected patches: {tower.num_patches} ({tower.num_patches_per_side}x{tower.num_patches_per_side})")
    print(f"Hidden size: {tower.hidden_size}")

    # Verify compression worked
    expected_patches = 12 * 12
    assert features.shape[1] == expected_patches, f"Expected {expected_patches} patches, got {features.shape[1]}"
    print("\n✓ Test passed!\n")


def test_with_compression_24_to_8():
    print("=" * 60)
    print("Test 3: CLIPVisionTower WITH compression (24x24 → 8x8)")
    print("=" * 60)

    args = Args(use_compression=True, input_size=24, output_size=8)
    tower = CLIPVisionTower('openai/clip-vit-large-patch14-336', args)

    # Create dummy input
    dummy_images = torch.randn(2, 3, 336, 336)

    print(f"\nInput shape: {dummy_images.shape}")

    # Forward pass
    features = tower(dummy_images)

    print(f"Output shape: {features.shape}")
    print(f"Expected patches: {tower.num_patches} ({tower.num_patches_per_side}x{tower.num_patches_per_side})")
    print(f"Hidden size: {tower.hidden_size}")

    # Verify compression worked
    expected_patches = 8 * 8
    assert features.shape[1] == expected_patches, f"Expected {expected_patches} patches, got {features.shape[1]}"
    print("\n✓ Test passed!\n")


if __name__ == "__main__":
    print("\nTesting CLIPVisionTower with ToConv compression integration\n")

    try:
        test_without_compression()
        test_with_compression_24_to_12()
        test_with_compression_24_to_8()

        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
