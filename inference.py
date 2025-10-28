"""Inference script for testing trained token compressor"""
import torch
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from vision_token_compression.models import (
    CLIPVisionEncoder,
    TokenCompressor,
    AutoEncoderDecoder
)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained models from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models (you may need to adjust these parameters)
    clip_encoder = CLIPVisionEncoder(
        model_name="openai/clip-vit-large-patch14-336",
        freeze=True
    )

    grid_size = clip_encoder.get_grid_size()
    hidden_dim = clip_encoder.get_hidden_size()
    output_grid_size = 6  # Adjust based on your training config

    compressor = TokenCompressor(
        input_grid_size=grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim
    )

    ae_decoder = AutoEncoderDecoder(
        compressed_grid_size=output_grid_size,
        original_grid_size=grid_size,
        hidden_dim=hidden_dim
    )

    # Load weights
    compressor.load_state_dict(checkpoint['compressor_state_dict'])
    ae_decoder.load_state_dict(checkpoint['ae_decoder_state_dict'])

    # Move to device
    clip_encoder = clip_encoder.to(device).eval()
    compressor = compressor.to(device).eval()
    ae_decoder = ae_decoder.to(device).eval()

    return clip_encoder, compressor, ae_decoder


@torch.no_grad()
def compress_image(
    image_path: str,
    clip_encoder: CLIPVisionEncoder,
    compressor: TokenCompressor,
    ae_decoder: AutoEncoderDecoder,
    device: torch.device
):
    """Compress an image and reconstruct it"""

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    processed = clip_encoder.preprocess(image)
    pixel_values = processed['pixel_values'].to(device)

    # Extract CLIP tokens
    original_tokens = clip_encoder(pixel_values)
    print(f"Original tokens shape: {original_tokens.shape}")

    # Compress
    compressed_tokens = compressor(original_tokens)
    print(f"Compressed tokens shape: {compressed_tokens.shape}")

    compression_ratio = original_tokens.shape[1] / compressed_tokens.shape[1]
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Reconstruct
    reconstructed_tokens = ae_decoder(compressed_tokens)
    print(f"Reconstructed tokens shape: {reconstructed_tokens.shape}")

    # Compute similarity
    original_flat = original_tokens.flatten(0, 1)
    reconstructed_flat = reconstructed_tokens.flatten(0, 1)
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_flat, reconstructed_flat, dim=-1
    ).mean()
    print(f"Reconstruction cosine similarity: {cosine_sim.item():.4f}")

    return {
        'original_tokens': original_tokens.cpu(),
        'compressed_tokens': compressed_tokens.cpu(),
        'reconstructed_tokens': reconstructed_tokens.cpu(),
        'cosine_similarity': cosine_sim.item(),
        'compression_ratio': compression_ratio
    }


def visualize_tokens(results: dict, save_path: str = None):
    """Visualize token compression results"""

    original = results['original_tokens'][0]  # (N, C)
    compressed = results['compressed_tokens'][0]  # (K^2, C)
    reconstructed = results['reconstructed_tokens'][0]  # (N, C)

    # Compute per-token similarity
    similarity = torch.nn.functional.cosine_similarity(
        original, reconstructed, dim=-1
    )

    # Reshape for visualization
    grid_size = int(np.sqrt(original.shape[0]))
    output_grid_size = int(np.sqrt(compressed.shape[0]))

    similarity_map = similarity.reshape(grid_size, grid_size).numpy()

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original token grid (show mean activation)
    original_map = original.mean(dim=-1).reshape(grid_size, grid_size).numpy()
    axes[0].imshow(original_map, cmap='viridis')
    axes[0].set_title(f'Original Tokens ({grid_size}x{grid_size})')
    axes[0].axis('off')

    # Compressed token grid
    compressed_map = compressed.mean(dim=-1).reshape(output_grid_size, output_grid_size).numpy()
    axes[1].imshow(compressed_map, cmap='viridis')
    axes[1].set_title(f'Compressed Tokens ({output_grid_size}x{output_grid_size})')
    axes[1].axis('off')

    # Reconstruction similarity map
    im = axes[2].imshow(similarity_map, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title(f'Reconstruction Similarity\n(Mean: {results["cosine_similarity"]:.4f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test token compressor on images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='compression_result.png',
                        help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    clip_encoder, compressor, ae_decoder = load_checkpoint(args.checkpoint, device)
    print("Models loaded successfully!")

    # Compress image
    print(f"\nCompressing image: {args.image}")
    results = compress_image(
        args.image,
        clip_encoder,
        compressor,
        ae_decoder,
        device
    )

    # Visualize
    print(f"\nCreating visualization...")
    visualize_tokens(results, args.output)

    print(f"\nSummary:")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}x")
    print(f"  Reconstruction similarity: {results['cosine_similarity']:.4f}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
