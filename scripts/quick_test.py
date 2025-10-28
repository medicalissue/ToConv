"""Quick test script to verify installation and setup"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_token_compression.models import (
    CLIPVisionEncoder,
    TokenCompressor,
    Discriminator,
    AutoEncoderDecoder
)
from vision_token_compression.losses import WGANGPLoss, AutoEncoderLoss


def test_models():
    """Test all models can be created and run"""
    print("=" * 80)
    print("TESTING VISION TOKEN COMPRESSION SYSTEM")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Test parameters
    batch_size = 2
    output_grid_size = 6

    print("\n" + "-" * 80)
    print("1. Testing CLIP Vision Encoder...")
    print("-" * 80)

    clip_encoder = CLIPVisionEncoder(
        model_name="openai/clip-vit-large-patch14-336",
        freeze=True
    ).to(device)

    grid_size = clip_encoder.get_grid_size()
    hidden_dim = clip_encoder.get_hidden_size()

    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Hidden dim: {hidden_dim}")

    # Test forward pass
    dummy_images = torch.randn(batch_size, 3, 336, 336).to(device)
    tokens = clip_encoder(dummy_images)
    print(f"   Output shape: {tokens.shape}")
    assert tokens.shape == (batch_size, grid_size ** 2, hidden_dim)
    print("   ✓ CLIP encoder test passed!")

    print("\n" + "-" * 80)
    print("2. Testing Token Compressor...")
    print("-" * 80)

    compressor = TokenCompressor(
        input_grid_size=grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim
    ).to(device)

    compressed = compressor(tokens)
    print(f"   Input shape: {tokens.shape}")
    print(f"   Output shape: {compressed.shape}")
    print(f"   Compression ratio: {tokens.shape[1] / compressed.shape[1]:.2f}x")
    assert compressed.shape == (batch_size, output_grid_size ** 2, hidden_dim)
    print("   ✓ Compressor test passed!")

    print("\n" + "-" * 80)
    print("3. Testing Discriminator...")
    print("-" * 80)

    discriminator = Discriminator(
        num_tokens=output_grid_size ** 2,
        hidden_dim=hidden_dim
    ).to(device)

    scores = discriminator(compressed)
    print(f"   Input shape: {compressed.shape}")
    print(f"   Output shape: {scores.shape}")
    assert scores.shape == (batch_size, 1)
    print("   ✓ Discriminator test passed!")

    print("\n" + "-" * 80)
    print("4. Testing AutoEncoder Decoder...")
    print("-" * 80)

    ae_decoder = AutoEncoderDecoder(
        compressed_grid_size=output_grid_size,
        original_grid_size=grid_size,
        hidden_dim=hidden_dim
    ).to(device)

    reconstructed = ae_decoder(compressed)
    print(f"   Input shape: {compressed.shape}")
    print(f"   Output shape: {reconstructed.shape}")
    assert reconstructed.shape == tokens.shape
    print("   ✓ AutoEncoder decoder test passed!")

    print("\n" + "-" * 80)
    print("5. Testing Loss Functions...")
    print("-" * 80)

    # WGAN-GP Loss
    wgan_loss = WGANGPLoss(lambda_gp=10.0)

    real_scores = discriminator(tokens)
    fake_scores = discriminator(compressed)

    from vision_token_compression.losses import compute_gradient_penalty
    gp = compute_gradient_penalty(discriminator, tokens, compressed, device)

    disc_loss, disc_info = wgan_loss.discriminator_loss(real_scores, fake_scores, gp)
    gen_loss, gen_info = wgan_loss.generator_loss(fake_scores)

    print(f"   Discriminator loss: {disc_loss.item():.4f}")
    print(f"   Generator loss: {gen_loss.item():.4f}")
    print(f"   Gradient penalty: {gp.item():.4f}")
    print("   ✓ WGAN-GP loss test passed!")

    # AutoEncoder Loss
    ae_loss_fn = AutoEncoderLoss(loss_type='hybrid')
    ae_loss, ae_info = ae_loss_fn(reconstructed, tokens)

    print(f"   AutoEncoder loss: {ae_loss.item():.4f}")
    print(f"   Cosine similarity: {ae_info['cosine_similarity']:.4f}")
    print("   ✓ AutoEncoder loss test passed!")

    print("\n" + "-" * 80)
    print("6. Testing Backward Pass...")
    print("-" * 80)

    # Test that gradients flow correctly
    total_loss = gen_loss + ae_loss
    total_loss.backward()

    print("   ✓ Backward pass successful!")

    # Check that compressor has gradients
    has_grads = any(p.grad is not None for p in compressor.parameters())
    assert has_grads, "Compressor should have gradients"
    print("   ✓ Compressor gradients computed!")

    # Check that CLIP encoder does NOT have gradients (frozen)
    clip_has_grads = any(p.grad is not None for p in clip_encoder.parameters())
    assert not clip_has_grads, "CLIP encoder should be frozen"
    print("   ✓ CLIP encoder correctly frozen!")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! System is ready for training.")
    print("=" * 80)

    # Print summary
    total_params = (
        sum(p.numel() for p in compressor.parameters()) +
        sum(p.numel() for p in discriminator.parameters()) +
        sum(p.numel() for p in ae_decoder.parameters())
    )

    print("\nModel Summary:")
    print(f"  Total trainable parameters: {total_params / 1e6:.2f}M")
    print(f"  Compression ratio: {grid_size ** 2 / output_grid_size ** 2:.2f}x")
    print(f"  Device: {device}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


if __name__ == "__main__":
    try:
        test_models()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
