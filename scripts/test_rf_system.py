"""Test script for RF-based Vision Token Compression System"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_token_compression.models import (
    CLIPVisionEncoder,
    TokenCompressor,
    RFDiscriminator,
    RFAutoEncoderDecoder
)
from vision_token_compression.losses import RFWGANGPLoss, RFAutoEncoderLoss


def test_rf_system():
    """Test the complete RF-based system"""
    print("=" * 80)
    print("TESTING RF-BASED VISION TOKEN COMPRESSION SYSTEM")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Configuration
    batch_size = 2
    compressed_grid_size = 6
    original_grid_size = 24
    output_grid_size = 6
    hidden_dim = 1024
    rf_size = 4

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Original grid: {original_grid_size}×{original_grid_size} = {original_grid_size**2} tokens")
    print(f"  Compressed grid: {compressed_grid_size}×{compressed_grid_size} = {compressed_grid_size**2} tokens")
    print(f"  RF size: {rf_size}×{rf_size} = {rf_size**2} tokens per RF")
    print(f"  Hidden dim: {hidden_dim}")

    # Test 1: Create models
    print("\n" + "-" * 80)
    print("1. Creating models...")
    print("-" * 80)

    clip_encoder = CLIPVisionEncoder(
        model_name="openai/clip-vit-large-patch14-336",
        freeze=True
    ).to(device)
    print(f"✓ CLIP encoder created")

    compressor = TokenCompressor(
        input_grid_size=original_grid_size,
        output_grid_size=output_grid_size,
        hidden_dim=hidden_dim
    ).to(device)
    print(f"✓ Token compressor created ({sum(p.numel() for p in compressor.parameters())/1e6:.2f}M params)")

    rf_discriminator = RFDiscriminator(
        hidden_dim=hidden_dim,
        num_layers=3
    ).to(device)
    print(f"✓ RF discriminator created ({sum(p.numel() for p in rf_discriminator.parameters())/1e6:.2f}M params)")

    rf_ae_decoder = RFAutoEncoderDecoder(
        hidden_dim=hidden_dim,
        rf_size=rf_size,
        num_layers=3
    ).to(device)
    print(f"✓ RF AutoEncoder decoder created ({sum(p.numel() for p in rf_ae_decoder.parameters())/1e6:.2f}M params)")

    # Test 2: Forward pass through CLIP
    print("\n" + "-" * 80)
    print("2. Testing CLIP encoding...")
    print("-" * 80)

    dummy_images = torch.randn(batch_size, 3, 336, 336).to(device)
    with torch.no_grad():
        original_tokens = clip_encoder(dummy_images)
    print(f"✓ CLIP output shape: {original_tokens.shape}")
    assert original_tokens.shape == (batch_size, 576, hidden_dim)

    # Test 3: Token compression
    print("\n" + "-" * 80)
    print("3. Testing token compression...")
    print("-" * 80)

    compressed_tokens = compressor(original_tokens)
    print(f"✓ Compressed shape: {compressed_tokens.shape}")
    assert compressed_tokens.shape == (batch_size, 36, hidden_dim)
    print(f"✓ Compression ratio: {original_tokens.shape[1] / compressed_tokens.shape[1]:.2f}x")

    # Test 4: RF discrimination
    print("\n" + "-" * 80)
    print("4. Testing RF discrimination...")
    print("-" * 80)

    rf_wgan_loss = RFWGANGPLoss(lambda_gp=10.0)
    disc_loss, disc_info = rf_wgan_loss.discriminator_loss(
        compressed_tokens=compressed_tokens,
        original_tokens=original_tokens,
        discriminator=rf_discriminator,
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size,
        device=device
    )
    print(f"✓ Discriminator loss: {disc_loss.item():.4f}")
    print(f"  - Wasserstein distance: {disc_info['wasserstein_distance']:.4f}")
    print(f"  - Gradient penalty: {disc_info['gradient_penalty']:.4f}")
    print(f"  - Real score: {disc_info['real_score_mean']:.4f}")
    print(f"  - Fake score: {disc_info['fake_score_mean']:.4f}")

    # Test 5: RF reconstruction
    print("\n" + "-" * 80)
    print("5. Testing RF reconstruction...")
    print("-" * 80)

    reconstructed_rfs = rf_ae_decoder(compressed_tokens)
    print(f"✓ Reconstructed RFs shape: {reconstructed_rfs.shape}")
    assert reconstructed_rfs.shape == (batch_size, 36, 16, hidden_dim)
    print(f"  - Each of 36 compressed tokens reconstructs 16 RF tokens")

    # Test 6: AutoEncoder loss
    print("\n" + "-" * 80)
    print("6. Testing RF AutoEncoder loss...")
    print("-" * 80)

    rf_ae_loss = RFAutoEncoderLoss(loss_type='hybrid', normalize=True)
    ae_loss, ae_info = rf_ae_loss(
        reconstructed_rfs=reconstructed_rfs,
        original_tokens=original_tokens,
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size
    )
    print(f"✓ AutoEncoder loss: {ae_loss.item():.4f}")
    print(f"  - Cosine similarity: {ae_info['cosine_similarity']:.4f}")
    print(f"  - RF similarity mean: {ae_info['rf_similarity_mean']:.4f}")
    print(f"  - RF similarity min: {ae_info['rf_similarity_min']:.4f}")
    print(f"  - RF similarity max: {ae_info['rf_similarity_max']:.4f}")

    # Test 7: Gradient flow
    print("\n" + "-" * 80)
    print("7. Testing gradient flow...")
    print("-" * 80)

    # Generator forward pass
    compressed_tokens_grad = compressor(original_tokens)
    gen_loss, gen_info = rf_wgan_loss.generator_loss(
        compressed_tokens=compressed_tokens_grad,
        discriminator=rf_discriminator
    )

    reconstructed_rfs_grad = rf_ae_decoder(compressed_tokens_grad)
    ae_loss_grad, _ = rf_ae_loss(
        reconstructed_rfs=reconstructed_rfs_grad,
        original_tokens=original_tokens,
        compressed_grid_size=compressed_grid_size,
        original_grid_size=original_grid_size
    )

    total_loss = gen_loss + ae_loss_grad
    total_loss.backward()

    # Check gradients
    has_comp_grads = any(p.grad is not None for p in compressor.parameters())
    has_dec_grads = any(p.grad is not None for p in rf_ae_decoder.parameters())
    has_clip_grads = any(p.grad is not None for p in clip_encoder.parameters())

    print(f"✓ Compressor has gradients: {has_comp_grads}")
    print(f"✓ Decoder has gradients: {has_dec_grads}")
    print(f"✓ CLIP is frozen (no gradients): {not has_clip_grads}")

    assert has_comp_grads, "Compressor should have gradients"
    assert has_dec_grads, "Decoder should have gradients"
    assert not has_clip_grads, "CLIP should be frozen"

    # Test 8: RF utilities
    print("\n" + "-" * 80)
    print("8. Testing RF utilities...")
    print("-" * 80)

    from vision_token_compression.utils import get_rf_indices, compute_rf_statistics

    # Test RF indices
    rf_idx_0 = get_rf_indices(0, compressed_grid_size, original_grid_size)
    rf_idx_35 = get_rf_indices(35, compressed_grid_size, original_grid_size)

    print(f"✓ RF 0 indices: {len(rf_idx_0)} tokens, first 4: {rf_idx_0[:4].tolist()}")
    print(f"✓ RF 35 indices: {len(rf_idx_35)} tokens, last 4: {rf_idx_35[-4:].tolist()}")

    # Test RF statistics
    with torch.no_grad():
        rf_stats = compute_rf_statistics(
            reconstructed_rfs,
            original_tokens,
            compressed_grid_size,
            original_grid_size
        )

    print(f"✓ RF statistics computed:")
    print(f"  - Mean similarity: {rf_stats['rf_similarity_mean']:.4f}")
    print(f"  - Excellent RFs (>0.9): {rf_stats['excellent_rfs']}/36")
    print(f"  - Good RFs (0.8-0.9): {rf_stats['good_rfs']}/36")
    print(f"  - Fair RFs (0.6-0.8): {rf_stats['fair_rfs']}/36")
    print(f"  - Poor RFs (<0.6): {rf_stats['poor_rfs']}/36")

    # Test 9: Memory check
    print("\n" + "-" * 80)
    print("9. Memory usage check...")
    print("-" * 80)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"✓ GPU memory allocated: {allocated:.2f} GB")
        print(f"✓ GPU memory reserved: {reserved:.2f} GB")

    # Summary
    print("\n" + "=" * 80)
    print("ALL RF SYSTEM TESTS PASSED!")
    print("=" * 80)

    print("\nSystem Summary:")
    print(f"  ✓ RF-based discrimination: 36 (compressed) vs 36 (sampled from RFs)")
    print(f"  ✓ RF-based reconstruction: 36 compressed → 36 RFs of 16 tokens each")
    print(f"  ✓ Total parameters: {(sum(p.numel() for p in compressor.parameters()) + sum(p.numel() for p in rf_discriminator.parameters()) + sum(p.numel() for p in rf_ae_decoder.parameters())) / 1e6:.2f}M")
    print(f"  ✓ Compression ratio: 16x (576 → 36 tokens)")
    print(f"  ✓ System is ready for training!")

    print("\nNext steps:")
    print("  1. Update ImageNet path in configs/rf_config.yaml")
    print("  2. Run: python train_rf.py")
    print("  3. Monitor RF reconstruction quality in wandb")


if __name__ == "__main__":
    try:
        test_rf_system()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
