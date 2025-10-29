"""Analyze reasonable OT loss baselines for token compression."""
import torch
from vision_token_compression.losses.sinkhorn_ot_loss import SinkhornOTLoss

# Current config settings
input_grid_size = 24   # 24x24 = 576 tokens
output_grid_size = 8   # 8x8 = 64 tokens
hidden_dim = 1024
epsilon = 0.1
normalize = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Config: {input_grid_size}×{input_grid_size} ({input_grid_size**2} tokens) → "
      f"{output_grid_size}×{output_grid_size} ({output_grid_size**2} tokens)")
print(f"Hidden dim: {hidden_dim}, Epsilon: {epsilon}, Normalize: {normalize}\n")

# Create loss function
ot_loss_fn = SinkhornOTLoss(epsilon=epsilon, normalize=normalize).to(device)

batch_size = 8
n_original = input_grid_size ** 2  # 576
n_compressed = output_grid_size ** 2  # 64

print("=" * 70)
print("BASELINE 1: Identical distributions (perfect compression)")
print("=" * 70)
# Same tokens - this is the best case (perfect compression)
tokens = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
loss_identical, info = ot_loss_fn(tokens, tokens)
print(f"OT loss (identical): {loss_identical.item():.6f}")
print(f"OT distance: {info['ot_distance']:.6f}")
print(f"→ This is the BEST case (perfect reconstruction)\n")

print("=" * 70)
print("BASELINE 2: Random uniform distributions")
print("=" * 70)
# Completely random distributions
compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
original = torch.randn(batch_size, n_original, hidden_dim).to(device)
loss_random, info = ot_loss_fn(compressed, original)
print(f"OT loss (random): {loss_random.item():.6f}")
print(f"OT distance: {info['ot_distance']:.6f}")
print(f"→ This is a BAD case (no structure preservation)\n")

print("=" * 70)
print("BASELINE 3: Slightly perturbed distributions (small noise)")
print("=" * 70)
# Add small noise to simulate slight compression artifacts
compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
# Create "compressed" version: downsample with small noise
original_base = compressed.repeat_interleave(n_original // n_compressed, dim=1)
noise_levels = [0.01, 0.05, 0.1, 0.3, 0.5]
for noise_level in noise_levels:
    original = original_base + noise_level * torch.randn_like(original_base)
    loss_noisy, info = ot_loss_fn(compressed, original)
    print(f"Noise level {noise_level:.2f}: OT loss = {loss_noisy.item():.6f}, "
          f"OT distance = {info['ot_distance']:.6f}")

print()
print("=" * 70)
print("BASELINE 4: Shifted distributions (systematic bias)")
print("=" * 70)
# Shifted distribution (systematic error)
compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
shift_amounts = [0.1, 0.5, 1.0, 2.0, 5.0]
for shift in shift_amounts:
    original = torch.randn(batch_size, n_original, hidden_dim).to(device) + shift
    loss_shifted, info = ot_loss_fn(compressed, original)
    print(f"Shift {shift:.1f}: OT loss = {loss_shifted.item():.6f}, "
          f"OT distance = {info['ot_distance']:.6f}")

print()
print("=" * 70)
print("BASELINE 5: Realistic compression scenarios")
print("=" * 70)
# Simulate realistic compression: keep structure but with some information loss
compressed = torch.randn(batch_size, n_compressed, hidden_dim).to(device)
# Normalize to unit sphere (like in real tokens)
compressed = torch.nn.functional.normalize(compressed, p=2, dim=-1)

# Scenario A: Good compression (preserve 95% of information)
original_good = compressed.repeat_interleave(n_original // n_compressed, dim=1)
original_good = original_good + 0.1 * torch.randn_like(original_good)  # 10% noise
original_good = torch.nn.functional.normalize(original_good, p=2, dim=-1)
loss_good, info = ot_loss_fn(compressed, original_good)
print(f"Good compression (10% noise): OT loss = {loss_good.item():.6f}, "
      f"OT distance = {info['ot_distance']:.6f}")

# Scenario B: Moderate compression (preserve 80% of information)
original_moderate = compressed.repeat_interleave(n_original // n_compressed, dim=1)
original_moderate = original_moderate + 0.3 * torch.randn_like(original_moderate)  # 30% noise
original_moderate = torch.nn.functional.normalize(original_moderate, p=2, dim=-1)
loss_moderate, info = ot_loss_fn(compressed, original_moderate)
print(f"Moderate compression (30% noise): OT loss = {loss_moderate.item():.6f}, "
      f"OT distance = {info['ot_distance']:.6f}")

# Scenario C: Poor compression (preserve 50% of information)
original_poor = compressed.repeat_interleave(n_original // n_compressed, dim=1)
original_poor = original_poor + 1.0 * torch.randn_like(original_poor)  # 100% noise
original_poor = torch.nn.functional.normalize(original_poor, p=2, dim=-1)
loss_poor, info = ot_loss_fn(compressed, original_poor)
print(f"Poor compression (100% noise): OT loss = {loss_poor.item():.6f}, "
      f"OT distance = {info['ot_distance']:.6f}")

print()
print("=" * 70)
print("SUMMARY: Reasonable OT Loss Ranges")
print("=" * 70)
print(f"✓ Excellent (<0.05): Very close distributions, minimal information loss")
print(f"✓ Good (0.05-0.10): Close distributions, acceptable compression quality")
print(f"○ Moderate (0.10-0.20): Some structural similarity preserved")
print(f"✗ Poor (>0.20): Significant distributional mismatch")
print("=" * 70)
