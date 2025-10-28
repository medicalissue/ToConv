"""Utility functions for Receptive Field operations"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def get_rf_indices(
    compressed_idx: int,
    compressed_grid_size: int,
    original_grid_size: int
) -> torch.Tensor:
    """
    Get original token indices for a compressed token's RF.

    Args:
        compressed_idx: Index in flattened compressed grid (0-35 for 6x6)
        compressed_grid_size: 6
        original_grid_size: 24

    Returns:
        indices: (16,) tensor of original token indices
    """
    rf_size = original_grid_size // compressed_grid_size

    # Convert to 2D position
    comp_i = compressed_idx // compressed_grid_size
    comp_j = compressed_idx % compressed_grid_size

    # RF region in original grid
    start_i = comp_i * rf_size
    start_j = comp_j * rf_size

    indices = []
    for di in range(rf_size):
        for dj in range(rf_size):
            orig_i = start_i + di
            orig_j = start_j + dj
            idx = orig_i * original_grid_size + orig_j
            indices.append(idx)

    return torch.tensor(indices, dtype=torch.long)


def visualize_rf_reconstruction(
    original_tokens: torch.Tensor,
    compressed_tokens: torch.Tensor,
    reconstructed_rfs: torch.Tensor,
    compressed_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize reconstruction quality for a specific RF.

    Args:
        original_tokens: (B, 576, 1024)
        compressed_tokens: (B, 36, 1024)
        reconstructed_rfs: (B, 36, 16, 1024)
        compressed_idx: Which RF to visualize (0-35)
        save_path: Path to save figure (optional)
    """
    # Extract RF
    rf_indices = get_rf_indices(compressed_idx, 6, 24)
    original_rf = original_tokens[0, rf_indices, :].cpu().numpy()  # (16, 1024)
    reconstructed_rf = reconstructed_rfs[0, compressed_idx, :, :].cpu().numpy()  # (16, 1024)
    compressed = compressed_tokens[0, compressed_idx, :].cpu().numpy()  # (1024,)

    # Compute similarities
    similarities = []
    for i in range(16):
        sim = F.cosine_similarity(
            torch.from_numpy(reconstructed_rf[i:i+1]),
            torch.from_numpy(original_rf[i:i+1]),
            dim=-1
        )
        similarities.append(sim.item())

    # Create figure
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Original RF tokens (heatmap)
    ax1 = fig.add_subplot(gs[:, 0])
    im1 = ax1.imshow(original_rf, aspect='auto', cmap='viridis')
    ax1.set_title(f'Original RF Tokens\n(RF #{compressed_idx})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hidden Dimension')
    ax1.set_ylabel('Token Index (0-15)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Plot 2: Reconstructed RF tokens (heatmap)
    ax2 = fig.add_subplot(gs[:, 1])
    im2 = ax2.imshow(reconstructed_rf, aspect='auto', cmap='viridis')
    ax2.set_title('Reconstructed RF Tokens', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hidden Dimension')
    ax2.set_ylabel('Token Index (0-15)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Plot 3: Per-token similarity
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in similarities]
    bars = ax3.bar(range(16), similarities, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.6)')
    ax3.set_title('Per-Token Cosine Similarity', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Token Index in RF')
    ax3.set_ylabel('Similarity')
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Statistics
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    stats_text = f"""
    RF #{compressed_idx} Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━
    Mean Similarity: {np.mean(similarities):.4f}
    Std Similarity:  {np.std(similarities):.4f}
    Min Similarity:  {np.min(similarities):.4f}
    Max Similarity:  {np.max(similarities):.4f}

    Tokens > 0.8:    {sum(1 for s in similarities if s > 0.8)}/16
    Tokens > 0.6:    {sum(1 for s in similarities if s > 0.6)}/16
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Plot 5: Compressed token visualization
    ax5 = fig.add_subplot(gs[:, 3])
    ax5.plot(compressed, alpha=0.7, linewidth=0.5)
    ax5.set_title('Compressed Token\n(1 token → 16 tokens)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Hidden Dimension')
    ax5.set_ylabel('Value')
    ax5.grid(alpha=0.3)

    plt.suptitle(f'RF Reconstruction Visualization (Compressed Grid Position: {compressed_idx})',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def compute_rf_statistics(
    reconstructed_rfs: torch.Tensor,
    original_tokens: torch.Tensor,
    compressed_grid_size: int = 6,
    original_grid_size: int = 24
) -> dict:
    """
    Compute detailed statistics for RF reconstruction.

    Args:
        reconstructed_rfs: (B, 36, 16, 1024)
        original_tokens: (B, 576, 1024)
        compressed_grid_size: 6
        original_grid_size: 24

    Returns:
        stats: Dictionary with per-RF and aggregate statistics
    """
    from vision_token_compression.losses.rf_autoencoder_loss import extract_rf_targets

    batch_size = original_tokens.shape[0]

    # Extract target RFs
    target_rfs = extract_rf_targets(
        original_tokens,
        compressed_grid_size,
        original_grid_size
    )

    # Per-RF cosine similarity
    rf_similarities = []
    for i in range(36):
        recon = reconstructed_rfs[:, i, :, :].reshape(-1, 1024)
        target = target_rfs[:, i, :, :].reshape(-1, 1024)
        sim = F.cosine_similarity(recon, target, dim=-1).mean()
        rf_similarities.append(sim.item())

    # Per-RF MSE
    rf_mse = []
    for i in range(36):
        mse = F.mse_loss(
            reconstructed_rfs[:, i, :, :],
            target_rfs[:, i, :, :]
        )
        rf_mse.append(mse.item())

    # Spatial statistics (per row/column)
    row_similarities = []
    for row in range(compressed_grid_size):
        row_rfs = [rf_similarities[row * compressed_grid_size + col] for col in range(compressed_grid_size)]
        row_similarities.append(np.mean(row_rfs))

    col_similarities = []
    for col in range(compressed_grid_size):
        col_rfs = [rf_similarities[row * compressed_grid_size + col] for row in range(compressed_grid_size)]
        col_similarities.append(np.mean(col_rfs))

    stats = {
        'rf_similarity_mean': float(np.mean(rf_similarities)),
        'rf_similarity_std': float(np.std(rf_similarities)),
        'rf_similarity_min': float(np.min(rf_similarities)),
        'rf_similarity_max': float(np.max(rf_similarities)),
        'rf_mse_mean': float(np.mean(rf_mse)),
        'rf_mse_std': float(np.std(rf_mse)),
        'per_rf_similarities': rf_similarities,
        'per_rf_mse': rf_mse,
        'row_similarities': row_similarities,
        'col_similarities': col_similarities,
        # Quality bins
        'excellent_rfs': sum(1 for s in rf_similarities if s > 0.9),
        'good_rfs': sum(1 for s in rf_similarities if 0.8 < s <= 0.9),
        'fair_rfs': sum(1 for s in rf_similarities if 0.6 < s <= 0.8),
        'poor_rfs': sum(1 for s in rf_similarities if s <= 0.6)
    }

    return stats


def create_rf_heatmap(
    rf_similarities: list,
    compressed_grid_size: int = 6,
    save_path: Optional[str] = None,
    title: str = "RF Reconstruction Quality Heatmap"
):
    """
    Create a heatmap showing reconstruction quality for all RFs.

    Args:
        rf_similarities: List of 36 similarity values
        compressed_grid_size: 6
        save_path: Path to save figure (optional)
        title: Figure title
    """
    # Reshape to grid
    heatmap_data = np.array(rf_similarities).reshape(compressed_grid_size, compressed_grid_size)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

    # Add text annotations
    for i in range(compressed_grid_size):
        for j in range(compressed_grid_size):
            idx = i * compressed_grid_size + j
            text = ax.text(j, i, f'{rf_similarities[idx]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=12)

    # Labels
    ax.set_xticks(np.arange(compressed_grid_size))
    ax.set_yticks(np.arange(compressed_grid_size))
    ax.set_xticklabels(np.arange(compressed_grid_size))
    ax.set_yticklabels(np.arange(compressed_grid_size))
    ax.set_xlabel('Column in Compressed Grid', fontsize=12)
    ax.set_ylabel('Row in Compressed Grid', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.set_xticks(np.arange(compressed_grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(compressed_grid_size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Statistics text
    stats_text = f"""
    Mean: {np.mean(rf_similarities):.4f}
    Std:  {np.std(rf_similarities):.4f}
    Min:  {np.min(rf_similarities):.4f}
    Max:  {np.max(rf_similarities):.4f}
    """
    plt.text(1.15, 0.5, stats_text, transform=ax.transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test RF utilities
    print("Testing RF Utilities...")

    # Test 1: get_rf_indices
    print("\n1. Testing get_rf_indices...")
    for idx in [0, 5, 17, 35]:
        indices = get_rf_indices(idx, 6, 24)
        print(f"RF {idx}: {len(indices)} indices, first 4: {indices[:4].tolist()}")
        assert len(indices) == 16
    print("✓ get_rf_indices test passed!")

    # Test 2: compute_rf_statistics
    print("\n2. Testing compute_rf_statistics...")
    batch_size = 2
    original_tokens = torch.randn(batch_size, 576, 1024)
    reconstructed_rfs = torch.randn(batch_size, 36, 16, 1024)

    stats = compute_rf_statistics(reconstructed_rfs, original_tokens)
    print(f"Mean similarity: {stats['rf_similarity_mean']:.4f}")
    print(f"Excellent RFs: {stats['excellent_rfs']}/36")
    print(f"Good RFs: {stats['good_rfs']}/36")
    print("✓ compute_rf_statistics test passed!")

    # Test 3: create_rf_heatmap
    print("\n3. Testing create_rf_heatmap...")
    dummy_similarities = [0.7 + 0.2 * np.random.rand() for _ in range(36)]
    create_rf_heatmap(dummy_similarities, save_path='/tmp/test_heatmap.png')
    print("✓ create_rf_heatmap test passed!")

    # Test 4: visualize_rf_reconstruction
    print("\n4. Testing visualize_rf_reconstruction...")
    compressed_tokens = torch.randn(1, 36, 1024)
    visualize_rf_reconstruction(
        original_tokens[:1],
        compressed_tokens,
        reconstructed_rfs[:1],
        compressed_idx=0,
        save_path='/tmp/test_rf_viz.png'
    )
    print("✓ visualize_rf_reconstruction test passed!")

    print("\n" + "=" * 60)
    print("All RF Utility tests passed!")
    print("=" * 60)
