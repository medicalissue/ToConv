"""
Receptive Field Cosine Similarity Loss

This loss maximizes cosine similarity between each compressed token and all tokens
in its corresponding receptive field from the original token grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RFCosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for Receptive Field Token Compression.

    For each compressed token, computes cosine similarity with all tokens in its
    corresponding receptive field and maximizes the average similarity.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        compressed_tokens: torch.Tensor,
        original_tokens: torch.Tensor,
        compressed_grid_size: Tuple[int, int] = (6, 6),
        original_grid_size: Tuple[int, int] = (24, 24),
        compute_stats: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine similarity loss between compressed tokens and their RFs.

        Args:
            compressed_tokens: (B, 36, 1024) - Compressed token grid
            original_tokens: (B, 576, 1024) - Original token grid from CLIP
            compressed_grid_size: Size of compressed grid (6, 6)
            original_grid_size: Size of original grid (24, 24)

        Returns:
            loss: Scalar loss tensor (negative mean similarity to maximize)
            info_dict: Dictionary with detailed metrics
        """
        batch_size = compressed_tokens.size(0)
        hidden_dim = compressed_tokens.size(2)

        comp_h, comp_w = compressed_grid_size
        orig_h, orig_w = original_grid_size

        # Calculate RF size
        rf_h = orig_h // comp_h  # 4
        rf_w = orig_w // comp_w  # 4
        rf_size = rf_h * rf_w    # 16

        # Reshape original tokens to 2D grid
        # (B, 576, 1024) -> (B, 24, 24, 1024)
        original_grid = original_tokens.reshape(batch_size, orig_h, orig_w, hidden_dim)

        # Vectorized RF extraction using unfold
        # Unfold extracts sliding windows without loops
        # Reshape to (B, D, H, W) for unfold
        original_grid_transposed = original_grid.permute(0, 3, 1, 2)  # (B, 1024, 24, 24)

        # Use unfold to extract all RFs at once
        # unfold(dim, size, step) extracts windows
        # dim=2 (height), size=rf_h, step=rf_h (non-overlapping)
        rf_unfolded = original_grid_transposed.unfold(2, rf_h, rf_h)  # (B, D, comp_h, W, rf_h)
        rf_unfolded = rf_unfolded.unfold(3, rf_w, rf_w)  # (B, D, comp_h, comp_w, rf_h, rf_w)

        # Rearrange to (B, comp_h, comp_w, rf_h, rf_w, D)
        rf_unfolded = rf_unfolded.permute(0, 2, 3, 4, 5, 1)

        # Reshape to (B, comp_h*comp_w, rf_h*rf_w, D) = (B, 36, 16, 1024)
        rf_tokens = rf_unfolded.reshape(batch_size, comp_h * comp_w, rf_size, hidden_dim)

        # Normalize tokens for cosine similarity
        # Compressed: (B, 36, 1024)
        compressed_norm = F.normalize(compressed_tokens, p=2, dim=-1)

        # RF tokens: (B, 36, 16, 1024)
        rf_norm = F.normalize(rf_tokens, p=2, dim=-1)

        # Compute cosine similarity between each compressed token and its RF tokens
        # compressed_norm: (B, 36, 1, 1024)
        # rf_norm: (B, 36, 16, 1024)
        # Result: (B, 36, 16) - similarity for each token in RF
        compressed_expanded = compressed_norm.unsqueeze(2)  # (B, 36, 1, 1024)

        # Compute dot product (cosine similarity since vectors are normalized)
        # (B, 36, 1, 1024) * (B, 36, 16, 1024) -> (B, 36, 16)
        similarities = (compressed_expanded * rf_norm).sum(dim=-1)  # (B, 36, 16)

        # Apply temperature scaling
        similarities = similarities / self.temperature

        # Average similarity per compressed token: (B, 36)
        avg_similarity_per_token = similarities.mean(dim=-1)

        # Overall average similarity: scalar
        mean_similarity = avg_similarity_per_token.mean()

        # Loss is negative similarity (we want to maximize similarity)
        loss = -mean_similarity

        # Lazy statistics computation (only when needed for logging)
        if compute_stats:
            min_similarity = similarities.min().item()
            max_similarity = similarities.max().item()
            std_similarity = similarities.std().item()
            min_per_token = avg_similarity_per_token.min().item()
            max_per_token = avg_similarity_per_token.max().item()
            std_per_token = avg_similarity_per_token.std().item()

            info_dict = {
                'cosine_sim_loss': loss.item(),
                'mean_similarity': mean_similarity.item(),
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'std_similarity': std_similarity,
                'min_per_token': min_per_token,
                'max_per_token': max_per_token,
                'std_per_token': std_per_token,
                'temperature': self.temperature
            }
        else:
            # Skip expensive statistics, only return essentials
            info_dict = {
                'cosine_sim_loss': loss.item(),
                'mean_similarity': mean_similarity.item(),
                'min_similarity': 0.0,  # Placeholder
                'max_similarity': 0.0,  # Placeholder
                'std_similarity': 0.0,  # Placeholder
                'temperature': self.temperature
            }

        return loss, info_dict


def extract_rf_tokens_for_similarity(
    original_tokens: torch.Tensor,
    compressed_grid_size: Tuple[int, int] = (6, 6),
    original_grid_size: Tuple[int, int] = (24, 24)
) -> torch.Tensor:
    """
    Extract all RF tokens for each compressed position.

    Args:
        original_tokens: (B, 576, 1024) - Original token grid
        compressed_grid_size: Size of compressed grid (6, 6)
        original_grid_size: Size of original grid (24, 24)

    Returns:
        rf_tokens: (B, 36, 16, 1024) - RF tokens for each compressed position
    """
    batch_size = original_tokens.size(0)
    hidden_dim = original_tokens.size(2)

    comp_h, comp_w = compressed_grid_size
    orig_h, orig_w = original_grid_size

    rf_h = orig_h // comp_h
    rf_w = orig_w // comp_w
    rf_size = rf_h * rf_w

    # Reshape to 2D grid
    original_grid = original_tokens.reshape(batch_size, orig_h, orig_w, hidden_dim)

    rf_tokens = []

    for i in range(comp_h):
        for j in range(comp_w):
            start_h = i * rf_h
            end_h = start_h + rf_h
            start_w = j * rf_w
            end_w = start_w + rf_w

            rf_region = original_grid[:, start_h:end_h, start_w:end_w, :]
            rf_flat = rf_region.reshape(batch_size, rf_size, hidden_dim)

            rf_tokens.append(rf_flat)

    return torch.stack(rf_tokens, dim=1)


def compute_pairwise_rf_similarity(
    compressed_tokens: torch.Tensor,
    original_tokens: torch.Tensor,
    compressed_grid_size: Tuple[int, int] = (6, 6),
    original_grid_size: Tuple[int, int] = (24, 24)
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for visualization.

    Args:
        compressed_tokens: (B, 36, 1024)
        original_tokens: (B, 576, 1024)
        compressed_grid_size: (6, 6)
        original_grid_size: (24, 24)

    Returns:
        similarity_matrix: (B, 36, 16) - Similarity of each compressed token
                          with each token in its RF
    """
    # Extract RF tokens
    rf_tokens = extract_rf_tokens_for_similarity(
        original_tokens, compressed_grid_size, original_grid_size
    )

    # Normalize
    compressed_norm = F.normalize(compressed_tokens, p=2, dim=-1)  # (B, 36, 1024)
    rf_norm = F.normalize(rf_tokens, p=2, dim=-1)  # (B, 36, 16, 1024)

    # Compute similarities
    compressed_expanded = compressed_norm.unsqueeze(2)  # (B, 36, 1, 1024)
    similarities = (compressed_expanded * rf_norm).sum(dim=-1)  # (B, 36, 16)

    return similarities
