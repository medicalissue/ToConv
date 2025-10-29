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
    corresponding receptive field. Loss = 1 - mean_similarity (range: 0 to 2).

    Uses actual convolution parameters (kernel_size, stride, padding) to compute
    the exact receptive field for each compressed token.
    """

    def __init__(self, kernel_size: int = 4, stride: int = 4, padding: int = 0):
        """
        Args:
            kernel_size: Convolution kernel size (determines RF size)
            stride: Convolution stride (determines RF spacing)
            padding: Convolution padding (affects RF positioning)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

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
            compressed_tokens: (B, comp_h*comp_w, hidden_dim) - Compressed token grid
            original_tokens: (B, orig_h*orig_w, hidden_dim) - Original token grid from CLIP
            compressed_grid_size: Size of compressed grid (e.g., (12, 12))
            original_grid_size: Size of original grid (e.g., (24, 24))
            compute_stats: Whether to compute detailed statistics

        Returns:
            loss: Scalar loss tensor (1 - mean_similarity)
            info_dict: Dictionary with detailed metrics
        """
        batch_size = compressed_tokens.size(0)
        hidden_dim = compressed_tokens.size(2)

        comp_h, comp_w = compressed_grid_size
        orig_h, orig_w = original_grid_size

        # Reshape original tokens to 2D grid: (B, N, D) -> (B, H, W, D)
        original_grid = original_tokens.reshape(batch_size, orig_h, orig_w, hidden_dim)

        # Extract RF tokens for each compressed position using actual conv parameters
        rf_tokens_list = []

        for i in range(comp_h):
            for j in range(comp_w):
                # Calculate RF bounds in original grid
                # Conv formula: output_pos = (input_pos + padding - kernel_size) / stride + 1
                # Inverse: input_start = output_pos * stride - padding
                start_h = i * self.stride - self.padding
                start_w = j * self.stride - self.padding
                end_h = start_h + self.kernel_size
                end_w = start_w + self.kernel_size

                # Clamp to valid original grid range
                start_h_clamped = max(0, start_h)
                start_w_clamped = max(0, start_w)
                end_h_clamped = min(orig_h, end_h)
                end_w_clamped = min(orig_w, end_w)

                # Extract RF region: (B, rf_h, rf_w, D)
                rf_region = original_grid[:, start_h_clamped:end_h_clamped,
                                         start_w_clamped:end_w_clamped, :]

                # Flatten RF: (B, rf_h * rf_w, D)
                rf_flat = rf_region.reshape(batch_size, -1, hidden_dim)

                rf_tokens_list.append(rf_flat)

        # Stack all RFs: list of (B, rf_size, D) -> (B, comp_h*comp_w, rf_size, D)
        # Note: rf_size may vary for boundary tokens, so we need to handle this
        max_rf_size = max(rf.size(1) for rf in rf_tokens_list)

        # Pad smaller RFs with zeros and create mask
        rf_tokens_padded = []
        rf_masks = []

        for rf in rf_tokens_list:
            current_rf_size = rf.size(1)
            if current_rf_size < max_rf_size:
                # Pad: (B, current_size, D) -> (B, max_size, D)
                padding = torch.zeros(batch_size, max_rf_size - current_rf_size,
                                     hidden_dim, device=rf.device, dtype=rf.dtype)
                rf_padded = torch.cat([rf, padding], dim=1)
                # Mask: 1 for valid tokens, 0 for padding
                mask = torch.cat([torch.ones(current_rf_size, device=rf.device),
                                 torch.zeros(max_rf_size - current_rf_size, device=rf.device)])
            else:
                rf_padded = rf
                mask = torch.ones(max_rf_size, device=rf.device)

            rf_tokens_padded.append(rf_padded)
            rf_masks.append(mask)

        # Stack: (B, comp_h*comp_w, max_rf_size, D)
        rf_tokens = torch.stack(rf_tokens_padded, dim=1)
        # Masks: (comp_h*comp_w, max_rf_size)
        rf_masks = torch.stack(rf_masks, dim=0)

        # Normalize tokens for cosine similarity
        # Compressed: (B, comp_h*comp_w, D)
        compressed_norm = F.normalize(compressed_tokens, p=2, dim=-1)

        # RF tokens: (B, comp_h*comp_w, max_rf_size, D)
        rf_norm = F.normalize(rf_tokens, p=2, dim=-1)

        # Compute cosine similarity between each compressed token and its RF tokens
        # compressed_norm: (B, comp_h*comp_w, 1, D)
        # rf_norm: (B, comp_h*comp_w, max_rf_size, D)
        # Result: (B, comp_h*comp_w, max_rf_size) - similarity for each token in RF
        compressed_expanded = compressed_norm.unsqueeze(2)  # (B, comp_h*comp_w, 1, D)

        # Compute dot product (cosine similarity since vectors are normalized)
        similarities = (compressed_expanded * rf_norm).sum(dim=-1)  # (B, comp_h*comp_w, max_rf_size)

        # Apply mask to ignore padded tokens
        # rf_masks: (comp_h*comp_w, max_rf_size)
        masked_similarities = similarities * rf_masks.unsqueeze(0)  # (B, comp_h*comp_w, max_rf_size)

        # Average similarity per compressed token (only over valid tokens)
        # Sum over RF dimension and divide by number of valid tokens
        sum_per_token = masked_similarities.sum(dim=-1)  # (B, comp_h*comp_w)
        count_per_token = rf_masks.sum(dim=-1)  # (comp_h*comp_w,)
        avg_similarity_per_token = sum_per_token / count_per_token.unsqueeze(0)  # (B, comp_h*comp_w)

        # Overall average similarity: scalar (range: -1 to 1)
        mean_similarity = avg_similarity_per_token.mean()

        # Loss: 1 - similarity (range: 0 to 2, lower is better)
        # Perfect similarity (1.0) → loss = 0
        # No correlation (0.0) → loss = 1
        # Opposite (-1.0) → loss = 2
        loss = 1 - mean_similarity

        # Lazy statistics computation (only when needed for logging)
        if compute_stats:
            # Only compute stats on valid (non-masked) similarities
            valid_similarities = masked_similarities[rf_masks.unsqueeze(0).expand_as(masked_similarities) > 0]
            min_similarity = valid_similarities.min().item() if valid_similarities.numel() > 0 else 0.0
            max_similarity = valid_similarities.max().item() if valid_similarities.numel() > 0 else 0.0
            std_similarity = valid_similarities.std().item() if valid_similarities.numel() > 0 else 0.0

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
                'std_per_token': std_per_token
            }
        else:
            # Skip expensive statistics, only return essentials
            info_dict = {
                'cosine_sim_loss': loss.item(),
                'mean_similarity': mean_similarity.item(),
                'min_similarity': 0.0,  # Placeholder
                'max_similarity': 0.0,  # Placeholder
                'std_similarity': 0.0   # Placeholder
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
