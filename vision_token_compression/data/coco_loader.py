"""
COCO DataLoader for Vision Token Compression Training
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pathlib import Path
from typing import Tuple, Optional
import random


class CocoDatasetForVision(CocoDetection):
    """
    COCO Dataset wrapper that returns only images (no annotations needed for vision tasks).
    This is used for unsupervised vision token compression training.
    """

    def __getitem__(self, index):
        """
        Returns:
            image: PIL Image
            label: Dummy label (0) since we only need images for token compression
        """
        img, _ = super().__getitem__(index)
        # Return dummy label 0 since we don't need annotations for compression
        return img, 0


def create_coco_dataloaders(
    root: str,
    annotation_file_train: str = None,
    annotation_file_val: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 336,
    pin_memory: bool = True,
    use_subset: bool = False,
    subset_size: int = 1000,
    val_subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create COCO train and validation dataloaders.

    Args:
        root: Path to COCO dataset root directory
        annotation_file_train: Path to train annotations JSON (e.g., 'annotations/instances_train2017.json')
        annotation_file_val: Path to val annotations JSON (e.g., 'annotations/instances_val2017.json')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image size (for CLIP ViT-L/14@336, use 336)
        pin_memory: Whether to pin memory for faster GPU transfer
        use_subset: Whether to use a subset of data (for debugging)
        subset_size: Size of training subset if use_subset=True
        val_subset_size: Size of validation subset (defaults to subset_size // 10)

    Returns:
        train_loader, val_loader
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"COCO root directory not found: {root}\n"
            f"Please update the 'data.coco_root' path in your config file."
        )

    # Default annotation paths
    if annotation_file_train is None:
        annotation_file_train = str(root / "annotations" / "instances_train2017.json")
    if annotation_file_val is None:
        annotation_file_val = str(root / "annotations" / "instances_val2017.json")

    # Image directories
    train_dir = root / "train2017"
    val_dir = root / "val2017"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"COCO train/val directories not found.\n"
            f"Expected structure:\n"
            f"  {root}/train2017/\n"
            f"  {root}/val2017/\n"
            f"  {root}/annotations/instances_train2017.json\n"
            f"  {root}/annotations/instances_val2017.json"
        )

    # COCO normalization (same as ImageNet - standard for most vision models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Resize to slightly larger
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    print(f"Loading COCO from: {root}")
    print(f"  Train annotations: {annotation_file_train}")
    print(f"  Val annotations: {annotation_file_val}")

    train_dataset = CocoDatasetForVision(
        root=str(train_dir),
        annFile=annotation_file_train,
        transform=train_transform
    )

    val_dataset = CocoDatasetForVision(
        root=str(val_dir),
        annFile=annotation_file_val,
        transform=val_transform
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create subsets if needed (for debugging)
    if use_subset:
        if val_subset_size is None:
            val_subset_size = subset_size // 10

        print(f"\n⚠️  Using SUBSET mode for debugging:")
        print(f"  Train subset: {subset_size} samples")
        print(f"  Val subset: {val_subset_size} samples")

        # Random subset
        train_indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(val_subset_size, len(val_dataset)))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test COCO dataloaders
    print("Testing COCO dataloaders...")
    print("NOTE: This requires COCO dataset to be downloaded")
    print("Download from: https://cocodataset.org/#download")
    print()

    # Example usage
    try:
        train_loader, val_loader = create_coco_dataloaders(
            root="/data/COCO",  # UPDATE THIS PATH
            batch_size=4,
            num_workers=0,
            use_subset=True,
            subset_size=20
        )

        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
            if batch_idx >= 2:
                break

        print("✓ COCO dataloader test passed!")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Please download COCO dataset first or update the path.")
