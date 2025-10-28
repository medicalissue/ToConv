"""
ImageNet DataLoader for Vision Token Compression Training
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional
import random


def create_imagenet_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 336,
    pin_memory: bool = True,
    use_subset: bool = False,
    subset_size: int = 1000,
    val_subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create ImageNet train and validation dataloaders.

    Args:
        root: Path to ImageNet dataset (should contain 'train' and 'val' folders)
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
            f"ImageNet root directory not found: {root}\n"
            f"Please update the 'data.imagenet_root' path in your config file."
        )

    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet train/val directories not found.\n"
            f"Expected structure:\n"
            f"  {root}/train/\n"
            f"  {root}/val/"
        )

    # ImageNet normalization (standard)
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
    print(f"Loading ImageNet from: {root}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Number of classes: {len(train_dataset.classes)}")

    # Create subsets if needed (for debugging)
    if use_subset:
        if val_subset_size is None:
            val_subset_size = subset_size // 10

        print(f"\n⚠️  Using SUBSET mode for debugging:")
        print(f"  Train subset: {subset_size} samples")
        print(f"  Val subset: {val_subset_size} samples")

        # Random subset
        train_indices = random.sample(range(len(train_dataset)), subset_size)
        val_indices = random.sample(range(len(val_dataset)), val_subset_size)

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


def create_dummy_dataloaders(
    batch_size: int = 32,
    num_batches: int = 10,
    image_size: int = 336,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy dataloaders for testing without ImageNet.

    Args:
        batch_size: Batch size
        num_batches: Number of batches to generate
        image_size: Image size
        num_workers: Number of workers (use 0 for dummy data)

    Returns:
        train_loader, val_loader with random data
    """
    from torch.utils.data import TensorDataset

    num_samples = batch_size * num_batches

    # Generate random images and labels
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 1000, (num_samples,))

    train_dataset = TensorDataset(images, labels)
    val_dataset = TensorDataset(images[:num_samples // 10], labels[:num_samples // 10])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    print(f"⚠️  Using DUMMY dataloaders (random data)")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dummy dataloaders
    print("Testing dummy dataloaders...")
    train_loader, val_loader = create_dummy_dataloaders(batch_size=4, num_batches=5)

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
        if batch_idx >= 2:
            break

    print("✓ Dataloader test passed!")
