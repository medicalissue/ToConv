from .imagenet_loader import create_imagenet_dataloaders
from .coco_loader import create_coco_dataloaders
from typing import Tuple
from torch.utils.data import DataLoader


def create_dataloaders(
    dataset_type: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    pin_memory: bool,
    use_subset: bool,
    subset_size: int,
    imagenet_root: str = None,
    coco_root: str = None,
    coco_annotation_train: str = None,
    coco_annotation_val: str = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders based on dataset type.

    Args:
        dataset_type: Either "imagenet" or "coco"
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image size
        pin_memory: Whether to pin memory for faster GPU transfer
        use_subset: Whether to use a subset of data (for debugging)
        subset_size: Size of subset if use_subset=True
        imagenet_root: Path to ImageNet dataset (required if dataset_type="imagenet")
        coco_root: Path to COCO dataset (required if dataset_type="coco")
        coco_annotation_train: Path to COCO train annotations
        coco_annotation_val: Path to COCO val annotations
        **kwargs: Additional arguments passed to the dataset-specific loader

    Returns:
        train_loader, val_loader
    """
    if dataset_type == "imagenet":
        if imagenet_root is None:
            raise ValueError("imagenet_root must be provided when dataset_type='imagenet'")

        return create_imagenet_dataloaders(
            root=imagenet_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=pin_memory,
            use_subset=use_subset,
            subset_size=subset_size,
            **kwargs
        )

    elif dataset_type == "coco":
        if coco_root is None:
            raise ValueError("coco_root must be provided when dataset_type='coco'")

        return create_coco_dataloaders(
            root=coco_root,
            annotation_file_train=coco_annotation_train,
            annotation_file_val=coco_annotation_val,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=pin_memory,
            use_subset=use_subset,
            subset_size=subset_size,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unsupported dataset_type: {dataset_type}. "
            f"Supported types: 'imagenet', 'coco'"
        )


__all__ = [
    'create_imagenet_dataloaders',
    'create_coco_dataloaders',
    'create_dataloaders'
]
