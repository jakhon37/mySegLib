"""
Semantic Segmentation Dataset
"""
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation tasks."""

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        transform=None,
        num_classes: int = 2,
        cache_in_memory: bool = False,
        use_onehot: bool = False,
    ):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks
            transform: Albumentations transform
            num_classes: Number of segmentation classes
            cache_in_memory: If True, cache samples in memory after first load
            use_onehot: If True, return one-hot encoded masks (legacy mode)
        """
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.images = [img for img in image_paths if img.lower().endswith(valid_extensions)]
        self.masks = [mask for mask in mask_paths if mask.lower().endswith(valid_extensions)]
        self.transform = transform
        self.num_classes = num_classes
        self.cache_in_memory = cache_in_memory
        self.use_onehot = use_onehot
        self.cache = {} if cache_in_memory else None

        assert len(self.images) == len(
            self.masks
        ), f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return from cache if available
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(image.shape[1::-1], Image.NEAREST)
        mask = np.array(mask)
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert mask to appropriate format
        if self.use_onehot:
            # Legacy one-hot encoding (for backward compatibility)
            mask_onehot = np.eye(self.num_classes)[mask]
            mask_onehot = torch.from_numpy(mask_onehot).float()
            mask_tensor = mask_onehot.permute(2, 0, 1)
        else:
            # Return class indices directly
            if torch.is_tensor(mask):  # ToTensorV2 already converted it
                mask_tensor = mask.long()
            else:
                mask_tensor = torch.from_numpy(mask.astype(np.int64))

        result = (image, mask_tensor)

        # Cache if enabled
        if self.cache_in_memory:
            self.cache[idx] = result

        return result


def get_image_mask_pairs(images_dir: str, masks_dir: str) -> tuple:
    """
    Get sorted lists of image and mask paths.

    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks

    Returns:
        Tuple of (image_paths, mask_paths)
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Get image files
    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(valid_extensions) and not f.endswith("_mask.tif")
    ]

    # Get mask files
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(valid_extensions)]

    # Pair by exact name so one missing file can't shift every later pair
    def base_name(filename):
        return os.path.splitext(filename)[0].replace("_mask", "")

    masks_by_name = {base_name(f): f for f in mask_files}
    image_files = sorted(image_files, key=base_name)
    unmatched = [f for f in image_files if base_name(f) not in masks_by_name]
    if unmatched:
        warnings.warn(f"Skipping {len(unmatched)} images without a mask, e.g. {unmatched[0]}")
    image_files = [f for f in image_files if base_name(f) in masks_by_name]
    mask_files = [masks_by_name[base_name(f)] for f in image_files]

    # Create full paths
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]

    return image_paths, mask_paths
