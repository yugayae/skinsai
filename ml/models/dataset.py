"""SkinAI canonical dataset loader and image transform pipeline.

Provides:
1. Canonical transforms for training (with augmentations) and validation/inference.
2. SkinLesionDataset wrapping DatasetManifest or manifest CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Canonical normalization constants (ImageNet standard)
CANONICAL_MEAN = [0.485, 0.456, 0.406]
CANONICAL_STD = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE = (224, 224)


def get_canonical_transforms(
    is_train: bool = False,
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    mean: List[float] = CANONICAL_MEAN,
    std: List[float] = CANONICAL_STD,
) -> transforms.Compose:
    """Returns canonical transform pipeline for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion image classification."""

    def __init__(
        self,
        manifest_items: List[Dict[str, Union[str, int]]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        image_dir: Optional[Union[str, Path]] = None,
    ):
        self.items = manifest_items
        self.class_to_idx = class_to_idx
        self.transform = transform or get_canonical_transforms(is_train=False)
        self.image_dir = Path(image_dir) if image_dir else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        image_id = str(item["image_id"])
        class_id = str(item["class_id"])
        target = self.class_to_idx[class_id]

        # Resolve image path
        img_path = None
        img_path_str = item.get("image_path")
        if isinstance(img_path_str, str) and img_path_str.strip() and img_path_str.lower() != "nan":
            candidate = Path(img_path_str)
            if candidate.exists():
                img_path = candidate

        if img_path is None and self.image_dir:
            candidate = self.image_dir / f"{image_id}.jpg"
            if candidate.exists():
                img_path = candidate

        if img_path is None:
            raise FileNotFoundError(f"Image file not found for {image_id}")

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)

        return tensor, target, image_id
