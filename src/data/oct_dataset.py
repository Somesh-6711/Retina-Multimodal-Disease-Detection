# src/data/oct_dataset.py

from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class KermanyOCTDataset(Dataset):
    """
    OCT dataset for the Kermany 2018 data.

    Expects directory structure like:

        root/
          CNV/
          DME/
          DRUSEN/
          NORMAL/

    Each subfolder contains images for that class.
    """

    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover class folders
        self.class_names = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        if not self.class_names:
            raise RuntimeError(f"No class folders found under {self.root_dir}")

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.rglob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    samples.append((img_path, cls_idx))

        if not samples:
            raise RuntimeError(f"No images found under {self.root_dir}")

        self.samples = samples
        print(f"[KermanyOCTDataset] Loaded {len(self.samples)} images from {self.root_dir}")
        print(f"[KermanyOCTDataset] Classes: {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Most OCT images are grayscale; convert to RGB for consistency
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


def get_oct_transforms(img_size: int = 224, train: bool = True):
    """
    Data augmentations for OCT images.

    Training:
      - Resize
      - Horizontal flip
      - Light geometric + intensity jitter
      - Normalize
    Validation:
      - Resize + Normalize only
    """
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
