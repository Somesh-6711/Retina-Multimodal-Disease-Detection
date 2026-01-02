# src/data/fundus_dataset.py

from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class APTOSFundusDataset(Dataset):
    """
    Dataset wrapper for APTOS 2019 Blindness Detection.

    Expects:
        - CSV with columns: id_code, diagnosis
        - Images in a directory, named <id_code>.png or .jpg

    Optionally:
        - 'indices' to select a subset (for train/val split)
        - 'transform' for augmentations / normalization
    """

    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        transform: Optional[Callable] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        if "id_code" not in df.columns or "diagnosis" not in df.columns:
            raise ValueError(
                f"Expected columns 'id_code' and 'diagnosis' in {csv_path}, "
                f"got {df.columns.tolist()}"
            )

        # Drop NaNs just in case
        df = df.dropna(subset=["id_code", "diagnosis"]).reset_index(drop=True)
        df["diagnosis"] = df["diagnosis"].astype(int)

        if indices is not None:
            df = df.iloc[list(indices)].reset_index(drop=True)

        self.df = df
        print(f"[APTOSFundusDataset] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_id: str) -> Image.Image:
        """
        Try common image extensions for APTOS: .png, .jpg, .jpeg
        """
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = self.img_dir / f"{img_id}{ext}"
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        raise FileNotFoundError(
            f"Image for id_code={img_id} not found with extensions "
            f".png/.jpg/.jpeg in {self.img_dir}"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = int(row["diagnosis"])

        img = self._load_image(img_id)
        img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            # Fallback: simple tensor conversion
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


def get_fundus_transforms(img_size: int = 224, train: bool = True):
    """
    Albumentations transforms for fundus images.

    Training:
      - Resize
      - Random rotations / flips
      - Brightness/contrast changes
      - Optional CLAHE
      - Normalize + ToTensorV2

    Validation / Test:
      - Resize
      - Normalize + ToTensorV2
    """
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.3),
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
