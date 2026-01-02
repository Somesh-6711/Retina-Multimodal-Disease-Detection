# src/configs/oct_config.py

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class OCTConfig:
    # Basic experiment settings
    seed: int = 42
    num_classes: int = 4              # NORMAL, CNV, DME, DRUSEN
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Backbone (from timm)
    model_name: str = "resnet18"      # lighter than EfficientNet for CPU

    # To keep CPU training manageable, cap per-class samples
    max_train_per_class: int = 800    # you can lower if too slow

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (resolved relative to repo root)
    repo_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = repo_root / "data" / "raw" / "oct" / "kermany2018"

    def __post_init__(self):
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"

        self.out_dir = self.repo_root / "outputs" / "oct"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.out_dir / "oct_resnet18_best.pth"

        if not self.train_dir.exists():
            raise FileNotFoundError(f"OCT train dir not found: {self.train_dir}")
        if not self.val_dir.exists():
            raise FileNotFoundError(f"OCT val dir not found: {self.val_dir}")

        print(f"[OCT CONFIG] Using device: {self.device}")
        print(f"[OCT CONFIG] Train dir: {self.train_dir}")
        print(f"[OCT CONFIG] Val dir:   {self.val_dir}")


def get_oct_config() -> OCTConfig:
    return OCTConfig()
