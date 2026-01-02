# src/configs/fundus_config.py

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class FundusConfig:
    # Basic experiment settings
    seed: int = 42
    num_classes: int = 5        # APTOS DR: 0-4
    img_size: int = 224         # resize shorter side or square crop
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 5         # start small, increase later
    lr: float = 1e-4
    weight_decay: float = 1e-5
    model_name: str = "tf_efficientnet_b0_ns"  # from timm

    # Train/val split
    val_ratio: float = 0.2

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (resolved relative to repo root)
    repo_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = repo_root / "data" / "raw" / "fundus" / "aptos2019"

    def __post_init__(self):
        self.train_csv = self.data_root / "train.csv"
        self.train_img_dir = self.data_root / "train_images"
        self.test_img_dir = self.data_root / "test_images"

        self.out_dir = self.repo_root / "outputs" / "fundus"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.out_dir / "fundus_effnet_b0_best.pth"
        self.log_path = self.out_dir / "fundus_training_log.txt"

        if not self.train_csv.exists():
            raise FileNotFoundError(f"train.csv not found at {self.train_csv}")
        if not self.train_img_dir.exists():
            raise FileNotFoundError(f"train_images directory not found at {self.train_img_dir}")

        print(f"[CONFIG] Using device: {self.device}")
        print(f"[CONFIG] Train CSV: {self.train_csv}")
        print(f"[CONFIG] Train images dir: {self.train_img_dir}")


def get_fundus_config() -> FundusConfig:
    return FundusConfig()
