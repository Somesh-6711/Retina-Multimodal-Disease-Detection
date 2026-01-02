# src/inference/fundus_gradcam.py

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import timm
from tqdm import tqdm

from src.configs.fundus_config import get_fundus_config
from src.data.fundus_dataset import APTOSFundusDataset, get_fundus_transforms
from src.explainability.gradcam_utils import GradCAMEffNet, GradCAMConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(num_examples: int = 8):
    cfg = get_fundus_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # === Recreate the same val split used in training ===
    df = pd.read_csv(cfg.train_csv)
    labels = df["diagnosis"].astype(int).values
    idx_all = np.arange(len(df))

    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )

    val_transforms = get_fundus_transforms(cfg.img_size, train=False)

    val_dataset = APTOSFundusDataset(
        csv_path=cfg.train_csv,
        img_dir=cfg.train_img_dir,
        transform=val_transforms,
        indices=val_idx,
    )

    # We'll just randomly pick some examples from val
    num_examples = min(num_examples, len(val_dataset))
    example_indices = random.sample(range(len(val_dataset)), num_examples)

    print(f"[INFO] Generating Grad-CAM for {num_examples} validation images...")

    # === Build model and load best weights ===
    model = timm.create_model(
        cfg.model_name,
        pretrained=False,
        in_chans=3,
        num_classes=cfg.num_classes,
    )
    state_dict = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    gradcam = GradCAMEffNet(
        model,
        GradCAMConfig(
            target_layer_name="conv_head",
            use_cuda=(device.type == "cuda"),
        ),
    )

    gradcam_dir = cfg.out_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    for local_idx in tqdm(example_indices, desc="Grad-CAM"):
        # Get transformed image + label
        img_tensor, label = val_dataset[local_idx]  # tensor, int
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

        # Get corresponding id_code (for naming)
        global_idx = val_idx[local_idx]
        img_id = df.iloc[global_idx]["id_code"]
        true_label = int(df.iloc[global_idx]["diagnosis"])

        # Forward pass to get predicted class
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())

        # Compute CAM for predicted class
        cam = gradcam.generate_cam(img_tensor, class_idx=pred_class)

        # For visualization, we need the *unnormalized* original image in RGB
        # Reload raw image from disk
        from src.data.fundus_dataset import APTOSFundusDataset as _APTOS

        # Use same loader logic as dataset
        ds_for_img = _APTOSFundusDatasetForImageOnly(df, cfg.train_img_dir)
        orig_img = ds_for_img.load_raw_image_by_id(img_id)

        overlay = GradCAMEffNet.overlay_heatmap_on_image(
            cam,
            orig_img,
            alpha=0.4,
        )

        # Save
        save_path = gradcam_dir / f"{img_id}_true{true_label}_pred{pred_class}_gradcam.png"
        GradCAMEffNet.save_overlay(overlay, str(save_path))

    print(f"[DONE] Grad-CAM examples saved under: {gradcam_dir}")


class _APTOSFundusDatasetForImageOnly:
    """
    Tiny helper class to reuse APTOS image loading logic for Grad-CAM visualization.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df
        self.img_dir = Path(img_dir)

    def _load_image(self, img_id: str) -> Image.Image:
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = self.img_dir / f"{img_id}{ext}"
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        raise FileNotFoundError(
            f"Image for id_code={img_id} not found in {self.img_dir}"
        )

    def load_raw_image_by_id(self, img_id: str) -> np.ndarray:
        img = self._load_image(img_id)
        return np.array(img)  # RGB uint8


if __name__ == "__main__":
    main(num_examples=8)
