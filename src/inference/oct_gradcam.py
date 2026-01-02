# src/inference/oct_gradcam.py

from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm

from src.configs.oct_config import get_oct_config
from src.data.oct_dataset import KermanyOCTDataset, get_oct_transforms
from src.explainability.gradcam_utils import GradCAMEffNet, GradCAMConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(num_examples: int = 8):
    cfg = get_oct_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # === Build validation dataset (we’ll take random examples) ===
    val_transforms = get_oct_transforms(cfg.img_size, train=False)
    val_dataset = KermanyOCTDataset(cfg.val_dir, transform=val_transforms)

    num_examples = min(num_examples, len(val_dataset))
    example_indices = random.sample(range(len(val_dataset)), num_examples)
    print(f"[OCT-GradCAM] Generating Grad-CAM for {num_examples} OCT images...")

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

    # Grad-CAM wrapper for ResNet (last conv block is `layer4`)
    gradcam = GradCAMEffNet(
        model,
        GradCAMConfig(
            target_layer_name="layer4",
            use_cuda=(device.type == "cuda"),
        ),
    )

    gradcam_dir = cfg.out_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # We need a helper to reload original OCT images at full resolution
    raw_helper = _OCTRawImageHelper(cfg.val_dir, val_dataset.class_names)

    for idx in tqdm(example_indices, desc="OCT-GradCAM"):
        img_tensor, label = val_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1,3,H,W)

        # We also need to know which file this came from
        img_path, _ = val_dataset.samples[idx]
        cls_idx = label
        cls_name = val_dataset.class_names[cls_idx]

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())

        cam = gradcam.generate_cam(img_tensor, class_idx=pred_class)

        # Load original raw grayscale or RGB image as RGB uint8
        orig_img = raw_helper.load_raw_rgb(img_path)

        overlay = GradCAMEffNet.overlay_heatmap_on_image(
            cam,
            orig_img,
            alpha=0.4,
        )

        save_path = gradcam_dir / f"{img_path.stem}_true{cls_name}_pred{val_dataset.class_names[pred_class]}_gradcam.png"
        GradCAMEffNet.save_overlay(overlay, str(save_path))

    print(f"[DONE] OCT Grad-CAM images saved under: {gradcam_dir}")


class _OCTRawImageHelper:
    """
    Small helper to reload original OCT images for visualization.
    """

    def __init__(self, root_dir: Path, class_names):
        self.root_dir = Path(root_dir)
        self.class_names = class_names

    def load_raw_rgb(self, img_path: Path) -> np.ndarray:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)


if __name__ == "__main__":
    main(num_examples=8)
