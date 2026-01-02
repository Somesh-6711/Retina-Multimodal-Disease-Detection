# src/training/train_fusion.py

"""
Train a simple multimodal fusion model:
- Fundus encoder: EfficientNet-B0 (pretrained, frozen)
- OCT encoder:    ResNet18 (pretrained, frozen)
- Fusion head:    MLP on concatenated embeddings

Binary task:
    0 = normal
    1 = disease

We construct "virtual multimodal patients" by pairing:
    (fundus image, OCT image) with the same binary label.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

import timm
from tqdm import tqdm

from src.configs.fundus_config import get_fundus_config
from src.configs.oct_config import get_oct_config
from src.data.fundus_dataset import APTOSFundusDataset, get_fundus_transforms
from src.data.oct_dataset import KermanyOCTDataset, get_oct_transforms


# ------------------ Utils ------------------ #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class FusionConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_pairs_per_class: int = 1500  # normal + disease (so max 3000 total)

    num_classes: int = 2  # normal vs disease

    def __post_init__(self):
        self.fundus_cfg = get_fundus_config()
        self.oct_cfg = get_oct_config()

        self.device = torch.device(
            self.fundus_cfg.device if self.fundus_cfg.device == self.oct_cfg.device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.repo_root: Path = self.fundus_cfg.repo_root
        self.out_dir: Path = self.repo_root / "outputs" / "fusion"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path: Path = self.out_dir / "fusion_fundus_oct_best.pth"

        print(f"[FUSION CONFIG] Using device: {self.device}")
        print(f"[FUSION CONFIG] Output dir:  {self.out_dir}")


def get_fusion_config() -> FusionConfig:
    return FusionConfig()


# ------------------ Fusion Dataset ------------------ #

class FusionPairDataset(Dataset):
    """
    Dataset of paired (fundus_image, oct_image, binary_label).

    - fundus_label_bin: 0 normal, 1 disease (from APTOS diagnosis)
    - oct_label_bin:    0 normal, 1 disease (from Kermany classes)
    """

    def __init__(
        self,
        fundus_dataset: APTOSFundusDataset,
        oct_dataset: KermanyOCTDataset,
        fundus_labels_bin: np.ndarray,
        oct_labels_bin: np.ndarray,
        max_pairs_per_class: int = 1500,
    ):
        self.fundus_dataset = fundus_dataset
        self.oct_dataset = oct_dataset

        self.fundus_labels_bin = fundus_labels_bin
        self.oct_labels_bin = oct_labels_bin

        # Indices by class
        self.fundus_idx_by_class = {
            c: np.where(self.fundus_labels_bin == c)[0].tolist() for c in [0, 1]
        }
        self.oct_idx_by_class = {
            c: np.where(self.oct_labels_bin == c)[0].tolist() for c in [0, 1]
        }

        # Determine how many pairs we can reasonably form per class
        max_pairs_possible = min(
            len(self.fundus_idx_by_class[0]),
            len(self.oct_idx_by_class[0]),
            len(self.fundus_idx_by_class[1]),
            len(self.oct_idx_by_class[1]),
            max_pairs_per_class,
        )

        if max_pairs_possible == 0:
            raise RuntimeError("Not enough samples to build fusion pairs.")

        self.samples: List[Tuple[int, int, int]] = []  # (fundus_idx, oct_idx, label_bin)
        for c in [0, 1]:  # normal, disease
            f_indices = self.fundus_idx_by_class[c]
            o_indices = self.oct_idx_by_class[c]
            for _ in range(max_pairs_possible):
                f_idx = random.choice(f_indices)
                o_idx = random.choice(o_indices)
                self.samples.append((f_idx, o_idx, c))

        random.shuffle(self.samples)
        print(
            f"[FUSION DATA] Built {len(self.samples)} pairs "
            f"({max_pairs_possible} per class, normal & disease)."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        f_idx, o_idx, label_bin = self.samples[idx]

        fundus_img, _ = self.fundus_dataset[f_idx]
        oct_img, _ = self.oct_dataset[o_idx]

        return fundus_img, oct_img, label_bin


# ------------------ Fusion Model ------------------ #

class FusionModel(nn.Module):
    """
    Wraps pre-trained fundus + OCT encoders (frozen) and
    trains a small MLP head on concatenated embeddings.
    """

    def __init__(
        self,
        fundus_model: nn.Module,
        oct_model: nn.Module,
        fundus_feat_dim: int,
        oct_feat_dim: int,
        num_classes: int = 2,
    ):
        super().__init__()
        self.fundus_model = fundus_model
        self.oct_model = oct_model

        # Freeze encoders
        for p in self.fundus_model.parameters():
            p.requires_grad = False
        for p in self.oct_model.parameters():
            p.requires_grad = False

        fusion_dim = fundus_feat_dim + oct_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def _encode_fundus(self, x: torch.Tensor) -> torch.Tensor:
        # timm EfficientNet: forward_features -> global_pool
        feats = self.fundus_model.forward_features(x)
        pooled = self.fundus_model.global_pool(feats)
        return pooled.view(pooled.size(0), -1)

    def _encode_oct(self, x: torch.Tensor) -> torch.Tensor:
        # timm ResNet18: forward_features -> global_pool
        feats = self.oct_model.forward_features(x)
        pooled = self.oct_model.global_pool(feats)
        return pooled.view(pooled.size(0), -1)

    def forward(self, x_fundus: torch.Tensor, x_oct: torch.Tensor) -> torch.Tensor:
        f_emb = self._encode_fundus(x_fundus)
        o_emb = self._encode_oct(x_oct)
        z = torch.cat([f_emb, o_emb], dim=1)
        logits = self.classifier(z)
        return logits


# ------------------ Main train loop ------------------ #

def main():
    cfg = get_fusion_config()
    set_seed(cfg.seed)

    device = cfg.device

    # ---------- Build base fundus dataset (train split only) ----------
    f_cfg = cfg.fundus_cfg
    fundus_transforms = get_fundus_transforms(f_cfg.img_size, train=True)

    df_fundus = pd.read_csv(f_cfg.train_csv)
    fundus_labels_orig = df_fundus["diagnosis"].astype(int).values
    # Binary: 0 = normal, 1 = any DR
    fundus_labels_bin = (fundus_labels_orig > 0).astype(int)

    fundus_dataset = APTOSFundusDataset(
        csv_path=f_cfg.train_csv,
        img_dir=f_cfg.train_img_dir,
        transform=fundus_transforms,
        indices=None,  # use all
    )

    # ---------- Build base OCT dataset (train subset) ----------
    o_cfg = cfg.oct_cfg
    oct_transforms = get_oct_transforms(o_cfg.img_size, train=True)

    full_oct_train = KermanyOCTDataset(o_cfg.train_dir, transform=oct_transforms)

    # Map OCT labels to binary: NORMAL=0, others=1
    oct_labels_orig = np.array([label for _, label in full_oct_train.samples])
    oct_labels_bin = np.zeros_like(oct_labels_orig)

    # NORMAL is whichever class name == "NORMAL"
    try:
        normal_idx = full_oct_train.class_names.index("NORMAL")
    except ValueError:
        raise RuntimeError(
            f"NORMAL class not found in OCT class names: {full_oct_train.class_names}"
        )

    oct_labels_bin[oct_labels_orig != normal_idx] = 1  # disease

    # Optionally limit OCT samples per class BEFORE pairing (to speed up)
    # We'll just reuse all; FusionPairDataset will handle per-class pairing counts.

    # ---------- Build fusion dataset ----------
    fusion_dataset = FusionPairDataset(
        fundus_dataset=fundus_dataset,
        oct_dataset=full_oct_train,
        fundus_labels_bin=fundus_labels_bin,
        oct_labels_bin=oct_labels_bin,
        max_pairs_per_class=cfg.max_pairs_per_class,
    )

    # Simple random split into train/val for fusion
    indices = np.arange(len(fusion_dataset))
    np.random.shuffle(indices)

    val_ratio = 0.2
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = torch.utils.data.Subset(fusion_dataset, train_indices)
    val_subset = torch.utils.data.Subset(fusion_dataset, val_indices)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[FUSION DATA] Train pairs: {len(train_subset)}, "
        f"Val pairs: {len(val_subset)}"
    )

    # ---------- Build fusion model ----------
    # Recreate fundus & OCT encoders and load pre-trained weights
    fundus_model = timm.create_model(
        f_cfg.model_name,
        pretrained=False,
        in_chans=3,
        num_classes=f_cfg.num_classes,
    )
    fundus_state = torch.load(f_cfg.checkpoint_path, map_location=device)
    fundus_model.load_state_dict(fundus_state)

    oct_model = timm.create_model(
        o_cfg.model_name,
        pretrained=False,
        in_chans=3,
        num_classes=o_cfg.num_classes,
    )
    oct_state = torch.load(o_cfg.checkpoint_path, map_location=device)
    oct_model.load_state_dict(oct_state)

    fundus_model.to(device)
    oct_model.to(device)

    # timm models expose embedding dim as num_features
    fundus_feat_dim = fundus_model.num_features
    oct_feat_dim = oct_model.num_features

    fusion_model = FusionModel(
        fundus_model=fundus_model,
        oct_model=oct_model,
        fundus_feat_dim=fundus_feat_dim,
        oct_feat_dim=oct_feat_dim,
        num_classes=cfg.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        fusion_model.classifier.parameters(),  # only fusion head
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== FUSION Epoch {epoch}/{cfg.num_epochs} =====")

        # ----- TRAIN -----
        fusion_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Train-Fusion", ncols=80)
        for f_imgs, o_imgs, labels_bin in pbar:
            f_imgs = f_imgs.to(device)
            o_imgs = o_imgs.to(device)
            labels_bin = labels_bin.to(device)

            optimizer.zero_grad()
            outputs = fusion_model(f_imgs, o_imgs)
            loss = criterion(outputs, labels_bin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels_bin.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_bin).sum().item()
            total += labels_bin.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"[Train-Fusion] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # ----- VALIDATION -----
        fusion_model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for f_imgs, o_imgs, labels_bin in tqdm(
                val_loader, desc="Val-Fusion", ncols=80
            ):
                f_imgs = f_imgs.to(device)
                o_imgs = o_imgs.to(device)
                labels_bin = labels_bin.to(device)

                outputs = fusion_model(f_imgs, o_imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels_bin).sum().item()
                val_total += labels_bin.size(0)

                all_labels.extend(labels_bin.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = val_correct / val_total
        print(f"[Val-Fusion] Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(fusion_model.state_dict(), cfg.checkpoint_path)
            print(
                f"[SAVE] New best fusion model -> {cfg.checkpoint_path} "
                f"(Val Acc={val_acc:.4f})"
            )

        print("\n[Val-Fusion] Classification report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=["normal", "disease"],
            )
        )

    print(f"\nFusion training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best fusion checkpoint: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
