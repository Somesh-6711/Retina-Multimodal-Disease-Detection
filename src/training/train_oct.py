# src/training/train_oct.py

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

import timm
from tqdm import tqdm

from src.configs.oct_config import get_oct_config
from src.data.oct_dataset import KermanyOCTDataset, get_oct_transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_balanced_subset(dataset: KermanyOCTDataset, max_per_class: int) -> Subset:
    """
    Take at most `max_per_class` samples per class to keep training manageable.
    """
    labels = np.array([label for _, label in dataset.samples])
    subset_indices = []

    for cls_idx in range(len(dataset.class_names)):
        cls_indices = np.where(labels == cls_idx)[0].tolist()
        random.shuffle(cls_indices)
        chosen = cls_indices[: max_per_class] if max_per_class > 0 else cls_indices
        subset_indices.extend(chosen)

    random.shuffle(subset_indices)
    print(
        f"[OCT] Using {len(subset_indices)} samples for training "
        f"(max {max_per_class} per class)."
    )
    return Subset(dataset, subset_indices)


def main():
    cfg = get_oct_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # ---- Datasets ----
    train_transforms = get_oct_transforms(cfg.img_size, train=True)
    val_transforms = get_oct_transforms(cfg.img_size, train=False)

    full_train_dataset = KermanyOCTDataset(cfg.train_dir, transform=train_transforms)
    val_dataset = KermanyOCTDataset(cfg.val_dir, transform=val_transforms)

    # Balanced subset for CPU-friendly training
    if cfg.max_train_per_class is not None and cfg.max_train_per_class > 0:
        train_dataset = build_balanced_subset(full_train_dataset, cfg.max_train_per_class)
    else:
        train_dataset = full_train_dataset

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    print(f"[OCT DATA] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- Model ----
    model = timm.create_model(
        cfg.model_name,
        pretrained=True,
        in_chans=3,
        num_classes=cfg.num_classes,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== OCT Epoch {epoch}/{cfg.num_epochs} =====")

        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Train-OCT", ncols=80)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"[Train-OCT] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val-OCT", ncols=80):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = val_correct / val_total
        print(f"[Val-OCT] Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"[SAVE] New best OCT model -> {cfg.checkpoint_path} (Val Acc={val_acc:.4f})")

        print("\n[Val-OCT] Classification report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=full_train_dataset.class_names,
            )
        )

    print(f"\nOCT training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best OCT checkpoint: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
