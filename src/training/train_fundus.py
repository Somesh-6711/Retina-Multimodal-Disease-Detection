# src/training/train_fundus.py

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import timm
from tqdm import tqdm

from src.configs.fundus_config import get_fundus_config
from src.data.fundus_dataset import APTOSFundusDataset, get_fundus_transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = get_fundus_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # ---- Build transforms ----
    train_transforms = get_fundus_transforms(cfg.img_size, train=True)
    val_transforms = get_fundus_transforms(cfg.img_size, train=False)

    # ---- Read full CSV once to create stratified split ----
    import pandas as pd

    df = pd.read_csv(cfg.train_csv)
    if "diagnosis" not in df.columns:
        raise ValueError(f"'diagnosis' column not found in {cfg.train_csv}")
    labels = df["diagnosis"].astype(int).values
    idx_all = np.arange(len(df))

    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )

    # ---- Datasets ----
    train_dataset = APTOSFundusDataset(
        csv_path=cfg.train_csv,
        img_dir=cfg.train_img_dir,
        transform=train_transforms,
        indices=train_idx,
    )

    val_dataset = APTOSFundusDataset(
        csv_path=cfg.train_csv,
        img_dir=cfg.train_img_dir,
        transform=val_transforms,
        indices=val_idx,
    )

    # ---- Dataloaders ----
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

    print(f"[DATA] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

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
        print(f"\n===== Epoch {epoch}/{cfg.num_epochs} =====")

        # ================== TRAIN ==================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Train", ncols=80)
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
        print(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # ================== VALIDATION ==================
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", ncols=80):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = val_correct / val_total
        print(f"[Val] Acc: {val_acc:.4f}")

        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"[SAVE] New best model -> {cfg.checkpoint_path} (Val Acc={val_acc:.4f})")

        # ---- Optionally: classification report each epoch ----
        print("\n[Val] Classification report:")
        print(classification_report(all_labels, all_preds))

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
