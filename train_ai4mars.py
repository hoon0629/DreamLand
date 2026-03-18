"""
Fine-tune DeepLabV3+ on AI4MARS Dataset
=========================================
Run this AFTER downloading AI4MARS from data.nasa.gov

Download: https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars
Expected structure:
  ai4mars/
    images/    ← rover .jpg images
    labels/    ← .png label maps (pixel values 0-3 = soil/bedrock/sand/rock)
    train.txt  ← list of training image IDs
    val.txt    ← list of validation image IDs

Usage:
  python train_ai4mars.py --data_dir ./ai4mars --epochs 30 --output ./checkpoints
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class AI4MARSDataset(Dataset):
    """
    Loads AI4MARS images + segmentation label maps.
    Label values: 0=soil, 1=bedrock, 2=sand, 3=rock
    """
    NUM_CLASSES = 4

    def __init__(self, data_dir: str, split: str = "train",
                 img_size: int = 512, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment

        split_file = self.data_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.ids = [l.strip() for l in f if l.strip()]
        else:
            # Fallback: use all images in images/
            img_dir = self.data_dir / "images"
            self.ids = [p.stem for p in img_dir.glob("*.jpg")]

        self.img_dir = self.data_dir / "images"
        self.lbl_dir = self.data_dir / "labels"

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        print(f"  AI4MARS {split}: {len(self.ids)} samples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        lbl = Image.open(self.lbl_dir / f"{img_id}.png")

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        lbl = lbl.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augmentation (training only)
        if self.augment:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
                lbl = TF.vflip(lbl)
            # Random color jitter
            img = T.ColorJitter(brightness=0.3, contrast=0.3,
                                saturation=0.2, hue=0.1)(img)

        img_tensor = self.normalize(T.ToTensor()(img))
        lbl_tensor = torch.from_numpy(np.array(lbl)).long()

        # Clamp labels to valid range (255 = unlabeled in some AI4MARS versions)
        lbl_tensor = lbl_tensor.clamp(0, self.NUM_CLASSES - 1)

        return img_tensor, lbl_tensor


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Build DeepLabV3+ with ResNet-101 backbone, adapted for 4 Mars classes."""
    weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet101(weights=weights)

    # Replace final classifier head for 4-class output
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class MarsSegLoss(nn.Module):
    """
    Combined cross-entropy + dice loss for Mars terrain segmentation.
    Class weights upweight rare/critical classes (rocks, sand).
    """
    # Upweight rocks (safety critical) and sand (slippage risk)
    CLASS_WEIGHTS = torch.tensor([1.0, 1.2, 2.0, 3.0])  # soil/bedrock/sand/rock

    def __init__(self, device: str = "cpu"):
        super().__init__()
        w = self.CLASS_WEIGHTS.to(device)
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=255)

    def dice_loss(self, pred, target, num_classes=4, eps=1e-6):
        pred_soft = torch.softmax(pred, dim=1)
        loss = 0.0
        for c in range(num_classes):
            p = pred_soft[:, c]
            t = (target == c).float()
            intersection = (p * t).sum()
            loss += 1 - (2 * intersection + eps) / (p.sum() + t.sum() + eps)
        return loss / num_classes

    def forward(self, pred, target):
        return 0.7 * self.ce(pred, target) + 0.3 * self.dice_loss(pred, target)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_miou(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int = 4) -> float:
    """Compute mean Intersection over Union (mIoU)."""
    pred = pred.argmax(1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious = []
    for c in range(num_classes):
        inter = np.sum((pred == c) & (target == c))
        union = np.sum((pred == c) | (target == c))
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training DeepLabV3+ on AI4MARS | device={device}")

    # Data
    train_ds = AI4MARSDataset(args.data_dir, split="train",
                               img_size=args.img_size, augment=True)
    val_ds   = AI4MARSDataset(args.data_dir, split="val",
                               img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_model(num_classes=4, pretrained=True).to(device)

    # Optimizer: lower LR for backbone, higher for new head
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.classifier.parameters()) +
                   list(model.aux_classifier.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = MarsSegLoss(device=device)

    # Checkpoint dir
    os.makedirs(args.output, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out["out"], lbls)
            if "aux" in out:
                loss += 0.4 * criterion(out["aux"], lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss, val_miou = 0.0, 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += criterion(out["out"], lbls).item()
                val_miou += compute_miou(out["out"], lbls)
        val_loss /= len(val_loader)
        val_miou /= len(val_loader)

        print(f"  Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}")

        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt = os.path.join(args.output, "deeplabv3_ai4mars_best.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "miou": best_miou}, ckpt)
            print(f"  ✅ New best mIoU={best_miou:.4f} → saved to {ckpt}")

    print(f"\n🏁 Training complete. Best mIoU: {best_miou:.4f}")
    print(f"   Checkpoint: {args.output}/deeplabv3_ai4mars_best.pth")
    print(f"\n   To use in pipeline, update MarsTerrainSegmenter._remap_to_mars_classes():")
    print(f"   → Load checkpoint and return model output directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./ai4mars")
    parser.add_argument("--output",   default="./checkpoints")
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size",   type=int, default=512)
    parser.add_argument("--lr",       type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
