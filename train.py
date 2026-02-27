"""
train.py  —  Duality AI Offroad Segmentation
UNet (ResNet50) + Combined CE + Tversky Loss

Usage:
    python train.py                          # uses config.py defaults
    python train.py --epochs 50 --lr 3e-5   # override hyperparams
    python train.py --resume                 # auto-resume from latest checkpoint
"""

import os, json, glob, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    VALUE_MAP, NUM_CLASSES, CLASS_NAMES, CLASS_WEIGHTS, PALETTE,
    TRAIN_DIR, VAL_DIR, KAGGLE_BASE,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    CKPT_FREQ, NUM_WORKERS, ENCODER_NAME, ENCODER_WEIGHTS,
    TVERSKY_ALPHA, TVERSKY_BETA,
    OUTPUT_DIR, CKPT_DIR, BEST_MODEL, FINAL_MODEL, HISTORY_JSON
)


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train UNet-ResNet50 segmentation")
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    p.add_argument("--lr",        type=float, default=LR)
    p.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    p.add_argument("--patience",  type=int,   default=PATIENCE)
    p.add_argument("--resume",    action="store_true", help="Auto-resume from latest checkpoint")
    p.add_argument("--data",      type=str,   default=None, help="Override data base path")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "Color_Images")
        self.mask_dir  = os.path.join(root_dir, "Segmentation")
        self.images    = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def convert_mask(self, mask):
        out = np.zeros_like(mask)
        for raw, new in VALUE_MAP.items():
            out[mask == raw] = new
        return out

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(
            cv2.imread(os.path.join(self.image_dir, self.images[idx])),
            cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.images[idx]),
                          cv2.IMREAD_UNCHANGED)
        mask = self.convert_mask(mask)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        return img, mask.long()


# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


# ── Loss functions ────────────────────────────────────────────────────────────
class TverskyLoss(nn.Module):
    def __init__(self, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1e-6):
        super().__init__()
        self.alpha = alpha; self.beta = beta; self.smooth = smooth

    def forward(self, preds, targets):
        preds  = torch.softmax(preds, dim=1)
        t_oh   = nn.functional.one_hot(targets, NUM_CLASSES).permute(0,3,1,2).float()
        TP     = (preds * t_oh).sum(dim=(2,3))
        FP     = (preds * (1-t_oh)).sum(dim=(2,3))
        FN     = ((1-preds) * t_oh).sum(dim=(2,3))
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(NUM_CLASSES):
        p = (pred == cls); t = (target == cls)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float("nan") if union == 0 else (inter/union).item())
    return np.nanmean(ious), ious


def compute_pixel_acc(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_iou):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:03d}.pth")
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "history": history, "best_iou": best_iou
    }, path)
    print(f"    💾 Checkpoint → {path}")


def find_latest_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pth")))
    return ckpts[-1] if ckpts else None


# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_curves(history, best_iou):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    e = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("UNet-ResNet50 Training History", fontsize=15, fontweight="bold")

    axes[0,0].plot(e, history["train_loss"], label="Train", color="steelblue", lw=2)
    axes[0,0].plot(e, history["val_loss"],   label="Val",   color="coral",     lw=2)
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(e, history["train_iou"], label="Train", color="steelblue", lw=2)
    axes[0,1].plot(e, history["val_iou"],   label="Val",   color="coral",     lw=2)
    axes[0,1].axhline(best_iou, color="green", ls="--", label=f"Best={best_iou:.4f}")
    axes[0,1].set_title("IoU"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(e, history["val_acc"], color="purple", lw=2)
    axes[1,0].set_title("Val Pixel Accuracy"); axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(e, history["lr"], color="darkorange", lw=2)
    axes[1,1].set_title("Learning Rate"); axes[1,1].set_yscale("log"); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve data paths (Kaggle vs local)
    base = args.data if args.data else (KAGGLE_BASE if os.path.isdir(KAGGLE_BASE) else None)
    train_dir = os.path.join(base, "train") if base else TRAIN_DIR
    val_dir   = os.path.join(base, "val")   if base else VAL_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = SegDataset(train_dir, train_transform)
    val_ds   = SegDataset(val_dir,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=NUM_WORKERS)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                     classes=NUM_CLASSES, activation=None).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw      = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=cw)
    tv_loss = TverskyLoss()
    combined_loss = lambda p, t: ce_loss(p, t) + tv_loss(p, t)

    # ── Optim ─────────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    history   = dict(train_loss=[], val_loss=[], train_iou=[], val_iou=[], val_acc=[], lr=[])
    best_iou  = 0.0
    patience_counter = 0
    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            history     = ckpt["history"]
            best_iou    = ckpt["best_iou"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch-1}  (best IoU={best_iou:.4f})")
        else:
            print("No checkpoint found — starting fresh.")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("=" * 65)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_losses, tr_ious = [], []

        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs} [Train]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = combined_loss(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tr_losses.append(loss.item())
            iou, _ = compute_iou(out.detach(), masks)
            tr_ious.append(iou)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        model.eval(); vl_losses, vl_ious, vl_accs = [], [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Ep {epoch:02d}/{args.epochs} [Val]  ", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    out = model(imgs)
                    vl_losses.append(combined_loss(out, masks).item())
                iou, _ = compute_iou(out, masks)
                vl_ious.append(iou); vl_accs.append(compute_pixel_acc(out, masks))

        tr_loss = np.mean(tr_losses); vl_loss = np.mean(vl_losses)
        tr_iou  = np.nanmean(tr_ious); vl_iou = np.nanmean(vl_ious)
        vl_acc  = np.mean(vl_accs)

        for k, v in zip(["train_loss","val_loss","train_iou","val_iou","val_acc","lr"],
                         [tr_loss, vl_loss, tr_iou, vl_iou, vl_acc, current_lr]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{args.epochs}  TrLoss={tr_loss:.4f}  VlLoss={vl_loss:.4f}  "
              f"TrIoU={tr_iou:.4f}  VlIoU={vl_iou:.4f}  VlAcc={vl_acc:.4f}  LR={current_lr:.2e}")

        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), BEST_MODEL)
            patience_counter = 0
            print(f"    ⭐ New best Val IoU={best_iou:.4f}  → {BEST_MODEL}")
        else:
            patience_counter += 1
            print(f"    ⏳ No improvement {patience_counter}/{args.patience}")

        if epoch % CKPT_FREQ == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_iou)

        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping at epoch {epoch}.")
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_iou)
            break

    torch.save(model.state_dict(), FINAL_MODEL)
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)

    save_curves(history, best_iou)

    print(f"\n{'='*60}")
    print(f"Training complete!  Best Val IoU: {best_iou:.4f}")
    print(f"  Best model  → {BEST_MODEL}")
    print(f"  Curves      → {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
