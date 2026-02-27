"""
test.py  —  Duality AI Offroad Segmentation
Runs inference on the test set using the best saved model.
Applies Test-Time Augmentation (horizontal flip average).

Usage:
    python test.py                             # uses best_model.pth
    python test.py --model path/to/model.pth   # use specific checkpoint
    python test.py --no-tta                    # disable TTA
    python test.py --save-masks                # also save raw class-ID masks
"""

import os, glob, argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from config import (
    VALUE_MAP, NUM_CLASSES, CLASS_NAMES, PALETTE,
    TEST_DIR, KAGGLE_BASE,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    ENCODER_NAME, ENCODER_WEIGHTS,
    OUTPUT_DIR, CKPT_DIR, BEST_MODEL, FINAL_MODEL
)


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Test UNet-ResNet50 segmentation")
    p.add_argument("--model",       type=str, default=None, help="Path to model weights")
    p.add_argument("--no-tta",      action="store_true",   help="Disable TTA")
    p.add_argument("--save-masks",  action="store_true",   help="Save raw class-ID mask PNGs")
    p.add_argument("--comparisons", type=int, default=8,   help="Number of comparison images to save")
    p.add_argument("--data",        type=str, default=None,help="Override test data path")
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
        fname = self.images[idx]
        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask.long(), fname


val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou(pred_probs, target):
    pred = torch.argmax(pred_probs, dim=1)
    ious = []
    for cls in range(NUM_CLASSES):
        p = (pred == cls); t = (target == cls)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float("nan") if union == 0 else (inter/union).item())
    return np.nanmean(ious), ious


# ── Visualisation ─────────────────────────────────────────────────────────────
PAL = np.array(PALETTE, dtype=np.uint8)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def mask_to_color(mask_np):
    h, w = mask_np.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask_np == c] = PAL[c]
    return out


def pick_model_path(arg_model):
    """Priority: CLI arg → best_model.pth → latest checkpoint → final_model.pth"""
    if arg_model and os.path.exists(arg_model):
        return arg_model, "CLI argument"
    if os.path.exists(BEST_MODEL):
        return BEST_MODEL, "best_model.pth (highest Val IoU)"
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
    if ckpts:
        return ckpts[-1], f"latest checkpoint ({os.path.basename(ckpts[-1])})"
    if os.path.exists(FINAL_MODEL):
        return FINAL_MODEL, "final_model.pth"
    raise FileNotFoundError("No model found. Run train.py first.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Paths
    base     = args.data if args.data else (KAGGLE_BASE if os.path.isdir(KAGGLE_BASE) else None)
    test_dir = os.path.join(base, "test") if base else TEST_DIR

    pred_dir    = os.path.join("predictions")
    color_dir   = os.path.join(pred_dir, "masks_color")
    raw_dir     = os.path.join(pred_dir, "masks_raw")
    compare_dir = os.path.join(pred_dir, "comparisons")
    for d in [pred_dir, color_dir, compare_dir]:
        os.makedirs(d, exist_ok=True)
    if args.save_masks:
        os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Model
    model_path, reason = pick_model_path(args.model)
    print(f"Loading: {model_path}  ({reason})")

    model  = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None,
                      classes=NUM_CLASSES, activation=None).to(device)
    ckpt   = torch.load(model_path, map_location=device)
    state  = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print("Model loaded ✅")

    # DataLoader — returns fname as 3rd element
    test_ds = SegDataset(test_dir, val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    print(f"Test samples: {len(test_ds)}")

    tta = not args.no_tta
    print(f"TTA: {'ON (horizontal flip average)' if tta else 'OFF'}")

    all_ious, all_class_ious = [], []
    saved_comparisons = 0

    with torch.no_grad():
        for imgs, masks, fnames in tqdm(test_loader, desc="Inference"):
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                p_orig = torch.softmax(model(imgs), dim=1)
                if tta:
                    p_flip = torch.flip(
                        torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1),
                        dims=[3])
                    probs = (p_orig + p_flip) / 2.0
                else:
                    probs = p_orig

            preds = torch.argmax(probs, dim=1)
            iou, class_iou = compute_iou(probs, masks)
            all_ious.append(iou); all_class_ious.append(class_iou)

            for i in range(imgs.shape[0]):
                base_name = os.path.splitext(fnames[i])[0]
                pred_np   = preds[i].cpu().numpy().astype(np.uint8)
                col       = mask_to_color(pred_np)

                # Colour mask PNG
                cv2.imwrite(os.path.join(color_dir, f"{base_name}_pred.png"),
                            cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

                # Raw class-ID mask (optional)
                if args.save_masks:
                    Image.fromarray(pred_np).save(os.path.join(raw_dir, f"{base_name}_mask.png"))

                # Side-by-side comparison (first N)
                if saved_comparisons < args.comparisons:
                    img_np  = imgs[i].cpu().permute(1,2,0).numpy()
                    img_np  = np.clip(img_np * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
                    gt_col  = mask_to_color(masks[i].cpu().numpy().astype(np.uint8))

                    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                    fig.suptitle(f"Test Sample #{saved_comparisons+1}  ({fnames[i]})",
                                 fontsize=12, fontweight="bold")
                    ax[0].imshow(img_np);   ax[0].set_title("Input");         ax[0].axis("off")
                    ax[1].imshow(gt_col);   ax[1].set_title("Ground Truth");  ax[1].axis("off")
                    ax[2].imshow(col);      ax[2].set_title("Prediction TTA" if tta else "Prediction")
                    ax[2].axis("off")

                    patches = [Patch(color=np.array(PALETTE[c])/255, label=CLASS_NAMES[c])
                               for c in range(NUM_CLASSES)]
                    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
                               bbox_to_anchor=(0.5, -0.04))
                    plt.tight_layout()
                    plt.savefig(os.path.join(compare_dir, f"comparison_{saved_comparisons+1:02d}.png"),
                                dpi=120, bbox_inches="tight")
                    plt.close()
                    saved_comparisons += 1

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    test_mean_iou = np.nanmean(all_ious)
    avg_class_iou = np.nanmean(all_class_ious, axis=0)

    print(f"\n{'='*50}")
    print(f"TEST RESULTS  ({'TTA' if tta else 'No TTA'})")
    print(f"Mean IoU : {test_mean_iou:.4f}")
    print(f"{'='*50}")
    for name, iou in zip(CLASS_NAMES, avg_class_iou):
        bar = "█" * int((iou if not np.isnan(iou) else 0) * 30)
        print(f"  {name:<20} {iou:.4f}  {bar}")

    # Save metrics text
    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"TEST RESULTS ({'TTA' if tta else 'No TTA'})\nMean IoU: {test_mean_iou:.4f}\n\n")
        for n, v in zip(CLASS_NAMES, avg_class_iou):
            f.write(f"  {n:<20}: {v:.4f}\n")

    # Save bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    valid   = [v if not np.isnan(v) else 0 for v in avg_class_iou]
    bars    = ax.bar(range(NUM_CLASSES), valid,
                     color=[np.array(PALETTE[i])/255 for i in range(NUM_CLASSES)],
                     edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("IoU Score", fontsize=12)
    ax.set_title(f"Per-Class IoU — UNet-ResNet50 (Mean IoU = {test_mean_iou:.4f})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(test_mean_iou, color="red", ls="--", lw=2, label=f"Mean={test_mean_iou:.4f}")
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_iou.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs saved to:")
    print(f"  {OUTPUT_DIR}/test_metrics.txt")
    print(f"  {OUTPUT_DIR}/per_class_iou.png")
    print(f"  {compare_dir}/  ({saved_comparisons} comparison images)")
    print(f"  {color_dir}/  (colour mask PNGs for all test images)")


if __name__ == "__main__":
    main()
