# Duality AI – Offroad Semantic Scene Segmentation
## Team Submission · February 2026

---

## Overview

This repository contains training and inference code for semantic segmentation of synthetic desert offroad scenes. The model classifies each pixel into one of **10 classes**: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.

**Architecture:** UNet with ResNet50 encoder (pretrained on ImageNet)  
**Loss:** CrossEntropyLoss (class-weighted) + TverskyLoss  
**Final Test Mean IoU:** 0.3191 (with TTA)  
**Best Val IoU:** 0.5751

---

## Repository Structure

```
duality_submission/
├── train.py           # Training script
├── test.py            # Inference + evaluation script
├── config.py          # All hyperparameters and paths
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── outputs/           # Generated PNGs and metrics (after running)
│   ├── training_curves.png
│   ├── per_class_iou.png
│   ├── augmentation_preview.png
│   ├── test_metrics.txt
│   └── training_history.json
├── predictions/       # Test set predictions (after running test.py)
│   ├── comparisons/   # Side-by-side input/GT/prediction images
│   └── masks_color/   # Colour-coded mask PNGs for all test images
└── checkpoints/       # Epoch checkpoints (auto-saved during training)
```

---

## Environment Setup

### Option 1 — pip (local)
```bash
pip install -r requirements.txt
```

### Option 2 — conda
```bash
conda create -n duality python=3.10
conda activate duality
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install segmentation-models-pytorch albumentations tqdm matplotlib
```

### Option 3 — Kaggle Notebook
```python
!pip install -q segmentation-models-pytorch albumentations
```
*(torch, torchvision, cv2, tqdm, matplotlib are pre-installed)*

---

## Dataset Setup

The dataset should be organised as follows:
```
data/
├── train/
│   ├── Color_Images/    # RGB images (.png)
│   └── Segmentation/    # Mask images (.png)
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
    ├── Color_Images/
    └── Segmentation/
```

**On Kaggle:** The scripts auto-detect the Kaggle input path  
`/kaggle/input/duality-offroad-segmentation/data/`  
No manual path changes needed.

**Locally:** Place the dataset at `./data/` or pass `--data /path/to/data`.

---

## Training

```bash
# Standard training (30 epochs, cosine LR)
python train.py

# Custom hyperparameters
python train.py --epochs 50 --lr 3e-5 --batch 8

# Resume from latest checkpoint (if previous run was interrupted)
python train.py --resume

# Custom data path
python train.py --data /path/to/dataset
```

**Training outputs saved to `outputs/`:**
- `training_curves.png` — Loss, IoU, Accuracy, LR plots
- `training_history.json` — Full epoch-by-epoch numbers
- `best_model.pth` — Weights at highest Val IoU
- `final_model.pth` — Weights at last epoch
- `checkpoints/ckpt_epoch_NNN.pth` — Periodic saves every 5 epochs

---

## Inference / Testing

```bash
# Run with TTA (default — horizontal flip average)
python test.py

# Run without TTA
python test.py --no-tta

# Use a specific model checkpoint
python test.py --model checkpoints/ckpt_epoch_025.pth

# Also save raw class-ID masks
python test.py --save-masks

# Save more comparison images (default 8)
python test.py --comparisons 16
```

**Inference outputs saved to `outputs/` and `predictions/`:**
- `outputs/test_metrics.txt` — Mean IoU + per-class breakdown
- `outputs/per_class_iou.png` — Bar chart
- `predictions/comparisons/` — Input / Ground Truth / Prediction side-by-side
- `predictions/masks_color/` — Colour-coded mask for every test image

---

## Reproducing Final Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (or skip if best_model.pth already exists)
python train.py

# 3. Evaluate on test set
python test.py

# Expected: Test Mean IoU ≈ 0.32 (with TTA)
```

---

## Expected Outputs

| Class           | IoU    |
|-----------------|--------|
| Background      | 0.000  |
| Trees           | 0.436  |
| Lush Bushes     | 0.000  |
| Dry Grass       | 0.469  |
| Dry Bushes      | 0.361  |
| Ground Clutter  | 0.000  |
| Logs            | 0.000  |
| Rocks           | 0.047  |
| Landscape       | 0.641  |
| Sky             | 0.981  |
| **Mean IoU**    | **0.319** |

---

## Model Details

| Parameter        | Value                        |
|------------------|------------------------------|
| Architecture     | UNet                         |
| Encoder          | ResNet50 (ImageNet pretrained)|
| Input resolution | 768 × 768                    |
| Loss             | CE (weighted) + Tversky      |
| Optimizer        | AdamW (lr=5e-5, wd=1e-4)    |
| Scheduler        | CosineAnnealingLR            |
| Epochs           | 30                           |
| Batch size       | 4                            |
| TTA              | Horizontal flip average      |

---

## Collaborators

GitHub usernames for repository access:
- `Maazsyedm`
- `rebekah-bogdanoff`
- `egold010`
