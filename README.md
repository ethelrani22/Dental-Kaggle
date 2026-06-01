# 🦷 Dental X-Ray AI Pipeline

A two-stage deep learning pipeline for **panoramic dental X-ray analysis** — detecting, segmenting, and classifying tooth anomalies.

---

## 📌 Project Overview

| Stage | Model | Task | Dataset |
|-------|-------|------|---------|
| **Stage 1** | Mask R-CNN (ResNet-50 FPN) | Tooth instance segmentation + quadrant numbering | Humans in the Loop (598 panoramic X-rays) |
| **Stage 2** | ResNet-18 binary classifier | Anomaly detection per tooth (Normal / Anomaly) | DENTEX Challenge 2023 |

At inference time, both stages run **sequentially on the same image**:
1. Stage 1 detects and segments every tooth → outputs bounding boxes + masks
2. Stage 2 takes each tooth crop from Stage 1 → classifies it as Normal or Anomaly

---

## 🗂️ Repository Structure

```
Dental-Kaggle/
├── README.md
├── stage1_segmentation/
│   └── teeth_segmentation.ipynb      # Mask R-CNN training + inference
├── stage2_anomaly/
│   └── anomaly_classifier.ipynb      # ResNet-18 anomaly classifier
└── docs/
    └── pipeline.md                   # How Stage 1 → Stage 2 connects
```

---

## 🔄 How the Two Stages Work Together

```
NEW PANORAMIC X-RAY
        │
        ▼
┌───────────────────────────────┐
│  STAGE 1  — Mask R-CNN        │
│  "Where are the teeth?"       │
│  → Bounding box per tooth     │
│  → Pixel mask per tooth       │
│  → Tooth ID (1–32)            │
│  → Quadrant (UL/UR/LL/LR)     │
└──────────────┬────────────────┘
               │  boxes + masks
               ▼
┌───────────────────────────────┐
│  STAGE 2  — ResNet-18         │
│  "Is this tooth healthy?"     │
│  → Crops each tooth region    │
│  → Binary classification      │
│  → NORMAL / ANOMALY + prob    │
└──────────────┬────────────────┘
               ▼
     FINAL OUTPUT PER TOOTH:
     T11 → 🟢 Normal  (P=0.08)
     T21 → 🔴 Anomaly (P=0.91)
     T22 → 🟢 Normal  (P=0.12)
```

---

## 📦 Datasets

### Stage 1 — Humans in the Loop
- **598** panoramic dental X-ray images
- Polygon annotations for each individual tooth
- 33 classes (32 teeth + background)
- Kaggle: [Teeth Segmentation on Dental X-Ray Images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images)

### Stage 2 — DENTEX Challenge 2023
- Tooth-level crops with disease labels
- Disease classes: Caries, Deep Caries, Periapical Lesion, Impacted Tooth
- Binary label: Normal (0) / Anomaly (1)
- Kaggle: [DENTEX Challenge 2023](https://www.kaggle.com/datasets/humansintheloop/dentex-challenge-2023)

---

## 🧠 Model Architecture

### Stage 1 — Mask R-CNN
- Backbone: ResNet-50 + FPN (pretrained on COCO)
- Custom heads: FastRCNNPredictor + MaskRCNNPredictor
- Classes: 33 (background + 32 teeth)
- Optimizer: Adam, lr=1e-4
- Epochs: 50
- Best checkpoint saved by: lowest validation loss

### Stage 2 — ResNet-18
- Backbone: ResNet-18 (pretrained on ImageNet)
- Custom head: Dropout(0.3) → Linear(512, 1)
- Loss: BCEWithLogitsLoss with pos_weight for class imbalance
- Optimizer: Adam, lr=1e-4, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (patience=3)
- Epochs: 15
- Best checkpoint saved by: highest validation F1

---

## ⚙️ Configuration

### Stage 1
```python
CONFIG = {
    "batch_size":      2,
    "lr":              1e-4,
    "epochs":          50,
    "conf_threshold":  0.6,
    "nms_iou":         0.3,
    "train_ratio":     0.7,
    "val_ratio":       0.15,
    "num_workers":     2,
    "padding":         20,
}
```

### Stage 2
```python
CONFIG_S2 = {
    "batch_size":        32,
    "lr":                1e-4,
    "epochs":            15,
    "img_size":          224,
    "binary_threshold":  0.5,
    "crop_padding":      8,
    "weight_decay":      1e-4,
    "lr_patience":       3,
    "lr_factor":         0.5,
}
```

---

## 📊 Data Split

| Split | Stage 1 | Stage 2 |
|-------|---------|----------|
| Train | 70% (418 images) | 80% |
| Val   | 15% (89 images)  | 10% |
| Test  | 15% (91 images)  | 10% |

---

## 💾 Output Files (Kaggle Working Dir)

| File | Description |
|------|-------------|
| `maskrcnn_teeth_best.pth` | Stage 1 best checkpoint (by val loss) |
| `maskrcnn_teeth_final.pth` | Stage 1 final epoch weights |
| `stage2_anomaly_best.pth` | Stage 2 best checkpoint (by val F1) |
| `stage2_anomaly_final.pth` | Stage 2 final epoch weights |
| `stage2_report.json` | Full training history + test metrics |
| `stage2_curves.png` | Loss + F1 training curves |
| `pipeline_sample_*.png` | Combined Stage 1 + 2 visualizations |

---

## 🚀 Running the Pipeline

Run the notebooks in order within the **same Kaggle session**:

1. Run all cells in `stage1_segmentation/teeth_segmentation.ipynb`
   - Trains Mask R-CNN for 50 epochs
   - `model` and `test_dataset` remain in memory
2. Run all cells in `stage2_anomaly/anomaly_classifier.ipynb`
   - Trains ResNet-18 classifier for 15 epochs
   - Final cells use both `model` (Stage 1) and `stage2_model` (Stage 2) together

---

## 📋 Evaluation Notes

- **Stage 1** is quantitatively evaluated on the HumansInTheLoop held-out test split (segmentation masks available as ground truth)
- **Stage 2** is quantitatively evaluated on the DENTEX held-out test split (disease labels available)
- **Combined pipeline** (Cell 13 of Stage 2) is a qualitative demonstration — HumansInTheLoop test images have no disease labels, so anomaly accuracy cannot be numerically computed at this stage

---

## 🛠️ Environment

- Python 3.12
- PyTorch 2.10.0+cu128
- Torchvision 0.25.0+cu128
- CUDA (Nvidia Tesla T4 on Kaggle)
- NumPy 2.0.2
- OpenCV, Pillow, Matplotlib

---

## 👤 Author

Ethelbert Rani — [GitHub](https://github.com/ethelrani22)
