# ================== CELL 1: SETUP ==================

# Optional: fix numpy / OpenCV compatibility if needed
# !pip uninstall -y numpy
# !pip install numpy==1.26.4
# !pip install opencv-python-headless==4.8.1.78

import os
import json
import random

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Numpy version: {np.__version__}")

# ================== DATASET PATHS ==================
BASE_DIR = "/kaggle/input/teeth-segmentation-on-dental-x-ray-images/Teeth Segmentation PNG/d2"
IMG_DIR = "/kaggle/input/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/Teeth Segmentation PNG/d2/img"
JSON_ANNOT_DIR = "/kaggle/input/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/Teeth Segmentation JSON/d2/ann"
COLORMAP_PATH = "/kaggle/input/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/Teeth Segmentation JSON/obj_class_to_machine_color.json"

print(f"✓ Image directory: {IMG_DIR}")
print(f"  Total images: {len(os.listdir(IMG_DIR))}")
print(f"✓ Annotation directory: {JSON_ANNOT_DIR}")
print(f"  Total annotations: {len(os.listdir(JSON_ANNOT_DIR))}")

with open(COLORMAP_PATH) as f:
    COLORMAP = json.load(f)
print(f"✓ Color map loaded with {len(COLORMAP)} tooth classes")

# ================== CONFIG ==================
CONFIG = {
    "batch_size": 2,
    "lr": 1e-4,
    "epochs": 50,
    "conf_threshold": 0.6,   # stricter to reduce false positives
    "nms_iou": 0.3,
    "train_ratio": 0.7,
    "val_ratio": 0.15,       # remaining = test
    "num_workers": 2,
    "padding": 20,
}

# ================== CELL 2: DATASET ==================

class TeethDataset(Dataset):
    """Dataset for teeth instance segmentation from panoramic X-rays"""

    def __init__(self, imgs_dir, ann_dir, file_list=None):
        self.imgs_dir = imgs_dir
        self.ann_dir = ann_dir

        if file_list is None:
            all_files = sorted([
                f for f in os.listdir(imgs_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
        else:
            all_files = file_list

        # Validate image-annotation pairs
        self.img_files = []
        print("Validating image-annotation pairs...")
        for img_file in all_files:
            ann_path = os.path.join(self.ann_dir, img_file + ".json")
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path) as f:
                    annotation = json.load(f)
                if "objects" in annotation and len(annotation["objects"]) > 0:
                    self.img_files.append(img_file)
            except Exception:
                continue

        print(f"✓ Found {len(self.img_files)} valid image-annotation pairs")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.imgs_dir, img_filename)
        ann_path = os.path.join(self.ann_dir, img_filename + ".json")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img, dtype=np.uint8).copy()
        height, width = img_array.shape[:2]

        # Load annotations
        with open(ann_path) as f:
            annotation = json.load(f)

        masks = []
        boxes = []
        labels = []

        for obj in annotation["objects"]:
            class_title = obj.get("classTitle", "")

            # Tooth labels are 1..32
            try:
                tooth_num = int(class_title)
                if not (1 <= tooth_num <= 32):
                    continue
            except ValueError:
                continue

            if "points" not in obj or "exterior" not in obj["points"]:
                continue
            exterior_points = obj["points"]["exterior"]
            if len(exterior_points) < 3:
                continue

            try:
                pts = np.array(exterior_points, dtype=np.int32).copy()
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)

                xs = pts[:, 0]
                ys = pts[:, 1]
                x_min = int(max(0, xs.min()))
                y_min = int(max(0, ys.min()))
                x_max = int(min(width - 1, xs.max()))
                y_max = int(min(height - 1, ys.max()))

                if x_max <= x_min or y_max <= y_min:
                    continue
                if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                    continue

                masks.append(mask.copy())
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                labels.append(tooth_num)
            except Exception:
                continue

        if len(masks) == 0:
            raise ValueError(f"No valid annotations for {img_filename}")

        masks = torch.from_numpy(np.stack(masks).copy()).to(torch.uint8)
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float32).copy())
        labels = torch.from_numpy(np.array(labels, dtype=np.int64).copy())

        target = {
            "masks": masks,
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        img_tensor = torch.from_numpy(img_array.copy()).permute(2, 0, 1).float() / 255.0

        return img_tensor, target

print("✓ TeethDataset class defined")

# ================== CELL 3: SPLIT & LOADERS ==================

all_img_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

random.seed(42)
shuffled_files = all_img_files.copy()
random.shuffle(shuffled_files)

n_total = len(shuffled_files)
n_train = int(n_total * CONFIG["train_ratio"])
n_val = int(n_total * CONFIG["val_ratio"])
n_test = n_total - n_train - n_val

train_files = shuffled_files[:n_train]
val_files = shuffled_files[n_train:n_train + n_val]
test_files = shuffled_files[n_train + n_val:]

print(f"Total images: {n_total}")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

train_dataset = TeethDataset(IMG_DIR, JSON_ANNOT_DIR, file_list=train_files)
val_dataset   = TeethDataset(IMG_DIR, JSON_ANNOT_DIR, file_list=val_files)
test_dataset  = TeethDataset(IMG_DIR, JSON_ANNOT_DIR, file_list=test_files)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=CONFIG["num_workers"],
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CONFIG["num_workers"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CONFIG["num_workers"],
)

print("✓ DataLoaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# ================== CELL 4: MODEL ==================

def get_model_instance_segmentation(num_classes: int):
    """Pretrained Mask R-CNN with custom heads."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

num_classes = 33  # 32 teeth + background
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model_instance_segmentation(num_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
print("✓ Mask R-CNN initialized on", device)

# ================== CELL 5: TRAINING ==================

best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(CONFIG["epochs"]):
    # ---------- TRAIN ----------
    model.train()
    epoch_train_loss = 0.0
    train_batches = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_train_loss += losses.item()
        train_batches += 1

    avg_train_loss = epoch_train_loss / max(1, train_batches)
    train_losses.append(avg_train_loss)

    # ---------- VALIDATION ----------
    # Put model back in training mode to compute losses
    model.train()  # Keep in train mode to get loss_dict
    epoch_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # Returns dict in train mode
            losses = sum(loss for loss in loss_dict.values())
            epoch_val_loss += losses.item()
            val_batches += 1

    avg_val_loss = epoch_val_loss / max(1, val_batches)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch+1:03d}/{CONFIG['epochs']} "
        f"- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # ---------- BEST CHECKPOINT ----------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            },
            "/kaggle/working/maskrcnn_teeth_best.pth",
        )
        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

print("✓ Training complete!")
# Save final epoch model as well (for experiment reproducibility)
torch.save(model.state_dict(), "/kaggle/working/maskrcnn_teeth_final.pth")
print("✓ Final model weights saved: maskrcnn_teeth_final.pth")

# ================== CELL 6: INFERENCE HELPERS ==================

def apply_nms(predictions, iou_threshold=CONFIG["nms_iou"]):
    """Apply Non-Maximum Suppression to reduce duplicate detections."""
    if len(predictions["boxes"]) == 0:
        return predictions

    keep_indices = nms(
        boxes=predictions["boxes"],
        scores=predictions["scores"],
        iou_threshold=iou_threshold,
    )

    return {
        k: v[keep_indices] for k, v in predictions.items()
    }

def run_inference(model, image_tensor, device, confidence_threshold=CONFIG["conf_threshold"]):
    """Run inference on a single image tensor (C,H,W in [0,1])."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        raw_pred = model(image_tensor)[0]

    # Confidence filter
    keep = raw_pred["scores"] >= confidence_threshold
    filtered = {k: v[keep] for k, v in raw_pred.items()}

    # NMS
    filtered = apply_nms(filtered)

    return {
        "boxes": filtered["boxes"].cpu().numpy(),
        "labels": filtered["labels"].cpu().numpy(),
        "masks": filtered["masks"].cpu().numpy(),
        "scores": filtered["scores"].cpu().numpy(),
    }

def crop_to_teeth_region(image, predictions, padding=CONFIG["padding"]):
    """Crop image to the union of all tooth boxes."""
    boxes = predictions["boxes"]
    if len(boxes) == 0:
        return image, None

    x_min = int(boxes[:, 0].min())
    y_min = int(boxes[:, 1].min())
    x_max = int(boxes[:, 2].max())
    y_max = int(boxes[:, 3].max())

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)

    crop_box = (x_min, y_min, x_max, y_max)
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image, crop_box

def color_teeth(image, predictions, crop_box=None):
    """Color each detected tooth with a vibrant palette."""
    if image.ndim == 2:
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        colored_image = image.copy()

    vibrant_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 128, 0], [128, 0, 255], [255, 0, 128],
        [0, 255, 128], [128, 255, 0], [0, 128, 255],
    ]

    for i, (mask, label, box, score) in enumerate(zip(
        predictions["masks"],
        predictions["labels"],
        predictions["boxes"],
        predictions["scores"],
    )):
        mask_binary = (mask[0] > 0.5).astype(np.uint8)

        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            mask_cropped = mask_binary[y1:y2, x1:x2]
        else:
            mask_cropped = mask_binary

        if mask_cropped.shape != colored_image.shape[:2]:
            continue

        color = vibrant_colors[i % len(vibrant_colors)]

        overlay = colored_image.copy()
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_cropped == 1,
                color[c],
                overlay[:, :, c],
            )

        colored_image = cv2.addWeighted(colored_image, 0.3, overlay, 0.7, 0)

        contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_image, contours, -1, color, 3)

    return colored_image

def split_into_quadrants(predictions, image_shape, crop_box=None):
    """Assign each tooth to UL, UR, LL, LR based on centroid."""
    if crop_box:
        x1, y1, x2, y2 = crop_box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    else:
        height, width = image_shape[:2]
        center_x = width // 2
        center_y = height // 2

    quadrants = {"UL": [], "UR": [], "LL": [], "LR": []}

    for i, (box, label, mask, score) in enumerate(zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["masks"],
        predictions["scores"],
    )):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        if cx < center_x and cy < center_y:
            q = "UL"
        elif cx >= center_x and cy < center_y:
            q = "UR"
        elif cx < center_x and cy >= center_y:
            q = "LL"
        else:
            q = "LR"

        quadrants[q].append({
            "index": i,
            "label": int(label),
            "box": box,
            "centroid": (cx, cy),
            "mask": mask,
            "score": float(score),
        })

    return quadrants, (center_x, center_y)

def number_teeth_in_quadrants(quadrants, center):
    """
    Number teeth 1-8 within each quadrant starting from the midline (center)
    and moving outward (FDI-like behavior).
    """
    numbered_quadrants = {}
    center_x, _ = center

    for q_name, teeth in quadrants.items():
        if not teeth:
            numbered_quadrants[q_name] = []
            continue

        # Left side quadrants: UL, LL -> teeth nearer to center have larger x
        if q_name in ["UL", "LL"]:
            sorted_teeth = sorted(teeth, key=lambda t: -t["centroid"][0])
        # Right side quadrants: UR, LR -> nearer to center have smaller x
        else:
            sorted_teeth = sorted(teeth, key=lambda t: t["centroid"][0])

        for num, tooth in enumerate(sorted_teeth, start=1):
            tooth["number"] = num

        numbered_quadrants[q_name] = sorted_teeth

    # Cap at 8 teeth per quadrant using confidence
    for q_name, teeth in numbered_quadrants.items():
        if len(teeth) > 8:
            teeth_sorted = sorted(teeth, key=lambda t: t["score"], reverse=True)
            numbered_quadrants[q_name] = teeth_sorted[:8]

    return numbered_quadrants

# --- ANOMALY MAPPING FUNCTIONS ---

def calculate_ioa(tooth_box, anomaly_box):
    """Calculate Intersection over Anomaly Area (IoA)."""
    tx1, ty1, tx2, ty2 = tooth_box
    ax1, ay1, ax2, ay2 = anomaly_box

    x_left = max(tx1, ax1)
    y_top = max(ty1, ay1)
    x_right = min(tx2, ax2)
    y_bottom = min(ty2, ay2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    anomaly_area = (ax2 - ax1) * (ay2 - ay1)

    if anomaly_area == 0:
        return 0.0

    return intersection_area / anomaly_area

def map_anomalies_to_teeth(anomalies, numbered_quadrants, min_overlap=0.15):
    """Matches a list of detected anomalies to specific teeth based on spatial overlap."""
    clinical_report = []

    for anomaly in anomalies:
        best_match = None
        highest_ioa = 0.0

        for q_name, teeth in numbered_quadrants.items():
            for tooth in teeth:
                ioa = calculate_ioa(tooth["box"], anomaly["box"])

                if ioa > highest_ioa and ioa > min_overlap:
                    highest_ioa = ioa
                    best_match = {
                        "quadrant": q_name,
                        "tooth_number": tooth["number"],
                        "fdi_notation": f"{q_name}-{tooth['number']}",
                        "overlap_percentage": round(ioa * 100, 2)
                    }

        finding = {
            "disease": anomaly["label"],
            "confidence": f"{anomaly['score']*100:.1f}%",
            "location": best_match if best_match else "Unassigned (e.g., gum tissue or interdental)"
        }
        clinical_report.append(finding)

    return clinical_report

# ================== CELL 7: VISUALIZATION & TEST ==================

def visualize_all_tasks(model, dataset, idx, device, confidence=CONFIG["conf_threshold"]):
    image_tensor, target = dataset[idx]
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

    predictions = run_inference(model, image_tensor, device, confidence_threshold=confidence)
    print(f"Detected {len(predictions['labels'])} teeth")
    print(f"Tooth IDs: {predictions['labels']}")

    cropped_img, crop_box = crop_to_teeth_region(image_np.copy(), predictions)
    colored_img = color_teeth(cropped_img.copy(), predictions, crop_box)
    quadrants, center = split_into_quadrants(predictions, image_np.shape, crop_box)
    numbered_quadrants = number_teeth_in_quadrants(quadrants, center)

    # Mock Anomaly Detection & Mapping
    # These represent coordinates where a disease was found
    mock_detected_anomalies = [
        {"box": [150, 300, 180, 330], "label": "Caries", "score": 0.88},
        {"box": [600, 400, 650, 450], "label": "Periapical Lesion", "score": 0.95}
    ]
    anomaly_report = map_anomalies_to_teeth(mock_detected_anomalies, numbered_quadrants)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np, cmap="gray")
    ax1.set_title("Original Panoramic X-ray", fontsize=18, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cropped_img, cmap="gray")
    ax2.set_title("Task 1: Cropped to Teeth Region", fontsize=18, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(colored_img)
    ax3.set_title("Task 2: Each Tooth Colored Differently", fontsize=18, fontweight="bold")
    ax3.axis("off")

    result_img = colored_img.copy()
    h, w = result_img.shape[:2]

    if crop_box:
        cx_crop = center[0] - crop_box[0]
        cy_crop = center[1] - crop_box[1]
    else:
        cx_crop, cy_crop = center

    cv2.line(result_img, (cx_crop, 0), (cx_crop, h), (255, 255, 0), 5)
    cv2.line(result_img, (0, cy_crop), (w, cy_crop), (255, 255, 0), 5)

    for q_name, teeth in numbered_quadrants.items():
        for tooth in teeth:
            cx = tooth["centroid"][0] - (crop_box[0] if crop_box else 0)
            cy = tooth["centroid"][1] - (crop_box[1] if crop_box else 0)
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            num = tooth["number"]
            font_scale = 2.0
            cv2.putText(result_img, str(num), (cx - 25, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 8)
            cv2.putText(result_img, str(num), (cx - 25, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 5)
            cv2.putText(result_img, str(num), (cx - 25, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 3)

    label_font_scale = 2.5
    cv2.putText(result_img, "UL", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), 8)
    cv2.putText(result_img, "UL", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 255), 5)

    cv2.putText(result_img, "UR", (w - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), 8)
    cv2.putText(result_img, "UR", (w - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 255), 5)

    cv2.putText(result_img, "LL", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), 8)
    cv2.putText(result_img, "LL", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 255), 5)

    cv2.putText(result_img, "LR", (w - 120, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), 8)
    cv2.putText(result_img, "LR", (w - 120, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 255), 5)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(result_img)
    ax4.set_title("Task 3 & 4: Quadrants + Numbering (1-8)", fontsize=18, fontweight="bold")
    ax4.axis("off")

    plt.show()

    print("\n" + "=" * 70)
    print("QUADRANT SUMMARY (Teeth numbered from midline outward):")
    print("=" * 70)
    for q_name in ["UL", "UR", "LL", "LR"]:
        teeth = numbered_quadrants[q_name]
        print(f"\n{q_name} Quadrant: {len(teeth)} teeth detected")
        if teeth:
            tooth_list = ", ".join(
                [f"#{t['number']}(ID:{t['label']})" for t in teeth]
            )
            print(f"  {tooth_list}")

    # Print the Anomaly Report
    print("\n" + "=" * 70)
    print("CLINICAL ANOMALY REPORT:")
    print("=" * 70)
    for finding in anomaly_report:
        print(f"Alert: {finding['disease']} ({finding['confidence']})")
        if isinstance(finding['location'], dict):
            print(f"Location: Tooth {finding['location']['fdi_notation']} (Overlap: {finding['location']['overlap_percentage']}%)")
        else:
            print(f"Location: {finding['location']}")
        print("-" * 30)


# Visualize a few samples from TEST set (unseen during training/validation)
num_samples = min(3, len(test_dataset))
for i in range(num_samples):
    print("\n" + "=" * 60)
    print(f"SAMPLE {i+1}/{num_samples}")
    print("=" * 60 + "\n")
    visualize_all_tasks(model, test_dataset, i, device, confidence=CONFIG["conf_threshold"])
