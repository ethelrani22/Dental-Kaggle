# Pipeline: How Stage 1 → Stage 2 Connect

## Overview

Both stages run in a single Kaggle notebook session. Stage 1 trains first, then Stage 2 trains using its own dataset. At the final cells, both trained models are used together on the same test images.

---

## Memory Lifecycle

| Variable | Created in | Used in |
|----------|------------|----------|
| `model` | Stage 1 Cell 4 | Stage 2 Cell 13 (final viz) |
| `test_dataset` | Stage 1 Cell 3 | Stage 2 Cell 13 (sample images) |
| `run_inference()` | Stage 1 Cell 6 | Stage 2 Cell 16 (bridge) |
| `stage2_model` | Stage 2 Cell 6 | Stage 2 Cell 13 (final viz) |
| `run_stage2_on_stage1_output()` | Stage 2 Cell 11 | Stage 2 Cell 13 (bridge) |

---

## The Bridge Function

`run_stage2_on_stage1_output()` in Stage 2 Cell 11 is the exact connector:

```python
def run_stage2_on_stage1_output(image_np, stage1_predictions,
                                stage2_model, device, threshold=0.5):
    results = []
    for box, tooth_id, score in zip(
        stage1_predictions["boxes"],   # ← from Stage 1
        stage1_predictions["labels"],  # ← from Stage 1
        stage1_predictions["scores"],  # ← from Stage 1
    ):
        crop = crop_tooth_from_box(image_np, box)    # crop using Stage 1 box
        pred = predict_tooth(crop, stage2_model, ...) # classify with Stage 2
        results.append({                              # merged result
            "tooth_id":            tooth_id,
            "box":                 box,
            "segmentation_score":  score,
            "anomaly_label":       pred["label"],
            "anomaly_probability": pred["probability_anomaly"],
        })
    return results
```

---

## Inference Flow at Cell 13

```python
# Both models already in memory from training
image_tensor, _ = test_dataset[i]          # Stage 1 test image

# Step 1 — Stage 1: detect + segment teeth
predictions = run_inference(model, image_tensor, device)
# predictions = { boxes, labels, masks, scores }

# Step 2 — Stage 2: classify each tooth crop
results = run_stage2_on_stage1_output(
    image_np, predictions, stage2_model, device
)
# results = [{ tooth_id, box, anomaly_label, anomaly_probability }, ...]

# Step 3 — Combined visualization
visualize_pipeline(image_tensor, model, stage2_model, device, sample_index=i)
```

---

## Dataset Separation (Why Two Datasets?)

No single public dataset provides both:
- Full polygon tooth segmentation masks (needed for Stage 1)
- Per-tooth disease labels (needed for Stage 2)

So we use:
- **HumansInTheLoop** → provides polygon masks → trains Stage 1
- **DENTEX** → provides disease labels on crops → trains Stage 2

They complement each other in the pipeline.

---

## Evaluation Scope

```
Stage 1 test eval  → quantitative (HumansInTheLoop test split has ground truth masks)
Stage 2 test eval  → quantitative (DENTEX test split has disease labels)
Combined Cell 13   → qualitative only (HumansInTheLoop has no disease labels)
```

For a fully quantitative end-to-end evaluation, a dataset with both segmentation masks AND disease labels on the same panoramic images would be needed.
