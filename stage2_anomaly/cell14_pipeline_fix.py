# ================== CELL 14 (FIXED): Full Pipeline Visualization ==================
# Uses Stage 1 test images (humansintheloop) — the domain Stage 1 was trained on.
# Stage 1 threshold lowered to 0.3 to handle cross-session inference.
# PASTE THIS ENTIRE CELL to REPLACE the existing Cell 14 in your Kaggle notebook.

import os, glob, random

# ── Stage 1 image directory (same dataset Stage 1 was trained on) ───────────
S1_IMG_DIR = (
    "/kaggle/input/datasets/humansintheloop/"
    "teeth-segmentation-on-dental-x-ray-images/"
    "Teeth Segmentation PNG/d2/img"
)

# ── Lower conf threshold to handle slight domain/session variation ──────────
_S1_CONF   = 0.3   # was 0.6 — Stage 1 scores lower when called cross-session
_S1_NMS    = 0.3

# ── Re-define _s1_infer with explicit model reference & lower threshold ────────
def _s1_infer_fixed(image_tensor):
    """Run Stage 1 Mask R-CNN (uses `model` or `stage1_model` whichever exists)."""
    from torchvision.ops import nms as _nms
    _m = globals().get("stage1_model") or globals().get("model")
    if _m is None:
        raise NameError(
            "Neither 'stage1_model' nor 'model' found in session.\n"
            "Run the Stage 1 model-loading cell first."
        )
    _m.eval()
    with torch.no_grad():
        raw = _m(image_tensor.to(device).unsqueeze(0))[0]
    keep = raw["scores"] >= _S1_CONF
    filt = {k: v[keep] for k, v in raw.items()}
    if len(filt["boxes"]) > 0:
        ki = _nms(filt["boxes"], filt["scores"], _S1_NMS)
        filt = {k: v[ki] for k, v in filt.items()}
    return {k: filt[k].cpu().numpy() for k in ["boxes", "labels", "masks", "scores"]}

# ── Collect Stage 1 test images ──────────────────────────────────────────────────
if os.path.isdir(S1_IMG_DIR):
    _all_s1 = sorted(
        glob.glob(os.path.join(S1_IMG_DIR, "*.png")) +
        glob.glob(os.path.join(S1_IMG_DIR, "*.jpg"))
    )
    random.seed(99)
    _sample_paths = random.sample(_all_s1, min(3, len(_all_s1)))
    print(f"✓ Using {len(_sample_paths)} images from Stage 1 dataset")
else:
    print(f"⚠ Stage 1 img dir not found:\n  {S1_IMG_DIR}")
    print("  Falling back to DENTEX images (detection may be lower)")
    _sample_paths = sorted(
        glob.glob(os.path.join(DENTEX_TRAIN_IMG, "*.png")) +
        glob.glob(os.path.join(DENTEX_TRAIN_IMG, "*.jpg"))
    )[:3]

# ── Patched visualize_full_pipeline ─────────────────────────────────────────────
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HIGHLIGHT   = (255,  50,  50)
_QUAD_LINE   = (255, 230,   0)
_NUM_NORMAL  = (  0, 255, 255)
_NUM_ANOMALY = (255,  50,  50)

def visualize_full_pipeline(img_path, save_prefix=None):
    print(f"  Loading: {os.path.basename(img_path)}")
    img_np  = np.array(Image.open(img_path).convert("RGB"))
    img_t   = TF.to_tensor(Image.fromarray(img_np))

    # ── Stage 1 ────────────────────────────────────────────────────────────────────
    s1 = _s1_infer_fixed(img_t)
    n_teeth = len(s1["boxes"])
    print(f"  Stage 1: {n_teeth} teeth detected  (conf≥{_S1_CONF})")
    if n_teeth == 0:
        print("  ⚠ No teeth detected — try lowering _S1_CONF further.")
        return

    cy_med = float(np.median([(b[1]+b[3])/2 for b in s1["boxes"]]))
    cx_med = float(np.median([(b[0]+b[2])/2 for b in s1["boxes"]]))
    center = (int(cx_med), int(cy_med))

    quads = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for i, box in enumerate(s1["boxes"]):
        tx, ty = (box[0]+box[2])/2, (box[1]+box[3])/2
        q = ("Q1" if tx < center[0] else "Q2") if ty < center[1] \
            else ("Q3" if tx < center[0] else "Q4")
        quads[q].append({"idx": i, "centroid": (int(tx), int(ty)), "box": box,
                         "mask": s1["masks"][i], "score": s1["scores"][i]})
    starts = {"Q1": 11, "Q2": 21, "Q3": 31, "Q4": 41}
    for q, teeth in quads.items():
        teeth_s = sorted(teeth, key=lambda t: t["centroid"][0],
                         reverse=(q in ("Q1","Q3")))
        for rank, tooth in enumerate(teeth_s):
            tooth["number"] = starts[q] + rank
            tooth["quadrant"] = q

    b = s1["boxes"]; pad = 20; H, W = img_np.shape[:2]
    cx1 = max(0, int(b[:,0].min())-pad); cy1 = max(0, int(b[:,1].min())-pad)
    cx2 = min(W, int(b[:,2].max())+pad); cy2 = min(H, int(b[:,3].max())+pad)
    crop_box   = (cx1, cy1, cx2, cy2)
    cropped_np = img_np[cy1:cy2, cx1:cx2]

    palette = [[255,60,60],[60,255,60],[60,60,255],[255,220,0],[220,0,255],
               [0,220,255],[255,140,0],[140,0,255],[255,0,140],[0,255,140],
               [140,255,0],[0,140,255]]
    colored_np = cropped_np.copy()
    for i, (mask, box) in enumerate(zip(s1["masks"], s1["boxes"])):
        mb = (mask[0] > 0.5).astype(np.uint8)
        mb_c = mb[cy1:cy2, cx1:cx2]
        if mb_c.shape != colored_np.shape[:2]: continue
        color = palette[i % len(palette)]
        ov = colored_np.copy()
        for c in range(3): ov[:,:,c] = np.where(mb_c==1, color[c], ov[:,:,c])
        colored_np = cv2.addWeighted(colored_np, 0.3, ov, 0.7, 0)
        cnts, _ = cv2.findContours(mb_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_np, cnts, -1, color, 2)

    # ── Stage 2 ────────────────────────────────────────────────────────────────────
    anomaly_map = {}
    for q, teeth in quads.items():
        for tooth in teeth:
            bx1,by1,bx2,by2 = tooth["box"]; cpd = 10
            crop_t = img_np[max(0,int(by1)-cpd):min(H,int(by2)+cpd),
                            max(0,int(bx1)-cpd):min(W,int(bx2)+cpd)]
            if crop_t.shape[0]<32 or crop_t.shape[1]<32: continue
            t2 = TF.to_tensor(Image.fromarray(crop_t)).to(device)
            stage2_model.eval()
            with torch.no_grad():
                pred2 = stage2_model([t2])[0]
            for b2, lbl, sc in zip(pred2["boxes"].cpu().numpy(),
                                    pred2["labels"].cpu().numpy(),
                                    pred2["scores"].cpu().numpy()):
                if sc < CONFIG2["conf_threshold"]: continue
                tnum = tooth["number"]
                if tnum not in anomaly_map:
                    anomaly_map[tnum] = {"quadrant": q, "tooth": tooth, "anomalies": []}
                anomaly_map[tnum]["anomalies"].append({
                    "label_name": ANOMALY_CLASSES.get(int(lbl), "Unknown"),
                    "score": float(sc)
                })

    print(f"  Stage 2: {len(anomaly_map)} anomalous teeth")
    for tnum, info in anomaly_map.items():
        for a in info["anomalies"]:
            print(f"    Tooth {tnum} ({info['quadrant']}): {a['label_name']}  conf={a['score']:.2f}")

    # ── Annotated result image ──────────────────────────────────────────────────────
    result_img = colored_np.copy()
    rh, rw = result_img.shape[:2]
    ox, oy = cx1, cy1
    ccx, ccy = center[0]-ox, center[1]-oy
    cv2.line(result_img, (ccx, 0),   (ccx, rh), _QUAD_LINE, 4)
    cv2.line(result_img, (0, ccy),   (rw,  ccy), _QUAD_LINE, 4)
    for ql, pos in {"Q1":(10,40),"Q2":(rw-80,40),
                     "Q3":(10,rh-20),"Q4":(rw-80,rh-20)}.items():
        cv2.putText(result_img, ql, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 5)
        cv2.putText(result_img, ql, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2)
    for q, teeth in quads.items():
        for tooth in teeth:
            tx, ty = tooth["centroid"][0]-ox, tooth["centroid"][1]-oy
            if not (0<=tx<rw and 0<=ty<rh): continue
            num  = tooth["number"]; is_a = num in anomaly_map
            if is_a:
                bx1r=max(0,  int(tooth["box"][0])-ox-5)
                by1r=max(0,  int(tooth["box"][1])-oy-5)
                bx2r=min(rw, int(tooth["box"][2])-ox+5)
                by2r=min(rh, int(tooth["box"][3])-oy+5)
                cv2.rectangle(result_img,(bx1r,by1r),(bx2r,by2r),_HIGHLIGHT,6)
                cv2.rectangle(result_img,(bx1r+4,by1r+4),(bx2r-4,by2r-4),(255,150,0),2)
            nc = _NUM_ANOMALY if is_a else _NUM_NORMAL
            cv2.putText(result_img, str(num), (tx-20,ty+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
            cv2.putText(result_img, str(num), (tx-20,ty+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, nc, 2)

    lines = ["ANOMALY REPORT", "─"*32]
    if anomaly_map:
        for tnum in sorted(anomaly_map):
            info = anomaly_map[tnum]
            for a in info["anomalies"]:
                lines.append(f"  Tooth {tnum:>2}  ({info['quadrant']}):  "
                              f"{a['label_name']}  [{a['score']*100:.0f}%]")
    else:
        lines.append("  No anomalies detected.")

    fig = plt.figure(figsize=(22, 18), facecolor="#1a1a2e")
    gs  = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08,
                            left=0.02, right=0.98, top=0.93, bottom=0.08)
    tkw = dict(fontsize=14, fontweight="bold", color="white",
               fontfamily="monospace", pad=8)
    ax1=fig.add_subplot(gs[0,0]); ax1.imshow(img_np,cmap="gray")
    ax1.set_title("Original Panoramic X-ray",**tkw);              ax1.axis("off")
    ax2=fig.add_subplot(gs[0,1]); ax2.imshow(cropped_np,cmap="gray")
    ax2.set_title("Cropped to Teeth Region  (Stage 1)",**tkw);    ax2.axis("off")
    ax3=fig.add_subplot(gs[1,0]); ax3.imshow(colored_np)
    ax3.set_title("Coloured Segmentation  (Stage 1)",**tkw);      ax3.axis("off")
    ax4=fig.add_subplot(gs[1,1]); ax4.imshow(result_img)
    ax4.set_title("Quadrants + Numbers + Anomaly Highlights  (Stage 1+2)",**tkw)
    ax4.axis("off")
    fig.suptitle("Dental X-ray Analysis Pipeline",
                 fontsize=20, fontweight="bold", color="white",
                 fontfamily="monospace", y=0.97)
    fig.text(0.02, 0.01, "\n".join(lines), fontsize=10, color="#f0f0f0",
             fontfamily="monospace", va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d0d1a",
                       edgecolor="#ff3232", linewidth=1.5))
    nn = np.array(_NUM_NORMAL)/255.; na = np.array(_HIGHLIGHT)/255.
    fig.legend(
        handles=[mpatches.Patch(color=nn, label="Normal tooth"),
                 mpatches.Patch(color=na, label="Anomalous tooth")],
        loc="lower right", fontsize=11,
        facecolor="#1a1a2e", edgecolor="gray", labelcolor="white", framealpha=0.9
    )
    plt.tight_layout()
    if save_prefix:
        out = f"{save_prefix}_pipeline.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✓ Saved → {out}")
    plt.show()
    return fig, anomaly_map

# ── Run ──────────────────────────────────────────────────────────────────────────────────
print("="*60)
print("  Running full pipeline...")
print("="*60)
for _i, _img_path in enumerate(_sample_paths):
    if not os.path.exists(_img_path):
        print(f"  ⚠ Not found: {_img_path} — skipping"); continue
    print(f"\n{'='*60}\nSAMPLE {_i+1}/{len(_sample_paths)}: "
          f"{os.path.basename(_img_path)}\n{'='*60}")
    try:
        visualize_full_pipeline(_img_path,
                                save_prefix=f"/kaggle/working/pipeline_sample{_i+1}")
    except Exception as _e:
        import traceback; traceback.print_exc()
    plt.close("all")
print("\n✓ CELL 14 COMPLETE — Full pipeline visualization done.")
