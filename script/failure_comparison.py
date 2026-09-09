# script/failure_comparison.py
"""
Draws ground-truth boxes (green) and model predictions (red) on the same
image, so failure modes are immediately visible without cross-referencing
label files manually.

Run from repo root: python script/failure_comparison.py
Output saved to: benchmarks/failure_analysis/
"""

import cv2
import glob
import os
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "model/best.pt"
VAL_IMAGES = "data/val/images"
VAL_LABELS = "data/val/labels"
OUT_DIR = "benchmarks/failure_analysis"
IMGSZ = 640
CONF_THRES = 0.25

CLASS_NAMES = {0: "case", 1: "speaker"}
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
BOX_THICKNESS = 6
FONT_SCALE = 1.2
FONT_THICKNESS = 3

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls = int(cls)
            xc, yc, w, h = [float(v) for v in (xc, yc, w, h)]
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def draw_label(img, text, x, y, color, img_w, img_h, above=True):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

    y_text = y - 10 if above else y + th + 15

    # flip if it would run off the top or bottom edge
    if y_text - th - 8 < 0:
        y_text = y + th + 15
    if y_text + 6 > img_h:
        y_text = y - 10

    # clamp fully on-screen as a last resort
    y_text = max(th + 8, min(y_text, img_h - 8))
    x_text = max(0, min(x, img_w - tw - 6))

    cv2.rectangle(img, (int(x_text), int(y_text - th - 8)), (int(x_text) + tw + 6, int(y_text) + 6), color, -1)
    cv2.putText(img, text, (int(x_text) + 3, int(y_text)), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

img_paths = sorted(glob.glob(f"{VAL_IMAGES}/*.jpg"))
print(f"Processing {len(img_paths)} validation images...\n")

for img_path in img_paths:
    img_name = Path(img_path).stem
    label_path = f"{VAL_LABELS}/{img_name}.txt"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gt_boxes = load_gt_boxes(label_path, w, h)
    for i, (cls, x1, y1, x2, y2) in enumerate(gt_boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, BOX_THICKNESS)
        draw_label(img, f"GT #{i+1}: {CLASS_NAMES.get(cls, cls)}", x1, y1, GT_COLOR, w, h, above=True)

    results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
    pred_boxes = results[0].boxes
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), PRED_COLOR, BOX_THICKNESS)
        draw_label(img, f"Pred #{i+1}: {CLASS_NAMES.get(cls, cls)} ({conf:.2f})", x1, y2, PRED_COLOR, w, h, above=False)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    flag = "  <-- MISMATCH" if n_gt != n_pred else ""
    print(f"{img_name}: GT={n_gt}, Pred={n_pred}{flag}")
    cv2.imwrite(f"{OUT_DIR}/{img_name}_compare.jpg", img)

print(f"\nSaved to {OUT_DIR}/")