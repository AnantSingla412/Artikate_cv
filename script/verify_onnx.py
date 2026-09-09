# script/verify_onnx.py
import numpy as np
import glob
from ultralytics import YOLO

IMGSZ = 640

img_paths = sorted(glob.glob(r"data\val\images\*.jpg"))
print(f"Testing on {len(img_paths)} validation images\n")

pt_model = YOLO(r"model\best.pt")
onnx_model = YOLO(r"model\best.onnx")

all_box_diffs = []
all_conf_diffs = []
mismatches = 0

for img_path in img_paths:
    pt_results = pt_model.predict(img_path, imgsz=IMGSZ, verbose=False)
    onnx_results = onnx_model.predict(img_path, imgsz=IMGSZ, verbose=False)

    pt_boxes = pt_results[0].boxes.xyxy.cpu().numpy()
    pt_conf = pt_results[0].boxes.conf.cpu().numpy()
    onnx_boxes = onnx_results[0].boxes.xyxy.cpu().numpy()
    onnx_conf = onnx_results[0].boxes.conf.cpu().numpy()

    if len(pt_boxes) == len(onnx_boxes) and len(pt_boxes) > 0:
        h, w = pt_results[0].orig_shape
        box_diff_pct = np.abs(pt_boxes - onnx_boxes) / np.array([w, h, w, h]) * 100
        max_box_diff_pct = box_diff_pct.max()
        max_conf_diff = np.abs(pt_conf - onnx_conf).max()
        all_box_diffs.append(max_box_diff_pct)
        all_conf_diffs.append(max_conf_diff)
        print(f"{img_path.split(chr(92))[-1]}: {len(pt_boxes)} dets, box diff={max_box_diff_pct:.2f}%, conf diff={max_conf_diff:.4f}")
    else:
        mismatches += 1
        print(f"{img_path.split(chr(92))[-1]}: ⚠️ detection count mismatch (PT={len(pt_boxes)}, ONNX={len(onnx_boxes)})")

print(f"\n--- Summary over {len(img_paths)} images ---")
print(f"Detection count mismatches: {mismatches}")
if all_box_diffs:
    print(f"Max box diff (any image): {max(all_box_diffs):.2f}% of image dimension")
    print(f"Mean box diff: {np.mean(all_box_diffs):.2f}%")
    print(f"Max conf diff (any image): {max(all_conf_diffs):.4f}")
    print(f"Mean conf diff: {np.mean(all_conf_diffs):.4f}")