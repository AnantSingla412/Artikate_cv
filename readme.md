# Artikate CV Assignment — Case & Speaker Detector

## Overview
Two-class object detector (case, speaker) trained on a self-captured dataset.

## Dataset
- Total images: 93
- Classes: 0 = case, 1 = speaker
- Per-class instance counts: (case: 72, speaker: 72)
- Train/val split: 75 / 18 (80/20, random seed=42)
- Split guarantee: split performed on raw captured images before any augmentation;
  each image is a distinct physical capture, so no image or crop of it appears
  in both train and val.

## Annotation
- Tool used: Roboflow
- Format: YOLOv8 (bounding boxes)

## Training Configuration — Run 1
 config for Run 1 (yolov8n, imgsz=640, epochs=100, batch=16, seed=42).

## Validation Results — Run 1 
| Metric       | Run 1 | 
|--------------|-------|
| Precision    | 0.980 |
| Recall       |  0.962|
| mAP@0.5      | 0.990 |
| mAP@0.5:0.95 | 0.858 |

## Known Gaps
- Validation mAP (0.99) is measured on only 19 images — a small sample size

### ONNX Export Verification
Verified ONNX Runtime output against PyTorch on all 18 validation images.

- 16/18 images: detection counts matched, with mean box coordinate difference
  of 1.15% of image dimension and mean confidence difference of 0.0225 —
  consistent with expected floating-point variation between PyTorch and
  ONNX Runtime backends.
- 2/18 images: detection count mismatch (one image found 2 vs 1 detections,
  another found 2 vs 3). Both occurred on borderline-confidence detections
  near the model's confidence threshold — a detection just above threshold
  in one backend can fall just below it in the other due to the same small
  numerical variation, causing it to appear/disappear entirely rather than
  just shift slightly.
- Confirmed via: side-by-side inference on the same images through both
  `best_run2.pt` (PyTorch) and `best_run2.onnx` (ONNX Runtime, CPUExecutionProvider),
  comparing box coordinates (as % of image dimension) and confidence scores.