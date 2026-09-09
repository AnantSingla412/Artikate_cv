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

## Export & Quantization

### Reduced-Precision Format
FP16 chosen over INT8. INT8 requires a calibration dataset and additional
tooling (representative-sample calibration in ONNX Runtime), which was not
justified given the dataset size and available time. FP16 is natively
supported by Ultralytics' export pipeline and by ONNX Runtime, and is a
well-supported target for edge devices such as Jetson.

### Benchmark: FP32 vs FP16

**1. Local CPU (AMD Ryzen 5 5500U, CPUExecutionProvider):**

| Metric            | FP32    | FP16   |
|--------------------------------------|
| Mean latency (ms) | 299.43  | 306.71 |
| P95 latency (ms)  | 347.58  | 367.62 |
| Model size (MB)   | 11.70   | 5.88   |
| mAP@0.5           | 0.9438  | 0.9438 |

**2. Colab CPU (consistency check):** Attempted on Colab's T4 GPU runtime,
but ONNX Runtime's default CPU-only package (`onnxruntime`, not
`onnxruntime-gpu`) was installed, so this run also executed on CPU
(confirmed by a "CUDAExecutionProvider not available" warning). Results
were consistent with the local run:

| Metric            | FP32   | FP16   |
|-------------------------------------|
| Mean latency (ms) | 242.58 | 266.19 |
| P95 latency (ms)  | 284.11 | 319.69 |
| Model size (MB)   | 11.70  | 5.88   |
| mAP@0.5           | 0.9438 | 0.9438 |

**3. Colab GPU (Tesla T4, CUDAExecutionProvider — corrected `onnxruntime-gpu` install):**

| Metric                                | FP32  | FP16  |
|-------------------                    ----------------|
| Mean latency (ms)                     | 9.78  | 9.27  |
| P95 latency (ms)                      | 10.41 | 10.56 |
| FPS                                   | 102.24| 107.91|
| mAP@0.5 (Ultralytics `.val()`)        |not run|not run|
| mAP@0.5 (custom per-class AP script)  | 0.9904 |0.9435|

### Accuracy Discrepancy Between Evaluation Methods

Ultralytics' `.val()` wrapper was unreliable when run directly against
exported ONNX files on GPU in this environment, so a custom per-class AP
script was written (raw ONNX Runtime inference + manual NMS + AP@0.5
calculation) to cross-check accuracy on GPU:


This custom script shows a real accuracy drop specifically on the "case"
class after FP16 quantization (-9.4 points AP), which the CPU benchmarks
above (using Ultralytics' `.val()` wrapper) did not detect — those reported
identical mAP (0.9438) for FP32 and FP16.

Caveat: the custom script used a low confidence threshold (0.001)
before NMS, producing an unusually high raw prediction count (~100+ per
image), which suggests its NMS pass may not exactly replicate Ultralytics'
internal evaluation pipeline. The two methods therefore disagree on
whether FP16 causes any accuracy loss. This was not fully resolved given
time constraints — it is reported here rather than silently picking the
more favorable number, and would need further investigation (e.g.
visualizing raw pre-NMS predictions) before trusting either number for a
production decision.

### Analysis
Accuracy impact of FP16 is inconclusive: CPU runs (via Ultralytics' `.val()`)
showed no accuracy drop, while a custom GPU evaluation showed a real,
class-specific drop on "case" (see discrepancy section above). This
inconsistency itself is a useful finding — it shows that quantization's
accuracy impact can depend on which evaluation pipeline is used, not just
which hardware.

Latency behaved differently by hardware: on both CPU runs, FP16 was
slightly *slower* than FP32, since consumer/server CPUs generally lack
native FP16 arithmetic — ONNX Runtime upcasts FP16 tensors back to FP32
internally, adding conversion overhead with no compute benefit. On GPU
(Tesla T4, which has native FP16 tensor cores), FP16 shows the expected
improvement: ~5% lower mean latency and ~5% higher throughput (102 → 108
FPS) versus FP32.


Model size is hardware-independent: FP16 halves file size (11.70 MB → 5.88
MB) in all cases, useful for storage and transfer regardless of runtime
speed.

---

## A4 — Failure Analysis

Annotated comparison images for all three failure cases (ground truth vs.
prediction overlay) are available at `data/failure_images/` in this repository.

### Image 1: 
- **Predicted:** Only the case is detected (0.67 conf). The speaker is completely
  missed — no detection at all, despite being clearly visible and centered in frame.
- **Expected:** Two objects — speaker (large, centered) and case (bottom-left).
- **Hypothesis:** Two contributing factors. First, the busy, patterned floral
  background behind the speaker differs from the simpler backgrounds used in most
  other training images. Second, and more specifically: most speaker training
  images were captured from a distance, so the grill's perforated texture — a key
  visual cue for this class — appears small and indistinct. The model likely
  learned to associate "speaker" partly with the grill pattern being recognizable
  at a certain scale; in this image, the pattern is present but was captured from
  far enough that it may not have looked similar to what the model saw at training
  scale, especially against the additional visual noise from the patterned
  background.
- **Fix:** Capture more speaker images with close-up, zoomed shots of the grill
  texture specifically, so the model learns to recognize the grill pattern at
  multiple scales — not just the distances most current training images were
  captured at. Also add more speaker images against busier/patterned backgrounds.

### Image 2:
- **Predicted:** Speaker correctly detected (0.97 conf, accurate box). Two spurious
  false-positive "case" detections (0.41, 0.34) on empty background/shadow regions.
- **Expected:** Only the speaker — no case present in this frame.
- **Hypothesis:** Model pattern-matches shadow/plain-surface texture as "case,"
  likely due to insufficient negative/background-only training examples with
  similar tone and low texture.
- **Fix:** Add background-only negative training images; raise deployment
  confidence threshold to ~0.5 to filter these low-confidence false positives.

### Image 3: 
- **Predicted:** Single speaker object split into three overlapping "speaker"
  boxes (0.62, 0.57, 0.27) instead of one clean detection.
- **Expected:** One speaker, filling nearly the entire frame.
- **Hypothesis:** NMS/scale failure on an extreme close-up — different regions of
  the object (top, middle, bottom) activate as separate detections that don't
  overlap enough with each other to be merged by standard NMS thresholds.
- **Fix:** Capture more extreme-close-up training examples so the model learns to
  output a single confident box for full-frame objects; consider tuning NMS IoU
  threshold for this deployment scenario.

---  
## How to Reproduce

Run every from root: python script/[file_name].py

### 1. Environment setup
```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Dataset split
Raw images and Roboflow-exported YOLO labels are already committed under
`data/raw/`. Generate the train/val split:
```bash
python script/split_dataset.py
```
This creates `data/train/` and `data/val/` (75/18 images, seed=42).

### 3. Train the detector
Training was run on Google Colab (Tesla T4 GPU). To reproduce:
```bash
python script/train.py --epochs 100 --batch 16 --imgsz 640 --name run2
```
Weights are saved to `runs/detect/run2/weights/best.pt`. The committed
final model is `model/best .pt`.

### 4. Export to ONNX (FP32)
```bash
python script/export_onnx.py
```

### 5. Verify ONNX output matches PyTorch
```bash
python script/verify_onnx.py
```
Runs inference through both `best.pt` and `best_quantized.onnx` on all
18 validation images and reports box/confidence differences.

### 6. Quantize to FP16
Run on a CUDA-enabled environment (FP16 export requires GPU):
```bash
python script/quantize.py
```

### 7. Benchmark FP32 vs FP16
```bash
python script/benchmark.py
```
Reports mean/p95 latency, file size, and mAP@0.5 for both formats. Run
once on CPU and once on a CUDA-enabled environment (with `onnxruntime-gpu`
installed, not the default `onnxruntime` package) to reproduce both the
CPU and GPU benchmark tables above.

### Notes
- Random seed fixed at 42 throughout (data split, training) for reproducibility.
- Exact numbers may vary slightly (±1-2%) on different hardware; the
  qualitative conclusions (FP16 helps size always, helps latency only on
  GPU) should hold.
