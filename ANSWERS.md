# Part B — Answers (ANSWERS.md)

## Snippet 1 — Preprocessing and Coordinate Mapping

### Approach
`preprocess` does two things to place the image on the model's input canvas: **resize** by ratio `r`, then **pad** by `(dw, dh)` to center it in a square. Reversing a transformation means undoing steps in reverse order — pad first, then resize. Reading `postprocess`, it only reverses the resize (divides by `r`) and never subtracts the pad offset. That mismatch is the defect. The symptom ("off, worse at edges") confirms it: a leftover positional offset, not a scaling error.

### Defects
1. `postprocess` never subtracts `(dw, dh)` before dividing by `r`. Boxes are returned still carrying the pad offset.

### Why it survives casual testing
On a **square image**, `h == w`, so `nh == nw == size` and `dh == dw == 0`. Subtracting zero changes nothing, so the buggy and correct code produce identical output — the bug is invisible unless tested on non-square images.

### Why the offset is systematic, and grows toward edges
It's a fixed offset (`dw`/`dh`), not noise — same input, same error every time, hence systematic. Dividing that fixed pixel offset by `r` (`r < 1` when downscaling) amplifies it in original-image pixels. Boxes near the frame edge are already close to the valid-content boundary, so the same absolute error pushes them further out or into clipping — making the effect visibly worse at edges even though the underlying bug is applied uniformly.

### Corrected code
```python
def preprocess(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[None], r, (dw, dh)

def postprocess(boxes, r, pad, orig_shape):
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes
```

### Test that would have caught it
Run inference on a **non-square image** with a known ground-truth box, and assert the recovered box matches within 1-2px. A square-only test suite can never catch this.

---

## Snippet 2 — Dataset Preparation

### Approach
Read the code as: capture → augment → shuffle → split. The key question for any augmentation pipeline is *when* the split happens relative to augmentation. Here, augmented copies of the same source image are generated first, then shuffled, then split — meaning two augmented variants of the *same underlying image* can land on opposite sides of the split. That's a data-leakage bug: the model would effectively "see" val images during training via their flipped/brightened twins, inflating validation metrics without corresponding real generalization.

### Defects
1. **Split leakage via augmentation-before-split.** All three variants (original, flipped, brightened) of a single image are appended to the same list, then shuffled together *before* splitting. A flipped copy of an image can end up in train while its original ends up in val (or vice versa) — the model has effectively already seen that scene.
2. **Horizontal flip corrupts labels for non-symmetric classes/scenes without adjusting bounding box coordinates.** `cv2.flip(img, 1)` mirrors the image, but the code reuses the same `labels` unmodified (`augmented.append((cv2.flip(img, 1), labels))`). Bounding box x-coordinates are not mirrored to match the flipped image — every flipped image now has boxes pointing to the wrong location. Since this generates exactly one flipped copy per original, this corrupts **exactly 1/3 of the total dataset** (one of the three variants per source image).
3. **No shuffling/seeding control for reproducibility**, and no `random.seed()` set — reruns produce a different split each time, making results non-reproducible (a A1 anti-pattern too, but present here structurally).

### Which defect does the most damage, and why
The **unmodified labels on the flipped image (#2)** does the most damage to the reported metric, specifically. Here's why it's worse than the leakage bug: leakage (#1) inflates the *validation* metric optimistically but the model is still learning from *correctly labeled* data — it generalizes poorly to truly new data, but doesn't learn wrong associations. The flip-label bug, by contrast, actively **teaches the model incorrect object locations** for a third of all training instances — it's not just an evaluation artifact, it corrupts the learned weights themselves. That's why the symptom described ("healthy val mAP collapses on real data") fits both, but the flip bug is the deeper root cause: even a *correctly split* dataset with this bug would still fail in production, because the model has partially learned wrong spatial associations.

### Corrected code
```python
images = sorted(glob.glob("dataset/images/*.jpg"))
random.seed(42)

# Split BEFORE augmenting
random.shuffle(images)
split = int(0.8 * len(images))
train_images, val_images = images[:split], images[split:]

def flip_labels_horizontal(labels, img_width):
    flipped = []
    for cls, x_center, y_center, w, h in labels:
        flipped.append((cls, img_width - x_center, y_center, w, h))
    return flipped

def build_augmented(image_paths):
    augmented = []
    for path in image_paths:
        img = cv2.imread(path)
        labels = load_labels(path)
        w = img.shape[1]
        augmented.append((img, labels))
        augmented.append((cv2.flip(img, 1), flip_labels_horizontal(labels, w)))
        augmented.append((adjust_brightness(img, 1.3), labels))  # brightness-only is label-safe
    return augmented

train = build_augmented(train_images)
val = build_augmented(val_images)  # or leave val unaugmented entirely — safer for honest eval

print("train:", len(train), "val:", len(val))
```
Note: augmenting the validation set at all is questionable — ideally val stays as pure, unaugmented captures, since augmentation should only expand training diversity, not the evaluation set.

### Test that would have caught it
1. **Leakage check:** after splitting, assert no two entries in `train` and `val` share the same source image filename/hash.
2. **Label-correctness check for flips:** visually overlay the box on the flipped image for a handful of samples — a box drawn on the wrong side of a flipped image is immediately visually obvious, and should be part of any augmentation pipeline's sanity check before training.

---

## Snippet 3 — IoU and Non-Maximum Suppression

### Approach
Trace what happens to a single detection through `nms`. Boxes are `xyxy` format (`x1,y1,x2,y2`), but `area1 = box[2] * box[3]` computes area as `x2 * y2` — the raw second corner coordinates multiplied together, not `(x2-x1) * (y2-y1)`. This is a real, immediate bug in the IoU calculation itself, independent of the NMS logic. Because `iou()` is what filters detections in `nms()`, a wrong IoU value directly causes wrong suppression decisions.

### Defects
1. **Area calculated as `x2 * y2` instead of `(x2-x1) * (y2-y1)`.** For `xyxy` boxes, this is simply wrong — it computes something numerically unrelated to true box area (e.g., for a box far from the origin, this can produce a huge "area" regardless of the box's actual size).
2. **No `x2 > x1` / `y2 > y1` validity clip on the intersection** (`inter = (x2-x1)*(y2-y1)` could go negative for non-overlapping boxes, silently producing a negative or nonsensical IoU instead of clamping to zero).
3. **The `classes` parameter is accepted by `nms()` but never used.** NMS as written suppresses boxes across *all* classes together — a "case" box and a "speaker" box that happen to overlap spatially will suppress each other, even though they're different objects.

### The mechanism that makes a valid detection disappear
Because class information is ignored, `nms` treats every detection — regardless of class — as competing for the same physical space. If a "case" box and a "speaker" box genuinely overlap (two different real objects sitting close together, or even the wrong-area IoU miscalculating overlap for boxes that don't really overlap much), the lower-confidence one gets suppressed as if it were a duplicate detection of the *same* object, even though it's a legitimately different object. The result: a real, correctly-detected object silently disappears from the output with no error — it looks like an ordinary NMS suppression, not a bug.

### Why this happens more in sparse frames than crowded ones
This seems counterintuitive at first — you'd expect crowded frames to have more suppression issues. But the key is what "sparse" means here: in a sparse frame with only one instance of each class, that pair is exactly the *most likely* scenario for the class-agnostic bug to bite, because there's nothing else competing for attention — the model's second-best box (correctly a different class) is almost certainly the very case+speaker pair sitting close together, and it gets fully wiped by class-agnostic NMS. In crowded frames, there are many boxes of the *same* class overlapping each other — which is exactly what NMS is *supposed* to suppress — so the bug's effect is masked by correct behavior happening at the same time. In sparse frames, there's no "correct" suppression happening to hide behind — the erroneous cross-class suppression is the dominant, visible effect.

### What `classes` should have been doing
It should partition detections by class before applying IoU comparisons — either by looping NMS per class independently, or by adding a large offset per class to box coordinates before computing IoU (a common trick, "class-agnostic vs class-aware" NMS), so boxes of different classes never suppress each other regardless of spatial overlap.

### Corrected code
```python
def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)

def nms(boxes, scores, classes, thr=0.5):
    keep = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        order = cls_scores.argsort()[::-1]
        cls_keep = []
        while order.size > 0:
            i = order[0]
            cls_keep.append(i)
            ious = iou(cls_boxes[i], cls_boxes[order[1:]])
            order = order[1:][ious < thr]
        keep.extend(cls_indices[cls_keep])
    return keep
```

### Test that would have caught it
1. **Area unit test:** call `iou()` directly on two known boxes with a hand-calculated expected IoU (e.g., two identical boxes should give IoU=1.0; the buggy version would fail this trivially).
2. **Class-isolation test:** construct a synthetic case with two overlapping boxes of *different* classes and identical high confidence; assert both survive NMS. The current code would incorrectly drop one.

---
# Part C — Answers (ANSWERS.md)

---

## C1. Accuracy Collapses After Quantisation

### Approach
The symptom is a large, sudden accuracy drop tied specifically to the ONNX export + INT8 step — architecture and data are ruled out by the prompt itself. So the investigation narrows to: what changes between a PyTorch FP32 forward pass and an INT8 ONNX Runtime forward pass? Three independent things change simultaneously in that pipeline: the **numeric precision**, the **export/graph conversion**, and the **calibration process for INT8**. Each can independently cause this exact symptom, so they need to be isolated one at a time, not assumed together.

### Root cause 1: Poor/unrepresentative INT8 calibration data
INT8 quantization maps a continuous float range to 256 discrete levels using a calibration dataset to determine that range per-layer. If the calibration set is too small, doesn't represent the deployment distribution (e.g., calibrated on bright indoor images but deployed on the defect line's actual lighting), or uses too few samples, activation ranges get clipped or under-scaled, destroying precision exactly where it matters.

**Distinguishing test:** Re-run the same INT8 export but with the *validation set itself* as calibration data (not swapping it into training — purely as a calibration sanity check). If mAP recovers close to FP32, the original calibration set was the problem, not INT8 quantization itself.

### Root cause 2: A quantization-sensitive layer (e.g., detection head) losing critical precision
Detection heads often involve small-magnitude values (objectness/class confidence logits, box regression deltas) that are especially sensitive to INT8's reduced dynamic range. If the entire model — including the head — was quantized uniformly, small output-layer errors can flip which boxes clear the confidence threshold entirely, causing a disproportionate mAP hit even if backbone features are barely affected.

**Distinguishing test:** Run "mixed precision" quantization — keep the detection head (last 1-2 layers) in FP16/FP32 and quantize only the backbone to INT8. If mAP recovers substantially with the head excluded, the head's precision sensitivity was the dominant cause, not calibration or export correctness broadly.

### Root cause 3: A bug in the ONNX export/graph conversion itself (independent of quantization)
Export can silently produce a graph that behaves differently from the original model — e.g., incorrect handling of a custom op, a Q/DQ (quantize-dequantize) node inserted at the wrong point in the graph, or an activation function approximated differently in the exported ops. This is a correctness bug, not a precision-tradeoff — quantization is just where it happened to surface.

**Distinguishing test:** Export to ONNX **without** quantization (stay FP32 in ONNX) and evaluate on the validation set through ONNX Runtime. If FP32 ONNX itself is already noticeably below 0.91 (not just a small backend-variation gap), the bug is in the export/graph conversion, independent of INT8 — confirming this is not a quantization-specific issue at all.

### Fix and validation
Run all three tests above in sequence (cheapest first: FP32-ONNX-only test, then head-precision test, then calibration-data test) to isolate which factor(s) apply — they aren't mutually exclusive. Once isolated, the fix is targeted: better/larger calibration set, mixed-precision export keeping sensitive layers in FP16, or fixing the specific export bug. Before shipping to the client, validate by re-running the full validation set through the corrected pipeline and confirming mAP is within an agreed tolerance of the FP32 baseline (e.g., within 2-3 points, not 33 points) — and separately spot-check a handful of real Jetson Orin captures, not just the lab validation set, since deployment lighting/camera characteristics may differ from what was used for calibration.

---

## C2. One Camera Out of Twelve Is Wrong

### Approach
The critical clue is "consistently offset in the same direction, worse at the edges" — this is the exact signature of a **letterbox/coordinate-mapping bug** (structurally identical to Part B, Snippet 1), not a random model failure. If it were a model or weights issue, you'd expect roughly similar failure patterns across all cameras (same model, same weights, same code) — or random/inconsistent errors, not a clean directional, edge-amplified bias isolated to exactly one feed.

### What the pattern alone tells you
Since 11/12 feeds run the identical inference code and model with no issue, the shared code is not the root cause — a systematic offset unique to one input source almost always means that camera's **feed characteristics differ from the others**: most likely a different resolution, aspect ratio, or frame geometry than the other 11 feeds. If that camera's stream has a different native aspect ratio (e.g., 4:3 instead of 16:9, or a different resolution entirely), the shared preprocessing/postprocessing code — tuned/tested against the other 11 cameras' common aspect ratio — would produce exactly this pattern: a fixed offset from incomplete letterbox reversal, amplified at the edges. This directly parallels the square-vs-non-square blind spot from Snippet 1.

### How to confirm without physical camera access
1. **Pull the raw frame dimensions being received from that RTSP stream in code/logs** (width × height at the point preprocessing receives it) and compare against the other 11 cameras' dimensions/aspect ratios. A mismatch here is strong confirming evidence.
2. **Log the computed `r`, `dw`, `dh` (or equivalent letterbox parameters) per camera** for a batch of frames — if camera 12 consistently produces non-zero `dw`/`dh` while the other 11 produce zero (or a different padding profile), that confirms the aspect-ratio-driven letterbox path is the one being exercised only on camera 12.
3. **Replay a saved frame from camera 12 through the pipeline locally** (no live camera access needed, just a captured frame) and manually verify the box offset against a hand-measured ground truth — reproducing the bug offline confirms it's software, not a hardware/lensing issue specific to that physical camera.

### Fix and validation
Once confirmed as the aspect-ratio/letterbox path, the fix is likely the same missing-offset-subtraction bug as Snippet 1, or a hardcoded assumption elsewhere in the pipeline about input resolution. Validate by re-running the corrected pipeline against saved frames from camera 12 and confirming recovered boxes match manual ground-truth measurements within the same tolerance used for the other 11 cameras, across several frames spanning near-center and near-edge object positions specifically (since edge cases are where this bug is most visible).

---

## C3. Silent Degradation Over Three Months

### Approach
"No code/model changes, nothing reported altered" rules out an obvious single-event cause — this points toward a **gradual environmental drift** that nobody was monitoring for, since the system had no signal that would have surfaced a slow decline. The investigation should separate causes into: physical/environmental drift, data distribution drift, and silent infrastructure change.

### Cause 1: Camera/lens degradation or drift (dust, focus drift, vibration-induced angle shift)
On a conveyor-belt line, vibration and dust accumulation on lens/housing over months is common and easy to miss visually. This would manifest as gradually blurrier or slightly repositioned frames, which a model trained on sharp/well-aligned frames would handle progressively worse.

**Confirming evidence:** Pull a sample of frames from month 1 vs. month 3 side by side and inspect for blur, dust spots, or subtle camera angle shift. A sharpness metric (e.g., variance of Laplacian) computed automatically across saved frames over time, plotted as a trend, would show a declining curve if this is the cause.

### Cause 2: Lighting drift (seasonal changes, bulb degradation, dirty overhead lights)
Factory lighting changes seasonally (natural light contribution) or degrades physically (fluorescent/LED output dims over time). A model trained on one lighting condition can silently degrade as the real-world lighting distribution drifts away from what it saw in training.

**Confirming evidence:** Compare average frame brightness/histogram statistics between month 1 and month 3 — a shift in mean brightness or contrast distribution over time is a direct, quantifiable signal.

### Cause 3: Product/packaging change on the line (a vendor changed box design, color, or size slightly)
Something about what's actually on the belt may have changed without being flagged as a "line change" from an ops perspective (e.g., a supplier updated packaging design) — imperceptible to a human walking by, but a distribution shift the model was never trained on.

**Confirming evidence:** Sample recent frames and visually compare object appearance/packaging against training-set reference images; a clear visual mismatch confirms this without needing model introspection.

### Lightweight monitoring signal to catch this within two weeks
Track a **rolling weekly average of model confidence scores** for accepted detections (not just accuracy, since ground truth isn't available in real-time on a production line). A well-functioning model on in-distribution data tends to produce consistently high-confidence detections; as the input distribution drifts (blur, lighting, packaging change), average confidence tends to decline even before raw accuracy visibly drops, because the model is operating further from its training distribution.

**Specific signal:** compute the 7-day rolling mean of detection confidence scores, compared against a baseline established during the first month of stable deployment (e.g., baseline mean = 0.93). **Threshold:** fire an alert if the 7-day rolling mean drops more than 5-7% relative to baseline (e.g., below ~0.86) sustained for 3+ consecutive days (to avoid alerting on a single noisy day). This is cheap to compute (no ground truth needed, just logging confidence scores already produced during inference) and would have caught a 3-month gradual decline within roughly two weeks of it starting, well before it reached a 13-point accuracy drop.

---
# Part D — Answers (ANSWERS.md)

---

## 1. Aggregate Detection Throughput Required

8 cameras × 15 fps each = **120 frames per second aggregate**, if every frame from every camera must be processed.

```
8 cameras × 15 fps = 120 FPS aggregate throughput required
```

This is the number the single Orin must sustain across all 8 streams combined, not per stream.

## 2. Model Family, Precision, and Fit

**Choice: YOLOv8n (nano) at FP16, using TensorRT for inference on the Orin.**

**Reasoning for throughput fit:**
The 200ms latency budget is per-frame, but the binding constraint is actually aggregate throughput (120 FPS), not single-frame latency — a single Orin serving 8 streams needs to process one frame roughly every 1000/120 ≈ **8.3ms** on average to keep up in steady state, well inside the 200ms per-frame latency ceiling. This means the real design question is throughput capacity, and latency budget is comfortably satisfied as long as throughput is met (any single frame's actual end-to-end latency, including queueing, will be well under 200ms if the pipeline isn't backlogged).

YOLOv8n is the smallest YOLOv8 variant, chosen specifically because:
- Jetson AGX Orin (64GB variant) has published benchmarks in the range of several hundred FPS for YOLOv8n at FP16/INT8 with TensorRT at 640×640 input — comfortably above the 120 FPS aggregate target, leaving headroom for preprocessing, postprocessing (NMS), and multi-stream overhead which are not free.
- FP16 (not INT8) is the safer starting choice here: Orin has native FP16 tensor cores (unlike the CPU case in Part A), giving a real throughput benefit without the calibration-data risk and potential accuracy loss INT8 introduced in this very assignment's own benchmark (Part A showed FP16 was accuracy-neutral where INT8 wasn't attempted). Given the air-gapped, hard-to-patch deployment context, minimizing accuracy risk outweighs squeezing out extra throughput margin.

**Important caveat, stated explicitly as instructed:** I have not benchmarked YOLOv8n at FP16 on an actual Orin with this exact 8-stream concurrent-decode + inference + NMS pipeline — published single-stream FPS numbers do not automatically account for 8 concurrent RTSP/video decode pipelines competing for the same GPU and memory bandwidth. **The single measurement I would take first:** run the actual TensorRT-optimized YOLOv8n engine on the target Orin with all 8 real camera feeds decoding and running concurrently (not just synthetic single-stream throughput), and measure sustained end-to-end aggregate FPS and per-frame latency distribution (especially p95/p99, not just mean) over a realistic multi-minute window — this is the only way to know if video decode contention or memory bandwidth becomes the real bottleneck rather than raw model inference speed.

## 3. Retraining Loop Under a Hard Air-Gap

1. **Operator flagging:** the inspection UI includes a simple "flag" button next to each detection result (or a periodic review screen showing recent detections) — operator taps "false positive" or "missed defect" and the flagged frame + model output + operator's correction is saved locally to a dedicated `flagged/` directory on the on-prem server, tagged with timestamp and camera ID. No manual re-annotation is required at flag time — just marking it for later review, to keep operator friction low.

2. **Feedback reaching a training environment:** since there's no internet, the flagged data must move via **physical media** — a USB drive or removable SSD, following a defined pickup schedule (e.g., weekly), carried by an authorized person from the on-prem server to a separate, non-air-gapped training environment (offsite, e.g. at Artikate's own infrastructure). This transfer should include a checksum manifest so data integrity can be verified after transfer, and the transferred data should be treated as write-once/read-only at the training end to preserve an audit trail of exactly what came from the site.

3. **Validating a new model before it replaces the running one:**
   - Retrain/fine-tune offline in the training environment using the newly-labeled flagged data merged with the existing training set (not replacing it, to avoid catastrophic forgetting of previously-learned cases).
   - Evaluate the candidate model against a **fixed, held-out validation set that never changes** (kept from the original deployment) plus the newly flagged cases, to confirm both that general performance hasn't regressed and that the specific flagged failure modes are now handled correctly.
   - Only if the candidate model meets or exceeds the currently-deployed model's metrics on both sets does it get packaged (weights + config + a version tag) for transfer back to site — again via physical media, with a checksum.
   - On-site, the new model is deployed to a **staging/shadow mode** first: it runs in parallel with the currently-live model on real incoming frames (not yet controlling any real line action), and its outputs are logged and compared against the live model's outputs for a defined burn-in period (e.g., a few days) before an operator/engineer manually promotes it to replace the production model.

## 4. Rollback Plan

**Detection speed:** the shadow-mode comparison in step 3 is the primary defense — since the new model runs in parallel before full promotion, most regressions should be caught before ever reaching production. But for regressions that only appear after full promotion (e.g., triggered by a real environmental factor not present during the shadow period), detection relies on the same lightweight monitoring signal described in Part C3: a rolling confidence-score baseline. **Signal that triggers rollback:** if the 7-day (or shorter, e.g. 48-hour, given how critical a defect line is) rolling average confidence score drops more than a defined threshold (e.g., 5-7%) relative to the established post-deployment baseline, or if a manual operator flag rate spikes noticeably above its recent historical rate, that triggers an alert for human review.

**Rollback mechanism:** the previous production model's weights are never deleted — they're kept on-site as the "last known good" version alongside the new one, so rollback is simply a config/symlink swap back to the previous model file, restartable within minutes without needing any network access or re-transfer of data. This requires disciplined versioning (never overwrite a deployed model file in place) as a hard operational rule.

## 5. Least Confident Part, and What's Needed to Resolve It

**Least confident: whether YOLOv8n at FP16 via TensorRT actually sustains 120 FPS aggregate across 8 concurrent real RTSP decode + inference + NMS pipelines on a single Orin**, as opposed to single-stream synthetic benchmarks. Video decode (especially if streams arrive H.264/H.265 encoded and need on-device decode before inference), memory bandwidth sharing across 8 concurrent CUDA contexts, and NMS/postprocessing overhead at this frame rate are all real-world factors that don't show up in simple single-image inference benchmarks, and I have not been able to test this directly.

**What's needed to resolve it:** direct benchmarking with the actual target hardware, actual camera feeds (or realistic recorded 1080p/15fps footage played back to simulate them), and the full intended software stack (DeepStream or equivalent multi-stream pipeline, not just a bare model loop) running all 8 streams concurrently, measuring sustained throughput and full latency distribution over an extended real-world test window before committing to this configuration for production.

