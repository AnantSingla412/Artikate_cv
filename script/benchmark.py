# scripts/benchmark.py
"""
Benchmarks FP32 vs FP16 ONNX models on the same hardware:
- latency per image (mean, p95)
- model file size
- validation accuracy (mAP50)

Run from repo root: python scripts/benchmark.py
"""

import time
import glob
import os
import numpy as np
from ultralytics import YOLO

IMGSZ = 640
N_WARMUP = 3

def benchmark_latency(model_path, img_paths):
    model = YOLO(model_path)
    latencies = []

    # warmup (first few runs are always slower - exclude from timing)
    for img_path in img_paths[:N_WARMUP]:
        model.predict(img_path, imgsz=IMGSZ, verbose=False)

    for img_path in img_paths:
        start = time.perf_counter()
        model.predict(img_path, imgsz=IMGSZ, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return np.array(latencies)

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def get_val_map(model_path, data_yaml=r"E:\artikate\data\data.yaml"):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=IMGSZ, verbose=False)
    return metrics.box.map50

if __name__ == "__main__":
    img_paths = sorted(glob.glob(r"E:\artikate\data\val\images\*.jpg"))
    print(f"Benchmarking on {len(img_paths)} validation images\n")

    results = {}
    for label, path in [("FP32", r"E:\artikate\model\best.onnx"),
                          ("FP16", r"E:\artikate\model\best_quantized.onnx")]:
        print(f"--- {label} ({path}) ---")
        latencies = benchmark_latency(path, img_paths)
        size_mb = get_file_size_mb(path)
        map50 = get_val_map(path)

        results[label] = {
            "mean_latency_ms": latencies.mean(),
            "p95_latency_ms": np.percentile(latencies, 95),
            "size_mb": size_mb,
            "map50": map50,
        }
        print(f"  Mean latency: {latencies.mean():.2f} ms")
        print(f"  P95 latency:  {np.percentile(latencies, 95):.2f} ms")
        print(f"  File size:    {size_mb:.2f} MB")
        print(f"  mAP50:        {map50:.4f}\n")

    print("--- Summary ---")
    print(f"{'Metric':<20}{'FP32':<15}{'FP16':<15}")
    for key in ["mean_latency_ms", "p95_latency_ms", "size_mb", "map50"]:
        print(f"{key:<20}{results['FP32'][key]:<15.4f}{results['FP16'][key]:<15.4f}")