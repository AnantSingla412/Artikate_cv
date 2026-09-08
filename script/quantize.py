"""
Exports a reduced-precision (FP16) ONNX version of the trained model.
FP16 chosen over INT8 because INT8 requires a calibration dataset and
extra tooling, which isn't justified given the dataset size and time
budget here. FP16 halves model size and is well supported by ONNX
Runtime and edge targets like Jetson.

Run from repo root: python scripts/quantize.py
"""

from ultralytics import YOLO

model = YOLO("models/best_run2.pt")
model.export(format="onnx", imgsz=640, opset=12, half=True)