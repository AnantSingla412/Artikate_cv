"""
Exports a reduced-precision (FP16) ONNX version of the trained model.
FP16 chosen over INT8. FP16 halves model size and is well supported by ONNX
Runtime and edge targets like Jetson.

Run from repo root: python script/quantize.py
"""

from ultralytics import YOLO

model = YOLO("model/best.pt")
model.export(format="onnx", imgsz=640, opset=12, half=True)