# scripts/export_onnx.py
"""
Exports the trained YOLOv8 model to ONNX format for inference via ONNX Runtime.

"""

from ultralytics import YOLO

model = YOLO("model/best.pt")
model.export(format="onnx", imgsz=640, opset=12)