# scripts/train.py
"""
Trains a YOLOv8 detector on the case/speaker dataset.
Run from repo root: python scripts/train.py
"""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano 


model.train(
    data=r"E:\artikate\data\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=42,
)