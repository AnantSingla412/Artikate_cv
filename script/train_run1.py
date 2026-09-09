# scripts/train.py
"""
Trains a YOLOv8 detector on the case/speaker dataset.
Run from repo root: python script/train_run1.py
"""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano 


model.train(
    data=r"data\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=42,
)