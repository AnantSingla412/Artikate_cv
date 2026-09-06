# Artikate CV Assignment — Case & Speaker Detector

## Overview
Two-class object detector (case, speaker) trained on a self-captured dataset.

## Dataset
- Total images: 93
- Classes: 0 = case, 1 = speaker
- Per-class instance counts: (case: 72, speaker: 72)
- Train/val split: 74 / 19 (80/20, random seed=42)
- Split guarantee: split performed on raw captured images before any augmentation;
  each image is a distinct physical capture, so no image or crop of it appears
  in both train and val.

## Annotation
- Tool used: Roboflow
- Format: YOLOv8 (bounding boxes)