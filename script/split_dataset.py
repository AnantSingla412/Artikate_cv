"""
Splits the annotated dataset into train/val sets.
Run BEFORE any augmentation (augmentation is applied at training time only,
via Ultralytics' training config), so no augmented crop of a val image
can leak into training.

Run from repo root: python script/split_dataset.py
"""

import random
import shutil
from pathlib import Path

random.seed(42)  # fixed seed so the split is reproducible

RAW_IMAGES = Path(r"data\raw\images")
RAW_LABELS = Path(r"data\raw\labels")
TRAIN_DIR = Path(r"data\train")
VAL_DIR = Path(r"data\val")
VAL_RATIO = 0.2


def main():
    # Collect all image filenames (without extension) from the raw export
    all_images = sorted(f.stem for f in RAW_IMAGES.glob("*.jpg"))
    print(f"Total images found: {len(all_images)}")

    random.shuffle(all_images)

    n_val = int(len(all_images) * VAL_RATIO)
    val_set = all_images[:n_val]
    train_set = all_images[n_val:]

    for split_name, split_files, out_dir in [
        ("train", train_set, TRAIN_DIR),
        ("val", val_set, VAL_DIR),
    ]:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

        for stem in split_files:
            shutil.copy(RAW_IMAGES / f"{stem}.jpg", out_dir / "images" / f"{stem}.jpg")
            label_path = RAW_LABELS / f"{stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, out_dir / "labels" / f"{stem}.txt")

    print(f"Train: {len(train_set)} images")
    print(f"Val:   {len(val_set)} images")


if __name__ == "__main__":
    main()