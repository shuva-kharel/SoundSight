#!/usr/bin/env python
"""
predict_banknote.py  --  Classify ONE banknote photo with the trained model.

STANDALONE sanity-check tool for train_banknote.py.

Usage:
    python predict_banknote.py path/to/photo.jpg
    python predict_banknote.py path/to/photo.jpg models/banknote_ncnn_model   # use NCNN

Prints the most likely Nepali-rupee class and its confidence, plus the top-3.
"""

import sys
from pathlib import Path

DEFAULT_MODEL = "models/banknote.pt"


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_banknote.py <image> [model_path]")
        raise SystemExit(1)

    image = sys.argv[1]
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_MODEL)

    if not Path(image).exists():
        print(f"[x] Image not found: {image}")
        raise SystemExit(1)
    if not model_path.exists():
        print(f"[x] Model not found: {model_path}\n    Train it first:  python train_banknote.py")
        raise SystemExit(1)

    from ultralytics import YOLO

    # task="classify" is required when loading the NCNN export (a model folder has
    # no task metadata, so Ultralytics would otherwise assume detection and crash).
    model = YOLO(str(model_path), task="classify")
    result = model.predict(image, imgsz=224, verbose=False)[0]
    probs = result.probs

    top1, conf = int(probs.top1), float(probs.top1conf)
    print(f"\n  => {model.names[top1]}   ({conf * 100:.1f}% confident)\n")

    print("  Top 3:")
    data = probs.data.tolist()
    for idx in list(probs.top5)[:3]:
        print(f"    {model.names[int(idx)]:<12} {data[int(idx)] * 100:5.1f}%")


if __name__ == "__main__":
    main()
