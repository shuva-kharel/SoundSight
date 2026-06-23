#!/usr/bin/env python
"""
train_banknote_detect.py  --  Train a banknote DETECTION model (Path A multi-note).

The classifier (train_banknote.py) gives ONE label per image, so it can't count
multiple notes in one frame. A DETECTION model outputs a box + denomination per
note, which natively handles several notes spread out in view. Money mode uses it
automatically when models/banknote_detect.pt (or the NCNN export) exists; otherwise
it falls back to Path B (contour + classifier).

==========================  DATA YOU NEED TO COLLECT  =========================
DETECTION labels = a bounding box AROUND EACH NOTE, class = its denomination.
Label with Roboflow (easiest) or LabelImg, export **YOLO format**.

  * Photograph **1-4 notes per image**, spread out and also touching/overlapping.
  * Vary layout, angle, lighting, background, both sides, worn/new.
  * Classes: rs5 rs10 rs20 rs50 rs100 rs500 rs1000  (NO background class -- in
    detection, "nothing" is simply no box, which is what stops false counts).
  * Aim for **300-600 labelled images** total (more is better); make sure every
    denomination appears many times and in multi-note scenes.
  * Keep a held-out val split of different photos.

Roboflow export gives you a data.yaml; point --data at it.

    python train_banknote_detect.py --data path/to/data.yaml --epochs 100
==============================================================================
"""

import argparse
import shutil
from pathlib import Path

BASE_MODEL = "yolo11s.pt"   # detection base; use yolo11n.pt for a lighter/faster model
RUN_DIR = Path("banknote_detect_runs")


def main():
    ap = argparse.ArgumentParser(description="Train a Nepali banknote DETECTION model.")
    ap.add_argument("--data", required=True, help="YOLO data.yaml (boxes + denomination classes)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="'auto' | 0 | cpu")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=device,
                project=str(RUN_DIR), name="train", exist_ok=True, patience=20,
                degrees=10, translate=0.1, scale=0.5, fliplr=0.0, flipud=0.0,
                hsv_v=0.5, hsv_s=0.6, mosaic=1.0)

    best = Path(RUN_DIR) / "train" / "weights" / "best.pt"
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / "banknote_detect.pt"
    shutil.copy(best, out)
    print(f"\n[i] Exporting NCNN for the Pi...")
    ncnn = Path(YOLO(str(best)).export(format="ncnn", imgsz=args.imgsz))
    ncnn_out = models_dir / "banknote_detect_ncnn_model"
    if ncnn_out.exists():
        shutil.rmtree(ncnn_out)
    shutil.move(str(ncnn), str(ncnn_out))

    print(f"\n================  DONE  ================")
    print(f"  Detection model : {out}")
    print(f"  NCNN (Pi)       : {ncnn_out}")
    print("Money mode now uses Path A (multi-note detection) automatically. "
          "Say 'count the notes'.")


if __name__ == "__main__":
    main()
