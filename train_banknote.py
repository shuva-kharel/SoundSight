#!/usr/bin/env python
"""
train_banknote.py  --  Train a Nepali rupee-note classifier (YOLO11 classification).

STANDALONE -- this is NOT part of the SoundSight app; it just shares the venv.

Transfer-learns from the ImageNet-pretrained `yolo11n-cls.pt` onto your own photos.

Dataset layout you create (Ultralytics classification format):

    dataset/
      train/
        rs5/*.jpg  rs10/*.jpg  rs20/*.jpg  rs50/*.jpg
        rs100/*.jpg  rs500/*.jpg  rs1000/*.jpg  background/*.jpg
      val/
        rs5/*.jpg ... background/*.jpg

Run:
    python train_banknote.py                 # 30 epochs on the GPU
    python train_banknote.py --epochs 50
    python train_banknote.py --device cpu    # if you have no CUDA

Outputs:
    models/banknote.pt              (PyTorch weights -- use with predict_banknote.py)
    models/banknote_ncnn_model/     (NCNN export -- fast on a Raspberry Pi CPU)
"""

import argparse
import shutil
from pathlib import Path

CLASSES = ["rs5", "rs10", "rs20", "rs50", "rs100", "rs500", "rs1000", "background"]
# OPTIONAL: add "note_partial" for half-visible / edge-of-frame notes -- create a
# dataset/<split>/note_partial/ folder and it will be picked up automatically. Money
# mode treats anything that isn't a clean denomination as "no note", so partials and
# background both fail the gate -> "hold a note steady" instead of a wrong value.
BASE_MODEL = "yolo11n-cls.pt"   # ImageNet-pretrained YOLO11 nano classifier (auto-downloads)
RUN_DIR = Path("banknote_runs")  # training artifacts (plots, weights, confusion matrix)


# --------------------------------------------------------------------------- #
# Heavy augmentation -- real notes are shot at angles, upside down, in bad light.
# These are passed straight to model.train(). The big ones for this problem:
#   auto_augment="randaugment" -> random rotation / brightness / contrast / shear / sharpness
#   degrees=180                -> a note can be photographed at ANY orientation
#   hsv_v / hsv_s              -> strong brightness & colour jitter (dim rooms, glare)
#   erasing                    -> fingers / folds / glare covering part of the note
#   flipud=0.5                 -> upside-down notes are common; fliplr=0 because a real
#                                 note is never seen mirror-imaged (the print would reverse)
# --------------------------------------------------------------------------- #
AUGMENT = dict(
    auto_augment="randaugment",
    degrees=180.0,
    translate=0.12,
    scale=0.5,
    shear=5.0,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.6,
    fliplr=0.0,
    flipud=0.5,
    erasing=0.5,
)


def print_collection_guide():
    print(
        """
========================  HOW MANY PHOTOS TO COLLECT  ========================
Aim for **80+ photos per note class** (150+ is noticeably better):
    rs5  rs10  rs20  rs50  rs100  rs500  rs1000

*** 'background' is the MOST IMPORTANT class and should be the LARGEST. ***
Collect **150+ (ideally 200-300) NO-NOTE photos** so the model learns to say
"no note" instead of HALLUCINATING a denomination on an empty scene. Vary them a
LOT -- this is what kills the "it says 100 at a blank wall" bug:
    empty hands, tables, floors, walls, doors, wallets (no note inside),
    other paper / receipts / books, clothing, phones, random clutter,
    bright AND dim, close AND far, blurry AND sharp.

For each NOTE class, vary EVERYTHING so the model generalises:
  * angle & rotation   -- flat, tilted, upside down, rotated every which way
  * lighting           -- bright, dim, shadow, yellow indoor light, glare
  * distance / framing -- close-up and far, centred and off to a side
  * condition          -- crisp new notes AND creased/worn/dirty ones
  * both sides         -- front and back of each denomination
  * a little blur       -- some slightly out-of-focus / motion-blurred shots

OPTIONAL 'note_partial' class: half-visible / cut-off notes, so a note sliding
into frame isn't read as a confident full denomination.

Split ~80% into dataset/train/<class>/ and ~20% into dataset/val/<class>/.
IMPORTANT: val photos must be DIFFERENT shots from train (ideally different
notes / sessions) -- otherwise the reported accuracy is a lie.

Money mode also requires high confidence + a margin over the runner-up AND a
majority vote across ~10 frames, so even an imperfect model won't blurt a value.
=============================================================================
"""
    )


def check_dataset(data_dir: Path) -> bool:
    train, val = data_dir / "train", data_dir / "val"
    if not train.is_dir() or not val.is_dir():
        print(f"[!] Missing '{train}' and/or '{val}'.")
        return False
    found = sorted(p.name for p in train.iterdir() if p.is_dir())
    if not found:
        print(f"[!] No class folders inside '{train}'.")
        return False
    counts = {c: len(list((train / c).glob("*.*"))) for c in found}
    print(f"[i] Found {len(found)} train classes: {', '.join(found)}")
    print("[i] Photos per train class:", ", ".join(f"{c}={n}" for c, n in counts.items()))
    thin = [c for c, n in counts.items() if n < 80]
    if thin:
        print(f"[!] Under 80 photos: {', '.join(thin)} -- accuracy will suffer; collect more.")
    # background must exist and be large -- it's what prevents "no note" hallucination.
    bg = counts.get("background", 0)
    if bg == 0:
        print("[!] NO 'background' class! The model WILL hallucinate denominations on empty "
              "scenes. Add dataset/{train,val}/background/ with 150+ no-note photos.")
    elif bg < 150:
        print(f"[!] background has only {bg} photos -- collect 150+ (ideally the LARGEST class) "
              "so 'no note' rejection is strong.")
    elif bg < max(counts.values()):
        print(f"[i] Tip: background ({bg}) isn't the largest class -- more no-note variety "
              "further reduces false denominations.")
    missing = [c for c in CLASSES if c not in found]
    if missing:
        print(f"[!] Note: expected classes not present yet: {', '.join(missing)}")
    return True


def resolve_device(requested: str):
    if requested != "auto":
        return requested
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def print_confusion_matrix(metrics, names):
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        print("[i] (confusion matrix unavailable)")
        return
    import numpy as np

    matrix = np.asarray(cm.matrix).astype(int)
    base = [names[i] for i in range(len(names))]
    # Ultralytics allocates one extra (unused) background row/col -- drop it so we
    # don't print a duplicate of the real "background" class.
    if matrix.shape[0] == len(base) + 1:
        matrix = matrix[:len(base), :len(base)]
    n = matrix.shape[0]
    labels = base[:n] if n <= len(base) else [str(i) for i in range(n)]
    w = max(10, max(len(l) for l in labels) + 2)
    print("\nConfusion matrix  (rows = PREDICTED, cols = TRUE) -- "
          "see confusion_matrix.png for the labelled plot:")
    print(" " * w + "".join(f"{l:>{w}}" for l in labels))
    for i in range(n):
        print(f"{labels[i]:>{w}}" + "".join(f"{int(matrix[i, j]):>{w}}" for j in range(n)))


def main():
    ap = argparse.ArgumentParser(description="Train a Nepali banknote classifier (YOLO11-cls).")
    ap.add_argument("--data", default="dataset", help="dataset root (with train/ and val/)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="auto", help="'auto' | 0 | cpu")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    print_collection_guide()

    data_dir = Path(args.data)
    if not check_dataset(data_dir):
        print("\n[x] Build the dataset above, then re-run.  python train_banknote.py")
        raise SystemExit(1)

    device = resolve_device(args.device)
    print(f"\n[i] Training on device: {device}  (RTX 5060 -> '0';  no GPU -> 'cpu')")

    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)  # transfer learning from ImageNet weights
    results = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project=str(RUN_DIR),
        name="train",
        exist_ok=True,
        patience=15,        # early-stop if val accuracy plateaus
        dropout=0.2,        # regularise -- datasets like this are small
        cos_lr=True,
        verbose=True,
        **AUGMENT,
    )

    # ---- evaluate the BEST checkpoint: accuracy + confusion matrix ----------
    best = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(data_dir), imgsz=args.imgsz, device=device,
                             split="val", project=str(RUN_DIR), name="val", exist_ok=True)
    top1 = float(getattr(metrics, "top1", 0.0))
    top5 = float(getattr(metrics, "top5", 0.0))
    print(f"\n================  VALIDATION  ================")
    print(f"  Top-1 accuracy: {top1 * 100:.2f}%")
    print(f"  Top-5 accuracy: {top5 * 100:.2f}%")
    print_confusion_matrix(metrics, best_model.names)

    # ---- export best -> models/banknote.pt  and  models/banknote_ncnn_model/ -
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pt_out = models_dir / "banknote.pt"
    shutil.copy(best, pt_out)

    print("\n[i] Exporting NCNN (fast on the Raspberry Pi CPU)...")
    ncnn_src = Path(best_model.export(format="ncnn", imgsz=args.imgsz))  # creates best_ncnn_model/
    ncnn_out = models_dir / "banknote_ncnn_model"
    if ncnn_out.exists():
        shutil.rmtree(ncnn_out)
    shutil.move(str(ncnn_src), str(ncnn_out))

    print(f"\n================  DONE  ================")
    print(f"  PyTorch model : {pt_out}")
    print(f"  NCNN model    : {ncnn_out}")
    print(f"  Run artifacts : {results.save_dir}  (results.png, confusion_matrix.png)")
    print(f"\nSanity-check a photo:\n  python predict_banknote.py path/to/note.jpg")


if __name__ == "__main__":
    main()
