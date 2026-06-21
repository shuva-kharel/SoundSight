#!/usr/bin/env python
"""
merge_new_datasets.py  --  Fold the extra Roboflow exports into dataset/ so that
`python train_banknote.py` trains on everything at once.

Sources:
  * new_dataset/            YOLO DETECTION export (images/ + labels/*.txt). The
                            denomination is the class index in each .txt, decoded
                            via data.yaml names. The whole image is copied (the
                            note fills most of the frame, which matches how the
                            Money mode captures a held note).
  * new_dataset_500_1000/   CLASSIFICATION export (train/500, train/1000, ...).

Split mapping (both sources):  train -> train ,  valid + test -> val .

The 'fake' class in new_dataset is SKIPPED -- the banknote classifier has no
'fake' class (counterfeit detection is a separate feature). Files are COPIED (the
originals are left intact) with a source prefix so names never collide. Re-running
is safe (it just overwrites the same prefixed copies).

    python merge_new_datasets.py --dry-run    # show the plan, copy nothing
    python merge_new_datasets.py              # do the merge
"""

import argparse
import collections
import shutil
from pathlib import Path

DEST = Path("dataset")
SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}

# --- new_dataset (detection) ------------------------------------------------ #
DET_DIR = Path("new_dataset")
DET_NAMES = ["fake", "fifty", "five", "fivehundred", "hundred", "ten", "thousand", "twenty"]
DET_TO_RS = {
    "fifty": "rs50", "five": "rs5", "fivehundred": "rs500", "hundred": "rs100",
    "ten": "rs10", "thousand": "rs1000", "twenty": "rs20",   # 'fake' deliberately absent
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def label_class(txt_path: Path):
    """Return the rs-class for a YOLO label file, or None to skip (fake/empty)."""
    idxs = collections.Counter()
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if line:
            idxs[int(line.split()[0])] += 1
    if not idxs:
        return None
    name = DET_NAMES[idxs.most_common(1)[0][0]]   # all images here are single-class
    return DET_TO_RS.get(name)                    # None for 'fake'


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def merge_detection(counts, dry):
    if not DET_DIR.is_dir():
        print(f"[!] {DET_DIR} not found, skipping.")
        return
    for split, dst_split in SPLIT_MAP.items():
        labels = DET_DIR / split / "labels"
        images = DET_DIR / split / "images"
        if not labels.is_dir():
            continue
        for txt in labels.glob("*.txt"):
            rs = label_class(txt)
            if rs is None:
                counts[("skip(fake/empty)", split)] += 1
                continue
            img = find_image(images, txt.stem)
            if img is None:
                counts[("skip(no-image)", split)] += 1
                continue
            counts[(rs, dst_split)] += 1
            if not dry:
                out = DEST / dst_split / rs / f"nd_{img.name}"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out)


# --- new_dataset_500_1000 (classification) ---------------------------------- #
CLS_DIR = Path("new_dataset_500_1000")
CLS_TO_RS = {"500": "rs500", "1000": "rs1000"}


def merge_classification(counts, dry):
    if not CLS_DIR.is_dir():
        print(f"[!] {CLS_DIR} not found, skipping.")
        return
    for split, dst_split in SPLIT_MAP.items():
        sdir = CLS_DIR / split
        if not sdir.is_dir():
            continue
        for cls, rs in CLS_TO_RS.items():
            src = sdir / cls
            if not src.is_dir():
                continue
            for img in src.iterdir():
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                counts[(rs, dst_split)] += 1
                if not dry:
                    out = DEST / dst_split / rs / f"nd5k_{img.name}"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, out)


def main():
    ap = argparse.ArgumentParser(description="Merge extra Roboflow exports into dataset/.")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, copy nothing")
    args = ap.parse_args()

    counts = collections.Counter()
    merge_detection(counts, args.dry_run)
    merge_classification(counts, args.dry_run)

    verb = "WOULD ADD" if args.dry_run else "ADDED"
    print(f"\n{verb} to dataset/ (per class, per split):")
    rs_order = ["rs5", "rs10", "rs20", "rs50", "rs100", "rs500", "rs1000"]
    for dst_split in ("train", "val"):
        line = ", ".join(f"{rs}={counts[(rs, dst_split)]}" for rs in rs_order)
        print(f"  {dst_split:5}: {line}")
    skipped = {k: v for k, v in counts.items() if str(k[0]).startswith("skip")}
    if skipped:
        print("  skipped:", ", ".join(f"{k[0]}[{k[1]}]={v}" for k, v in sorted(skipped.items())))
    if args.dry_run:
        print("\n(dry run -- nothing copied. Re-run without --dry-run to apply.)")
    else:
        print("\n[i] Done. Originals in new_dataset/ and new_dataset_500_1000/ are untouched;")
        print("    delete them once you've confirmed training looks good.")


if __name__ == "__main__":
    main()
