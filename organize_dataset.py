#!/usr/bin/env python
"""
organize_dataset.py  --  Sort flat Roboflow-export images into per-class folders.

The banknote photos arrived dumped flat into dataset/train/ and dataset/val/ with
the denomination encoded in the FILENAME (a merged Roboflow export), e.g.

    train_fifty_102_jpg.rf.<hash>.jpg     -> rs50
    fivehun_train_115_jpg.rf.<hash>.jpg   -> rs500   (NOT rs5!)
    hundred_valid_221_jpg.rf.<hash>.jpg   -> rs100
    IMG_20190327_124905_jpg.rf.<hash>.jpg -> (no denomination in name -> review)

Ultralytics classification needs  dataset/<split>/<class>/*.jpg , so this moves
each file into the right class folder by matching whole underscore-tokens against
known denomination words. Files with zero or several denomination words go to
dataset/_unsorted_review/<split>/ for you to sort by hand (none are deleted).

Idempotent: only loose files in the split root are touched; re-running is safe.

    python organize_dataset.py --dry-run     # show the plan, move nothing
    python organize_dataset.py               # actually move the files
"""

import argparse
import collections
import shutil
from pathlib import Path

# whole underscore-token -> class. Matching whole tokens (not substrings) is what
# keeps "fivehundred"/"fivehun" out of the rs5 bucket.
KEYWORD_TO_CLASS = {
    "five": "rs5",
    "ten": "rs10",
    "twenty": "rs20",
    "fifty": "rs50",
    "hundred": "rs100",
    "fivehun": "rs500",
    "fivehundred": "rs500",
    "thousand": "rs1000",
    "background": "background",
    "bg": "background",
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def classify(filename: str):
    """Return the class for a filename, or None if it isn't unambiguous."""
    head = filename.lower().split(".rf.", 1)[0]   # drop the roboflow "_jpg.rf.<hash>" tail
    classes = {KEYWORD_TO_CLASS[t] for t in head.split("_") if t in KEYWORD_TO_CLASS}
    return next(iter(classes)) if len(classes) == 1 else None


def main():
    ap = argparse.ArgumentParser(description="Sort flat banknote images into class folders.")
    ap.add_argument("--data", default="dataset", help="dataset root containing train/ and val/")
    ap.add_argument("--dry-run", action="store_true", help="print the plan but move nothing")
    args = ap.parse_args()

    root = Path(args.data)
    review = root / "_unsorted_review"
    grand = collections.Counter()

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[!] skip: {split_dir} does not exist")
            continue
        loose = [p for p in split_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
        counts = collections.Counter()
        for p in loose:
            cls = classify(p.name)
            dest_dir = (review / split) if cls is None else (split_dir / cls)
            counts[cls or "_REVIEW_"] += 1
            grand[cls or "_REVIEW_"] += 1
            if not args.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest_dir / p.name))
        print(f"\n=== {split}: {len(loose)} loose files ===")
        for k in ["rs5", "rs10", "rs20", "rs50", "rs100", "rs500", "rs1000",
                  "background", "_REVIEW_"]:
            if counts.get(k):
                tag = "  -> _unsorted_review/" if k == "_REVIEW_" else ""
                print(f"  {k:<12} {counts[k]}{tag}")

    verb = "WOULD MOVE" if args.dry_run else "MOVED"
    print(f"\n{verb} totals: " + ", ".join(f"{k}={v}" for k, v in sorted(grand.items())))
    if args.dry_run:
        print("\n(dry run -- nothing changed. Re-run without --dry-run to apply.)")
    elif grand.get("_REVIEW_"):
        print(f"\n[i] {grand['_REVIEW_']} file(s) had no clear denomination in their "
              f"name and went to {review}/ -- sort those by hand if you want them.")


if __name__ == "__main__":
    main()
