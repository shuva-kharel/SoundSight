#!/usr/bin/env python
"""
face_enroll.py  --  Enroll a known person into SoundSight's face database (opt-in).

CONSENT FIRST: this stores a face embedding ONLY for people who agree to be
recognized. It is entirely local (faces_db/faces.json) -- nothing is uploaded.

    python face_enroll.py --name Ram                 # capture from the webcam (press SPACE)
    python face_enroll.py --name Sita --images ./sita # enroll from a folder of photos
    python face_enroll.py --list                       # who is enrolled
    python face_enroll.py --forget Ram                 # remove a person

Uses the same FaceMatcher/provider as the live app (faces.py).
"""

import argparse
import sys
from pathlib import Path

import cv2

import camera as cam
from faces import DB_FILE, FaceMatcher


def _consent():
    print("\n=== CONSENT ===")
    print("This saves a numeric face signature so SoundSight can greet this person")
    print("by name. It is LOCAL only (faces_db/), never uploaded. Only enroll people")
    print("who have agreed. Delete faces_db/ at any time to forget everyone.\n")


def enroll_from_images(matcher, name, folder):
    n = 0
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        ok, msg = matcher.enroll(name, img)
        print(f"  {p.name}: {msg}")
        n += 1 if ok else 0
    print(f"Enrolled {n} sample(s) for {name}.")


def enroll_from_webcam(matcher, name):
    capt, idx, backend = cam.find_camera()
    if capt is None:
        print("No camera found. Try `python pi_app.py --list-cameras`.")
        return
    print(f"Camera {idx} ({backend}). Look at the camera; press SPACE to capture, Q to quit.")
    saved = 0
    try:
        while True:
            ok, frame = capt.read()
            if not ok:
                continue
            cv2.imshow("Enroll (SPACE=capture, Q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                ok, msg = matcher.enroll(name, frame)
                print(" ", msg)
                saved += 1 if ok else 0
            elif key in (ord("q"), 27):
                break
    finally:
        capt.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} sample(s) for {name}.")


def main():
    ap = argparse.ArgumentParser(description="Enroll known faces (opt-in, local).")
    ap.add_argument("--name", help="person's name to enroll")
    ap.add_argument("--images", help="folder of photos to enroll from (instead of webcam)")
    ap.add_argument("--list", action="store_true", help="list enrolled people")
    ap.add_argument("--forget", help="remove a person from the database")
    args = ap.parse_args()

    matcher = FaceMatcher(enabled=True)

    if args.list:
        if not matcher.db:
            print("No one enrolled yet.")
        for name, vecs in matcher.db.items():
            print(f"  {name}: {len(vecs)} sample(s)")
        return
    if args.forget:
        if args.forget in matcher.db:
            del matcher.db[args.forget]
            matcher._save_db()
            print(f"Removed {args.forget}.")
        else:
            print(f"{args.forget} is not enrolled.")
        return
    if not args.name:
        ap.error("give --name (and optionally --images), or use --list / --forget")
    if not matcher.available:
        print("Face recognition isn't installed. Install one provider:")
        print("  pip install insightface onnxruntime      (best, GPU)")
        print("  pip install face_recognition             (lighter)")
        sys.exit(1)

    _consent()
    if args.images:
        enroll_from_images(matcher, args.name, args.images)
    else:
        enroll_from_webcam(matcher, args.name)
    print(f"Database: {DB_FILE}")


if __name__ == "__main__":
    main()
