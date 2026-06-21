"""
camera.py  --  OS-aware, robust webcam opening for SoundSight.

Windows (dev PC) and the Raspberry Pi (V4L2) need different OpenCV backends, and the
Fantech USB cam needs MJPG on Linux but breaks if MJPG is forced under Windows MSMF.
This module hides all of that:

  * Windows -> DirectShow (CAP_DSHOW) first (fixes the MSMF "can't grab frame
    -1072875772" error), then MSMF, then ANY.
  * Linux/Pi -> V4L2 first (+ MJPG fourcc, 640x480), then ANY.
  * Every open is VALIDATED by actually reading a non-empty test frame before it's
    accepted, then ~5 warm-up frames are discarded (some cams return black at first).

  open_camera(index)   -> (cap, backend_name) or (None, None)
  find_camera()        -> (cap, index, backend_name) -- tries 0,1,2
  list_cameras()       -> probe 0..4 and print which return real frames
"""

import logging
import sys

import cv2

log = logging.getLogger("soundsight.camera")

IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

FRAME_W, FRAME_H = 640, 480
WARMUP_FRAMES = 5


def _backends():
    """Ordered (name, flag) backends to try for THIS OS."""
    if IS_WINDOWS:
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    if IS_LINUX:
        return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def _try_open(index, flag):
    """Open one index with one backend; return a cap only if it yields a real frame."""
    try:
        cap = cv2.VideoCapture(index) if flag == cv2.CAP_ANY else cv2.VideoCapture(index, flag)
    except cv2.error:
        return None
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if IS_LINUX:   # Fantech needs MJPG on the Pi; do NOT force it on Windows
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ok, frame = cap.read()
    if not ok or frame is None or getattr(frame, "size", 0) == 0:
        cap.release()
        return None
    return cap


def _warmup(cap, n=WARMUP_FRAMES):
    for _ in range(n):
        cap.read()


def open_camera(index):
    """Open one camera index using the OS-preferred backends. Returns
    (cap, backend_name) or (None, None)."""
    for name, flag in _backends():
        cap = _try_open(index, flag)
        if cap is not None:
            _warmup(cap)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("Camera opened: index %d, backend %s, %dx%d", index, name, w, h)
            return cap, name
    return None, None


def find_camera(indices=(0, 1, 2)):
    """Try several indices; return (cap, index, backend_name) for the first that
    returns a real frame, else (None, None, None) with a clear log."""
    for i in indices:
        cap, name = open_camera(i)
        if cap is not None:
            return cap, i, name
    tried = ", ".join(b[0] for b in _backends())
    log.error("No working camera among indices %s (backends tried: %s). "
              "Is the webcam plugged in and not in use by another app?", list(indices), tried)
    return None, None, None


def list_cameras(max_index=4):
    """Probe indices 0..max_index and print which return real frames (--list-cameras)."""
    print(f"Probing camera indices 0..{max_index} on {sys.platform} "
          f"(backends: {', '.join(b[0] for b in _backends())}) ...")
    found = []
    for i in range(max_index + 1):
        cap, name = open_camera(i)
        if cap is not None:
            print(f"  index {i}: WORKS  (backend {name})")
            found.append(i)
            cap.release()
        else:
            print(f"  index {i}: no frame")
    print("Found working camera(s): " + (", ".join(map(str, found)) if found else "NONE"))
    return found
