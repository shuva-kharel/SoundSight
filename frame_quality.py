"""
frame_quality.py  --  Shared frame preprocessing / quality assessment for SoundSight.

A wearable camera sees bad light, motion blur and tilt. EVERY mode (Navigate, Read,
Describe, Money) runs its frames through this one layer so we fix the root cause once
instead of per-mode. Pure functions + tunable constants, no app state, so it imports
cleanly on the laptop AND the Raspberry Pi (only needs OpenCV + NumPy).

  assess(frame)        -> {brightness, blur_score, ok, reason}   (gate junk frames)
  enhance(frame)       -> frame   (CLAHE adaptive contrast + auto-gamma; the single
                                   biggest real-world accuracy win -- run before inference)
  read_preprocess(f)   -> gray    (CLAHE + upscale + deskew, for OCR)
  binarize(gray)       -> gray    (adaptive threshold; OCR low-confidence rescue only)
  deskew(gray)         -> (gray, angle)
  rotate90(frame, k)   -> frame   (k=1/2/3 -> 90/180/270, for the OCR sideways retry)
  frame_diff(a, b)     -> 0..1    (mean abs diff; used to spot a frozen/dead camera)

All thresholds are module constants so they can be tuned without touching callers.
"""

import cv2
import numpy as np

# --- quality thresholds ----------------------------------------------------- #
# Kept LENIENT on purpose: a wearable camera + JPEG downscale lowers the
# variance-of-Laplacian a lot, so a high BLUR_MIN false-flags perfectly usable
# frames. These only catch GENUINELY unusable frames (covered lens / near-black /
# heavy blur). Brightness is reliable; blur is content-dependent, so blur is used
# ONLY to gate Read/Money (where sharpness matters) -- it never blocks Navigate or
# Describe (see BLOCKING_REASONS).
BLUR_MIN = 35.0       # variance-of-Laplacian below this  -> too blurry (Read/Money only)
DARK_MIN = 20.0       # mean luma below this              -> too dark (covered/near-black)
BRIGHT_MAX = 245.0    # mean luma above this              -> too bright (fully blown out)

# Reasons that should actually BLOCK a mode. Blur is deliberately excluded here:
# YOLO/Gemini cope with soft frames, and blur scoring is unreliable on compressed
# webcam frames -- so it must not stop Navigate/Describe (callers that care about
# sharpness, like Read, check for "too_blurry" explicitly).
BLOCKING_REASONS = ("too_dark", "too_bright", "no_frame")

# --- enhancement ------------------------------------------------------------ #
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
DARK_GAMMA_TRIGGER = 70.0   # if still this dark after CLAHE, brighten with gamma
DARK_GAMMA = 0.6            # <1 brightens

# --- OCR preprocessing ------------------------------------------------------ #
READ_UPSCALE_MIN_W = 1000   # upscale 2x if the frame is narrower than this
DESKEW_MAX_DEG = 15.0       # ignore implausibly large estimated skews
ADAPT_BLOCK = 31            # adaptive-threshold neighbourhood (odd)
ADAPT_C = 15                # adaptive-threshold constant subtracted

_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def _luma_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def assess(frame):
    """Cheap per-frame quality check. brightness = mean luma; blur_score =
    variance of the Laplacian (low = blurry). `ok` is False with a `reason` of
    too_dark / too_bright / too_blurry / no_frame so callers can skip junk."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"brightness": 0.0, "blur_score": 0.0, "ok": False, "reason": "no_frame"}
    gray = _luma_gray(frame)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < DARK_MIN:
        reason = "too_dark"
    elif brightness > BRIGHT_MAX:
        reason = "too_bright"
    elif blur_score < BLUR_MIN:
        reason = "too_blurry"
    else:
        reason = "ok"
    return {"brightness": round(brightness, 1), "blur_score": round(blur_score, 1),
            "ok": reason == "ok", "reason": reason}


def _apply_gamma(frame, gamma):
    inv = 1.0 / max(gamma, 1e-3)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def enhance(frame, enabled=True):
    """Adaptive-contrast enhance (CLAHE on the L channel) + auto-gamma if dark.
    Returns the frame unchanged when disabled or on any error (never raises)."""
    if not enabled or frame is None or getattr(frame, "size", 0) == 0:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        if float(l.mean()) < DARK_GAMMA_TRIGGER:   # still dark -> brighten
            out = _apply_gamma(out, DARK_GAMMA)
        return out
    except cv2.error:
        return frame


def deskew(gray):
    """Estimate text skew from the orientation of dark pixels and rotate to level.
    Returns (gray, angle_deg). Tiny or implausibly large skews are ignored."""
    try:
        inv = cv2.bitwise_not(gray)
        thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(thr)
        if coords is None:
            return gray, 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        angle = float(angle)
        if abs(angle) < 0.5 or abs(angle) > DESKEW_MAX_DEG:
            return gray, 0.0
        h, w = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        return rot, round(angle, 2)
    except cv2.error:
        return gray, 0.0


def read_preprocess(frame):
    """Prepare a frame for OCR: grayscale -> CLAHE -> 2x upscale (if small) ->
    light deskew. Returns a grayscale image. (Hard binarization is deliberately
    NOT applied here -- it hurts EasyOCR on photographic backgrounds; it's kept in
    binarize() for the low-confidence rescue pass.)"""
    gray = _luma_gray(frame)
    gray = _CLAHE.apply(gray)
    h, w = gray.shape[:2]
    if w < READ_UPSCALE_MIN_W:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray, _ = deskew(gray)
    return gray


def binarize(gray):
    """Adaptive threshold -- a second representation tried in the OCR rescue pass."""
    try:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C)
    except cv2.error:
        return gray


_ROT = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate90(frame, k):
    """Rotate by k*90 degrees (k in 1,2,3). Used to OCR sideways/upside-down text."""
    return cv2.rotate(frame, _ROT[k]) if k in _ROT else frame


def frame_diff(a, b):
    """Mean absolute difference of two frames, normalized 0..1. ~0 means the
    camera is frozen/dead (identical consecutive frames)."""
    if a is None or b is None or getattr(a, "shape", None) != getattr(b, "shape", None):
        return 1.0
    return float(np.mean(cv2.absdiff(a, b))) / 255.0
