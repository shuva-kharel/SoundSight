"""
distance.py  --  Camera-only distance estimation for SoundSight (NO ultrasonic).

Two tiers, profile-gated, degrade gracefully:

  TIER 1  Geometric known-size distance  (Pi + laptop; always available)
          distance_m = real_size_m * focal_px / pixel_box_size, from a per-class
          REAL_HEIGHTS/REAL_WIDTHS table and focal_px derived from the camera FOV
          (or set by --calibrate). Smoothed per ByteTrack id (EMA). Per-class
          reliability flag (rigid known-size = high; deformable/unknown = low).

  TIER 2  Monocular depth model  (laptop only; Pi reaches it via /remote/depth)
          Depth Anything V2-Small (ViT-S, **Apache-2.0**) via the transformers
          depth-estimation pipeline on CUDA/FP16. It outputs RELATIVE (scale-
          ambiguous) depth; we make it METRIC by FUSION: solve a single scale `s`
          so relative*s matches the Tier-1 geometric distance of high-reliability
          anchor objects (robust median), then apply `s` to the whole map -> every
          object (even unknown-size) gets metric distance, and free-space ahead is
          measurable. No anchor visible -> fall back to per-object geometric.

  python distance.py --calibrate   # how to set focal_px / FOV for your camera
"""

import logging
import math

import cv2
import numpy as np

log = logging.getLogger("soundsight.distance")

# =========================================================================== #
# KNOBS (all here)
# =========================================================================== #
CAMERA_HFOV_DEG = 70.0      # horizontal field of view of your webcam (~60-78 typical)
FOCAL_PX = None             # set this to override the FOV-derived focal length (px). --calibrate

# distance tiers (meters) -- replace area_ratio urgency
DIST_VERY_CLOSE = 1.0       # < 1.0 m  -> very close (hazard)
DIST_NEAR = 3.0             # 1.0-3.0 m -> near ; > 3.0 m -> far
UNIT = "meters"             # "meters" | "steps"
STEP_M = 0.75               # 1 step ~= 0.75 m (used when UNIT="steps")
ARM_REACH = 0.7             # < this -> "right in front, reach out"

DIST_EMA = 0.5              # per-track distance smoothing (0..1; higher = snappier)
SCALE_EMA = 0.3             # depth->metric scale smoothing
MAX_TRACKS = 300            # cap the per-track EMA dict

# TIER 2 depth model (laptop only)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"   # Apache-2.0. Alts:
#   "depth-anything/Depth-Anything-V2-Base-hf"  -> more accurate, more VRAM
#   Apple "Depth Pro"                            -> heavier, higher accuracy
DEPTH_INPUT_W = 518         # resize width fed to the depth model (lower = faster/less VRAM)
DEPTH_EVERY_N = 2           # run depth every Nth Navigate frame (raise if VRAM/latency tight)

# Real-world sizes in METERS. Height is used for upright objects; width as a backup.
REAL_HEIGHTS = {
    "person": 1.65, "bicycle": 1.1, "car": 1.5, "motorcycle": 1.1, "bus": 3.1,
    "truck": 3.2, "dog": 0.5, "cat": 0.3, "horse": 1.6, "cow": 1.5,
    "chair": 0.9, "couch": 0.8, "bed": 0.6, "dining table": 0.75, "table": 0.75,
    "door": 2.0, "stairs": 1.0, "bench": 0.85, "pole": 2.5, "traffic light": 0.9,
    "stop sign": 0.75, "fire hydrant": 0.75, "refrigerator": 1.7, "tv": 0.6,
    "laptop": 0.02, "backpack": 0.45, "handbag": 0.3, "suitcase": 0.6,
    "bottle": 0.25, "cup": 0.1, "bowl": 0.07, "potted plant": 0.4, "toilet": 0.7,
    "sink": 0.2, "microwave": 0.3, "oven": 0.7, "umbrella": 0.9,
}
REAL_WIDTHS = {
    "car": 1.8, "bus": 2.55, "truck": 2.5, "motorcycle": 0.8, "door": 0.9,
    "tv": 1.0, "refrigerator": 0.7, "dining table": 1.2, "table": 1.2, "couch": 2.0,
    "bed": 1.5, "laptop": 0.33, "person": 0.5,
}
# rigid, known-size classes -> reliable distance (also used as fusion anchors)
HIGH_RELIABILITY = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle", "door", "stop sign",
    "traffic light", "refrigerator", "tv", "chair", "bench", "fire hydrant",
}


def focal_px(frame_w):
    """Focal length in pixels: explicit FOCAL_PX, else from horizontal FOV + width."""
    if FOCAL_PX:
        return float(FOCAL_PX)
    return (frame_w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)


def geometric_distance(label, box, frame_w):
    """Tier-1 distance (m) for one box, and a reliability flag. None if class unknown."""
    f = focal_px(frame_w)
    h_px = max(box[3] - box[1], 1.0)
    w_px = max(box[2] - box[0], 1.0)
    cands = []
    if label in REAL_HEIGHTS:
        cands.append(REAL_HEIGHTS[label] * f / h_px)
    if label in REAL_WIDTHS:
        cands.append(REAL_WIDTHS[label] * f / w_px)
    if not cands:
        return None, False
    # median of the available estimates; rigid classes are trustworthy
    return float(np.median(cands)), (label in HIGH_RELIABILITY)


def distance_tier(d):
    """meters -> 'very close' | 'near' | 'far' (matches the existing urgency words)."""
    if d is None:
        return None
    if d < DIST_VERY_CLOSE:
        return "very close"
    if d <= DIST_NEAR:
        return "near"
    return "far"


def round_distance_m(d):
    """Round per the SoundSight spec so we never speak raw floats:
      < 0.5 m -> None (caller says 'very close')
      < 2.0 m -> nearest 0.25 m
      >= 2.0 m -> nearest 0.5 m
    """
    if d is None or d < 0.5:
        return None
    step = 0.25 if d < 2.0 else 0.5
    return round(d / step) * step


def _fmt_m(r):
    """Format a rounded metre value: '1', '1.25', '2.5' (no trailing .0)."""
    r = round(r, 2)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.2f}".rstrip("0").rstrip(".")


def spoken_distance(d):
    """Human, HONEST phrasing: rounded meters or steps. '' if unknown.
    NEVER a raw float -- rounded to 0.25 m (<2 m) / 0.5 m (>=2 m); 'very close'
    under 0.5 m; 'right in front' within arm's reach."""
    if d is None:
        return ""
    if UNIT == "steps":
        n = max(1, round(d / STEP_M))
        return f"about {n} step" + ("s" if n != 1 else "")
    if d < 0.5:
        return "very close"
    if d < ARM_REACH:
        return "right in front"
    r = round_distance_m(d)
    if r is None:
        return "very close"
    return f"about {_fmt_m(r)} meter" + ("s" if abs(r - 1.0) > 1e-9 else "")


def nearest_with_distance(detections):
    """The detection with the smallest known distance_m, or None."""
    cand = [d for d in (detections or []) if d.get("distance_m") is not None]
    return min(cand, key=lambda d: d["distance_m"]) if cand else None


def spoken_distance_ne(d):
    """Nepali (Devanagari) distance phrase, same rounding as spoken_distance(). '' if unknown."""
    if d is None:
        return ""
    dev = lambda s: str(s).translate(str.maketrans("0123456789", "०१२३४५६७८९"))
    if UNIT == "steps":
        return f"करिब {dev(max(1, round(d / STEP_M)))} पाइला"
    if d < 0.5:
        return "धेरै नजिक"
    if d < ARM_REACH:
        return "ठीक अगाडि"
    r = round_distance_m(d)
    return f"करिब {dev(_fmt_m(r)) if r is not None else ''} मिटर".strip()


def _is_clipped(box, frame_w, frame_h, margin=3):
    """True if the box touches a frame edge -- the object continues out of view, so a
    known-size geometric estimate (which needs the whole object) is unreliable."""
    x1, y1, x2, y2 = box
    return (x1 <= margin or y1 <= margin
            or x2 >= frame_w - margin or y2 >= frame_h - margin)


def _proximity_cap(area_ratio, clipped):
    """Upper bound on distance (m) implied by how much of the frame a CLIPPED box
    fills. A clipped box that fills the view is physically close, so this corrects the
    geometric over-estimate at close range. Returns None when no cap applies."""
    if not clipped or area_ratio is None:
        return None
    if area_ratio >= 0.50:
        return 0.4     # fills the view -> within arm's reach
    if area_ratio >= 0.32:
        return 0.8
    if area_ratio >= 0.18:
        return 1.5
    return None


def _median_depth(depth, box):
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    y2 = min(depth.shape[0], y2)
    x2 = min(depth.shape[1], x2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    return float(np.median(roi)) if roi.size else None


class GeometricDistance:
    """Tier-1 estimator with per-ByteTrack-id EMA smoothing."""

    def __init__(self):
        self._ema = {}   # track_id -> smoothed distance

    def estimate(self, det, frame_w):
        d, reliable = geometric_distance(det["label"], det["box"], frame_w)
        if d is None:
            return None, False
        tid = det.get("track_id")
        if tid is not None:
            prev = self._ema.get(tid)
            d = d if prev is None else (DIST_EMA * d + (1 - DIST_EMA) * prev)
            self._ema[tid] = d
            if len(self._ema) > MAX_TRACKS:        # bound memory
                for k in list(self._ema)[:-MAX_TRACKS]:
                    del self._ema[k]
        return d, reliable


class DepthEstimator:
    """Tier-2 Depth Anything V2 (LAPTOP ONLY). Lazy, FP16. Returns a relative-depth
    map (higher value = NEARER, i.e. inverse-depth) resized to the frame."""

    def __init__(self, model=DEPTH_MODEL):
        self.model = model
        self.pipe = None
        self._tried = False

    def available(self):
        return self._load() is not None

    def _load(self):
        if not self._tried:
            self._tried = True
            try:
                import torch
                from transformers import pipeline
                self.pipe = pipeline("depth-estimation", model=self.model,
                                     device=0, torch_dtype=torch.float16)
                log.info("Depth: loaded %s on GPU (FP16, Apache-2.0)", self.model)
            except Exception as exc:
                log.warning("Depth model unavailable (%s) -- Tier-2 disabled, using geometric.", exc)
                self.pipe = None
        return self.pipe

    def relative_depth(self, frame_bgr):
        pipe = self._load()
        if pipe is None:
            return None
        try:
            from PIL import Image
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if w > DEPTH_INPUT_W:                  # downscale for speed/VRAM
                rgb = cv2.resize(rgb, (DEPTH_INPUT_W, int(h * DEPTH_INPUT_W / w)))
            out = pipe(Image.fromarray(rgb))
            depth = out["predicted_depth"]
            depth = depth.squeeze().float().cpu().numpy()  # relative; higher = nearer
            return cv2.resize(depth, (w, h))               # back to frame size
        except Exception as exc:
            log.warning("Depth inference failed (%s) -- using geometric this frame.", exc)
            return None


class DistanceEstimator:
    """Orchestrates Tier 1 (+ Tier 2 fusion on the laptop). One per session.

    annotate(detections, frame, frame_idx) sets det['distance_m'] and
    det['dist_source'] ('depth'|'geom') on every detection, updates self.free_space_m
    (laptop depth only), and returns the metric scale `s` in use (or None)."""

    def __init__(self, profile="laptop", use_depth=True):
        self.geo = GeometricDistance()
        self.depth = DepthEstimator() if (profile == "laptop" and use_depth) else None
        self.scale = None             # smoothed depth->metric scale factor s
        self.free_space_m = None
        self._frame = 0

    def annotate(self, detections, frame, frame_idx=None):
        self._frame += 1
        frame_h, frame_w = frame.shape[:2]
        # Tier 1: geometric for everyone (always)
        for d in detections:
            dm, rel = self.geo.estimate(d, frame_w)
            # CLOSE-RANGE FIX: geometric assumes the WHOLE object is visible. A box that
            # fills/clips the frame is only partly visible -> the height formula badly
            # OVER-estimates (e.g. a face 30 cm away reads ~2 m). When the box is clipped
            # at a frame edge AND large, cap the distance by how much of the view it fills.
            clipped = _is_clipped(d["box"], frame_w, frame_h)
            d["_clipped"] = clipped
            if dm is not None:
                cap = _proximity_cap(d.get("area_ratio"), clipped)
                if cap is not None and dm > cap:
                    dm, rel = cap, False    # capped distance isn't a trustworthy anchor
            d["distance_m"] = round(dm, 2) if dm is not None else None
            d["dist_reliable"] = rel
            d["dist_source"] = "geom" if dm is not None else None

        # Tier 2 (laptop): run depth every Nth frame, fuse to metric
        if self.depth is None or (self._frame % DEPTH_EVERY_N != 0):
            return self.scale
        rel = self.depth.relative_depth(frame)
        if rel is None:
            return self.scale

        # fuse: s = median( geometric_distance * relative_depth ) over reliable anchors.
        # Skip CLIPPED boxes -- their geometric distance is unreliable and would poison s.
        ratios = []
        for d in detections:
            if not d.get("dist_reliable") or d.get("_clipped"):
                continue
            rd = _median_depth(rel, d["box"])
            if rd and rd > 1e-6 and d["distance_m"]:
                ratios.append(d["distance_m"] * rd)     # distance = s / rel  ->  s = distance*rel
        if ratios:
            s = float(np.median(ratios))
            self.scale = s if self.scale is None else (SCALE_EMA * s + (1 - SCALE_EMA) * self.scale)

        if self.scale:
            # metric distance for EVERY object from the scaled map (even unknown-size)
            for d in detections:
                rd = _median_depth(rel, d["box"])
                if rd and rd > 1e-6:
                    d["distance_m"] = round(self.scale / rd, 2)
                    d["dist_source"] = "depth"
            self.free_space_m = self._free_space(rel)
        return self.scale

    def _free_space(self, rel):
        """Clear distance straight ahead: nearest thing in the center-bottom path."""
        h, w = rel.shape[:2]
        region = rel[int(h * 0.55):h, int(w * 0.4):int(w * 0.6)]   # where you'd step
        if region.size == 0 or not self.scale:
            return None
        nearest_rel = float(np.percentile(region, 90))   # robust 'closest' in the path
        return round(self.scale / nearest_rel, 2) if nearest_rel > 1e-6 else None

    def free_space_phrase(self):
        d = self.free_space_m
        if d is None:
            return None
        if d < DIST_VERY_CLOSE:
            return (f"Obstacle ahead, about {round(d, 1)} meters.",
                    f"अगाडि अवरोध, करिब {round(d, 1)} मिटर।", True)
        return (f"Path clear about {round(d)} meters ahead.",
                f"बाटो करिब {round(d)} मिटर सम्म खाली छ।", False)


def _calibrate_help():
    print("""
=========================  CAMERA DISTANCE CALIBRATION  =======================
Tier-1 distance needs the camera's focal length in pixels (focal_px). Two ways:

  A) From field-of-view (default, no measuring):
       set CAMERA_HFOV_DEG in distance.py to your webcam's horizontal FOV
       (check the spec sheet; common laptop/USB cams are ~65-78 deg).
       focal_px is then computed from the frame width automatically.

  B) One-shot measurement (more accurate):
       1. Put a person (1.65 m) or a known-width object exactly D meters away.
       2. Note the object's pixel HEIGHT h (or width) in a 'frame_w'-wide frame.
       3. focal_px = (pixel_size * D) / real_size_m
          e.g. person 1.65 m at 3.0 m showing 250 px tall ->
               focal_px = 250 * 3.0 / 1.65 = 454
       4. set FOCAL_PX = 454 in distance.py.

Then distances read out as 'about N meters'. The depth model (laptop) refines them.
==============================================================================
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SoundSight camera distance estimation")
    ap.add_argument("--calibrate", action="store_true", help="how to set focal_px / FOV")
    args = ap.parse_args()
    if args.calibrate:
        _calibrate_help()
    else:
        print("Tier-1 focal_px @640 wide =", round(focal_px(640)), "px (HFOV=%.0f deg)" % CAMERA_HFOV_DEG)
        print("Run with --calibrate for setup help.")
