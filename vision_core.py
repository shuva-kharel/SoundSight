"""
vision_core.py  --  Pure AI / vision logic for SoundSight.

Hardware-agnostic and contains NO web code, so it ports to a Raspberry Pi
unchanged. The web layer (server.py) only does transport.

The whole detection pipeline lives behind VisionCore.detect_and_rank(frame):
    model inference -> allowlist + part removal -> per-class confidence ->
    part-containment suppression -> temporal smoothing -> priority ranking.
server.py calls VisionCore.detect() (an alias) for the boxes and
select_announcements() for the speech, so server.py never has to change.

------------------------------------------------------------------------------
HOW TO TUNE
------------------------------------------------------------------------------
* Switch detectors:    set MODEL_MODE below to "coco" | "openvocab" | "oiv7".
* Detect everything:    DETECT_ALL=True (default) keeps every class the model
                        knows; NAVIGATION_CLASSES then only prioritizes ranking.
* Tighten/loosen:       edit CONF_THRESHOLDS / DEFAULT_CONF (per-class floors).
* Body parts on people: removed by geometry (a small box inside a person box),
                        not by a name list -- see CONTAINMENT_IOS / _MAX_AREA.
* See what's happening: every 15th frame the console prints RAW (before filter)
                        vs KEPT (after), plus the full class list on startup.
"""

import logging
import os
import platform
import shutil
import subprocess
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import frame_quality as fq

log = logging.getLogger("soundsight.vision")

# Pick the best available device once, at import time, and announce it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Raspberry Pi / ARM detection -> auto-run the light path (NCNN nano, cpu, imgsz 320)
# and NEVER pass device="cuda" on the Pi. See VisionCore.__init__.
IS_ARM = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")

# =========================================================================== #
# FEATURE PROFILE  --  auto-detected; gates HEAVY (laptop GPU) vs LIGHT (Pi) models
# =========================================================================== #
# "laptop" = a real GPU is present -> heavy, accurate models (used by the laptop app
#            AND the compute server the Pi offloads to).
# "pi"     = ARM / no CUDA -> light models only; heavy models NEVER load here.
# Switching profiles needs NO code edits -- it's purely auto-detected here.
FEATURE_PROFILE = "laptop" if (DEVICE == "cuda" and not IS_ARM) else "pi"

# Per-feature model config. Dial size up/down here; lighter alternatives noted.
# (The Pi's own models are fixed below and must not change.)
LAPTOP_DETECT_MODEL = "yolo11m"   # heavy Navigate/detect. Lighter: "yolo11s"; bigger: "yolo11l" (8GB ok)
LAPTOP_DETECT_IMGSZ = 640
LAPTOP_DETECT_HALF = True         # FP16 on GPU: faster + ~half the VRAM
LAPTOP_OCR_GPU = True             # EasyOCR on GPU (heavier models). PaddleOCR is an optional upgrade.
LAPTOP_FACE_MODEL = "buffalo_l"   # InsightFace full model on GPU
LAPTOP_VLM_MODEL = "llava"        # local Ollama VLM when offline; lighter: "moondream"; bigger: "qwen2-vl"
PI_DETECT_ACCURACY = "fast"       # yolo11n NCNN @320 on CPU -- DO NOT change (the Pi's safety path)

# Frame width used by the zone logic. detect_and_rank() keeps it in sync with the
# actual frame it processes.
FRAME_W = 416


def zone_for(cx):
    if cx < FRAME_W / 3:
        return "on your left"
    if cx > 2 * FRAME_W / 3:
        return "on your right"
    return "ahead"


# =========================================================================== #
# DETECTION CONFIG
# =========================================================================== #
# "coco"      -> YOLO11 .pt (80 COCO classes) at the MODEL_ACCURACY size. Default.
# "coco-ncnn" -> the same model exported to NCNN -- much faster on the Raspberry
#                Pi's ARM CPU. Used by pi_app.py (with accuracy="fast" = nano).
# "openvocab" -> YOLO-World (yolov8s-worldv2.pt) restricted to OPENVOCAB_CLASSES.
#                Falls back to "coco" if the weights/deps aren't available.
# "oiv7"      -> yolov8s-oiv7.pt (~600 classes).
MODEL_MODE = "coco"

# Speed/accuracy of the COCO model. Tradeoff: nano is ~2-3x faster than medium but
# misses more small/distant objects. Pi stays on "fast" (nano, NCNN).
#   "fast" -> yolo11n (Pi / NCNN) | "balanced" -> yolo11s (laptop default) |
#   "accurate" -> yolo11m (laptop GPU only)
MODEL_ACCURACY = "balanced"
ACCURACY_WEIGHTS = {"fast": "yolo11n", "balanced": "yolo11s", "accurate": "yolo11m"}
# Per-tier inference resolution and test-time augmentation (TTA). Smaller imgsz =
# faster (Pi/NCNN runs "fast" at 320). "accurate" turns on augment=True (TTA): a
# real accuracy bump but ~2-3x slower -- laptop GPU only, NEVER the Pi.
ACCURACY_IMGSZ = {"fast": 320, "balanced": 640, "accurate": 640}
ACCURACY_AUGMENT = {"fast": False, "balanced": False, "accurate": True}

IMG_SIZE = 640        # default inference resolution (per-tier value overrides this)
PREDICT_CONF = 0.25   # model floor: surface everything >=0.25 so the debug log shows
                      # near-threshold objects; the per-class thresholds do the real cut.

# Detect EVERYTHING the model knows. When True, there is NO hand-written allowlist
# -- the model's own class list is the allowlist, so common objects (bottle, cup,
# chair, bag, phone...) are never dropped. (A too-narrow allowlist + high
# thresholds is exactly what made it announce only "person".) NAVIGATION_CLASSES
# below is then used ONLY to prioritize ranking, never to drop a detection.
DETECT_ALL = True

# Open-vocabulary prompt for "openvocab" mode -- exactly the things a blind user
# needs to walk around. Edit freely; YOLO-World detects whatever you list.
OPENVOCAB_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "door", "stairs",
    "chair", "table", "sofa", "bed", "pole", "wall", "fence", "dog", "cat",
    "traffic light", "stop sign", "bench", "backpack", "bottle", "cup", "laptop",
    "tv", "refrigerator", "sink", "toilet", "potted plant", "curb", "pothole",
    "obstacle",
]

# Priority classes (canonical, lowercase). Used ONLY to rank what to announce
# first -- NEVER to drop detections (see DETECT_ALL). Edit freely.
NAVIGATION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "door", "stairs",
    "chair", "table", "couch", "bed", "pole", "wall", "fence", "dog", "cat",
    "traffic light", "stop sign", "bench", "backpack", "bottle", "cup", "laptop",
    "tv", "refrigerator", "sink", "toilet", "plant", "curb", "pothole", "obstacle",
}

# Nicer spoken words ONLY. This never drops a class: a label with no entry here
# just keeps its own lowercased name (so "bottle", "cup", "chair" pass untouched).
SYNONYMS = {
    "sofa": "couch",
    "dining table": "table",
    "potted plant": "plant",
    "houseplant": "plant",
    "television": "tv",
    "coffee cup": "cup",
    "mug": "cup",
    "cell phone": "phone",
    # Open Images labels people as Man/Woman/Boy/Girl -- group to "person".
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
}

# Body parts worn on a person ("glasses on a face") are removed PURELY BY GEOMETRY
# in detect_and_rank -- a small box mostly inside a larger person box. There is no
# name blocklist, so a standalone bottle/cup/bag in the scene is never dropped.
CONTAINMENT_IOS = 0.70       # box must be >70% inside the person box, AND
CONTAINMENT_MAX_AREA = 0.30  # ...be smaller than 30% of that person's area

# Per-class confidence floors. Default is LOW so everyday objects appear; only
# people and vehicles are nudged up slightly. Unlisted classes inherit 0.35.
DEFAULT_CONF = 0.35
CONF_THRESHOLDS = {
    "person": 0.40,
    "car": 0.40, "bus": 0.40, "truck": 0.40, "motorcycle": 0.40, "bicycle": 0.40,
}

# Importance for ranking what to *announce* (higher = more important).
DEFAULT_IMPORTANCE = 1
IMPORTANCE = {
    "person": 3, "car": 3, "bus": 3, "truck": 3, "motorcycle": 3, "bicycle": 3,
    "stairs": 3, "pole": 3, "curb": 3, "pothole": 3, "obstacle": 3,
    "door": 2, "wall": 2, "fence": 2, "bench": 2, "dog": 2, "cat": 2,
    "traffic light": 2, "stop sign": 2,
}


# --------------------------------------------------------------------------- #
# Proximity / urgency
# --------------------------------------------------------------------------- #
FAR_MAX = 0.05          # area_ratio < 0.05          -> "far"
NEAR_MAX = 0.20         # 0.05 <= area_ratio <= 0.20 -> "near"
                        # area_ratio > 0.20          -> "very close"
URGENCY_FAR = "far"
URGENCY_NEAR = "near"
URGENCY_VERY_CLOSE = "very close"
URGENCY_RANK = {URGENCY_FAR: 0, URGENCY_NEAR: 1, URGENCY_VERY_CLOSE: 2}

APPROACH_MARGIN = 0.01  # area_ratio growth needed to count as "approaching"

# CLASS-AWARE urgency. area_ratio alone is misleading: a cup filling 20% of the
# frame is NOT the hazard a person filling 20% is. We scale each object's
# area_ratio by its typical real-world importance BEFORE classifying urgency, so
# "very close" warnings are calibrated -- a nearby person/vehicle/door fires, a
# nearby bottle/phone does not. >1 makes a class reach "very close" sooner; <1
# makes it need to fill much more of the frame. Tune freely.
URGENCY_SCALE = {
    # hazards: easy to trigger
    "person": 1.3, "car": 1.4, "bus": 1.4, "truck": 1.4, "motorcycle": 1.3,
    "bicycle": 1.2, "dog": 1.2, "stairs": 1.4, "pole": 1.3, "door": 1.1,
    "bench": 1.0, "fence": 1.0, "wall": 1.0, "obstacle": 1.3, "curb": 1.2,
    "pothole": 1.2, "traffic light": 1.0, "stop sign": 1.0,
    # small handheld / tabletop things: de-weighted so they aren't "hazards"
    "bottle": 0.4, "cup": 0.4, "phone": 0.4, "remote": 0.3, "mouse": 0.3,
    "book": 0.4, "clock": 0.4, "scissors": 0.3, "keyboard": 0.5, "laptop": 0.6,
    "bowl": 0.4, "spoon": 0.3, "fork": 0.3, "knife": 0.5, "banana": 0.3,
    "apple": 0.3, "orange": 0.3, "toothbrush": 0.3, "cell phone": 0.4,
}
DEFAULT_URGENCY_SCALE = 0.8   # unknown classes: slightly de-weighted vs. a person


def urgency_scale_of(label):
    return URGENCY_SCALE.get(label, DEFAULT_URGENCY_SCALE)


def classify_urgency(area_ratio):
    if area_ratio > NEAR_MAX:
        return URGENCY_VERY_CLOSE
    if area_ratio >= FAR_MAX:
        return URGENCY_NEAR
    return URGENCY_FAR


def _area(box_or_det):
    box = box_or_det["box"] if isinstance(box_or_det, dict) else box_or_det
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def _ios(inner, outer):
    """Intersection over `inner`'s own area: how much of `inner` sits inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / a if a > 0 else 0.0


def conf_threshold(canonical):
    return CONF_THRESHOLDS.get(canonical, DEFAULT_CONF)


def importance_of(canonical):
    return IMPORTANCE.get(canonical, DEFAULT_IMPORTANCE)


# --------------------------------------------------------------------------- #
# Announcement tuning (event-driven announcer -- see AnnouncementManager)
# --------------------------------------------------------------------------- #
REFRESH_SILENCE = 8.0   # restate the single most important object after this many silent seconds
ZONE_DEADBAND   = 0.08  # fraction of frame width past the 1/3 or 2/3 line before a zone flip sticks
URGENCY_MARGIN  = 0.20  # area_ratio must cross a tier boundary by this fraction before the tier flips
MAX_PER_CYCLE   = 2     # speak at most this many events per frame
FORGET          = 8     # drop per-track / per-group memory after this many missing frames
MIN_GAP         = 1.0   # min seconds between (non-urgent) speech, so events don't talk over each other

# When very close, these are genuine hazards and take the first speaking slot.
HAZARD_CLASSES = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "stairs", "pole", "obstacle", "curb", "pothole",
}

ANNOUNCE_COOLDOWN = 2.5    # legacy Announcer only
VERY_CLOSE_COOLDOWN = 1.0  # legacy Announcer only

# PATH weighting: objects in the walking path (center third AND lower half of the
# frame) matter more than ones up high or off to the edges. Adds 0..PATH_WEIGHT*2
# to an object's ranking so the thing you're about to walk into is announced first.
PATH_WEIGHT = 1

# --------------------------------------------------------------------------- #
# Robustness / performance knobs (shared by server.py + pi_app.py)
# --------------------------------------------------------------------------- #
ENHANCE_FRAMES = True       # run frame_quality.enhance() before inference (auto-off under load)
FPS_FLOOR = 4.0             # Pi: below this, drop imgsz and disable enhance until it recovers
MAX_TRACKS = 200            # hard cap on per-track/-group memory (long-session safety)

# "Camera view is dark/unclear" guidance: only after the view stays bad this long,
# and at most once per cooldown, so it guides without nagging.
VIEW_BAD_SECONDS = 2.0
VIEW_WARN_COOLDOWN = 8.0
VIEW_DARK_SPEECH = ("Camera view is dark", "क्यामेरा दृश्य अँध्यारो छ")
VIEW_BRIGHT_SPEECH = ("Camera view is too bright", "क्यामेरा दृश्य अति उज्यालो छ")
VIEW_BLUR_SPEECH = ("Camera view is unclear", "क्यामेरा दृश्य अस्पष्ट छ")
_VIEW_SPEECH = {"too_dark": VIEW_DARK_SPEECH, "too_bright": VIEW_BRIGHT_SPEECH,
                "too_blurry": VIEW_BLUR_SPEECH, "no_frame": VIEW_DARK_SPEECH}


class QualityGate:
    """Per-session frame-quality gate (one per Navigate connection / Pi loop).

    check(frame, now) -> (assessment, speak_or_None). The caller should SKIP
    detection only when assessment["blocking"] is True (genuinely unusable: dark /
    blown-out / no frame -- NOT merely soft-focus, which YOLO handles fine). If a
    BLOCKING view persists VIEW_BAD_SECONDS, a single gentle bilingual warning is
    returned (rate-limited), so it guides without nagging on normal frames.
    """

    def __init__(self, bad_seconds=VIEW_BAD_SECONDS, cooldown=VIEW_WARN_COOLDOWN):
        self.bad_seconds = bad_seconds
        self.cooldown = cooldown
        self.bad_since = None
        self.last_warn = -1e9

    def check(self, frame, now):
        a = dict(fq.assess(frame))
        a["blocking"] = a["reason"] in fq.BLOCKING_REASONS   # blur is NOT blocking
        if not a["blocking"]:
            self.bad_since = None
            return a, None
        if self.bad_since is None:
            self.bad_since = now
        speak = None
        if (now - self.bad_since) >= self.bad_seconds and (now - self.last_warn) >= self.cooldown:
            en, ne = _VIEW_SPEECH.get(a["reason"], VIEW_DARK_SPEECH)
            speak = {"text": en, "text_ne": ne, "rate": 1.0, "urgent": False, "urgency": URGENCY_NEAR}
            self.last_warn = now
        return a, speak


class Announcer:
    """LEGACY time-based cooldown -- superseded by AnnouncementManager (kept for
    reference). Prevents repeating the same object+zone within `cooldown` seconds.
    """

    def __init__(self, cooldown=ANNOUNCE_COOLDOWN):
        self.cooldown = cooldown
        self.last = {}

    def consider(self, label, zone, now, cooldown=None):
        cd = self.cooldown if cooldown is None else cooldown
        key = (label, zone)
        if now - self.last.get(key, -999) >= cd:
            self.last[key] = now
            return f"{label} {zone}"
        return None


# ----- natural-language helpers -------------------------------------------- #
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_IRREGULAR_PLURALS = {"person": "people", "man": "men", "woman": "women",
                      "child": "children", "mouse": "mice", "foot": "feet"}


def _number_word(n):
    return _NUMBER_WORDS.get(n, str(n))


def _pluralize(name, n):
    if n == 1:
        return name
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def _phrase(name, zone, urgency, count, approaching=False):
    """Natural wording for one (possibly multi-member) group.
    Returns (text, rate, urgent) -- same fields select_announcements always had."""
    # A very-close warning is about the single nearest hazard, not the count.
    # (Comma rather than an em-dash: safe for espeak-ng + Windows console logs.)
    if urgency == URGENCY_VERY_CLOSE:
        if zone == "ahead":
            return f"Careful, {name} right in front", 1.3, True
        return f"Careful, {name} {zone}", 1.3, True
    head = name if count == 1 else f"{_number_word(count)} {_pluralize(name, count)}"
    if approaching:
        return f"{head} {zone}, getting closer", 1.0, False
    if urgency == URGENCY_NEAR:
        return f"{head} {zone}, close", 1.0, False
    return f"{head} {zone}", 1.0, False


# --------------------------------------------------------------------------- #
# Nepali phrasing  (used when the browser's Language toggle is set to नेपाली).
# We build a Nepali string for every announcement *in parallel* with the English
# one (see _phrase) and ship both as {text, text_ne}; the browser picks which to
# speak. Object names not in NE_NAMES fall back to their English label -- gTTS
# will still voice it, just imperfectly -- so an unmapped class is never dropped.
# --------------------------------------------------------------------------- #
NE_NAMES = {
    "person": "मानिस", "bicycle": "साइकल", "car": "गाडी", "motorcycle": "मोटरसाइकल",
    "bus": "बस", "truck": "ट्रक", "train": "रेल", "boat": "डुङ्गा", "airplane": "हवाईजहाज",
    "chair": "कुर्सी", "table": "टेबल", "couch": "सोफा", "bed": "ओछ्यान",
    "dog": "कुकुर", "cat": "बिरालो", "bird": "चरा", "horse": "घोडा", "cow": "गाई",
    "sheep": "भेडा", "elephant": "हात्ती", "bear": "भालु",
    "bottle": "बोतल", "cup": "कप", "bowl": "कचौरा", "knife": "चक्कु",
    "spoon": "चम्चा", "fork": "काँटा", "laptop": "ल्यापटप", "tv": "टिभी",
    "phone": "फोन", "keyboard": "किबोर्ड", "mouse": "माउस", "remote": "रिमोट",
    "book": "किताब", "clock": "घडी", "scissors": "कैंची",
    "backpack": "झोला", "handbag": "ब्याग", "suitcase": "सुटकेस", "umbrella": "छाता",
    "door": "ढोका", "stairs": "भर्‍याङ", "bench": "बेन्च", "pole": "खम्बा",
    "wall": "पर्खाल", "fence": "बार", "traffic light": "ट्राफिक बत्ती",
    "stop sign": "स्टप साइन", "refrigerator": "फ्रिज", "sink": "सिंक",
    "toilet": "शौचालय", "plant": "बिरुवा",
}
NE_ZONES = {"ahead": "अगाडि", "on your left": "बायाँतिर", "on your right": "दायाँतिर"}
NE_NUMS = {2: "दुई", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छ", 7: "सात", 8: "आठ", 9: "नौ"}


def _phrase_ne(name, zone, urgency, count, approaching=False):
    """Nepali (Devanagari) wording mirroring _phrase(); returns just the string."""
    nm = NE_NAMES.get(name, name)
    z = NE_ZONES.get(zone, zone)
    if urgency == URGENCY_VERY_CLOSE:
        if zone == "ahead":
            return f"सावधान, {nm} ठीक अगाडि"
        return f"सावधान, {nm} {z}"
    head = nm if count == 1 else f"{NE_NUMS.get(count, count)} {nm}"
    if approaching:
        return f"{head} {z}, नजिक आउँदै"
    if urgency == URGENCY_NEAR:
        return f"{head} {z}, नजिक"
    return f"{head} {z}"


class _TrackState:
    """Debounced per-object memory (one per stable track_id)."""
    __slots__ = ("label", "zone", "urgency", "area", "uarea", "path",
                 "approaching", "last_seen")

    def __init__(self, label, zone, urgency, area, uarea, path, last_seen):
        self.label, self.zone, self.urgency = label, zone, urgency
        self.area, self.uarea, self.path = area, uarea, path
        self.approaching, self.last_seen = False, last_seen


class _GroupState:
    """What we've already announced about a (label, zone) group."""
    __slots__ = ("introduced", "urgency", "ids", "approaching_announced", "last_seen")

    def __init__(self):
        self.introduced = False
        self.urgency = URGENCY_FAR
        self.ids = set()
        self.approaching_announced = False
        self.last_seen = 0


class AnnouncementManager:
    """Event-driven announcer with persistent object tracking. One per connection.

    Speaks about a group ONLY on an event -- first sight (NEW), urgency ESCALATION,
    a new member / count CHANGE, an APPROACH, or a REFRESH after REFRESH_SILENCE
    seconds of silence. A stable, unchanging object is never repeated. Per-object
    zone and urgency are debounced (hysteresis) so an object hovering on a boundary
    doesn't flip-flop. This replaces the time-cooldown Announcer.
    """

    def __init__(self):
        self.tracks = {}   # track_key -> _TrackState  (track_key = track_id, or (label,zone) fallback)
        self.groups = {}   # (label, zone) -> _GroupState
        self.frame = 0
        self.last_spoken = -1e9

    # --- hysteresis (deadbands) ------------------------------------------- #
    def _debounce_zone(self, current, cx):
        fw = FRAME_W or 1
        left_b, right_b, db = fw / 3.0, 2 * fw / 3.0, ZONE_DEADBAND * fw
        if current == "ahead":
            if cx < left_b - db:
                return "on your left"
            if cx > right_b + db:
                return "on your right"
            return "ahead"
        if current == "on your left":
            if cx > right_b + db:
                return "on your right"
            if cx > left_b + db:
                return "ahead"
            return "on your left"
        if current == "on your right":
            if cx < left_b - db:
                return "on your left"
            if cx < right_b - db:
                return "ahead"
            return "on your right"
        return zone_for(cx)

    def _debounce_urgency(self, current, area_ratio):
        # Tier only changes once area_ratio clears a boundary by URGENCY_MARGIN, so
        # jitter near 0.05 / 0.20 doesn't re-trigger. Can jump tiers (e.g. far ->
        # very close) if something appears suddenly close.
        cur = URGENCY_RANK[current]
        # highest tier justified by the upper boundaries (+margin)
        if area_ratio > NEAR_MAX * (1 + URGENCY_MARGIN):
            up = URGENCY_VERY_CLOSE
        elif area_ratio > FAR_MAX * (1 + URGENCY_MARGIN):
            up = URGENCY_NEAR
        else:
            up = None
        if up is not None and URGENCY_RANK[up] > cur:
            return up
        # lowest tier justified by the lower boundaries (-margin)
        if area_ratio < FAR_MAX / (1 + URGENCY_MARGIN):
            down = URGENCY_FAR
        elif area_ratio < NEAR_MAX / (1 + URGENCY_MARGIN):
            down = URGENCY_NEAR
        else:
            down = None
        if down is not None and URGENCY_RANK[down] < cur:
            return down
        return current

    # --- per-frame processing --------------------------------------------- #
    def process(self, detections, approaching, now):
        self.frame += 1
        f = self.frame

        # 1) update debounced per-track state
        for det in detections:
            key = det.get("track_id")
            if key is None:
                key = (det["label"], zone_for(det["cx"]))    # predict() fallback keying
            raw_zone = zone_for(det["cx"])
            ar = det["area_ratio"]
            uar = det.get("urgency_area", ar)        # class-aware area for urgency
            path = det.get("path_score", 0)          # 0..2: in the walking path?
            appr = (det["label"], raw_zone) in approaching
            t = self.tracks.get(key)
            if t is None:
                t = _TrackState(det["label"], raw_zone, classify_urgency(uar), ar, uar, path, f)
                self.tracks[key] = t
            else:
                t.zone = self._debounce_zone(t.zone, det["cx"])
                t.urgency = self._debounce_urgency(t.urgency, uar)
                t.label, t.area, t.uarea, t.path = det["label"], ar, uar, path
            t.approaching, t.last_seen = appr, f

        # 2) forget stale tracks (+ hard cap so a long session can't grow unbounded)
        for k in [k for k, t in self.tracks.items() if f - t.last_seen >= FORGET]:
            del self.tracks[k]
        if len(self.tracks) > MAX_TRACKS:
            for k in sorted(self.tracks, key=lambda k: self.tracks[k].last_seen)[:-MAX_TRACKS]:
                del self.tracks[k]

        # 3) group ALIVE tracks by (label, debounced zone); nearest member drives urgency
        groups = {}
        for tkey, t in self.tracks.items():
            g = groups.setdefault((t.label, t.zone),
                                  {"ids": set(), "area": -1.0, "urgency": URGENCY_FAR,
                                   "approaching": False, "path": 0})
            g["ids"].add(tkey)
            if t.area > g["area"]:
                g["area"], g["urgency"] = t.area, t.urgency
            if t.approaching:
                g["approaching"] = True
            g["path"] = max(g["path"], t.path)

        # 4) detect events (+ silent bookkeeping so a later rise re-fires)
        candidates = []
        for gk, g in groups.items():
            label, _zone = gk
            gs = self.groups.get(gk)
            if gs is None:
                gs = _GroupState()
                self.groups[gk] = gs
            gs.last_seen = f
            cur, prev = URGENCY_RANK[g["urgency"]], URGENCY_RANK[gs.urgency]
            if not gs.introduced:
                event = "new"
            elif cur > prev:
                event = "escalation"
            elif g["ids"] - gs.ids:                 # a genuinely new member joined
                event = "change"
            elif g["approaching"] and not gs.approaching_announced:
                event = "approaching"
            else:
                event = None
            if event is None:                       # silent updates, no speech
                if cur < prev:
                    gs.urgency = g["urgency"]        # de-escalation is silent
                gs.ids &= g["ids"]                   # members that left, silently
                if not g["approaching"]:
                    gs.approaching_announced = False
                continue
            hazard = g["urgency"] == URGENCY_VERY_CLOSE and label in HAZARD_CLASSES
            # rank: hazard first, then urgency, then PATH (in your walking line),
            # then class importance, then apparent size.
            candidates.append(((hazard, cur, PATH_WEIGHT * g["path"],
                                importance_of(label), g["area"]), event, gk, g))

        # forget stale group memory (+ hard cap)
        for gk in [gk for gk, gs in self.groups.items() if f - gs.last_seen >= FORGET]:
            del self.groups[gk]
        if len(self.groups) > MAX_TRACKS:
            for gk in sorted(self.groups, key=lambda k: self.groups[k].last_seen)[:-MAX_TRACKS]:
                del self.groups[gk]

        # 5) rank events, apply min-gap, cap, and commit the ones we speak
        candidates.sort(key=lambda c: c[0], reverse=True)
        if candidates and (now - self.last_spoken) < MIN_GAP and not candidates[0][0][0]:
            candidates = []   # hold a non-urgent burst (a very-close hazard bypasses this)

        spoken, reasons = [], []
        for _rank, event, gk, g in candidates[:MAX_PER_CYCLE]:
            label, zone = gk
            text, rate, urgent = _phrase(label, zone, g["urgency"], len(g["ids"]),
                                         approaching=(event == "approaching"))
            text_ne = _phrase_ne(label, zone, g["urgency"], len(g["ids"]),
                                 approaching=(event == "approaching"))
            spoken.append({"text": text, "text_ne": text_ne, "rate": rate,
                           "urgent": urgent, "urgency": g["urgency"]})
            reasons.append(event)
            gs = self.groups[gk]
            gs.introduced, gs.urgency, gs.ids = True, g["urgency"], set(g["ids"])
            if g["approaching"]:
                gs.approaching_announced = True
            self.last_spoken = now

        # 6) refresh: if we've been silent too long, restate the single top object
        if not spoken and groups and (now - self.last_spoken) >= REFRESH_SILENCE:
            gk, g = max(groups.items(), key=lambda kv: (
                kv[1]["urgency"] == URGENCY_VERY_CLOSE and kv[0][0] in HAZARD_CLASSES,
                URGENCY_RANK[kv[1]["urgency"]], PATH_WEIGHT * kv[1]["path"],
                importance_of(kv[0][0]), kv[1]["area"]))
            label, zone = gk
            text, rate, urgent = _phrase(label, zone, g["urgency"], len(g["ids"]))
            text_ne = _phrase_ne(label, zone, g["urgency"], len(g["ids"]))
            spoken.append({"text": text, "text_ne": text_ne, "rate": rate,
                           "urgent": urgent, "urgency": g["urgency"]})
            reasons.append("refresh")
            self.last_spoken = now

        if spoken:  # observability: WHY each thing was said
            log.info("SPEAK %s", ", ".join(f"[{r}] {s['text']!r}" for r, s in zip(reasons, spoken)))
        return spoken


class ApproachDetector:
    """Flags objects getting closer (area_ratio growing), one per connection.

    PI PORT: replace approaching_objects() with ultrasonic distance deltas --
    keep the signature (detections in, set of (label, zone) keys out).
    """

    def __init__(self, margin=APPROACH_MARGIN):
        self.margin = margin
        self.prev = {}

    def approaching_objects(self, detections):
        current = {}
        for det in detections:
            key = (det["label"], zone_for(det["cx"]))
            current[key] = max(current.get(key, 0.0), det["area_ratio"])
        approaching = {
            key for key, ratio in current.items()
            if key in self.prev and ratio > self.prev[key] + self.margin
        }
        self.prev = current
        return approaching


class TemporalTracker:
    """Confirms objects across frames to kill single-frame flickers/ghosts.

    An object (keyed by class + approximate zone) is "confirmed" only after it
    appears in at least `confirm` of the last `window` frames, and is forgotten
    after `forget` consecutive missing frames. Relaxed to 2-of-3 here because the
    persistent tracker (ByteTrack) already removes single-frame blips upstream.
    """

    def __init__(self, confirm=2, window=3, forget=6):
        self.confirm, self.window, self.forget = confirm, window, forget
        self.frame = 0
        self.seen = {}  # key -> deque of frame indices it was seen in

    def update(self, keys_this_frame):
        self.frame += 1
        f = self.frame
        for k in keys_this_frame:
            dq = self.seen.setdefault(k, deque(maxlen=self.window))
            if not dq or dq[-1] != f:
                dq.append(f)

        confirmed = set()
        for k, dq in list(self.seen.items()):
            if sum(1 for i in dq if i > f - self.window) >= self.confirm:
                confirmed.add(k)
            if dq and (f - dq[-1]) >= self.forget:
                del self.seen[k]
        if len(self.seen) > MAX_TRACKS:   # long-session safety cap
            for k in sorted(self.seen, key=lambda k: self.seen[k][-1])[:-MAX_TRACKS]:
                del self.seen[k]
        return confirmed


# --------------------------------------------------------------------------- #
# Detection pipeline
# --------------------------------------------------------------------------- #
class VisionCore:
    """Loads a detector once and runs the full clean/stable pipeline.

    NOTE: the TemporalTracker state lives here, which assumes a single live
    Navigate client (true for this prototype). For multiple simultaneous clients
    you'd give each connection its own tracker.
    """

    def __init__(self, mode=MODEL_MODE, accuracy=MODEL_ACCURACY):
        self.mode = mode
        self.accuracy = accuracy
        # AUTO LIGHT PATH on the Pi/ARM (no CUDA): force NCNN nano + imgsz 320 so a
        # fresh Pi "just works" even via server.py, and we never try device=cuda.
        if IS_ARM and DEVICE == "cpu" and self.mode == "coco":
            log.info("ARM + no CUDA detected -> light path: coco-ncnn, fast (imgsz 320).")
            self.mode, self.accuracy = "coco-ncnn", "fast"
        # Pi/NCNN must never run the heavy "accurate" (yolo11m + TTA) tier.
        if self.mode == "coco-ncnn" and self.accuracy == "accurate":
            log.warning("Pi/NCNN can't run 'accurate' (yolo11m + TTA) -- using 'fast'.")
            self.accuracy = "fast"
        # PROFILE-based model selection: laptop coco -> HEAVY (yolo11m@640 FP16);
        # everything else (incl. the Pi's coco-ncnn) -> the per-accuracy light model.
        if self.mode == "coco" and FEATURE_PROFILE == "laptop":
            self._stem = LAPTOP_DETECT_MODEL
            self.imgsz = LAPTOP_DETECT_IMGSZ
            self.half = LAPTOP_DETECT_HALF and DEVICE == "cuda"
        else:
            self._stem = ACCURACY_WEIGHTS.get(self.accuracy, "yolo11s")
            self.imgsz = ACCURACY_IMGSZ.get(self.accuracy, IMG_SIZE)
            self.half = False
        self.augment = ACCURACY_AUGMENT.get(self.accuracy, False)
        self.model = self._load_model()
        self.names = self.model.names
        self.tracker = TemporalTracker()
        self._use_tracking = True  # flips to False if track() isn't supported
        self.frame_count = 0
        self.enhance_frames = ENHANCE_FRAMES
        self._last_assess = None
        self._last_enhanced = False
        log.info("Detection: profile=%s, mode=%s, model=%s, imgsz=%d, half=%s, augment=%s, "
                 "device=%s, DETECT_ALL=%s", FEATURE_PROFILE, self.mode, self._stem, self.imgsz,
                 self.half, self.augment, DEVICE.upper(), DETECT_ALL)
        # Print the full class list so you can SEE exactly what the model can find.
        known = sorted(str(n).lower() for n in self.names.values())
        log.info("Model knows %d classes: %s", len(known), ", ".join(known))
        if DEVICE == "cpu":
            log.warning("Running on CPU -- detection will be slower.")

    def _load_model(self):
        if self.mode == "openvocab":
            try:
                from ultralytics import YOLOWorld

                m = YOLOWorld("yolov8s-worldv2.pt")
                m.set_classes(OPENVOCAB_CLASSES)
                log.info("Loaded YOLO-World with %d navigation classes", len(OPENVOCAB_CLASSES))
                return m
            except Exception as exc:
                log.warning("openvocab unavailable (%s) -- falling back to coco", exc)
                self.mode = "coco"
        if self.mode == "oiv7":
            return YOLO("yolov8s-oiv7.pt")

        stem = self._stem
        if self.mode == "coco-ncnn":
            ncnn_dir = f"{stem}_ncnn_model"  # NCNN is much faster on the Pi's ARM CPU
            if not os.path.isdir(ncnn_dir):
                log.info("Exporting %s -> NCNN (one-time)...", stem)
                YOLO(f"{stem}.pt").export(format="ncnn")
            return YOLO(ncnn_dir, task="detect")
        if self.mode != "coco":
            log.warning("Unknown MODEL_MODE '%s' -- using coco", self.mode)
            self.mode = "coco"
        try:
            return YOLO(f"{stem}.pt")  # auto-downloads on first run
        except Exception as exc:   # heavy model OOM / download fail -> fall back lighter
            if stem != "yolo11s":
                log.warning("Heavy model %s failed (%s) -- falling back to yolo11s.", stem, exc)
                self._stem, self.half = "yolo11s", False
                return YOLO("yolo11s.pt")
            raise

    def set_imgsz(self, imgsz):
        """Runtime imgsz change (Pi perf guard drops 320->256 under load)."""
        if imgsz != self.imgsz:
            log.info("imgsz %d -> %d", self.imgsz, imgsz)
            self.imgsz = imgsz

    def _infer(self, frame_bgr):
        """Run persistent tracking (stable track_ids) if available, else predict()."""
        kw = dict(imgsz=self.imgsz, conf=PREDICT_CONF, device=DEVICE, augment=self.augment,
                  half=self.half, iou=0.5, agnostic_nms=True, max_det=50, verbose=False)
        if self._use_tracking:
            try:
                return self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **kw)[0]
            except Exception as exc:
                log.warning("Tracking unavailable (%s) -- using predict() + (label,zone) keys", exc)
                self._use_tracking = False
        return self.model.predict(frame_bgr, **kw)[0]

    def detect(self, frame_bgr, enhance=None, assessment=None):
        """Public entry: CLAHE-enhance the frame (root-cause accuracy win), then run
        the full detect/rank pipeline. `enhance` overrides self.enhance_frames
        (callers that already enhanced -- /ocr, /describe -- pass False to avoid
        doubling). Never raises: on any error it logs and returns []."""
        self._last_assess = assessment
        try:
            do_enh = self.enhance_frames if enhance is None else enhance
            self._last_enhanced = bool(do_enh)
            if do_enh:
                frame_bgr = fq.enhance(frame_bgr, True)
            return self.detect_and_rank(frame_bgr)
        except Exception as exc:   # one bad frame must never kill the Navigate loop
            log.exception("detect() failed on a frame: %s", exc)
            return []

    def detect_and_rank(self, frame_bgr):
        """Full pipeline -> confirmed detections, ranked by priority. Each dict:
        {label, confidence, cx, cy, box, area_ratio, urgency, track_id} with
        `label` grouped to its canonical spoken word and a stable `track_id`
        (or None in predict() fallback).
        """
        global FRAME_W
        frame_h, frame_w = frame_bgr.shape[:2]
        FRAME_W = frame_w
        frame_area = float(frame_w * frame_h)

        result = self._infer(frame_bgr)

        raw = []      # (class_lower, confidence) for EVERY raw detection -- for the debug log
        dropped = {}  # reason -> count

        # --- Stage 1: confidence filter (NO allowlist drop when DETECT_ALL) ----
        kept = []
        for box in result.boxes:
            low = str(self.names[int(box.cls[0])]).lower()
            conf = float(box.conf[0])
            raw.append((low, conf))

            canonical = SYNONYMS.get(low, low)  # nicer word; never removes a class
            if not DETECT_ALL and canonical not in NAVIGATION_CLASSES:
                dropped["not-allowed"] = dropped.get("not-allowed", 0) + 1
                continue
            if conf < conf_threshold(canonical):
                dropped["low-confidence"] = dropped.get("low-confidence", 0) + 1
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            area_ratio = ((x2 - x1) * (y2 - y1) / frame_area) if frame_area else 0.0
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # class-aware urgency: scale the area by the class's hazard weight first
            urgency_area = area_ratio * urgency_scale_of(canonical)
            # path score: in the center third AND lower half = in the walking line
            path_score = ((frame_w / 3 <= cx <= 2 * frame_w / 3) +
                          (cy >= frame_h / 2))
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
            kept.append({
                "label": canonical,
                "confidence": round(conf, 3),
                "cx": cx, "cy": cy,
                "box": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "urgency_area": round(urgency_area, 4),
                "path_score": int(path_score),
                "urgency": classify_urgency(urgency_area),
                "track_id": track_id,
            })

        # --- Stage 2: geometry-only part suppression (no name blocklist) -------
        # Drop a NON-person box only if it's mostly inside a LARGER person box and
        # clearly smaller (glasses/face/hands on a person). A standalone bottle,
        # cup or bag -- not inside a person, or not tiny -- is always kept.
        persons = [k for k in kept if k["label"] == "person"]
        survivors = []
        for k in kept:
            if k["label"] != "person" and any(
                _ios(k["box"], p["box"]) > CONTAINMENT_IOS
                and _area(k) < CONTAINMENT_MAX_AREA * _area(p)
                for p in persons
            ):
                dropped["on-person"] = dropped.get("on-person", 0) + 1
                continue
            survivors.append(k)

        # --- Stage 3: temporal smoothing (confirm 2-of-4, forget after 6) ------
        keys = {(k["label"], zone_for(k["cx"])) for k in survivors}
        confirmed_keys = self.tracker.update(keys)
        confirmed = [k for k in survivors if (k["label"], zone_for(k["cx"])) in confirmed_keys]
        if len(survivors) - len(confirmed):
            dropped["unconfirmed"] = len(survivors) - len(confirmed)

        # --- Stage 4: rank (closeness, then path, then priority class, then size) -
        confirmed.sort(
            key=lambda d: (
                URGENCY_RANK[d["urgency"]],
                PATH_WEIGHT * d.get("path_score", 0),  # things in the walking line first
                importance_of(d["label"]),
                d["label"] in NAVIGATION_CLASSES,  # NAVIGATION_CLASSES is a priority hint only
                _area(d),
            ),
            reverse=True,
        )

        self._debug_log(raw, survivors, confirmed, dropped)
        return confirmed

    def _debug_log(self, raw, kept, confirmed, dropped):
        # Every 15th frame, dump what the model SAW (raw, before filtering) vs what
        # SURVIVED (kept) plus the active track_ids, so you can tune thresholds and
        # see tracking working. If raw lists many objects but kept is only
        # "person", the filter is too aggressive.
        self.frame_count += 1
        if self.frame_count % 15 != 0:
            return
        raw_str = ", ".join(f"{c}:{cf:.2f}" for c, cf in raw[:25]) or "(none)"
        kept_str = ", ".join(sorted(k["label"] for k in kept)) or "(none)"
        drops = ", ".join(f"{r} x{n}" for r, n in dropped.items() if n) or "none"
        ids = [d["track_id"] for d in confirmed if d.get("track_id") is not None]
        a = self._last_assess
        qual = (f"brightness={a['brightness']} blur={a['blur_score']} ({a['reason']})"
                if a else "(not assessed)")
        log.info("FRAME quality: %s | enhance=%s | imgsz=%d", qual, self._last_enhanced, self.imgsz)
        log.info("RAW (%d before filter): %s", len(raw), raw_str)
        log.info("KEPT (%d after filter): %s   | dropped: %s", len(kept), kept_str, drops)
        log.info("active track_ids: %s", ids or "(none -- predict fallback)")


# --------------------------------------------------------------------------- #
# What to speak  --  rate-limit + cap, on the already-ranked confirmed list.
# --------------------------------------------------------------------------- #
def select_announcements(detections, manager, approaching, now):
    """Produce this frame's (calm, event-driven) speech items.

    `detections` are the confirmed, ranked detections from detect_and_rank, each
    carrying a stable `track_id`. `manager` is the per-connection
    AnnouncementManager that remembers what's already been said and only speaks on
    real events. Returns a list of {text, rate, urgent, urgency} (unchanged shape),
    so the browser / Pi speech code doesn't change.
    """
    return manager.process(detections, approaching, now)


# --------------------------------------------------------------------------- #
# Scene description (Describe mode)
# --------------------------------------------------------------------------- #
# A vision-language model turns one frame into a short spoken description. This
# is the only part that calls an external AI service (Google Gemini). It is
# isolated behind SceneDescriber.describe() so you can swap providers -- a local
# VLM, Anthropic, etc. -- WITHOUT touching server.py or the browser.
DESCRIBE_MODEL = "gemini-2.5-flash"  # richer: "gemini-3.5-flash"
DESCRIBE_TIMEOUT_MS = 15000          # per-request timeout; Gemini's MINIMUM is 10s, so keep >10000
DESCRIBE_RETRIES = 1                 # one extra online attempt before the offline fallback
DESCRIBE_PROMPT = (
    "You are the eyes of a blind person wearing this camera. In one or two short, "
    "calm sentences, tell them what is right in front of them. Call out HAZARDS first "
    "(people, vehicles, poles, low obstacles), then any STEPS or STAIRS and which way "
    "they go, then DOORWAYS, and which direction is clear to walk. If there is any sign "
    "or printed text, READ IT ALOUD exactly. Be concrete and specific; give directions "
    "as left / right / ahead. Do not begin with 'The image' or 'I see'."
)


def describe_from_detections(detections, lang="en"):
    """OFFLINE scene description, synthesized from the current Navigate detections.

    Used when Gemini has no key or can't be reached, so Describe is ALWAYS useful
    (critical where there's no WiFi). e.g. "In front of you: two people ahead, a
    chair on your left." Mirrors the Navigate grouping/plurals, bilingual.
    """
    groups = {}
    for d in detections or []:
        key = (d["label"], zone_for(d["cx"]))
        groups[key] = groups.get(key, 0) + 1
    if not groups:
        return "The path ahead looks clear." if lang != "ne" else "अगाडिको बाटो खाली देखिन्छ।"
    items = sorted(groups.items(), key=lambda kv: (importance_of(kv[0][0]), kv[1]), reverse=True)
    parts = []
    for (label, zone), n in items[:5]:
        if lang == "ne":
            nm = NE_NAMES.get(label, label)
            head = nm if n == 1 else f"{NE_NUMS.get(n, n)} {nm}"
            parts.append(f"{head} {NE_ZONES.get(zone, zone)}")
        else:
            head = f"a {label}" if n == 1 else f"{_number_word(n)} {_pluralize(label, n)}"
            parts.append(f"{head} {zone}")
    if lang == "ne":
        return "तपाईंको अगाडि: " + ", ".join(parts) + "।"
    return "In front of you: " + ", ".join(parts) + "."


class SceneDescriber:
    """Describe a frame with a vision-language model (Google Gemini), with an
    OFFLINE fallback so Describe never just fails.

    Online: Gemini, with an ~8s timeout and one retry. Offline / no key / network
    down: a description synthesized from the current Navigate detections
    (describe_from_detections). SWAP POINT: replace the online block to change
    providers -- keep the signature (frame, lang, detections) -> spoken string.
    """

    def __init__(self, model=DESCRIBE_MODEL):
        self.model = model
        self._client = None  # built lazily on first describe()

    def _client_or_none(self):
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return None
            from google import genai  # lazy: vision_core imports fine without the pkg
            from google.genai import types

            try:   # http_options(timeout=...) so a stalled request can't hang Describe
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=DESCRIBE_TIMEOUT_MS),
                )
            except Exception:   # older SDKs without http_options -- still works, just no timeout
                self._client = genai.Client(api_key=api_key)
            log.info("SceneDescriber using Gemini model: %s", self.model)
        return self._client

    def describe(self, frame_bgr, lang="en", detections=None):
        """Return a short spoken description. Tries Gemini (timeout + 1 retry); on
        no key / network failure, falls back to an offline summary of `detections`.
        """
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return ("Could not read the camera image." if lang != "ne"
                    else "क्यामेरा छवि पढ्न सकिएन।")

        client = self._client_or_none()
        if client is not None:
            from google.genai import types

            prompt = DESCRIBE_PROMPT
            if lang == "ne":
                prompt += " Respond ONLY in Nepali, written in Devanagari script."
            part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
            for attempt in range(1 + DESCRIBE_RETRIES):
                try:
                    resp = client.models.generate_content(model=self.model, contents=[part, prompt])
                    text = (resp.text or "").strip()
                    if text:
                        log.info("Describe source: gemini (attempt %d)", attempt + 1)
                        return text
                except Exception as exc:
                    log.warning("Describe attempt %d failed: %s", attempt + 1, exc)

        # LAPTOP offline: try a LOCAL vision-language model via Ollama (no internet).
        # The Pi never runs a VLM -- it offloads to the laptop's /remote/describe.
        if FEATURE_PROFILE == "laptop":
            vlm = self._ollama_describe(buf.tobytes(), lang)
            if vlm:
                log.info("Describe source: ollama (%s)", LAPTOP_VLM_MODEL)
                return vlm

        # graceful fallback -- always say something useful from the detections
        log.info("Describe source: offline fallback (from %d detections)", len(detections or []))
        return describe_from_detections(detections, lang)

    def _ollama_describe(self, jpg_bytes, lang):
        """Local VLM via Ollama (http://localhost:11434). Returns text or '' if Ollama
        isn't running / the model isn't pulled. Config: LAPTOP_VLM_MODEL."""
        try:
            import base64
            import json
            import urllib.request

            prompt = DESCRIBE_PROMPT + (" Respond ONLY in Nepali (Devanagari)." if lang == "ne" else "")
            body = json.dumps({
                "model": LAPTOP_VLM_MODEL, "prompt": prompt, "stream": False,
                "images": [base64.b64encode(jpg_bytes).decode()],
            }).encode()
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return (json.loads(r.read()).get("response") or "").strip()
        except Exception as exc:
            log.info("Ollama VLM unavailable (%s) -- run `ollama pull %s`?", exc, LAPTOP_VLM_MODEL)
            return ""


# --------------------------------------------------------------------------- #
# Offline text-to-speech (Nepali)  --  espeak-ng
# --------------------------------------------------------------------------- #
# English is spoken OFFLINE by the browser's Web Speech API; the only thing that
# needed the internet was Nepali (we used gTTS). This replaces it with espeak-ng,
# a tiny OFFLINE engine that ships a Nepali voice and reads Devanagari directly.
# No model download, no GPU, and it's the SAME engine the Raspberry Pi uses
# (pi_app.py) -- so the whole app runs offline; only Describe (a cloud vision
# model) needs the network.
#   Windows:   winget install eSpeak-NG.eSpeak-NG
#   Pi/Linux:  sudo apt install espeak-ng
# (MMS-TTS was the first idea but Meta never released a Nepali checkpoint -- only
# Newari, a different language -- so it would mispronounce Nepali.)
ESPEAK_VOICES = {"ne": "ne", "en": "en"}   # app language -> espeak-ng voice code
ESPEAK_WPM = 150                            # words/min; a touch slower than default for clarity
_WIN_ESPEAK_PATHS = (
    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
    r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
)


def find_espeak():
    """Locate the espeak-ng (or espeak) binary: on PATH (Pi/Linux) or the standard
    Windows install dir. Returns the path, or None if it isn't installed. Shared by
    OfflineTTS (server) and pi_app so discovery is identical everywhere."""
    return (shutil.which("espeak-ng") or shutil.which("espeak")
            or next((p for p in _WIN_ESPEAK_PATHS if os.path.exists(p)), None))


class OfflineTTS:
    """Synthesize speech to WAV bytes with espeak-ng, fully offline (no internet).

    Built once in server.py; the espeak-ng binary is located once and reused.
    Returns WAV bytes the browser plays via <audio>.

    PI NOTE: this is already the Pi-friendly engine -- the same espeak-ng the Pi
    speaks English through. Keep the contract (text, lang) -> WAV bytes.
    """

    def __init__(self):
        self._exe_path = None

    def _exe(self):
        if self._exe_path is None:
            cand = find_espeak()
            if cand is None:
                raise RuntimeError(
                    "espeak-ng not found. Install it -- Windows: "
                    "'winget install eSpeak-NG.eSpeak-NG'; Pi/Linux: "
                    "'sudo apt install espeak-ng'."
                )
            self._exe_path = cand
            log.info("Offline TTS using espeak-ng: %s", cand)
        return self._exe_path

    def synth(self, text, lang="ne"):
        """Return spoken `text` as WAV bytes. Text is piped via stdin (UTF-8) so
        long strings and Devanagari aren't mangled by command-line quoting."""
        text = (text or "").strip()
        if not text:
            return b""
        voice = ESPEAK_VOICES.get(lang, lang)
        proc = subprocess.run(
            [self._exe(), "-v", voice, "-s", str(ESPEAK_WPM), "--stdout"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                "espeak-ng failed: " + proc.stderr.decode("utf-8", "ignore")[:200]
            )
        return _fix_wav_sizes(proc.stdout)


def _fix_wav_sizes(raw: bytes) -> bytes:
    """Patch the RIFF/data length fields in a WAV.

    espeak-ng streams to stdout and can't seek back to fill in the chunk sizes, so
    it leaves placeholders -- which makes some players report a bogus duration or
    refuse to seek/replay. We rewrite both sizes from the actual byte length.
    """
    import struct

    if len(raw) < 44 or raw[:4] != b"RIFF":
        return raw
    b = bytearray(raw)
    struct.pack_into("<I", b, 4, len(b) - 8)            # RIFF chunk size
    i = b.find(b"data")
    if i != -1 and i + 8 <= len(b):
        struct.pack_into("<I", b, i + 4, len(b) - (i + 8))  # data chunk size
    return bytes(b)


# --------------------------------------------------------------------------- #
# Money mode -- Nepali banknote classifier (trained by train_banknote.py)
# --------------------------------------------------------------------------- #
# Reads a single held note and says its value. The model is the one you train with
# train_banknote.py (models/banknote.pt). It is loaded ONCE at server startup, not
# per request. Each class maps to a spoken phrase in BOTH English and Nepali; the
# browser's Language toggle decides which one is voiced (English via Web Speech,
# Nepali via the offline /tts). Below MIN_CONF we refuse to guess.
BANKNOTE_MODEL = "models/banknote.pt"

# --- anti-hallucination knobs (a classifier ALWAYS outputs a class, so without
#     these it invents a denomination on an empty wall/hand). All tunable. -------
MONEY_CONF = 0.85          # top-1 must beat this confidence...
MONEY_MARGIN = 0.30        # ...AND beat top-2 by at least this margin
MONEY_VOTES = 10           # frames to sample for temporal voting (~1s at 10 fps)
MONEY_MIN_AGREE = 6        # the same denomination must win at least this many votes
BANKNOTE_ROI = True        # crop the central region before classifying (drop edge noise)
BANKNOTE_ROI_FRAC = 0.85   # keep this central fraction of width/height
MONEY_OCR_VERIFY = False   # optional: check the note's printed digits match (server sets the
                           # OCR fn; off by default, and skip on the Pi -- too slow)

# trained class name -> (English phrase, Nepali phrase)
BANKNOTE_SPEECH = {
    "rs5":    ("Five rupees",         "पाँच रुपैयाँ"),
    "rs10":   ("Ten rupees",          "दस रुपैयाँ"),
    "rs20":   ("Twenty rupees",       "बीस रुपैयाँ"),
    "rs50":   ("Fifty rupees",        "पचास रुपैयाँ"),
    "rs100":  ("One hundred rupees",  "एक सय रुपैयाँ"),
    "rs500":  ("Five hundred rupees", "पाँच सय रुपैयाँ"),
    "rs1000": ("One thousand rupees", "एक हजार रुपैयाँ"),
    "background": ("No note",          "कुनै नोट छैन"),
}
# the digits we expect to see printed on each denomination (for OCR cross-check)
BANKNOTE_DIGITS = {"rs5": "5", "rs10": "10", "rs20": "20", "rs50": "50",
                   "rs100": "100", "rs500": "500", "rs1000": "1000"}
NO_NOTE_SPEECH = ("No note detected. Hold a note steady in good light.",
                  "कुनै नोट देखिएन। राम्रो उज्यालोमा नोट स्थिर राख्नुहोस्।")
UNSURE_SPEECH = ("Couldn't read the note clearly, please try again.",
                 "नोट प्रस्ट पढ्न सकिएन, कृपया फेरि प्रयास गर्नुहोस्।")
DARK_NOTE_SPEECH = ("Too dark or blurry. Move to better light and hold steady.",
                    "अति अँध्यारो वा अस्पष्ट छ। उज्यालोमा स्थिर राख्नुहोस्।")
NO_MODEL_SPEECH = ("Money model not trained yet. Run train banknote.",
                   "मनी मोडेल तालिम भएको छैन।")


class BanknoteClassifier:
    """Classify a held Nepali rupee note, with strong rejection of "no note".

    Loaded ONCE at startup. A bare classifier hallucinates a denomination on empty
    frames, so Money mode here uses, in order: a quality pre-check, a central-ROI
    crop, a confidence+margin gate, a "background" class -> no note, and (the key
    fix) TEMPORAL VOTING over ~10 frames -- a value is announced only if it wins a
    clear majority. Disagreement -> "try again", never a guess.
    """

    def __init__(self, model_path=BANKNOTE_MODEL, conf=MONEY_CONF, margin=MONEY_MARGIN,
                 min_agree=MONEY_MIN_AGREE, roi=BANKNOTE_ROI):
        self.conf, self.margin, self.min_agree, self.roi = conf, margin, min_agree, roi
        self.ocr_verify_fn = None   # server may set an OCR fn when MONEY_OCR_VERIFY
        self.model = None
        if os.path.exists(model_path):
            self.model = YOLO(model_path, task="classify")   # task -> never a detector
            log.info("Money: banknote classifier loaded from %s (classes: %s)",
                     model_path, list(self.model.names.values()))
        else:
            log.warning("Money: %s not found -- train it with train_banknote.py "
                        "(Money mode will say 'not trained yet').", model_path)

    # ---- helpers ---------------------------------------------------------- #
    def _roi_crop(self, frame):
        if not self.roi or frame is None:
            return frame
        h, w = frame.shape[:2]
        m = (1.0 - BANKNOTE_ROI_FRAC) / 2.0
        return frame[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))]

    def _classify_one(self, frame):
        """Return (name, conf, margin) for ONE frame (top-1 conf and its lead over top-2)."""
        probs = self.model.predict(self._roi_crop(frame), imgsz=224, verbose=False)[0].probs
        vals = probs.data.tolist()
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        top1 = order[0]
        conf = float(vals[top1])
        second = float(vals[order[1]]) if len(order) > 1 else 0.0
        return self.model.names[top1], conf, conf - second

    def _gate(self, name, conf, margin):
        """A single reading is trustworthy only if it's confident, clearly ahead of
        the runner-up, and not the background class."""
        return name != "background" and conf >= self.conf and margin >= self.margin

    def _result(self, ok, name, conf, en_ne, **extra):
        en, ne = en_ne
        out = {"available": True, "ok": ok, "class": name,
               "confidence": round(conf, 4), "text": en, "text_ne": ne}
        out.update(extra)
        return out

    # ---- single frame (used by the tally "add", and as a fallback) -------- #
    def classify(self, frame_bgr):
        """Gated single-frame classify. ok only if confident + clear + a real note."""
        if self.model is None:
            return self._result(False, None, 0.0, NO_MODEL_SPEECH, available=False)
        name, conf, margin = self._classify_one(frame_bgr)
        log.info("Money 1-frame: %s conf=%.3f margin=%.3f", name, conf, margin)
        if self._gate(name, conf, margin):
            return self._result(True, name, conf, BANKNOTE_SPEECH[name], margin=round(margin, 4))
        # confident background -> explicitly "no note"; otherwise "no note detected"
        return self._result(False, name, conf, NO_NOTE_SPEECH, margin=round(margin, 4))

    # ---- temporal voting over many frames (the main Money-mode path) ------- #
    def classify_voted(self, frames):
        """Vote across ~MONEY_VOTES frames. Announce a denomination only if it wins
        >= min_agree gated votes; else 'no note' (mostly background) or 'unsure'."""
        if self.model is None:
            return self._result(False, None, 0.0, NO_MODEL_SPEECH, available=False, votes=[])
        if not frames:
            return self._result(False, None, 0.0, NO_NOTE_SPEECH, votes=[])
        from collections import Counter
        votes, confs, bg, details = Counter(), {}, 0, []
        for f in frames:
            name, conf, margin = self._classify_one(f)
            ok = self._gate(name, conf, margin)
            details.append((name, round(conf, 2), round(margin, 2), ok))
            if name == "background":
                bg += 1
            if ok:
                votes[name] += 1
                confs.setdefault(name, []).append(conf)
        log.info("Money votes (%d frames): %s | background=%d | tally=%s",
                 len(frames), details, bg, dict(votes))
        if votes:
            name, n = votes.most_common(1)[0]
            if n >= self.min_agree:
                avg = sum(confs[name]) / len(confs[name])
                if MONEY_OCR_VERIFY and self.ocr_verify_fn and not self._ocr_ok(frames[-1], name):
                    log.info("Money: OCR digits disagree with %s -> reject", name)
                    return self._result(False, name, avg, UNSURE_SPEECH, votes=details)
                log.info("Money ACCEPT %s (%d/%d votes, avg conf %.3f)", name, n, len(frames), avg)
                return self._result(True, name, avg, BANKNOTE_SPEECH[name],
                                    votes=details, agree=n)
        # no denomination won the majority
        if bg >= max(1, len(frames) // 2):
            return self._result(False, "background", 0.0, NO_NOTE_SPEECH, votes=details)
        return self._result(False, None, 0.0, UNSURE_SPEECH, votes=details)

    def _ocr_ok(self, frame, name):
        """Cross-check: does the note crop's OCR contain the expected digits?"""
        try:
            text = self.ocr_verify_fn(self._roi_crop(frame)) or ""
        except Exception:
            return True   # OCR failure shouldn't block a confident vote
        digits = "".join(ch for ch in text if ch.isdigit())
        return BANKNOTE_DIGITS.get(name, "") in digits
