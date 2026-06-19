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
* Change WHAT is seen:  edit NAVIGATION_CLASSES (whole objects worth announcing)
                        and OPENVOCAB_CLASSES (the YOLO-World prompt vocabulary).
* Hide body parts:      add class names to PART_CLASSES.
* Tighten/loosen:       edit CONF_THRESHOLDS (per-class confidence floors).
* See what's happening: detection logs (raw vs kept + why things were dropped)
                        print ~once a second to the console.
"""

import logging
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger("soundsight.vision")

# Pick the best available device once, at import time, and announce it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# "coco"     -> yolo11s.pt (YOLO11, 80 COCO classes). Default: fast, accurate,
#               and almost every COCO class is navigation-relevant.
# "openvocab"-> YOLO-World (yolov8s-worldv2.pt) restricted to OPENVOCAB_CLASSES.
#               Best fit (only ever detects what a blind user needs); falls back
#               to "coco" automatically if the weights/deps aren't available.
# "oiv7"     -> yolov8s-oiv7.pt (~600 classes) leaning hard on the allowlist +
#               part suppression below to stay sane.
MODEL_MODE = "coco"

IMG_SIZE = 640        # inference resolution
PREDICT_CONF = 0.35   # low model floor; real cutoffs are the per-class thresholds below

# Open-vocabulary prompt for "openvocab" mode -- exactly the things a blind user
# needs to walk around. Edit freely; YOLO-World detects whatever you list.
OPENVOCAB_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "door", "stairs",
    "chair", "table", "sofa", "bed", "pole", "wall", "fence", "dog", "cat",
    "traffic light", "stop sign", "bench", "backpack", "bottle", "cup", "laptop",
    "tv", "refrigerator", "sink", "toilet", "potted plant", "curb", "pothole",
    "obstacle",
]

# Whole objects worth announcing (canonical, lowercase). ANY detection whose
# (synonym-mapped) class isn't in here is dropped, in every mode.
NAVIGATION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "door", "stairs",
    "chair", "table", "couch", "bed", "pole", "wall", "fence", "dog", "cat",
    "traffic light", "stop sign", "bench", "backpack", "bottle", "cup", "laptop",
    "tv", "refrigerator", "sink", "toilet", "plant", "curb", "pothole", "obstacle",
}

# Map model-specific / synonymous labels (lowercased) -> one spoken word.
SYNONYMS = {
    "sofa": "couch",
    "dining table": "table",
    "potted plant": "plant",
    "houseplant": "plant",
    "television": "tv",
    "coffee cup": "cup",
    "mug": "cup",
    # Open Images labels people as Man/Woman/Boy/Girl -- an assistive tool
    # shouldn't announce a model's (often wrong) guess at gender.
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
}

# Parts / worn items / attributes -- ALWAYS discarded (this is the "glasses,
# sunglasses, human face, footwear..." spam the OIV7 model produced).
PART_CLASSES = {
    "glasses", "sunglasses", "goggles", "human face", "human head", "human hair",
    "human hand", "human arm", "human leg", "human nose", "human eye", "human ear",
    "human mouth", "clothing", "footwear", "hat", "sock", "sleeve", "human body",
}

# Worn/held items: if mostly inside a (larger) person box they're being carried,
# not obstacles -- suppressed by part-containment below.
CONTAINMENT_SUPPRESS_CLASSES = {"backpack", "bottle", "cup", "laptop"}

# Per-class confidence floors (canonical class -> threshold). Anything below its
# class's floor is dropped; unlisted classes use DEFAULT_CONF.
DEFAULT_CONF = 0.55  # small / ambiguous (bottle, cup, laptop, backpack, plant, signs, dog/cat)
CONF_THRESHOLDS = {
    "person": 0.40,
    # vehicles
    "car": 0.45, "bus": 0.45, "truck": 0.45, "motorcycle": 0.45, "bicycle": 0.45,
    # furniture / fixtures
    "chair": 0.45, "table": 0.45, "couch": 0.45, "bed": 0.45, "bench": 0.45,
    "sink": 0.45, "toilet": 0.45, "refrigerator": 0.45, "tv": 0.45,
    # structure / hazards
    "stairs": 0.50, "door": 0.50, "pole": 0.50, "obstacle": 0.50,
    "wall": 0.50, "fence": 0.50, "curb": 0.50, "pothole": 0.50,
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
# Announcement timing
# --------------------------------------------------------------------------- #
ANNOUNCE_COOLDOWN = 2.5    # don't repeat the same object+zone within this window
VERY_CLOSE_COOLDOWN = 1.0  # ...but very-close hazards re-warn this often


class Announcer:
    """Prevents repeating the same object+zone within `cooldown` seconds."""

    def __init__(self, cooldown=ANNOUNCE_COOLDOWN):
        self.cooldown = cooldown
        self.last = {}

    def consider(self, label, zone, now, cooldown=None):
        # cooldown=None -> the normal 2.5s; very-close warnings pass 1.0s.
        cd = self.cooldown if cooldown is None else cooldown
        key = (label, zone)
        if now - self.last.get(key, -999) >= cd:
            self.last[key] = now
            return f"{label} {zone}"
        return None


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
    after `forget` consecutive missing frames.
    """

    def __init__(self, confirm=3, window=5, forget=5):
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

    def __init__(self, mode=MODEL_MODE):
        self.mode = mode
        self.model = self._load_model()
        self.names = self.model.names
        self.tracker = TemporalTracker()
        self._last_log = 0.0
        log.info(
            "Detection: mode=%s, %d model classes, device=%s",
            self.mode, len(self.names), DEVICE.upper(),
        )
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
                return YOLO("yolo11s.pt")
        if self.mode == "oiv7":
            return YOLO("yolov8s-oiv7.pt")
        if self.mode != "coco":
            log.warning("Unknown MODEL_MODE '%s' -- using coco", self.mode)
            self.mode = "coco"
        return YOLO("yolo11s.pt")  # auto-downloads on first run

    def detect_and_rank(self, frame_bgr):
        """Full pipeline -> confirmed, navigation-relevant detections, ranked by
        priority. Each dict: {label, confidence, cx, cy, box, area_ratio, urgency}
        with `label` already grouped to its canonical spoken word.
        """
        global FRAME_W
        frame_h, frame_w = frame_bgr.shape[:2]
        FRAME_W = frame_w
        frame_area = float(frame_w * frame_h)

        results = self.model.predict(
            frame_bgr, imgsz=IMG_SIZE, conf=PREDICT_CONF, device=DEVICE, verbose=False
        )[0]

        raw_count = len(results.boxes)
        dropped = {}  # reason -> count, for tuning logs

        # --- Stage 1: allowlist + part blocklist + per-class confidence -------
        kept = []
        for box in results.boxes:
            low = self.names[int(box.cls[0])].lower()
            if low in PART_CLASSES:
                dropped["part"] = dropped.get("part", 0) + 1
                continue
            canonical = SYNONYMS.get(low, low)
            if canonical not in NAVIGATION_CLASSES:
                dropped["not-navigation"] = dropped.get("not-navigation", 0) + 1
                continue
            conf = float(box.conf[0])
            if conf < conf_threshold(canonical):
                dropped["low-confidence"] = dropped.get("low-confidence", 0) + 1
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            area_ratio = ((x2 - x1) * (y2 - y1) / frame_area) if frame_area else 0.0
            kept.append({
                "label": canonical,
                "confidence": round(conf, 3),
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "box": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "urgency": classify_urgency(area_ratio),
            })

        # --- Stage 2: part-containment suppression (worn/held inside a person) -
        persons = [k for k in kept if k["label"] == "person"]
        survivors = []
        for k in kept:
            if k["label"] in CONTAINMENT_SUPPRESS_CLASSES and any(
                _area(p) > _area(k) and _ios(k["box"], p["box"]) > 0.6 for p in persons
            ):
                dropped["worn-on-person"] = dropped.get("worn-on-person", 0) + 1
                continue
            survivors.append(k)

        # --- Stage 3: temporal smoothing (confirm 3-of-5, forget after 5) ------
        keys = {(k["label"], zone_for(k["cx"])) for k in survivors}
        confirmed_keys = self.tracker.update(keys)
        confirmed = [k for k in survivors if (k["label"], zone_for(k["cx"])) in confirmed_keys]
        if len(survivors) - len(confirmed):
            dropped["unconfirmed"] = len(survivors) - len(confirmed)

        # --- Stage 4: rank for announcing (closeness, then importance, then size)
        confirmed.sort(
            key=lambda d: (URGENCY_RANK[d["urgency"]], importance_of(d["label"]), _area(d)),
            reverse=True,
        )

        self._log(raw_count, len(confirmed), dropped)
        return confirmed

    # Back-compat alias so server.py (boxes from vision.detect) is unchanged.
    def detect(self, frame_bgr):
        return self.detect_and_rank(frame_bgr)

    def _log(self, raw, kept, dropped):
        # Throttle to ~1/sec so the console stays readable while still tunable.
        now = time.time()
        if now - self._last_log < 1.0:
            return
        self._last_log = now
        reasons = ", ".join(f"{r} x{c}" for r, c in dropped.items() if c) or "none"
        log.info("[%s] raw=%d kept=%d | dropped: %s", self.mode, raw, kept, reasons)


# --------------------------------------------------------------------------- #
# What to speak  --  rate-limit + cap, on the already-ranked confirmed list.
# --------------------------------------------------------------------------- #
def select_announcements(detections, announcer, approaching, now, max_items=2):
    """
    `detections` is already confirmed + priority-ranked by detect_and_rank().
    Apply the per-object+zone cooldown (very-close bypasses to 1s), phrase it,
    and cap at `max_items` so it stays calm.

    Phrasing:
      * very close -> "Careful, {name} {zone}, very close" (rate 1.3, urgent)
      * approaching -> "{name} {zone}, getting closer"
      * otherwise  -> "{name} {zone}"

    Never emits a generic "something detected": every confirmed object already
    has a real spoken name (its canonical class), but we guard anyway.
    """
    spoken = []
    for det in detections:
        name = det["label"]
        if not name:  # no spoken name -> stay silent
            continue
        zone = zone_for(det["cx"])
        urgency = det["urgency"]
        very_close = urgency == URGENCY_VERY_CLOSE

        cooldown = VERY_CLOSE_COOLDOWN if very_close else None  # None -> 2.5s
        if announcer.consider(name, zone, now, cooldown=cooldown) is None:
            continue

        if very_close:
            text, rate, urgent = f"Careful, {name} {zone}, very close", 1.3, True
        elif (name, zone) in approaching:
            text, rate, urgent = f"{name} {zone}, getting closer", 1.0, False
        else:
            text, rate, urgent = f"{name} {zone}", 1.0, False

        spoken.append({"text": text, "rate": rate, "urgent": urgent, "urgency": urgency})
        if len(spoken) >= max_items:
            break
    return spoken


# --------------------------------------------------------------------------- #
# Scene description (Describe mode)
# --------------------------------------------------------------------------- #
# A vision-language model turns one frame into a short spoken description. This
# is the only part that calls an external AI service (Google Gemini). It is
# isolated behind SceneDescriber.describe() so you can swap providers -- a local
# VLM, Anthropic, etc. -- WITHOUT touching server.py or the browser.
DESCRIBE_MODEL = "gemini-2.5-flash"  # richer: "gemini-3.5-flash"
DESCRIBE_PROMPT = (
    "You are the eyes of a blind person wearing this camera. In one or two short, "
    "calm sentences, describe what is in front of them and anything important to "
    "know: people, obstacles, doorways, stairs, signs and what they say. Be "
    "concrete and specific. Do not begin with 'The image' or 'I see'."
)


class SceneDescriber:
    """Describe a frame with a vision-language model (Google Gemini).

    SWAP POINT: to change providers, replace the body of describe() -- keep the
    signature (BGR frame in, spoken-style string out) and nothing else changes.
    The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) at first use.
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

            self._client = genai.Client(api_key=api_key)
            log.info("SceneDescriber using Gemini model: %s", self.model)
        return self._client

    def describe(self, frame_bgr):
        """Return a short description of the frame, or a spoken-friendly error."""
        client = self._client_or_none()
        if client is None:
            return "Describe needs a Gemini API key. Set GEMINI_API_KEY and restart the server."

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return "Could not read the camera image."

        from google.genai import types

        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                    DESCRIBE_PROMPT,
                ],
            )
            return (resp.text or "").strip() or "I couldn't describe the scene."
        except Exception as exc:  # network/quota/key errors -> speak something useful
            log.warning("Describe (Gemini) failed: %s", exc)
            return f"Describe failed: {exc}"
