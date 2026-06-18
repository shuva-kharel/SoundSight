"""
vision_core.py  --  Pure AI / vision logic for SoundSight.

This module is intentionally hardware-agnostic and contains NO web code, so it
ports to a Raspberry Pi (or any other host) unchanged. It only knows how to:
  * load a YOLO model and run object detection on a BGR frame,
  * turn a detection's horizontal position into a spoken zone,
  * decide *what* to announce (priority + anti-repeat cooldown).

The web layer (server.py) is responsible for transport (WebSocket / HTTP),
frame decoding, and anything browser-specific.
"""

import logging
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger("soundsight.vision")

# Pick the best available device once, at import time, and announce it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Frame width used by the zone logic below. Frames arriving from the browser are
# downscaled to 416px wide, so this is the *actual* frame width. detect() keeps
# it in sync with whatever frame it is actually handed (see detect()).
FRAME_W = 416


# --------------------------------------------------------------------------- #
# Announcement logic  --  EXACT behavior, do not change.
# --------------------------------------------------------------------------- #
def zone_for(cx):
    if cx < FRAME_W / 3:
        return "on your left"
    if cx > 2 * FRAME_W / 3:
        return "on your right"
    return "ahead"


# How long to stay quiet about the same object+zone once it's been announced.
# A constant scene (e.g. "person ahead") is spoken once, then not repeated until
# this many seconds pass -- so it doesn't chant "careful, careful, ...".
ANNOUNCE_COOLDOWN = 6.0


class Announcer:
    """Announce each object+zone once, then stay quiet.

    It re-speaks the same object+zone only when EITHER (a) it becomes more urgent
    than last time (escalation -- e.g. it just got "very close"), so genuine
    danger is never silenced, OR (b) the refresh cooldown elapses, as a gentle
    reminder. New objects or objects in a new zone are announced right away.
    """

    def __init__(self, cooldown=ANNOUNCE_COOLDOWN):
        self.cooldown = cooldown
        self.last = {}  # (label, zone) -> (last_time, last_urgency_rank)

    def consider(self, label, zone, now, urgency_rank=0):
        key = (label, zone)
        last_time, last_rank = self.last.get(key, (-1e9, -1))
        escalated = urgency_rank > last_rank
        if escalated or (now - last_time) >= self.cooldown:
            self.last[key] = (now, urgency_rank)
            return f"{label} {zone}"
        return None


# --------------------------------------------------------------------------- #
# Proximity / urgency
# --------------------------------------------------------------------------- #
# Closeness is estimated from how much of the frame an object's box fills:
#     area_ratio = box_area / frame_area
FAR_MAX = 0.05          # area_ratio < 0.05            -> "far"
NEAR_MAX = 0.20         # 0.05 <= area_ratio <= 0.20   -> "near"
                        # area_ratio > 0.20            -> "very close"

URGENCY_FAR = "far"
URGENCY_NEAR = "near"
URGENCY_VERY_CLOSE = "very close"

# Speaking priority is urgency-first (a very-close hazard outranks a distant
# person), then people, then bounding-box area. This keeps the closest danger
# from being dropped by the per-frame cap.
URGENCY_RANK = {URGENCY_FAR: 0, URGENCY_NEAR: 1, URGENCY_VERY_CLOSE: 2}

# How much area_ratio must grow vs. the previous frame to count as "approaching"
# (a small margin so box jitter doesn't trigger false alarms).
APPROACH_MARGIN = 0.01


def classify_urgency(area_ratio):
    if area_ratio > NEAR_MAX:
        return URGENCY_VERY_CLOSE
    if area_ratio >= FAR_MAX:
        return URGENCY_NEAR
    return URGENCY_FAR


class ApproachDetector:
    """
    Flags objects that are getting closer, one instance per connection.

    Right now "closer" is inferred purely from vision: an object's area_ratio
    growing frame-over-frame. On the Raspberry Pi you can REPLACE the body of
    approaching_objects() with real ultrasonic-sensor readings -- keep the same
    signature (detections in, a set of (label, zone) keys out) and nothing else
    in the pipeline has to change.
    """

    def __init__(self, margin=APPROACH_MARGIN):
        self.margin = margin
        self.prev = {}  # (label, zone) -> largest area_ratio seen last frame

    def approaching_objects(self, detections):
        """Return the set of (label, zone) keys whose area_ratio grew this frame.

        PI PORT: swap this body for ultrasonic distance deltas -- return the set
        of object keys the sensor reports are getting closer.
        """
        current = {}
        for det in detections:
            key = (det["label"], zone_for(det["cx"]))
            current[key] = max(current.get(key, 0.0), det["area_ratio"])

        approaching = {
            key
            for key, ratio in current.items()
            if key in self.prev and ratio > self.prev[key] + self.margin
        }
        self.prev = current
        return approaching


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
# Default model: Open Images V7 (~600 classes) so Read/Navigate know everyday
# objects -- pen, mobile phone, headphones, book, laptop, mouse, cup, bottle...
# Swap to "yolo11n.pt" for the lighter 80-class COCO model (faster, fewer types),
# or a YOLO-World model if you want to set your own open-vocabulary class list.
DETECT_MODEL = "yolov8s-oiv7.pt"
CONF_THRESHOLD = 0.45
IMG_SIZE = 640  # 600-class model + small objects (pens, earbuds) need the resolution


class VisionCore:
    """Loads YOLO once and runs detection. Reusable on Pi or server."""

    def __init__(self, model_path=DETECT_MODEL):
        # Auto-downloads the weights on first run.
        self.model = YOLO(model_path)
        self.names = self.model.names
        log.info(
            "YOLO model '%s' (%d classes) loaded on device: %s",
            model_path, len(self.names), DEVICE.upper(),
        )
        if DEVICE == "cpu":
            log.warning("Running on CPU -- detection will be slower.")

    def detect(self, frame_bgr):
        """
        Run object detection on a single BGR frame.

        Returns a list of dicts:
            {label, confidence, cx, cy, box[x1, y1, x2, y2], area_ratio, urgency}
        with confidence < CONF_THRESHOLD already filtered out.
        """
        # Keep the global frame width (used by zone_for) honest about the frame
        # we are actually processing.
        global FRAME_W
        frame_h, frame_w = frame_bgr.shape[:2]
        FRAME_W = frame_w
        frame_area = float(frame_w * frame_h)

        results = self.model.predict(
            frame_bgr,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLD:  # belt-and-suspenders
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = self.names[int(box.cls[0])]
            area_ratio = ((x2 - x1) * (y2 - y1) / frame_area) if frame_area else 0.0
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "box": [x1, y1, x2, y2],
                    "area_ratio": round(area_ratio, 4),
                    "urgency": classify_urgency(area_ratio),
                }
            )
        return detections


# --------------------------------------------------------------------------- #
# What to speak  --  priority + cap, kept here so the Pi port speaks the same.
# --------------------------------------------------------------------------- #
def _area(det):
    x1, y1, x2, y2 = det["box"]
    return (x2 - x1) * (y2 - y1)


# Open Images labels people as Man/Woman/Boy/Girl/etc. We treat them all as
# "person": it keeps the people-first priority working, and an assistive tool
# shouldn't announce a model's (often wrong) guess at someone's gender.
PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "human", "human body", "human face"}


def _is_person(label):
    return label.lower() in PERSON_LABELS


def select_announcements(detections, announcer, approaching, now, max_items=2):
    """
    Given this frame's detections, decide what to say.

    Priority: urgency first (very close > near > far), then people, then largest
    objects by bounding-box area -- so the closest hazard is never crowded out by
    the per-frame cap. Caps at `max_items` spoken items per frame so it stays calm.

    Phrasing (the Announcer decides *whether* to speak, see ANNOUNCE_COOLDOWN):
      * very close -> "Careful, {label} {zone}, very close" (rate 1.3, urgent)
      * approaching -> "{label} {zone}, getting closer"
      * otherwise  -> "{label} {zone}"

    `approaching` is the set of (label, zone) keys from ApproachDetector.

    Returns a list of dicts: {text, rate, urgent, urgency} so the browser can
    speak very-close warnings faster (rate 1.3) and interrupt for them.
    """
    ordered = sorted(
        detections,
        key=lambda d: (URGENCY_RANK[d["urgency"]], _is_person(d["label"]), _area(d)),
        reverse=True,
    )

    spoken = []
    for det in ordered:
        zone = zone_for(det["cx"])
        urgency = det["urgency"]
        # Escalation (urgency went up) speaks immediately; otherwise the Announcer
        # keeps a constant scene quiet until the refresh cooldown elapses.
        if announcer.consider(det["label"], zone, now, urgency_rank=URGENCY_RANK[urgency]) is None:
            continue

        # Title-Case Open Images labels -> plain lowercase; people -> "person".
        name = "person" if _is_person(det["label"]) else det["label"].lower()
        if urgency == URGENCY_VERY_CLOSE:
            text, rate, urgent = f"Careful, {name} {zone}, very close", 1.3, True
        elif (det["label"], zone) in approaching:
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
# VLM, Anthropic, etc. -- WITHOUT touching server.py or the browser. It ports to
# the Pi unchanged (the Pi can call the same cloud API).
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
