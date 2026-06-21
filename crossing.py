"""
crossing.py  --  Street-crossing safety for SoundSight.

COCO already detects "traffic light"; this module crops each one and classifies its
lit color by HSV (red / amber / green / unknown), then turns that into SAFE spoken
guidance. It is a sub-behavior of Navigate and answers the "can I cross" voice
command. Pure OpenCV + NumPy, so it runs on the Pi NCNN path unchanged.

SAFETY RULES (deliberate):
  * Never give an absolute guarantee -- green says "looks clear", and always adds a
    caution to listen for vehicles.
  * Hysteresis: a state must hold for CONFIRM_FRAMES before we announce a change, so
    flicker can't flip red<->green. Announce only on CONFIRMED CHANGES.
  * Fail safe: no light / unclear color / conflicting lights -> "be careful", never
    a guessed green. Every such path is logged with the reason.

  CrossingMonitor().update(detections, frame, now) -> announcement dict or None
  CrossingMonitor().query(detections, frame)       -> on-demand spoken status (voice cmd)
  classify_light_color(crop_bgr)                    -> (state, confidence)
"""

import logging

import cv2
import numpy as np

log = logging.getLogger("soundsight.crossing")

# --- config knobs ----------------------------------------------------------- #
CONFIRM_FRAMES = 4        # same state must hold this many frames before announcing a change
MIN_AREA_RATIO = 0.004    # a light must fill at least this fraction of the frame (gate out
                          # distant lights so they don't trigger crossing guidance)
MIN_LIT_FRACTION = 0.04   # the dominant color must be at least this fraction of the crop,
                          # else the light is "unknown" (off / too far / glare)
DOMINANCE_RATIO = 1.5     # dominant color must beat the runner-up by this factor
SAT_MIN, VAL_MIN = 80, 110  # only count vivid, bright (i.e. LIT) pixels

# HSV hue ranges (OpenCV H is 0..179). Red wraps around 0.
_HSV = {
    "red":   [((0, SAT_MIN, VAL_MIN), (10, 255, 255)), ((170, SAT_MIN, VAL_MIN), (179, 255, 255))],
    "amber": [((11, SAT_MIN, VAL_MIN), (32, 255, 255))],
    "green": [((40, SAT_MIN, VAL_MIN), (90, 255, 255))],
}

# bilingual spoken lines (text, text_ne)
SAY = {
    "red":    ("Red light, please wait.", "रातो बत्ती, कृपया पर्खनुहोस्।"),
    "amber":  ("Amber light, please wait.", "पहेँलो बत्ती, कृपया पर्खनुहोस्।"),
    "green":  ("Green light, looks clear to cross. Vehicles may still be moving, listen before you step.",
               "हरियो बत्ती, काट्न ठीक देखिन्छ। गाडी चलिरहेका हुन सक्छन्, पाइला चाल्नु अघि सुन्नुहोस्।"),
    "changed_stop": ("Light changed, stop and wait.", "बत्ती परिवर्तन भयो, रोकिनुहोस् र पर्खनुहोस्।"),
    "unclear": ("Traffic light unclear, please be careful.", "ट्राफिक बत्ती अस्पष्ट छ, कृपया सावधान हुनुहोस्।"),
    "none":    ("No traffic light seen, please be careful.", "ट्राफिक बत्ती देखिएन, कृपया सावधान हुनुहोस्।"),
}


def _mask_count(hsv, ranges):
    total = 0
    for lo, hi in ranges:
        total += int(cv2.inRange(hsv, np.array(lo), np.array(hi)).sum() // 255)
    return total


def classify_light_color(crop_bgr):
    """Return (state, confidence) for a traffic-light crop. state in
    red/amber/green/unknown; confidence is the dominant color's pixel fraction."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    px = max(crop_bgr.shape[0] * crop_bgr.shape[1], 1)
    counts = {c: _mask_count(hsv, r) for c, r in _HSV.items()}
    state = max(counts, key=counts.get)
    top = counts[state]
    runner = max((v for c, v in counts.items() if c != state), default=0)
    frac = top / px
    if frac < MIN_LIT_FRACTION or top < DOMINANCE_RATIO * max(runner, 1):
        return "unknown", round(frac, 3)
    return state, round(frac, 3)


def _is_pedestrian_shape(box):
    """Heuristic: pedestrian signals are roughly square; vehicle stacks are tall.
    COCO can't truly separate them, so this only *prefers* squarer boxes."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    if w <= 0 or h <= 0:
        return False
    return (h / w) < 1.6


class CrossingMonitor:
    """Per-Navigate-session crossing state with hysteresis + mid-cross monitoring."""

    def __init__(self, confirm_frames=CONFIRM_FRAMES, min_area=MIN_AREA_RATIO):
        self.confirm_frames = confirm_frames
        self.min_area = min_area
        self.confirmed = None      # last CONFIRMED state we announced
        self._cand = None          # candidate state being counted toward confirmation
        self._count = 0

    # ---- read all gated lights in the frame -> a single aggregate state ------ #
    def _aggregate(self, detections, frame):
        lights = [d for d in detections
                  if d.get("label") == "traffic light" and d.get("area_ratio", 0) >= self.min_area]
        if not lights:
            return "none", []
        h, w = frame.shape[:2]
        readings = []
        for d in lights:
            x1, y1, x2, y2 = (int(max(0, v)) for v in d["box"])
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            state, conf = classify_light_color(frame[y1:y2, x1:x2])
            readings.append({"state": state, "conf": conf,
                             "ped": _is_pedestrian_shape(d["box"]), "area": d.get("area_ratio", 0)})
        if not readings:
            return "none", []
        # Prefer pedestrian-shaped lights when any are present (improvement #1).
        considered = [r for r in readings if r["ped"]] or readings
        known = [r for r in considered if r["state"] != "unknown"]
        if not known:
            return "unknown", readings
        states = {r["state"] for r in known}
        if len(states) > 1:
            return "conflict", readings   # lights disagree -> fail safe
        return known[0]["state"], readings

    def update(self, detections, frame, now=None):
        """Per-frame. Returns an announcement dict only on a CONFIRMED state change."""
        agg, readings = self._aggregate(detections, frame)
        # Map non-color aggregates: don't spam "no light" every frame -- only colors
        # drive confirmed announcements; none/unknown/conflict reset the candidate.
        candidate = agg if agg in ("red", "amber", "green") else None

        if candidate is None:
            self._cand, self._count = None, 0
            return None

        if candidate == self._cand:
            self._count += 1
        else:
            self._cand, self._count = candidate, 1

        if self._count < self.confirm_frames:
            return None
        if candidate == self.confirmed:
            return None   # already announced this confirmed state

        prev = self.confirmed
        self.confirmed = candidate
        # mid-cross safety: green -> amber/red means STOP, announce urgently (#4)
        if prev == "green" and candidate in ("amber", "red"):
            log.info("CROSSING change green->%s: STOP", candidate)
            en, ne = SAY["changed_stop"]
            return {"text": en, "text_ne": ne, "rate": 1.3, "urgent": True, "urgency": "very close"}
        en, ne = SAY[candidate]
        log.info("CROSSING confirmed %s", candidate)
        return {"text": en, "text_ne": ne, "rate": 1.0,
                "urgent": candidate != "green", "urgency": "near"}

    def query(self, detections, frame):
        """On-demand status for the 'can I cross' voice command. Bypasses hysteresis
        but stays fail-safe; always returns something to say."""
        agg, readings = self._aggregate(detections, frame)
        if agg in ("red", "amber", "green"):
            key = agg
        elif agg == "conflict":
            log.info("CROSSING query: conflicting lights %s", [r["state"] for r in readings])
            key = "unclear"
        elif agg == "unknown":
            log.info("CROSSING query: color unclear")
            key = "unclear"
        else:
            log.info("CROSSING query: no light detected")
            key = "none"
        en, ne = SAY[key]
        return {"text": en, "text_ne": ne, "rate": 1.0,
                "urgent": key != "green", "urgency": "near"}
