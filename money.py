"""
money.py  --  Stateful money handling for SoundSight Money mode.

Single-note "how much is this" stays in vision_core.BanknoteClassifier. This module
adds a SESSION TALLY on top:
  * add a note's value to a running total ("added 500, total 650")
  * speak the total / breakdown ("two 500s and one 100, total 1100")
  * clear / undo last
  * "I need to pay X" -> change due, with a suggested give-back breakdown
  * double-count guard so the same note isn't added twice in a row

All amounts are spoken in BOTH English and Nepali. MoneyTally is pure logic (fully
unit-testable); count_notes()/detect_notes() do MULTI-note counting (Path A detection
model if present, else Path B contour + classifier).
"""

import logging
from collections import Counter

log = logging.getLogger("soundsight.money")

DENOMS = [1000, 500, 100, 50, 20, 10, 5]   # Nepali rupee notes, high -> low
CLASS_VALUE = {"rs5": 5, "rs10": 10, "rs20": 20, "rs50": 50,
               "rs100": 100, "rs500": 500, "rs1000": 1000}
ADD_COOLDOWN = 2.5     # s: same value within this window is treated as the same note (no double add)


def class_to_value(name):
    return CLASS_VALUE.get(name)


def to_devanagari(n):
    """Integer -> Devanagari digits, e.g. 650 -> '६५०' (for clear Nepali speech)."""
    return str(int(n)).translate(str.maketrans("0123456789", "०१२३४५६७८९"))


_NUM_EN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
           6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _en_count(n):
    return _NUM_EN.get(n, str(n))


def breakdown_text(counts):
    """counts: {value: how_many} -> ('two 500s and one 100', 'दुई ५००, एक १००')."""
    if not counts:
        return "no notes", "कुनै नोट छैन"
    parts_en, parts_ne = [], []
    for v in sorted(counts, reverse=True):
        c = counts[v]
        parts_en.append(f"{_en_count(c)} {v}{'s' if c > 1 else ''}")
        parts_ne.append(f"{to_devanagari(c)} वटा {to_devanagari(v)}")
    en = " and ".join(parts_en)
    ne = ", ".join(parts_ne)
    return en, ne


def give_back_breakdown(amount):
    """Greedy note breakdown to hand back `amount` -> ({value: count}, en, ne)."""
    counts, rem = {}, int(amount)
    for d in DENOMS:
        if rem // d:
            counts[d] = rem // d
            rem -= d * counts[d]
    en, ne = breakdown_text(counts)
    return counts, en, ne


class MoneyTally:
    """Running tally of added note values, with a double-count cooldown."""

    def __init__(self, cooldown=ADD_COOLDOWN):
        self.notes = []
        self.cooldown = cooldown
        self._last_value, self._last_time = None, -1e9

    def total(self):
        return sum(self.notes)

    def add(self, value, now=0.0):
        """Add one note value. Refuses an identical value within the cooldown
        (guards against double-counting the same note). Returns a spoken result."""
        if value is None:
            return {"ok": False, "reason": "no_value", "total": self.total(),
                    "text": "Hold the note steady and try again.",
                    "text_ne": "नोट स्थिर राखेर फेरि प्रयास गर्नुहोस्।"}
        if value == self._last_value and (now - self._last_time) < self.cooldown:
            return {"ok": False, "reason": "duplicate", "total": self.total(),
                    "text": f"Already added {value}. Move the note away and back to add another.",
                    "text_ne": f"{to_devanagari(value)} पहिले नै जोडिएको छ। अर्को जोड्न नोट हटाएर फेरि देखाउनुहोस्।"}
        self.notes.append(int(value))
        self._last_value, self._last_time = value, now
        tot = self.total()
        return {"ok": True, "added": int(value), "total": tot,
                "text": f"Added {value}, total {tot}.",
                "text_ne": f"{to_devanagari(value)} जोडियो, जम्मा {to_devanagari(tot)} रुपैयाँ।"}

    def undo(self):
        if not self.notes:
            return {"ok": False, "total": 0, "text": "Nothing to undo.",
                    "text_ne": "हटाउन केही छैन।"}
        removed = self.notes.pop()
        self._last_value, self._last_time = None, -1e9
        tot = self.total()
        return {"ok": True, "removed": removed, "total": tot,
                "text": f"Removed {removed}, total {tot}.",
                "text_ne": f"{to_devanagari(removed)} हटाइयो, जम्मा {to_devanagari(tot)} रुपैयाँ।"}

    def clear(self):
        self.notes.clear()
        self._last_value, self._last_time = None, -1e9
        return {"ok": True, "total": 0, "text": "Cleared. Total is zero.",
                "text_ne": "खाली गरियो। जम्मा शून्य।"}

    def total_spoken(self):
        tot = self.total()
        en_b, ne_b = breakdown_text(Counter(self.notes))
        return {"total": tot,
                "text": f"You have {en_b}. Total {tot} rupees.",
                "text_ne": f"तपाईंसँग {ne_b} छ। जम्मा {to_devanagari(tot)} रुपैयाँ।"}

    def change_for(self, price):
        """Change due for a purchase of `price` against the current total."""
        price = int(price)
        tot = self.total()
        if tot < price:
            short = price - tot
            return {"ok": False, "short": short,
                    "text": f"You have {tot}, that is {short} short of {price}.",
                    "text_ne": f"तपाईंसँग {to_devanagari(tot)} छ, {to_devanagari(price)} भन्दा {to_devanagari(short)} कम।"}
        change = tot - price
        if change == 0:
            return {"ok": True, "change": 0,
                    "text": f"Pay {price}, exact amount, no change.",
                    "text_ne": f"{to_devanagari(price)} तिर्नुहोस्, ठ्याक्कै, फिर्ता छैन।"}
        _, en_b, ne_b = give_back_breakdown(change)
        return {"ok": True, "change": change,
                "text": f"Pay {price}, change {change}: {en_b}.",
                "text_ne": f"{to_devanagari(price)} तिर्नुहोस्, फिर्ता {to_devanagari(change)}: {ne_b}।"}


# =========================================================================== #
# MULTI-NOTE counting (announce each note + sum). Two paths:
#   A) a banknote DETECTION model (box + denomination per note) -- best, needs a
#      detection-trained model (see train_banknote_detect.py).
#   B) contour segmentation + the existing CLASSIFIER on each note-like crop --
#      works with the current model, no retrain (less robust).
# Temporal stability: the SET of notes must be stable across a majority of frames
# before we finalize, so the total doesn't flicker mid-count.
# =========================================================================== #
MONEY_MODEL_TYPE = "classify"          # "detect" | "classify" (auto-flips to detect if a model loads)
BANKNOTE_DETECT_MODEL = "models/banknote_detect.pt"  # or models/banknote_detect_ncnn_model/
MIN_NOTE_AREA = 0.015                  # a note must fill >= this fraction of the frame
NOTE_ASPECT = (1.5, 3.2)               # banknote long/short side ratio (~2:1)
COUNT_FRAMES = 8                       # frames sampled for temporal stability
COUNT_STABLE_FRAC = 0.55               # same note-set must appear in this fraction of frames


def _note_candidates(frame):
    """Path B: find note-like rectangles by contour + aspect. Returns [(box, crop)]."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    area = float(h * w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_NOTE_AREA * area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 8:
            continue
        if not (NOTE_ASPECT[0] <= max(rw, rh) / min(rw, rh) <= NOTE_ASPECT[1]):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        crop = frame[y:y + bh, x:x + bw]
        if crop.size:
            out.append(((x, y, x + bw, y + bh), crop))
    return out


def detect_notes(frame, classifier=None, detect_model=None):
    """All notes in ONE frame -> [{value, conf, box}]. Path A if detect_model given,
    else Path B (contour + classifier with the high MONEY_CONF/margin gate)."""
    if detect_model is not None:
        res = detect_model.predict(frame, imgsz=640, verbose=False)[0]
        out = []
        for b in res.boxes:
            name = detect_model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            v = class_to_value(name)
            from vision_core import MONEY_CONF as _MC
            if v and conf >= _MC:
                out.append({"value": v, "conf": round(conf, 3),
                            "box": [float(x) for x in b.xyxy[0]]})
        return out
    if classifier is None or classifier.model is None:
        return []
    out = []
    for box, crop in _note_candidates(frame):
        name, conf, margin = classifier._classify_one(crop)
        if classifier._gate(name, conf, margin):
            out.append({"value": class_to_value(name), "conf": round(conf, 3), "box": list(box)})
    return out


def summarize_notes(notes):
    """[{value,...}] -> spoken breakdown + total (EN + NE), with grouping/plurals."""
    if not notes:
        return {"stable": True, "count": 0, "total": 0,
                "text": "No notes detected.", "text_ne": "कुनै नोट देखिएन।", "notes": []}
    counts = Counter(n["value"] for n in notes)
    total = sum(n["value"] for n in notes)
    en_b, ne_b = breakdown_text(counts)
    return {"stable": True, "count": len(notes), "total": total,
            "text": f"{en_b}. Total {total} rupees.",
            "text_ne": f"{ne_b}। जम्मा {to_devanagari(total)} रुपैयाँ।",
            "notes": [{"value": n["value"], "conf": n.get("conf", 0)} for n in notes]}


def count_notes(frames, classifier=None, detect_model=None):
    """Temporal-stable multi-note count over `frames`. Finalizes only when the SAME
    note-set wins a majority of frames; otherwise asks to hold steady. Logs the
    per-frame sets, the chosen stable set, and the total."""
    if not frames:
        return summarize_notes([])
    per_frame = []
    for f in frames:
        notes = detect_notes(f, classifier, detect_model)
        per_frame.append((tuple(sorted(n["value"] for n in notes)), notes))
    sets = Counter(k for k, _ in per_frame)
    stable_key, n = sets.most_common(1)[0]
    log.info("Money count: per-frame=%s | stable=%s (%d/%d) ",
             [list(k) for k, _ in per_frame], list(stable_key), n, len(frames))
    if n < COUNT_STABLE_FRAC * len(frames):
        return {"stable": False, "count": None, "total": None,
                "text": "Hold the notes steady, I'm counting.",
                "text_ne": "नोटहरू स्थिर राख्नुहोस्, गन्दै छु।"}
    notes = next(nz for k, nz in per_frame if k == stable_key)
    res = summarize_notes(notes)
    log.info("Money count: total=%d count=%d notes=%s", res["total"], res["count"],
             [nn["value"] for nn in res["notes"]])
    return res
