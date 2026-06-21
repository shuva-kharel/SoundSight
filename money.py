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
unit-testable); classify_notes() bridges to the banknote classifier (single note, or
best-effort multi-note via tiling).
"""

import logging
from collections import Counter

log = logging.getLogger("soundsight.money")

DENOMS = [1000, 500, 100, 50, 20, 10, 5]   # Nepali rupee notes, high -> low
CLASS_VALUE = {"rs5": 5, "rs10": 10, "rs20": 20, "rs50": 50,
               "rs100": 100, "rs500": 500, "rs1000": 1000}
ADD_COOLDOWN = 2.5     # s: same value within this window is treated as the same note (no double add)
TILE_CONF = 0.80       # multi-note tiling needs HIGH confidence (tiling is noisy)


def class_to_value(name):
    return CLASS_VALUE.get(name)


def to_devanagari(n):
    """Integer -> Devanagari digits, e.g. 650 -> '६५०' (for clear Nepali speech)."""
    return str(int(n)).translate(str.maketrans("0123456789", "०१२३४५६७८९"))


def _en_word(value, count):
    return f"{count} {value}{'s' if count > 1 else ''}"   # "two 500s" -> we pass count as word below


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


def classify_notes(frame, classifier, multi=False, grid=3, min_conf=TILE_CONF):
    """Bridge to the banknote classifier. Returns a list of {value, conf, class}.

    Single (default): classify the whole frame -> 0 or 1 note. Multi (best-effort):
    tile the frame into grid x grid cells and classify each, keeping only HIGH-conf
    cells -- a crude stand-in until a banknote *detection* model is trained. Tiling
    is noisy, hence the high threshold; document this limitation to users.
    """
    if classifier is None or classifier.model is None:
        return []
    if not multi:
        r = classifier.classify(frame)
        v = class_to_value(r.get("class"))
        return [{"value": v, "conf": r.get("confidence", 0.0), "class": r.get("class")}] if (r.get("ok") and v) else []
    h, w = frame.shape[:2]
    found = []
    for gy in range(grid):
        for gx in range(grid):
            tile = frame[gy * h // grid:(gy + 1) * h // grid, gx * w // grid:(gx + 1) * w // grid]
            r = classifier.classify(tile)
            v = class_to_value(r.get("class"))
            if v and r.get("confidence", 0.0) >= min_conf:
                found.append({"value": v, "conf": r.get("confidence", 0.0), "class": r.get("class")})
    return found
