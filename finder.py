"""
finder.py  --  Object Finder mode for SoundSight.

A sub-app of Navigate that lets the user pick ONE detected object from a spoken
pick-list and get guided to it (direction + distance + an audio beacon that speeds
up as the target gets centered AND closer). Pure logic + stdlib + `distance` only
(NO torch/cv2 model code), so the whole generic path runs ON THE PI unchanged and
is unit-testable. The caller (pi_app.py / server.py / index.html) drives it with the
SAME detection dicts Navigate already produces:
    {label, cx, cy, box, area_ratio, distance_m, track_id, ...}

State machine:  idle -> scanning -> picking -> tracking -> (success|lost) -> idle

  f = ObjectFinder(profile="pi", frame_w=416)
  f.start_scan(now)                      # F key / "find mode"
  f.update_scan(dets, now)               # call each frame; returns pick-list when done
  f.choose("A" | "cup" | "the chair")    # pick from the list (key/voice)
  f.find_class("cup", dets, now)         # "find a cup" -> skip the list, lock nearest
  f.guide(dets, now)                     # each frame while tracking -> {text, beacon, found, lost}
  f.room_scan(dets)                      # "what's in this room" -> list + count
  f.exit()                               # Esc/Q / "stop" -> back to Navigate

Personal objects ("find my keys") use PersonalObjectDB (laptop embeddings, objects_db/);
on the Pi alone it degrades gracefully ("that needs the laptop").
"""
import logging
import time

import distance as _dist

log = logging.getLogger("soundsight.finder")

# --- config knobs (all here) ------------------------------------------------ #
FINDER_BEACON = True        # audio beacon on by default (the last-meter guide)
SCAN_SECONDS = 2.5          # collect detections this long before offering the list
LOST_SECONDS = 3.0          # target out of frame this long -> pause beacon, "lost it"
ARM_REACH = 0.7             # within this distance (m) AND centered -> success
SUCCESS_AREA = 0.32         # area_ratio fallback for success when distance is unknown
CENTER_BAND = 0.20          # |cx-center| < this fraction of half-width == "centered"
GUIDE_MIN_GAP = 1.6         # seconds between spoken guidance lines (beacon fills the gaps)
GUIDE_DIST_MAX = 6.0        # distance (m) at which the beacon is slowest
BEACON_MIN_MS = 120         # fastest beacon interval (centered + within reach)
BEACON_MAX_MS = 750         # slowest beacon interval (off-centre / far)
MAX_LIST = 8                # most pick-list entries we offer
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Room inference for the room-scan ("what's in this room"): a room is named by the
# objects present (mirrors vision_core.ROOM_HINTS but kept local so finder stays
# torch-free for the Pi).
ROOM_CONTEXT = {
    "kitchen": {"refrigerator", "sink", "oven", "microwave", "toaster"},
    "bedroom": {"bed"},
    "bathroom": {"toilet", "sink"},
    "living room": {"couch", "tv", "sofa"},
    "dining area": {"dining table", "table"},
    "office": {"laptop", "keyboard", "mouse", "monitor"},
}


def _zone(cx, frame_w):
    if cx < frame_w / 3:
        return "on your left"
    if cx > 2 * frame_w / 3:
        return "on your right"
    return "ahead"


def _clock(cx, frame_w):
    """Coarse clock direction across the forward field of view: 10 (far left) .. 12
    (centre) .. 2 (far right)."""
    frac = max(0.0, min(1.0, cx / max(frame_w, 1)))
    hours = [10, 11, 12, 1, 2]
    return hours[int(round(frac * (len(hours) - 1)))]


def _dsuffix(dm):
    sp = _dist.spoken_distance(dm)
    return f", {sp}" if sp and sp not in ("right in front", "very close") else ""


def infer_room(detections):
    labels = {d.get("label") for d in (detections or [])}
    best, best_n = None, 0
    for room, hints in ROOM_CONTEXT.items():
        n = len(labels & hints)
        if n > best_n:
            best, best_n = room, n
    return best


class ObjectFinder:
    """Per-session Object Finder. One active target at a time; never blocks the
    Navigate loop (every method returns quickly and hazards are handled by the
    caller, which calls guide() only when no hazard is speaking)."""

    def __init__(self, profile="pi", frame_w=416, scan_seconds=SCAN_SECONDS,
                 beacon=FINDER_BEACON):
        self.profile = profile
        self.frame_w = frame_w
        self.scan_seconds = scan_seconds
        self.beacon_enabled = beacon
        self.reset()

    def reset(self):
        self.state = "idle"          # idle | scanning | picking | tracking
        self._scan_end = 0.0
        self._cands = {}             # key -> latest det seen during the scan
        self.pick = []               # [{key, letter, label, zone, distance_m, track_id, cx}]
        self.target = None           # {letter,label,track_id,zone}
        self._last_guide_t = 0.0
        self._last_seen_t = 0.0
        self._last_phrase = ""

    @property
    def active(self):
        return self.state != "idle"

    def set_frame_w(self, w):
        if w:
            self.frame_w = int(w)

    # --- step 1: scan + pick-list ----------------------------------------- #
    def start_scan(self, now):
        """Enter Find mode: begin a short scan. Returns the spoken prompt."""
        self.reset()
        self.state = "scanning"
        self._scan_end = now + self.scan_seconds
        return "Scanning. Hold still and look around slowly."

    def _key(self, det):
        tid = det.get("track_id")
        return tid if tid is not None else (det["label"], _zone(det["cx"], self.frame_w))

    def update_scan(self, detections, now):
        """Call each frame while scanning. Returns None until the scan window ends,
        then ('list', text, entries) or ('empty', text, []) once."""
        if self.state != "scanning":
            return None
        for d in detections or []:
            self._cands[self._key(d)] = d
        if now < self._scan_end:
            return None
        return self._build_list()

    def _build_list(self):
        # distinct objects, nearest first (unknown distance last), capped
        objs = list(self._cands.values())
        objs.sort(key=lambda d: (d.get("distance_m") if d.get("distance_m") is not None else 1e6,
                                 -float(d.get("area_ratio") or 0.0)))
        self.pick = []
        for i, d in enumerate(objs[:MAX_LIST]):
            self.pick.append({
                "key": self._key(d), "letter": LETTERS[i], "label": d["label"],
                "zone": _zone(d["cx"], self.frame_w), "distance_m": d.get("distance_m"),
                "track_id": d.get("track_id"), "cx": d["cx"],
            })
        if not self.pick:
            self.state = "idle"
            txt = "I don't see anything clearly. Try turning slowly and find again."
            self._last_phrase = txt
            return ("empty", txt, [])
        self.state = "picking"
        parts = [f"{p['letter']} - {p['label']} {p['zone']}{_dsuffix(p['distance_m'])}"
                 for p in self.pick]
        txt = "I see: " + ", ".join(parts) + ". Say the letter or press the key to track it."
        self._last_phrase = txt
        return ("list", txt, list(self.pick))

    # --- step 2: choose (letter / name) ----------------------------------- #
    def choose(self, token):
        """Lock onto a pick-list entry by letter ('A') or name ('cup'/'the chair').
        Returns a confirmation string, or None if it couldn't be resolved."""
        if not self.pick or not token:
            return None
        t = token.strip().lower()
        entry = None
        if len(t) == 1 and t.upper() in LETTERS:                    # letter
            entry = next((p for p in self.pick if p["letter"] == t.upper()), None)
        if entry is None:                                           # name (substring)
            t2 = t.replace("the ", "").replace("a ", "").strip()
            entry = next((p for p in self.pick if t2 and (t2 in p["label"] or p["label"] in t2)), None)
        if entry is None:
            return None
        return self._lock(entry["label"], entry["track_id"], entry["zone"], entry["letter"])

    def find_class(self, class_name, detections, now):
        """'find a cup': skip the list, lock onto the NEAREST instance of the class
        in the current view. Returns a confirmation string, or a not-found message."""
        name = (class_name or "").strip().lower().replace("the ", "").replace("a ", "").strip()
        cands = [d for d in (detections or [])
                 if name and (name in d["label"] or d["label"] in name)]
        if not cands:
            self.state = "idle"
            return None
        d = min(cands, key=lambda d: (d.get("distance_m") if d.get("distance_m") is not None else 1e6,
                                      -float(d.get("area_ratio") or 0.0)))
        self._last_seen_t = now
        return self._lock(d["label"], d.get("track_id"), _zone(d["cx"], self.frame_w), None)

    def _lock(self, label, track_id, zone, letter):
        self.target = {"label": label, "track_id": track_id, "zone": zone, "letter": letter}
        self.state = "tracking"
        self._last_guide_t = 0.0
        self._last_seen_t = time.time()
        txt = f"Tracking the {label} {zone}." if zone else f"Tracking the {label}."
        self._last_phrase = txt
        return txt

    # --- step 3: guide ----------------------------------------------------- #
    def _match(self, detections):
        """Find the locked target in this frame. Prefer the ByteTrack id (stable);
        fall back to the nearest same-label box if the id was lost (predict mode)."""
        if not self.target:
            return None
        tid = self.target.get("track_id")
        if tid is not None:
            for d in detections or []:
                if d.get("track_id") == tid:
                    return d
            # id lost -> fall through to label match (so we don't immediately give up)
        same = [d for d in (detections or []) if d["label"] == self.target["label"]]
        if not same:
            return None
        return min(same, key=lambda d: (d.get("distance_m") if d.get("distance_m") is not None else 1e6))

    def _beacon(self, cx, dm):
        fw = self.frame_w
        centered = 1.0 - min(1.0, abs(cx - fw / 2) / (fw / 2))
        close = 0.3 if dm is None else (1.0 - max(0.0, min(1.0, dm / GUIDE_DIST_MAX)))
        score = 0.5 * centered + 0.5 * close
        interval = int(BEACON_MAX_MS - score * (BEACON_MAX_MS - BEACON_MIN_MS))
        return {"active": self.beacon_enabled, "interval_ms": interval,
                "centered": round(centered, 2), "close": round(close, 2)}

    def guide(self, detections, now):
        """Per-frame guidance while tracking. Returns:
        {text|None, beacon, found, lost, done}. `text` is throttled so it doesn't
        chatter; the beacon fills the gaps. Hazard override is the CALLER's job."""
        if self.state != "tracking":
            return {"text": None, "beacon": {"active": False}, "found": False, "lost": False, "done": False}
        det = self._match(detections)
        if det is None:
            if now - self._last_seen_t > LOST_SECONDS:
                txt = "Lost it. Turn slowly."
                self._last_phrase = txt
                # stay in tracking so it re-acquires if it comes back; beacon paused
                return {"text": txt, "beacon": {"active": False}, "found": False, "lost": True, "done": False}
            return {"text": None, "beacon": {"active": False}, "found": False, "lost": False, "done": False}

        self._last_seen_t = now
        cx = det["cx"]
        dm = det.get("distance_m")
        area = float(det.get("area_ratio") or 0.0)
        centered = abs(cx - self.frame_w / 2) <= CENTER_BAND * (self.frame_w / 2)
        within = (dm is not None and dm < ARM_REACH) or (dm is None and area >= SUCCESS_AREA)

        if centered and within:
            label = self.target["label"]
            self.state = "idle"
            txt = f"{label} right in front, reach out."
            self._last_phrase = txt
            return {"text": txt, "beacon": {"active": False}, "found": True, "lost": False, "done": True}

        beacon = self._beacon(cx, dm)
        text = None
        if now - self._last_guide_t >= GUIDE_MIN_GAP:
            text = self._guide_phrase(det, centered, dm)
            self._last_guide_t = now
            self._last_phrase = text
        return {"text": text, "beacon": beacon, "found": False, "lost": False, "done": False}

    def _guide_phrase(self, det, centered, dm):
        label = self.target["label"]
        cx = det["cx"]
        if dm is not None and dm < ARM_REACH:
            return f"{label} right in front, about half a meter, reach out"
        zone = _zone(cx, self.frame_w)
        if centered:
            return f"{label} ahead{_dsuffix(dm)}, getting closer"
        return f"{label} {zone}{_dsuffix(dm)}, at your {_clock(cx, self.frame_w)} o'clock"

    def repeat(self):
        return self._last_phrase or "Nothing to repeat."

    def exit(self):
        self.reset()
        return "Find mode off."

    # --- room scan -------------------------------------------------------- #
    def room_scan(self, detections):
        """'What's in this room': list notable objects (position + distance) and the
        total count, with an inferred room name when possible. Returns the text."""
        dets = detections or []
        if not dets:
            txt = "I don't see anything clearly here."
            self._last_phrase = txt
            return txt
        groups = {}
        for d in dets:
            z = _zone(d["cx"], self.frame_w)
            g = groups.setdefault((d["label"], z), {"n": 0, "dist": None})
            g["n"] += 1
            dm = d.get("distance_m")
            if dm is not None and (g["dist"] is None or dm < g["dist"]):
                g["dist"] = dm
        items = sorted(groups.items(),
                       key=lambda kv: (kv[1]["dist"] if kv[1]["dist"] is not None else 1e6))
        parts = []
        for (label, zone), g in items[:8]:
            head = label if g["n"] == 1 else f"{_count_word(g['n'])} {_plural(label, g['n'])}"
            parts.append(f"{head} {zone}{_dsuffix(g['dist'])}")
        room = infer_room(dets)
        prefix = f"In this {room}: " if room else "In this area: "
        n = len(dets)
        txt = prefix + ", ".join(parts) + f". I can see {n} object" + ("s" if n != 1 else "") + "."
        self._last_phrase = txt
        return txt


_NUMS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine"}


def _count_word(n):
    return _NUMS.get(n, str(n))


def _plural(name, n):
    if n == 1:
        return name
    if name == "person":
        return "people"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


# =========================================================================== #
# PERSONAL objects  --  "remember this as my keys" / "find my keys"
# =========================================================================== #
# Feature-embedding match. On the LAPTOP we use a CLIP image embedding when
# transformers is available (best), else a fast OpenCV color+texture histogram so
# it still works offline. Embeddings are stored per-name in objects_db/. The Pi has
# no GPU/model, so it OFFLOADS to the laptop via /remote/find; alone it says so.
import os

OBJECTS_DB = "objects_db"
PERSONAL_MATCH_MIN = 0.55     # min cosine similarity to call it a match


class PersonalObjectDB:
    def __init__(self, db_dir=OBJECTS_DB):
        self.db_dir = db_dir
        self._embed = None      # callable(crop_bgr) -> 1D float vector (lazy)
        self._engine = None

    def _ensure_embedder(self):
        if self._embed is not None:
            return self._embed
        # Try CLIP (transformers); fall back to a color/edge histogram (cv2 only).
        try:
            import numpy as np
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev).eval()

            def embed(crop_bgr):
                import cv2
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                inp = proc(images=Image.fromarray(rgb), return_tensors="pt").to(dev)
                with torch.no_grad():
                    v = model.get_image_features(**inp)[0].cpu().numpy()
                return v / (np.linalg.norm(v) + 1e-8)

            self._engine = "clip"
            log.info("Personal objects: CLIP embeddings on %s.", dev)
        except Exception as exc:
            log.info("Personal objects: CLIP unavailable (%s) -- using histogram features.", exc)
            import cv2
            import numpy as np

            def embed(crop_bgr):
                hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                v = hist.flatten().astype("float32")
                return v / (np.linalg.norm(v) + 1e-8)

            self._engine = "histogram"
        self._embed = embed
        return self._embed

    def register(self, name, crop_bgr):
        """Store an embedding for a named personal object. Returns (ok, message)."""
        try:
            import numpy as np
            embed = self._ensure_embedder()
            os.makedirs(self.db_dir, exist_ok=True)
            vec = embed(crop_bgr)
            np.save(os.path.join(self.db_dir, _safe(name) + ".npy"), vec)
            return True, f"Okay, I'll remember your {name}."
        except Exception as exc:
            log.warning("register('%s') failed: %s", name, exc)
            return False, "I couldn't save that object."

    def match(self, name, crops):
        """Best-matching crop index for a stored name among candidate crops, or None.
        `crops` is a list of BGR arrays. Returns (index, score) or (None, 0)."""
        try:
            import numpy as np
            path = os.path.join(self.db_dir, _safe(name) + ".npy")
            if not os.path.exists(path) or not crops:
                return None, 0.0
            ref = np.load(path)
            embed = self._ensure_embedder()
            best_i, best_s = None, 0.0
            for i, c in enumerate(crops):
                v = embed(c)
                s = float(np.dot(ref, v))
                if s > best_s:
                    best_i, best_s = i, s
            if best_s >= PERSONAL_MATCH_MIN:
                return best_i, best_s
            return None, best_s
        except Exception as exc:
            log.warning("match('%s') failed: %s", name, exc)
            return None, 0.0

    def known(self):
        if not os.path.isdir(self.db_dir):
            return []
        return [f[:-4] for f in os.listdir(self.db_dir) if f.endswith(".npy")]


def _safe(name):
    return "".join(c for c in (name or "obj").lower() if c.isalnum() or c in "-_") or "obj"
