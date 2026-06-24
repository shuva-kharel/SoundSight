"""
faces.py  --  Opt-in known-face recognition for SoundSight (modular; NOT in vision_core).

Greets enrolled people by name during Navigate ("Ram is ahead of you"). Everything is
LOCAL -- no cloud. People are only recognized if they opted in via face_enroll.py.

Providers (auto-detected, lazy): InsightFace buffalo_l (laptop GPU, best) -> the
face_recognition library (lighter) -> disabled (returns "unknown") if neither is
installed. So importing this module never fails, even on a bare Pi.

  FaceMatcher.identify(frame, person_dets, now, frame_idx)
      -> attaches det["name"] to matched person tracks (cached per track_id; every
         N frames; re-greet cooldown). Used continuously in Navigate on the laptop.
  FaceMatcher.who_is_here(frame)
      -> on-demand list of {name, zone, closeness}, nearest/most-central first
         (the "who's here" voice command; the Pi uses ONLY this, not continuous).
  FaceMatcher.enroll(name, frame)
      -> embed the largest face and save it to faces_db/ (used by face_enroll.py and
         the "remember this person as {name}" voice command).

Privacy: embeddings live in faces_db/faces.json (git-ignored). Delete the file to
forget everyone.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("soundsight.faces")

# --- config knobs ----------------------------------------------------------- #
FACE_RECOGNITION_ENABLED = True   # laptop default True; pi_app sets it False (on-demand only)
MATCH_THRESHOLD = 0.45            # cosine similarity to accept a name (InsightFace ~0.45-0.5)
LOWCONF_MARGIN = 0.06             # within this BELOW threshold -> "not sure who", don't guess
FACE_EVERY_N = 10                 # only run recognition every N Navigate frames (it's heavy)
REGREET_SECONDS = 60.0           # re-greet a known person only after this long out of sight
DB_DIR = Path("faces_db")
DB_FILE = DB_DIR / "faces.json"


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _iou_center_in(face_box, person_box):
    """True if the face center sits inside the person box (link face -> person)."""
    fx = (face_box[0] + face_box[2]) / 2
    fy = (face_box[1] + face_box[3]) / 2
    return (person_box[0] <= fx <= person_box[2]) and (person_box[1] <= fy <= person_box[3])


# --------------------------------------------------------------------------- #
# Provider abstraction (lazy import so a bare environment still loads this file)
# --------------------------------------------------------------------------- #
class _Provider:
    name = "none"

    def faces(self, frame_bgr):
        """Return [(bbox[x1,y1,x2,y2], embedding np.array), ...] for the frame."""
        return []


class _InsightFace(_Provider):
    name = "insightface"

    def __init__(self):
        from insightface.app import FaceAnalysis  # heavy; only imported if available
        self.app = FaceAnalysis(name="buffalo_l")
        try:
            self.app.prepare(ctx_id=0)     # GPU
        except Exception:
            self.app.prepare(ctx_id=-1)    # CPU fallback
        log.info("Faces: using InsightFace buffalo_l")

    def faces(self, frame_bgr):
        out = []
        for f in self.app.get(frame_bgr):
            out.append((list(map(int, f.bbox)), np.asarray(f.normed_embedding, dtype=np.float32)))
        return out


class _FaceRecognition(_Provider):
    name = "face_recognition"

    def __init__(self):
        import face_recognition  # noqa: F401  (presence check)
        self._fr = face_recognition
        log.info("Faces: using face_recognition library")

    def faces(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]
        locs = self._fr.face_locations(rgb)          # (top, right, bottom, left)
        encs = self._fr.face_encodings(rgb, locs)
        out = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            out.append(([left, top, right, bottom], np.asarray(enc, dtype=np.float32)))
        return out


def _load_provider():
    for builder in (_InsightFace, _FaceRecognition):
        try:
            return builder()
        except Exception as exc:
            log.info("Faces: %s unavailable (%s)", builder.name, exc)
    log.warning("Faces: no provider installed -- recognition disabled "
                "(pip install insightface onnxruntime, or face_recognition).")
    return None


# --------------------------------------------------------------------------- #
class FaceMatcher:
    def __init__(self, enabled=FACE_RECOGNITION_ENABLED, threshold=MATCH_THRESHOLD):
        self.enabled = enabled
        self.threshold = threshold
        self._provider = None
        self._loaded = False
        self.db = self._load_db()                 # {name: [embedding, ...]}
        self._track_names = {}                     # track_id -> (name, last_seen_time)

    # ---- provider / db ---------------------------------------------------- #
    def _provider_or_none(self):
        if not self._loaded:
            self._loaded = True
            self._provider = _load_provider() if self.enabled else None
        return self._provider

    @property
    def available(self):
        return self._provider_or_none() is not None

    def _load_db(self):
        if DB_FILE.exists():
            try:
                raw = json.loads(DB_FILE.read_text(encoding="utf-8"))
                return {k: [np.asarray(v, np.float32) for v in vs] for k, vs in raw.items()}
            except Exception as exc:
                log.warning("Faces: could not read %s (%s)", DB_FILE, exc)
        return {}

    def _save_db(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        raw = {k: [v.tolist() for v in vs] for k, vs in self.db.items()}
        DB_FILE.write_text(json.dumps(raw), encoding="utf-8")

    # ---- matching --------------------------------------------------------- #
    def _match(self, emb):
        """Return (name, score). name is 'unknown' below threshold, or 'unsure' in
        the low-confidence margin (so we never assert a wrong name)."""
        best_name, best = "unknown", 0.0
        for name, vecs in self.db.items():
            for v in vecs:
                s = _cosine(emb, v)
                if s > best:
                    best_name, best = name, s
        if best >= self.threshold:
            return best_name, best
        if best >= self.threshold - LOWCONF_MARGIN:
            return "unsure", best
        return "unknown", best

    # ---- continuous (Navigate) ------------------------------------------- #
    def identify(self, frame_bgr, person_dets, now=None, frame_idx=0):
        """Attach det['name'] to matched person tracks. Runs only every FACE_EVERY_N
        frames and only for person tracks not already named (cached by track_id)."""
        now = now or time.time()
        # Nobody enrolled -> running face detection (heavy, esp. on CPU) is pure waste:
        # there is no one to match. Skip entirely until at least one face is enrolled.
        if not person_dets or not self.available or not self.db:
            return
        # prune stale track names; allow re-greet after REGREET_SECONDS away
        self._track_names = {tid: (n, t) for tid, (n, t) in self._track_names.items()
                             if now - t < REGREET_SECONDS}
        # reuse cached names without re-embedding
        unnamed = []
        for d in person_dets:
            tid = d.get("track_id")
            if tid is not None and tid in self._track_names:
                d["name"] = self._track_names[tid][0]
                self._track_names[tid] = (self._track_names[tid][0], now)
            else:
                unnamed.append(d)
        if not unnamed or frame_idx % FACE_EVERY_N != 0:
            return
        faces = self._provider.faces(frame_bgr)
        for d in unnamed:
            for fbox, emb in faces:
                if _iou_center_in(fbox, d["box"]):
                    name, score = self._match(emb)
                    if name not in ("unknown", "unsure"):
                        d["name"] = name
                        if d.get("track_id") is not None:
                            self._track_names[d["track_id"]] = (name, now)
                    break

    # ---- on-demand ("who's here") ---------------------------------------- #
    def who_is_here(self, frame_bgr):
        """Return [{name, zone, closeness, area}] for KNOWN faces, nearest first."""
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        out = []
        for fbox, emb in self._provider.faces(frame_bgr):
            name, score = self._match(emb)
            if name in ("unknown",):
                continue
            cx = (fbox[0] + fbox[2]) / 2
            area = max((fbox[2] - fbox[0]) * (fbox[3] - fbox[1]), 0) / float(w * h)
            zone = "on your left" if cx < w / 3 else "on your right" if cx > 2 * w / 3 else "ahead"
            closeness = "close" if area > 0.06 else "nearby" if area > 0.02 else "far"
            out.append({"name": name, "zone": zone, "closeness": closeness,
                        "area": area, "score": round(score, 3),
                        "centrality": abs(cx - w / 2)})
        # nearest / most-central first
        out.sort(key=lambda r: (-r["area"], r["centrality"]))
        return out

    # ---- enrollment ------------------------------------------------------- #
    def enroll(self, name, frame_bgr):
        """Embed the LARGEST face in the frame and store it under `name`. Returns
        (ok, message)."""
        if not self.available:
            return False, "Face recognition is not installed."
        faces = self._provider.faces(frame_bgr)
        if not faces:
            return False, "No face detected. Face the camera and try again."
        # largest face
        fbox, emb = max(faces, key=lambda fe: (fe[0][2] - fe[0][0]) * (fe[0][3] - fe[0][1]))
        self.db.setdefault(name, []).append(emb)
        self._save_db()
        log.info("Faces: enrolled '%s' (%d sample(s))", name, len(self.db[name]))
        return True, f"Saved {name}."
