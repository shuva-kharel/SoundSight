"""
server.py  --  FastAPI web layer for SoundSight.

Owns transport only. All AI lives in vision_core (so it ports to the Pi):
  * GET  /              -> serves index.html (the whole UI)
  * WS   /ws/navigate   -> base64 JPEG frames in, JSON detections out
  * POST /ocr           -> one JPEG in, {text, lang} out (Nepali + English)
  * POST /tts           -> text in, WAV audio out (offline Nepali speech, espeak-ng)
  * POST /describe      -> one JPEG in, scene description out (Gemini)

The browser owns the webcam (getUserMedia) and the speech (Web Speech API for
English; it plays the /tts WAV for Nepali). Everything is offline except Describe.
"""

import argparse
import base64
import datetime
import ipaddress
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

import commands
import finder as finder_mod
import frame_quality as fq
import labels as labels_mod
import money as money_mod
import nepali_phrases
from crossing import CrossingMonitor
from faces import FaceMatcher
from vision_core import DEVICE, FEATURE_PROFILE, NE_ZONES, zone_for
from vision_core import (
    AnnouncementManager,
    ApproachDetector,
    BanknoteClassifier,
    OfflineTTS,
    QualityGate,
    SceneDescriber,
    VisionCore,
    get_sub_mode,
    how_far_phrase,
    scan_area_phrase,
    select_announcements,
    set_sub_mode,
)

# OCR tuning (the /ocr path).
OCR_CONF = 0.30          # drop OCR fragments below this per-box confidence
OCR_RESCUE_CONF = 0.50   # if mean conf is under this (or empty), retry rotated 90/180/270
OCR_ROI = True           # crop to the central region before OCR (focus on the held label)
OCR_ROI_FRAC = 0.85      # keep this central fraction (gentle; lower it to crop more)
# Engine: laptop defaults to PaddleOCR (PP-OCRv4, faster + better Devanagari) when
# installed; otherwise EasyOCR. The Pi always uses EasyOCR (CPU). Both pre-warm at
# startup so the FIRST Read is fast, never a 30 s model download.
import os as _os
OCR_ENGINE = _os.environ.get("OCR_ENGINE", "auto")   # "auto" | "paddle" | "easyocr"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("soundsight.server")

HERE = Path(__file__).parent

# Load a .env file from the project root (if present) so secrets like
# GEMINI_API_KEY are picked up automatically -- no need to set them by hand each
# session. Real environment variables still work and take precedence.
try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except ImportError:
    pass  # python-dotenv is optional

app = FastAPI(title="SoundSight")

# CORS: allow the Pi's browser dashboard (any LAN origin) to call /remote/* directly.
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["X-Processing-Time"])
except Exception as _exc:   # CORS is a nicety, never fatal
    log.warning("CORS middleware not enabled: %s", _exc)

# Live request accounting for /remote/ping (so the Pi dashboard can show what the
# laptop is doing) + an X-Processing-Time header on every response.
_inflight = {"n": 0}


@app.middleware("http")
async def _timing(request: Request, call_next):
    _inflight["n"] += 1
    t0 = time.time()
    try:
        resp = await call_next(request)
    finally:
        _inflight["n"] -= 1
    ms = (time.time() - t0) * 1000.0
    resp.headers["X-Processing-Time"] = f"{ms:.0f}"
    if request.url.path.startswith("/remote/") and ms > 250:
        log.warning("SLOW %s: %.0f ms (target <150ms)", request.url.path, ms)
    return resp


@app.exception_handler(Exception)
async def graceful_error(request: Request, exc: Exception):
    """SAFETY NET: any unhandled error in a one-shot endpoint returns a spoken-
    friendly message (HTTP 200) instead of a 500, so a single mode's hiccup never
    crashes the UX. Navigate is a separate WebSocket, already wrapped per-frame."""
    log.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=200, content={
        "ok": False, "available": True,
        "text": "Sorry, that didn't work. Please try again.",
        "text_ne": "माफ गर्नुहोस्, त्यो भएन। फेरि प्रयास गर्नुहोस्।"})


def _gpu_diagnostics():
    """Loud, actionable check. Three outcomes: (1) GPU works -> log and return;
    (2) CUDA reports available but the torch build lacks kernels for this GPU's arch
    (e.g. RTX 50-series sm_120) -> it silently runs on CPU -> warn + fix; (3) no CUDA."""
    reason = "no CUDA build / GPU not detected"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            archs = torch.cuda.get_arch_list()
            if sm in archs:
                log.info("GPU: %s (%s, CUDA %s) -- GPU acceleration ACTIVE.", name, sm, torch.version.cuda)
                return
            reason = (f"{name} is {sm}, but this torch build only supports {archs} -- "
                      f"so kernels fall back to CPU")
    except Exception as exc:
        reason = f"torch CUDA query failed: {exc}"
    log.warning("=" * 64)
    log.warning("RUNNING ON CPU -- the GPU is NOT accelerating inference.")
    log.warning("  Why: %s.", reason)
    log.warning("  Effect: detection runs the light CPU model (yolo11n@320); for full")
    log.warning("  accuracy + speed + the depth distance model, enable the GPU.")
    log.warning("  RTX 50-series (Blackwell/sm_120) needs the CUDA 12.8 torch build:")
    log.warning("    pip uninstall -y torch torchvision")
    log.warning("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    log.warning("  Verify:  python -c \"import torch;print(torch.cuda.is_available(),torch.cuda.get_arch_list())\"")
    log.warning("=" * 64)


_gpu_diagnostics()

# Load YOLO once at startup (logs which device it uses).
vision = VisionCore()

# Describe mode (Gemini). Cheap to construct; the API client is built lazily on
# the first /describe call (and only if a key is set).
describer = SceneDescriber()

# Offline Nepali speech (espeak-ng). Cheap to construct; the binary is located on
# the first /tts call. No model download, no internet.
tts_engine = OfflineTTS()

# Money mode: the trained banknote classifier, loaded ONCE here at startup (not
# per request). If models/banknote.pt doesn't exist yet, Money mode reports
# 'not trained' until you run train_banknote.py.
banknote = BanknoteClassifier()

# Optional banknote DETECTION model for multi-note counting (Path A). If absent,
# /money/count falls back to Path B (contour segmentation + the classifier).
banknote_detect = None
for _p in (money_mod.BANKNOTE_DETECT_MODEL, "models/banknote_detect_ncnn_model"):
    if Path(_p).exists():
        from ultralytics import YOLO
        banknote_detect = YOLO(_p, task="detect")
        log.info("Money: banknote DETECTION model loaded from %s (multi-note Path A)", _p)
        break

# Latest Navigate detections, cached so Describe's OFFLINE fallback and the "can I
# cross" voice command can use them WITHOUT re-running the shared tracking model
# (which would corrupt Navigate's ByteTrack state). Updated by the Navigate loop.
latest_nav = {"detections": [], "t": 0.0, "cross": None, "fw": 416}
personal_db = finder_mod.PersonalObjectDB()   # personal-object embeddings (/find/register, /remote/find)


def _say(text, ne=None):
    """Wrap a plain string as a speak item (the browser/Pi speech shape)."""
    return {"text": text, "text_ne": ne or text, "rate": 1.0,
            "urgent": False, "urgency": "near"}


def _finder_state(finder):
    """Serialise ObjectFinder state for the browser HUD (pick-list + locked target)."""
    return {
        "state": finder.state,
        "pick": [{"letter": p["letter"], "label": p["label"], "zone": p["zone"],
                  "distance_m": p["distance_m"]} for p in finder.pick],
        "target": (finder.target or {}).get("label"),
    }
NAV_FRESH_SECS = 5.0   # only use cached detections/crossing this recent

# Server-side money tally (single-user demo). Voice/keys add notes & compute change.
tally = money_mod.MoneyTally()

# Opt-in known-face recognition (laptop). Degrades to "disabled" if no provider is
# installed (insightface / face_recognition) -- never blocks startup.
face_matcher = FaceMatcher(enabled=True)

# OCR is pre-warmed in a background thread at startup (prewarm_ocr below) so the
# FIRST Read is fast -- no 30 s model download surprise mid-demo. Still lazy-safe:
# get_ocr_reader() builds it on demand if the warm-up hasn't finished yet.
_ocr_reader = None
_ocr_engine = None        # "paddle" | "easyocr" (which one actually loaded)


class _PaddleReader:
    """Adapter so PaddleOCR exposes the same readtext(img, detail=1) -> [(box,text,conf)]
    interface EasyOCR uses, keeping _ocr_pass/_ocr_best unchanged."""

    def __init__(self, gpu):
        from paddleocr import PaddleOCR
        # PP-OCRv4 multilingual handles Devanagari + Latin; angle classifier on.
        self._ocr = PaddleOCR(use_angle_cls=True, lang="devanagari", use_gpu=gpu, show_log=False)

    def readtext(self, img, detail=1):
        out = self._ocr.ocr(img, cls=True) or []
        rows = out[0] if out and isinstance(out[0], list) else out
        res = []
        for line in (rows or []):
            try:
                box, (text, conf) = line[0], line[1]
                res.append((box, text, float(conf)))
            except Exception:
                continue
        return res


def get_ocr_reader():
    global _ocr_reader, _ocr_engine
    if _ocr_reader is not None:
        return _ocr_reader
    from vision_core import FEATURE_PROFILE, LAPTOP_OCR_GPU
    gpu = (FEATURE_PROFILE == "laptop" and LAPTOP_OCR_GPU)
    want_paddle = OCR_ENGINE == "paddle" or (OCR_ENGINE == "auto" and FEATURE_PROFILE == "laptop")
    if want_paddle:
        try:
            log.info("Initializing PaddleOCR (PP-OCRv4, devanagari, gpu=%s)...", gpu)
            _ocr_reader = _PaddleReader(gpu)
            _ocr_engine = "paddle"
            log.info("PaddleOCR ready.")
            return _ocr_reader
        except Exception as exc:
            log.warning("PaddleOCR unavailable (%s) -- falling back to EasyOCR.", exc)
    import easyocr
    log.info("Initializing EasyOCR for Nepali + English (gpu=%s)...", gpu)
    _ocr_reader = easyocr.Reader(["ne", "en"], gpu=gpu)   # one reader handles both scripts
    _ocr_engine = "easyocr"
    log.info("EasyOCR ready.")
    return _ocr_reader


def prewarm_ocr():
    """Load the OCR model in a background thread at startup so the first Read is fast."""
    import threading

    def _warm():
        try:
            log.info("Pre-warming OCR model (one-time, in background)...")
            get_ocr_reader()
            log.info("OCR pre-warmed and READY (engine=%s).", _ocr_engine)
        except Exception as exc:
            log.warning("OCR pre-warm failed (%s) -- will retry on first Read.", exc)

    threading.Thread(target=_warm, daemon=True).start()


@app.on_event("startup")
async def _startup_prewarm():
    prewarm_ocr()


def detect_lang(text: str) -> str:
    """Flag the script by majority of letters: 'ne' if Devanagari letters
    outnumber Latin letters, else 'en'.

    EasyOCR doesn't tag language per word, so we infer from the Unicode block
    (Devanagari = U+0900..U+097F). Devanagari *digits* (U+0966..U+096F) are
    ignored: the bilingual model often emits e.g. '१' for '1' in otherwise-English
    text, and a stray digit shouldn't flip the whole result to Nepali. The browser
    uses this to pick speech: English -> Web Speech API; Nepali -> the /tts MP3.
    """
    deva = sum(1 for ch in text if "ऀ" <= ch <= "ॿ" and not ("०" <= ch <= "९"))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return "ne" if deva > latin else "en"


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode raw JPEG bytes into a BGR frame."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_b64_frame(text: str) -> np.ndarray:
    """Decode a (possibly data-URL prefixed) base64 JPEG string into a BGR frame."""
    if "," in text:  # strip "data:image/jpeg;base64," prefix if present
        text = text.split(",", 1)[1]
    return decode_jpeg(base64.b64decode(text))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.websocket("/ws/navigate")
async def ws_navigate(ws: WebSocket):
    await ws.accept()
    manager = AnnouncementManager()  # per-connection event/tracking memory
    approach = ApproachDetector()    # per-connection approach-history state
    gate = QualityGate()             # per-connection bad-frame guard
    crossing_mon = CrossingMonitor() # per-connection traffic-light state
    from finder import ObjectFinder
    finder = ObjectFinder(profile="laptop")  # per-connection Object Finder state
    last_dets = []                   # most recent detections (for find-by-class/room)
    greeted = set()                  # track_ids already greeted by name this session
    frames, t0 = 0, time.time()
    log.info("Navigate client connected.")

    async def _control(msg):
        """Handle a Find-mode control message from the browser; returns a speak item
        to send back, or None."""
        cmd = msg.get("cmd")
        now = time.time()
        finder.set_frame_w(latest_nav.get("fw") or 416)
        if cmd == "find_start":
            return _say(finder.start_scan(now))
        if cmd == "find_class":
            conf = finder.find_class(msg.get("name", ""), last_dets, now)
            return _say(conf or (f"I don't see a {msg.get('name')}. " + finder.start_scan(now)))
        if cmd == "find_choose":
            return _say(finder.choose(msg.get("pick", "")) or "Say the letter or the name.")
        if cmd == "room":
            return _say(finder.room_scan(last_dets))
        if cmd == "find_exit":
            finder.exit()
            return _say("Find mode off.")
        return None

    try:
        while True:
            text = await ws.receive_text()
            # A control message is JSON ({"cmd": ...}); a frame is base64 (not JSON).
            if text and text[0] == "{":
                try:
                    item = await _control(__import__("json").loads(text))
                    await ws.send_json({"boxes": [], "speak": [item] if item else [],
                                        "finder": _finder_state(finder)})
                    continue
                except Exception:
                    pass
            frame = decode_b64_frame(text)
            if frame is None:
                continue

            # One bad frame must never break the loop -- wrap the whole step.
            try:
                now = time.time()
                assess, warn = gate.check(frame, now)
                if assess.get("blocking"):
                    # Genuinely unusable frame (dark/blown-out/no frame): skip
                    # detection; pass the gentle warning if due. Soft-focus frames
                    # are NOT blocked -- detection still runs on them.
                    await ws.send_json({
                        "boxes": [], "speak": [warn] if warn else [],
                        "fw": int(frame.shape[1]), "fh": int(frame.shape[0]),
                        "quality": assess,
                    })
                    continue

                detections = vision.detect(frame, assessment=assess)  # enhances internally
                last_dets = detections
                approaching = approach.approaching_objects(detections)
                speak = select_announcements(detections, manager, approaching, now)
                # In Find mode, ambient navigate yields -- only HAZARDS still interrupt.
                if finder.active:
                    speak = [s for s in speak if s.get("urgency") == "very close"]

                # Known-face greetings: name confirmed person tracks (heavy -> only
                # every N frames inside identify()), greet each new track once,
                # nearest/most-central first. Faces stay modular (faces.py).
                persons = [d for d in detections if d["label"] == "person"]
                if persons and face_matcher.available:
                    face_matcher.identify(frame, persons, now, frames)
                    fw = frame.shape[1]
                    named = sorted((d for d in persons if d.get("name")
                                    and d.get("track_id") not in greeted),
                                   key=lambda d: -d.get("area_ratio", 0))
                    face_speak = []
                    for d in named:
                        greeted.add(d.get("track_id"))
                        z = zone_for(d["cx"])   # FRAME_W was set by the detect() above
                        close = d.get("area_ratio", 0) > 0.05
                        en = f"{d['name']} {z}" + (", close" if close else "")
                        ne = f"{d['name']} {NE_ZONES.get(z, z)}" + (", नजिक" if close else "")
                        face_speak.append({"text": en, "text_ne": ne, "rate": 1.0,
                                           "urgent": False, "urgency": "near"})
                    speak = face_speak + speak
                    greeted.intersection_update({d.get("track_id") for d in persons})

                # Street-crossing sub-behavior: use the RAW frame for true light color
                # (CLAHE would shift hues). Auto-announced ONLY in STREET sub-mode;
                # confirmed state changes are spoken first.
                street = get_sub_mode() == "street"
                cross_ann = crossing_mon.update(detections, frame, now) if street else None
                if cross_ann:
                    speak = [cross_ann] + speak
                has_light = any(d["label"] == "traffic light" for d in detections)
                latest_nav["cross"] = crossing_mon.query(detections, frame) if has_light else None
                latest_nav["detections"], latest_nav["t"] = detections, now  # Describe/cross use these
                latest_nav["free_space"] = getattr(vision.distance, "free_space_m", None)
                latest_nav["fw"] = int(frame.shape[1])
                finder.set_frame_w(frame.shape[1])

                # --- Object Finder: scan -> pick-list -> guide + beacon -------- #
                beacon = None
                if finder.state == "scanning":
                    res = finder.update_scan(detections, now)
                    if res:
                        speak = [_say(res[1])] + speak
                elif finder.state == "tracking":
                    g = finder.guide(detections, now)
                    beacon = g["beacon"]
                    if g["text"] and not any(s.get("urgency") == "very close" for s in speak):
                        speak = [_say(g["text"])] + speak

                await ws.send_json({
                    "boxes": [
                        {"label": d["label"], "confidence": d["confidence"],
                         "box": d["box"], "urgency": d["urgency"],
                         "distance_m": d.get("distance_m"), "track_id": d.get("track_id")}
                        for d in detections
                    ],
                    "speak": speak,
                    "fw": int(frame.shape[1]),
                    "fh": int(frame.shape[0]),
                    "quality": assess,
                    "sub_mode": get_sub_mode(),
                    "light": crossing_mon.confirmed,
                    "finder": _finder_state(finder),
                    "beacon": beacon,
                })

                frames += 1
                if frames % 30 == 0:
                    # WINDOWED fps (rate over the last 30 frames) -- not a cumulative
                    # average, which would stay low forever after a slow startup.
                    fps = 30 / (time.time() - t0)
                    t0 = time.time()
                    log.info("Navigate FPS: %.1f  (last frame: %d detections, %s)",
                             fps, len(detections), assess["reason"])
            except WebSocketDisconnect:
                raise
            except Exception as exc:   # log and keep going -- don't drop the client
                log.exception("Navigate frame error (continuing): %s", exc)
    except WebSocketDisconnect:
        log.info("Navigate client disconnected after %d frames.", frames)


def _ocr_pass(reader, img):
    """One EasyOCR pass with per-box confidence. Returns (text, mean_conf):
    drops fragments below OCR_CONF, then orders boxes top-to-bottom, left-to-right
    so the result reads in natural order."""
    results = reader.readtext(img, detail=1)  # [(box, text, conf), ...]
    good = [(b, t, c) for (b, t, c) in results if c >= OCR_CONF and t.strip()]
    if not good:
        return "", 0.0

    def top_left(box):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(min(ys) / 20), min(xs))  # bucket rows so a line reads L->R

    good.sort(key=lambda r: top_left(r[0]))
    text = " ".join(t.strip() for _, t, _ in good)
    mean_conf = sum(c for _, _, c in good) / len(good)
    return text, mean_conf


def _center_roi(frame, frac=OCR_ROI_FRAC):
    """Crop the central `frac` of the frame -- the user points at the label/sign, so
    trimming the noisy border focuses OCR. Gentle by default; tune OCR_ROI_FRAC."""
    if not OCR_ROI or frame is None:
        return frame
    h, w = frame.shape[:2]
    m = (1.0 - frac) / 2.0
    return frame[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))]


def _ocr_best(reader, frame):
    """ROI-crop -> preprocess (CLAHE/upscale/deskew) -> OCR; if empty or
    low-confidence, retry rotated 90/180/270 (notes & signs are often held sideways)
    and keep the best. Returns (text, mean_conf, lang)."""
    base = fq.read_preprocess(_center_roi(frame))
    best_text, best_conf = _ocr_pass(reader, base)
    if not best_text or best_conf < OCR_RESCUE_CONF:
        for k in (1, 2, 3):
            t, c = _ocr_pass(reader, fq.rotate90(base, k))
            if t and (c > best_conf or not best_text):
                best_text, best_conf = t, c
    lang = detect_lang(best_text) if best_text else "en"
    return best_text, best_conf, lang


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), label: bool = Form(False)):
    """Bilingual (Nepali + English) OCR -> {text, lang, conf}. Refuses to read junk
    frames (too dark/blurry) and guides the user instead of returning garbage.

    label=True runs the medicine/packaged-goods parser (labels.py) ON TOP of the
    OCR text and returns a product/expiry/dosage summary (normal Read is unchanged)."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"text": "No image received.", "lang": "en", "conf": 0.0}

    assess = fq.assess(frame)
    if not assess["ok"] and assess["reason"] in ("too_dark", "too_blurry", "no_frame"):
        log.info("OCR skipped: frame %s (brightness=%s blur=%s)",
                 assess["reason"], assess["brightness"], assess["blur_score"])
        return {"text": "Hold the label flat and steady in better light, then try again."
                if label else "Hold steady and move to better light, then try again.",
                "lang": "en", "conf": 0.0, "quality": assess["reason"]}

    reader = get_ocr_reader()
    t0 = time.time()
    text, conf, lang = await run_in_threadpool(_ocr_best, reader, frame)
    log.info("OCR: %d chars, mean_conf=%.2f, lang=%s, label=%s in %.2fs",
             len(text), conf, lang, label, time.time() - t0)
    if not text:
        return {"text": "No readable text found.", "lang": "en", "conf": 0.0}

    if label:   # parse expiry/dosage/product on top of the raw OCR text
        parsed = labels_mod.parse_label(text)
        log.info("Label: product=%r expiry=%s dosage=%s",
                 parsed["product"], parsed["expiry"], parsed["dosage"])
        return {"text": parsed["text"], "text_ne": parsed["text_ne"], "lang": lang,
                "conf": round(conf, 3), "label": parsed}
    return {"text": text, "lang": lang, "conf": round(conf, 3)}


def _gtts_mp3(text):
    """gTTS Nepali MP3 bytes, or None if gTTS isn't installed / there's no internet."""
    try:
        import io

        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="ne").write_to_fp(buf)
        return buf.getvalue()
    except Exception as exc:
        log.info("gTTS unavailable (offline?): %s", exc)
        return None


@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("ne")):
    """Speak `text`, choosing the best Nepali source: a PRE-RECORDED file for the
    fixed vocabulary (best, offline) -> gTTS (good, online) for arbitrary OCR text
    -> espeak-ng (offline fallback). Returns audio bytes (mp3 or wav)."""
    text = (text or "").strip()
    if not text:
        return PlainTextResponse("", status_code=400)
    if lang == "ne":
        text = nepali_phrases.clean_ocr_ne(text)   # normalize OCR'd Devanagari first

    # 1) pre-recorded fixed phrase (denominations / navigation / warnings)
    pre = nepali_phrases.prerecorded_path(text)
    if pre is not None:
        log.info("TTS[prerecorded]: %r -> %s", text, pre.name)
        mime = "audio/mpeg" if pre.suffix == ".mp3" else "audio/wav"
        return Response(content=pre.read_bytes(), media_type=mime)

    # 2) gTTS online for arbitrary Nepali text (nicer than espeak)
    if lang == "ne":
        mp3 = await run_in_threadpool(_gtts_mp3, text)
        if mp3:
            log.info("TTS[gtts]: %d chars", len(text))
            return Response(content=mp3, media_type="audio/mpeg")

    # 3) espeak-ng offline fallback (low quality, but always works)
    try:
        audio = await run_in_threadpool(tts_engine.synth, text, lang)
        log.info("TTS[espeak]: %d chars (lang=%s)", len(text), lang)
        return Response(content=audio, media_type="audio/wav")
    except Exception as exc:
        log.warning("Offline TTS failed: %s", exc)
        return PlainTextResponse(f"TTS failed: {exc}", status_code=503)


@app.post("/describe", response_class=PlainTextResponse)
async def describe(file: UploadFile = File(...), lang: str = Form("en")):
    frame = decode_jpeg(await file.read())
    if frame is None:
        return "No image received."
    t0 = time.time()

    # Enhance once (don't double-enhance inside detect). Only refuse if it's
    # genuinely too dark / no image -- Gemini handles soft-focus fine, so blur does
    # NOT block Describe.
    enhanced = fq.enhance(frame, vision.enhance_frames)
    assess = fq.assess(frame)
    if assess["reason"] in ("too_dark", "no_frame"):
        msg = ("It's too dark to describe. Move to better light and hold steady."
               if lang != "ne"
               else "वर्णन गर्न अति अँध्यारो छ। उज्यालोमा स्थिर राखेर फेरि प्रयास गर्नुहोस्।")
        log.info("Describe skipped: frame %s (brightness=%s)", assess["reason"], assess["brightness"])
        return msg

    # Offline fallback uses the LATEST Navigate detections (if recent) -- we don't
    # re-run the shared tracking model here, which would corrupt Navigate's tracking.
    fresh = (time.time() - latest_nav["t"]) < NAV_FRESH_SECS
    detections = latest_nav["detections"] if fresh else []
    # Gemini call is blocking network I/O -> off the event loop (never stalls Navigate).
    text = await run_in_threadpool(describer.describe, enhanced, lang, detections)
    log.info("Describe: %d chars (lang=%s, %d nav-dets fresh=%s) in %.2fs",
             len(text), lang, len(detections), fresh, time.time() - t0)
    return text


@app.post("/money")
async def money(files: list[UploadFile] = File(...)):
    """Classify a held banknote with TEMPORAL VOTING. The browser sends ~10 frames
    over ~1s; we enhance each, quality-gate the batch, and only announce a value if
    one denomination wins a clear majority -- so an empty wall/hand never yields a
    denomination. Returns {available, ok, class, confidence, text, text_ne, votes}.
    """
    frames = []
    for f in files:
        fr = decode_jpeg(await f.read())
        if fr is not None:
            frames.append(fr)
    if not frames:
        return {"available": True, "ok": False, "class": None, "confidence": 0.0,
                "text": "Could not read the camera image.", "text_ne": "क्यामेरा छवि पढ्न सकिएन।"}

    # Sanity pre-check on the sharpest frame: refuse junk before feeding the model.
    best = max(frames, key=lambda fr: fq.assess(fr)["blur_score"])
    a = fq.assess(best)
    if not a["ok"] and a["reason"] in ("too_dark", "too_blurry", "no_frame"):
        from vision_core import DARK_NOTE_SPEECH
        log.info("Money skipped: frame %s (brightness=%s blur=%s)",
                 a["reason"], a["brightness"], a["blur_score"])
        return {"available": True, "ok": False, "class": None, "confidence": 0.0,
                "text": DARK_NOTE_SPEECH[0], "text_ne": DARK_NOTE_SPEECH[1], "quality": a["reason"]}

    enhanced = [fq.enhance(fr, vision.enhance_frames) for fr in frames]
    t0 = time.time()
    result = await run_in_threadpool(banknote.classify_voted, enhanced)
    log.info("Money: class=%s conf=%.2f ok=%s (%d frames) in %.2fs",
             result.get("class"), result.get("confidence", 0.0), result.get("ok"),
             len(frames), time.time() - t0)
    return result


@app.get("/remote/health")
def remote_health():
    """Health check for the distributed-compute client (Pi). Reports the profile and
    which heavy models are loaded so the Pi knows it reached a real GPU server."""
    return {"ok": True, "profile": FEATURE_PROFILE, "device": DEVICE,
            "models": {"detect": vision._stem, "imgsz": vision.imgsz, "half": vision.half,
                       "banknote_detect": banknote_detect is not None}}


@app.get("/remote/ping")
def remote_ping():
    """Lightweight status for the Pi dashboard: GPU, which models are loaded, and how
    many requests are in flight right now (so judges can see what the laptop is doing)."""
    return {
        "status": "ok",
        "gpu": DEVICE == "cuda",
        "device": DEVICE,
        "queue_depth": max(0, _inflight["n"] - 1),   # minus this ping itself
        "models_loaded": {
            "detect": vision._stem,
            "depth": getattr(getattr(vision, "distance", None), "depth", None) is not None,
            "ocr": _ocr_reader is not None,
            "ocr_engine": _ocr_engine,
            "faces": getattr(face_matcher, "available", False),
            "banknote_detect": banknote_detect is not None,
        },
    }


@app.post("/remote/detect")
async def remote_detect(file: UploadFile = File(...)):
    """High-accuracy one-shot detection on the laptop GPU (heavy model). Returns the
    detection list for the Pi to use. (The Pi's own Navigate stays local & light.)"""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"detections": []}
    enhanced = fq.enhance(frame, vision.enhance_frames)
    dets = await run_in_threadpool(vision.detect, enhanced, False)
    return {"detections": [
        {"label": d["label"], "confidence": d["confidence"], "box": d["box"],
         "urgency": d.get("urgency"), "area_ratio": d.get("area_ratio"),
         "cx": d.get("cx"), "cy": d.get("cy"),
         "distance_m": d.get("distance_m")} for d in dets],
        "free_space_m": getattr(vision.distance, "free_space_m", None)}


@app.post("/sos")
async def sos(lat: str = Form(None), lon: str = Form(None)):
    """Emergency. The browser already raised a loud LOCAL alert (works offline); this
    is the ONLINE dispatch hook. By default it just logs -- wire a real channel here
    (Twilio SMS / email / webhook) with credentials. Degrades gracefully: with no
    channel configured, `dispatched` is False and the local alert is the fallback."""
    loc = (f"{lat},{lon}" if lat and lon else "unknown")
    maps = f"https://maps.google.com/?q={lat},{lon}" if lat and lon else None
    log.warning("SOS TRIGGERED -- location: %s  %s", loc, maps or "")
    # SWAP POINT: send an SMS/email/webhook to an emergency contact here.
    return {"ok": True, "location": loc, "maps": maps, "dispatched": False}


@app.post("/money/count")
async def money_count(files: list[UploadFile] = File(...)):
    """Count MULTIPLE notes in the frame burst and sum them. Path A (detection model)
    if one is loaded, else Path B (contour + classifier). Temporal-stable: finalizes
    only when the note-set is stable across the frames. Returns breakdown + total."""
    frames = []
    for f in files:
        fr = decode_jpeg(await f.read())
        if fr is not None:
            frames.append(fq.enhance(fr, vision.enhance_frames))
    if not frames:
        return {"stable": True, "count": 0, "total": 0,
                "text": "Could not read the camera image.", "text_ne": "क्यामेरा छवि पढ्न सकिएन।"}
    t0 = time.time()
    result = await run_in_threadpool(money_mod.count_notes, frames, banknote, banknote_detect)
    log.info("Money count: total=%s count=%s stable=%s (%d frames) in %.2fs",
             result.get("total"), result.get("count"), result.get("stable"),
             len(frames), time.time() - t0)
    return result


@app.post("/command")
async def command(text: str = Form(...), conf: float = Form(1.0)):
    """Parse a recognized voice transcript with the SHARED parser (commands.py) so
    the browser and the Pi use identical command logic. Returns
    {action, target, wake, matched, conf}."""
    result = commands.parse_command(text)
    result["conf"] = conf
    log.info("Voice: %r (conf=%.2f) -> action=%s target=%s wake=%s",
             text, conf, result["action"], result["target"], result["wake"])
    return result


@app.post("/mode")
async def mode(mode: str = Form(...)):
    """Switch the Navigate sub-mode (street | public | home). Bilingual confirmation."""
    m = set_sub_mode(mode)
    if not m:
        return {"ok": False, "text": "Unknown mode.", "text_ne": "अज्ञात मोड।", "sub_mode": get_sub_mode()}
    ne = {"street": "सडक मोड", "public": "सार्वजनिक मोड", "home": "घर मोड"}[m]
    log.info("Navigate sub-mode -> %s", m)
    return {"ok": True, "text": f"{m.capitalize()} mode", "text_ne": ne, "sub_mode": m}


@app.post("/howfar")
async def howfar():
    """'How far' -- nearest object distance + free-space ahead, from the latest
    Navigate frame (no second inference). Distances are already metric on the laptop."""
    fresh = (time.time() - latest_nav["t"]) < NAV_FRESH_SECS
    dets = latest_nav["detections"] if fresh else []
    en, ne = how_far_phrase(dets, getattr(vision, "distance", None), "en")
    return {"text": en, "text_ne": ne, "urgent": False, "urgency": "near"}


@app.post("/scan")
async def scan():
    """'Scan area' -- list what's around with rough positions + distances (PUBLIC mode)."""
    fresh = (time.time() - latest_nav["t"]) < NAV_FRESH_SECS
    dets = latest_nav["detections"] if fresh else []
    en, ne = scan_area_phrase(dets, "en")
    return {"text": en, "text_ne": ne, "urgent": False, "urgency": "near"}


@app.post("/find/register")
async def find_register(name: str = Form(...), file: UploadFile = File(...)):
    """'Remember this as my {name}' -- store a feature embedding of the object in the
    centre of the frame (laptop only; uses CLIP if available, else a colour histogram)."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"ok": False, "text": "No image received."}
    h, w = frame.shape[:2]
    crop = frame[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]   # centre region
    ok, msg = await run_in_threadpool(personal_db.register, name, crop)
    return {"ok": ok, "text": msg}


@app.post("/remote/find")
async def remote_find(name: str = Form(...), file: UploadFile = File(...)):
    """Locate a PERSONAL object the Pi can't match locally: detect candidate boxes,
    embed each crop, match against the stored embedding, return its position+distance."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"found": False, "text": "No image received."}
    if name not in personal_db.known():
        return {"found": False, "text": f"I don't have your {name} saved yet."}
    dets = await run_in_threadpool(vision.detect, frame, False)
    crops, boxes = [], []
    for d in dets:
        x1, y1, x2, y2 = (int(max(0, v)) for v in d["box"])
        c = frame[y1:min(frame.shape[0], y2), x1:min(frame.shape[1], x2)]
        if c.size:
            crops.append(c)
            boxes.append(d)
    idx, score = await run_in_threadpool(personal_db.match, name, crops)
    if idx is None:
        return {"found": False, "text": f"I don't see your {name} right now."}
    d = boxes[idx]
    z = zone_for(d["cx"])
    dm = d.get("distance_m")
    sp = (", " + __import__("distance").spoken_distance(dm)) if dm is not None else ""
    return {"found": True, "text": f"Your {name} {z}{sp}.",
            "box": d["box"], "cx": d["cx"], "distance_m": dm, "score": round(score, 2)}


@app.post("/cross")
async def cross():
    """On-demand 'can I cross' status, from the Navigate loop's latest traffic-light
    reading (no second inference -> no ByteTrack corruption). Fail-safe."""
    fresh = (time.time() - latest_nav["t"]) < NAV_FRESH_SECS
    if fresh and latest_nav.get("cross"):
        return latest_nav["cross"]
    return {"text": "Start Navigate so I can watch the traffic light. For now, please be careful.",
            "text_ne": "ट्राफिक बत्ती हेर्न Navigate सुरु गर्नुहोस्। अहिलेलाई कृपया सावधान हुनुहोस्।",
            "urgent": False, "urgency": "near"}


@app.post("/faces/who")
async def faces_who(file: UploadFile = File(...)):
    """'Who's here' -- identify enrolled faces in one frame, nearest first. Uses the
    separate face model (no ByteTrack interference). Speaks position + closeness."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"text": "No image received.", "text_ne": "तस्बिर छैन।"}
    if not face_matcher.available:
        return {"text": "Face recognition is not installed on this device.",
                "text_ne": "यो यन्त्रमा अनुहार पहिचान उपलब्ध छैन।"}
    people = await run_in_threadpool(face_matcher.who_is_here, frame)
    if not people:
        return {"text": "I don't recognize anyone here.", "text_ne": "यहाँ कसैलाई चिनिनँ।"}
    en = ", ".join(f"{p['name']} {p['zone']}, {p['closeness']}" for p in people[:3])
    ne = ", ".join(f"{p['name']} {NE_ZONES.get(p['zone'], p['zone'])}" for p in people[:3])
    log.info("Faces who: %s", [p["name"] for p in people])
    return {"text": en, "text_ne": ne, "people": people}


@app.post("/faces/enroll")
async def faces_enroll(name: str = Form(...), file: UploadFile = File(...)):
    """'Remember this person as {name}' -- enroll the largest face from one frame."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"ok": False, "text": "No image received.", "text_ne": "तस्बिर छैन।"}
    if not face_matcher.available:
        return {"ok": False, "text": "Face recognition is not installed on this device.",
                "text_ne": "यो यन्त्रमा अनुहार पहिचान उपलब्ध छैन।"}
    ok, msg = await run_in_threadpool(face_matcher.enroll, name, frame)
    return {"ok": ok, "text": msg if ok else msg,
            "text_ne": (f"{name} सम्झेँ।" if ok else "अनुहार भेटिएन, फेरि प्रयास गर्नुहोस्।")}


@app.post("/money/tally")
async def money_tally(op: str = Form(...), amount: str = Form(None), file: UploadFile = File(None)):
    """Stateful money tally. op = add | total | clear | undo | pay.
    'add' classifies the frame (needs conf > 0.7) and adds its value; 'pay' uses
    `amount` to compute change. Amounts are returned in English + Nepali."""
    if op == "add":
        if file is None:
            return {"ok": False, "text": "No image for add.", "text_ne": "तस्बिर छैन।", "total": tally.total()}
        frame = decode_jpeg(await file.read())
        if frame is None:
            return {"ok": False, "text": "Could not read the image.", "text_ne": "तस्बिर पढ्न सकिएन।", "total": tally.total()}
        frame = fq.enhance(frame, vision.enhance_frames)
        r = await run_in_threadpool(banknote.classify, frame)
        if not r.get("ok"):    # confidence <= 0.7 -> don't add, guide the user
            return {"ok": False, "text": r["text"], "text_ne": r["text_ne"], "total": tally.total()}
        res = tally.add(money_mod.class_to_value(r.get("class")), time.time())
        res["class"] = r.get("class")
        log.info("Money tally add %s -> total %d", r.get("class"), res.get("total"))
        return res
    if op == "total":
        return tally.total_spoken()
    if op == "clear":
        return tally.clear()
    if op == "undo":
        return tally.undo()
    if op == "pay":
        try:
            return tally.change_for(int(amount or 0))
        except (TypeError, ValueError):
            return {"ok": False, "text": "Tell me an amount to pay.", "text_ne": "तिर्ने रकम भन्नुहोस्।"}
    return {"ok": False, "text": f"Unknown money operation {op!r}.", "text_ne": "अज्ञात आदेश।"}


CERT_FILE = HERE / "cert.pem"
KEY_FILE = HERE / "key.pem"


def _cert_covers_ips(want_ips):
    """True if the existing cert already lists every IP in want_ips in its SAN, so
    we don't needlessly regenerate it. Returns False if the cert is missing/unreadable."""
    if not (CERT_FILE.exists() and KEY_FILE.exists()):
        return False
    try:
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        return all(ip in have for ip in want_ips)
    except Exception:
        return False


def ensure_self_signed_cert(extra_ips=None):
    """
    Create a self-signed cert covering localhost (+ any extra LAN IPs) if one doesn't
    exist yet, or regenerate it if a needed LAN IP isn't already in the cert's SAN.

    Why HTTPS at all? getUserMedia (camera/mic) only runs in a *secure context*:
    https://, or http://localhost. A browser on another device (phone, the Pi)
    hitting http://<laptop-lan-ip>:8000 is NOT secure, so the camera is blocked.
    Serving HTTPS -- with the LAN IP in the cert's SAN -- makes that a secure context.
    The cert is self-signed, so you'll click through a one-time "not private" warning.
    """
    want_ips = ["127.0.0.1"] + list(extra_ips or [])
    if _cert_covers_ips(want_ips):
        return

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    log.info("Generating self-signed certificate for %s (cert.pem / key.pem)...",
             ", ".join(["localhost"] + want_ips))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    sans = [x509.DNSName("localhost")]
    for ip in want_ips:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            log.warning("Skipping invalid IP for cert SAN: %r", ip)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def run_selftest():
    """Print a PASS/FAIL report of the moving parts, then exit. The camera is
    browser-owned here, so it's reported N/A (use `python pi_app.py --selftest` for
    a camera check)."""
    import os as _os

    import numpy as _np

    print("\n==================  SoundSight server self-test  ==================")
    results = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:22} {detail}")

    dummy = (_np.random.rand(480, 640, 3) * 255).astype("uint8")
    check("camera", lambda: (True, "N/A - browser provides frames (getUserMedia)"))
    check("frame_quality", lambda: (fq.assess(dummy)["ok"] is not None, "assess()/enhance() run"))
    check("YOLO inference", lambda: (len(vision.detect(dummy)) >= 0,
                                     f"mode={vision.mode} imgsz={vision.imgsz} device ok"))
    check("banknote model", lambda: (True,
          "loaded" if banknote.model is not None else "not trained yet (run train_banknote.py)"))
    check("offline TTS (espeak-ng)", lambda: (len(tts_engine.synth("परीक्षण", "ne")) > 0, "Nepali WAV synthesized"))
    check("EasyOCR", lambda: (get_ocr_reader() is not None, "Nepali+English reader ready"))
    check("Faces (InsightFace)", lambda: (True,
          "buffalo_l ready" if face_matcher.available else "DISABLED - pip install insightface onnxruntime"))
    has_key = bool(_os.environ.get("GEMINI_API_KEY") or _os.environ.get("GOOGLE_API_KEY"))
    check("Gemini key (Describe)", lambda: (True,
          "present - online Describe enabled" if has_key else "MISSING - Describe uses offline fallback"))
    print("==================================================================")
    print(f"  {sum(results)}/{len(results)} checks passed "
          f"({'all good' if all(results) else 'see FAIL lines above'}).\n")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SoundSight server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve plain HTTP instead of HTTPS (only if your browser doesn't force HTTPS)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run a PASS/FAIL check of model/OCR/TTS/Gemini and exit",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="COMPUTE SERVER mode: bind 0.0.0.0 (plain HTTP) so the Pi can offload to this "
             "laptop over the LAN -- fast, no per-frame TLS. Prints this machine's LAN IP.",
    )
    parser.add_argument(
        "--lan-web",
        action="store_true",
        help="LAN WEB mode: serve the browser app over HTTPS on 0.0.0.0 so a browser on "
             "another device (phone/Pi) can use its camera (getUserMedia needs HTTPS).",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()

    # LAN-WEB mode: serve the browser app over HTTPS on all interfaces so a browser
    # on ANOTHER device (phone, the Pi) gets a *secure context* and getUserMedia
    # (camera) works. Plain http:// from a non-localhost address can't open the camera.
    if args.lan_web:
        from remote import local_ip
        ip = local_ip() or "<this-laptop-ip>"
        ensure_self_signed_cert(extra_ips=[ip] if ip != "<this-laptop-ip>" else None)
        log.info("LAN WEB (profile=%s) on https://0.0.0.0:8000", FEATURE_PROFILE)
        log.info("Open the web app from any device:  https://%s:8000", ip)
        log.info("  (first visit: accept the self-signed cert warning -- it's your own laptop)")
        log.info("The Pi can also offload here:  export COMPUTE_SERVER_URL=https://%s:8000", ip)
        uvicorn.run(app, host="0.0.0.0", port=8000,
                    ssl_keyfile=str(KEY_FILE), ssl_certfile=str(CERT_FILE))
    # COMPUTE-SERVER mode: plain HTTP on all interfaces. Fast (no per-frame TLS
    # handshake) for the Pi's real-time /remote/detect offload loop. Not for browsers
    # on other devices -- they can't use the camera over plain http (use --lan-web).
    elif args.lan:
        from remote import local_ip
        ip = local_ip() or "<this-laptop-ip>"
        log.info("=" * 60)
        log.info("Compute server ready at http://%s:8000 -- point Pi at this address.", ip)
        log.info("  device=%s  detect=%s  (profile=%s)", DEVICE, vision._stem, FEATURE_PROFILE)
        log.info("  Pi:  python pi_app.py --find-server   (auto-discovers this server)")
        log.info("  or:  export COMPUTE_SERVER_URL=http://%s:8000", ip)
        log.info("  Browser web app on another device:  python server.py --lan-web (HTTPS)")
        log.info("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    # Bind to localhost so the browser treats it as a secure context (getUserMedia).
    elif args.http:
        log.info("Serving HTTP -> open http://localhost:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        ensure_self_signed_cert()
        log.info("Serving HTTPS -> open https://localhost:8000")
        log.info("First visit: click 'Advanced' -> 'Proceed to localhost (unsafe)'.")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            ssl_keyfile=str(KEY_FILE),
            ssl_certfile=str(CERT_FILE),
        )
