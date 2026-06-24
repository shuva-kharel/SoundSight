"""
pi_app.py  --  Raspberry Pi entrypoint for SoundSight (headless, offline).

Runs the SAME vision_core pipeline as the web server, but instead of a browser it
reads the Pi camera directly and speaks through an OFFLINE text-to-speech engine
(espeak-ng). It uses the NCNN nano model on the Pi's CPU for speed.

  Run on the Pi:   python pi_app.py
  Self-test:       python pi_app.py --selftest    (camera/model/audio PASS-FAIL report)
  Stop:            Ctrl+C
  Audio:           sudo apt install espeak-ng   (falls back to printing if absent)

This is the Pi twin of server.py's /ws/navigate loop: same per-session objects
(AnnouncementManager + ApproachDetector + QualityGate) and the same detect() /
select_announcements() calls, so the event-driven, anti-repeat behavior is
identical on-device. It adds the on-device robustness the browser doesn't need:
  * frame_quality.enhance() + a bad-frame gate (skip junk, say "view is dark")
  * camera health: auto-reopen a frozen / yanked USB cam, spoken once
  * perf guard: if FPS drops below FPS_FLOOR (or the SoC is hot), drop imgsz and
    turn enhance off until it recovers -- so the Pi stays responsive
  * every step wrapped so one bad frame can never kill the loop
"""

import argparse
import faulthandler
import importlib
import logging
import os
import shutil
import subprocess
import sys
import time

faulthandler.enable()

# Native math/video libraries can oversubscribe the Pi and occasionally crash
# inside C/C++ extension code. Keep them conservative before cv2/torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _import_module(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        log.exception("Failed to import %s: %s", name, exc)
        raise


def _import_camera():
    return _import_module("camera")


def _import_frame_quality():
    return _import_module("frame_quality")


def _import_remote():
    return _import_module("remote")


def _import_vision_core():
    return _import_module("vision_core")


def _import_commands():
    return _import_module("commands")


def find_espeak_local():
    """Find espeak-ng without importing vision_core/torch."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        return exe
    for path in ("/usr/bin/espeak-ng", "/usr/local/bin/espeak-ng"):
        if os.path.exists(path):
            return path
    return None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("soundsight.pi")

# --- tunables (all here so they're easy to find) ---------------------------- #
FRAME_INTERVAL = 0.15     # ~6-7 fps target; tracking persists across frames
CAM_FAIL_LIMIT = 45       # consecutive failed reads before we reopen the camera (then back off)
# Pi OFFLINE voice (Vosk) is OFF by default -- it needs `vosk` + `sounddevice` + a
# downloaded model on the Pi (see README_PI.md). Cleanly disabled (not flaky) until
# then. Set True (after installing those) to enable hands-free voice on the Pi.
VOICE_ENABLED = False
# Pre-load EasyOCR in a background thread at startup so the FIRST Read is fast (no
# 30 s model download surprise mid-demo). Disable on a low-RAM Pi: PI_PREWARM_OCR=0.
PREWARM_OCR = os.environ.get("PI_PREWARM_OCR", "1") not in ("0", "false", "False")
PERF_WINDOW = 30          # measure FPS over this many frames for the perf guard
IMG_SIZE_FULL = 320       # Pi "fast" tier imgsz...
IMG_SIZE_LOW = 256        # ...dropped to this under load
TEMP_HIGH_C = 80.0        # SoC temp above this also triggers the low-power mode
TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"


class Speaker:
    """Offline espeak-ng speaker with adjustable speed/volume and a stop() for
    barge-in. Detached so synthesis never blocks the capture loop; falls back to
    printing if espeak-ng isn't installed."""

    def __init__(self):
        self.exe = self._find_espeak()
        self.wpm = 165       # speed (words/min)
        self.amp = 130       # volume (espeak -a, 0..200)
        self.proc = None
        if self.exe:
            log.info("Offline TTS: %s", self.exe)
        else:
            log.warning("espeak-ng not found -- printing announcements "
                        "(install with: sudo apt install espeak-ng)")

    def _find_espeak(self):
        return find_espeak_local()

    def speak(self, text, voice="en"):
        if not text:
            return
        if not self.exe:
            print("SAY:", text)
            return
        self.proc = subprocess.Popen(
            [self.exe, "-s", str(self.wpm), "-a", str(self.amp), "-v", voice, text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def adjust(self, action):
        if action == "louder":   self.amp = min(200, self.amp + 40)
        elif action == "softer": self.amp = max(20, self.amp - 40)
        elif action == "faster": self.wpm = min(280, self.wpm + 30)
        elif action == "slower": self.wpm = max(90, self.wpm - 30)


class Beacon:
    """Find-mode audio beacon: short beeps whose interval the finder shrinks as the
    target gets centered AND closer. Generates a tiny sine WAV once (stdlib, no
    numpy) and plays it via `aplay` (alsa-utils). A daemon thread paces the beeps so
    it never blocks the capture loop. Silently no-ops if aplay is missing."""

    def __init__(self):
        self.exe = shutil.which("aplay")
        self._interval = None       # ms between beeps; None = silent
        self._stop = False
        self._wav = self._make_wav(880, 0.06) if self.exe else None
        self._hi = self._make_wav(1320, 0.05) if self.exe else None
        if self.exe and self._wav:
            import threading
            threading.Thread(target=self._loop, daemon=True).start()
        elif not self.exe:
            log.info("Beacon OFF (aplay not found; install alsa-utils for the find beep).")

    def _make_wav(self, freq, secs):
        import math
        import struct
        import tempfile
        import wave
        rate = 16000
        path = os.path.join(tempfile.gettempdir(), f"ss_beacon_{freq}.wav")
        try:
            with wave.open(path, "w") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                frames = bytearray()
                for i in range(int(rate * secs)):
                    val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / rate))
                    frames += struct.pack("<h", val)
                w.writeframes(bytes(frames))
            return path
        except Exception:
            return None

    def _play(self, path):
        try:
            subprocess.Popen([self.exe, "-q", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _loop(self):
        while not self._stop:
            iv = self._interval
            if iv is None:
                time.sleep(0.05)
                continue
            self._play(self._wav)
            time.sleep(max(0.08, iv / 1000.0))

    def update(self, beacon):
        """beacon = {active, interval_ms} from ObjectFinder.guide()."""
        self._interval = beacon.get("interval_ms") if beacon and beacon.get("active") else None

    def off(self):
        self._interval = None

    def success(self):
        """Two quick high beeps = found it."""
        if self.exe and self._hi:
            self._play(self._hi)
            import threading
            threading.Timer(0.12, lambda: self._play(self._hi)).start()


# --------------------------------------------------------------------------- #
# Offline voice control (Vosk) -- runs in a background thread
# --------------------------------------------------------------------------- #
def find_vosk_model():
    """Locate a Vosk model dir: $VOSK_MODEL_DIR, else ./models_vosk/* or ./vosk-model*."""
    import os
    from pathlib import Path

    env = os.environ.get("VOSK_MODEL_DIR")
    if env and Path(env).is_dir():
        return env
    for base in (Path("models_vosk"), Path(".")):
        if base.is_dir():
            for d in sorted(base.glob("vosk-model*")):
                if d.is_dir():
                    return str(d)
    return None


def start_voice(on_command, on_partial=None):
    """Start Vosk offline recognition in a daemon thread. Calls on_command(text) on
    each final transcript, and on_partial() as soon as speech is detected (for
    BARGE-IN -- stop the assistant when the user starts talking). Returns the
    thread, or None if voice is unavailable (graceful). FULLY OFFLINE -- Vosk needs
    no internet once the model is downloaded."""
    try:
        import json
        import queue
        import threading

        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except Exception as exc:
        log.warning("Voice control OFF: %s. (pip install vosk sounddevice; see README_PI.md)", exc)
        return None

    model_dir = find_vosk_model()
    if not model_dir:
        log.warning("Voice control OFF: no Vosk model found. Download one and set "
                    "VOSK_MODEL_DIR (see README_PI.md).")
        return None

    def worker():
        try:
            rec = KaldiRecognizer(Model(model_dir), 16000)
            q = queue.Queue()
            sd.default.samplerate = 16000
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                                   channels=1, callback=lambda d, *_: q.put(bytes(d))):
                log.info("Voice control ON (Vosk offline, model=%s). Say 'hey sight ...'.", model_dir)
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text:
                            try:
                                on_command(text)
                            except Exception as exc:   # a bad command must never kill voice
                                log.warning("voice dispatch error: %s", exc)
                    elif on_partial:                   # speech detected mid-utterance
                        if json.loads(rec.PartialResult()).get("partial", "").strip():
                            try:
                                on_partial()
                            except Exception:
                                pass
        except Exception as exc:
            log.warning("Voice thread stopped: %s", exc)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th


def soc_temp_c():
    """Pi SoC temperature in C, or None if unavailable (e.g. on a laptop)."""
    try:
        with open(TEMP_PATH) as fh:
            return int(fh.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def env_camera_index():
    value = os.environ.get("CAMERA_INDEX")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        log.warning("Ignoring invalid CAMERA_INDEX=%r; expected an integer.", value)
        return None


def camera_indices(camera_index):
    return (camera_index,) if camera_index is not None else (0, 1, 2)


def _build_dispatch(speaker, banknote, crossing_mon, mtally, shared, fq, remote, vision=None, vc=None):
    """Return on_command(text): parse a voice transcript and run the matching Pi
    handler. On-demand heavy features (Read/Money/Faces/Describe) OFFLOAD to the
    laptop compute server when it's reachable, else fall back to the Pi's local
    models. Navigate/crossing/distance stay 100% local. Same parser (commands.py)
    as the laptop -- no duplicated command logic."""
    from commands import parse_command
    from vision_core import describe_from_detections, how_far_phrase, scan_area_phrase, set_sub_mode
    finder_obj = shared.get("finder")

    def _ocr_pass(reader, img):
        """One pass: drop fragments < 0.30 conf, order top->bottom, left->right."""
        res = reader.readtext(img, detail=1)
        good = [(b, t, c) for (b, t, c) in res if c >= 0.30 and t.strip()]
        if not good:
            return "", 0.0
        good.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        return " ".join(t.strip() for _, t, _ in good), sum(c for _, _, c in good) / len(good)

    def ocr_read(frame):
        if frame is None:
            return "No camera image."
        # quality gate: refuse junk frames, guide the user (don't read garbage)
        a = fq.assess(frame)
        if not a.get("ok") and a.get("reason") in ("too_dark", "too_blurry", "no_frame"):
            return "Hold steady in better light and try again."
        reader = shared.get("ocr_reader")
        if reader is None:                       # warm-up not finished -> load now
            import easyocr
            speaker.speak("Loading reader")
            reader = easyocr.Reader(["ne", "en"], gpu=False)   # Pi = CPU
            shared["ocr_reader"], shared["ocr_ready"] = reader, True
        base = fq.read_preprocess(frame)
        text, conf = _ocr_pass(reader, base)
        if not text or conf < 0.50:              # rotate-retry: signs/notes held sideways
            for k in (1, 2, 3):
                t, c = _ocr_pass(reader, fq.rotate90(base, k))
                if t and (c > conf or not text):
                    text, conf = t, c
        return text or "No readable text found."

    def sample_frames(n=10, gap=0.1):
        """Grab ~n recent frames over ~1s for banknote temporal voting."""
        out = []
        for _ in range(n):
            f = shared["frame"]
            if f is not None:
                out.append(f)
            time.sleep(gap)
        return out

    def on_command(text):
        cmd = parse_command(text)
        if not cmd["wake"]:          # wake word required (ignore background speech)
            return
        speaker.stop()               # barge-in: stop whatever is playing before replying
        a, tgt = cmd["action"], cmd["target"]
        frame, dets = shared["frame"], shared["dets"]
        say = lambda m: (shared.__setitem__("last", m), speaker.speak(m))
        if a == "none":
            speaker.speak("Yes?"); return
        if a == "stop":
            shared["find"] = None
            if finder_obj is not None and finder_obj.active:
                finder_obj.exit(); shared["beacon"].off(); shared["pick"] = []
                say("Find mode off, back to navigate")
            return
        if a in ("louder", "softer", "faster", "slower"):
            speaker.adjust(a); return
        if a == "navigate":
            say("Navigating"); return
        if a == "describe":
            # OFFLOAD: laptop VLM/Gemini; fall back to a local detection summary.
            r = remote.describe(frame, "en") if frame is not None else None
            if r is None:
                remote.note_fallback_once(speaker.speak)
                r = describe_from_detections(dets, "en")
            say(r); return
        if a == "cross":     # ALWAYS local (safety, uses the local Navigate detections)
            say(crossing_mon.query(dets, frame)["text"] if frame is not None else "No view yet."); return
        if a == "how_far":   # nearest object distance + clear path ahead (local)
            est = getattr(vision, "distance", None) if vision is not None else None
            say(how_far_phrase(dets, est, "en")[0]); return
        if a == "scan":      # list what's around with positions + distances (local)
            say(scan_area_phrase(dets, "en")[0]); return
        if a in ("mode_street", "mode_public", "mode_home"):
            name = a.split("_", 1)[1]
            if set_sub_mode(name):
                shared["sub_mode"] = name
                say(f"{name.capitalize()} mode")
            else:
                say("Unknown mode")
            return
        if a == "read" or a == "label":
            rr = remote.ocr(frame, label=(a == "label")) if frame is not None else None
            if rr is not None:
                say(rr.get("text", "No readable text found.")); return
            remote.note_fallback_once(speaker.speak)
            say(ocr_read(frame)); return
        if a == "money":
            say("Hold the note steady")
            frames = sample_frames()
            rr = remote.money(frames)                       # OFFLOAD heavy detector
            if rr is not None:
                say(rr.get("text", "")); return
            remote.note_fallback_once(speaker.speak)
            say(banknote.classify_voted(frames)["text"]); return   # local temporal voting
        if a == "money_add":
            say("Hold the note steady")
            r = banknote.classify_voted(sample_frames())    # tally is local state
            if not r.get("ok"): say(r["text"]); return
            from money import class_to_value
            say(mtally.add(class_to_value(r["class"]), time.time())["text"]); return
        if a == "money_count":
            say("Counting the notes")
            frames = sample_frames(8)
            rr = remote.money_count(frames)                 # OFFLOAD multi-note
            if rr is not None:
                say(rr.get("text", "")); return
            remote.note_fallback_once(speaker.speak)
            from money import count_notes
            say(count_notes(frames, banknote)["text"]); return   # local Path B
        if a == "money_total": say(mtally.total_spoken()["text"]); return
        if a == "money_clear": say(mtally.clear()["text"]); return
        if a == "money_undo":  say(mtally.undo()["text"]); return
        if a == "money_pay":
            try: say(mtally.change_for(int(tgt or 0))["text"])
            except (TypeError, ValueError): say("Tell me an amount.")
            return
        # --- Object Finder mode ------------------------------------------------
        if a == "find_mode":
            if finder_obj is None: say("Find mode is unavailable."); return
            say(finder_obj.start_scan(time.time())); return
        if a == "find":      # "find a cup" -> immediate lock; else fall into a scan
            if finder_obj is None: say("Find mode is unavailable."); return
            conf = finder_obj.find_class(tgt or "", dets, time.time())
            if conf: say(conf)
            else: say(f"I don't see a {tgt} right now. " + finder_obj.start_scan(time.time()))
            return
        if a == "track":
            if finder_obj is None: return
            conf = finder_obj.choose(tgt or "")
            say(conf or "I didn't catch which one. Say the letter or the name."); return
        if a == "room_scan":
            if finder_obj is None: say("Unavailable."); return
            say(finder_obj.room_scan(dets)); return
        if a == "found":
            if finder_obj is not None and finder_obj.active:
                finder_obj.exit(); shared["beacon"].off(); shared["pick"] = []
            say("Great, back to navigate"); return
        if a == "repeat":
            speaker.speak(shared["last"] or "Nothing to repeat yet."); return
        if a == "sos":
            speaker.amp = 200   # max volume for the emergency alert
            say("Emergency. This person needs help. Please assist."); return
        if a == "help":
            say("You can say navigate, what's in front, read, how much, add this note, total, "
                "can I cross, find, repeat, louder, slower, stop, or for help say help me."); return
        if a == "who":
            rr = remote.faces(frame) if frame is not None else None   # OFFLOAD to laptop InsightFace
            if rr is not None:
                say(rr.get("text", "I don't recognize anyone here.")); return
            say("Face recognition needs the laptop, which is offline."); return
        if a == "remember":
            say("Enrolling a face needs the laptop. Use it nearby and online."); return

    return on_command


def _start_dashboard(web, port):
    """Start the live dashboard. Prefer the full FastAPI/WebSocket pi_server; if
    fastapi/uvicorn aren't installed, fall back to the stdlib pi_web preview. Returns
    an object with a uniform .update(frame, detections, announce, **fields) method, or
    None if web is disabled / unavailable."""
    if not web:
        return None
    try:
        import pi_server
        return pi_server.start(pi_server.DashboardState(), port=port)
    except Exception as exc:
        log.warning("Full dashboard (pi_server) unavailable (%s) -- using stdlib preview. "
                    "Install fastapi+uvicorn for the WebSocket dashboard.", exc)
        try:
            from pi_web import WebPreview
            return WebPreview(port=port).start()
        except Exception as exc2:
            log.warning("Web preview also failed (%s) -- running headless.", exc2)
            return None


def run(server_url=None, camera_index=None, show=False, web=False, web_port=8080, no_enhance=False):
    # Import the Pi-only dependencies lazily so startup errors are reported cleanly.
    cam = _import_camera()
    fq = _import_frame_quality()
    remote_mod = _import_remote()
    vc = _import_vision_core()
    from crossing import CrossingMonitor
    from money import MoneyTally
    from vision_core import BanknoteClassifier, FPS_FLOOR
    from vision_core import AnnouncementManager, ApproachDetector, QualityGate, VisionCore, select_announcements

    vision = VisionCore(mode="coco-ncnn", accuracy="fast")
    if no_enhance:
        vision.enhance_frames = False   # skip the OpenCV CLAHE path (cv2 segfault dodge)
        log.info("Frame enhance DISABLED (--no-enhance): skipping OpenCV CLAHE.")
    manager = AnnouncementManager()
    approach = ApproachDetector()
    gate = QualityGate()
    speaker = Speaker()
    speak = speaker.speak

    # Distributed compute: offload heavy on-demand AI to the laptop if reachable.
    remote = remote_mod.RemoteCompute(server_url)
    if remote.enabled:
        log.info("Compute offload target: %s (checking...)", remote.url)
        remote.health(force=True)
    else:
        log.info("No COMPUTE_SERVER_URL -- running fully on-device (offload disabled).")

    # voice command components (shared state updated each frame by the loop)
    import finder as finder_mod
    banknote = BanknoteClassifier()
    crossing_mon = CrossingMonitor()
    mtally = MoneyTally()
    finder_obj = finder_mod.ObjectFinder(profile="pi")
    beacon = Beacon()
    shared = {"frame": None, "dets": [], "find": None, "last": "",
              "sub_mode": vc.get_sub_mode(), "light": None,
              "finder": finder_obj, "beacon": beacon, "pick": [],
              "ocr_reader": None, "ocr_ready": False}

    # Pre-warm EasyOCR on a background thread so the first Read is fast (no 30 s
    # download mid-demo). Offload still prefers the laptop; this is the local fallback.
    if PREWARM_OCR:
        def _warm_ocr():
            try:
                import easyocr
                log.info("Pre-warming EasyOCR on the Pi (one-time, ~20 s, background)...")
                shared["ocr_reader"] = easyocr.Reader(["ne", "en"], gpu=False)
                shared["ocr_ready"] = True
                log.info("Pi OCR pre-warmed and READY.")
            except Exception as exc:
                log.warning("Pi OCR pre-warm failed (%s) -- will load on first Read.", exc)
        import threading
        threading.Thread(target=_warm_ocr, daemon=True).start()

    if VOICE_ENABLED:
        start_voice(_build_dispatch(speaker, banknote, crossing_mon, mtally, shared, fq, remote, vision, vc),
                    on_partial=speaker.stop)   # on_partial -> barge-in
    else:
        log.info("Pi voice control DISABLED (VOICE_ENABLED=False). "
                 "Install vosk+sounddevice+a model and set VOICE_ENABLED=True to enable.")

    indices = camera_indices(camera_index)
    cap, cam_index, backend = cam.find_camera(indices)   # OS-aware, validated open
    if cap is None:
        log.error("No camera found. Try `python pi_app.py --list-cameras` to locate it.")
        return
    log.info("SoundSight Pi running on camera %d (%s). Ctrl+C to stop.", cam_index, backend)

    preview = _start_dashboard(web, web_port)

    fails = 0                 # consecutive failed reads
    low_power = False         # perf guard active?
    win_t0, win_frames = time.time(), 0
    cur_fps = 0.0
    prev_frame = None

    try:
        while True:
            t0 = time.time()
            try:
                ok, frame = cap.read()

                # --- camera health: reopen a dead / yanked camera ------------- #
                if not ok or frame is None:
                    fails += 1
                    if fails == CAM_FAIL_LIMIT:
                        speak("Camera disconnected")
                        log.warning("Camera read failed %dx -- reopening (another app using it?)...", fails)
                    if fails >= CAM_FAIL_LIMIT:
                        cap.release()
                        time.sleep(1.0)   # back off so we don't spin on a dead device
                        newcap, cam_index, backend = cam.find_camera(indices)
                        if newcap is not None:
                            cap = newcap
                            fails = 0
                            speak("Camera reconnected")
                            log.info("Camera reopened on index %d (%s).", cam_index, backend)
                    else:
                        time.sleep(0.05)
                    continue
                if fails:
                    fails = 0
                # frozen camera (identical consecutive frames) -> treat as a bad view
                frozen = fq.frame_diff(prev_frame, frame) < 0.002 if prev_frame is not None else False
                prev_frame = frame

                # --- quality gate: skip only genuinely unusable frames -------- #
                now = time.time()
                assess, warn = gate.check(frame, now)
                if frozen or assess.get("blocking"):
                    if warn:
                        speak(warn["text"])
                    time.sleep(0.05)
                    continue

                # --- detect + announce (same pipeline as the server) ---------- #
                detections = vision.detect(frame, assessment=assess)  # enhances internally
                shared["frame"], shared["dets"] = frame, detections    # for voice commands
                if show:   # terminal visibility: what the Pi is seeing right now
                    log.info("SEE: %s", ", ".join(
                        f"{d['label']}({_closeness(d)})" for d in detections) or "(nothing)")
                approaching = approach.approaching_objects(detections)
                announced = None
                finding = finder_obj.active   # Find mode suppresses AMBIENT navigate
                for item in select_announcements(detections, manager, approaching, now):
                    # In Find mode only HAZARDS (very close) still interrupt.
                    if finding and item.get("urgency") != "very close":
                        continue
                    speak(item["text"])
                    announced = item["text"]

                # --- STREET sub-mode: auto traffic-light state (confirmed changes) -
                if vc.get_sub_mode() == "street":
                    light = crossing_mon.update(detections, frame, now)
                    if light:
                        speaker.stop()            # safety message takes priority
                        speak(light["text"])
                        announced = light["text"]
                    shared["light"] = crossing_mon.confirmed

                # --- Object Finder: scan -> pick-list -> guide + beacon -------- #
                if finder_obj.state == "scanning":
                    res = finder_obj.update_scan(detections, now)
                    if res:                       # scan window ended -> speak the list
                        speak(res[1]); shared["pick"] = res[2]
                        shared["last"] = res[1]
                elif finder_obj.state == "tracking":
                    g = finder_obj.guide(detections, now)
                    beacon.update(g["beacon"])
                    if g["text"] and not announced:   # hazard this frame wins the voice
                        speak(g["text"]); shared["last"] = g["text"]
                    if g["done"]:
                        beacon.off(); beacon.success(); shared["pick"] = []
                else:
                    beacon.off()

                if preview is not None:   # live dashboard of the Pi's own camera
                    preview.update(frame, detections, announced,
                                   sub_mode=vc.get_sub_mode(), light=shared.get("light"),
                                   mode=("FIND" if finding else "NAVIGATE"),
                                   pick=shared.get("pick"),
                                   target=(finder_obj.target or {}).get("label") if finding else None,
                                   traffic_light=shared.get("light"),
                                   find_target=(finder_obj.target or {}).get("label") if finding else None,
                                   fps=cur_fps,
                                   laptop_link=remote.link_status,
                                   laptop_ms=remote.last_latency_ms,
                                   ocr_ready=shared.get("ocr_ready", False))

            except Exception as exc:   # one bad frame must never kill the loop
                log.exception("Loop error (continuing): %s", exc)
                time.sleep(0.05)
                continue

            # --- perf guard: keep the Pi above the FPS floor ------------------ #
            win_frames += 1
            if win_frames >= PERF_WINDOW:
                fps = win_frames / (time.time() - win_t0)
                cur_fps = fps
                temp = soc_temp_c()
                hot = temp is not None and temp >= TEMP_HIGH_C
                if (fps < FPS_FLOOR or hot) and not low_power:
                    low_power = True
                    vision.set_imgsz(IMG_SIZE_LOW)
                    vision.enhance_frames = False
                    speak("Low power mode")
                    log.warning("Perf guard ON: fps=%.1f temp=%s -> imgsz=%d, enhance off",
                                fps, temp, IMG_SIZE_LOW)
                elif low_power and fps > FPS_FLOOR + 1.5 and not hot:
                    low_power = False
                    vision.set_imgsz(IMG_SIZE_FULL)
                    vision.enhance_frames = not no_enhance   # keep it off if --no-enhance
                    log.info("Perf guard OFF: fps=%.1f -> imgsz=%d, enhance %s",
                             fps, IMG_SIZE_FULL, "off" if no_enhance else "on")
                else:
                    log.info("Pi FPS: %.1f%s", fps, f" temp={temp:.0f}C" if temp else "")
                win_t0, win_frames = time.time(), 0

            dt = time.time() - t0
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        cap.release()


def _zone_from_det(det, frame_w):
    cx = det.get("cx")
    if cx is None:
        box = det.get("box") or [0, 0, 0, 0]
        cx = (float(box[0]) + float(box[2])) / 2.0
    if cx < frame_w / 3:
        return "on your left"
    if cx > 2 * frame_w / 3:
        return "on your right"
    return "ahead"


def _closeness(det):
    urgency = det.get("urgency")
    if urgency:
        return urgency
    area = float(det.get("area_ratio") or 0.0)
    if area > 0.20:
        return "very close"
    if area >= 0.05:
        return "near"
    return "far"


class RemoteAnnouncementManager:
    """Tiny Torch-free announcer for laptop-offloaded detections."""

    def __init__(self, min_gap=1.2, refresh=6.0):
        self.min_gap = min_gap
        self.refresh = refresh
        self.last_spoken = 0.0
        self.memory = {}

    def choose(self, detections, frame_w, now):
        if not detections or (now - self.last_spoken) < self.min_gap:
            return None
        ranked = sorted(
            detections,
            key=lambda d: (
                _closeness(d) == "very close",
                float(d.get("area_ratio") or 0.0),
                float(d.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        for det in ranked[:5]:
            label = str(det.get("label") or "object")
            zone = _zone_from_det(det, frame_w)
            close = _closeness(det)
            state = (zone, close)
            prev_state, prev_t = self.memory.get(label, (None, 0.0))
            if state != prev_state or close == "very close" or (now - prev_t) >= self.refresh:
                self.memory[label] = (state, now)
                self.last_spoken = now
                if close == "far":
                    return f"{label} {zone}"
                return f"{label} {zone}, {close}"
        return None


def run_remote_only(server_url=None, camera_index=None, show=False, web=False, web_port=8080):
    """Torch-free Pi mode: camera on the Pi, detection on the laptop server."""
    cam = _import_camera()
    fq = _import_frame_quality()
    remote_mod = _import_remote()

    speaker = Speaker()
    remote = remote_mod.RemoteCompute(server_url)
    if not remote.enabled:
        log.error("Remote-only mode needs COMPUTE_SERVER_URL or --find-server.")
        return 2
    log.info("REMOTE-ONLY mode: camera/audio on Pi, AI detection on %s", remote.url)
    if not remote.health(force=True):
        log.error("Laptop compute server is offline: %s", remote.url)
        return 2

    cap, cam_index, backend = cam.find_camera(camera_indices(camera_index))
    if cap is None:
        log.error("No camera found. Try `python pi_app.py --list-cameras`.")
        return 2
    log.info("SoundSight Pi remote-only running on camera %d (%s). Ctrl+C to stop.", cam_index, backend)

    preview = _start_dashboard(web, web_port)

    manager = RemoteAnnouncementManager()
    offline_spoken = False
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            assess = fq.assess(frame)
            if assess.get("reason") in ("too_dark", "no_frame"):
                time.sleep(0.1)
                continue

            result = remote.detect(frame)
            if result is None:
                if not offline_spoken:
                    speaker.speak("Laptop is offline")
                    offline_spoken = True
                if preview is not None:
                    preview.update(frame, [], "Laptop is offline", mode="NAVIGATE",
                                   laptop_link=remote.link_status, laptop_ms=remote.last_latency_ms)
                time.sleep(1.0)
                continue
            offline_spoken = False

            detections = result.get("detections") or []
            msg = manager.choose(detections, frame.shape[1], time.time())
            if msg:
                speaker.speak(msg)
            if show:
                log.info("SEE: %s", ", ".join(
                    f"{d.get('label')}({_closeness(d)})" for d in detections) or "(nothing)")
            if preview is not None:
                preview.update(frame, detections, msg, mode="NAVIGATE",
                               laptop_link=remote.link_status, laptop_ms=remote.last_latency_ms,
                               fps=(1.0 / max(1e-3, time.time() - t0)))

            dt = time.time() - t0
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        cap.release()
    return 0


def selftest(camera_index=None):
    """PASS/FAIL report: camera opens + returns a real frame, model loads + runs one
    inference, and audio (espeak-ng) is available. Exits 0 if all pass."""
    cam = _import_camera()
    remote_mod = _import_remote()

    print("\n==================  SoundSight Pi self-test  ==================")
    results = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:20} {detail}")

    cap, cam_index, backend = cam.find_camera(camera_indices(camera_index))

    def cam_check():
        if cap is None:
            return False, "no camera found (try --list-cameras)"
        return True, f"index {cam_index}, backend {backend}, real frame received"

    check("camera", cam_check)

    def model_check():
        code = (
            "from vision_core import VisionCore\n"
            "import numpy as np\n"
            "v = VisionCore(mode='coco-ncnn', accuracy='fast')\n"
            "dets = v.detect((np.random.rand(240, 320, 3) * 255).astype('uint8'))\n"
            "print(f'inference ran (imgsz={v.imgsz}, dets={len(dets)})')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        err = proc.stderr or ""
        detail = (proc.stdout or err or "").strip().splitlines()
        detail = detail[-1] if detail else f"exit={proc.returncode}"
        if "null bytes" in err:
            detail = "corrupted venv file (NUL bytes) -- run: python pi_app.py --repair-venv"
        elif proc.returncode in (-6, -11) or "double-linked list" in err or "Aborted" in err:
            detail = ("native crash (segfault/abort in torch/lap/opencv -- usually numpy-2 ABI) "
                      "-- run: python pi_app.py --reinstall-ml")
        return proc.returncode == 0, detail

    check("model load + infer", model_check)
    check("audio (espeak-ng)", lambda: (
        bool(find_espeak_local()),
        find_espeak_local() or "MISSING -- sudo apt install espeak-ng"))

    def remote_check():
        rc = remote_mod.RemoteCompute()
        if not rc.enabled:
            return True, "offload disabled (no COMPUTE_SERVER_URL) -- runs on-device"
        import time as _t
        t0 = _t.time()
        ok = rc.health(force=True)
        return True, (f"laptop ONLINE at {rc.url} ({(_t.time()-t0)*1000:.0f} ms round-trip)"
                      if ok else f"laptop OFFLINE ({rc.url}) -- will run on-device")

    check("compute offload", remote_check)
    temp = soc_temp_c()
    check("SoC temp", lambda: (True, f"{temp:.0f}C" if temp is not None else "N/A (not a Pi)"))

    if cap is not None:
        cap.release()
    print("==============================================================")
    print(f"  {sum(results)}/{len(results)} checks passed "
          f"({'all good' if all(results) else 'see FAIL lines above'}).\n")
    raise SystemExit(0 if all(results) else 1)


def diag_imports():
    """Run risky native imports in child processes so a segfault is isolated and
    reported instead of taking this diagnostic command down with it."""
    checks = [
        ("numpy", "import numpy; print('numpy', numpy.__version__)"),
        ("cv2", "import cv2; print('cv2', cv2.__version__)"),
        ("torch", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"),
        ("ultralytics", "from ultralytics import YOLO; print('ultralytics YOLO import ok')"),
        ("vision_core", "import vision_core; print('vision_core import ok')"),
        ("camera", "import camera; print('camera import ok')"),
    ]
    print("\n================ SoundSight Pi native import diagnostic ================")
    failed = False
    for name, code in checks:
        proc = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        ok = proc.returncode == 0
        failed = failed or not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:12} exit={proc.returncode}")
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            print("       " + out.replace("\n", "\n       "))
        if err:
            print("       " + err.replace("\n", "\n       "))
    print("=======================================================================")
    raise SystemExit(1 if failed else 0)


def torch_import_ok():
    """(ok, stderr): does `import torch` succeed in a clean child process?
    Returns the child's stderr too so the caller can detect the specific failure
    (e.g. NUL-byte corruption) and give targeted repair advice."""
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", "import torch; print(torch.__version__)"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    return proc.returncode == 0, (proc.stderr or "")


# --------------------------------------------------------------------------- #
# venv repair -- fix files corrupted by an interrupted pip install / bad SD card
# --------------------------------------------------------------------------- #
# Symptom: `SyntaxError: source code string cannot contain null bytes` on import.
# A half-written .py file (NUL bytes) anywhere in site-packages takes down torch /
# ultralytics / vision_core. We scan for those files and force-reinstall ONLY the
# affected distributions -- much faster and safer than rebuilding the whole venv.

def _site_packages_dirs():
    import sysconfig
    dirs = []
    try:
        dirs.append(sysconfig.get_paths()["purelib"])
    except Exception:
        pass
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p) and p not in dirs:
            dirs.append(p)
    return [d for d in dirs if os.path.isdir(d)]


def _file_owner_map():
    """Reverse map: absolute file path -> the pip distribution that installed it,
    built from each installed package's RECORD (its own list of files). This is
    authoritative -- it correctly attributes isympy.py -> sympy and cv2/* ->
    opencv-python-headless -- unlike guessing a package name from the filename
    (which could send pip after a random unrelated PyPI package)."""
    owners = {}
    try:
        from importlib import metadata
    except Exception:
        return owners
    for dist in metadata.distributions():
        try:
            name = (dist.metadata["Name"] or "").strip()
        except Exception:
            name = ""
        if not name:
            continue
        for f in (dist.files or []):
            try:
                p = os.path.normcase(os.path.realpath(dist.locate_file(f)))
            except Exception:
                continue
            owners[p] = name
    return owners


def scan_corrupt_files():
    """Walk site-packages for .py files containing NUL bytes (the corruption
    signature). Returns (corrupt, unowned):
      corrupt  = {distribution_name: [paths...]}   -- reinstallable via pip
      unowned  = [paths...]                         -- stray files no package owns"""
    owners = _file_owner_map()
    corrupt = {}
    unowned = []
    for sp in _site_packages_dirs():
        for root, _dirs, files in os.walk(sp):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                except OSError:
                    continue
                if b"\x00" not in data:
                    continue
                key = os.path.normcase(os.path.realpath(path))
                dist = owners.get(key)
                if dist:
                    corrupt.setdefault(dist, []).append(path)
                else:
                    unowned.append(path)
    return corrupt, unowned


def repair_venv(apply=False):
    """Find NUL-corrupted files in the venv and (if apply) force-reinstall the
    affected packages. Run with --repair-venv to actually fix; --scan-corrupt to
    only report. Exits 0 if the venv is clean (or was repaired)."""
    print("\n================ SoundSight venv corruption scan ================")
    print("Scanning site-packages for files with NUL bytes (interrupted install / bad SD card)...")
    corrupt, unowned = scan_corrupt_files()
    if not corrupt and not unowned:
        print("  [OK] No corrupted .py files found.")
        ok, err = torch_import_ok()
        if ok:
            print("  [OK] `import torch` succeeds.")
        else:
            print("  [!] No NUL-byte files, but torch still fails to import:")
            print("      " + err.strip().replace("\n", "\n      "))
            print("      This may be a torch/Python version mismatch rather than corruption.")
            print("      Try: pip install --force-reinstall --no-cache-dir -r requirements_pi.txt")
        print("=================================================================")
        raise SystemExit(0 if ok else 1)

    dists = sorted(corrupt)
    total_bad = sum(len(v) for v in corrupt.values()) + len(unowned)
    if dists:
        print(f"  [!] Corrupted packages found: {', '.join(dists)}")
        for dist, paths in sorted(corrupt.items()):
            print(f"      - {dist}: {len(paths)} file(s), e.g. {paths[0]}")
    if unowned:
        print(f"  [!] {len(unowned)} corrupted file(s) belong to no installed package (stray):")
        for p in unowned[:10]:
            print(f"      - {p}")
        if len(unowned) > 10:
            print(f"      ... and {len(unowned) - 10} more")

    # A handful of corrupt files = interrupted install (repairable). Hundreds across
    # many packages = the filesystem/SD card is rotting; reinstalling won't hold.
    widespread = total_bad >= 200 or len(dists) >= 8

    if not apply:
        print("\n  Re-run with --repair-venv to reinstall these packages, or manually:")
        if dists:
            print(f"    pip install --force-reinstall --no-cache-dir {' '.join(dists)}")
        if widespread:
            print("  NOTE: corruption is widespread -- if it returns after repair, the SD card")
            print("        is failing; reflash a fresh card and run `bash setup_pi.sh`.")
        print("=================================================================")
        raise SystemExit(1)

    # Reinstall via the SAME interpreter's pip so we hit this venv, not system pip.
    # Only reinstall real distributions -- never guess a package for a stray file.
    if dists:
        cmd = [sys.executable, "-m", "pip", "install",
               "--force-reinstall", "--no-cache-dir"] + dists
        print("\n  Repairing:", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print("  [FAIL] pip reinstall failed (exit %d)." % rc)
            print("        Check network (the Pi must reach pypi.org / piwheels.org) and disk space,")
            print("        then re-run. If only piwheels DNS fails, retrying usually works.")
            print("=================================================================")
            raise SystemExit(rc)
    else:
        print("\n  No reinstallable packages -- only stray files (see below).")

    # Verify the corruption is actually gone.
    still_bad, still_unowned = scan_corrupt_files()
    ok, err = torch_import_ok()
    if still_bad:
        print("  [!] Still corrupted after reinstall:", ", ".join(sorted(still_bad)))
        print("      The SD card is likely failing. Reflash a fresh card and run `bash setup_pi.sh`.")
        print("=================================================================")
        raise SystemExit(1)
    if still_unowned:
        print(f"  [!] {len(still_unowned)} stray corrupted file(s) remain (no package owns them).")
        print("      These aren't imported by torch, but to be safe delete them, e.g.:")
        print(f"        rm '{still_unowned[0]}'")
    print("  [OK] No package-owned corrupted files remain.")
    print("  [%s] `import torch` %s." % ("OK" if ok else "!!",
          "succeeds" if ok else "still fails:\n      " + err.strip()))
    if ok:
        print("  Now run:  python pi_app.py --selftest")
    print("=================================================================")
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------- #
# native ML-stack repair -- fix `corrupted double-linked list` / SIGABRT aborts
# --------------------------------------------------------------------------- #
# Symptom: torch IMPORTS fine, but the first inference aborts in a C/C++ op
# (torchvision::nms, lap, ...) with "corrupted double-linked list" / "Aborted".
# This is a broken or MISMATCHED native build -- usually a GPU/CUDA torch wheel on
# the Pi (it must be the CPU build) or a torchvision compiled against a different
# torch. The fix is a clean, matched CPU reinstall of torch + torchvision.

def _run_py(code, timeout=120):
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def ml_stack_info():
    """Return (text, cuda_build): torch/torchvision/numpy versions and whether torch
    is a CUDA build (it should be None / CPU on a Pi)."""
    proc = _run_py(
        "import torch, torchvision, numpy\n"
        "print('torch', torch.__version__)\n"
        "print('torchvision', torchvision.__version__)\n"
        "print('numpy', numpy.__version__)\n"
        "print('cuda_build', torch.version.cuda)\n",
        timeout=60)
    out = (proc.stdout or "").strip()
    cuda_build = None
    for line in out.splitlines():
        if line.startswith("cuda_build "):
            val = line.split(" ", 1)[1].strip()
            cuda_build = None if val in ("None", "") else val
    return (out or (proc.stderr or "").strip()), cuda_build


def nms_ok():
    """Run the exact native op that aborts (torchvision NMS) in a child process.
    Returns (ok, detail). returncode -6 == SIGABRT (the 'corrupted double-linked
    list' crash)."""
    proc = _run_py(
        "import torch, torchvision\n"
        "b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]])\n"
        "s = torch.tensor([0.9, 0.8])\n"
        "keep = torchvision.ops.nms(b, s, 0.5)\n"
        "print('nms ok', keep.tolist())\n",
        timeout=60)
    if proc.returncode == 0:
        return True, (proc.stdout or "").strip().splitlines()[-1]
    if proc.returncode == -6:
        return False, "SIGABRT (corrupted double-linked list) -- native torch/torchvision build is broken"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else f"exit={proc.returncode}")


def lap_ok():
    """Run the native `lap` linear-assignment op the ByteTrack tracker uses (the other
    op that aborts with 'corrupted double-linked list'). (ok, detail)."""
    proc = _run_py(
        "import numpy as np\n"
        "import lap\n"
        "c = np.array([[1., 2.], [3., 4.]], dtype='float64')\n"
        "cost, x, y = lap.lapjv(c, extend_cost=True)\n"
        "print('lap ok', cost)\n",
        timeout=60)
    if proc.returncode == 0:
        return True, (proc.stdout or "").strip().splitlines()[-1]
    if proc.returncode == -6:
        return False, "SIGABRT -- native `lap`/numpy ABI mismatch (rebuild lap against this numpy)"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else f"exit={proc.returncode}")


def cv2_ok():
    """Run the exact OpenCV ops frame_quality.enhance() uses (cvtColor/split/CLAHE/
    merge) in a child process. These segfault under a numpy-ABI mismatch. (ok, detail)."""
    proc = _run_py(
        "import numpy as np, cv2\n"
        "f = (np.random.rand(64, 64, 3) * 255).astype('uint8')\n"
        "lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)\n"
        "l, a, b = cv2.split(lab)\n"
        "l = cv2.createCLAHE(2.0, (8, 8)).apply(l)\n"
        "cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)\n"
        "print('cv2 ok', cv2.__version__)\n",
        timeout=60)
    if proc.returncode == 0:
        return True, (proc.stdout or "").strip().splitlines()[-1]
    if proc.returncode == -11:
        return False, "SIGSEGV in OpenCV -- numpy-ABI mismatch (cv2 built for a different numpy)"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else f"exit={proc.returncode}")


def _numpy_major():
    proc = _run_py("import numpy; print(numpy.__version__)", timeout=30)
    try:
        return int((proc.stdout or "").strip().split(".")[0])
    except (ValueError, IndexError):
        return None


def _opencv_dist():
    """Which OpenCV distribution is installed (so we reinstall the right one)."""
    from importlib import metadata
    for name in ("opencv-python-headless", "opencv-python",
                 "opencv-contrib-python-headless", "opencv-contrib-python"):
        try:
            metadata.version(name)
            return name
        except metadata.PackageNotFoundError:
            continue
    return "opencv-python-headless"


def _native_health():
    """(all_ok, lines): run every native op the Pi uses that can abort/segfault."""
    nms, nms_d = nms_ok()
    lap, lap_d = lap_ok()
    cv, cv_d = cv2_ok()
    lines = [
        f"  [{'PASS' if nms else 'FAIL'}] torchvision NMS: {nms_d}",
        f"  [{'PASS' if lap else 'FAIL'}] lap (tracker):   {lap_d}",
        f"  [{'PASS' if cv  else 'FAIL'}] opencv (enhance): {cv_d}",
    ]
    return (nms and lap and cv), lines


def reinstall_ml(apply=False):
    """Diagnose and (if apply) fix the native stack so inference stops crashing.

    Checks the three native ops that crash on this Pi -- torchvision NMS, `lap`
    (tracker) and OpenCV (enhance) -- which all link numpy. The usual root cause is
    a numpy-2 ABI mismatch (these wheels were built for numpy 1.x), so the fix pins
    numpy<2 first, then, if needed, reinstalls a matched CPU torch/torchvision/lap +
    OpenCV against that numpy."""
    print("\n============== SoundSight native ML-stack check ==============")
    info, cuda_build = ml_stack_info()
    print(info.replace("\n", "\n  ") if info else "  (could not read torch versions)")
    if cuda_build is not None:
        print(f"  [!] torch is a CUDA/GPU build (cuda={cuda_build}). The Pi has NO GPU.")
    healthy, lines = _native_health()
    print("\n".join(lines))
    npmaj = _numpy_major()
    numpy2 = npmaj is not None and npmaj >= 2
    if numpy2:
        print(f"  [i] numpy is {npmaj}.x -- compiled wheels (cv2/lap/torchvision/scipy) built for")
        print("      numpy 1.x crash under numpy 2.x. Pinning numpy<2 is the likely fix.")

    if healthy and cuda_build is None:
        print("\n  Native stack is healthy. If inference still crashes, run --selftest.")
        print("=============================================================")
        raise SystemExit(0)

    opencv = _opencv_dist()
    step1 = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "numpy<2"]
    step2 = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             "numpy<2", "torch", "torchvision", "lap", opencv]
    if not apply:
        print("\n  Re-run with --reinstall-ml to fix automatically, or do it manually:")
        if numpy2:
            print("    1) " + " ".join(step1) + "    # most likely fix on its own")
        print("    2) " + " ".join(step2))
        print("=============================================================")
        raise SystemExit(1)

    # Step 1: if numpy is 2.x, downgrade it first -- often realigns the EXISTING
    # numpy-1 wheels (cv2/lap/torchvision) with no other reinstall needed.
    if numpy2:
        print("\n  [1/2] Pinning numpy<2:\n   ", " ".join(step1))
        if subprocess.run(step1).returncode == 0:
            healthy, lines = _native_health()
            print("\n  After numpy<2:")
            print("\n".join(lines))
            if healthy:
                print("  [OK] Fixed by the numpy downgrade alone. Run:  python pi_app.py --selftest")
                print("=============================================================")
                raise SystemExit(0)
            print("  Still not healthy -- reinstalling the native wheels against numpy<2...")

    # Step 2: reinstall the matched CPU natives + OpenCV, constrained to numpy<2.
    print("\n  [2/2] Reinstalling natives against numpy<2:\n   ", " ".join(step2))
    rc = subprocess.run(step2).returncode
    if rc != 0:
        print("  [FAIL] pip reinstall failed (exit %d). Check network / disk space." % rc)
        print("=============================================================")
        raise SystemExit(rc)

    info, cuda_build = ml_stack_info()
    healthy, lines = _native_health()
    print("\n  After reinstall:")
    print(info.replace("\n", "\n  ") if info else "  (could not read torch versions)")
    print("\n".join(lines))
    if healthy and cuda_build is None:
        print("  [OK] Native stack fixed. Now run:  python pi_app.py --selftest")
        print("=============================================================")
        raise SystemExit(0)

    print("\n  [!] Still not healthy. Last resorts:")
    print("  - ARM-CPU wheels from piwheels:")
    print("      pip install --force-reinstall --no-cache-dir \\")
    print("        --index-url https://www.piwheels.org/simple \\")
    print(f"        --extra-index-url https://pypi.org/simple 'numpy<2' torch torchvision lap {opencv}")
    print("  - Run a minimal, crash-free path now:  python pi_app.py --no-track --no-enhance")
    print("  - If crashes persist across libraries, the SD card is failing -- reflash + setup_pi.sh.")
    print("  - Or do all detection on the laptop:  python pi_app.py --find-server")
    print("=============================================================")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="SoundSight Raspberry Pi entrypoint")
    ap.add_argument("--selftest", action="store_true",
                    help="check camera/model/audio/offload and exit with a PASS/FAIL report")
    ap.add_argument("--diag-imports", action="store_true",
                    help="isolate native import crashes for cv2/torch/ultralytics")
    ap.add_argument("--scan-corrupt", action="store_true",
                    help="scan the venv for NUL-corrupted files (cause of 'null bytes' errors) and report")
    ap.add_argument("--repair-venv", action="store_true",
                    help="scan AND force-reinstall any NUL-corrupted packages, then verify torch imports")
    ap.add_argument("--check-ml", action="store_true",
                    help="check the native stack (torchvision NMS, lap, opencv) for the segfault/"
                         "'corrupted double-linked list' crashes; reports numpy version")
    ap.add_argument("--reinstall-ml", action="store_true",
                    help="fix native crashes: pin numpy<2 and reinstall matched CPU torch/torchvision/"
                         "lap/opencv, then re-verify all three native ops")
    ap.add_argument("--remote-only", action="store_true",
                    help="Torch-free mode: use the laptop compute server for detection")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe camera indices 0-4 (OS-aware backend) and exit")
    ap.add_argument("--camera-index", type=int, default=env_camera_index(),
                    help="open this camera index directly, e.g. 1 from --list-cameras")
    ap.add_argument("--find-server", action="store_true",
                    help="scan the LAN for the laptop compute server, then run using it")
    ap.add_argument("--no-track", action="store_true",
                    help="run detection without the ByteTrack tracker (avoids the native `lap` "
                         "abort); detection still works, you just lose stable track ids")
    ap.add_argument("--no-enhance", action="store_true",
                    help="skip the OpenCV CLAHE frame-enhance step (avoids a cv2 segfault on a "
                         "flaky native stack); slightly less accuracy in dim light")
    ap.add_argument("--show", action="store_true",
                    help="print detections to the terminal each frame so you can SEE what it detects")
    ap.add_argument("--web", action="store_true",
                    help="(default ON) serve the live dashboard at http://<pi-ip>:PORT")
    ap.add_argument("--no-web", action="store_true",
                    help="headless: do NOT start the web dashboard (Pi without a display)")
    ap.add_argument("--web-port", type=int, default=8080,
                    help="port for the live dashboard (default 8080)")
    args = ap.parse_args()
    if args.no_track:
        os.environ["SOUNDSIGHT_NO_TRACK"] = "1"   # read by VisionCore before the model loads
    if args.diag_imports:
        diag_imports()
    elif args.repair_venv:
        repair_venv(apply=True)
    elif args.scan_corrupt:
        repair_venv(apply=False)
    elif args.reinstall_ml:
        reinstall_ml(apply=True)
    elif args.check_ml:
        reinstall_ml(apply=False)
    elif args.list_cameras:
        cam = _import_camera()
        cam.list_cameras()
    elif args.selftest:
        selftest(args.camera_index)
    else:
        server_url = None
        if args.find_server:
            remote_mod = _import_remote()
            server_url = remote_mod.find_server()      # auto-discover on the subnet
            if server_url:
                log.info("Using compute server: %s", server_url)
            else:
                log.warning("No compute server found -- running on-device.")
        server_url = server_url or os.environ.get("COMPUTE_SERVER_URL")

        # --remote-only: everything on the laptop (Torch-free Pi). Explicit opt-in.
        if args.remote_only:
            raise SystemExit(run_remote_only(server_url, args.camera_index,
                                             show=args.show, web=(not args.no_web), web_port=args.web_port))

        # Otherwise we want the HYBRID: the Pi runs the light Navigate model locally and
        # offloads only the heavy on-demand features (Read/Money/Describe/Faces) to the
        # laptop. That needs Torch on the Pi. If Torch is broken, fall back to remote-only
        # (when a server is configured) so the demo still runs.
        ok, err = torch_import_ok()
        if not ok:
            if "null bytes" in err or "source code string cannot contain" in err:
                log.error("Torch import fails: a venv file is corrupted (NUL bytes).")
                log.error("FIX IT IN PLACE:  python pi_app.py --repair-venv")
            elif err.strip():
                log.error("Torch can't be imported on this Pi: %s", err.strip().splitlines()[-1])
                log.error("Rebuild the venv cleanly:  bash setup_pi.sh")
            if server_url:
                log.warning("Torch is unavailable -> falling back to REMOTE-ONLY (all detection on the laptop).")
                raise SystemExit(run_remote_only(server_url, args.camera_index,
                                                 show=args.show, web=(not args.no_web), web_port=args.web_port))
            log.error("No compute server either. Start one and use --find-server:")
            log.error("  laptop:  python server.py --lan")
            log.error("  Pi:      python pi_app.py --find-server")
            raise SystemExit(1)

        if server_url:
            log.info("HYBRID mode: Pi runs Navigate locally, offloads heavy features to %s", server_url)
        run(server_url, args.camera_index, show=args.show, web=(not args.no_web), web_port=args.web_port,
            no_enhance=args.no_enhance)


if __name__ == "__main__":
    main()
