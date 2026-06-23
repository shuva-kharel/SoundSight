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
        try:
            return _import_module("vision_core").find_espeak()
        except Exception as exc:
            log.warning("Offline TTS unavailable: %s", exc)
            return None

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


def _build_dispatch(speaker, banknote, crossing_mon, mtally, shared, fq, remote):
    """Return on_command(text): parse a voice transcript and run the matching Pi
    handler. On-demand heavy features (Read/Money/Faces/Describe) OFFLOAD to the
    laptop compute server when it's reachable, else fall back to the Pi's local
    models. Navigate/crossing stay 100% local. Same parser (commands.py) as the
    laptop -- no duplicated command logic."""
    from commands import parse_command
    from vision_core import describe_from_detections

    _ocr = {"reader": None}

    def ocr_read(frame):
        if frame is None:
            return "No camera image."
        if _ocr["reader"] is None:
            import easyocr
            speaker.speak("Loading reader")
            _ocr["reader"] = easyocr.Reader(["ne", "en"], gpu=False)   # Pi = CPU
        g = fq.read_preprocess(frame)
        lines = _ocr["reader"].readtext(g, detail=0)
        return " ".join(lines).strip() or "No readable text found."

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
            shared["find"] = None; return
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
        if a == "find":
            shared["find"] = (tgt or "").lower(); say("Looking for " + (tgt or "")); return
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


def run(server_url=None):
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
    banknote = BanknoteClassifier()
    crossing_mon = CrossingMonitor()
    mtally = MoneyTally()
    shared = {"frame": None, "dets": [], "find": None, "last": ""}
    if VOICE_ENABLED:
        start_voice(_build_dispatch(speaker, banknote, crossing_mon, mtally, shared, fq, remote),
                    on_partial=speaker.stop)   # on_partial -> barge-in
    else:
        log.info("Pi voice control DISABLED (VOICE_ENABLED=False). "
                 "Install vosk+sounddevice+a model and set VOICE_ENABLED=True to enable.")

    cap, cam_index, backend = cam.find_camera()   # OS-aware, validated open
    if cap is None:
        log.error("No camera found. Try `python pi_app.py --list-cameras` to locate it.")
        return
    log.info("SoundSight Pi running on camera %d (%s). Ctrl+C to stop.", cam_index, backend)

    fails = 0                 # consecutive failed reads
    low_power = False         # perf guard active?
    win_t0, win_frames = time.time(), 0
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
                        newcap, cam_index, backend = cam.find_camera()
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
                approaching = approach.approaching_objects(detections)
                for item in select_announcements(detections, manager, approaching, now):
                    speak(item["text"])

                # --- active object FIND guidance (set by a voice command) ----- #
                if shared["find"]:
                    tgt = shared["find"]
                    hits = [d for d in detections if tgt in d["label"].lower() or d["label"].lower() in tgt]
                    if hits:
                        d = max(hits, key=lambda d: d.get("area_ratio", 0))
                        from vision_core import zone_for
                        z = zone_for(d["cx"])
                        if d.get("area_ratio", 0) > 0.20 and z == "ahead":
                            speak(f"{tgt} right in front. Found it."); shared["find"] = None
                        else:
                            speak(f"{tgt} {z}")

            except Exception as exc:   # one bad frame must never kill the loop
                log.exception("Loop error (continuing): %s", exc)
                time.sleep(0.05)
                continue

            # --- perf guard: keep the Pi above the FPS floor ------------------ #
            win_frames += 1
            if win_frames >= PERF_WINDOW:
                fps = win_frames / (time.time() - win_t0)
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
                    vision.enhance_frames = True
                    log.info("Perf guard OFF: fps=%.1f -> imgsz=%d, enhance on", fps, IMG_SIZE_FULL)
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


def selftest():
    """PASS/FAIL report: camera opens + returns a real frame, model loads + runs one
    inference, and audio (espeak-ng) is available. Exits 0 if all pass."""
    cam = _import_camera()
    remote_mod = _import_remote()
    vc = _import_vision_core()
    from vision_core import VisionCore, find_espeak

    print("\n==================  SoundSight Pi self-test  ==================")
    results = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:20} {detail}")

    cap, cam_index, backend = cam.find_camera()

    def cam_check():
        if cap is None:
            return False, "no camera found (try --list-cameras)"
        return True, f"index {cam_index}, backend {backend}, real frame received"

    check("camera", cam_check)

    vision_holder = {}

    def model_check():
        vision_holder["v"] = VisionCore(mode="coco-ncnn", accuracy="fast")
        import numpy as _np
        dets = vision_holder["v"].detect((_np.random.rand(240, 320, 3) * 255).astype("uint8"))
        return len(dets) >= 0, f"inference ran (imgsz={vision_holder['v'].imgsz})"

    check("model load + infer", model_check)
    check("audio (espeak-ng)", lambda: (
        bool(find_espeak()),
        find_espeak() or "MISSING -- sudo apt install espeak-ng (Pi) / winget (Windows)"))

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


def main():
    ap = argparse.ArgumentParser(description="SoundSight Raspberry Pi entrypoint")
    ap.add_argument("--selftest", action="store_true",
                    help="check camera/model/audio/offload and exit with a PASS/FAIL report")
    ap.add_argument("--diag-imports", action="store_true",
                    help="isolate native import crashes for cv2/torch/ultralytics")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe camera indices 0-4 (OS-aware backend) and exit")
    ap.add_argument("--find-server", action="store_true",
                    help="scan the LAN for the laptop compute server, then run using it")
    args = ap.parse_args()
    if args.diag_imports:
        diag_imports()
    elif args.list_cameras:
        cam = _import_camera()
        cam.list_cameras()
    elif args.selftest:
        selftest()
    else:
        server_url = None
        if args.find_server:
            remote_mod = _import_remote()
            server_url = remote_mod.find_server()      # auto-discover on the subnet
            if server_url:
                log.info("Using compute server: %s", server_url)
            else:
                log.warning("No compute server found -- running on-device.")
        run(server_url)


if __name__ == "__main__":
    main()
