# SoundSight — Verification Report (ground truth)

Honest current state of the code, not the aspirational spec. "Tested" = logic
unit-tested on this dev machine with stubbed native libs (no GPU/camera/torch here).
It does **not** mean hardware-validated. Items needing a real Pi + laptop + camera are
🟡 **HW-UNVERIFIED**.

Legend: ✅ implemented & logic-tested · 🟡 implemented, HW-unverified · ⛔ specified but NOT implemented

---

## A. Feature status

### Navigate (Pi + laptop) — ✅ core
| Capability | Where | Profile | Status |
|---|---|---|---|
| Camera open (V4L2/MJPG, auto-probe 0–2, warm-up, auto-reopen) | `camera.py`, `pi_app.run()` | pi | 🟡 (index 0 confirmed in user logs) |
| yolo11n NCNN @320 CPU + ByteTrack | `vision_core.VisionCore` | pi | 🟡 (runs once native stack healthy — §D) |
| Geometric distance per detection | `distance.py` → `vision_core` Stage 4a | both | ✅ wired on every det |
| Depth-fusion distance (Depth Anything V2-Small) | `distance.DepthEstimator` | laptop | ⛔ **OFF by default** (cost ~0.7 s/frame → 2.5 FPS; re-enable `SOUNDSIGHT_DEPTH=1`) |
| **Distance spoken**, rounded (0.25 m <2 m, 0.5 m ≥2 m, "very close" <0.5 m) | `distance.spoken_distance`, `vision_core._phrase` | both | ✅ tested |
| Close-range clip-cap (frame-filling/clipped box → "very close", fixes geom over-estimate) | `distance._proximity_cap` | both | ✅ tested |
| **GPU diagnostic** at startup (CPU → slow + geometric-only; prints cu128 fix for RTX 50xx) | `server._gpu_diagnostics` | laptop | ✅ |
| Distance drives urgency (<1/1–3/>3 m) + ranking (closer first) | `vision_core.detect_and_rank`, `AnnouncementManager` | both | ✅ tested |
| Sub-modes STREET/PUBLIC/HOME (priorities + room context) | `vision_core.set_sub_mode/importance_of` | both | ✅ tested |
| espeak-ng TTS, detached (never blocks loop) | `pi_app.Speaker` | pi | 🟡 |
| "how far" / "scan area" | `vision_core.how_far_phrase/scan_area_phrase`, `/howfar`,`/scan` | both | ✅ tested |

### Traffic light / crossing — ✅
| HSV classify + fail-safe (conflict→"unclear", none→"No traffic light visible") | `crossing.py` | ✅ |
| Hysteresis **5 frames** before announcing | `crossing.CONFIRM_FRAMES=5` | ✅ |
| Auto-announce confirmed change **only in STREET** | `pi_app.run()`, `server.ws_navigate` | ✅ gating / 🟡 color on HW |
| C key / "can I cross" on-demand query | `index.html doCross`→`/cross`; Pi voice `cross` | 🟡 |

### Find mode (Object Finder) — ✅ logic
| Scan 2.5 s → numbered pick-list w/ position+distance | `finder.start_scan/update_scan` | ✅ tested |
| "find a {cup}" → immediate lock, no list | `finder.find_class` | ✅ tested |
| Choose by letter (A) or name | `finder.choose` | ✅ tested |
| ByteTrack lock; guidance (zone+distance+clock); beacon; success/lost | `finder.guide` + `pi_app.Beacon` (aplay) | ✅ logic / 🟡 beep on HW |
| Room scan + count + room inference | `finder.room_scan` | ✅ tested |
| Personal find (CLIP/histogram, `objects_db/`) | `finder.PersonalObjectDB`, `/find/register`, `/remote/find` | 🟡 HW-UNVERIFIED |
| Browser Find UI over WS (server-side finder) | `server.ws_navigate` + `index.html` | ✅ F/A–Z/R/Esc keys, pick-list panel, Web Audio beacon, voice (find/track/room) — JS parses; render needs HW |
| Voice commands (find_mode/track/room_scan/found) | `commands.py` | ✅ tested |

### Pi live dashboard — ✅ `pi_server.py` (FastAPI + WS)
| `pi_server.py` :8080 — `/` page, `/stream` MJPEG, **`/status` WebSocket @200 ms** | `pi_server.py` | ✅ DashboardState tested; WS needs HW |
| Thread-safe `DashboardState` written by loop (zero coupling) | `pi_server.DashboardState` | ✅ tested |
| `pi_dashboard.html`: video + mode/sub-mode, FPS, **laptop ONLINE/OFFLINE+latency banner**, last 6 spoken (timestamps), detections+distance, traffic-light circle, find pick-list, OCR panel; WS auto-reconnect | `pi_dashboard.html` | ✅ JS parses + elements present; render needs HW |
| Web **ON by default**, `--no-web` headless; stdlib `pi_web` fallback if fastapi absent | `pi_app._start_dashboard` | ✅ tested (fallback) |

### Pi↔Laptop offload — ✅ hardened
| HTTP `--lan` / HTTPS `--lan-web`; `remote.py` speaks both; `find_server` https→http | `server.py`,`remote.py` | ✅ tested (mock handshake) |
| **2-strike OFFLINE state machine** (`link_status` online/connecting/offline, anti-flap) | `remote.RemoteCompute` | ✅ tested |
| **Per-endpoint timeouts** (detect/faces 1.5 s, ocr/money/find 5 s, describe 20 s) | `remote.py` | ✅ |
| **Bounded heavy inflight** (drop past `MAX_INFLIGHT_HEAVY=2`) | `remote._guard(heavy=True)` | ✅ tested |
| Latency cuts: **JPEG q75, 640 px** transfer | `remote._encode` | ✅ |
| `/remote/health`, **`/remote/ping`** (gpu, models_loaded, queue_depth) | `server.py` | ✅ |
| **`X-Processing-Time` header** + slow-call (>250 ms) warning + CORS + LAN-IP banner | `server._timing` middleware | ✅ |
| `/remote/detect` returns detections + **distance + track_id** in one response | `server.py` | ✅ |
| Graceful local fallback on any error/timeout (returns None) | `remote._guard` | ✅ |

### Read mode — ✅ reworked (pre-warm + accuracy)
| Capability | Where | Status |
|---|---|---|
| **Pre-warm at startup** (background thread, kills 30 s delay) | `server.prewarm_ocr` (@startup), `pi_app.run()` thread | ✅ (logic; HW load time needs device) |
| `ocr_ready` LOADING→READY in dashboard HUD | `pi_app` → DashboardState; `/remote/ping.ocr_engine` | ✅ |
| **PaddleOCR** (PP-OCRv4 devanagari) on laptop, EasyOCR fallback; Pi = EasyOCR | `server._PaddleReader`, `OCR_ENGINE` | 🟡 (guarded; PaddleOCR not installed here) |
| Quality gate (refuse dark/blurry, guide user) | `server.ocr`, `pi_app.ocr_read` | ✅ |
| Enhance (CLAHE/upscale/deskew) + ROI crop-to-text | `frame_quality.read_preprocess`, `server._center_roi` | ✅ |
| Rotate-retry 90/180/270 on low conf; drop <`OCR_CONF`; reading-order sort; read once | `server._ocr_best/_ocr_pass`, `pi_app.ocr_read` | ✅ (both Pi + laptop now) |
| Offline speech: English espeak-ng (Pi) / Web Speech (browser); Nepali espeak `-v ne` | `pi_app.Speaker`, `index.html` | 🟡 (pre-recorded NE phrase files ⛔; espeak NE is the offline path) |

### Money / Describe / Faces — 🟡 pre-existing
| `/money`,`/money/count`,`/money/tally` temporal vote | `server.py`,`money.py` | 🟡 |
| `/describe` Gemini + offline detection-summary fallback | `vision_core.SceneDescriber` | 🟡 |
| `/faces/who`,`/faces/enroll` InsightFace | `faces.py` | 🟡 |

### Environment repair tooling — ✅
`--scan-corrupt`/`--repair-venv` (NUL corruption), `--check-ml`/`--reinstall-ml`
(numpy-2 ABI aborts), `--no-track`/`--no-enhance` safe-mode, `--show`. `pi_app.py`, compile-clean.

---

## B. Config flags & current defaults

| Flag / const | File | Default | Meaning |
|---|---|---|---|
| `FEATURE_PROFILE` | vision_core | auto (laptop if CUDA & !ARM) | heavy vs light models |
| `LAPTOP_DETECT_MODEL` | vision_core | **`yolo11m`** | laptop detector (yolo11l upgrade ⛔) |
| `DEPTH_MODEL` | distance | **`Depth-Anything-V2-Small-hf`** | laptop depth (Base ⛔) |
| `DIST_VERY_CLOSE / DIST_NEAR` | distance | 1.0 / 3.0 m | urgency tiers |
| `CAMERA_HFOV_DEG / FOCAL_PX` | distance | 70° / None | geometric calibration |
| `_sub_mode` | vision_core | `street` | Navigate sub-mode |
| `CONFIRM_FRAMES` | crossing | 5 | traffic-light hysteresis |
| `FINDER_BEACON / SCAN_SECONDS / ARM_REACH` | finder | True / 2.5 s / 0.7 m | find tuning |
| `web` (dashboard) | pi_app | **ON** (`--no-web` to disable) | live dashboard :8080 |
| `--web-port` | pi_app | 8080 | dashboard port |
| `REMOTE_TIMEOUT` | remote | 3.0 s | default offload timeout |
| `LIGHT_TIMEOUT / HEAVY_TIMEOUT / DESCRIBE_TIMEOUT` | remote | 1.5 / 5.0 / 20.0 s | per-endpoint |
| `HEALTH_INTERVAL / HEALTH_FAIL_LIMIT` | remote | 5 s / 2 | health re-check / strikes to OFFLINE |
| `MAX_INFLIGHT_HEAVY` | remote | 2 | bound on queued heavy calls |
| `JPEG_QUALITY / MAX_SEND_W` | remote | 75 / 640 px | transfer-size latency cut |
| `OCR_CONF / OCR_RESCUE_CONF` | server | 0.30 / 0.50 | drop fragments / rotate-retry threshold |
| `OCR_ENGINE` | server | `auto` (paddle on laptop, else easyocr) | OCR backend |
| `PREWARM_OCR` (`PI_PREWARM_OCR`) | pi_app | True | background-load EasyOCR at Pi startup |
| `VOICE_ENABLED` | pi_app | **False** | Pi Vosk voice off by default |
| `--no-track` / `--no-enhance` | pi_app | off | native-crash safe mode |

---

## C. Still NOT implemented (deferred)

1. **Read-mode** — DONE except: PaddleOCR is guarded/optional (not installed/tested here),
   and pre-recorded Nepali phrase files (espeak `-v ne` is the offline NE path instead).
2. **Final integration pass** — unified `ModeManager`, single priority speech-queue object,
   `--demo` scripted mode, expanded PASS/FAIL/SKIP `--selftest` (current selftest checks
   camera/model/audio/offload/temp only).
3. **Model upgrades** — yolo11l, Depth-V2-Base, PaddleOCR, InsightFace FP16 verify, VRAM
   budget log, per-model startup warm-up, every-frame depth piggybacked on `/remote/detect`,
   face names attached to track ids in `/remote/detect`.
**Now DONE that earlier reports listed as ⛔:** `pi_server.py` dashboard, `/remote/ping`,
`X-Processing-Time`, JPEG-75/640 px transfer, 2-strike OFFLINE machine, per-endpoint timeouts,
Read-mode pre-warm + PaddleOCR option, **browser Find-mode UI (keys + pick-list + Web Audio beacon)**.

`/ocr`,`/money`,`/describe` are the real endpoint names (the Pi's `remote.py` calls these
correctly); there are no `/remote/ocr` etc. aliases.

---

## D. Known blocking issue on the Pi (environment, not code)

Pi venv crashes from **numpy 2.x ABI mismatch** with numpy-1-built wheels (`lap`/`torchvision`/
`cv2` aborts), plus earlier SD-card corruption. Fix:
```
pip install --no-cache-dir 'numpy<2'      # primary
python pi_app.py --check-ml               # NMS + lap + opencv must PASS
python pi_app.py --reinstall-ml           # rebuild native stack if needed
```
`requirements_pi.txt` pins `numpy<2`. Guaranteed demo path meanwhile: laptop
`python server.py --lan`, Pi `python pi_app.py --find-server` (dashboard on by default →
`http://<pi-ip>:8080`) — camera+audio on Pi, detection on laptop, sidesteps the native stack.

---

## E. Verified this session (unit tests, stubbed natives)

Passed: distance rounding (0.25/0.5/very close), distance→urgency, distance spoken in
announcements, sub-mode importance (street vs home), how-far/scan, room inference, finder
scan→pick-list→choose→guide→beacon→success, lost-after-3 s, command parsing, HTTP+HTTPS
offload handshake, **2-strike OFFLINE state machine + heavy-inflight drop**, **DashboardState
snapshot/spoken-log**, **pi_dashboard.html JS parse + required elements**. Every touched
`.py` compiles.

**Not verifiable here (need Pi + laptop + camera + GPU):** real YOLO/NCNN inference & FPS,
depth accuracy, EasyOCR, espeak/aplay audio, InsightFace, camera capture, live WebSocket
push, MJPEG render, end-to-end round-trip latency.
