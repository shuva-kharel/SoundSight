# 👁️ SoundSight

A real-time, voice-driven **visual assistant for blind and low-vision users**. It watches
through a camera and *talks* — describing what's ahead, reading text and labels, identifying
and counting money, recognising traffic lights, greeting known faces, finding objects, and
warning about hazards — in **English and Nepali**, hands-free.

It runs in two places that work together:

- **Laptop (RTX 5060)** — the full app in your browser, *and* an optional **compute server**
  that does the heavy AI on the GPU.
- **Raspberry Pi 4** — a wearable that runs the light real-time loop locally and **offloads**
  heavy on-demand tasks to the laptop over Wi-Fi (falling back to on-device models if the
  laptop isn't reachable).

> **Honest status:** see [FEATURES.md](FEATURES.md) for the exact, audited state of every
> feature (working / partial / disabled-by-flag) and which device it runs on. This README
> describes the design and how to run it.

---

## How it's put together

```
            ┌──────────────────────────── LAPTOP (RTX 5060) ────────────────────────────┐
 browser ◄──┤  server.py  (FastAPI)                                                      │
 (camera,   │   • Navigate WebSocket  • Read/OCR  • Money  • Describe  • Faces  • Crossing │
  Web       │   • HEAVY models (profile=laptop): yolo11m FP16, EasyOCR-GPU, InsightFace,  │
  Speech)   │     Depth Anything V2, Gemini/Ollama VLM                                    │
            │                                                                            │
            │  server.py --lan  ► also a COMPUTE SERVER on the LAN: /remote/*            │
            └───────────────▲────────────────────────────────────────────────────────────┘
                            │  Wi-Fi (phone/laptop hotspot) — frames out, results back
                            │  (remote.py; auto-falls back to on-device if unreachable)
            ┌───────────────┴──────────────── RASPBERRY PI 4 ──────────────────────────┐
            │  pi_app.py                                                                │
            │   • USB camera (V4L2+MJPG)  • LIGHT local loop ALWAYS:                    │
            │     yolo11n NCNN @320 (CPU) + ByteTrack + geometric distance + espeak-ng  │
            │   • Navigate + hazards = 100% local (never network-dependent)             │
            │   • Read / Money / Faces / Describe = OFFLOAD to the laptop, or on-device │
            └───────────────────────────────────────────────────────────────────────────┘
```

**The split is automatic** via `FEATURE_PROFILE` in [vision_core.py](vision_core.py):
a machine with CUDA = **`laptop`** (heavy models); ARM / no-CUDA = **`pi`** (light models).
No code edits to switch.

---

## Features

| Mode | What it does | Voice command | Where it runs |
|---|---|---|---|
| **Navigate** | Continuously detects people/objects/vehicles, tracks them, and speaks **event-driven** ("two people ahead", "chair on your left") — once, not on a loop. Hazards interrupt. | always on (key **1**) | Pi-local + laptop |
| **Read** | OCR of signs/text (Nepali + English), with enhance/deskew/rotate-retry; refuses blurry junk. | "read this" (key **2**) | laptop / offload |
| **Label reading** | On a Read, parses **expiry/MFG dates** (incl. **Bikram Sambat**), **dosage**, product name → "this expired 2 months ago". | "read the label" | laptop / offload |
| **Describe** | One-sentence scene description via **Gemini** (online); offline → local **Ollama VLM** or a summary of current detections. | "what's in front" (key **3**) | laptop (Gemini=internet) |
| **Money — identify** | Reads a held note ("Five hundred rupees") with a confidence+margin gate and **temporal voting** so it **never invents a value** at a wall/hand. | "how much" (key **4**) | Pi/laptop/offload |
| **Money — count** | Sums **multiple** notes laid out in frame → "two 500s and one 100, total 1100". | "count the notes" | laptop/offload |
| **Money — tally** | Running total across captures + change: "add this note", "total", "undo", "pay 500". | "add this note" / "total" | Pi/laptop |
| **Crossing** | Classifies a traffic light's colour (HSV) with hysteresis + **fail-safe** wording ("green … *looks* clear, listen for vehicles"). | "can I cross" (key **C**) | Pi/laptop |
| **Faces** | Greets enrolled people by name in Navigate; "who's here". Opt-in, local only. | "who's here" / "remember this person as Ram" | laptop / offload |
| **Find** | Guides you to a detected object by direction + closeness. | "find a cup" | laptop |
| **Distance** | Camera-only distance: geometric (known sizes) everywhere, **Depth Anything V2** fused to metres on the laptop. | (module; see FEATURES.md) | geom both · depth laptop |
| **SOS** | Loud repeated emergency alert + location, works offline. | "help me" / "emergency" | laptop |
| **TTS** | English = browser Web Speech; **Nepali** = pre-recorded clips → gTTS → espeak-ng (offline). | — | both |
| **Voice control** | Wake word "hey sight", barge-in, live caption, one-at-a-time **priority speech** (hazards > answers > ambient), fuzzy bilingual parser. | toggle (key **7**) | laptop browser (Pi: Vosk, off by default) |
| **DEMO mode** | Steps the hero sequence on cue so a live demo can't stall. | key **D** | laptop |

---

## Quick start — Laptop (the full app)

```bash
# 1. virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows;  Linux/mac: source .venv/bin/activate

# 2. dependencies
pip install -r requirements.txt

# 3. (optional) GPU build of PyTorch for an RTX 50-series (Blackwell) — see "GPU notes" below

# 4. (optional) Describe online — put your Gemini key in a .env file:
#       GEMINI_API_KEY=your-key-here      (get one at https://aistudio.google.com/apikey)

# 5. install espeak-ng (offline Nepali voice):  winget install eSpeak-NG.eSpeak-NG

# 6. run
python server.py                    # then open https://localhost:8000
```

On first visit the self-signed cert shows a warning → **Advanced → Proceed to localhost**.
Allow camera access, then press **7** (or 🎙️ Assistant) for hands-free, or **1** to Navigate.

Verify everything first:
```bash
python server.py --selftest         # PASS/FAIL: models, OCR, TTS, Faces, Gemini key
```

---

## Quick start — Raspberry Pi 4

Full guide: **[README_PI.md](README_PI.md)**. In short:

```bash
git clone <your-repo> soundsight && cd soundsight
bash setup_pi.sh                    # venv + apt (espeak-ng, libgl1, portaudio) + pip
source .venv/bin/activate
# copy yolo11n_ncnn_model/ from the laptop (or it auto-exports on first run)
python pi_app.py --list-cameras     # find your USB camera index
python pi_app.py --selftest         # camera + model + audio + laptop-link check
python pi_app.py                    # run. Navigate + distance + hazards are 100% local.
```
The Pi auto-runs the **light path** (yolo11n NCNN @320, CPU). Audio comes out the 3.5 mm jack
(`sudo raspi-config` → Audio). Pi offline voice (Vosk) is **off by default** (`VOICE_ENABLED`
in `pi_app.py`) — enable it after installing `vosk` + a model on the Pi.

---

## Pi + Laptop together (distributed compute)

The Pi does the real-time loop locally and offloads heavy on-demand work to the laptop GPU.

```bash
# 1. Make a hotspot (phone or laptop). Connect BOTH devices to it.

# 2. On the LAPTOP — start the compute server on the LAN (binds 0.0.0.0, prints its IP):
python server.py --lan
#    -> "Point the Pi at it: export COMPUTE_SERVER_URL=http://192.168.x.x:8000"

# 3. On the PI — auto-discover the laptop and run (or set the URL by hand):
python pi_app.py --find-server
#    or:  export COMPUTE_SERVER_URL=http://<laptop-ip>:8000 && python pi_app.py
```

- **Read / Money / Faces / Describe / Find** run on the laptop GPU (fast, accurate) and are
  spoken on the Pi. **Navigate + hazards stay local** and smooth.
- If the laptop disappears mid-run, those features **fall back to the Pi's own models**, speak
  a one-time *"using on-device mode"*, and reconnect automatically. Nothing hangs or crashes.
- Internet (phone) is needed **only** for Gemini Describe; the laptop uses a local Ollama VLM
  when offline. Pi→laptop is pure LAN (no internet).

`python pi_app.py --selftest` reports the laptop **ONLINE/OFFLINE** + a sample round-trip time.

---

## How the key parts work

- **Navigate is event-driven.** Every object gets a stable ByteTrack id; the `AnnouncementManager`
  speaks only on a *change* (new object, urgency escalation, zone/count change, or a refresh
  after silence) — so it never loops "person nearby" at you. Urgency is **class-aware** (a
  nearby person is "very close"; a same-size bottle is not) and **path-weighted** (things in
  your walking line first). All in [vision_core.py](vision_core.py).
- **Frame quality** ([frame_quality.py](frame_quality.py)) CLAHE-enhances every frame before
  inference and gates genuinely unusable frames (dark/blown-out) — blur is tolerated (YOLO
  handles soft focus), so it doesn't nag on normal frames.
- **Money never hallucinates.** A classifier always outputs *some* class, so Money uses a
  confidence + margin gate, a "background → no note" rule, and **majority voting over ~10
  frames** — a wall or hand yields "no note", never a denomination. ([money.py](money.py),
  `BanknoteClassifier` in vision_core.py)
- **Nepali speech is tiered for quality**: pre-recorded clips in `audio/ne/` for the fixed
  vocabulary → gTTS (online) for arbitrary text → espeak-ng (offline). ([nepali_phrases.py](nepali_phrases.py),
  regenerate with `python gen_voice.py`)
- **Offload is stdlib-only** ([remote.py](remote.py)) — short timeouts, any error → local
  fallback, never a hang.
- **Robustness:** a global error handler makes every endpoint return a spoken-friendly message
  instead of a crash; the camera auto-reopens; heavy models fall back to light on load failure;
  memory is capped on all tracker/announcer state.

---

## Offline vs online

| Runs offline | Needs internet |
|---|---|
| Navigate, distance, hazards, Crossing | **Describe via Gemini** (use a phone hotspot) |
| Read/OCR, Money (all), Faces | gTTS (only for *arbitrary* Nepali text; fixed phrases are pre-recorded) |
| English speech + **Nepali speech** (espeak-ng / pre-recorded) | SOS dispatch (the local loud alert works offline) |
| Voice commands (browser, or Pi Vosk), Pi↔laptop LAN offload | |

---

## Profiles & models (laptop = heavy, Pi = light)

`FEATURE_PROFILE` auto-selects per device. Dial models via the `LAPTOP_*` constants at the top
of [vision_core.py](vision_core.py) (each has a lighter alternative noted):

| Feature | Laptop (`laptop`) | Pi (`pi`) |
|---|---|---|
| Detection | yolo11m @640 **FP16** | yolo11n **NCNN @320** (CPU) |
| OCR | EasyOCR **GPU** | EasyOCR CPU |
| Faces | InsightFace **buffalo_l** | offload-only |
| Depth | Depth Anything V2-Small | geometric only |
| Describe (offline) | Ollama VLM | offload / Gemini |

FP16 keeps VRAM small (yolo11m ≈ 75 MB) so it fits the 8 GB budget alongside the other models.

---

## Self-test & demo

```bash
python server.py --selftest     # laptop: 8/8 checks (model, OCR, TTS, Faces, Gemini)
python pi_app.py --selftest     # pi: camera frame, NCNN model, audio, laptop link + round-trip
python pi_app.py --list-cameras # probe camera indices (OS-aware backend)
```
In the browser, key **D** steps the scripted demo (read → crossing → who's here → money).

---

## Project layout

| File | Purpose |
|---|---|
| `vision_core.py` | All core AI: detection, tracking, urgency, announcer, banknote classifier, OfflineTTS, SceneDescriber, FEATURE_PROFILE |
| `server.py` | FastAPI: the web app **and** the `--lan` compute server (`/remote/*`) |
| `pi_app.py` | Raspberry Pi entrypoint (light local loop + offload + selftest) |
| `index.html` | Browser UI: camera, Web Speech, voice control, priority speech queue, all modes |
| `frame_quality.py` | Shared CLAHE enhance + quality gate + deskew |
| `commands.py` | One shared bilingual voice-command parser (laptop + Pi) |
| `remote.py` | Pi→laptop compute-offload client (stdlib only) |
| `camera.py` | OS-aware camera open (Windows DSHOW / Pi V4L2) |
| `crossing.py` · `money.py` · `labels.py` · `faces.py` · `distance.py` · `nepali_phrases.py` | Per-feature modules |
| `train_banknote.py` · `train_banknote_detect.py` · `predict_banknote.py` | Banknote model training/inference |
| `setup_pi.sh` · `requirements_pi.txt` · `README_PI.md` | Raspberry Pi packaging |
| `FEATURES.md` | **Audited, honest status of every feature** |

---

## GPU notes (RTX 50-series / Blackwell)

`ultralytics` installs a CPU-only `torch` by default. For a Blackwell GPU (sm_120) you need a
CUDA 12.8 build; on Python 3.14 use the nightly wheel:
```bash
pip install --pre --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## What's intentionally NOT here
No auth, no database, no Docker — a focused assistive prototype. Some heavy features depend on
extra installs (InsightFace for Faces — already added; Vosk for Pi voice — off by default);
unreliable paths are disabled behind flags rather than left flaky. See FEATURES.md.
