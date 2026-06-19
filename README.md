# 👁️ SoundSight

A browser-testable, real-time visual assistant for blind users.

The **browser** owns the webcam (`getUserMedia`) and the voice (Web Speech API).
The **Python backend** does all the AI (YOLO object detection + EasyOCR). They
talk over a WebSocket. The vision/AI code lives in [`vision_core.py`](vision_core.py)
with **no web code**, so it ports to a Raspberry Pi unchanged — the web layer in
[`server.py`](server.py) is the only thing you replace on-device.

```
browser ──(downscaled JPEG frames)──►  server.py  ──►  vision_core.py (YOLO)
   ▲                                       │
   └──────────(JSON: boxes + speak)────────┘
```

## Modes
- **Navigate** — streams frames, draws bounding boxes, and speaks e.g.
  `"person ahead"`, `"chair on your left"`. It stays calm: max 2 items/frame, each
  object+zone is announced at most once per **2.5s** cooldown, and **very-close**
  hazards (`area_ratio > 0.20`) bypass that to re-warn every **1s**.

### Detection pipeline (navigation-focused, anti-spam)
The whole pipeline lives behind `VisionCore.detect_and_rank()` in
[`vision_core.py`](vision_core.py) and is built to be *clean and stable* rather
than detect-everything:
1. **Model** — set by `MODEL_MODE` (top of the file):
   - `"coco"` *(default)* — `yolo11s.pt`, 80 COCO classes; fast, accurate, and
     almost all nav-relevant.
   - `"openvocab"` — YOLO-World restricted to `OPENVOCAB_CLASSES` (best fit; only
     detects what a blind user needs). First use auto-installs CLIP via
     ultralytics — you may need to restart the server once.
   - `"oiv7"` — `yolov8s-oiv7.pt` (~600 classes) behind the same filters.
2. **Allowlist** — only `NAVIGATION_CLASSES` (person, vehicles, doors, stairs,
   furniture, poles…) survive; everything else is dropped, and synonyms are
   grouped to one spoken word (`sofa→couch`, `dining table→table`).
3. **Part removal** — `PART_CLASSES` (glasses, human face/hand/arm, footwear,
   clothing…) are always discarded, plus worn/held items mostly inside a person
   box are suppressed.
4. **Per-class confidence** — `CONF_THRESHOLDS` (person 0.40, vehicles/furniture
   0.45, structure 0.50, small/ambiguous 0.55).
5. **Temporal smoothing** — an object is only announced after it's seen in **3 of
   the last 5 frames**, killing single-frame flickers/ghosts.
6. **Ranking** — by closeness, then importance (person/vehicles/stairs/poles
   highest), capped at 2 spoken items.

The console logs `raw=… kept=… dropped: …` (~once/second) so you can tune
thresholds. To detect non-navigation items again (e.g. **pen**, **headphones**),
add them to `NAVIGATION_CLASSES` and use `"oiv7"` or `"openvocab"` mode — the
default is deliberately navigation-only.
- **Read** — captures one full-res frame, runs **bilingual OCR (Nepali +
  English)** via `POST /ocr` (returns `{text, lang}`), and speaks it. English is
  spoken with the browser's Web Speech API; **Nepali** (Devanagari) is sent to
  `POST /tts`, which uses **gTTS** to return an MP3 the browser plays — because
  the offline browser voice reads Devanagari poorly. **gTTS needs internet**; for
  the Pi, swap `_synth_mp3()` in [`server.py`](server.py) for an offline Nepali
  TTS (espeak-ng / Piper / Coqui), same contract.
- **Repeat** (key **4** or **R**) — re-speaks the last Read result without
  re-capturing (handy if you missed it or want it slower to follow).
- **Language** (key **5** or **L**) — cycles **Auto → English → नेपाली**. Auto
  trusts the OCR's detected script; pick English or Nepali to force the voice when
  auto-detection guesses wrong. Each press speaks a confirmation; then press
  **Repeat** to re-hear the last text in the chosen language.
- **Describe** — captures one frame, sends it to a vision-language model
  (Google **Gemini**) via `POST /describe`, and speaks a short scene description.
  Needs a `GEMINI_API_KEY` (see below). The provider is isolated in
  `SceneDescriber.describe()` ([`vision_core.py`](vision_core.py)) — swap it for a
  local VLM or another API without touching the server or browser.

### Proximity urgency (Navigate)
Each detection's closeness is estimated from how much of the frame it fills,
`area_ratio = box_area / frame_area`, and classified:

| urgency | area_ratio | box color | speech |
|---|---|---|---|
| far | `< 0.05` | green | `"{label} {zone}"` |
| near | `0.05 – 0.20` | yellow | `"{label} {zone}"` |
| very close | `> 0.20` | red (thicker) | `"Careful, {label} {zone}, very close"` |

- **Very-close** warnings bypass the normal cooldown (re-announced every **1s**),
  are spoken **faster** (`rate 1.3`), and **interrupt** calmer speech.
- An object whose `area_ratio` is **growing** frame-over-frame is flagged as
  approaching: `"{label} {zone}, getting closer"`.
- The approach test lives in one clean swap point —
  `ApproachDetector.approaching_objects()` in [`vision_core.py`](vision_core.py).
  On the Pi, replace its body with **ultrasonic-sensor** distance deltas; keep
  the signature (detections in, set of `(label, zone)` keys out) and the rest of
  the pipeline is unchanged.

## Setup & run

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell:  .venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) enable Describe mode -- add your Gemini API key
#    Copy .env.example to .env and put your key in it:
#        GEMINI_API_KEY=your-key-here
#    Get a key at https://aistudio.google.com/apikey

# 4. run the server
python server.py
```

> **Describe mode** needs a `GEMINI_API_KEY`. The simplest way is a **`.env` file**
> in the project root (the server auto-loads it on startup):
>
> ```
> GEMINI_API_KEY=your-key-here
> ```
>
> `.env` is git-ignored so your key isn't committed. (You can also set the env var
> by hand instead: PowerShell `$env:GEMINI_API_KEY="..."`, bash
> `export GEMINI_API_KEY="..."`.) Without a key, Navigate and Read still work;
> Describe just says "set GEMINI_API_KEY and restart". The model defaults to
> `gemini-2.5-flash` (change `DESCRIBE_MODEL` in [`vision_core.py`](vision_core.py)
> to `gemini-3.5-flash` for richer descriptions).

Then open **https://localhost:8000**. The cert is self-signed, so on the first
visit click **Advanced → Proceed to localhost (unsafe)** — this is your own
machine. Allow camera access, press **1** (or click **Navigate**), and within a
second you should see boxes and hear *"person ahead"*.

> **Why HTTPS?** `getUserMedia` works on plain `http://localhost`, but modern
> browsers (Chrome's HTTPS-First mode, HSTS, etc.) auto-upgrade `localhost` to
> `https://`. Hitting a plain-HTTP server that way gives `ERR_SSL_PROTOCOL_ERROR`.
> Serving real HTTPS sidesteps the browser. If you're *sure* your browser won't
> upgrade, you can run plain HTTP instead:
>
> ```bash
> python server.py --http      # then open http://localhost:8000
> ```
>
> Open `localhost`, not your LAN IP.

The first run downloads the detection model (`yolo11s.pt`, ~19 MB) automatically.
The first time you use **Read**, EasyOCR downloads its Nepali + English models —
Navigate is unaffected.

## Logging
On startup the backend prints which device YOLO uses (`CUDA` or `CPU`). During
Navigate it logs FPS every 30 frames and the OCR character count / latency per
read.

## GPU notes (your RTX 5060 / Blackwell)

`ultralytics` installs `torch` for you, but the **default wheel is CPU-only**
(`torch x.y.z+cpu`), so the startup log says `CPU`. Blackwell (RTX 50-series,
`sm_120`) also needs a **CUDA 12.8** build. On **Python 3.14** there is *no
stable* cu128 wheel yet (stable tops out at Python 3.13), so you must use the
**nightly** cu128 build — it does ship a Windows `cp314` wheel:

```bash
# install the GPU build over the CPU one (~2.5 GB download)
pip install --pre --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify the GPU is seen:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 5060 Laptop GPU
```

Then restart `python server.py`; the log should now read
`loaded on device: CUDA`.

### Python version
You're on **Python 3.14**. The CPU stack installs fine there, and the **nightly**
cu128 GPU build above has a `cp314` Windows wheel. If you'd rather use the
*stable* GPU stack (cu128 up to Python 3.13), recreate the venv with 3.13:

```bash
py -3.13 -m venv .venv
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Porting to Raspberry Pi
Keep [`vision_core.py`](vision_core.py) as-is. Replace [`server.py`](server.py)'s
transport with your hardware loop: grab frames from the Pi camera, call
`VisionCore.detect()` + `select_announcements()`, and drive a speaker/haptics
instead of the browser. The `Announcer`, zones, and priority logic come along
unchanged.

## What's intentionally NOT here
No auth, no database, no Docker — minimal and working by design.
