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
- **Navigate** — streams frames, draws boxes, and speaks. It is **event-driven**:
  it announces a thing **once** ("two people ahead", "chair on your left") and then
  stays quiet — it only speaks again when something actually *changes* (see below),
  so it never loops "caution, person nearby" at you.

### Detection pipeline
Everything lives behind `VisionCore.detect_and_rank()` in
[`vision_core.py`](vision_core.py):
1. **Persistent tracking** — `model.track(persist=True, bytetrack)` gives every
   object a stable `track_id` across frames (with `iou=0.5, agnostic_nms=True,
   max_det=50`). Tracking removes single-frame false positives (a blip never earns
   an id) and is what makes event-driven speech possible. Falls back to
   `predict()` + `(label,zone)` keys if a model can't track.
2. **Model** — `MODEL_MODE`: `"coco"` *(default, `.pt`)* · `"coco-ncnn"` *(Pi)* ·
   `"openvocab"` (YOLO-World) · `"oiv7"`. Size via `MODEL_ACCURACY`:
   `"fast"`=yolo11n (Pi/NCNN) · `"balanced"`=yolo11s (laptop default) ·
   `"accurate"`=yolo11m (GPU). Bigger = more accurate, slower.
3. **Detect everything** — `DETECT_ALL = True`: no allowlist drop; every class the
   model knows (bottle, cup, chair, bag, phone…) is detected. `NAVIGATION_CLASSES`
   only *prioritizes* ranking; `SYNONYMS` give nicer words and never remove a class.
4. **Per-class confidence** — `DEFAULT_CONF = 0.35`; person/vehicles 0.40.
5. **Geometry-only part removal** — a non-person box is dropped only if >70% inside
   a larger person box **and** <30% of its area (glasses/face on a person). A
   standalone bottle/cup/bag is never dropped.
6. **Temporal smoothing** — secondary confirm at **2-of-3** frames (tracking already
   stabilizes).

### Event-driven announcements (the anti-repeat core)
`AnnouncementManager` keeps per-`track_id` memory and speaks **only on an event**:
- **NEW** — a confirmed object/group not introduced yet.
- **ESCALATION** — its urgency rose a tier (far→near→very close). De-escalation is
  **silent**.
- **ZONE CHANGE / COUNT** — it moved zones (held past a deadband) or a group gained a
  member ("two people ahead").
- **REFRESH** — after `REFRESH_SILENCE` (8 s) of silence, it restates the single most
  important object so you're never left guessing.

A stable object that doesn't change is **never repeated**. **Hysteresis** deadbands
(`ZONE_DEADBAND`, `URGENCY_MARGIN`) stop boundary/threshold flip-flop. Objects are
**grouped + counted** with plurals; events are ranked **very-close hazards first**,
capped at `MAX_PER_CYCLE` (2). Tune it all via the constants at the top of
[`vision_core.py`](vision_core.py) (`REFRESH_SILENCE, ZONE_DEADBAND, URGENCY_MARGIN,
MAX_PER_CYCLE, FORGET, MIN_GAP`). Every 15th frame the console logs `RAW`/`KEPT` +
active `track_ids`; whenever it speaks it logs **why** (`[new]/[escalation]/[change]/
[refresh]`).

### Raspberry Pi
[`pi_app.py`](pi_app.py) is the on-device entrypoint: it reads the Pi camera and
speaks through **espeak-ng** (offline), running the same pipeline in `"coco-ncnn"`
(NCNN nano on the ARM CPU). `python pi_app.py` — `sudo apt install espeak-ng` for
audio. It exports the NCNN model once on first run.
- **Read** — captures one full-res frame, runs **bilingual OCR (Nepali +
  English)** via `POST /ocr` (returns `{text, lang}`), and speaks it. English is
  spoken with the browser's Web Speech API; **Nepali** (Devanagari) goes to
  `POST /tts`, which is **tiered** for the best quality: (1) a **pre-recorded file**
  from `audio/ne/` for the fixed vocabulary (denominations, navigation, warnings —
  best, offline); (2) **gTTS** for arbitrary OCR text (good, online); (3)
  **espeak-ng** offline fallback (low quality, always works). Generate the
  pre-recorded set once with `python gen_voice.py` (online), and edit the table in
  [`nepali_phrases.py`](nepali_phrases.py). OCR'd Devanagari is normalized first.
  Install espeak-ng: Windows `winget install eSpeak-NG.eSpeak-NG`, Pi/Linux
  `sudo apt install espeak-ng`.
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
- **Money** (key **4** or **M**) — reads the held Nepali rupee note and says its
  value ("Five hundred rupees" / "पाँच सय रुपैयाँ"). It will **never invent a
  denomination** at a wall/hand: `BanknoteClassifier` ([`vision_core.py`](vision_core.py))
  uses a quality pre-check, a central-ROI crop, a **confidence + margin gate**
  (`MONEY_CONF` 0.85, `MONEY_MARGIN` 0.30), a **background → "no note"** class, and
  the key fix — **temporal voting** over ~10 frames (`MONEY_VOTES`/`MONEY_MIN_AGREE`):
  a value is announced only if it wins a clear majority, else *"no note detected"* /
  *"couldn't read clearly, try again"*. The browser sends the frame burst to
  `POST /money`. Train with [`train_banknote.py`](train_banknote.py) — **collect a
  large, varied `background` (no-note) class** for the strongest rejection.

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

## Accuracy & robustness layer
A wearable camera sees bad light, motion blur and tilt, so every mode runs frames
through one shared QA layer, [`frame_quality.py`](frame_quality.py):
- **`enhance()`** — CLAHE adaptive contrast (+ auto-gamma when dark), applied before
  every inference/OCR. The single biggest real-world accuracy win. Toggle with
  `ENHANCE_FRAMES`; the Pi auto-disables it under load.
- **`assess()`** — brightness + blur (variance of Laplacian). Junk frames are
  **skipped** instead of emitting false positives; if a bad view persists ~2 s the
  app says *"camera view is dark/unclear"* once (rate-limited, bilingual).

Per-mode upgrades, all degrade gracefully on the Pi (`coco-ncnn`, CPU):
- **Navigate** — model tiers (`fast`=yolo11n@320 / `balanced`=yolo11s@640 /
  `accurate`=yolo11m@640 +TTA; Pi never uses `accurate`). **Class-aware urgency**
  (`URGENCY_SCALE`): a nearby *person/vehicle* triggers "very close", a same-size
  *bottle/phone* does not. **Path weighting**: things in your walking line (center +
  lower frame) are announced first. ByteTrack + the event-driven `AnnouncementManager`
  (no spam) are unchanged; per-track memory is capped (`MAX_TRACKS`).
- **Read** — CLAHE → 2× upscale → deskew before EasyOCR; per-box confidence filter
  (`OCR_CONF`), natural top→bottom/left→right ordering, and a **sideways rescue**
  (retries 90/180/270° if low-confidence). Refuses to read blurry/dark junk.
- **Describe** — **works offline**: with no key or no internet it synthesizes a
  spoken description from the current detections (`describe_from_detections`, e.g.
  *"In front of you: two people ahead, a chair on your left."*); online it adds an
  ~8 s timeout + one retry and a hazard/stairs/doorway/sign-reading prompt.

Camera health (covered/frozen/yanked cam) and a Pi **perf guard** (drop imgsz 320→256
and disable enhance below `FPS_FLOOR`, restore when it recovers) keep the loop alive;
every mode's errors are isolated so one can't crash Navigate. All knobs are top-of-file
constants in [`frame_quality.py`](frame_quality.py) / [`vision_core.py`](vision_core.py).

## Hands-free voice control (the assistant)
Press **7** / 🎤 (or `V`) to start listening. It's an always-on assistant with
**Navigate as the ambient default**; commands switch modes and return to Navigate.
- **Wake word**: say **"hey sight …"** so background speech is ignored. ("hey sight,
  read this" / "hey sight, how much".) Bare "hey sight" → "Yes?".
- **Barge-in**: start talking and the assistant stops speaking and listens.
- **Priority speech**: only one thing speaks at a time — hazards > command answers >
  ambient navigation.
- **Bilingual**: English *and* Nepali keywords (पढ = read, कति = how much, अगाडि के छ =
  what's ahead, बाटो काट = cross, यो को हो = who's here, नोट जोड = add note…).
- **Commands**: navigate · what's in front (Describe) · read this · read the label ·
  how much · add this note / total / clear / undo / "pay 500" · can I cross · find my
  {object} · who's here · remember this person as {name} · repeat · louder/softer/
  slower/faster · stop · **help**. A live caption shows what was heard.
- **Keys** also work: `C` crossing · `H` help · `D` steps the **DEMO** sequence
  (read → crossing → who's here → banknote) so a live demo can't stall.

The shared parser is [`commands.py`](commands.py) — the browser POSTs transcripts to
`/command` and the Pi calls it directly, so both use identical logic.

### Crossing, labels, money tally, faces
- **Crossing** ([`crossing.py`](crossing.py)) — classifies a traffic light's color by
  HSV, with **hysteresis** (confirmed changes only) and **fail-safe** wording
  ("green … *looks* clear, listen for vehicles"; unclear/conflicting → "be careful").
  Runs inside Navigate and answers "can I cross".
- **Label reading** ([`labels.py`](labels.py)) — "read the label" parses expiry/MFG
  dates (incl. **Bikram Sambat**), dosage, and product name → "This expired 2 months
  ago", etc. Normal Read is unchanged.
- **Money tally** ([`money.py`](money.py)) — "add this note / total / undo / pay 500"
  keeps a running sum and computes change with a note breakdown, in EN + NE. Requires
  classifier confidence > 0.7 to add.
- **Faces** (opt-in, local) ([`faces.py`](faces.py)) — greets enrolled people by name
  in Navigate and answers "who's here". Enroll first (consent-first):
  `python face_enroll.py --name Ram`. Needs a provider:
  `pip install insightface onnxruntime` (best) or `pip install face_recognition`.
  Disabled gracefully if neither is installed.

## Self-test
Verify the moving parts before a demo (PASS/FAIL per check):
```bash
python server.py --selftest   # model, frame-QA, banknote, espeak-ng, EasyOCR, Gemini key
python pi_app.py --selftest    # camera opens + returns a frame, NCNN model, audio, SoC temp
python pi_app.py --list-cameras  # probe camera indices 0-4 (OS-aware backend)
```

## Logging
On startup the backend prints which device YOLO uses (`CUDA` or `CPU`). During
Navigate the 15-frame debug log adds **frame brightness + blur score**, whether
**enhance** ran, the imgsz, **why** each object was announced
(`new/escalation/change/refresh`), OCR **mean confidence** per Read, and whether
Describe used **Gemini or the offline fallback**.

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
