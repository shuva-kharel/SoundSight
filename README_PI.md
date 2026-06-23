# 👁️ SoundSight on Raspberry Pi 4

Run SoundSight headless and **offline** on a Raspberry Pi 4 with a USB webcam
(e.g. Fantech). The laptop's CUDA setup is NOT assumed — the Pi auto-runs the light
path (NCNN nano model, `device=cpu`, `imgsz 320`).

## Requirements

- **Raspberry Pi 4** (2 GB+), **64-bit Raspberry Pi OS** (aarch64). 32-bit will NOT work.
- A USB webcam, a **3.5 mm speaker/earphone** (or USB audio), and a USB mic for voice.
- A heatsink/fan is strongly recommended (sustained inference heats the SoC).

## 1. Clone & set up

```bash
git clone <your-repo-url> soundsight && cd soundsight
bash setup_pi.sh          # creates .venv, installs apt + pip deps, prints next steps
source .venv/bin/activate
```

`setup_pi.sh` installs the apt packages SoundSight needs: `espeak-ng` (offline TTS),
`alsa-utils`, `libgl1` + `libglib2.0-0` (OpenCV), `portaudio19-dev` (mic for Vosk),
then `pip install -r requirements_pi.txt`.

> On ARM, `ultralytics` pulls the **aarch64 CPU build of torch** automatically.
> Do **not** install any `+cu` / GPU wheel.

## 2. Copy the NCNN model (faster first run)

Copy the exported nano model from your laptop so the Pi doesn't have to export it:

```bash
# from the laptop, in the project folder:
scp -r yolo11n_ncnn_model  pi@<pi-ip>:~/soundsight/
```

If it's missing, the Pi **auto-exports** it from `yolo11n.pt` on first run (slower, one-time).
For Money mode, also copy `models/banknote_ncnn_model/` (or `models/banknote.pt`).

## 3. Voice model (offline, optional but recommended)

Voice control uses **Vosk**, which runs **fully offline** once a model is downloaded:

```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models_vosk
```

SoundSight auto-finds a model in `./models_vosk/` (or set `export VOSK_MODEL_DIR=/path`).
Without a model, the app still runs — just without voice (it says so in the log).

## 4. Audio out the 3.5 mm jack

```bash
sudo raspi-config        # System Options -> Audio -> select the 3.5mm jack (Headphones)
# or:  amixer cset numid=3 1     # 1 = 3.5mm jack, 2 = HDMI, 0 = auto
speaker-test -t sine -f 440 -l 1 # quick check; espeak-ng "hello" should be audible
```

## 5. Verify, then run

```bash
python pi_app.py --selftest      # PASS/FAIL: camera frame, NCNN model+infer, audio, temp
python pi_app.py --diag-imports  # if startup segfaults: isolate cv2/torch/ultralytics
python pi_app.py --list-cameras  # which index your webcam is on (0/1/2…)
python pi_app.py --camera-index 1 # run with the working camera index. Ctrl+C to stop.
```

### `SyntaxError: source code string cannot contain null bytes`

If `--selftest` / `--diag-imports` shows torch (or `ultralytics` / `vision_core`)
failing with **"source code string cannot contain null bytes"**, a Python file in
the venv is **corrupted** — half-written by an interrupted `pip install`, a power
cut, or a failing SD card. It is _not_ a torch/Python version problem. Fix it in
place by reinstalling only the corrupted packages:

```bash
python pi_app.py --scan-corrupt   # report which packages are corrupted (read-only)
python pi_app.py --repair-venv    # force-reinstall them, then verify torch imports
python pi_app.py --selftest       # confirm model load + infer now PASS
```

If `--repair-venv` says files are _still_ corrupted after reinstalling, the SD card
is likely failing — reflash to a fresh card and re-run `bash setup_pi.sh`.

### `corrupted double-linked list` / `Fatal Python error: Aborted` during inference

If torch _imports_ fine but the **first detection aborts** with `corrupted
double-linked list` (or a SIGABRT inside `torchvision::nms` / `lap`), the native
torch/torchvision build is broken or mismatched — typically a **GPU/CUDA torch
wheel on the Pi** (it must be the CPU build; the tell-tale is `nvidia-*` / `pynvml`
packages being installed) or a torchvision compiled against a different torch.
Fix it with a clean, matched CPU reinstall:

```bash
python pi_app.py --check-ml       # shows torch/torchvision versions + runs the NMS op that aborts
python pi_app.py --reinstall-ml   # reinstalls matched CPU torch+torchvision, then re-verifies NMS
python pi_app.py --selftest       # confirm 'model load + infer' now PASS
```

`--reinstall-ml` pulls the CPU wheels from PyTorch's CPU index. If it's still
unhealthy afterward, it prints the piwheels (Raspberry-Pi CPU) fallback command.

### torch segfaults / can't be fixed: run laptop-offload mode

If `--diag-imports` shows `torch` **segfaulting** (not the null-bytes error above),
run laptop-offload mode instead:

```bash
# laptop
python server.py --lan

# Raspberry Pi
python pi_app.py --find-server --camera-index 0
```

In this mode the Pi uses its camera and speaker, but sends detection to the laptop
and does not import Torch/Ultralytics on the Pi.

There are two LAN modes on the laptop, pick by _which device holds the camera_:

- `python server.py --lan` — **plain HTTP** on `0.0.0.0`. Fast (no per-frame TLS) for
  the **Pi's** real-time offload loop (`--find-server` / `--remote-only`). Use this
  when the **camera is on the Pi**.
- `python server.py --lan-web` — **HTTPS** on `0.0.0.0` with a self-signed cert that
  includes the laptop's LAN IP. Use this when you open the **web app in a browser**
  on another device (phone, the Pi's browser): `getUserMedia` (camera) only works in
  a secure context (`https://` or `http://localhost`), so plain HTTP can't access the
  camera from another device. First visit, accept the one-time cert warning.

The Pi's `remote.py` client and `--find-server` discovery speak **either** HTTP or
HTTPS automatically, so the Pi can offload to whichever mode the laptop is running.

Camera open is OS-aware (V4L2 + MJPG on the Pi), tries indices 0/1/2, validates a real
frame, warms up, and auto-reopens a yanked/frozen cam. Voice: say **"hey sight …"**.

## What's offline vs online

| Feature                                                        | Offline on the Pi?                                     |
| -------------------------------------------------------------- | ------------------------------------------------------ |
| Navigate (detection, tracking, urgency, crossing)              | ✅ yes                                                 |
| Read (EasyOCR) + label parsing                                 | ✅ yes                                                 |
| Money (banknote classifier, tally)                             | ✅ yes                                                 |
| English speech (espeak-ng) + Nepali speech (espeak-ng `-v ne`) | ✅ yes                                                 |
| Voice commands (Vosk)                                          | ✅ yes (after the model download)                      |
| **Describe (Gemini)**                                          | ❌ needs internet + `GEMINI_API_KEY`. Falls back to an |
| offline summary of current detections.                         |
| **Nepali gTTS** (not used — espeak-ng covers Nepali offline)   | n/a                                                    |

## Performance tips

- The **perf guard** auto-drops `imgsz` 320→256 and disables frame-enhance if FPS falls
  below the floor or the SoC gets hot, then restores when it recovers.
- Add swap if you hit memory pressure during the first EasyOCR/NCNN load:
  `sudo dphys-swapfile swapoff && sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon`
- Keep the Pi cool (heatsink + fan); throttling shows up as low FPS in the log.

## Command reference

### `pi_app.py` (Raspberry Pi)

| Command                             | What it does                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `python pi_app.py`                  | Run on-device (camera + NCNN model + offline TTS). Default.                                            |
| `python pi_app.py --camera-index 0` | Run with a specific camera index (from `--list-cameras`). Also `CAMERA_INDEX=0`.                       |
| `python pi_app.py --find-server`    | Auto-discover the laptop on the LAN and offload detection to it (no Torch on the Pi).                  |
| `python pi_app.py --remote-only`    | Force laptop-offload mode (uses `COMPUTE_SERVER_URL`). Torch-free.                                     |
| `python pi_app.py --selftest`       | PASS/FAIL: camera frame, model load + infer, audio, offload, SoC temp.                                 |
| `python pi_app.py --list-cameras`   | Probe camera indices 0–4 and report which one works.                                                   |
| **Diagnostics & repair**            |                                                                                                        |
| `python pi_app.py --diag-imports`   | Run native imports (numpy/cv2/torch/ultralytics) in isolated child procs to pinpoint a crash.          |
| `python pi_app.py --scan-corrupt`   | Report venv files with NUL bytes (cause of `source code string cannot contain null bytes`). Read-only. |
| `python pi_app.py --repair-venv`    | Reinstall the NUL-corrupted packages found above, then verify `import torch`.                          |
| `python pi_app.py --check-ml`       | Show torch/torchvision versions + run the NMS op that triggers `corrupted double-linked list` aborts.  |
| `python pi_app.py --reinstall-ml`   | Reinstall a matched **CPU** torch+torchvision to fix native inference aborts, then re-verify.          |

### `server.py` (laptop)

| Command                       | What it does                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `python server.py`            | Local web app over **HTTPS** on `localhost:8000` (browser owns the camera).                 |
| `python server.py --http`     | Same, plain HTTP on `localhost` (only if your browser doesn't force HTTPS).                 |
| `python server.py --lan`      | **Compute server** for the Pi: plain HTTP on `0.0.0.0:8000`. Fast (no per-frame TLS).       |
| `python server.py --lan-web`  | **Web app over the LAN**: HTTPS on `0.0.0.0:8000` so a phone/Pi browser can use its camera. |
| `python server.py --selftest` | PASS/FAIL check of model / OCR / TTS / Gemini, then exit.                                   |

### Environment variables

| Variable             | Purpose                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `COMPUTE_SERVER_URL` | Laptop compute server URL the Pi offloads to, e.g. `http://192.168.1.50:8000` (http or https). |
| `CAMERA_INDEX`       | Camera index to open (same as `--camera-index`).                                               |
| `REMOTE_ENABLED`     | `0` to disable offload entirely (always run on-device). Default `1`.                           |
| `REMOTE_TIMEOUT`     | Per-call connect+read timeout in seconds for offload. Default `3.0`.                           |
| `VOSK_MODEL_DIR`     | Path to a downloaded Vosk model dir (offline voice control).                                   |
| `GEMINI_API_KEY`     | Enables online **Describe** (otherwise falls back to an offline detection summary).            |

### Common recipes

```bash
# On-device, fully offline (camera on the Pi):
python pi_app.py --camera-index 0

# Pi camera, laptop does detection over the LAN:
python server.py --lan                 # laptop
python pi_app.py --find-server         # Pi (auto-finds the laptop)

# Browser (phone/Pi) owns the camera, laptop does detection:
python server.py --lan-web             # laptop, then open https://<laptop-ip>:8000

# Something broke after a power cut / bad SD card:
python pi_app.py --selftest            # see which stage fails, follow its hint
python pi_app.py --repair-venv         # 'null bytes' import errors
python pi_app.py --reinstall-ml        # 'corrupted double-linked list' inference aborts
```
