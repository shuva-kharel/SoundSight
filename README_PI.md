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
python pi_app.py --list-cameras  # which index your webcam is on (0/1/2…)
python pi_app.py                 # run. Ctrl+C to stop.
```
Camera open is OS-aware (V4L2 + MJPG on the Pi), tries indices 0/1/2, validates a real
frame, warms up, and auto-reopens a yanked/frozen cam. Voice: say **"hey sight …"**.

## What's offline vs online
| Feature | Offline on the Pi? |
|---|---|
| Navigate (detection, tracking, urgency, crossing) | ✅ yes |
| Read (EasyOCR) + label parsing | ✅ yes |
| Money (banknote classifier, tally) | ✅ yes |
| English speech (espeak-ng) + Nepali speech (espeak-ng `-v ne`) | ✅ yes |
| Voice commands (Vosk) | ✅ yes (after the model download) |
| **Describe (Gemini)** | ❌ needs internet + `GEMINI_API_KEY`. Falls back to an
  offline summary of current detections. |
| **Nepali gTTS** (not used — espeak-ng covers Nepali offline) | n/a |

## Performance tips
- The **perf guard** auto-drops `imgsz` 320→256 and disables frame-enhance if FPS falls
  below the floor or the SoC gets hot, then restores when it recovers.
- Add swap if you hit memory pressure during the first EasyOCR/NCNN load:
  `sudo dphys-swapfile swapoff && sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon`
- Keep the Pi cool (heatsink + fan); throttling shows up as low FPS in the log.
