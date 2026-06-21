#!/usr/bin/env bash
# setup_pi.sh -- one-shot setup for SoundSight on a fresh Raspberry Pi 4 (64-bit OS).
# Creates a venv, installs apt + pip deps, and prints the next steps.
set -e

echo "==================================================================="
echo "  SoundSight Raspberry Pi setup"
echo "==================================================================="

# 0) sanity: 64-bit OS?
if [ "$(getconf LONG_BIT)" != "64" ]; then
  echo "[!] This is a 32-bit OS. SoundSight needs 64-bit Raspberry Pi OS (aarch64)."
  echo "    Re-flash with the 64-bit image, then re-run this script."
  exit 1
fi

# 1) system packages: audio (espeak-ng), OpenCV runtime libs, mic capture (portaudio)
echo "[1/3] Installing apt packages (needs sudo)..."
sudo apt-get update
sudo apt-get install -y \
  espeak-ng alsa-utils libgl1 libglib2.0-0 \
  python3-venv python3-pip portaudio19-dev

# 2) python venv + deps
echo "[2/3] Creating virtual environment (.venv) and installing Python deps..."
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_pi.txt

# 3) done
echo "[3/3] Done."
echo
echo "Next steps:"
echo "  1. Activate the venv:   source .venv/bin/activate"
echo "  2. (faster first run) copy yolo11n_ncnn_model/ from your laptop into this"
echo "     folder. If absent, it auto-exports from yolo11n.pt on first run."
echo "  3. (voice) download a small Vosk model, e.g.:"
echo "       wget https://alphacephality.github.io/vosk/models/vosk-model-small-en-us-0.15.zip"
echo "       unzip vosk-model-small-en-us-0.15.zip -d models_vosk"
echo "     (set VOSK_MODEL_DIR or place it at ./models_vosk/<model> -- see README_PI.md)"
echo "  4. Verify everything:   python pi_app.py --selftest"
echo "  5. Find your camera:    python pi_app.py --list-cameras"
echo "  6. Run:                 python pi_app.py"
