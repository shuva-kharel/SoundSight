"""
server.py  --  FastAPI web layer for SoundSight.

Owns transport only. All AI lives in vision_core (so it ports to the Pi):
  * GET  /              -> serves index.html (the whole UI)
  * WS   /ws/navigate   -> base64 JPEG frames in, JSON detections out
  * POST /ocr           -> one JPEG in, {text, lang} out (Nepali + English)
  * POST /tts           -> text in, MP3 audio out (Nepali speech via gTTS)
  * POST /describe      -> one JPEG in, scene description out (Gemini)

The browser owns the webcam (getUserMedia) and the speech (Web Speech API for
English; it plays the /tts MP3 for Nepali).
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
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, Response

from vision_core import (
    Announcer,
    ApproachDetector,
    SceneDescriber,
    VisionCore,
    select_announcements,
)

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

# Load YOLO once at startup (logs which device it uses).
vision = VisionCore()

# Describe mode (Gemini). Cheap to construct; the API client is built lazily on
# the first /describe call (and only if a key is set).
describer = SceneDescriber()

# EasyOCR is heavy (downloads models on first use), so we build it lazily on the
# first /ocr call instead of blocking server startup. Navigate works instantly.
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # imported here so startup stays fast

        log.info("Initializing EasyOCR for Nepali + English (first use downloads the Devanagari model)...")
        # 'ne' = Nepali (Devanagari script), 'en' = English. EasyOCR lets English
        # ride along with the Devanagari model, so one reader handles both.
        _ocr_reader = easyocr.Reader(["ne", "en"])
        log.info("EasyOCR ready.")
    return _ocr_reader


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
    announcer = Announcer()       # one cooldown state per connection
    approach = ApproachDetector()  # one approach-history state per connection
    frames, t0 = 0, time.time()
    log.info("Navigate client connected.")

    try:
        while True:
            text = await ws.receive_text()
            frame = decode_b64_frame(text)
            if frame is None:
                continue

            detections = vision.detect(frame)
            approaching = approach.approaching_objects(detections)
            speak = select_announcements(detections, announcer, approaching, time.time())

            await ws.send_json(
                {
                    "boxes": [
                        {
                            "label": d["label"],
                            "confidence": d["confidence"],
                            "box": d["box"],
                            "urgency": d["urgency"],
                        }
                        for d in detections
                    ],
                    "speak": speak,
                    "fw": int(frame.shape[1]),
                    "fh": int(frame.shape[0]),
                }
            )

            # FPS logging every ~30 frames.
            frames += 1
            if frames % 30 == 0:
                fps = frames / (time.time() - t0)
                log.info("Navigate FPS: %.1f  (last frame: %d detections)", fps, len(detections))
    except WebSocketDisconnect:
        log.info("Navigate client disconnected after %d frames.", frames)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """Run bilingual (Nepali + English) OCR, returning {text, lang}."""
    frame = decode_jpeg(await file.read())
    if frame is None:
        return {"text": "", "lang": "en"}
    reader = get_ocr_reader()
    t0 = time.time()
    # readtext blocks; keep it off the event loop.
    lines = await run_in_threadpool(lambda: reader.readtext(frame, detail=0))
    text = " ".join(lines).strip()
    log.info("OCR: %d chars in %.2fs", len(text), time.time() - t0)
    if not text:
        return {"text": "No text found.", "lang": "en"}
    return {"text": text, "lang": detect_lang(text)}


def _synth_mp3(text: str, lang: str) -> bytes:
    """Synthesize speech to MP3 bytes with gTTS.

    NOTE: gTTS calls Google's online TTS service, so it REQUIRES INTERNET.
    PI / OFFLINE SWAP POINT: replace this body with an offline Nepali TTS engine
    (e.g. espeak-ng with a Nepali voice, or a Piper/Coqui Nepali model). Keep the
    contract -- (text, lang) in, MP3 bytes out -- and /tts and the browser are
    unchanged.
    """
    import io

    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("ne")):
    """Synthesize `text` to an MP3 and stream it back (used for Nepali speech)."""
    text = (text or "").strip()
    if not text:
        return PlainTextResponse("", status_code=400)
    try:
        t0 = time.time()
        audio = await run_in_threadpool(_synth_mp3, text, lang)
        log.info("TTS: %d chars (lang=%s) -> %d bytes in %.2fs", len(text), lang, len(audio), time.time() - t0)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as exc:  # offline / unsupported lang -> let the browser fall back
        log.warning("TTS failed (needs internet?): %s", exc)
        return PlainTextResponse(f"TTS failed: {exc}", status_code=503)


@app.post("/describe", response_class=PlainTextResponse)
async def describe(file: UploadFile = File(...)):
    frame = decode_jpeg(await file.read())
    if frame is None:
        return ""
    t0 = time.time()
    # The Gemini call is blocking network I/O -- run it off the event loop so it
    # doesn't stall the Navigate WebSocket.
    text = await run_in_threadpool(describer.describe, frame)
    log.info("Describe: %d chars in %.2fs", len(text), time.time() - t0)
    return text


CERT_FILE = HERE / "cert.pem"
KEY_FILE = HERE / "key.pem"


def ensure_self_signed_cert():
    """
    Create a self-signed cert for localhost if one doesn't exist yet.

    Why HTTPS at all? getUserMedia works on plain http://localhost, BUT modern
    browsers (Chrome HTTPS-First, HSTS, etc.) often auto-upgrade localhost to
    https://. When that hits a plain-HTTP server you get ERR_SSL_PROTOCOL_ERROR.
    Serving real HTTPS sidesteps the browser entirely. The cert is self-signed,
    so you'll click through a one-time "your connection is not private" warning.
    """
    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    log.info("Generating self-signed certificate (cert.pem / key.pem)...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
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


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SoundSight server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve plain HTTP instead of HTTPS (only if your browser doesn't force HTTPS)",
    )
    args = parser.parse_args()

    # Bind to localhost so the browser treats it as a secure context (getUserMedia).
    if args.http:
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
