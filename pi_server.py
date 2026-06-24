"""
pi_server.py  --  live dashboard server that runs ON THE PI (FastAPI, port 8080).

Serves a single dashboard page + an MJPEG video stream + a /status WebSocket that
pushes the Pi's live state (mode, sub-mode, FPS, laptop link, spoken log, detections
with distances, traffic light, find pick-list, OCR) at ~5 Hz. Judges open
http://<pi-ip>:8080 on the Pi's own browser OR any laptop on the same network and
watch exactly what the Pi is processing -- live.

ZERO COUPLING with the detection loop: the loop only ever calls DashboardState.update()
/ add_spoken() (cheap, lock-guarded). This server reads that snapshot. The web server
runs in a background thread, so it can never stall Navigate.

  state = DashboardState()
  start(state, port=8080)                 # launches uvicorn in a daemon thread
  state.update(frame, detections, announce, mode=..., laptop=..., ...)  # from the loop

Requires fastapi + uvicorn (added to requirements_pi.txt). pi_app falls back to the
stdlib pi_web.WebPreview if they're not installed.
"""
import asyncio
import logging
import threading
import time
from collections import deque
from pathlib import Path

log = logging.getLogger("soundsight.dashboard")

HERE = Path(__file__).resolve().parent
SPOKEN_HISTORY = 6


class DashboardState:
    """Thread-safe latest-state holder written by the detection loop, read by the web
    server. Holds the annotated JPEG + a status dict + a rolling spoken-line log."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._spoken = deque(maxlen=SPOKEN_HISTORY)
        self._status = {
            "mode": "NAVIGATE", "sub_mode": "street", "fps": 0.0,
            "laptop_link": "disabled", "laptop_ms": None, "ocr_ready": False,
            "detections": [], "traffic_light": None, "find_target": None,
            "pick": [], "last_spoken": [], "timestamp": 0.0,
        }

    # --- written by the detection loop ------------------------------------ #
    def add_spoken(self, text):
        if not text:
            return
        with self._lock:
            self._spoken.append({"text": text, "ts": time.strftime("%H:%M:%S")})

    def update(self, frame=None, detections=None, announce=None, **fields):
        """One cheap call per frame. `announce` (if new) is appended to the spoken log;
        `fields` overwrite status keys (mode, sub_mode, fps, laptop_link, laptop_ms,
        ocr_ready, traffic_light, find_target, pick, target)."""
        jpeg = None
        if frame is not None:
            jpeg = self._encode(frame, detections, fields.get("target"))
        dets = [{"label": d.get("label"), "distance": d.get("distance_m"),
                 "zone": _zone(d.get("cx", 0), (frame.shape[1] if frame is not None else 416)),
                 "urgency": d.get("urgency"), "box": d.get("box")}
                for d in (detections or [])]
        with self._lock:
            if jpeg is not None:
                self._jpeg = jpeg
            if announce:
                self._spoken.append({"text": announce, "ts": time.strftime("%H:%M:%S")})
            self._status["detections"] = dets
            for k in ("mode", "sub_mode", "fps", "laptop_link", "laptop_ms",
                      "ocr_ready", "traffic_light", "find_target", "pick"):
                if k in fields:
                    self._status[k] = fields[k]
            if "pick" in fields:
                self._status["pick"] = [{"letter": p["letter"], "label": p["label"],
                                         "zone": p["zone"], "distance": p["distance_m"]}
                                        for p in (fields["pick"] or [])]
            self._status["timestamp"] = time.time()

    def _encode(self, frame, detections, target):
        try:
            import cv2
            from pi_web import _annotate          # reuse the box/label/distance drawing
            img = _annotate(frame, detections, target)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes() if ok else None
        except Exception as exc:
            log.debug("dashboard encode skipped: %s", exc)
            return None

    # --- read by the web server ------------------------------------------- #
    @property
    def jpeg(self):
        with self._lock:
            return self._jpeg

    def snapshot(self):
        with self._lock:
            s = dict(self._status)
            s["last_spoken"] = list(self._spoken)
            return s


_state = DashboardState()   # module-global the FastAPI app reads (set by start())


def _build_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, StreamingResponse

    app = FastAPI(title="SoundSight Pi Dashboard")

    @app.get("/")
    def index():
        return FileResponse(HERE / "pi_dashboard.html")

    @app.get("/stream")
    def stream():
        async def gen():
            while True:
                jpeg = _state.jpeg
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.07)   # ~14 fps cap
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/status")
    async def status_ws(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(_state.snapshot())
                await asyncio.sleep(0.2)    # push @5 Hz
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log.debug("status ws closed: %s", exc)

    return app


def start(state=None, port=8080, host="0.0.0.0"):
    """Launch the dashboard in a daemon thread. Returns the DashboardState the loop
    should write to. Raises ImportError if fastapi/uvicorn aren't installed (caller
    falls back to the stdlib preview)."""
    global _state
    import uvicorn   # raises ImportError if missing -> caller falls back

    if state is not None:
        _state = state
    app = _build_app()

    def _run():
        try:
            uvicorn.run(app, host=host, port=port, log_level="warning")
        except Exception as exc:
            log.warning("Dashboard server stopped: %s", exc)

    threading.Thread(target=_run, daemon=True).start()
    log.info("Pi dashboard LIVE -> http://<pi-ip>:%d  (open on the Pi or any device on the LAN)", port)
    return _state


def _zone(cx, frame_w):
    if cx < frame_w / 3:
        return "on your left"
    if cx > 2 * frame_w / 3:
        return "on your right"
    return "ahead"
