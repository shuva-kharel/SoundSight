"""
pi_web.py  --  tiny live preview server for the headless Pi app.

`pi_app.py` runs on-device detection (and offloads heavy work to the laptop), but it
has no screen. For a DEMO you want to SEE the Pi's own camera + what it's detecting
from another device. This serves exactly that: an MJPEG video stream of the Pi's
annotated frames plus a small status panel, at http://<pi-ip>:8080.

Why MJPEG and not the browser camera (getUserMedia)? Because the frames come FROM
the Pi here -- the page just shows an <img src="/stream">. That needs no HTTPS /
secure context, so plain http works on any phone/laptop on the LAN.

  WebPreview(port).start()                 # background HTTP server (daemon thread)
  WebPreview.update(frame, detections, announce)   # call each loop iteration

Stdlib + cv2 only (cv2 is already a Pi dependency). Never blocks the capture loop.
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("soundsight.web")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0f14; color:#e6edf3; font:14px/1.4 system-ui,sans-serif; }
  header { padding:10px 14px; background:#11161d; border-bottom:1px solid #222b36;
           display:flex; align-items:center; gap:10px; }
  header b { font-size:16px; } header span { color:#7d8a99; }
  .wrap { display:flex; flex-wrap:wrap; gap:14px; padding:14px; }
  .video { flex:1 1 480px; min-width:320px; }
  .video img { width:100%; border-radius:10px; background:#000; display:block; }
  .panel { flex:0 1 280px; min-width:220px; background:#11161d; border:1px solid #222b36;
           border-radius:10px; padding:12px; height:max-content; }
  .panel h3 { margin:0 0 8px; font-size:13px; color:#7d8a99; text-transform:uppercase; letter-spacing:.04em; }
  .say { font-size:18px; font-weight:600; min-height:1.4em; }
  ul { list-style:none; margin:8px 0 0; padding:0; }
  li { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #1b2230; }
  .u { color:#ff6b6b; font-weight:600; } .n { color:#ffd166; } .f { color:#8aa0b5; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#3fb950; margin-right:6px; }
</style></head><body>
<header><span class="dot"></span><b>__TITLE__</b><span>live on-device preview</span></header>
<div class="wrap">
  <div class="video"><img src="/stream" alt="Pi camera stream"></div>
  <div class="panel">
    <h3>Now saying</h3><div class="say" id="say">…</div>
    <h3 style="margin-top:14px">Detections</h3><ul id="dets"></ul>
  </div>
</div>
<script>
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('say').textContent = s.announce || '—';
    const ul = document.getElementById('dets');
    ul.innerHTML = (s.dets||[]).map(d=>{
      const cls = d.urgency==='very close'?'u':(d.close==='near'?'n':'f');
      const tag = d.urgency || d.close || '';
      return `<li><span>${d.label}</span><span class="${cls}">${tag}</span></li>`;
    }).join('') || '<li><span class="f">nothing in view</span></li>';
  }catch(e){}
}
setInterval(tick, 500); tick();
</script></body></html>"""


def _closeness(det):
    if det.get("urgency"):
        return det["urgency"]
    area = float(det.get("area_ratio") or 0.0)
    if area > 0.20:
        return "very close"
    if area >= 0.05:
        return "near"
    return "far"


def _annotate(frame, detections):
    """Draw boxes + labels on a copy of the frame. Red = very close, amber = near."""
    import cv2

    img = frame.copy()
    for d in detections or []:
        box = d.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        close = _closeness(d)
        color = (60, 60, 255) if close == "very close" else (40, 200, 230) if close == "near" else (80, 200, 80)
        label = str(d.get("label") or "object")
        conf = d.get("confidence")
        txt = f"{label} {conf:.2f}" if isinstance(conf, (int, float)) else label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        ytxt = max(14, y1 - 6)
        cv2.putText(img, txt, (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


class WebPreview:
    """Serves an MJPEG stream (/stream), a status JSON (/status) and a viewer page (/).
    Thread-safe: update() is called from the capture loop, HTTP handlers read the
    latest frame under a lock. Failures here never propagate to the capture loop."""

    def __init__(self, port=8080, title="SoundSight Pi", quality=70):
        self.port = port
        self.title = title
        self.quality = quality
        self._lock = threading.Lock()
        self._jpeg = None
        self._status = {"announce": "", "dets": []}
        self._httpd = None

    def update(self, frame, detections=None, announce=None):
        try:
            import cv2

            img = _annotate(frame, detections)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                return
            dets = [{"label": d.get("label"), "urgency": d.get("urgency"),
                     "close": _closeness(d)} for d in (detections or [])]
            with self._lock:
                self._jpeg = buf.tobytes()
                self._status["dets"] = dets
                if announce:
                    self._status["announce"] = announce
        except Exception as exc:   # a preview hiccup must never kill the loop
            log.debug("web preview update skipped: %s", exc)

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _headers(self, ctype, body_len=None):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                if body_len is not None:
                    self.send_header("Content-Length", str(body_len))
                self.end_headers()

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.replace("__TITLE__", preview.title).encode()
                    self._headers("text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                elif self.path == "/status":
                    with preview._lock:
                        body = json.dumps(preview._status).encode()
                    self._headers("application/json", len(body))
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, private")
                self.end_headers()
                try:
                    while True:
                        with preview._lock:
                            jpeg = preview._jpeg
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpeg)).encode() +
                                             b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.07)   # ~14 fps cap; loop sets the real rate
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return   # viewer closed the tab -- fine

        try:
            self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as exc:
            log.warning("Web preview OFF -- port %d unavailable (%s).", self.port, exc)
            return None
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        log.info("Pi web preview LIVE -> open  http://<pi-ip>:%d  on any device on the LAN", self.port)
        return self

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
