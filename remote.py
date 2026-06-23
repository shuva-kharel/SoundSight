"""
remote.py  --  Distributed-compute client: the Pi offloads heavy, ON-DEMAND AI to the
laptop's compute server over the LAN (phone/laptop hotspot). Safety stays local.

The Pi's real-time loop (Navigate / distance) NEVER uses this -- only on-demand
features (Read / Money / Faces / Describe). If the laptop is unreachable or a call
times out, the caller falls back to the Pi's own local models; this module just
returns None on any failure so the app never hangs or crashes.

Stdlib only (urllib + socket) so it adds no deps on the Pi.

  RemoteCompute(url).health()                  -> bool (cached, periodic recheck)
  .ocr(frame[,label]) / .money(frames) / .money_count(frames) / .faces(frame)
  .describe(frame,lang) / .detect(frame)       -> parsed result, or None -> use local
  find_server(port)                            -> auto-discover the server on the subnet
"""

import concurrent.futures
import json
import logging
import os
import socket
import time
import urllib.request

import cv2

log = logging.getLogger("soundsight.remote")

# --- config (env-overridable so you don't edit code at the venue) ----------- #
COMPUTE_SERVER_URL = os.environ.get("COMPUTE_SERVER_URL", "")  # e.g. http://192.168.1.50:8000
REMOTE_ENABLED = os.environ.get("REMOTE_ENABLED", "1") not in ("0", "false", "False")
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "3.0"))   # connect+read timeout (s)
HEALTH_INTERVAL = 15.0     # re-check the server at most this often
JPEG_QUALITY = 80          # send small JPEGs to keep WiFi payloads fast
MAX_SEND_W = 960           # downscale wide frames before sending


def _encode(frame, q=JPEG_QUALITY, max_w=MAX_SEND_W):
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _multipart(fields, files):
    """Build a multipart/form-data body (stdlib, no requests)."""
    boundary = "----soundsight%d" % int(time.time() * 1e6)
    out = b""
    for k, v in (fields or {}).items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    for name, fname, data in (files or []):
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{fname}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return out, f"multipart/form-data; boundary={boundary}"


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


class RemoteCompute:
    def __init__(self, url=None, enabled=REMOTE_ENABLED, timeout=REMOTE_TIMEOUT):
        self.url = (url or COMPUTE_SERVER_URL or "").rstrip("/")
        self.enabled = bool(enabled and self.url)
        self.timeout = timeout
        self.online = False
        self._last_check = 0.0
        self._announced_fallback = False

    # --- health (cached) --------------------------------------------------- #
    def health(self, force=False):
        if not self.enabled:
            return False
        now = time.time()
        if not force and (now - self._last_check) < HEALTH_INTERVAL:
            return self.online
        self._last_check = now
        try:
            with urllib.request.urlopen(self.url + "/remote/health", timeout=1.5) as r:
                info = json.loads(r.read())
            was = self.online
            self.online = bool(info.get("ok"))
            if self.online and not was:
                log.info("Compute server ONLINE: %s (profile=%s, detect=%s)",
                         self.url, info.get("profile"), (info.get("models") or {}).get("detect"))
            return self.online
        except Exception:
            self.online = False
            return False

    # --- raw calls --------------------------------------------------------- #
    def _post_json(self, path, fields=None, files=None, timeout=None):
        body, ctype = _multipart(fields, files)
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": ctype})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            data = json.loads(r.read())
        log.info("remote %s ok in %.0f ms", path, (time.time() - t0) * 1000)
        return data

    def _post_text(self, path, fields=None, files=None, timeout=None):
        body, ctype = _multipart(fields, files)
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": ctype})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            txt = r.read().decode("utf-8", "ignore")
        log.info("remote %s ok in %.0f ms", path, (time.time() - t0) * 1000)
        return txt

    def _guard(self, fn):
        """Run a remote call only if online; any error -> None (caller goes local)."""
        if not self.health():
            return None
        try:
            return fn()
        except Exception as exc:
            log.info("remote call failed (%s) -> falling back to local", exc)
            self.online = False
            return None

    # --- features (None return => use the Pi's local model) ---------------- #
    def ocr(self, frame, label=False):
        return self._guard(lambda: self._post_json(
            "/ocr", {"label": "true" if label else "false"}, [("file", "f.jpg", _encode(frame))]))

    def money(self, frames):
        files = [("files", f"f{i}.jpg", _encode(f)) for i, f in enumerate(frames)]
        return self._guard(lambda: self._post_json("/money", {}, files, timeout=max(self.timeout, 6)))

    def money_count(self, frames):
        files = [("files", f"c{i}.jpg", _encode(f)) for i, f in enumerate(frames)]
        return self._guard(lambda: self._post_json("/money/count", {}, files, timeout=max(self.timeout, 8)))

    def faces(self, frame):
        return self._guard(lambda: self._post_json("/faces/who", {}, [("file", "f.jpg", _encode(frame))]))

    def describe(self, frame, lang="en"):
        return self._guard(lambda: self._post_text(
            "/describe", {"lang": lang}, [("file", "f.jpg", _encode(frame))], timeout=20))

    def detect(self, frame):
        return self._guard(lambda: self._post_json(
            "/remote/detect", {}, [("file", "f.jpg", _encode(frame))], timeout=max(self.timeout, 6)))

    def note_fallback_once(self, speak):
        """Speak a subtle notice the FIRST time we fall back to on-device mode."""
        if not self._announced_fallback:
            self._announced_fallback = True
            speak("Using on-device mode")


def _probe(ip, port, timeout):
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/remote/health", timeout=timeout) as r:
            if json.loads(r.read()).get("ok"):
                return f"http://{ip}:{port}"
    except Exception:
        pass
    return None


def find_server(port=8000, timeout=0.4):
    """Scan the local /24 subnet for the compute server. Returns its URL or None."""
    base = local_ip()
    if not base:
        return None
    prefix = base.rsplit(".", 1)[0]
    log.info("Scanning %s.0/24 for a compute server on port %d ...", prefix, port)
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        futs = {ex.submit(_probe, f"{prefix}.{i}", port, timeout): i for i in range(1, 255)}
        for fut in concurrent.futures.as_completed(futs):
            url = fut.result()
            if url:
                log.info("Found compute server: %s", url)
                return url
    log.warning("No compute server found on %s.0/24:%d", prefix, port)
    return None
