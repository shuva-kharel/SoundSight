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
import ssl
import time
import urllib.request

log = logging.getLogger("soundsight.remote")

# The laptop serves the LAN over HTTPS with a SELF-SIGNED cert (so browsers on the
# LAN get a secure context for the camera). The Pi trusts it explicitly -- it's our
# own laptop on our own LAN -- so verification is disabled for https URLs only.
_UNVERIFIED_SSL = ssl.create_default_context()
_UNVERIFIED_SSL.check_hostname = False
_UNVERIFIED_SSL.verify_mode = ssl.CERT_NONE


def _urlopen(req, timeout):
    """urllib.request.urlopen that accepts the laptop's self-signed cert for https
    URLs (and is a no-op wrapper for plain http)."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    if url.startswith("https"):
        return urllib.request.urlopen(req, timeout=timeout, context=_UNVERIFIED_SSL)
    return urllib.request.urlopen(req, timeout=timeout)

# --- config (env-overridable so you don't edit code at the venue) ----------- #
COMPUTE_SERVER_URL = os.environ.get("COMPUTE_SERVER_URL", "")  # e.g. http://192.168.1.50:8000
REMOTE_ENABLED = os.environ.get("REMOTE_ENABLED", "1") not in ("0", "false", "False")
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "3.0"))   # default connect+read timeout (s)
LIGHT_TIMEOUT = 1.5        # detect / faces: fast, never stall the realtime loop
HEAVY_TIMEOUT = 5.0        # ocr / money / find: bigger models, bigger payloads
DESCRIBE_TIMEOUT = 20.0    # VLM / Gemini caption can legitimately take longer
HEALTH_INTERVAL = 5.0      # re-check the server at most this often (s)
HEALTH_FAIL_LIMIT = 2      # consecutive failed checks before we declare OFFLINE (anti-flap)
MAX_INFLIGHT_HEAVY = 2     # never pile up unbounded heavy work -- drop the request past this
JPEG_QUALITY = 75          # send small JPEGs to keep WiFi payloads fast (latency cut)
MAX_SEND_W = 640           # downscale wide frames before sending (latency cut)


def _encode(frame, q=JPEG_QUALITY, max_w=MAX_SEND_W):
    import cv2

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
        self._fail_count = 0          # consecutive health/call failures (2-strike)
        self._offline_since = None
        self.last_latency_ms = None   # last health round-trip (for the dashboard)
        self._inflight_heavy = 0      # bounded so heavy work can't pile up
        self._announced_fallback = False

    @property
    def link_status(self):
        """'online' | 'connecting' (degraded / not-yet-confirmed) | 'offline' | 'disabled'."""
        if not self.enabled:
            return "disabled"
        if not self.online:
            # never confirmed yet, or 1 strike = still trying; >=2 strikes = offline
            return "offline" if self._fail_count >= HEALTH_FAIL_LIMIT else "connecting"
        return "connecting" if self._fail_count > 0 else "online"

    def _mark_ok(self, latency_ms=None):
        was = self.online
        self.online = True
        self._fail_count = 0
        self._offline_since = None
        if latency_ms is not None:
            self.last_latency_ms = latency_ms
        if not was:
            log.info("Compute server ONLINE: %s (%.0f ms round-trip)", self.url,
                     latency_ms if latency_ms is not None else -1)
        return True

    def _mark_fail(self, reason=""):
        """One strike. Only flip to OFFLINE after HEALTH_FAIL_LIMIT consecutive
        failures, so a single dropped packet doesn't kill offloading (anti-flap)."""
        self._fail_count += 1
        if self._fail_count >= HEALTH_FAIL_LIMIT and self.online:
            self.online = False
            self._offline_since = time.time()
            log.warning("Compute server OFFLINE after %d failed checks (%s) -- using local fallback.",
                        self._fail_count, reason or "timeout")
        return self.online

    # --- health (cached, 2-strike) ---------------------------------------- #
    def health(self, force=False):
        if not self.enabled:
            return False
        now = time.time()
        if not force and (now - self._last_check) < HEALTH_INTERVAL:
            return self.online
        self._last_check = now
        t0 = time.time()
        try:
            with _urlopen(self.url + "/remote/health", timeout=LIGHT_TIMEOUT) as r:
                info = json.loads(r.read())
            if info.get("ok"):
                return self._mark_ok((time.time() - t0) * 1000)
            return self._mark_fail("not ok")
        except Exception as exc:
            return self._mark_fail(str(exc))

    # --- raw calls --------------------------------------------------------- #
    def _post_json(self, path, fields=None, files=None, timeout=None):
        body, ctype = _multipart(fields, files)
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": ctype})
        t0 = time.time()
        with _urlopen(req, timeout=timeout or self.timeout) as r:
            data = json.loads(r.read())
        log.info("remote %s ok in %.0f ms", path, (time.time() - t0) * 1000)
        return data

    def _post_text(self, path, fields=None, files=None, timeout=None):
        body, ctype = _multipart(fields, files)
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": ctype})
        t0 = time.time()
        with _urlopen(req, timeout=timeout or self.timeout) as r:
            txt = r.read().decode("utf-8", "ignore")
        log.info("remote %s ok in %.0f ms", path, (time.time() - t0) * 1000)
        return txt

    def _guard(self, fn, heavy=False):
        """Run a remote call only if online; any error -> None (caller goes local).
        A failed call counts toward the 2-strike OFFLINE machine (not an instant flip).
        Heavy calls are bounded (MAX_INFLIGHT_HEAVY) so they can't pile up unbounded."""
        if not self.health():
            return None
        if heavy:
            if self._inflight_heavy >= MAX_INFLIGHT_HEAVY:
                log.info("remote heavy call dropped (queue full, %d in flight)", self._inflight_heavy)
                return None
            self._inflight_heavy += 1
        try:
            result = fn()
            self._fail_count = 0          # a good call clears strikes
            return result
        except Exception as exc:
            log.info("remote call failed (%s) -> falling back to local", exc)
            self._mark_fail(str(exc))
            return None
        finally:
            if heavy:
                self._inflight_heavy = max(0, self._inflight_heavy - 1)

    # --- features (None return => use the Pi's local model) ---------------- #
    # heavy=True calls are bounded (drop past MAX_INFLIGHT_HEAVY); timeouts are
    # per-endpoint so a slow link never stalls the realtime Navigate loop.
    def ocr(self, frame, label=False):
        return self._guard(lambda: self._post_json(
            "/ocr", {"label": "true" if label else "false"},
            [("file", "f.jpg", _encode(frame))], timeout=HEAVY_TIMEOUT), heavy=True)

    def money(self, frames):
        files = [("files", f"f{i}.jpg", _encode(f)) for i, f in enumerate(frames)]
        return self._guard(lambda: self._post_json("/money", {}, files, timeout=HEAVY_TIMEOUT), heavy=True)

    def money_count(self, frames):
        files = [("files", f"c{i}.jpg", _encode(f)) for i, f in enumerate(frames)]
        return self._guard(lambda: self._post_json("/money/count", {}, files, timeout=HEAVY_TIMEOUT), heavy=True)

    def faces(self, frame):
        return self._guard(lambda: self._post_json(
            "/faces/who", {}, [("file", "f.jpg", _encode(frame))], timeout=LIGHT_TIMEOUT))

    def find(self, name, frame):
        """Locate a PERSONAL object on the laptop (CLIP/feature match). None -> Pi can't."""
        return self._guard(lambda: self._post_json(
            "/remote/find", {"name": name}, [("file", "f.jpg", _encode(frame))],
            timeout=HEAVY_TIMEOUT), heavy=True)

    def describe(self, frame, lang="en"):
        return self._guard(lambda: self._post_text(
            "/describe", {"lang": lang}, [("file", "f.jpg", _encode(frame))],
            timeout=DESCRIBE_TIMEOUT), heavy=True)

    def detect(self, frame):
        # realtime path: short timeout so a slow frame is dropped, not queued.
        return self._guard(lambda: self._post_json(
            "/remote/detect", {}, [("file", "f.jpg", _encode(frame))], timeout=LIGHT_TIMEOUT))

    def note_fallback_once(self, speak):
        """Speak a subtle notice the FIRST time we fall back to on-device mode."""
        if not self._announced_fallback:
            self._announced_fallback = True
            speak("Using on-device mode")


def _probe(ip, port, timeout):
    # Prefer HTTPS (what the laptop serves by default in --lan mode), fall back to
    # plain HTTP for a server started with `--lan --http`.
    for scheme in ("https", "http"):
        url = f"{scheme}://{ip}:{port}"
        try:
            with _urlopen(url + "/remote/health", timeout=timeout) as r:
                if json.loads(r.read()).get("ok"):
                    return url
        except Exception:
            continue
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
