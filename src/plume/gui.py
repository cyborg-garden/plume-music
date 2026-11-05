"""``plume gui``: a local page for starting takes and reading them.

The page opens no device and starts nothing until a person clicks Start.  This
process never opens an audio device itself: Start runs ``plume lead`` as a child, and
everything about that child lives in :mod:`plume.guitake`.  The other way to hear
anything is a person clicking Listen, which makes their own browser fetch a WAV from
here.

HTTP lets any website try to reach this, unlike plume's unix socket, and the WAVs
are recordings of a room.  A DNS-rebinding page could find the random port and
read them.  So, on every route, before anything else:

- bind 127.0.0.1 only, on a random port;
- ``Host`` must match exactly, which is what stops DNS rebinding;
- ``Origin``, if sent, must match exactly, and every POST must send it;
- ``Sec-Fetch-Site``, if sent, must be same-origin.  Only the token link and the
  page itself may also be ``none``, which is how a terminal opens them;
- no CORS headers, ever;
- a one-time token in the first URL becomes an HttpOnly, SameSite=Strict cookie
  named for the port, and a redirect strips the token from the address bar;
- cookies ignore ports, so every other server on 127.0.0.1 is sent that cookie.
  It therefore proves nothing by itself.  Every ``/api`` route also needs
  ``X-Plume-Token``, a separate value the redirect hands over in the URL fragment,
  which no server ever receives, and the page keeps in sessionStorage, which is
  per origin.  Only ``<audio>``, which cannot set a header, sends it as ``?k=``;
- a POST also needs a JSON content type;
- take IDs come from the server's own listing.  No client ever names a path;
- the request path holds the token, so the stdlib's request logging is replaced:
  method, route template and status, nothing else.

``plume gui url`` prints a fresh link, so losing the cookie cannot lock anyone out.
It needs no running request: a token is a nonce signed with a key kept in a 0600
file, so whoever can read that file, and nobody else, can mint one.

Standard library only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
import sys
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import soundfile as sf

from . import guitake, record
from .realtime import REC_HUMAN, REC_MACHINE, list_sessions

STATE_DIR_ENV = "PLUME_GUI_STATE_DIR"

GET_ROUTES = ("/", "/auth", "/static/{name}", "/api/session", "/api/takes",
              "/api/take/{id}", "/api/take/{id}/audio/{track}", "/api/home", "/api/live", "/api/setup")
POST_OPS = frozenset({"keep", "rename", "start", "stop", "cue", "setup/listen", "setup/answer",
                      "setup/cancel", "setup/reset"})

_STATIC_DIR = Path(__file__).with_name("gui_static")
#: Every file the page may load, and its type.  Nothing else under the folder is served.
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "plume.css": "text/css; charset=utf-8",
    "theme.js": "text/javascript; charset=utf-8",
    "stone.js": "text/javascript; charset=utf-8",
    "layout.js": "text/javascript; charset=utf-8",
    "session.js": "text/javascript; charset=utf-8",
    "live.js": "text/javascript; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "fonts/schibsted-grotesk-latin-400-normal.woff2": "font/woff2",
    "fonts/schibsted-grotesk-latin-500-normal.woff2": "font/woff2",
    "fonts/dm-mono-latin-400-normal.woff2": "font/woff2",
}
_TRACKS = {"human": REC_HUMAN, "machine": REC_MACHINE}
TOGETHER = "together"
# What leaves room for both when they peak at once, and is only clipped if they really do.
_TOGETHER_GAIN = 0.7
_TOGETHER_FRAMES = 65536
_WAV_HEADER = 44

_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
        "connect-src 'self'; media-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'")

_MAX_BODY_BYTES = 4096
_MAX_DRAIN_BYTES = 1024 * 1024
_REQUEST_TIMEOUT_S = 30
_CHUNK_BYTES = 256 * 1024
_ID_RE = re.compile(r"[0-9a-f]{16}")
_TAKE_RE = re.compile(r"/api/take/([^/]*)")
_AUDIO_RE = re.compile(r"/api/take/([^/]*)/audio/([^/]*)")
_TOKEN_RE = re.compile(r"([0-9a-f]{32})\.([0-9a-f]{64})")
_RANGE_RE = re.compile(r"bytes=(\d{0,18})-(\d{0,18})")


class GuiError(RuntimeError):
    """A clean, user-facing failure."""


def platform_problem() -> str | None:
    """Its sockets and its private files need these, and Windows has neither."""
    if not hasattr(socket, "AF_UNIX") or not hasattr(os, "O_NOFOLLOW"):
        return "plume gui needs macOS or Linux, and this computer is neither."
    return None


# ------------------------------------------------------------------ the state file

def state_path() -> Path:
    """The newest bridge.  Each one also has a file of its own: see :func:`fresh_url`."""
    env = os.environ.get(STATE_DIR_ENV)
    base = Path(env) if env else Path.home() / ".plume"
    return base / "gui.json"


def _own_state_path(port: int) -> Path:
    return state_path().with_name(f"gui-{int(port)}.json")


def _write_state(doc: dict, path: Path | None = None) -> None:
    path = state_path() if path is None else path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # O_NOFOLLOW: a planted symlink must not redirect the key somewhere readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, json.dumps(doc).encode())
    finally:
        os.close(fd)


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _sign(key: bytes, nonce: str) -> str:
    return hmac.new(key, nonce.encode(), hashlib.sha256).hexdigest()


def _mint(key: bytes) -> str:
    nonce = secrets.token_hex(16)
    return f"{nonce}.{_sign(key, nonce)}"


def _read_state(path: Path) -> tuple[bytes, int, object] | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        with os.fdopen(fd) as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None
            doc = json.load(f)
        return bytes.fromhex(doc["key"]), int(doc["port"]), doc.get("pid")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _port_answers(port: int) -> bool:
    """Is anything listening there?  Connects and leaves, and sends nothing."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except (OSError, ValueError, OverflowError):
        return False


def fresh_url() -> str:
    """A new one-time link to a running bridge, from a 0600 file alone.

    The newest bridge wins.  If that one has gone, any other still running is a way
    back in, because a bridge that is alive must never read as "none is running".
    """
    latest = state_path()
    try:
        others = sorted(latest.parent.glob("gui-*.json"),
                        key=lambda p: p.lstat().st_mtime_ns, reverse=True)
    except OSError:
        others = []
    for path in (latest, *others):
        state = _read_state(path)
        # The pid alone proves little: a file left by a kill can name a pid reused since.
        if state is not None and _pid_alive(state[2]) and _port_answers(state[1]):
            return f"http://127.0.0.1:{state[1]}/auth?token={_mint(state[0])}"
    raise GuiError("no plume gui is running. Start one with 'plume gui'.")


def _open_folder(path: Path) -> int:
    """A handle on the folder itself, so a later swap of its name changes nothing."""
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)


def _still_real(path: Path) -> bool:
    """True if this is a folder now, with no symlink anywhere on the way to it."""
    try:
        return path.resolve() == path and stat.S_ISDIR(os.lstat(path).st_mode)
    except (OSError, RuntimeError):
        return False


def _live_folders() -> set[str]:
    """Folders a running session is recording into, from the registry."""
    out = set()
    for session in list_sessions(prune=False):
        if session.record_dir:
            try:
                out.add(os.path.realpath(session.record_dir))
            except (OSError, ValueError):
                pass
    return out


# --------------------------------------------------------------------- the library

class Library:
    """Takes, known by IDs this server gives out.  Paths never come from a client."""

    def __init__(self, roots=(), pinned=None):
        self.roots = [Path(r) for r in roots]
        self.pinned = None if pinned is None else Path(pinned).resolve()
        self._lock = threading.Lock()
        #: Held across disk work, so it is not the lock every reader needs.
        self._keep_lock = threading.Lock()
        self._paths: dict[str, Path] = {}
        self._cache: dict[str, tuple] = {}

    @staticmethod
    def _id_of(path: Path) -> str:
        # fsencode: a folder name need not be valid UTF-8 on Linux.
        return hashlib.sha256(os.fsencode(path)).hexdigest()[:16]

    def _scan(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if self.pinned is not None and _still_real(self.pinned) and record.is_take_dir(self.pinned):
            found[self._id_of(self.pinned)] = self.pinned
        for root in self.roots:
            try:
                children = sorted(root.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    # A symlink could point anywhere, so it is not part of the library.
                    if child.is_symlink() or not child.is_dir() or not record.is_take_dir(child):
                        continue
                    resolved = child.resolve()
                except OSError:
                    continue
                found[self._id_of(resolved)] = resolved
        with self._lock:
            self._paths = found
        return found

    def path_of(self, take_id: str) -> Path | None:
        if not _ID_RE.fullmatch(take_id):
            return None
        with self._lock:
            path = self._paths.get(take_id)
        # A folder can become a symlink after it was listed, so a remembered path
        # proves nothing about today.
        if path is None or not _still_real(path):
            path = self._scan().get(take_id)
        return path

    def pinned_id(self) -> str | None:
        return None if self.pinned is None else self._id_of(self.pinned)

    def id_of_folder(self, folder: Path) -> str | None:
        """The ID of a take this library can see, or None."""
        take_id = self._id_of(Path(folder).resolve())
        return take_id if self.path_of(take_id) is not None else None

    @staticmethod
    def _signature(path: Path) -> tuple:
        out = []
        for name in (record.REC_MANIFEST, record.REC_DECISIONS, record.REC_ENDING, record.REC_CUES,
                     record.REC_KEEP, REC_HUMAN, REC_MACHINE):
            try:
                st = os.lstat(path / name)
                out.append((st.st_mtime_ns, st.st_size))
            except OSError:
                out.append(None)
        return tuple(out)

    def take(self, take_id: str, live: set[str] | None = None) -> dict | None:
        path = self.path_of(take_id)
        if path is None:
            return None
        is_live = str(path) in (_live_folders() if live is None else live)
        signature = (self._signature(path), is_live)
        with self._lock:
            cached = self._cache.get(take_id)
        if cached and cached[0] == signature:
            return cached[1]
        fallback = int(take_id[:8], 16)
        doc = record.read_take(path, fallback_seed=fallback, live=is_live)
        doc["id"] = take_id
        home = str(Path.home())
        where = str(path)
        doc["where"] = "~" + where[len(home):] if where.startswith(home + os.sep) else where
        with self._lock:
            self._cache[take_id] = (signature, doc)
        return doc

    def shelf(self) -> list[dict]:
        out = []
        live = _live_folders()
        for take_id in self._scan():
            doc = self.take(take_id, live)
            if doc is not None:
                out.append({"id": take_id, **record.summary(doc)})
        out.sort(key=lambda t: (t["started_at"] or 0.0, t["folder"]), reverse=True)
        return out

    def write_keep(self, take_id: str, *, kept=None, name=None) -> dict | None:
        path = self.path_of(take_id)
        if path is None:
            return None
        with self._keep_lock:
            doc = record.read_keep(path)
            if kept is not None:
                doc["kept"] = kept
            if name is not None:
                doc["name"] = record.clean_name(name)
            folder = _open_folder(path)
            # A name nobody could have planted, and O_EXCL in case they guess it:
            # a symlink waiting here must not aim this write at another file.
            tmp = f".{record.REC_KEEP}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                             dir_fd=folder)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(json.dumps(doc))
                    os.replace(tmp, record.REC_KEEP, src_dir_fd=folder, dst_dir_fd=folder)
                except OSError:
                    try:
                        os.unlink(tmp, dir_fd=folder)
                    except OSError:
                        pass
                    raise
            finally:
                os.close(folder)
        return doc


# ---------------------------------------------------------------------- the handler

class _Refused(Exception):
    def __init__(self, status: int, message: str, *, extra: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra or {}


class _Handler(BaseHTTPRequestHandler):
    server_version = "plume"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # The stdlib's default, HTTP/0.9, answers errors with no status line or headers.
    default_request_version = "HTTP/1.1"
    timeout = _REQUEST_TIMEOUT_S
    #: set per request, for the log.  Never the real path.
    route = "?"

    @property
    def gui(self) -> "GuiServer":
        return self.server.gui        # type: ignore[attr-defined]

    # -- logging: the stdlib writes the request path, and the path holds the token

    def log_message(self, format, *args):  # noqa: A002, ARG002
        pass

    def log_error(self, format, *args):  # noqa: A002, ARG002
        pass

    def log_request(self, code="-", size="-"):  # noqa: ARG002
        pass

    def _log(self, status: int) -> None:
        method = self.command if self.command and self.command.isalpha() else "?"
        self.gui.log(f"{method.upper() if len(method) <= 10 else '?'} {self.route} {status}")

    # -- responses

    def _send(self, status: int, body: bytes = b"", ctype: str | None = None,
              headers: dict | None = None) -> None:
        self.send_response(status)
        self._security_headers()
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)
        self._log(status)

    def _security_headers(self) -> None:
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _json(self, doc, status: int = 200) -> None:
        body = json.dumps(doc, allow_nan=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _refuse(self, refused: _Refused) -> None:
        # A refused POST's body may be unread, and must not parse as a next request.
        self.close_connection = True
        body = json.dumps({"error": refused.message}).encode()
        self._send(refused.status, body, "application/json; charset=utf-8",
                   {"Connection": "close", **refused.extra})

    def send_error(self, code, message=None, explain=None):  # noqa: ARG002
        """Used by the stdlib for malformed requests.  Echoes nothing it was sent."""
        self.close_connection = True
        self.route = "?"
        try:
            self._send(int(code), json.dumps({"error": "bad request"}).encode(),
                       "application/json; charset=utf-8", {"Connection": "close"})
        except OSError:
            pass

    # -- the checks, in order, before any route

    def _check_common(self, *, allow_none: bool = False) -> None:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0] != self.gui.host:
            raise _Refused(403, "wrong host")
        origins = self.headers.get_all("Origin") or []
        if len(origins) > 1 or (origins and origins[0] != self.gui.origin):
            raise _Refused(403, "wrong origin")
        sites = self.headers.get_all("Sec-Fetch-Site") or []
        allowed = ("same-origin", "none") if allow_none else ("same-origin",)
        if len(sites) > 1 or (sites and sites[0] not in allowed):
            raise _Refused(403, "cross-site request")

    def _session(self) -> str | None:
        """The CSRF value of the session this request's cookie proves, if any.

        Every cookie of this name counts.  Any other server on this address can plant
        one with a narrower path, and the browser then sends that one first.
        """
        cookies = self.headers.get_all("Cookie") or []
        wanted = f"{self.gui.cookie_name}="
        for header in cookies:
            for part in header.split(";"):
                part = part.strip()
                if part.startswith(wanted):
                    csrf = self.gui.session_csrf(part[len(wanted):])
                    if csrf is not None:
                        return csrf
        return None

    def _require_session(self) -> str:
        csrf = self._session()
        if csrf is None:
            raise _Refused(401, "no session. Run 'plume gui url' for a new link.")
        return csrf

    def _require_page_key(self, csrf: str, *, query: str | None = None) -> None:
        """The cookie reaches every server on this address.  This value does not."""
        sent = self.headers.get("X-Plume-Token")
        if sent is None and query is not None:
            values = parse_qs(query, keep_blank_values=True).get("k", [])
            sent = values[0] if len(values) == 1 else None
        if sent is None or not hmac.compare_digest(sent.encode(), csrf.encode()):
            raise _Refused(401, "no session. Run 'plume gui url' for a new link.")

    # -- routing

    def _dispatch(self, handler) -> None:
        try:
            handler()
        except _Refused as refused:
            self._refuse(refused)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except OSError:
            self.close_connection = True
        except Exception:  # noqa: BLE001 - one bad take must not drop the connection bare
            self._refuse(_Refused(500, "that did not work"))

    def do_GET(self):  # noqa: N802
        self._dispatch(self._get)

    def do_POST(self):  # noqa: N802
        self._dispatch(self._post)

    def _not_allowed(self):
        self.route = "?"
        self._check_host_only_for_refusal()
        raise _Refused(405, "method not allowed", extra={"Allow": "GET, POST"})

    def _check_host_only_for_refusal(self) -> None:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0] != self.gui.host:
            raise _Refused(403, "wrong host")

    def do_HEAD(self):  # noqa: N802
        self._dispatch(self._not_allowed)

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = do_BREW = do_HEAD  # noqa: N815

    def parse_request(self):
        ok = super().parse_request()
        # "GET /path" with no version gets no headers in the old protocol, so it is
        # refused here, by the same path as any other malformed request.
        if ok and len(self.requestline.split()) < 3:
            self.send_error(400)
            return False
        # Any other method the stdlib would answer with 501: same answer as the rest.
        if ok and not hasattr(self, f"do_{self.command}"):
            self._dispatch(self._not_allowed)
            return False
        return ok

    def _get(self) -> None:
        self.route = "?"
        parts = urlsplit(self.path)
        path = parts.path
        if not self.path.startswith("/") or self.path.startswith("//"):
            self._check_common()
            raise _Refused(404, "not found")
        if path == "/auth":
            self.route = "/auth"
            self._check_common(allow_none=True)
            self._auth(parts.query)
            return
        if path == "/":
            self.route = "/"
            self._check_common(allow_none=True)
            self._require_session()
            self._static("index.html")
            return
        self._check_common()
        csrf = self._require_session()
        if path.startswith("/static/"):
            self.route = "/static/{name}"
            name = path[len("/static/"):]
            if name not in STATIC_FILES:
                raise _Refused(404, "not found")
            self._static(name)
            return
        found = _AUDIO_RE.fullmatch(path)
        self._require_page_key(csrf, query=parts.query if found else None)
        if path == "/api/session":
            self.route = "/api/session"
            self._json({"csrf": self._session(), "open_take": self.gui.library.pinned_id()})
            return
        if path == "/api/takes":
            self.route = "/api/takes"
            self._json({"takes": self.gui.library.shelf()})
            return
        if path == "/api/home":
            self.route = "/api/home"
            self._json(self.gui.takes.home())
            return
        if path == "/api/live":
            self.route = "/api/live"
            self._json(self.gui.takes.live())
            return
        if path == "/api/setup":
            self.route = "/api/setup"
            self._setup_view(parts.query)
            return
        if found:
            self.route = "/api/take/{id}/audio/{track}"
            self._audio(found.group(1), found.group(2))
            return
        found = _TAKE_RE.fullmatch(path)
        if found:
            self.route = "/api/take/{id}"
            doc = self.gui.library.take(found.group(1))
            if doc is None:
                raise _Refused(404, "no such take")
            self._json(doc)
            return
        raise _Refused(404, "not found")

    def _post(self) -> None:
        self.route = "?"
        op = self.path[len("/api/"):] if self.path.startswith("/api/") else ""
        if op in POST_OPS:
            self.route = f"/api/{op}"
        self._check_common()
        # Required here, where a GET may leave it out.
        if not self.headers.get("Origin"):
            raise _Refused(403, "origin required")
        csrf = self._require_session()
        if op not in POST_OPS:
            raise _Refused(404, "not found")
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise _Refused(415, "content type must be application/json")
        sent = self.headers.get("X-Plume-Token") or ""
        if not hmac.compare_digest(sent.encode(), csrf.encode()):
            raise _Refused(403, "bad token")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise _Refused(400, "bad length") from None
        if length > _MAX_BODY_BYTES:
            # Read a bounded amount, so the sender gets this answer and not a reset.
            self.rfile.read(min(length, _MAX_DRAIN_BYTES))
            raise _Refused(413, "too long")
        try:
            body = json.loads(self.rfile.read(max(0, length)))
        except (ValueError, RecursionError):
            raise _Refused(400, "body must be JSON") from None
        if not isinstance(body, dict):
            raise _Refused(400, "body must be an object")
        if op in ("keep", "rename"):
            self._keep_or_rename(op, body)
        else:
            self._take_op(op, body)

    def _setup_view(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True).get("device", [])
        try:
            self._json(self.gui.takes.setup(values[0] if len(values) == 1 else None))
        except guitake.Refused as refused:
            self._json({"error": refused.message}, refused.status)

    def _take_op(self, op: str, body: dict) -> None:
        takes = self.gui.takes
        try:
            if op == "start":
                self._json(takes.start(body))
            elif op == "stop":
                self._json(takes.stop())
            elif op == "setup/listen":
                self._json(takes.listen(body))
            elif op == "setup/answer":
                self._json(takes.answer(body))
            elif op in ("setup/cancel", "setup/reset"):
                if body:
                    raise guitake.Refused(400, "That is not something setup understands.")
                self._json(takes.cancel_listen() if op == "setup/cancel" else takes.reset_setup())
            else:
                self._json(takes.cue(body))
        except guitake.Refused as refused:
            doc = {"error": refused.message}
            if refused.live is not None:
                doc["live"] = refused.live
            self._json(doc, refused.status)

    def _keep_or_rename(self, op: str, body: dict) -> None:
        if not isinstance(body.get("take"), str):
            raise _Refused(400, "body needs a take")
        kept, name = None, None
        if op == "keep":
            kept = body.get("kept")
            if not isinstance(kept, bool):
                raise _Refused(400, "kept must be true or false")
        else:
            name = body.get("name")
            if not isinstance(name, str):
                raise _Refused(400, "name must be text")
        try:
            doc = self.gui.library.write_keep(body["take"], kept=kept, name=name)
        except OSError:
            raise _Refused(500, "could not write in the take's folder") from None
        if doc is None:
            raise _Refused(404, "no such take")
        self._json(doc)

    # -- routes

    def _auth(self, query: str) -> None:
        tokens = parse_qs(query, keep_blank_values=True).get("token", [])
        token = tokens[0] if len(tokens) == 1 else ""
        value = self.gui.redeem(token)
        if value is not None:
            cookie = f"{self.gui.cookie_name}={value}; Path=/; HttpOnly; SameSite=Strict"
            # In the fragment, which browsers keep to themselves.  The page removes it.
            where = f"/#k={self.gui.session_csrf(value)}"
            self._send(303, headers={"Location": where, "Set-Cookie": cookie})
            return
        if self._session() is not None:
            self._send(303, headers={"Location": "/"})
            return
        raise _Refused(403, "That link was already used or is not valid. "
                            "Run 'plume gui url' for a new one.")

    def _static(self, name: str) -> None:
        try:
            body = (_STATIC_DIR / name).read_bytes()
        except OSError:
            raise _Refused(404, "not found") from None
        self._send(200, body, STATIC_FILES[name])

    def _together(self, take_id: str) -> None:
        """Both recordings as one, lined up and mixed as it is sent, so a listen writes nothing."""
        folder = self.gui.library.path_of(take_id)
        doc = self.gui.library.take(take_id) if folder is not None else None
        if doc is None or not doc.get("together"):
            raise _Refused(404, "no such recording")
        try:
            human_file = sf.SoundFile(_open_track(folder, REC_HUMAN))
            machine_file = sf.SoundFile(_open_track(folder, REC_MACHINE))
        except (OSError, RuntimeError, ValueError):
            raise _Refused(404, "no such recording") from None
        with human_file as human, machine_file as machine:
            if human.samplerate != machine.samplerate:
                raise _Refused(404, "no such recording")
            shift = record.playing_delay_frames(folder, human.samplerate)
            frames = max(human.frames - shift, machine.frames)
            if frames <= 0:
                raise _Refused(404, "no such recording")
            size = _WAV_HEADER + frames * 2
            start, end, status = 0, size - 1, 200
            headers = {"Accept-Ranges": "bytes"}
            wanted = self.headers.get("Range")
            if wanted is not None:
                bounds = _parse_range(wanted, size)
                if bounds is None:
                    self._send(416, headers={"Content-Range": f"bytes */{size}"})
                    return
                start, end = bounds
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(end - start + 1))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if start < _WAV_HEADER:
                self.wfile.write(_wav_header(frames, human.samplerate)[start:end + 1])
            at = max(start, _WAV_HEADER)
            while at <= end:
                frame = (at - _WAV_HEADER) // 2
                count = min(_TOGETHER_FRAMES, frames - frame)
                pcm = _mixed_pcm(human, machine, shift, frame, count)[(at - _WAV_HEADER) % 2:]
                pcm = pcm[:end + 1 - at]
                self.wfile.write(pcm)
                at += len(pcm)
            self._log(status)

    def _audio(self, take_id: str, track: str) -> None:
        if track == TOGETHER:
            self._together(take_id)
            return
        if track not in _TRACKS:
            raise _Refused(404, "no such recording")
        folder = self.gui.library.path_of(take_id)
        if folder is None:
            raise _Refused(404, "no such take")
        try:
            # By handle, so the folder's name cannot change under this.  O_NOFOLLOW: a
            # symlink in a take folder could point at anything.  O_NONBLOCK: opening
            # a FIFO would wait for ever, before the regular-file check below.
            where = _open_folder(folder)
            try:
                fd = os.open(_TRACKS[track], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=where)
            finally:
                os.close(where)
        except OSError:
            raise _Refused(404, "no such recording") from None
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise _Refused(404, "no such recording")
            size = st.st_size
            start, end, status = 0, size - 1, 200
            headers = {"Accept-Ranges": "bytes"}
            wanted = self.headers.get("Range")
            if wanted is not None:
                bounds = _parse_range(wanted, size)
                if bounds is None:
                    self._send(416, headers={"Content-Range": f"bytes */{size}"})
                    return
                start, end = bounds
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            length = max(0, end - start + 1)
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(length))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(_CHUNK_BYTES, left))
                if not chunk:
                    # The file shrank.  The promised length cannot be kept now.
                    self.close_connection = True
                    break
                self.wfile.write(chunk)
                left -= len(chunk)
            self._log(status)


def _open_track(folder: Path, name: str):
    """A recording opened by handle, refusing a symlink, a FIFO or anything but a file."""
    where = _open_folder(folder)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=where)
    finally:
        os.close(where)
    f = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
        f.close()
        raise OSError("not a file")
    return f


def _wav_header(frames: int, rate: int) -> bytes:
    data = frames * 2
    return (b"RIFF" + (36 + data).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + rate.to_bytes(4, "little")
            + (rate * 2).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
            + b"data" + data.to_bytes(4, "little"))


def _read_mono(track, start: int, count: int) -> np.ndarray:
    """``count`` samples from ``start``, silence past either end of the file."""
    out = np.zeros(count, dtype=np.float32)
    if start >= track.frames:
        return out
    track.seek(start)
    got = track.read(count, dtype="float32", always_2d=True).mean(axis=1)
    out[:len(got)] = got
    return out


def _mixed_pcm(human, machine, shift: int, frame: int, count: int) -> bytes:
    mixed = (_read_mono(human, frame + shift, count) + _read_mono(machine, frame, count)) * _TOGETHER_GAIN
    return (np.clip(mixed, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """One byte range, or None.  Several ranges in one request are not offered."""
    found = _RANGE_RE.fullmatch(header.strip())
    if not found or size <= 0:
        return None
    first, last = found.groups()
    if first == "" and last == "":
        return None
    if first == "":
        count = int(last)
        if count == 0:
            return None
        return max(0, size - count), size - 1
    start = int(first)
    end = size - 1 if last == "" else min(int(last), size - 1)
    if start >= size or end < start:
        return None
    return start, end


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    gui: "GuiServer"

    def handle_error(self, request, client_address):  # noqa: ARG002
        # The stdlib prints a traceback, which can hold the request path.
        self.gui.log("? ? 500")


class GuiServer:
    """The bridge: an HTTP server on loopback, its session state, and its library."""

    def __init__(self, *, roots=(), pinned=None, log=None, write_state: bool = True,
                 spawner=None, lister=None, sessions=None, stopper=None, **take_options):
        self.library = Library(roots, pinned)
        self.takes = guitake.TakeRunner(
            self.library.roots[0] if self.library.roots else default_library(),
            log_path=lambda: state_path().with_name("take.log"),
            setup_path=lambda: state_path().with_name("setup.json"),
            id_of=self.library.id_of_folder, spawner=spawner, lister=lister,
            sessions=sessions, stopper=stopper, **take_options)
        self.log = log if log is not None else (lambda line: print(line, file=sys.stderr, flush=True))
        self._write_state_file = write_state
        self._key = secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._used: set[str] = set()
        self._sessions: dict[str, str] = {}
        self.httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self.port = 0
        self.host = ""
        self.origin = ""
        self.cookie_name = ""

    # -- lifecycle

    def start(self) -> "GuiServer":
        httpd = _Server(("127.0.0.1", 0), _Handler)
        httpd.gui = self
        self.httpd = httpd
        self.port = httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.origin = f"http://{self.host}"
        self.cookie_name = f"plume_{self.port}"
        if self._write_state_file:
            try:
                doc = {"pid": os.getpid(), "port": self.port, "key": self._key.hex()}
                _write_state(doc)
                _write_state(doc, _own_state_path(self.port))
            except OSError:
                httpd.server_close()
                raise
        self._thread = threading.Thread(target=httpd.serve_forever, name="plume-gui",
                                        kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.takes.shutdown()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._write_state_file:
            self._forget_state()

    def _forget_state(self) -> None:
        """Remove this bridge's files, and the newest-bridge one if it is still this."""
        path = state_path()
        try:
            _own_state_path(self.port).unlink()
        except OSError:
            pass
        try:
            with open(path, encoding="utf-8") as f:
                mine = json.load(f).get("port") == self.port
            if mine:
                path.unlink()
        except (OSError, ValueError, AttributeError):
            pass

    # -- tokens and sessions

    def mint_token(self) -> str:
        return _mint(self._key)

    def first_url(self) -> str:
        return f"{self.origin}/auth?token={self.mint_token()}"

    def redeem(self, token: str) -> str | None:
        """A new session's cookie value, once, for a token this bridge's key signed."""
        found = _TOKEN_RE.fullmatch(token)
        if not found:
            return None
        nonce, signature = found.groups()
        if not hmac.compare_digest(signature, _sign(self._key, nonce)):
            return None
        with self._lock:
            if nonce in self._used:
                return None
            self._used.add(nonce)
            value = secrets.token_hex(32)
            self._sessions[value] = secrets.token_hex(32)
        return value

    def session_csrf(self, value: str) -> str | None:
        with self._lock:
            for known, csrf in self._sessions.items():
                if hmac.compare_digest(known.encode(), value.encode()):
                    return csrf
        return None


def default_library() -> Path:
    return Path.home() / "Music" / "Plume"


class _Closing(BaseException):
    """SIGTERM or SIGHUP: leave the same way Ctrl-C does."""


def _closing_signal(signum, frame):  # noqa: ARG001
    raise _Closing


def _catch_closing_signals() -> dict:
    """A closed terminal must not orphan the take this bridge started."""
    previous = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                previous[sig] = signal.signal(sig, _closing_signal)
            except ValueError:      # not the main thread: then it is not ours to set
                pass
    return previous


def _stop_the_take(server: GuiServer, write) -> None:
    def say(line: str) -> None:
        try:
            write(line)
        except OSError:              # the terminal may be what went away
            pass

    try:
        if server.takes.live(poll=False)["state"] in ("starting", "waiting", "playing", "stopping"):
            say("Stopping the take...")
        stopped = server.takes.shutdown()
    except (KeyboardInterrupt, _Closing):
        stopped = False
    if not stopped:
        say("The take is still running. 'plume stop' ends it.")


def serve(*, take=None, takes=None, open_browser: bool = True, wait=None, **server_options) -> None:
    """Run until Ctrl-C, which also stops the take this started.

    Opens no audio device, and starts nothing until a person clicks Start.
    """
    def write(line: str) -> None:
        print(line, flush=True)

    problem = platform_problem()
    if problem is not None:
        raise GuiError(problem)
    pinned = None
    if take is not None:
        pinned = Path(take)
        if not pinned.is_dir() or not record.is_take_dir(pinned):
            raise GuiError(f"{take} is not a take folder (no session.json, decisions.ndjson, "
                           "ending.json or human.wav in it).")
    roots = [Path(takes) if takes is not None else default_library()]
    server = GuiServer(roots=roots, pinned=pinned, **server_options).start()
    previous = _catch_closing_signals()
    try:
        opened = False
        if open_browser:
            import webbrowser  # noqa: PLC0415

            opened = webbrowser.open(server.first_url())
        if opened:
            write("plume gui is open in your browser. It starts nothing until you click Start.")
            write("Lost the page? 'plume gui url' prints a new link. Ctrl-C here to quit.")
        else:
            # Asked for, or no browser.  This one-time link goes to the player's terminal only.
            write(server.first_url())
            write("Open that link once. 'plume gui url' prints a new one. Ctrl-C here to quit.")
        if wait is None:
            threading.Event().wait()
        else:
            wait(server)
    except (KeyboardInterrupt, _Closing):
        pass
    finally:
        _stop_the_take(server, write)
        try:
            server.close()
        except (KeyboardInterrupt, _Closing):
            pass
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
