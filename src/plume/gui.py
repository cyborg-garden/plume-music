"""``plume gui``: a local page for starting takes and reading them.

The page opens no device and starts nothing until a person clicks Start.  This
process never opens an audio device itself: Start runs ``plume lead`` as a child, and
everything about that child lives in :mod:`plume.guitake`.  The other way to hear
anything is a person clicking Listen, which makes their own browser fetch a WAV from
here.

HTTP lets any website try to reach this, unlike plume's cue socket, and the WAVs
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
- a POST also needs a JSON content type.  The one exception is a video being handed
  over, which is the file's own bytes and says so; it passes every other check first;
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
import math
import secrets
import shutil
import socket
import stat
import sys
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import soundfile as sf

from . import fx as _fx
from . import guitake, oscompat, record, takevideo
from .realtime import (MIX_CHUNK, REC_HUMAN, REC_MACHINE, REC_MIX, REC_STREAM, is_own_mix, list_sessions,
                       own_mix_note, wav_header)

STATE_DIR_ENV = "PLUME_GUI_STATE_DIR"

GET_ROUTES = ("/", "/auth", "/static/{name}", "/api/session", "/api/takes",
              "/api/take/{id}", "/api/take/{id}/audio/{track}", "/api/home", "/api/live", "/api/setup",
              "/api/guide", "/api/take/{id}/video", "/api/take/{id}/video/preview")
POST_OPS = frozenset({"keep", "rename", "start", "stop", "cue", "setup/listen", "setup/answer",
                      "setup/cancel", "setup/reset", "style", "reveal", "video/place", "video/cancel",
                      "video/show"})
#: The one POST whose body is not JSON: a video, as its own bytes, for the take named.
UPLOAD_ROUTE = "/api/video/{id}"

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
    "guide.js": "text/javascript; charset=utf-8",
    "video.js": "text/javascript; charset=utf-8",
    "fonts/schibsted-grotesk-latin-400-normal.woff2": "font/woff2",
    "fonts/schibsted-grotesk-latin-500-normal.woff2": "font/woff2",
    "fonts/dm-mono-latin-400-normal.woff2": "font/woff2",
}
_GUIDE_DIR = Path(__file__).with_name("guide")
#: The guide's pages, in reading order.  Each is ``guide/<id>.md`` in the package, written in
#: the small Markdown the page renders itself.  The page asks for them by this list only.
GUIDE_PAGES = ("getting-started", "playing", "sound", "terminal", "troubleshooting")
_TRACKS = {"human": REC_HUMAN, "machine": REC_MACHINE}
TOGETHER = "together"
# What leaves room for both when they peak at once.
_TOGETHER_GAIN = 0.7
# The band over that.  Both tracks at 0.7 left the band buried: a direct-input page take
# (2026-11-05) had machine.wav about 12 dB (LUFS) under human.wav, and the owner asked for
# the other instruments a little louder with her still slightly on top.
_TOGETHER_BAND_DB = 3.0
# Where the sum is soft-limited rather than hard-clipped when both peak at once.  The
# limiter is untouched below 3 dB under it (0.71 of full scale).
_TOGETHER_CEILING_DB = 0.0
_TOGETHER_FRAMES = 65536
_WAV_HEADER = 44
#: A WAV's lengths are 32 bits wide, so nothing longer than this can be written as one.
_WAV_MAX_BYTES = 0xFFFFFFFF
#: Raised when a saved mix would come out different for a reason the three numbers
#: above do not show.  Those numbers are part of the rule too: see :func:`_mix_rule`.
_MIX_RULE = 1
_MIX_TMP_RE = re.compile(r"\." + re.escape(REC_MIX) + r"\.[0-9a-f]{16}\.tmp")
#: A half-written mix nothing has touched for this long was left by a run that died.
_MIX_TMP_STALE_S = 60.0

REC_VIDEO = takevideo.REC_VIDEO
#: A video being taken in, and one being written, each under a name nobody could plant.
_VIDEO_IN_RE = re.compile(r"\.video-in\.[0-9a-f]{16}\.tmp")
_VIDEO_OUT_RE = re.compile(r"\." + re.escape(REC_VIDEO) + r"\.[0-9a-f]{16}\.tmp")
#: Either of those that nothing has touched for this long was left by a run that died.
#: Long, because a video waiting for a nudge is one nothing is touching.
_VIDEO_TMP_STALE_S = 3600.0
#: While a bridge holds a copy, it says so this often by touching the copy, so that
#: another bridge on the same takes never takes it for a leftover, however long it is held.
_VIDEO_HEARTBEAT_S = 60.0
#: The largest video taken in: an hour of a phone's best.
_VIDEO_MAX_BYTES = 16 * 1024 ** 3
#: Room the disk must have beyond the video and the copy made of it.
_VIDEO_SPARE_BYTES = 64 * 1024 * 1024
_VIDEO_PREVIEW_S = 8.0
#: A preview starts this long before the player's first note.
_VIDEO_PREVIEW_LEAD_S = 1.0
#: How far either side of an unsure match the nudge reaches, and its step.
_VIDEO_NUDGE_S = 5.0
_VIDEO_NUDGE_STEP_S = 0.01
#: With no sound to match, the nudge covers the whole video, in coarser steps.
_VIDEO_PLACE_STEP_S = 0.05
_VIDEO_WORKING = ("receiving", "listening", "making")
#: The most cancel waits for work under way to let go.  Its tool is ended at once, so this
#: is a moment on any machine; the bound is only for a machine that has stopped.
_VIDEO_CANCEL_WAIT_S = 30.0

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
_VIDEO_RE = re.compile(r"/api/take/([^/]*)/video")
_PREVIEW_RE = re.compile(r"/api/take/([^/]*)/video/preview")
_UPLOAD_RE = re.compile(r"/api/video/([^/]*)")
_NUMBER_RE = re.compile(r"-?\d{1,6}(\.\d{1,6})?")
_TOKEN_RE = re.compile(r"([0-9a-f]{32})\.([0-9a-f]{64})")
_RANGE_RE = re.compile(r"bytes=(\d{0,18})-(\d{0,18})")


class GuiError(RuntimeError):
    """A clean, user-facing failure."""


class MixInTheWay(GuiError):
    """The take folder holds a ``mix.wav`` plume does not know for its own, so it stays.

    The player's own, or one of plume's that is no longer exactly the file plume wrote:
    edited, damaged, or saved again by another program.  Also one that is locked.
    """


class MixInUse(GuiError):
    """Plume's own older mix could not be replaced: another program has it open."""


#: Whether a refused rename over an old mix means another program has that mix open.
#: On Windows it does.  Elsewhere an open file never stops a rename, so it means
#: something else: a locked file, or a folder that keeps its files.
_RENAME_REFUSAL_IS_IN_USE = oscompat.IS_WINDOWS


class _Abandoned(Exception):
    """A mix given up before it was put in place, because a take began in its folder."""


class _NotReplaced(PermissionError):
    """A finished mix that could not be renamed into place.  Only that step raises it."""


class VideoRefused(GuiError):
    """A video that was not taken in, or a step that could not be taken.

    ``why`` is a short code the page has words for; ``status`` is the HTTP answer.
    """

    def __init__(self, status: int, why: str):
        super().__init__(why)
        self.status, self.why = status, why


class _VideoJob:
    """One video on its way into one take.  The library keeps at most one per take."""

    def __init__(self, take_id: str, folder: Path, keep_sound: bool):
        self.take_id, self.folder, self.keep_sound = take_id, folder, keep_sound
        self.state = "receiving"
        self.progress: float | None = 0.0
        self.why: str | None = None
        #: The video as it was handed over, by its name in the take's folder.
        self.source: str | None = None
        #: The video being written, by its name in the take's folder, while it is.
        self.writing: str | None = None
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        #: Set when nothing more will happen to this job without being asked.
        self.settled = threading.Event()
        self.probe: takevideo.Probe | None = None
        self.start_s: float | None = None
        self.confidence: float | None = None
        self.drift_ppm: float | None = None
        self.listen_s = 0.0
        self.take_s = 0.0
        #: One preview or one save at a time.
        self.busy = threading.Lock()


def _free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


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
    oscompat.chmod_private(path.parent, 0o700)
    # O_NOFOLLOW: a planted symlink must not redirect the key somewhere readable.
    fd = oscompat.open_private(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        oscompat.fchmod_private(fd, 0o600)
        os.write(fd, json.dumps(doc).encode())
    finally:
        os.close(fd)


def _pid_alive(pid) -> bool:
    # Never os.kill(pid, 0) here: on Windows that is a Ctrl-C to the bridge it asks about.
    return oscompat.pid_alive(pid)


def _sign(key: bytes, nonce: str) -> str:
    return hmac.new(key, nonce.encode(), hashlib.sha256).hexdigest()


def _mint(key: bytes) -> str:
    nonce = secrets.token_hex(16)
    return f"{nonce}.{_sign(key, nonce)}"


def _read_state(path: Path) -> tuple[bytes, int, object] | None:
    try:
        fd = oscompat.open_private(path, oscompat.read_flags())
    except OSError:
        return None
    try:
        with os.fdopen(fd, encoding="utf-8") as f:
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


def _open_folder(path: Path):
    """A handle on the folder itself, so a later swap of its name changes nothing.

    Where folders cannot be opened by handle (Windows), the handle is the path,
    checked here to be a real folder with no symlink on the way.  The calls below
    then refuse a symlink at the file itself.  That is a narrower guarantee than a
    handle, and the reason it is acceptable there: making a symlink on Windows
    takes an administrator or developer mode.
    """
    if oscompat.DIR_FDS:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | oscompat.O_NOFOLLOW
                       | oscompat.O_NONBLOCK)
    if not _still_real(Path(path)):
        raise NotADirectoryError(str(path))
    return Path(path)


def _close_folder(where) -> None:
    if isinstance(where, int):
        os.close(where)


def _open_in(where, name: str, flags: int, mode: int = 0o600) -> int:
    """Open ``name`` inside a folder from :func:`_open_folder`, never through a symlink."""
    if isinstance(where, int):
        return oscompat.open_private(name, flags, mode, dir_fd=where)
    return oscompat.open_private(Path(where) / name, flags, mode)


def _replace_in(where, src: str, dst: str) -> None:
    # Retried briefly where a reader holding the old file open makes it fail.
    if isinstance(where, int):
        oscompat.replace(src, dst, src_dir_fd=where, dst_dir_fd=where)
    else:
        oscompat.replace(Path(where) / src, Path(where) / dst)


def _unlink_in(where, name: str) -> None:
    if isinstance(where, int):
        os.unlink(name, dir_fd=where)
    else:
        os.unlink(Path(where) / name)


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

    def __init__(self, roots=(), pinned=None, ffmpeg=None):
        #: Finds the ffmpeg to run, or None.  Asked once, the first time it matters.
        self._find_ffmpeg = takevideo.find_ffmpeg if ffmpeg is None else ffmpeg
        self._ffmpeg: tuple | None = None
        self._videos: dict[str, _VideoJob] = {}
        self._heartbeat: threading.Thread | None = None
        #: Where a line about something that went wrong goes.  The bridge sets its own.
        self.log = lambda line: None
        self._heartbeat_stop = threading.Event()
        self.roots = [Path(r) for r in roots]
        self.pinned = None if pinned is None else Path(pinned).resolve()
        self._lock = threading.Lock()
        #: Held across disk work, so it is not the lock every reader needs.
        self._keep_lock = threading.Lock()
        #: One per take, so a long mix holds up nobody else's.
        self._mix_locks: dict[str, threading.Lock] = {}
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
                     record.REC_KEEP, record.REC_STREAM, REC_HUMAN, REC_MACHINE):
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
        found = self._scan()
        self.sweep_leftovers(found)
        for take_id in found:
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
                fd = _open_in(folder, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(json.dumps(doc))
                    _replace_in(folder, tmp, record.REC_KEEP)
                except OSError:
                    try:
                        _unlink_in(folder, tmp)
                    except OSError:
                        pass
                    raise
            finally:
                _close_folder(folder)
        return doc

    def _mix_lock(self, take_id: str) -> threading.Lock:
        with self._lock:
            return self._mix_locks.setdefault(take_id, threading.Lock())

    def write_mix(self, take_id: str) -> str | None:
        """Save both recordings as one file in the take's folder, and say its name.

        The same samples Together plays: the player moved earlier by the take's round
        trip, the band lifted over her, the sum soft-limited.  None when there is nothing
        to mix: one recording missing, two sample rates, more than a WAV can hold, or a
        take still being recorded, whose folder belongs to the session writing it.  A
        room mic take gets one too; :func:`band_may_be_twice` says when to warn of that.

        Only plume's own ``mix.wav`` is ever replaced.  One the player put there raises
        :class:`MixInTheWay`, and an older one of plume's that another program holds
        open raises :class:`MixInUse`; in both the file is as it was.

        Written under another name and renamed, so ``mix.wav`` is whole or absent.  The
        file ends by saying what it was made from, and is left alone only while that is
        still exactly what is here: see :func:`_mix_facts`.
        """
        path = self.path_of(take_id)
        if path is None:
            return None
        with self._mix_lock(take_id):
            # Asked with the lock held, and again before the mix is put in place: asked
            # before, a take could start here in between.
            if str(path) in _live_folders():
                return None
            raw = []
            try:
                try:
                    for name in (REC_HUMAN, REC_MACHINE):
                        raw.append(_open_track(path, name))
                    stats = [os.fstat(f.fileno()) for f in raw]
                    human, machine = [sf.SoundFile(f) for f in raw]
                except (OSError, RuntimeError, ValueError):
                    return None
                if human.samplerate != machine.samplerate:
                    return None
                shift = record.playing_delay_frames(path, human.samplerate)
                frames = max(human.frames - shift, machine.frames)
                facts = _mix_facts(path, *stats, shift=shift, frames=frames, rate=human.samplerate)
                if frames <= 0 or _WAV_HEADER + frames * 2 + len(_mix_note(facts)) > _WAV_MAX_BYTES:
                    return None
                theirs = path / REC_MIX
                # Read once: this hashes the whole mix, and says both whose it is and
                # what it was made from.
                note = own_mix_note(theirs)
                if not _mix_is_current(note, facts):
                    there = os.path.lexists(theirs)
                    if there and note is None:
                        raise MixInTheWay(REC_MIX)
                    # Readable by whoever can read the recordings it is made from.
                    mode = stat.S_IMODE(stats[0].st_mode)
                    try:
                        self._save_mix(path, human, machine, shift, frames, facts, mode)
                    except _Abandoned:
                        return None
                    except _NotReplaced:
                        # The new mix was written whole and could not take the old one's
                        # place.  Failing to write it at all is not this, and is not caught.
                        # Windows refuses the rename while a player holds the old one open,
                        # and only there is that what a refusal means.  Elsewhere it is a
                        # locked file or a folder that will not let go of it: a mix plume
                        # cannot replace, which is what the page then says.  A mix marked
                        # read-only is refused on Windows as well, and nothing has it open.
                        if there:
                            in_use = _RENAME_REFUSAL_IS_IN_USE and _may_be_written(theirs)
                            raise (MixInUse if in_use else MixInTheWay)(REC_MIX) from None
                        raise
            finally:
                for f in raw:
                    f.close()
        return REC_MIX

    @staticmethod
    def _save_mix(path: Path, human, machine, shift: int, frames: int, facts: dict,
                  mode: int) -> None:
        folder = _open_folder(path)
        # As for keep.json: a name nobody could have planted, and O_EXCL in case they did.
        tmp = f".{REC_MIX}.{secrets.token_hex(8)}.tmp"
        try:
            _sweep_mix_leftovers(path, folder)
            fd = _open_in(folder, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as f:
                    oscompat.fchmod_private(f.fileno(), mode)
                    # The note's length does not depend on the digest, so the header can
                    # say how long the file will be before the samples exist.
                    f.write(_wav_header(frames, human.samplerate, extra=len(_mix_note(facts))))
                    heard = hashlib.sha256()
                    for frame in range(0, frames, _TOGETHER_FRAMES):
                        count = min(_TOGETHER_FRAMES, frames - frame)
                        pcm = _mixed_pcm(human, machine, shift, frame, count)
                        heard.update(pcm)          # as they are written: nothing is read back
                        f.write(pcm)
                    f.write(_mix_note(facts, heard.hexdigest()))
                if str(path) in _live_folders():
                    # A take started here while this was being written.  The folder is
                    # that session's now, and its recordings are no longer these.
                    raise _Abandoned(REC_MIX)
                try:
                    _replace_in(folder, tmp, REC_MIX)
                except PermissionError as e:
                    raise _NotReplaced(*e.args) from None
            except BaseException:
                try:
                    _unlink_in(folder, tmp)
                except OSError:
                    pass
                raise
        finally:
            _close_folder(folder)


    # ---------------------------------------------------------------- a video

    def start_heartbeat(self) -> None:
        """Keep every copy this bridge holds looking fresh, for as long as it holds it."""
        if self._heartbeat is None:
            self._heartbeat_stop.clear()
            self._heartbeat = threading.Thread(target=self._beat, name="plume-video-held", daemon=True)
            self._heartbeat.start()

    def stop_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat_stop.set()
            self._heartbeat.join(timeout=30)
            self._heartbeat = None

    def _beat(self) -> None:
        while not self._heartbeat_stop.wait(_VIDEO_HEARTBEAT_S):
            try:
                self.say_held()
            except Exception:  # noqa: BLE001 - one bad moment must not end the heartbeat for good
                self.log("video: could not mark held copies; trying again")

    def say_held(self) -> None:
        """Touch each copy a job here holds: the video taken in, and one being written."""
        with self._lock:
            held = [(job.folder, name) for job in self._videos.values()
                    for name in (job.source, job.writing) if name is not None]
        for folder, name in held:
            _touch(folder / name)

    def sweep_leftovers(self, found: dict[str, Path] | None = None) -> None:
        """Remove copies of videos that a run which died left in any take's folder.

        Such a copy is the film as it was handed over, where it was filmed and all,
        under a hidden name.  Left there, it goes wherever the folder goes.  Only names
        plume makes, only plain files, only old ones, and never in a take this bridge is
        working on or holding a video for: one waiting for a nudge can wait all day.  A
        copy another bridge holds is never old: that bridge touches it while it holds it.
        """
        for take_id, path in (self._scan() if found is None else found).items():
            with self._lock:
                if take_id in self._videos:
                    continue
            try:
                folder = _open_folder(path)
            except OSError:
                continue
            try:
                _sweep_video_leftovers(path, folder)
            finally:
                _close_folder(folder)

    def ffmpeg(self) -> str | None:
        with self._lock:
            if self._ffmpeg is None:
                self._ffmpeg = (self._find_ffmpeg(),)
            return self._ffmpeg[0]

    def _can_video(self, path: Path) -> bool:
        """A take that is finished and has both recordings, each a plain file."""
        if str(path) in _live_folders():
            return False
        return all(record._is_regular(path / name) for name in (REC_HUMAN, REC_MACHINE))

    def video_status(self, take_id: str) -> dict | None:
        """Where a take's video has got to, for the page.  None for no such take."""
        path = self.path_of(take_id)
        if path is None:
            return None
        there = os.path.lexists(path / REC_VIDEO)
        own = there and takevideo.is_own_video(path / REC_VIDEO)
        doc = {"state": "idle", "can": self._can_video(path), "ready": self.ffmpeg() is not None,
               "have": bool(own), "theirs": there and not own, "max_bytes": _VIDEO_MAX_BYTES}
        with self._lock:
            job = self._videos.get(take_id)
        if job is None:
            return doc
        doc.update(state=job.state, keep_sound=job.keep_sound)
        if job.state in _VIDEO_WORKING and job.cancel.is_set():
            # Asked to stop, and its tool not yet ended: said, so nobody presses Stop again.
            doc.update(state="stopping", progress=None)
        elif job.state in _VIDEO_WORKING:
            doc["progress"] = job.progress
        elif job.state == "failed":
            doc["why"] = job.why
        elif job.state == "done":
            doc.update(file=REC_VIDEO, start_s=job.start_s, confidence=job.confidence,
                       drift_ppm=job.drift_ppm)
        elif job.state == "unsure":
            silent = job.why == "silent"
            if silent:
                low, high, step = -job.take_s, job.probe.duration_s, _VIDEO_PLACE_STEP_S
            else:
                low, high, step = job.start_s - _VIDEO_NUDGE_S, job.start_s + _VIDEO_NUDGE_S, _VIDEO_NUDGE_STEP_S
            doc.update(why=job.why, start_s=job.start_s, confidence=job.confidence, min_s=round(low, 3),
                       max_s=round(high, 3), step_s=step, listen=not silent)
        return doc

    def video_receive(self, take_id: str, body, length: int, keep_sound: bool) -> dict:
        """Take a video in from ``body``, ``length`` bytes of it, and start the work.

        It is written, as it arrives, to a file in the take's folder under a name of
        plume's own choosing: nothing the sender calls it is ever seen.  Raises
        :class:`VideoRefused` when it is not taken in, and then nothing is left behind.
        """
        path = self.path_of(take_id)
        if path is None:
            raise VideoRefused(404, "no_take")
        if not self._can_video(path):
            raise VideoRefused(409, "no_take")
        if self.ffmpeg() is None:
            raise VideoRefused(409, "no_tool")
        if os.path.lexists(path / REC_VIDEO) and not takevideo.is_own_video(path / REC_VIDEO):
            raise VideoRefused(409, "theirs")
        if length > _VIDEO_MAX_BYTES:
            raise VideoRefused(413, "too_big")
        job = _VideoJob(take_id, path, keep_sound)
        with self._lock:
            old = self._videos.get(take_id)
            if old is not None and (old.state in _VIDEO_WORKING or not old.settled.is_set()):
                raise VideoRefused(409, "busy")
            self._videos[take_id] = job
        try:
            folder = _open_folder(path)
        except OSError:
            self._video_forget(job)
            raise VideoRefused(500, "failed") from None
        try:
            if old is not None:
                self._video_drop_source(old, folder)
            _sweep_video_leftovers(path, folder)
            try:
                # The video, and the copy that will be made of it.
                if _free_bytes(path) < 2 * length + _VIDEO_SPARE_BYTES:
                    raise VideoRefused(507, "no_room")
                name = f".video-in.{secrets.token_hex(8)}.tmp"
                fd = _open_in(folder, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                job.source = name
                with os.fdopen(fd, "wb") as f:
                    left = length
                    while left > 0:
                        if job.cancel.is_set():
                            raise VideoRefused(409, "cancelled")
                        chunk = body.read(min(_CHUNK_BYTES, left))
                        if not chunk:
                            raise VideoRefused(400, "cut_short")
                        f.write(chunk)
                        left -= len(chunk)
                        job.progress = 1.0 - left / length
            except BaseException:
                self._video_drop_source(job, folder)
                self._video_forget(job)
                raise
        except OSError:
            raise VideoRefused(500, "failed") from None
        finally:
            _close_folder(folder)
        job.state, job.progress = "listening", None
        job.thread = threading.Thread(target=self._video_work, args=(job,), name="plume-video", daemon=True)
        job.thread.start()
        return self.video_status(take_id)

    def _video_forget(self, job: _VideoJob) -> None:
        with self._lock:
            if self._videos.get(job.take_id) is job:
                del self._videos[job.take_id]
        job.settled.set()

    @staticmethod
    def _video_drop_source(job: _VideoJob, folder=None) -> None:
        """Remove the video a job took in.  It is plume's copy; the player's is where it was."""
        name, job.source = job.source, None
        if name is None:
            return
        try:
            where = _open_folder(job.folder) if folder is None else folder
            try:
                _unlink_in(where, name)
            finally:
                if folder is None:
                    _close_folder(where)
        except OSError:
            pass

    def _video_work(self, job: _VideoJob) -> None:
        """Read the video, find the take in it, and save it if the place is sure."""
        try:
            ffmpeg, source = self.ffmpeg(), job.folder / job.source
            cancelled = job.cancel.is_set
            job.probe = takevideo.probe(ffmpeg, source, cancelled=cancelled)
            if cancelled():
                raise takevideo.Cancelled
            with _open_track(job.folder, REC_HUMAN) as f:
                # Never shorter than the recording is: two bytes a sample at a low rate.
                long = os.fstat(f.fileno()).st_size / 16000.0
                take = takevideo.decode(ffmpeg, f, cancelled=cancelled, duration_s=long)
            heard = (takevideo.decode(ffmpeg, source, cancelled=cancelled, duration_s=job.probe.duration_s)
                     if job.probe.has_audio
                     else np.zeros(0, dtype=np.float32))
            found = takevideo.align(heard, take)
            if cancelled():
                raise takevideo.Cancelled
            job.listen_s, job.take_s = takevideo.first_sound_s(take), len(take) / takevideo.ALIGN_RATE
            job.start_s, job.confidence, job.drift_ppm = found.start_s, found.confidence, found.drift_ppm
            if found.confident:
                self._video_make(job)
            else:
                job.why = found.why or "unsure"
                job.state = "unsure"
        except takevideo.Cancelled:
            self._video_drop_source(job)
            self._video_forget(job)
        except takevideo.VideoError as e:
            self._video_fail(job, e.why)
        except Exception:  # noqa: BLE001 - whatever went wrong, the take is as it was and she can try again
            self._video_fail(job, "failed")
        finally:
            job.settled.set()

    def _video_fail(self, job: _VideoJob, why: str) -> None:
        self._video_drop_source(job)
        job.why, job.progress = why, None
        job.state = "failed"

    def _video_make(self, job: _VideoJob) -> None:
        """Write the video again with the take in it, as ``video.mp4`` in the take's folder.

        The take's part is the mix Together plays, by the same rule: the player moved
        earlier by the round trip, the band lifted over her, the sum soft-limited.  It
        is laid in so that her recording's first sample falls at ``job.start_s``.

        Only plume's own ``video.mp4`` is ever replaced, and that is asked again at the
        last moment: one the player saved while this ran stays, and this fails.
        """
        job.state, job.progress = "making", 0.0
        path, ffmpeg = job.folder, self.ffmpeg()
        raw = []
        try:
            try:
                for name in (REC_HUMAN, REC_MACHINE):
                    raw.append(_open_track(path, name))
                mode = stat.S_IMODE(os.fstat(raw[0].fileno()).st_mode)
                human, machine = [sf.SoundFile(f) for f in raw]
            except (OSError, RuntimeError, ValueError):
                raise takevideo.VideoError("failed") from None
            if human.samplerate != machine.samplerate:
                raise takevideo.VideoError("failed")
            rate = human.samplerate
            shift = record.playing_delay_frames(path, rate)

            def mixed(frame: int, count: int) -> bytes:
                return _mixed_pcm(human, machine, shift, frame, count)

            def progress(share: float) -> None:
                job.progress = share

            folder = _open_folder(path)
            tmp = f".{REC_VIDEO}.{secrets.token_hex(8)}.tmp"
            job.writing = tmp
            try:
                # Made here first, so the name the tool is given is a plain file of plume's.
                os.close(_open_in(folder, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
                try:
                    takevideo.write_video(
                        ffmpeg, path / job.source, path / tmp, mixed=mixed, rate=rate,
                        band_at=takevideo.video_frame_of(job.start_s, shift, rate),
                        duration_s=job.probe.duration_s, keep_sound=job.keep_sound and job.probe.has_audio,
                        codec=job.probe.codec, picture=job.probe.picture,
                        cancelled=job.cancel.is_set, progress=progress)
                    fd = _open_in(folder, tmp, os.O_WRONLY | os.O_APPEND | oscompat.O_BINARY, 0o600)
                    with os.fdopen(fd, "ab") as f:
                        oscompat.fchmod_private(f.fileno(), mode)
                        f.write(takevideo.stamp({"start_s": round(job.start_s, 6), "sound": job.keep_sound}))
                    there = os.path.lexists(path / REC_VIDEO)
                    if there and not takevideo.is_own_video(path / REC_VIDEO):
                        raise takevideo.VideoError("theirs")
                    try:
                        _replace_in(folder, tmp, REC_VIDEO)
                    except PermissionError:
                        raise takevideo.VideoError("in_use" if there else "failed") from None
                except BaseException:
                    try:
                        _unlink_in(folder, tmp)
                    except OSError:
                        pass
                    raise
            except OSError:
                raise takevideo.VideoError("failed") from None
            finally:
                _close_folder(folder)
        finally:
            for f in raw:
                f.close()
        self._video_drop_source(job)
        job.progress = None
        job.state = "done"

    def video_place(self, take_id: str, start_s: float, keep_sound: bool) -> dict | None:
        """Save a video that was waiting for a nudge, with the take at ``start_s``."""
        if self.path_of(take_id) is None:
            return None
        with self._lock:
            job = self._videos.get(take_id)
            if job is None or job.state != "unsure" or not job.busy.acquire(blocking=False):
                raise VideoRefused(409, "busy" if job is not None and job.state in _VIDEO_WORKING else "nothing")
            if not -job.take_s - 1.0 <= start_s <= job.probe.duration_s + 1.0:
                job.busy.release()
                raise VideoRefused(400, "bad_place")
            job.start_s, job.keep_sound = float(start_s), keep_sound
            job.state, job.progress = "making", 0.0
            job.settled.clear()

        def work() -> None:
            try:
                self._video_make(job)
            except takevideo.Cancelled:
                self._video_drop_source(job)
                self._video_forget(job)
            except takevideo.VideoError as e:
                self._video_fail(job, e.why)
            except Exception:  # noqa: BLE001 - as in _video_work
                self._video_fail(job, "failed")
            finally:
                job.busy.release()
                job.settled.set()

        job.thread = threading.Thread(target=work, name="plume-video", daemon=True)
        job.thread.start()
        return self.video_status(take_id)

    def video_preview(self, take_id: str, start_s: float) -> bytes | None:
        """A few seconds of how a nudged video would sound, as a WAV.  Writes nothing.

        The video's own sound with the take over it, from just before the player's
        first note, so what she hears is whether the two of her are together.
        """
        with self._lock:
            job = self._videos.get(take_id)
        if job is None or job.state != "unsure" or not job.busy.acquire(blocking=False):
            return None
        raw = []
        try:
            if not -job.take_s - 1.0 <= start_s <= job.probe.duration_s + 1.0:
                raise VideoRefused(400, "bad_place")
            for name in (REC_HUMAN, REC_MACHINE):
                raw.append(_open_track(job.folder, name))
            human, machine = [sf.SoundFile(f) for f in raw]
            rate = human.samplerate
            if machine.samplerate != rate or job.source is None:
                return None
            shift = record.playing_delay_frames(job.folder, rate)
            from_s = max(0.0, start_s + job.listen_s - _VIDEO_PREVIEW_LEAD_S)
            sound = takevideo.preview(
                self.ffmpeg(), job.folder / job.source, rate=rate, from_s=from_s, seconds=_VIDEO_PREVIEW_S,
                mixed=lambda frame, count: _mixed_pcm(human, machine, shift, frame, count),
                band_at=takevideo.video_frame_of(start_s, shift, rate), has_audio=job.probe.has_audio)
            pcm = (np.clip(_fx.limiter(sound, _TOGETHER_CEILING_DB), -1.0, 1.0) * 32767.0).astype("<i2")
            return _wav_header(len(pcm), rate) + pcm.tobytes()
        except (OSError, RuntimeError, ValueError, takevideo.VideoError):
            return None
        finally:
            for f in raw:
                f.close()
            job.busy.release()

    def video_cancel(self, take_id: str) -> dict | None:
        """Stop a take's video, wherever it has got to, and remove what was taken in.

        A finished ``video.mp4`` is not touched: it is hers now.  Work under way is
        stopped where it stands, tool and all, and cleans up after itself.
        """
        if self.path_of(take_id) is None:
            return None
        with self._lock:
            job = self._videos.get(take_id)
            working = job is not None and job.state in _VIDEO_WORKING
            # One request waits for a video to stop.  Any other asking meanwhile is told it
            # is stopping, and does not wait too.
            waits = working and not job.cancel.is_set()
            if job is not None:
                job.cancel.set()
                if not working:
                    del self._videos[take_id]
        if job is not None and not working:
            self._video_drop_source(job)
            job.settled.set()
        stopped = True
        if working:
            # The tool is ended from outside, so this is a moment.  Waited for, so that
            # "stopped" is true by the time it is said, and the next video is not refused.
            stopped = waits and job.settled.wait(_VIDEO_CANCEL_WAIT_S)
        doc = self.video_status(take_id)
        return {**{k: doc[k] for k in ("can", "ready", "have", "theirs", "max_bytes")},
                "state": "idle" if stopped else "stopping"}

    def video_wait(self, take_id: str, timeout: float | None = None) -> bool:
        """Wait until a take's video needs nothing more done.  True if that came."""
        with self._lock:
            job = self._videos.get(take_id)
        return True if job is None else job.settled.wait(timeout)

    def close_videos(self, timeout: float = 30.0) -> None:
        """Stop every video under way and remove every one that was waiting.

        Every one is told first, which ends each tool at once, and then all are waited
        for together, against one deadline: never a full wait per video.
        """
        with self._lock:
            jobs = list(self._videos.values())
            for job in jobs:
                job.cancel.set()
            waiting = [job for job in jobs if job.state not in _VIDEO_WORKING]
            for job in waiting:
                del self._videos[job.take_id]
        for job in waiting:
            self._video_drop_source(job)
            job.settled.set()
        deadline = time.monotonic() + timeout
        for job in jobs:
            job.settled.wait(max(0.0, deadline - time.monotonic()))


def _mix_rule() -> str:
    """How a mix is made, as text.  A mix made by another rule is made again."""
    return f"{_MIX_RULE}:{_TOGETHER_GAIN}:{_TOGETHER_BAND_DB}:{_TOGETHER_CEILING_DB}"


def _identity(st) -> list[int]:
    return [st.st_size, st.st_mtime_ns]


def _mix_facts(path: Path, human, machine, *, shift: int, frames: int, rate: int) -> dict:
    """Everything a mix's samples depend on, as the note records it.

    The rule, the delay in frames that was actually used, the length and rate, and each
    file that went in by its size and its time to the nanosecond (``human`` and
    ``machine`` are those files' ``fstat``).  "Is the mix newer?" would be the wrong
    question: a recording put back from a copy keeps its old date, and a change to the
    rule changes no date at all.  The delay is here as a number, not only as its file's
    size and date, because a file can be rewritten and keep both.
    """
    try:
        stream = _identity(os.lstat(path / REC_STREAM))
    except OSError:
        stream = None
    return {"rule": _mix_rule(), "shift": int(shift), "frames": int(frames), "rate": int(rate),
            "human": _identity(human), "machine": _identity(machine), "stream": stream}


def _mix_note(facts: dict, digest: str = "0" * 64) -> bytes:
    """The chunk that ends a saved mix: its facts, and a SHA-256 of its samples.

    The digest is what lets plume know the file again for exactly what it wrote.  It is
    always 64 characters, so the note is the same length whatever the samples are.  The
    note is inside the mix, after the samples, so it can never describe another file.
    """
    body = json.dumps({**facts, "sha256": digest}, sort_keys=True).encode()
    body += b" " * (len(body) % 2)          # a chunk is an even number of bytes
    return MIX_CHUNK + len(body).to_bytes(4, "little") + body


def _mix_is_current(note: dict | None, facts: dict) -> bool:
    """True when ``note`` is that of plume's own mix of exactly these files, by this rule.

    ``note`` is what :func:`plume.realtime.own_mix_note` read from ``mix.wav``: None
    unless the file is whole and plume's, down to its samples.  What is left to ask is
    whether it was made from what is here now.
    """
    return note is not None and {k: v for k, v in note.items() if k != "sha256"} == facts


def _may_be_written(path: Path) -> bool:
    """False when the file is marked read-only, which on Windows is the read-only attribute."""
    try:
        return bool(os.lstat(path).st_mode & stat.S_IWUSR)
    except OSError:
        return True


def _sweep_mix_leftovers(path: Path, folder) -> None:
    """Remove half-written mixes a killed run left in this take's folder.

    Only names this module makes, only plain files, and only ones nothing has written
    to for a while: another bridge may be half way through one of its own.
    """
    try:
        names = os.listdir(path)
        now = time.time()
        for name in names:
            if not _MIX_TMP_RE.fullmatch(name):
                continue
            st = os.lstat(path / name)
            if stat.S_ISREG(st.st_mode) and now - st.st_mtime > _MIX_TMP_STALE_S:
                _unlink_in(folder, name)
    except OSError:
        pass


def _touch(path: Path) -> None:
    """Mark a copy of plume's as in use now.  Never through a link, where that can be asked."""
    try:
        if os.utime in os.supports_follow_symlinks:
            os.utime(path, follow_symlinks=False)
        elif not os.path.islink(path):
            os.utime(path)
    except OSError:
        pass            # gone already: the work it belonged to has ended


def _sweep_video_leftovers(path: Path, folder) -> None:
    """Remove videos a killed run left half taken in or half written in this take's folder.

    As for mixes: only names this module makes, only plain files, only old ones.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return
    now = time.time()
    for name in names:
        if not (_VIDEO_IN_RE.fullmatch(name) or _VIDEO_OUT_RE.fullmatch(name)):
            continue
        # Each on its own: one held open elsewhere, or gone already, keeps none of the rest.
        try:
            st = os.lstat(path / name)
            if stat.S_ISREG(st.st_mode) and now - st.st_mtime > _VIDEO_TMP_STALE_S:
                _unlink_in(folder, name)
        except OSError:
            continue


def _something_inside(folder: Path) -> str | None:
    """A file a take folder holds, to show the folder by.  The player's recording first."""
    for name in (REC_HUMAN, REC_MACHINE, record.REC_MANIFEST, record.REC_DECISIONS, record.REC_ENDING):
        if record._is_regular(folder / name):
            return name
    return None


def band_may_be_twice(folder: Path) -> bool:
    """True when a mix of this take may hold the band twice.

    A plugged-in instrument is the player alone.  A mic is not, unless the take's
    manifest says the band could not reach it: ``input.isolated`` being ``true``.
    Nothing writes that yet, so today every mic take answers True.  It is read here,
    in the place and under the name the manifest's ``input`` block would carry it, so
    that a take which does say so is believed.  Anything else is a mic that could
    hear the band.
    """
    manifest = record._read_json(Path(folder) / record.REC_MANIFEST) or {}
    if record.recorded_from(manifest) == "input":
        return False
    audio = manifest.get("input")
    return not (isinstance(audio, dict) and audio.get("isolated") is True)


def _reveal(folder: Path, select: str | None) -> None:
    """Show a folder in the system's file manager.  Never under pytest: it opens a window."""
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        raise GuiError("the real file manager does not open under pytest")
    oscompat.reveal(folder, select)


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
        preview = _PREVIEW_RE.fullmatch(path)
        # An audio element cannot set a header, so what one plays may carry the key instead.
        self._require_page_key(csrf, query=parts.query if found or preview else None)
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
        if path == "/api/guide":
            self.route = "/api/guide"
            self._json({"pages": guide_pages()})
            return
        if found:
            self.route = "/api/take/{id}/audio/{track}"
            self._audio(found.group(1), found.group(2))
            return
        if preview:
            self.route = "/api/take/{id}/video/preview"
            self._video_preview(preview.group(1), parts.query)
            return
        found = _VIDEO_RE.fullmatch(path)
        if found:
            self.route = "/api/take/{id}/video"
            doc = self.gui.library.video_status(found.group(1))
            if doc is None:
                raise _Refused(404, "no such take")
            self._json(doc)
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
        parts = urlsplit(self.path)
        upload = _UPLOAD_RE.fullmatch(parts.path) if op not in POST_OPS else None
        if op in POST_OPS:
            self.route = f"/api/{op}"
        elif upload:
            self.route = UPLOAD_ROUTE
        self._check_common()
        # Required here, where a GET may leave it out.
        if not self.headers.get("Origin"):
            raise _Refused(403, "origin required")
        csrf = self._require_session()
        if op not in POST_OPS and not upload:
            raise _Refused(404, "not found")
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        wanted = "application/octet-stream" if upload else "application/json"
        if ctype != wanted:
            raise _Refused(415, f"content type must be {wanted}")
        sent = self.headers.get("X-Plume-Token") or ""
        if not hmac.compare_digest(sent.encode(), csrf.encode()):
            raise _Refused(403, "bad token")
        if upload:
            self._video_upload(upload.group(1), parts.query)
            return
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
        elif op == "reveal":
            self._show_files(body)
        elif op.startswith("video/"):
            self._video_op(op, body)
        else:
            self._take_op(op, body)

    def _setup_view(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True).get("device", [])
        try:
            self._json(self.gui.takes.setup(values[0] if len(values) == 1 else None, check_output=True))
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
            elif op == "style":
                self._json(takes.choose_style(body))
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
            if refused.field is not None:
                doc["field"] = refused.field
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

    def _show_files(self, body: dict) -> None:
        """Show the take's folder, with its mix in it if one can be made.

        The body names a take by the ID this server gave out and nothing else, so the
        only folder that can open is one the library listed.
        """
        if set(body) != {"take"} or not isinstance(body["take"], str):
            raise _Refused(400, "body needs a take, and only a take")
        folder = self.gui.library.path_of(body["take"])
        if folder is None:
            raise _Refused(404, "no such take")
        why = None
        try:
            mix = self.gui.library.write_mix(body["take"])
        except MixInTheWay:
            mix, why = None, "theirs"
        except MixInUse:
            mix, why = None, "in_use"
        except (OSError, RuntimeError, ValueError, ArithmeticError):
            mix = None          # a folder that cannot be written in can still be shown
        if mix is None and why is None and is_own_mix(folder / REC_MIX):
            # No mix was made, and an older one of plume's is still here.  "No mix was
            # saved" would be true and would leave out the thing the person will find.
            why = "stale"
        try:
            # Always by a file inside it.  Asked for bare, the folder is shown as an item
            # in the folder it sits in, and the person is looking at the wrong place.
            self.gui.revealer(folder, REC_MIX if mix or why else _something_inside(folder))
        except (OSError, GuiError):
            raise _Refused(500, "could not open the folder") from None
        twice = mix is not None and band_may_be_twice(folder)
        self._json({"shown": True, "mix": mix is not None, "band_twice": twice, "why": why})

    # -- a video

    def _video_refuse(self, refused: VideoRefused) -> None:
        self.close_connection = True          # what was not read must not parse as a next request
        body = json.dumps({"error": "that video was not taken in", "why": refused.why}).encode()
        self._send(refused.status, body, "application/json; charset=utf-8", {"Connection": "close"})

    def _video_upload(self, take_id: str, query: str) -> None:
        """A video handed over for a take: the body is the file, and that is all it is.

        The query says whether to keep the video's sound and nothing else.  No name
        comes with it, and the take is one the library listed.
        """
        asked = parse_qs(query, keep_blank_values=True)
        if set(asked) != {"sound"} or asked["sound"] not in (["0"], ["1"]):
            raise _Refused(400, "say only whether to keep the sound")
        if self.headers.get("Transfer-Encoding"):
            raise _Refused(411, "the video's length is needed")
        sent = self.headers.get_all("Content-Length") or []
        if len(sent) != 1 or not sent[0].isascii() or not sent[0].isdigit() or int(sent[0]) <= 0:
            raise _Refused(411 if not sent else 400, "the video's length is needed")
        try:
            doc = self.gui.library.video_receive(take_id, self.rfile, int(sent[0]),
                                                 asked["sound"] == ["1"])
        except VideoRefused as refused:
            if (refused.status in (409, 507) and refused.why != "cancelled"
                    and int(sent[0]) <= _VIDEO_MAX_BYTES):
                # A browser still sending cannot read an answer: it sees the line drop, and
                # she is told nothing she can use.  So what is left is read, and let go of.
                left = int(sent[0])
                while left > 0:
                    chunk = self.rfile.read(min(_CHUNK_BYTES, left))
                    if not chunk:
                        break
                    left -= len(chunk)
            self._video_refuse(refused)
            return
        self._json(doc, 202)

    def _video_op(self, op: str, body: dict) -> None:
        library = self.gui.library
        wanted = {"take", "start_s", "sound"} if op == "video/place" else {"take"}
        if set(body) != wanted or not isinstance(body["take"], str):
            raise _Refused(400, "body needs a take" + (", a place and a choice" if len(wanted) > 1 else ", and only a take"))
        take_id = body["take"]
        try:
            if op == "video/place":
                start, sound = body["start_s"], body["sound"]
                if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start)
                        or not isinstance(sound, bool)):
                    raise _Refused(400, "a place is a number, and the choice is true or false")
                doc = library.video_place(take_id, float(start), sound)
                if doc is None:
                    raise _Refused(404, "no such take")
                self._json(doc, 202)
            elif op == "video/cancel":
                doc = library.video_cancel(take_id)
                if doc is None:
                    raise _Refused(404, "no such take")
                self._json(doc)
            else:
                folder = library.path_of(take_id)
                if folder is None or not takevideo.is_own_video(folder / REC_VIDEO):
                    raise _Refused(404, "no video for that take")
                try:
                    self.gui.revealer(folder, REC_VIDEO)
                except (OSError, GuiError):
                    raise _Refused(500, "could not open the folder") from None
                self._json({"shown": True})
        except VideoRefused as refused:
            self._json({"error": "that could not be done", "why": refused.why}, refused.status)

    def _video_preview(self, take_id: str, query: str) -> None:
        if self.gui.library.path_of(take_id) is None:
            raise _Refused(404, "no such take")
        values = parse_qs(query, keep_blank_values=True).get("at", [])
        if len(values) != 1 or not _NUMBER_RE.fullmatch(values[0]):
            raise _Refused(400, "a place is a number")
        try:
            wav = self.gui.library.video_preview(take_id, float(values[0]))
        except VideoRefused:
            raise _Refused(400, "that place is outside the video") from None
        if wav is None:
            raise _Refused(404, "nothing to listen to")
        self._send(200, wav, "audio/wav")

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
                fd = _open_in(where, _TRACKS[track], oscompat.read_flags())
            finally:
                _close_folder(where)
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
        fd = _open_in(where, name, oscompat.read_flags())
    finally:
        _close_folder(where)
    # Checked on the descriptor: a folder of that name opens, and then cannot be made
    # into a file object, which would leave the descriptor open for good.
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        os.close(fd)
        raise
    if not regular:
        os.close(fd)
        raise OSError("not a file")
    try:
        return os.fdopen(fd, "rb")
    except OSError:
        os.close(fd)
        raise


def _wav_header(frames: int, rate: int, *, extra: int = 0) -> bytes:
    """``extra`` is how many bytes of other chunks follow the samples."""
    return wav_header(frames, rate, extra=extra)


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
    band = 10.0 ** (_TOGETHER_BAND_DB / 20.0)
    mixed = (_read_mono(human, frame + shift, count) + _read_mono(machine, frame, count) * band) * _TOGETHER_GAIN
    # Sample by sample, so a range request reads the same bytes as the whole file.
    mixed = _fx.limiter(mixed, _TOGETHER_CEILING_DB)
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
                 spawner=None, lister=None, sessions=None, stopper=None, revealer=None,
                 ffmpeg=None, **take_options):
        self.library = Library(roots, pinned, ffmpeg)
        self.library.log = lambda line: self.log(line)
        #: ``(folder, file name or None)``: opens the system's file manager.  Tests pass a fake.
        self.revealer = _reveal if revealer is None else revealer
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
        # A run that died may have left a copy of a video in any take.  None waits for
        # her next video to that take to be cleared.
        self.library.sweep_leftovers()
        self.library.start_heartbeat()
        self._thread = threading.Thread(target=httpd.serve_forever, name="plume-gui",
                                        kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.takes.shutdown()
        self.library.close_videos()
        self.library.stop_heartbeat()
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


def guide_pages() -> list[dict]:
    """The guide as text, in reading order.  Words only: the page renders them with textContent."""
    return [{"id": name, "text": (_GUIDE_DIR / f"{name}.md").read_text(encoding="utf-8")}
            for name in GUIDE_PAGES]


def default_library() -> Path:
    return Path.home() / "Music" / "Plume"


class _Closing(BaseException):
    """SIGTERM, SIGHUP or SIGBREAK: leave the same way Ctrl-C does."""


def _closing_signal(signum, frame):  # noqa: ARG001
    raise _Closing


def _catch_closing_signals() -> dict:
    """A closed terminal must not orphan the take this bridge started."""
    previous = {}
    for name in oscompat.closing_signals():
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
            # In short waits, not one long one: on Windows a lock wait with no timeout
            # cannot be interrupted, so Ctrl-C and Ctrl-Break would never be handled.
            forever = threading.Event()
            while not forever.wait(0.25):
                pass
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
