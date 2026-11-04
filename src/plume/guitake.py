"""One take at a time for ``plume gui``: choose a device, start it, watch it, stop it.

The page opens no device and starts nothing until a person clicks Start.  That click
is the consent ``realtime.audio_output_allowed`` asks for, because the child has no
terminal to be asked through.  So :func:`default_spawner` is the one place here that
passes consent on, and only the Start route, behind the bridge's session and page key
checks, reaches it.  ``PLUME_ALLOW_AUDIO=0`` still wins.

Nothing a client sends becomes a flag.  A device is one of the server's own IDs, a
channel is a number in range, a key or style is parsed down to its canonical spelling,
and argv is a list that no shell ever reads.

Everything the take says about itself comes from what any watcher can read: the
session registry, the take folder, and ``watch.Listener``.  The bridge itself never
loads PortAudio.  ``plume inputs --json`` runs in a short-lived child, which is also
why a device plugged in after the bridge started still shows up.

A process that nobody can reach is the failure this file exists to avoid.  So the child
keeps its own 60 minute cap and still answers ``plume stop``, closing the bridge stops
the take it started, and a take that needs a person to end it stops itself when no page
has asked after it for :data:`HEARTBEAT_S`.

Everything here has only run against fakes.  Raw errors and stderr stay in a 0600 log
and are never sent to the page.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .compose import parse_prompt
from .realtime import (AUDIO_CONSENT_ENV, AUDIO_CONSENT_OFF, REC_CUES, REC_DECISIONS, REC_ENDING,
                       REC_HUMAN, REC_MACHINE, REC_MANIFEST, REC_MIDI, REC_STREAM, REC_VOICE,
                       LiveSession, RealtimeError, list_sessions, send_cue, stop_session)
from .styles_lib import list_styles
from .theory import parse_key, spell

# The page's words.  Nothing else about a failure ever reaches it.
DEVICE_MISSING = "Plume could not find that device. Check it is plugged in."
NO_INPUT = "Plume could not find a microphone or an audio interface."
OPEN_FAILED = "Plume could not open the device. Check it is set to 48 kHz."
AUDIO_PART_MISSING = "Plume needs its audio part: pip install with the play extra."
NO_LISTING = "Plume could not ask this computer for its audio devices."
SOUND_OFF = "This terminal has sound turned off."
TERMINAL_TAKE = "A take is already running from a terminal."
ALREADY_RUNNING = "A take is already running."
DID_NOT_START = "The take did not start."
NO_COUNT = ("Still waiting for a count. A knock may not reach a pickup, "
            "so try chucking muted strings.")
OPENING = "Still opening the device. This computer may be asking to use the microphone."
STALLED = "The audio device stopped answering."
ENDED_EARLY = "The take ended early."
NO_PAGE = "No page was open, so plume stopped this take."
WEDGED = "Still stopping. Run plume stop in a terminal."
BAD_KEY = "Plume did not understand that key. Try F major or A minor."
NO_HEAR_ME = "Hear me needs an audio interface. On a built-in mic it would feed back."

#: A take needing a person to end it, with no page asking this long, stops itself.
HEARTBEAT_S = 180.0
#: The recording grows at least once a second (REC_AUDIO_FLUSH_S) while the device runs.
STALL_S = 10.0
#: stop_session's ladder takes at most about 55 s.
WEDGED_S = 60.0
NO_COUNT_S = 60.0
FLAT_AFTER_S = 15.0
OPENING_AFTER_S = 15.0
#: Quieter than this all along means nothing is plugged into that channel.  A guess: no
#: real interface has been measured against it.
FLAT_DB = -70.0
_FLOOR_DB = -120.0
_LEVEL_WINDOW_S = 0.25

LIST_TIMEOUT_S = 5.0
SHUTDOWN_TIMEOUT_S = 70.0
_MAX_TEXT = 80
_LOG_TAIL_BYTES = 8192

CUES = ("busier", "sparser")
_FIELDS = frozenset({"device", "channel", "key", "style", "hear_me"})
#: A row may carry more than a page needs.  Only these fields leave the bridge.
_ROW_FIELDS = ("bar", "t_s", "bpm", "section", "stage", "key", "chord", "style", "style_guess",
               "key_guess", "you_db", "band_db", "you_busy", "you_notes", "you_source", "input",
               "you_trusted")
_STREAM_FIELDS = ("input_ms", "output_ms", "reported_ms", "block_ms", "calibrated_ms")

#: Software that shows up as a many-channel device.  "zoomaudio", not "zoom": Zoom also
#: make real interfaces.
_VIRTUAL = ("blackhole", "soundflower", "loopback", "aggregate", "zoomaudio", "zoom audio",
            "microsoft teams", "teams audio", "vb-cable", "virtual")
_BUILT_IN = ("built-in", "builtin", "macbook", "imac", "mac mini", "mac studio", "mac pro",
             "internal microphone")

#: What the child said, mapped to the page's words.  First match wins.
_FAILURES = (
    (("sounddevice is not available",), AUDIO_PART_MISSING),
    (("refusing to play live audio",), SOUND_OFF),
    (("another plume session",), TERMINAL_TAKE),
    (("no audio device matching", "has no inputs", "cannot form a duplex", "has channels 1 to",
      "choosing a channel", "no audio input device", "no audio output device",
      "audio device query failed"), DEVICE_MISSING),
    (("portaudio", "paerrorcode", "error opening stream", "invalid sample rate"), OPEN_FAILED),
)


class TakeError(RuntimeError):
    """A clean failure.  Its text is one of this file's own sentences."""


class SoundOff(TakeError):
    """``PLUME_ALLOW_AUDIO`` says no, and a click does not overrule that."""


class Refused(Exception):
    """A request this bridge turns down, and the state it was turned down in."""

    def __init__(self, status: int, message: str, *, live: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.live = status, message, live


# --------------------------------------------------------------- reaching the real world

def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def default_spawner(argv: list[str], log_path: Path):
    """Start the real take.  The one place a click becomes consent."""
    if _under_pytest():
        raise TakeError("the real spawner does not run under pytest")
    if os.environ.get(AUDIO_CONSENT_ENV, "").strip().lower() in AUDIO_CONSENT_OFF:
        raise SoundOff(SOUND_OFF)
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # O_NOFOLLOW: a planted symlink must not point stderr at something else.
    log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(log, 0o600)
        return subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
            env={**os.environ, AUDIO_CONSENT_ENV: "1"},
            # Its own session: a Ctrl-C in this terminal must not reach it as well as
            # us, or the band gets two interrupts, and a second one cuts its ending short.
            start_new_session=True)
    finally:
        os.close(log)


def default_lister() -> list[dict]:
    """What ``plume inputs --json`` says, from a child that opens no stream."""
    if _under_pytest():
        raise TakeError("the real device lister does not run under pytest")
    try:
        done = subprocess.run([sys.executable, "-m", "plume", "inputs", "--json"],
                              stdin=subprocess.DEVNULL, capture_output=True, check=False,
                              timeout=LIST_TIMEOUT_S, start_new_session=True)
        doc = json.loads(done.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise TakeError(NO_LISTING) from None
    if not isinstance(doc, dict):
        raise TakeError(NO_LISTING)
    if "sounddevice is not available" in str(doc.get("error", "")):
        raise TakeError(AUDIO_PART_MISSING)
    devices = doc.get("devices")
    if not isinstance(devices, list):
        raise TakeError(NO_LISTING)
    return devices


# ------------------------------------------------------------------- choosing a device

def _kind(name: str, inputs: int, outputs: int) -> str:
    low = name.lower()
    if any(mark in low for mark in _VIRTUAL):
        return "virtual"
    if any(mark in low for mark in _BUILT_IN):
        return "builtin"
    return "interface" if inputs >= 2 and outputs >= 2 else "other"


def describe_devices(listed) -> list[dict]:
    """The inputs a person can pick, each with an ID that is not a name.

    ``plume lead`` finds a device by substring and takes the first match, so a name that
    would land on an earlier device cannot be offered: choosing it would open the other.
    """
    rows = []
    for item in listed if isinstance(listed, list) else []:
        try:
            name, inputs, outputs = str(item["name"]), int(item["inputs"]), int(item["outputs"])
        except (KeyError, TypeError, ValueError):
            continue
        if name:
            rows.append({"name": name, "inputs": inputs, "outputs": outputs})
    out = []
    for row in rows:
        low = row["name"].lower()
        hits = [r for r in rows if low in r["name"].lower()]
        if row["inputs"] < 1 or next(r for r in hits if r["inputs"] > 0) is not row:
            continue
        kind = _kind(row["name"], row["inputs"], row["outputs"])
        # With --device the output is found by the same substring, and has to be this one too.
        if kind == "interface" and next(r for r in hits if r["outputs"] > 0) is not row:
            kind = "other"
        out.append({"id": hashlib.sha256(row["name"].encode()).hexdigest()[:16], **row, "kind": kind,
                    "mode": "direct" if kind == "interface" else "room",
                    "hear_me": kind == "interface"})
    return out


def default_device(devices: list[dict]) -> dict | None:
    """The only interface, or else the built-in mic.  None means the system default."""
    interfaces = [d for d in devices if d["kind"] == "interface"]
    if len(interfaces) == 1:
        return interfaces[0]
    return next((d for d in devices if d["kind"] == "builtin"), None)


@dataclass(frozen=True)
class Route:
    name: str | None        # None: the system default input, in room mode
    mode: str               # "direct" | "room"
    channel: int
    hear_me: bool


def _route(body: dict, devices: list[dict]) -> Route:
    wanted = body.get("device")
    if wanted is None:
        device = default_device(devices)
    else:
        device = next((d for d in devices if d["id"] == wanted), None)
        if device is None:
            raise Refused(400, DEVICE_MISSING)
    channel = body.get("channel")
    if channel is None:
        channel = 1
    most = 1 if device is None else device["inputs"]
    if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= most:
        raise Refused(400, f"That device has channels 1 to {most}.")
    mode = "room" if device is None else device["mode"]
    hear_me = body.get("hear_me")
    if hear_me is None:
        hear_me = mode == "direct"
    if not isinstance(hear_me, bool):
        raise Refused(400, "hear_me must be true or false")
    if hear_me and mode != "direct":
        raise Refused(400, NO_HEAR_ME)
    return Route(None if device is None else device["name"], mode, channel, hear_me)


# ------------------------------------------------------------- free text, made canonical

def _is_detect(text: str) -> bool:
    return text.strip().lower() in ("", "detect")


def canonical_key(text: str) -> str | None:
    """"F major" from whatever was typed, None for detect.  Only this ever reaches argv."""
    if _is_detect(text):
        return None
    text = text.strip()
    named = parse_prompt(f"in {text}").key
    try:
        if named is None:            # 'Fm' or a bare 'F': the prompt grammar needs a mode word
            key = parse_key(text)
            named = f"{spell(key.tonic, key)} {key.mode.replace('_', ' ')}"
        parse_key(named)
    except ValueError:
        raise Refused(400, BAD_KEY) from None
    if parse_prompt(f"in {named}").key != named:
        raise Refused(400, BAD_KEY)
    return named


def canonical_style(text: str) -> str | None:
    """A style's own name from whatever was typed, None for detect."""
    if _is_detect(text):
        return None
    known = list_styles()
    styles = parse_prompt(text).styles
    if not styles or styles[0] not in known:
        raise Refused(400, f"Plume did not understand that style. Try {', '.join(known)}.")
    return styles[0]


def _text_field(body: dict, name: str) -> str:
    value = body.get(name)
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > _MAX_TEXT:
        raise Refused(400, f"{name} must be short text")
    return value


def build_argv(route: Route, folder: Path, *, key: str | None, style: str | None,
               seed: int) -> list[str]:
    """Every value is one token, glued to its flag, so none can read as a flag of its own.

    No ``--max-minutes``: the take keeps the session cap.  After ``--`` a prompt is only
    ever a prompt.
    """
    argv = [sys.executable, "-m", "plume", "lead", "--tap-start", "--detect",
            f"--record={folder}/", f"--seed={int(seed)}"]
    if route.mode == "direct":
        argv += [f"--device={route.name}", f"--input-channel={route.channel}", "--direct"]
        if route.hear_me:
            argv.append("--monitor")
    elif route.name is not None:
        argv.append(f"--input-device={route.name}")
        if route.channel != 1:
            argv.append(f"--input-channel={route.channel}")
    words = [w for w in (style, f"in {key}" if key else None) if w]
    if words:
        argv += ["--", " ".join(words)]
    return argv


# ---------------------------------------------------------------------- the take folder

def pretty_path(path: Path) -> str:
    home, where = str(Path.home()), str(path)
    return "~" + where[len(home):] if where.startswith(home + os.sep) else where


def make_take_folder(root: Path, now: datetime) -> Path:
    """A new folder, by exclusive mkdir, so a used one can never be recorded over."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    for n in range(1, 1000):
        path = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        return path
    raise FileExistsError(str(root))


def _frames(path: Path) -> int:
    import soundfile as sf  # noqa: PLC0415 - the bridge need not load it to read a shelf

    try:
        with sf.SoundFile(str(path)) as f:
            return int(f.frames)
    except (OSError, RuntimeError, ValueError):
        return 0


def _tail_level(path: Path) -> tuple[int, float | None]:
    """(frames so far, level of the last moment in dB).  Reads only what is on disk."""
    import soundfile as sf  # noqa: PLC0415

    try:
        with sf.SoundFile(str(path)) as f:
            frames = int(f.frames)
            count = min(frames, int(_LEVEL_WINDOW_S * f.samplerate))
            if count <= 0:
                return frames, None
            f.seek(frames - count)
            y = f.read(count, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return 0, None
    if y.size == 0:
        return frames, None
    power = float((y.astype("float64") ** 2).mean())
    return frames, round(max(_FLOOR_DB, 10.0 * math.log10(power)) if power > 1e-12 else _FLOOR_DB, 1)


def _clear_fresh_folder(folder: Path) -> None:
    """A take that made no sound leaves no folder on the shelf.

    Only names a take writes, in a folder this run made, and only if they hold nothing:
    anything else stops the rmdir, which is the point.
    """
    wavs = (REC_HUMAN, REC_MACHINE, REC_VOICE)
    logs = (REC_DECISIONS, REC_CUES, REC_MIDI)
    for name in (*wavs, *logs, REC_MANIFEST, REC_ENDING, REC_STREAM):
        path = folder / name
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if (name in wavs and _frames(path) > 0) or (name in logs and path.stat().st_size > 0):
                continue
            path.unlink()
        except OSError:
            pass
    try:
        folder.rmdir()
    except OSError:
        pass


def classify_failure(log_path: Path) -> str:
    """The page's sentence for what the child wrote.  Never the words it wrote."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - _LOG_TAIL_BYTES))
            text = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return DID_NOT_START
    for marks, message in _FAILURES:
        if any(mark in text for mark in marks):
            return message
    return DID_NOT_START


def _finite(value):
    """JSON has no NaN, and a reading must not cost the whole live view."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


# ----------------------------------------------------------------------------- the take

@dataclass
class _Take:
    proc: object
    folder: Path
    route: Route
    key: str | None
    style: str | None
    started: float
    phase: str = "starting"          # "starting" | "waiting" | "playing"
    opened: float | None = None
    frames: int = 0
    size: int = -1
    grew: float = 0.0
    loud: bool = False
    level_db: float | None = None
    row: dict | None = None
    stream: dict | None = None
    hint: str | None = None
    stop_asked: bool = False
    stop_since: float = 0.0
    note: str | None = None
    listener: object = None


class TakeRunner:
    """The state of the one take this bridge may be running, for every page that asks."""

    def __init__(self, root, *, log_path, id_of, spawner=None, lister=None, sessions=None,
                 stopper=None, cue=None, listener=None, clock=time.monotonic, now=datetime.now,
                 tick_s: float = 0.5, watch: bool = True):
        self.root = Path(root)
        self._log_path = log_path
        self._id_of = id_of
        self._spawn = default_spawner if spawner is None else spawner
        self._list = default_lister if lister is None else lister
        self._sessions = (lambda: list_sessions(prune=False)) if sessions is None else sessions
        self._stopper = stop_session if stopper is None else stopper
        self._cue = send_cue if cue is None else cue
        self._listener = listener
        self._clock = clock
        self._now = now
        self._tick_s = float(tick_s)
        self._watching = bool(watch)
        self._lock = threading.Lock()
        self._list_lock = threading.Lock()
        self._take: _Take | None = None
        self._result: dict | None = None
        self._preparing = False
        self._cancelled = False
        self._closing = False
        self._last_poll = clock()
        self._last_stream: dict | None = None
        self._stopping_others: set[int] = set()

    # -- what a page reads

    def home(self) -> dict:
        problem = None
        try:
            devices = self._devices()
        except TakeError as e:
            devices, problem = [], str(e)
        chosen = default_device(devices)
        if problem is None and not devices:
            problem = NO_INPUT
        with self._lock:
            stream = self._last_stream
        return {
            "devices": devices,
            "device": None if chosen is None else chosen["id"],
            "mode": "room" if chosen is None else chosen["mode"],
            "channel": 1,
            "hear_me": {"allowed": chosen is not None and chosen["hear_me"],
                        "default": chosen is not None and chosen["hear_me"],
                        "reported_ms": None if stream is None else stream.get("reported_ms")},
            "folder": pretty_path(self.root),
            "styles": list_styles(),
            "problem": problem,
        }

    def live(self, *, poll: bool = True) -> dict:
        """The state document.  A page asking is also what proves a page is still there."""
        with self._lock:
            if poll:
                self._last_poll = self._clock()
            doc = self._doc()
            quiet = self._take is None and not self._preparing
        if quiet:
            others = self._others()
            if others:
                return self._elsewhere(others)
        return doc

    def _doc(self) -> dict:
        take = self._take
        if take is None:
            if self._preparing:
                return {"state": "starting"}
            return dict(self._result or {"state": "idle"})
        doc = {
            "state": "stopping" if take.stop_asked else take.phase,
            "device": take.route.name, "mode": take.route.mode,
            "channel": take.route.channel, "hear_me": take.route.hear_me,
            "key": take.key, "style": take.style,
            "level_db": take.level_db, "row": take.row, "stream": take.stream,
            "hint": take.hint,
        }
        if take.stop_asked:
            doc["message"] = WEDGED if self._clock() - take.stop_since >= WEDGED_S else None
        return doc

    def _elsewhere(self, others: list) -> dict:
        with self._lock:
            stopping = others[0].pid in self._stopping_others
        return {"state": "elsewhere", "pid": others[0].pid, "message": TERMINAL_TAKE,
                "stopping": stopping}

    def _others(self) -> list:
        with self._lock:
            mine = None if self._take is None else self._take.proc.pid
        try:
            return [r for r in self._sessions() if r.pid != mine]
        except OSError:
            return []

    def _devices(self) -> list[dict]:
        # One child at a time, however many pages are open.
        with self._list_lock:
            return describe_devices(self._list())

    # -- start

    def start(self, body: dict) -> dict:
        if not isinstance(body, dict) or set(body) - _FIELDS:
            raise Refused(400, "That is not something Start understands.")
        with self._lock:
            if self._closing:
                raise Refused(409, "plume gui is closing.", live=self._doc())
            if self._take is not None or self._preparing:
                raise Refused(409, ALREADY_RUNNING, live=self._doc())
            self._preparing = True
            self._cancelled = False
        try:
            return self._start(body)
        except BaseException:
            with self._lock:
                self._preparing = False
            raise

    def _start(self, body: dict) -> dict:
        others = self._others()
        if others:
            raise Refused(409, TERMINAL_TAKE, live=self._elsewhere(others))
        key = canonical_key(_text_field(body, "key"))
        style = canonical_style(_text_field(body, "style"))
        try:
            devices = self._devices()
        except TakeError as e:
            return self._failed(str(e))
        if not devices:
            return self._failed(NO_INPUT)
        route = _route(body, devices)
        try:
            folder = make_take_folder(self.root, self._now())
        except OSError:
            return self._failed(f"Could not make a folder in {pretty_path(self.root)}.")
        argv = build_argv(route, folder, key=key, style=style, seed=secrets.randbelow(100000))
        with self._lock:
            if self._cancelled:          # Stop was clicked while the devices were being listed
                self._preparing = False
                self._result = None
                cancelled = True
            else:
                cancelled = False
        if cancelled:
            _clear_fresh_folder(folder)
            return self.live(poll=False)
        try:
            proc = self._spawn(argv, Path(self._log_path()))
        except SoundOff:
            _clear_fresh_folder(folder)
            return self._failed(SOUND_OFF)
        except (TakeError, OSError, ValueError):
            _clear_fresh_folder(folder)
            return self._failed(DID_NOT_START)
        now = self._clock()
        take = _Take(proc, folder, route, key, style, started=now, grew=now)
        with self._lock:
            self._take = take
            self._preparing = False
            self._result = None
            self._last_poll = now
            cancelled = self._cancelled
            doc = self._doc()
        if cancelled:
            self._ask_stop(take)
        if self._watching:
            threading.Thread(target=self._watch, args=(take,), name="plume-gui-take",
                             daemon=True).start()
        return doc

    def _failed(self, message: str) -> dict:
        with self._lock:
            self._preparing = False
            self._result = {"state": "failed", "message": message}
            return dict(self._result)

    # -- stop

    def stop(self) -> dict:
        with self._lock:
            take = self._take
            if take is None and self._preparing:
                self._cancelled = True
                return {"state": "starting"}
        if take is not None:
            self._ask_stop(take)
            return self.live(poll=False)
        others = self._others()
        if others:
            self._stop_other(others[0])
        return self.live(poll=False)

    def _ask_stop(self, take: _Take, note: str | None = None) -> None:
        """Once only.  A second SIGINT would arm the session's hard exit and cut its ending."""
        with self._lock:
            if take.stop_asked:
                return
            take.stop_asked = True
            take.stop_since = self._clock()
            take.note = note
        threading.Thread(target=self._stop_child, args=(take,), name="plume-gui-stop",
                         daemon=True).start()

    def _record_of(self, pid: int):
        try:
            return next((r for r in self._sessions() if r.pid == pid), None)
        except OSError:
            return None

    def _stop_child(self, take: _Take) -> None:
        # No record yet means it is still starting: the ladder goes straight to SIGINT,
        # and the pid is safe to signal, because an unreaped child's pid cannot be reused.
        rec = self._record_of(take.proc.pid) or LiveSession(
            pid=take.proc.pid, kind="lead", record_dir=str(take.folder))
        try:
            self._stopper(rec)
        except Exception:  # noqa: BLE001 - the page says "still stopping" from the clock
            pass

    def _stop_other(self, rec) -> None:
        with self._lock:
            if rec.pid in self._stopping_others:
                return
            self._stopping_others.add(rec.pid)

        def run():
            try:
                self._stopper(rec)
            except Exception:  # noqa: BLE001
                pass
            finally:
                with self._lock:
                    self._stopping_others.discard(rec.pid)

        threading.Thread(target=run, name="plume-gui-stop", daemon=True).start()

    def shutdown(self, *, timeout: float = SHUTDOWN_TIMEOUT_S) -> bool:
        """Stop the take this bridge started, and wait.  True if nothing is left running.

        Only the first call waits, so one that gave up, or was interrupted, is not repeated
        on the way out.
        """
        with self._lock:
            waited_before = self._closing
            self._closing = True
            self._cancelled = True
            take = self._take
        if take is None:
            return True
        self._ask_stop(take)
        until = time.monotonic() + (0.0 if waited_before else float(timeout))
        while take.proc.poll() is None:
            if time.monotonic() >= until:
                return False
            time.sleep(0.02)
        self.tick()
        return True

    # -- cues

    def cue(self, body: dict) -> dict:
        cue = body.get("cue") if isinstance(body, dict) else None
        if not isinstance(cue, str) or cue not in CUES:
            raise Refused(400, "That is not a cue the page offers.")
        with self._lock:
            take = self._take
            ok = take is not None and take.phase == "playing" and not take.stop_asked
            doc = self._doc()
        if not ok:
            raise Refused(409, "No take is playing.", live=doc)
        rec = self._record_of(take.proc.pid)
        if rec is None or not rec.socket:
            raise Refused(409, "The band did not answer that cue.", live=doc)
        try:
            self._cue(cue, socket_path=rec.socket, source="gui")
        except RealtimeError:
            raise Refused(409, "The band did not answer that cue.", live=doc) from None
        return {"ok": True, "cue": cue}

    # -- watching it

    def _watch(self, take: _Take) -> None:
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a bad read must not leave a take unwatched
                pass
            with self._lock:
                if self._take is not take:
                    return
            time.sleep(self._tick_s)

    def tick(self) -> None:
        """One look at the child and its folder."""
        with self._lock:
            take = self._take
        if take is None:
            return
        code = take.proc.poll()
        if code is not None:
            self._land(take, code)
            return
        now = self._clock()
        grew, level = self._read_human(take)
        row = self._read_row(take)
        stream = None if take.stream is not None else self._read_stream(take)
        ending = (take.folder / REC_ENDING).exists()
        played = self._has_rows(take)
        stop_note = None
        ask = False
        with self._lock:
            if grew:
                take.grew = now
            if level is not None:
                take.level_db = level
                take.loud = take.loud or level > FLAT_DB
            if row is not None:
                take.row = row
            if stream is not None:
                take.stream = self._last_stream = stream
            if take.phase == "starting" and take.frames > 0:
                take.phase, take.opened = "waiting", now
            if take.phase == "waiting" and played:
                take.phase = "playing"
            take.hint = self._hint(take, now)
            if not take.stop_asked and not ending:
                unattended = take.phase != "playing" or take.route.mode == "room"
                if unattended and now - self._last_poll >= HEARTBEAT_S:
                    ask, stop_note = True, NO_PAGE
                elif take.phase != "starting" and now - take.grew >= STALL_S:
                    ask, stop_note = True, STALLED
        if ask:
            self._ask_stop(take, stop_note)

    @staticmethod
    def _hint(take: _Take, now: float) -> str | None:
        if take.phase == "starting":
            return OPENING if now - take.started >= OPENING_AFTER_S else None
        if take.phase != "waiting":
            return None
        waited = now - take.opened
        if not take.loud and waited >= FLAT_AFTER_S:
            return f"Nothing on channel {take.route.channel}. Try another channel in the menu."
        return NO_COUNT if waited >= NO_COUNT_S else None

    @staticmethod
    def _read_human(take: _Take) -> tuple[bool, float | None]:
        """(did the recording grow, how loud is its newest moment)."""
        path = take.folder / REC_HUMAN
        try:
            size = path.stat().st_size
        except OSError:
            return False, None
        if size == take.size:
            return False, None
        take.size = size
        take.frames, level = _tail_level(path)
        return take.frames > 0, level

    @staticmethod
    def _has_rows(take: _Take) -> bool:
        try:
            return (take.folder / REC_DECISIONS).stat().st_size > 0
        except OSError:
            return False

    def _read_row(self, take: _Take) -> dict | None:
        if not self._has_rows(take):
            return None
        try:
            if take.listener is None:
                if self._listener is None:
                    from .watch import Listener  # noqa: PLC0415 - heavy, and only a take needs it
                    self._listener = Listener
                take.listener = self._listener(take.folder, every=1)
            rows = take.listener.poll()
        except Exception:  # noqa: BLE001 - a view of the take must never hurt the take
            return None
        if not rows:
            return None
        return {name: _finite(rows[-1].get(name)) for name in _ROW_FIELDS}

    @staticmethod
    def _read_stream(take: _Take) -> dict | None:
        try:
            doc = json.loads((take.folder / REC_STREAM).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict):
            return None
        return {name: (doc[name] if isinstance(doc.get(name), (int, float))
                       and not isinstance(doc.get(name), bool)
                       and math.isfinite(doc[name]) else None)
                for name in _STREAM_FIELDS}

    def _land(self, take: _Take, code: int) -> None:
        """It has gone.  Say where that left things, once."""
        folder = take.folder
        made_sound = _frames(folder / REC_HUMAN) > 0 or _frames(folder / REC_MACHINE) > 0
        result = None
        if made_sound:
            try:
                take_id = self._id_of(folder)
            except Exception:  # noqa: BLE001 - it has ended, and the page must hear that
                take_id = None
            if take_id is not None:
                note = take.note
                if note is None and code != 0 and not take.stop_asked:
                    note = ENDED_EARLY
                result = {"state": "done", "take": take_id, "note": note}
        if result is None:
            _clear_fresh_folder(folder)
            if take.stop_asked:
                result = {"state": "idle"}
            else:
                result = {"state": "failed", "message": classify_failure(Path(self._log_path()))}
        with self._lock:
            if self._take is take:
                self._take = None
                self._result = result
