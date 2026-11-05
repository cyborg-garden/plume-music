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

The first press of Start is the setup: one line at a time, and the answers stay in a small
0600 file in the state folder, keyed by device ID, so it holds no device's name.  Listening
is the only part that opens anything: a child of its own, only from a POST the page makes on
a click, for a few seconds, closed before the next step.  A setup left halfway costs nothing,
because Start always works, with Hear me off.  Plume cannot tell headphones from speakers, so
Hear me takes a person's yes, never a device's name.

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
import re
import secrets
import stat
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
from .theory import parse_key

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
FLAT_CHANNEL = "Nothing on channel {channel}. Press Stop, then try another channel in the menu."
FLAT_CHANNEL_HOME = "Nothing came in on channel {channel}. Try another channel."
FLAT_MIC = ("Plume hears nothing. Check this computer lets it use the microphone, "
            "or press Stop and pick another device.")
FLAT_MIC_HOME = "Plume heard nothing. Check this computer lets it use the microphone, or pick another device."
KEPT_NOTHING = "The band had not come in yet, so plume kept nothing."
BAD_KEY = "Plume did not understand that key. Try F major or A minor."
ONLY_MAJOR_MINOR = "Plume plays major and minor keys. Try F major or A minor."
NO_HEAR_ME = "Hear me needs headphones on. Through speakers it would feed back."
NO_READING = "Plume could not read that device. Try again, or just start."
STILL_LISTENING = "Plume is still closing the device. Try again in a moment."
PLAY_SOMETHING = "Play something."
HOT_GAIN = "A bit hot. Turn the gain down a little and play again."
LOW_GAIN = "Barely hearing you. Turn the gain up a little and play again."
#: A mic like this has no gain to turn.
HOT_ROOM = "A bit hot. Move back a little and play again."
LOW_ROOM = "Barely hearing you. Move closer and play again."
NOTHING_GAIN = "Nothing came in on {name}. Check the cable and the gain, then play again."
NOTHING_ROOM = "Nothing came in on {name}. Check this computer lets plume use the microphone."
LATE_GUESS = ("Try wired headphones. This output looks wireless (a guess), "
              "and wireless ones can arrive late.")
LATE_REPORTED = ("Try wired headphones. This output reports about {ms} ms, "
                 "and wireless ones can arrive late.")
COUNT_LINE = "Knock or clap 8 even beats."
COUNT_PICKUP = "A knock may not reach a pickup, so chuck muted strings."

#: A take needing a person to end it, with no page asking this long, stops itself.
HEARTBEAT_S = 180.0
#: The recording grows at least once a second (REC_AUDIO_FLUSH_S) while the device runs.
STALL_S = 10.0
#: stop_session's ladder takes at most about 55 s.
WEDGED_S = 60.0
NO_COUNT_S = 60.0
FLAT_AFTER_S = 15.0
#: Stopped before the band came in, this much recording with real sound in it is still hers.
#: Anything less is a false start, and leaves nothing on the shelf.
KEEP_UNCOUNTED_S = 60.0
OPENING_AFTER_S = 15.0
#: Quieter than this all along means nothing is plugged into that channel.  A guess: no
#: real interface has been measured against it.
FLAT_DB = -70.0
_FLOOR_DB = -120.0
_LEVEL_WINDOW_S = 0.25

#: How long the first step listens.  Long enough to play something, short enough to wait for.
LISTEN_S = 4.0
#: On top of that.  A computer asking about the microphone can hold the open for a while.
METER_GRACE_S = 15.0
#: How long leaving waits for a child to let go of its device.
RELEASE_S = 3.0
#: Two interfaces get listened to at once, so she plays once.  Each is its own child.
MAX_LISTEN_DEVICES = 4
#: Peaks.  docs/DIRECT-INPUT-TEST.md asks for about -18 to -6 dB, and 0 dB is clipping.
#: Guesses: no real interface has been measured against them.  LOW_DB is where a guitar's rms
#: drops under plume's -45 dB silence floor, and NOTHING_DB sits with FLAT_DB's rms.
HOT_DB = -1.0
LOW_DB = -33.0
NOTHING_DB = -60.0
#: A yes or no in setup can use a look at the devices this recent.  Each look is a child, and a
#: second's wait after every answer is what would make this feel like a form.  Only the answer
#: about headphones, and the one that finishes setup, wait for a fresh look.
LOOK_RECENT_S = 20.0
#: An output this late is hard to play to.  A guess too: wired ones report far less than this.
#: The output's own number: a stream's reported_ms adds the input's to it.
LATE_MS = 80.0
SETUP_VERSION = 1

LIST_TIMEOUT_S = 5.0
SHUTDOWN_TIMEOUT_S = 70.0
_MAX_TEXT = 80
_LOG_TAIL_BYTES = 8192

CUES = ("busier", "sparser")
#: Runs this plume whatever folder the bridge was started from: without -P, ``-m`` puts that
#: folder first on the path, and a plume.py in it would run with the click's consent.
PLUME_CHILD = (sys.executable, "-P", "-m", "plume")
_TYPED_TONIC = re.compile(r"[A-Ga-g][#sb]?")
#: No word about headphones: a page can be hours old, on other gear.  That is said to setup, and
#: saved there, where a change in what is plugged in can take it back.
_FIELDS = frozenset({"device", "channel", "key", "style", "hear_me"})
#: A row may carry more than a page needs.  Only these fields leave the bridge.
_ROW_FIELDS = ("bar", "t_s", "bpm", "section", "stage", "key", "chord", "style", "style_guess",
               "key_guess", "you_db", "band_db", "you_busy", "you_notes", "you_source", "input",
               "you_trusted", "playing", "playing_trusted", "fade_db")
_STREAM_FIELDS = ("input_ms", "output_ms", "reported_ms", "block_ms", "calibrated_ms")

#: Software that shows up as a many-channel device.  "zoomaudio", not "zoom": Zoom also
#: make real interfaces.
_VIRTUAL = ("blackhole", "soundflower", "loopback", "aggregate", "zoomaudio", "zoom audio",
            "microsoft teams", "teams audio", "vb-cable", "voicemeeter", "virtual", "monitor of",
            "null output", "dummy")
#: The computer's own, or what comes with it: a mic and speakers in one box, which would hear
#: the band.  Any computer's words, not one maker's.  A name not here is taken at its word, and
#: being wrong is caught by hearing nothing at setup, or by the take's own routing check.
_BUILT_IN = ("built-in", "builtin", "internal", "microphone array", "mic array", "laptop", "notebook",
             "macbook", "imac", "mac mini", "mac studio", "mac pro", "hda intel", "realtek",
             "conexant", "smart sound", "webcam", "camera", "facetime", "display", "hdmi", "dock",
             "iphone", "ipad")
#: Not a card at all: the sound system's own names for whatever is current.
_SYSTEM_NAMES = frozenset({"default", "sysdefault", "pulse", "pipewire", "jack"})
#: A mic, or wireless: the band belongs on the system's output, and it hears the room.
_ROOM_ONLY = re.compile(r"\b(?:mic|microphone)\b|bluetooth|wireless|airpods|buds|hands-?free",
                        re.IGNORECASE)
_WIRELESS = re.compile(r"bluetooth|wireless|airpods|buds|hands-?free", re.IGNORECASE)

#: What the child said, mapped to the page's words.  First match wins.
_FAILURES = (
    (("sounddevice is not available",), AUDIO_PART_MISSING),
    (("refusing to play live audio", "refusing to open a live input"), SOUND_OFF),
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
        done = subprocess.run([*PLUME_CHILD, "inputs", "--json"],
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


def default_meter(name: str, seconds: float, cancelled: threading.Event) -> dict | None:
    """What ``plume inputs --meter`` heard, from a child that makes no sound.

    The click that reached this is the consent, the same as Start's.  None means it was
    called off, and the child is gone by the time this returns either way.
    """
    if _under_pytest():
        raise TakeError("the real meter does not run under pytest")
    if os.environ.get(AUDIO_CONSENT_ENV, "").strip().lower() in AUDIO_CONSENT_OFF:
        raise SoundOff(SOUND_OFF)
    if cancelled.is_set():
        return None
    try:
        proc = subprocess.Popen(
            # Glued to its flag, so a name cannot read as a flag of its own.
            [*PLUME_CHILD, "inputs", f"--meter={name}", f"--seconds={seconds}", "--json"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, AUDIO_CONSENT_ENV: "1"}, start_new_session=True)
    except OSError:
        raise TakeError(NO_READING) from None
    until = time.monotonic() + float(seconds) + METER_GRACE_S
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancelled.is_set():
                    return None
                if time.monotonic() >= until:
                    raise TakeError(OPENING) from None
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    try:
        doc = json.loads(out)
    except ValueError:
        raise TakeError(_failure_words(err.decode("utf-8", "replace"), NO_READING)) from None
    if isinstance(doc, dict) and "error" in doc:
        raise TakeError(_failure_words(str(doc["error"]), NO_READING))
    return doc


# ------------------------------------------------------------------- choosing a device

def _kind(name: str, inputs: int, outputs: int) -> str:
    """No list of makers: inputs, not software, and not the computer's own is an interface.

    With no output of its own it cannot be ``--device``, and a mic or a wireless headset would
    take the band away from where she is listening, so those stay room mics.
    """
    low = name.lower()
    if any(mark in low for mark in _VIRTUAL):
        return "virtual"
    if low in _SYSTEM_NAMES or any(mark in low for mark in _BUILT_IN):
        return "builtin"
    return "interface" if inputs >= 1 and outputs >= 1 and not _ROOM_ONLY.search(name) else "other"


def _rows(listed) -> list[dict]:
    """A listing, down to the fields that make sense.  Anything else is dropped here."""
    rows = []
    for item in listed if isinstance(listed, list) else []:
        try:
            name, inputs, outputs = str(item["name"]), int(item["inputs"]), int(item["outputs"])
        except (KeyError, TypeError, ValueError):
            continue
        if not name:
            continue
        row = {"name": name, "inputs": inputs, "outputs": outputs}
        ms = item.get("output_ms")
        if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(ms) and ms > 0:
            row["output_ms"] = float(ms)
        if item.get("default_output") is True:
            row["default_output"] = True
        rows.append(row)
    return rows


def device_list_mark(rows: list[dict]) -> str:
    """Changes when what is plugged in changes, or where the system sends its sound.

    Software coming and going does not count.
    """
    real = sorted((r["name"], r["inputs"], r["outputs"], r.get("default_output", False))
                  for r in rows if _kind(r["name"], r["inputs"], r["outputs"]) != "virtual")
    return hashlib.sha256(json.dumps(real).encode()).hexdigest()[:16]


def describe_devices(listed) -> list[dict]:
    """The inputs a person can pick, each with an ID that is not a name.

    ``plume lead`` finds a device by substring and takes the first match, so a name that
    would land on an earlier device cannot be offered: choosing it would open the other.
    ``hear_me`` is only ever a person's answer, so it starts False for every name.
    """
    rows = _rows(listed)
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
        out.append({"id": hashlib.sha256(row["name"].encode()).hexdigest()[:16], "name": row["name"],
                    "inputs": row["inputs"], "outputs": row["outputs"], "kind": kind,
                    "mode": "direct" if kind == "interface" else "room", "hear_me": False})
    return out


def default_device(devices: list[dict], preferred: str | None = None) -> dict | None:
    """The one setup heard her on, or the only interface, or else the built-in mic.

    None means the system default.
    """
    if preferred is not None:
        kept = next((d for d in devices if d["id"] == preferred), None)
        if kept is not None:
            return kept
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
    channels: int = 1       # how many this device has, so what the page can offer her


def _route(body: dict, devices: list[dict], *, preferred=None, saved_for=lambda device: {}) -> tuple:
    """(the route, the device it came from).  What setup learned fills in what she did not say.

    Hear me needs a yes about headphones, saved for this device as things are now.  Plume cannot
    tell headphones from speakers, and a stale page must not cause feedback.
    """
    wanted = body.get("device")
    if wanted is None:
        device = default_device(devices, preferred)
    else:
        device = next((d for d in devices if d["id"] == wanted), None)
        if device is None:
            raise Refused(400, DEVICE_MISSING)
    saved = saved_for(device)
    most = 1 if device is None else device["inputs"]
    channel = body.get("channel")
    if channel is None:
        channel = _saved_channel(device, saved)
    if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= most:
        raise Refused(400, f"That device has channels 1 to {most}.")
    mode = "room" if device is None else device["mode"]
    hear_me, headphones = body.get("hear_me"), saved.get("headphones")
    if hear_me is not None and not isinstance(hear_me, bool):
        raise Refused(400, "hear_me must be true or false")
    if hear_me is None:
        hear_me = saved.get("hear_me") is True and headphones is True
    if hear_me and headphones is not True:
        raise Refused(400, NO_HEAR_ME)
    return Route(None if device is None else device["name"], mode, channel, hear_me, most), device


# --------------------------------------------------------------------------------- setup

_ANSWERS = {"headphones": ("yes", "no"), "hears_self": ("yes", "no"), "volume": ("fixed", "still"),
            "latency": ("ok",), "hear_me": ("on", "off")}
_YES_NO = [{"answer": "yes", "label": "Yes"}, {"answer": "no", "label": "No"}]
_STEPS = {
    "headphones": ("Headphones on?", _YES_NO),
    "hears_self": ("Can you hear yourself?", _YES_NO),
    "volume": ("Check your headphone volume.", [{"answer": "fixed", "label": "That was it"},
                                                {"answer": "still", "label": "Still nothing"}]),
}
#: After a level needs a hand or nothing came in: she is never kept here.
_TRY_OR_START = [{"answer": "again", "label": "Play again"}, {"answer": "start", "label": "Start anyway"}]
_DEVICE_KEY = re.compile(r"[0-9a-f]{16}|default")
_ROOM_ANSWERS = ("headphones", "hears_self", "still", "hear_me", "late_said")


def _setup_key(device: dict | None) -> str:
    return "default" if device is None else device["id"]


def _clean_entry(entry) -> dict:
    """Only fields this file wrote, of the right type.  A hand-edited file earns no Hear me."""
    if not isinstance(entry, dict):
        return {}
    out = {}
    if isinstance(entry.get("list"), str):
        out["list"] = entry["list"]
    channel = entry.get("channel")
    if isinstance(channel, int) and not isinstance(channel, bool) and channel >= 1:
        out["channel"] = channel
    if entry.get("level") in ("ok", "hot", "low"):
        out["level"] = entry["level"]
    for name in ("headphones", "hears_self", "still", "hear_me", "late_said"):
        if isinstance(entry.get(name), bool):
            out[name] = entry[name]
    if out.get("headphones") is not True:
        out.pop("hear_me", None)
    return out


class SetupStore:
    """The answers, per device, in one small file.  It is fine for this to fail.

    A file that cannot be read is a first run, and one that cannot be written leaves the
    answers in memory for as long as this bridge lasts.
    """

    def __init__(self, path_of=None):
        self._path_of = path_of
        self._lock = threading.Lock()
        self._doc: dict | None = None

    def _read(self) -> dict:
        empty = {"last": None, "devices": {}}
        if self._path_of is None:
            return empty
        try:
            # O_NOFOLLOW and O_NONBLOCK: a planted symlink or a FIFO is not this file.
            fd = os.open(self._path_of(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return empty
        try:
            with os.fdopen(fd, encoding="utf-8") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    return empty
                doc = json.load(f)
        except (OSError, ValueError):
            return empty
        if not isinstance(doc, dict) or doc.get("version") != SETUP_VERSION:
            return empty
        devices = doc.get("devices")
        if not isinstance(devices, dict):
            return empty
        kept = {k: _clean_entry(v) for k, v in devices.items()
                if isinstance(k, str) and _DEVICE_KEY.fullmatch(k)}
        last = doc.get("last")
        return {"last": last if last in kept else None, "devices": kept}

    def _loaded(self) -> dict:
        if self._doc is None:
            self._doc = self._read()
        return self._doc

    def _write(self) -> None:
        if self._path_of is None:
            return
        try:
            path = Path(self._path_of())
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(path.parent, 0o700)
            # A name nobody could have planted, and a rename that replaces a symlink rather
            # than writing through it.
            tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"version": SETUP_VERSION, **self._doc}))
                os.replace(tmp, path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass

    def last(self) -> str | None:
        with self._lock:
            return self._loaded()["last"]

    def entry(self, key: str) -> dict:
        with self._lock:
            return dict(self._loaded()["devices"].get(key, {}))

    def put(self, key: str, entry: dict, *, last: bool = False) -> None:
        with self._lock:
            doc = self._loaded()
            doc["devices"][key] = _clean_entry(entry)
            if last:
                doc["last"] = key
            self._write()

    def forget(self) -> None:
        with self._lock:
            self._doc = {"last": None, "devices": {}}
            self._write()


def _saved_channel(device: dict | None, saved: dict) -> int:
    most = 1 if device is None else device["inputs"]
    return saved["channel"] if saved.get("channel", most + 1) <= most else 1


def _next_step(entry: dict, late: str | None) -> str:
    if entry.get("level") != "ok":
        return "listen"
    if "headphones" not in entry:
        return "headphones"
    if not entry["headphones"]:
        return "count"              # the late line is about headphones she just said she isn't using
    if "hears_self" not in entry:
        return "hears_self"
    if not entry["hears_self"] and "still" not in entry:
        return "volume"
    return "latency" if late is not None and not entry.get("late_said") else "count"


def _late_line(device: dict | None, rows: list[dict], output_ms: float | None) -> str | None:
    """One line if her sound may come back late.  Only an output she is really using counts."""
    if device is not None and device["mode"] == "direct":
        output = next((r for r in rows if r["name"] == device["name"]), None)
    else:
        output = next((r for r in rows if r.get("default_output")), None)
    known = [ms for ms in (output_ms, None if output is None else output.get("output_ms"))
             if ms is not None]
    if known and max(known) >= LATE_MS:
        return LATE_REPORTED.format(ms=round(max(known)))
    if output is not None and _WIRELESS.search(output["name"]):
        return LATE_GUESS
    return None


def _reading(doc, device: dict) -> tuple[int, float] | None:
    """(the loudest channel, its peak) from what a meter said, or None if it made no sense."""
    rows = doc.get("channels") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    best = None
    for row in rows:
        channel, peak = (row.get("channel"), row.get("peak_db")) if isinstance(row, dict) else (None, None)
        if (isinstance(channel, bool) or not isinstance(channel, int)
                or not 1 <= channel <= device["inputs"] or isinstance(peak, bool)
                or not isinstance(peak, (int, float)) or not math.isfinite(peak)):
            return None
        if best is None or peak > best[1]:
            best = (channel, float(peak))
    return best


class _Listening:
    """One listen in progress: how to call it off, and how to know the device was let go."""

    def __init__(self):
        self.cancel = threading.Event()
        self.done = threading.Event()


# ------------------------------------------------------------- free text, made canonical

def _is_detect(text: str) -> bool:
    return text.strip().lower() in ("", "detect")


def canonical_key(text: str) -> str | None:
    """"F major" from whatever was typed, None for detect.  Only this ever reaches argv.

    What the band will really commit to, spelled the way it was typed.  It plays major and
    minor, so another mode is turned down here rather than quietly swapped for one of them.
    """
    if _is_detect(text):
        return None
    text = text.strip()
    named = parse_prompt(f"in {text}").key
    try:
        # 'Fm' or a bare 'F' or 'Eb maj': the prompt grammar needs a whole mode word
        key = parse_key(named or text)
    except (ValueError, KeyError):
        raise Refused(400, BAD_KEY) from None
    if key.mode not in ("major", "minor"):
        raise Refused(400, ONLY_MAJOR_MINOR)
    tonic = _TYPED_TONIC.match(named or text).group(0)
    named = f"{tonic[0].upper()}{tonic[1:].replace('s', '#')} {key.mode}"
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
    argv = [*PLUME_CHILD, "lead", "--tap-start", "--detect",
            f"--record={folder}/", f"--seed={int(seed)}"]
    if route.mode == "direct":
        argv += [f"--device={route.name}", f"--input-channel={route.channel}", "--direct"]
    elif route.name is not None:
        argv.append(f"--input-device={route.name}")
        if route.channel != 1:
            argv.append(f"--input-channel={route.channel}")
    if route.hear_me:
        argv.append("--monitor")
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


def _seconds(path: Path) -> float:
    import soundfile as sf  # noqa: PLC0415

    try:
        with sf.SoundFile(str(path)) as f:
            return f.frames / f.samplerate
    except (OSError, RuntimeError, ValueError, ZeroDivisionError):
        return 0.0


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


def _clear_fresh_folder(folder: Path, *, false_start: bool = False) -> None:
    """A take that never started, or a false start, leaves no folder on the shelf.

    Only names a take writes, in a folder this run made, and only if they hold nothing:
    anything else stops the rmdir, which is the point.  ``false_start`` lets audio go too:
    the child has gone, and it stopped before there was a band to record with her.
    """
    wavs = () if false_start else (REC_HUMAN, REC_MACHINE, REC_VOICE)
    logs = (REC_DECISIONS, REC_CUES, REC_MIDI)
    for name in (REC_HUMAN, REC_MACHINE, REC_VOICE, *logs, REC_MANIFEST, REC_ENDING, REC_STREAM):
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


def _failure_words(text: str, otherwise: str) -> str:
    low = text.lower()
    for marks, message in _FAILURES:
        if any(mark in low for mark in marks):
            return message
    return otherwise


def classify_failure(log_path: Path) -> str:
    """The page's sentence for what the child wrote.  Never the words it wrote."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - _LOG_TAIL_BYTES))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return DID_NOT_START
    return _failure_words(text, DID_NOT_START)


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
                 tick_s: float = 0.5, watch: bool = True, meter=None, setup_path=None):
        self.root = Path(root)
        self._log_path = log_path
        self._id_of = id_of
        self._spawn = default_spawner if spawner is None else spawner
        self._list = default_lister if lister is None else lister
        self._meter = default_meter if meter is None else meter
        self._setup = SetupStore(setup_path)
        self._listening: _Listening | None = None
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
        self._reported: dict[str | None, float] = {}
        self._output_ms: dict[str | None, float] = {}
        self._stopping_others: set[int] = set()
        self._last_look: tuple[float, tuple] | None = None

    # -- what a page reads

    def home(self) -> dict:
        problem = None
        try:
            devices, rows, mark = self._look()
        except TakeError as e:
            devices, rows, mark, problem = [], [], "", str(e)
        preferred = self._preferred(mark)
        with self._lock:
            # Each device's own number.  One device's delay says nothing about another's.
            reported, output = dict(self._reported), dict(self._output_ms)
        devices = [self._device_view(d, reported, output, rows, mark) for d in devices]
        chosen = default_device(devices, preferred)
        if problem is None and not devices:
            problem = NO_INPUT
        saved = self._saved(chosen, mark)
        return {
            "devices": devices,
            "device": None if chosen is None else chosen["id"],
            "mode": "room" if chosen is None else chosen["mode"],
            "channel": _saved_channel(chosen, saved),
            "hear_me": {"allowed": saved.get("headphones") is True,
                        "default": saved.get("hear_me") is True,
                        "reported_ms": None if chosen is None else chosen["reported_ms"]},
            "folder": pretty_path(self.root),
            "styles": list_styles(),
            "problem": problem,
            "setup": self._setup_doc(chosen, rows, mark, output),
        }

    def _device_view(self, device: dict, reported: dict, output: dict, rows: list[dict], mark: str) -> dict:
        saved = self._saved(device, mark)
        late = _late_line(device, rows, output.get(device["name"]))
        return {**device, "reported_ms": reported.get(device["name"]), "hear_me": saved.get("hear_me") is True,
                "headphones": saved.get("headphones"), "channel": _saved_channel(device, saved),
                "setup_done": _next_step(saved, late) == "count"}

    def _saved(self, device: dict | None, mark: str) -> dict:
        """What is known about this device, as of the devices there are now.

        When they change she is heard again.  A room mic's sound goes wherever the system
        sends it, which may just have become speakers, so its answers go too.
        """
        entry = self._setup.entry(_setup_key(device))
        if entry and entry.get("list") != mark:
            entry.pop("level", None)
            if device is None or device["mode"] == "room":
                for name in _ROOM_ANSWERS:
                    entry.pop(name, None)
        return entry

    def _late_for(self, device: dict | None, rows: list[dict], output: dict) -> str | None:
        return _late_line(device, rows, None if device is None else output.get(device["name"]))

    def _keep(self, device: dict | None, entry: dict, rows: list[dict], mark: str, *, last: bool = False) -> None:
        """Save what is known.  Once she is through, a delay a take reports later stops nobody."""
        entry["list"] = mark
        if _next_step(entry, self._late_for(device, rows, self._output_now())) == "count":
            entry["late_said"] = True
        self._setup.put(_setup_key(device), entry, last=last)

    def _setup_doc(self, device: dict | None, rows: list[dict], mark: str, output: dict) -> dict:
        saved = self._saved(device, mark)
        late = self._late_for(device, rows, output)
        step = _next_step(saved, late)
        direct = device is not None and device["mode"] == "direct"
        line, choices, note = PLAY_SOMETHING, [], None
        if step in _STEPS:
            line, choices = _STEPS[step]
        elif step == "latency":
            line, choices = late, [{"answer": "ok", "label": "OK"}]
        elif step == "count":
            line, note = COUNT_LINE, COUNT_PICKUP if direct else None
        return {"step": step, "line": line, "note": note, "choices": choices,
                "done": step == "count", "seconds": LISTEN_S,
                "device": None if device is None else device["id"],
                "device_name": None if device is None else device["name"],
                "mode": "direct" if direct else "room",
                "channel": _saved_channel(device, saved),
                "hear_me": saved.get("hear_me") is True}

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

    def _look(self) -> tuple[list[dict], list[dict], str]:
        """(the devices a person can pick, everything listed, its mark)."""
        # One child at a time, however many pages are open.
        with self._list_lock:
            listed = self._list()
        rows = _rows(listed)
        look = describe_devices(listed), rows, device_list_mark(rows)
        with self._lock:
            self._last_look = (self._clock(), look)
        return look

    def _look_recent(self) -> tuple[list[dict], list[dict], str]:
        with self._lock:
            last = self._last_look
        if last is not None and self._clock() - last[0] < LOOK_RECENT_S:
            return last[1]
        return self._look()

    # -- setup

    def setup(self, device_id: str | None = None) -> dict:
        """Where this device has got to.  Opens nothing."""
        try:
            devices, rows, mark = self._look()
        except TakeError:
            devices, rows, mark = [], [], ""
        return self._setup_doc(self._named(device_id, devices, mark), rows, mark, self._output_now())

    def _output_now(self) -> dict:
        with self._lock:
            return dict(self._output_ms)

    def _preferred(self, mark: str) -> str | None:
        """The device setup heard her on, while what is plugged in is what it was then."""
        last = self._setup.last()
        return last if last is not None and self._setup.entry(last).get("list") == mark else None

    def _named(self, device_id, devices: list[dict], mark: str) -> dict | None:
        if device_id is None:
            return default_device(devices, self._preferred(mark))
        device = next((d for d in devices if d["id"] == device_id), None)
        if device is None:
            raise Refused(400, DEVICE_MISSING)
        return device

    def answer(self, body: dict) -> dict:
        """One answer from a person.  Hear me only ever follows from a yes about headphones."""
        if not isinstance(body, dict) or set(body) - {"device", "step", "answer"}:
            raise Refused(400, "That is not something setup understands.")
        step, answer = body.get("step"), body.get("answer")
        if not isinstance(step, str) or not isinstance(answer, str) or answer not in _ANSWERS.get(step, ()):
            raise Refused(400, "That is not an answer setup asked for.")
        if step == "hear_me":
            return self._menu_hear_me(body.get("device"), answer == "on")
        # She may have just plugged them in because it asked, so this one waits for a fresh look.
        fresh = step == "headphones"
        try:
            devices, rows, mark = self._look() if fresh else self._look_recent()
        except TakeError as e:
            raise Refused(409, str(e)) from None
        device = self._named(body.get("device"), devices, mark)
        entry, asked_again = self._in_the_flow(device, mark)
        if not asked_again or fresh:
            self._apply_answer(entry, step, answer)
        self._keep(device, entry, rows, mark)
        doc = self._setup_doc(device, rows, mark, self._output_now())
        return doc if fresh or not doc["done"] else self._still_so(device, doc, mark)

    def _in_the_flow(self, device: dict | None, mark: str) -> tuple[dict, bool]:
        """What is known partway through, and whether a yes about headphones just lapsed.

        Unlike :meth:`_saved`, she keeps her level: she was only just heard on this device, and
        something plugged in elsewhere is no reason to play again.  A room mic's answers still go,
        because its sound may have just changed where it comes out.
        """
        entry = self._setup.entry(_setup_key(device))
        if not entry or entry.get("list") == mark or not (device is None or device["mode"] == "room"):
            return entry, False
        for name in _ROOM_ANSWERS:
            entry.pop(name, None)
        return entry, True

    def _still_so(self, device: dict | None, doc: dict, mark: str) -> dict:
        """Setup is about to say done, and the band follows, so it says so of the devices now."""
        try:
            devices, rows, now = self._look()
        except TakeError as e:
            raise Refused(409, str(e)) from None
        if now == mark:
            return doc
        device = self._named(None if device is None else device["id"], devices, now)
        entry, _ = self._in_the_flow(device, now)
        self._keep(device, entry, rows, now)
        return self._setup_doc(device, rows, now, self._output_now())

    @staticmethod
    def _apply_answer(entry: dict, step: str, answer: str) -> None:
        if step == "headphones":
            for name in ("hears_self", "still"):
                entry.pop(name, None)
            entry["headphones"], entry["hear_me"] = answer == "yes", False
        elif step == "hears_self":
            if entry.get("headphones") is not True:
                raise Refused(409, "Setup has moved on. Please start again.")
            entry.pop("still", None)
            entry["hears_self"], entry["hear_me"] = answer == "yes", False
        elif step == "volume":
            if entry.get("headphones") is not True or entry.get("hears_self") is not False:
                raise Refused(409, "Setup has moved on. Please start again.")
            if answer == "fixed":
                entry["hears_self"], entry["hear_me"] = True, False
            else:
                entry["still"], entry["hear_me"] = True, True
        else:
            entry["late_said"] = True

    def _menu_hear_me(self, device_id, on: bool) -> dict:
        """The tick in the menu.  On is a yes about headphones for this device, as things are now.

        It settles what setup would have asked, so a finished setup stays finished.
        """
        try:
            devices, rows, mark = self._look()
        except TakeError as e:
            raise Refused(409, str(e)) from None
        device = self._named(device_id, devices, mark)
        entry = self._saved(device, mark)
        if on:
            entry.update(headphones=True, hears_self=False, still=True, hear_me=True)
        else:
            entry["hear_me"] = False
            if entry.get("headphones") is True:
                entry["hears_self"] = True
                entry.pop("still", None)
        self._keep(device, entry, rows, mark)
        return self._setup_doc(device, rows, mark, self._output_now())

    def reset_setup(self) -> dict:
        """Set up again.  The next press of Start begins from the first step."""
        self._release(self._call_off_listening())
        self._setup.forget()
        return self.setup()

    def cancel_listen(self) -> dict:
        """She left.  When this returns, no device is open on setup's account."""
        self._release(self._call_off_listening())
        return self.setup()

    def _call_off_listening(self) -> _Listening | None:
        with self._lock:
            job = self._listening
        if job is not None:
            job.cancel.set()
        return job

    @staticmethod
    def _release(job: _Listening | None) -> bool:
        return job is None or job.done.wait(RELEASE_S)

    def listen(self, body: dict) -> dict:
        """Step 1.  The one thing here that opens a device, for :data:`LISTEN_S` seconds.

        Only ever from a POST the page makes on a click, which is the consent for this input.
        """
        if not isinstance(body, dict) or set(body) - {"device"}:
            raise Refused(400, "That is not something setup understands.")
        wanted = body.get("device")
        if wanted is not None and not isinstance(wanted, str):
            raise Refused(400, DEVICE_MISSING)
        job = _Listening()
        with self._lock:
            if self._closing:
                raise Refused(409, "plume gui is closing.", live=self._doc())
            if self._take is not None or self._preparing:
                raise Refused(409, ALREADY_RUNNING, live=self._doc())
            prior, self._listening = self._listening, job
        try:
            if prior is not None:
                prior.cancel.set()
                if not self._release(prior):
                    raise Refused(409, STILL_LISTENING)
            others = self._others()
            if others:
                raise Refused(409, TERMINAL_TAKE, live=self._elsewhere(others))
            try:
                devices, rows, mark = self._look()
            except TakeError as e:
                return self._heard("failed", None, rows=[], mark="", line=str(e))
            if wanted is not None:
                tried = [self._named(wanted, devices, mark)]
            else:
                tried = self._candidates(devices, mark)
            if not tried:
                return self._heard("failed", None, rows=rows, mark=mark, line=NO_INPUT)
            readings = self._read_meters(tried, job)
            return self._judge(job, tried, readings, devices, rows, mark)
        finally:
            job.done.set()
            with self._lock:
                if self._listening is job:
                    self._listening = None

    def _candidates(self, devices: list[dict], mark: str) -> list[dict]:
        """Who to listen to with nothing chosen.  Never software, and never anything arbitrary:
        every interface at once, so that hearing her is what picks one.  Something new plugged
        in may be why she is here, so the one she was heard on is only tried alongside it."""
        preferred = self._preferred(mark)
        if preferred is not None:
            kept = [d for d in devices if d["id"] == preferred]
            if kept:
                return kept
        last = self._setup.last()
        known = next((d for d in devices if d["id"] == last), None)
        tried = [d for d in devices if d["kind"] == "interface" and d is not known]
        tried = ([] if known is None else [known]) + tried
        if tried:
            return tried[:MAX_LISTEN_DEVICES]
        chosen = default_device(devices)
        if chosen is not None:
            return [chosen]
        return [d for d in devices if d["kind"] != "virtual"][:1]

    def _read_meters(self, tried: list[dict], job: _Listening) -> list:
        """A (channel, peak), or a sentence, or None for numbers that made no sense."""
        out: list = [None] * len(tried)

        def one(n: int, device: dict) -> None:
            try:
                out[n] = _reading(self._meter(device["name"], LISTEN_S, job.cancel), device)
            except TakeError as e:
                out[n] = str(e)
            except Exception:  # noqa: BLE001 - a failed step must not cost her the page
                out[n] = NO_READING

        if len(tried) == 1:
            one(0, tried[0])
            return out
        workers = [threading.Thread(target=one, args=(n, d), name="plume-gui-meter", daemon=True)
                   for n, d in enumerate(tried)]
        for worker in workers:
            worker.start()
        until = time.monotonic() + LISTEN_S + METER_GRACE_S + RELEASE_S
        for worker in workers:
            worker.join(max(0.0, until - time.monotonic()))
        return out

    def _judge(self, job: _Listening, tried: list[dict], readings: list, devices: list[dict],
               rows: list[dict], mark: str) -> dict:
        others = [{"id": d["id"], "name": d["name"]} for d in devices
                  if d["kind"] != "virtual" and d not in tried]
        if job.cancel.is_set():
            return self._heard("cancelled", tried[0], rows=rows, mark=mark)
        heard = [(r[1], r[0], d) for r, d in zip(readings, tried)
                 if isinstance(r, tuple) and r[1] >= NOTHING_DB]
        if not heard:
            if not any(isinstance(r, tuple) for r in readings):
                said = next((r for r in readings if isinstance(r, str)), NO_READING)
                return self._heard("failed", tried[0], rows=rows, mark=mark, line=said, others=others)
            names = " or ".join(d["name"] for d in tried)
            words = NOTHING_GAIN if tried[0]["mode"] == "direct" else NOTHING_ROOM
            return self._heard("nothing", tried[0], rows=rows, mark=mark, line=words.format(name=names),
                               others=others)
        # An interface that hears her beats a mic that also does: a mic hears everything.
        peak, channel, device = max(heard, key=lambda h: (h[2]["kind"] == "interface", h[0]))
        level = "hot" if peak >= HOT_DB else "low" if peak < LOW_DB else "ok"
        entry = self._saved(device, mark)
        # The channel is right whatever the gain is, so a Start from here uses it.
        entry.update(channel=channel, level=level)
        self._keep(device, entry, rows, mark, last=True)
        line = None
        if level != "ok":
            direct = device["mode"] == "direct"
            line = {"hot": HOT_GAIN if direct else HOT_ROOM, "low": LOW_GAIN if direct else LOW_ROOM}[level]
        return self._heard(level, device, rows=rows, mark=mark, line=line, channel=channel,
                           peak_db=round(peak, 1))

    def _heard(self, result: str, device: dict | None, *, rows: list[dict], mark: str, line: str | None = None,
               others: list | None = None, channel: int | None = None, peak_db: float | None = None) -> dict:
        doc = self._setup_doc(device, rows, mark, self._output_now())
        doc["heard"] = {"result": result, "channel": channel, "peak_db": peak_db}
        doc["others"] = others or []
        if line is not None:
            doc["step"], doc["line"], doc["note"] = "listen", line, None
            doc["done"], doc["choices"] = False, _TRY_OR_START
        return doc

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
            listening = self._listening
        try:
            if listening is not None:
                listening.cancel.set()
                if not self._release(listening):
                    raise Refused(409, STILL_LISTENING)
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
            devices, _, mark = self._look()
        except TakeError as e:
            return self._failed(str(e))
        if not devices:
            return self._failed(NO_INPUT)
        route, _ = _route(body, devices, preferred=self._preferred(mark),
                          saved_for=lambda d: self._saved(d, mark))
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
            listening = self._listening
        if listening is not None:
            listening.cancel.set()
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
            listening = self._listening
        if listening is not None:
            listening.cancel.set()
            self._release(listening)
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
                take.stream = stream
                if stream["reported_ms"] is not None:
                    self._reported[take.route.name] = stream["reported_ms"]
                if stream["output_ms"] is not None:
                    self._output_ms[take.route.name] = stream["output_ms"]
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
            return (FLAT_CHANNEL.format(channel=take.route.channel) if take.route.channels > 1
                    else FLAT_MIC)
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
        """It has gone.  Say where that left things, once.

        It is a take once the band has played a bar.  Before that it is a false start, unless
        she played for a good while and the count just never caught: that recording is hers.
        """
        folder = take.folder
        result = None
        if self._has_rows(take) or (take.loud and _seconds(folder / REC_HUMAN) >= KEEP_UNCOUNTED_S):
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
            _clear_fresh_folder(folder, false_start=True)
            if take.stop_asked:
                result = self._false_start(take)
            else:
                result = {"state": "failed", "message": classify_failure(Path(self._log_path()))}
        with self._lock:
            if self._take is take:
                self._take = None
                self._result = result

    def _false_start(self, take: _Take) -> dict:
        """Back on home, in words that say what to try, if anything needs saying."""
        flat = (not take.loud and take.phase == "waiting"
                and self._clock() - take.opened >= FLAT_AFTER_S)
        if flat and take.note is None:
            message = (FLAT_CHANNEL_HOME.format(channel=take.route.channel)
                       if take.route.channels > 1 else FLAT_MIC_HOME)
            return {"state": "idle", "message": message, "menu": True}
        message = take.note or (KEPT_NOTHING if take.loud else None)
        return {"state": "idle", "message": message} if message else {"state": "idle"}
