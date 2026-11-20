"""Live MIDI output: the band as notes, for another instrument to play.

Plume decides what the band plays a bar at a time and renders it with its own
synths.  Each bar's notes are already known before the audio is made
(``DuetEngine.last_events``), so this module sends those notes out of a MIDI port
as well: one channel per band part, so a DAW such as Ableton Live can put its own
instrument on each part.  See docs/PLAYING.md, "Sending the band to Ableton".

Four parts:

- :func:`bar_notes` turns one bar's events into notes: channel, pitch, velocity,
  and the stream samples they start and stop on.  Pure.
- :class:`Schedule` holds notes waiting to be sent and hands out the messages
  that are due, keeping every note-on paired with exactly one note-off.  Pure.
- :class:`BandMidi` owns the port and a sender thread.  The planner hands it a
  bar about a bar ahead; the sender sends each message when plume's own sound of
  that note would reach the speaker, plus an offset the player sets.
- :class:`BandFile` keeps what the port was sent, so a take with ``--record`` has
  the band as a standard MIDI file, ``band.mid``, beside its audio.

Nothing here runs in the audio callback.  The sender reads the playhead the
callback advances, which is a plain integer, and works out the wall-clock time of
any stream sample from when it saw the playhead move (:class:`StreamClock`).

``mido`` with ``python-rtmidi`` is the optional ``midi`` extra, imported only when a
port is asked for.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import math
import os
import struct
import threading
import time
from pathlib import Path

from .midi import MidiError, _import_mido

#: What ``--midi-out`` takes to make a new port of plume's own instead of naming one.
VIRTUAL = "virtual"
#: The name of that port, as other programs list it.
VIRTUAL_NAME = "plume"

#: Channel per band part, counted from 0 (a DAW shows these one higher).
ROLE_CHANNELS = {"bass": 0, "pad": 1, "keys": 1, "lead": 2, "arp": 3, "shimmer": 4,
                 "sub": 5}
#: Any other part with pitches.
OTHER_CHANNEL = 6
#: General MIDI keeps channel 10 for drums.
DRUM_CHANNEL = 9
#: Drum parts and their General MIDI notes.  A part not here is not sent.
DRUM_NOTES = {"kick": 36, "tick": 37, "click": 37, "snare": 38, "clap": 39, "hat": 42,
              "foot": 44, "openhat": 46, "crash": 49}
#: Parts with no notes to send: a noise bed is a sound, not a part to play.
SKIP_ROLES = frozenset({"crackle"})
#: How long a drum hit is held.  A hit has no length, but plenty of samplers want
#: a note-off, and a stuck one would hold a choke group.
DRUM_LEN_S = 0.05
#: The shortest note sent, so an off can never land before its on.
MIN_LEN_S = 0.005

#: How often the sender looks at the playhead and the queue, in seconds.
POLL_S = 0.001
#: How long the sender's reading of the stream clock remembers a sighting.  Long
#: enough to catch the earliest callback, short enough to follow clock drift.
CLOCK_WINDOW_S = 2.0
#: How long to wait on a port opening, listing or closing, like plume.midi.
PORT_TIMEOUT_S = 2.0

#: Ticks per quarter note in a take's ``band.mid``: under a millisecond at any tempo
#: plume plays.
TICKS_PER_BEAT = 960
#: The file's tempo when no bar said one.
DEFAULT_FILE_BPM = 120.0
#: The most note messages a take's file keeps, each a few bytes until the file is
#: written: about a hundred thousand notes, far more than an hour of the band sends.
#: Past it new notes are left out and every note already in still ends.
MAX_FILE_MESSAGES = 200_000
#: The most bar tempos kept for the file's tempo: more bars than the longest take has.
MAX_FILE_TEMPOS = 16_384
#: How long a take's ending waits on its ``band.mid`` being written.
FILE_WRITE_S = 15.0

#: Message order at the same sample: a note-off goes before a note-on, so a note
#: struck again on the beat it ends is a new note, not a cut-off one.
_OFF, _ON = 0, 1
_CC_SUSTAIN = 64
_CC_ALL_NOTES_OFF = 123


def channel_for(role: str) -> int | None:
    """The channel a band part is sent on, or None if it is not sent."""
    if role in SKIP_ROLES:
        return None
    if role in DRUM_NOTES:
        return DRUM_CHANNEL
    return ROLE_CHANNELS.get(role, OTHER_CHANNEL)


def velocity_of(event: dict) -> int:
    """A MIDI velocity from an event's velocity and its part's level, both 0..1."""
    v = float(event.get("velocity", 0.8)) * float(event.get("level", 1.0))
    return max(1, min(127, int(round(127.0 * v))))


def bar_notes(events, at: int, sr: int) -> list[tuple[int, int, int, int, int]]:
    """One bar's events as ``(on_sample, off_sample, channel, note, velocity)``.

    ``at`` is the stream sample the bar starts on: where its audio is written.  An
    event's ``onset_s`` counts from there, as it does in the render.
    """
    out = []
    for ev in events:
        role = str(ev.get("role", ""))
        if role == "announce":                    # the cue gesture plays as its part
            role = str(ev.get("voice_role", ""))
        ch = channel_for(role)
        if ch is None:
            continue
        if ch == DRUM_CHANNEL:
            note, dur = DRUM_NOTES[role], DRUM_LEN_S
        else:
            if ev.get("midi") is None:
                continue
            note, dur = int(ev["midi"]), max(MIN_LEN_S, float(ev.get("dur_s", 0.0)))
        if not 0 <= note <= 127:
            continue
        on = int(at) + int(round(float(ev.get("onset_s", 0.0)) * sr))
        off = max(on + 1, on + int(round(dur * sr)))
        out.append((on, off, ch, note, velocity_of(ev)))
    return out


class Schedule:
    """Notes waiting to be sent, and the notes sounding now.

    Every note-on is paired with exactly one note-off: a note struck while the
    same note on the same channel still sounds is ended first, and the earlier
    note's own off is then dropped.  :meth:`release` ends whatever sounds.
    """

    def __init__(self):
        #: [sample, order, seq, kind, channel, note, velocity, note id, bar key]
        self._pending: list = []
        self._seq = itertools.count()
        self._ids = itertools.count()
        #: (channel, note) -> the id of the note sounding there
        self._sounding: dict[tuple[int, int], int] = {}
        self.channels: set[int] = set()

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def sounding(self) -> int:
        return len(self._sounding)

    def add(self, notes, key=None) -> None:
        for on, off, ch, note, vel in notes:
            nid = next(self._ids)
            heapq.heappush(self._pending, (on, _ON, next(self._seq), "on", ch, note, vel,
                                           nid, key))
            heapq.heappush(self._pending, (off, _OFF, next(self._seq), "off", ch, note, 0,
                                           nid, key))

    def move(self, key, delta: int, new_key=None) -> None:
        """Move the unsent messages of the bar ``key`` by ``delta`` samples, and
        call that bar ``new_key`` from now on if one is given."""
        new_key = key if new_key is None else new_key
        if not delta and new_key == key:
            return
        self._pending = [((e[0] + delta,) + e[1:8] + (new_key,)) if e[8] == key else e
                         for e in self._pending]
        heapq.heapify(self._pending)

    def next_sample(self) -> int | None:
        return self._pending[0][0] if self._pending else None

    def due(self, upto: float) -> list[tuple]:
        """The messages at or before stream sample ``upto``, as ``(kind, ch, a, b)``."""
        return [m for _s, m in self.due_timed(upto)]

    def due_timed(self, upto: float) -> list[tuple]:
        """:meth:`due`, each message with the stream sample it is for:
        ``(sample, (kind, ch, a, b))``."""
        out = []
        while self._pending and self._pending[0][0] <= upto:
            s, _o, _q, kind, ch, note, vel, nid, _k = heapq.heappop(self._pending)
            where = (ch, note)
            if kind == "on":
                if where in self._sounding:
                    out.append((s, ("off", ch, note, 0)))
                self._sounding[where] = nid
                self.channels.add(ch)
                out.append((s, ("on", ch, note, vel)))
            elif self._sounding.get(where) == nid:
                del self._sounding[where]
                out.append((s, ("off", ch, note, 0)))
        return out

    def unsent_off(self, ch: int, note: int) -> None:
        """An off that did not reach the port: the note still counts as sounding, so
        the next strike of it, or :meth:`release`, ends it."""
        self._sounding.setdefault((ch, note), -1)

    def release(self) -> list[tuple]:
        """Forget what is waiting; end what sounds, then all notes off and the
        sustain pedal up on every channel this ever played on."""
        self._pending = []
        out = [("off", ch, note, 0) for ch, note in sorted(self._sounding)]
        self._sounding.clear()
        for ch in sorted(self.channels):
            out.append(("cc", ch, _CC_SUSTAIN, 0))
            out.append(("cc", ch, _CC_ALL_NOTES_OFF, 0))
        return out


class StreamClock:
    """Which wall-clock time a stream sample is handed to the device.

    The playhead moves by whole blocks.  A device may run one callback per block,
    or several back to back once per period of its own (and several at start-up,
    to fill its buffer).  Either way, the samples handed over in one burst start
    playing together, from the first one.  So a sighting of the playhead moving, at
    time ``now``, dates the sample it moved FROM: that one went at or before
    ``now``.  Moves seen less than half a block apart are one burst and only its
    first sighting counts, so a burst seen in two halves cannot date its middle as
    its start.  The earliest reading over :data:`CLOCK_WINDOW_S` is the best guess,
    and the window lets a slow drift between the audio and system clocks through.
    """

    def __init__(self, sr: int, block: int, *, window_s: float = CLOCK_WINDOW_S):
        self._sr = float(sr)
        self._block = int(block)
        self._window = float(window_s)
        self._last: int | None = None
        self._moved_at = -float("inf")
        #: (seen at, offset) with offsets rising: the front is the window's least
        self._low: collections.deque = collections.deque()

    @property
    def ready(self) -> bool:
        return bool(self._low)

    def observe(self, playhead: int, now: float) -> None:
        if self._last is not None and playhead != self._last:
            if now - self._moved_at >= 0.5 * self._block / self._sr:
                c = now - self._last / self._sr
                while self._low and self._low[-1][1] >= c:
                    self._low.pop()
                self._low.append((now, c))
            self._moved_at = now
        self._last = playhead
        while len(self._low) > 1 and now - self._low[0][0] > self._window:
            self._low.popleft()

    def time_of(self, sample: float) -> float:
        return self._low[0][1] + sample / self._sr

    def sample_at(self, t: float) -> float:
        return (t - self._low[0][1]) * self._sr


def _track_names() -> dict[int, str]:
    """What each channel's track in ``band.mid`` is called: the parts sent on it."""
    names: dict[int, list[str]] = {}
    for role, ch in ROLE_CHANNELS.items():
        names.setdefault(ch, []).append(role)
    out = {ch: " and ".join(roles) for ch, roles in names.items()}
    out[OTHER_CHANNEL] = "other"
    out[DRUM_CHANNEL] = "drums"
    return out


#: One kept message: stream sample, on (1) or off (0), channel, note, velocity.
_KEPT = struct.Struct("<qBBBB")


class BandFile:
    """What a take's port was sent, written as a standard MIDI file when it ends.

    The sender hands over each note message with the stream sample it is for: where
    plume's own sound of it starts, so sample 0 is the first sample of the take's
    ``machine.wav`` and ``human.wav``, whatever the latency and offset did to when
    the port got it.  Messages are kept packed, a few bytes each, up to
    ``max_messages`` (with room always left to end every note), and the file is
    written once, at the end: a MIDI file's tracks carry their lengths, so it cannot
    grow as the take does.  It goes to a temporary name first, so an ending cut short
    leaves no half file.

    One track per channel, named for its parts, and one tempo: the median of the
    tempos the bars were planned at.  Times are converted at that tempo from the
    stream sample, so every note sits at the right time in a DAW even when the take's
    tempo moved; only the bar lines are approximate then.

    Only notes are kept.  The All Notes Off and sustain messages a take starts and
    ends with are for a live port; in the file every note ends with its own off.
    Never raises from :meth:`add`; :meth:`write` notes a failure in :attr:`errors`.
    """

    def __init__(self, path, *, sr: int = 48000, max_messages: int = MAX_FILE_MESSAGES):
        self.path = Path(path)
        self.sr = int(sr)
        self.max_messages = int(max_messages)
        self.dropped = 0
        self.errors: list[str] = []
        self._kept = bytearray()
        self._n = 0
        self._sounding: set[tuple[int, int]] = set()
        self._tempos: list[float] = []
        self._last = 0
        self._done = False
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._n

    def tempo(self, bpm) -> None:
        """One bar's tempo, for the file's."""
        try:
            bpm = float(bpm)
        except (TypeError, ValueError):
            return
        if math.isfinite(bpm) and 10.0 <= bpm <= 400.0:
            with self._lock:
                if len(self._tempos) < MAX_FILE_TEMPOS:
                    self._tempos.append(bpm)

    def add(self, sample, kind: str, ch: int, note: int, vel: int) -> None:
        """One message as it went to the port.  Anything but a note is not kept, nor
        an off for a note not sounding."""
        if kind not in ("on", "off"):
            return
        where = (int(ch), int(note))
        sample = max(0, int(sample))
        with self._lock:
            if self._done:
                return
            if kind == "on":
                if where in self._sounding:          # the sender ends it first; still
                    self._keep(sample, 0, *where, 0)
                    self._sounding.discard(where)
                # Room for this on, its off and the off of every note now sounding.
                if self._n + len(self._sounding) + 2 > self.max_messages:
                    self.dropped += 1
                    return
                self._sounding.add(where)
                self._keep(sample, 1, *where, max(1, min(127, int(vel))))
            elif where in self._sounding:
                self._sounding.discard(where)
                self._keep(sample, 0, *where, 0)

    def end(self, sample=None) -> None:
        """End every note still sounding, at ``sample`` or the last message's."""
        with self._lock:
            at = self._last if sample is None else max(self._last, int(sample))
            for ch, note in sorted(self._sounding):
                self._keep(at, 0, ch, note, 0)
            self._sounding.clear()

    def _keep(self, sample: int, on: int, ch: int, note: int, vel: int) -> None:
        self._kept += _KEPT.pack(sample, on, ch, note, vel)
        self._n += 1
        self._last = max(self._last, sample)

    def bpm(self) -> float:
        with self._lock:
            tempos = sorted(self._tempos)
        return tempos[len(tempos) // 2] if tempos else DEFAULT_FILE_BPM

    def write(self, end_at=None, *, mido=None) -> bool:
        """End every note, write the file, and keep nothing more.  Once only."""
        if self._done:
            return False
        self.end(end_at)
        with self._lock:
            self._done = True
            kept, self._kept = self._kept, bytearray()
        try:
            if mido is None:
                from .midi import _import_mido  # noqa: PLC0415
                mido = _import_mido()
            # Ticks from the tempo as the file stores it, whole microseconds a beat:
            # from the exact one an hour would drift by milliseconds.
            tempo_us = int(mido.bpm2tempo(self.bpm()))
            per_tick = self.sr * tempo_us / (1e6 * TICKS_PER_BEAT)    # samples a tick
            by_ch: dict[int, list[tuple[int, int, int, int, int]]] = {}
            for i, (s, on, ch, note, vel) in enumerate(_KEPT.iter_unpack(kept)):
                by_ch.setdefault(ch, []).append((s, i, on, note, vel))
            del kept
            f = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
            head = mido.MidiTrack()
            head.append(mido.MetaMessage("track_name", name="plume band", time=0))
            head.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
            head.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                                         time=0))
            head.append(mido.MetaMessage("end_of_track", time=0))
            f.tracks.append(head)
            names = _track_names()
            for ch in sorted(by_ch):
                track = mido.MidiTrack()
                track.append(mido.MetaMessage("track_name", name=names.get(ch, "other"),
                                              time=0))
                prev = 0
                # In the order they were sent, which pairs every on with its off.  A
                # bar queued late under a negative offset can go out behind one sent
                # before it; it is then put no earlier than that one.
                for s, _i, on, note, vel in by_ch[ch]:
                    tick = max(prev, int(round(s / per_tick)))
                    track.append(mido.Message("note_on" if on else "note_off",
                                              channel=ch, note=note, velocity=vel,
                                              time=tick - prev))
                    prev = tick
                track.append(mido.MetaMessage("end_of_track", time=0))
                f.tracks.append(track)
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                f.save(str(tmp))
                os.replace(tmp, self.path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        except Exception as e:             # noqa: BLE001 - never take the take down
            if len(self.errors) < 20:
                self.errors.append(f"band.mid: {e}")
            return False
        return True


def _message(mido, kind: str, ch: int, a: int, b: int):
    if kind == "on":
        return mido.Message("note_on", channel=ch, note=a, velocity=b)
    if kind == "off":
        return mido.Message("note_off", channel=ch, note=a, velocity=0)
    return mido.Message("control_change", channel=ch, control=a, value=b)


def _bounded(fn, timeout: float, what: str):
    """``fn()`` on its own thread, waited on for ``timeout``: a driver for a gone
    device may never return."""
    answer: list = []

    def run() -> None:
        try:
            answer.append((True, fn()))
        except Exception as e:             # noqa: BLE001 - the backend's own type
            answer.append((False, e))

    t = threading.Thread(target=run, name="plume-midi-out-port", daemon=True)
    t.start()
    t.join(timeout)
    if not answer:
        raise MidiError(f"the MIDI driver did not {what} in {timeout:g} seconds")
    ok, value = answer[0]
    if not ok:
        raise value
    return value


def resolve_output(want: str) -> tuple[str, bool]:
    """``(port name, is virtual)`` for what ``--midi-out`` said: ``virtual``, an
    exact port name, or part of one."""
    if want == VIRTUAL:
        return VIRTUAL_NAME, True
    mido = _import_mido()
    try:
        names = list(_bounded(mido.get_output_names, PORT_TIMEOUT_S, "list its ports"))
    except MidiError:
        raise
    except ImportError as e:
        raise MidiError(f"live MIDI needs mido and python-rtmidi: "
                        f"pip install -e '.[midi]' ({e})") from None
    except Exception as e:                 # noqa: BLE001 - a backend's own error type
        raise MidiError(f"could not list MIDI output ports: {e}") from None
    if want in names:
        return want, False
    hits = [n for n in names if want.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0], False
    if hits:
        raise MidiError(f"--midi-out {want} matches more than one port: {', '.join(hits)}")
    known = ", ".join(names) or "none"
    raise MidiError(f"no MIDI output port matching {want} (there is: {known}); "
                    f"--midi-out {VIRTUAL} makes one called {VIRTUAL_NAME!r} where "
                    "the system allows it")


class BandMidi:
    """The band, sent out of one MIDI port while a take plays.

    ``only``: plume's own band is silent (:meth:`heard`), so the port's instruments
    are the band.  ``offset_s`` moves every message later (or earlier, negative)
    than the moment plume's own sound of it would reach the speaker: a DAW adds
    its own output latency to a note it is sent live.

    The planner calls :meth:`bar` and :meth:`move`; the sender thread does the
    rest.  :meth:`close` ends every sounding note whatever happened before it.

    ``file``: a :class:`BandFile` that keeps every note sent, written by :meth:`close`
    when the take had started.
    """

    def __init__(self, port: str, *, virtual: bool = False, only: bool = False,
                 offset_s: float = 0.0, sr: int = 48000, block: int = 256,
                 clock=time.perf_counter, poll_s: float = POLL_S,
                 file: BandFile | None = None):
        self.port = port
        self.file = file
        self.virtual = bool(virtual)
        self.only = bool(only)
        self.offset_s = float(offset_s)
        self.latency_s = 0.0
        self.sent = 0
        self.errors: list[str] = []
        self._sr = int(sr)
        self._clock = clock
        self._poll_s = float(poll_s)
        self._schedule = Schedule()
        self._stream = StreamClock(sr, block)
        self._lock = threading.Lock()
        #: Held for every batch sent, so the final release can never be overtaken
        #: by a batch the sender had already begun.
        self._send_lock = threading.Lock()
        self._closing = threading.Event()
        self._released = False
        self._thread: threading.Thread | None = None
        self._playhead = None
        self._handle = None
        self._mido = None

    # ------------------------------------------------------------ the port

    def open(self) -> "BandMidi":
        mido = _import_mido()
        self._mido = mido
        try:
            if self.virtual:
                self._handle = _bounded(
                    lambda: mido.open_output(self.port, virtual=True),
                    PORT_TIMEOUT_S, "open a port")
            else:
                self._handle = _bounded(lambda: mido.open_output(self.port),
                                        PORT_TIMEOUT_S, "open a port")
        except MidiError:
            raise
        except (NotImplementedError, AttributeError, TypeError) as e:
            raise MidiError(f"this system cannot make a virtual MIDI port ({e}); make "
                            "a loopback port (loopMIDI on Windows) and name it with "
                            "--midi-out") from None
        except Exception as e:             # noqa: BLE001 - a backend's own error type
            raise MidiError(f"could not open MIDI output {self.port}: {e}") from None
        return self

    def start(self, playhead, *, latency_s: float = 0.0, thread: bool = True) -> "BandMidi":
        """Start sending.  ``playhead`` gives the stream sample the device has been
        handed up to; ``latency_s`` is the stream's output latency."""
        self._playhead = playhead
        self.latency_s = max(0.0, float(latency_s))
        # A note left hanging by a take that was killed outright ends here.
        clear = [m for ch in (*sorted(set(ROLE_CHANNELS.values())), OTHER_CHANNEL,
                              DRUM_CHANNEL)
                 for m in (("cc", ch, _CC_SUSTAIN, 0), ("cc", ch, _CC_ALL_NOTES_OFF, 0))]
        try:
            _bounded(lambda: self._send(clear), PORT_TIMEOUT_S, "send")
        except Exception as e:             # noqa: BLE001 - the take still plays
            self._note(f"midi out: {e}")
        if thread:
            self._thread = threading.Thread(target=self._run, name="plume-midi-out",
                                            daemon=True)
            self._thread.start()
        return self

    def close(self, *, timeout: float = 1.0, file_timeout: float = FILE_WRITE_S) -> None:
        """Stop sending, end every sounding note, write the take's file, close the
        port.  Safe twice.

        Bounded by about three times ``timeout`` plus ``file_timeout`` whatever the
        driver does: a send that never returns costs the release, never the caller.
        Once closing, no note-on goes out, so nothing can follow the final All Notes
        Off.  The notes are let go of first, so writing a long take's file never holds
        a note on; the file needs nothing from the port: in it every note still
        sounding ends where the take did.
        """
        self._closing.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        band_file, self.file = self.file, None
        end_at = None
        if band_file is not None and self._playhead is not None:
            try:
                end_at = int(self._playhead())
            except Exception:              # noqa: BLE001 - the last message's, then
                end_at = None
        try:
            _bounded(self._release, timeout, "take the notes off")
        except Exception as e:             # noqa: BLE001 - it is going anyway
            self._note(f"midi out: {e}")
        if band_file is not None and self._playhead is not None:
            try:
                _bounded(lambda: band_file.write(end_at), file_timeout, "write band.mid")
            except Exception as e:         # noqa: BLE001 - the take is over
                self._note(f"midi out: {e}")
            for e in band_file.errors:
                self._note(f"midi out: {e}")
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                _bounded(handle.close, timeout, "close the port")
            except Exception:              # noqa: BLE001 - it is going anyway
                pass

    def _release(self) -> None:
        with self._send_lock:
            if self._released:
                return
            self._released = True
            with self._lock:
                msgs = self._schedule.release()
            self._send(msgs)

    # ------------------------------------------------------------ the planner

    def bar(self, at: int, events, *, key=None, bpm=None) -> None:
        """Queue one bar's events, its audio starting at stream sample ``at``.
        ``key`` names the bar for :meth:`move`; by default its sample.  ``bpm`` is
        the tempo it was planned at, for the take's file."""
        band_file = self.file
        if band_file is not None and bpm is not None:
            band_file.tempo(bpm)
        notes = bar_notes(events, at, self._sr)
        if self._playhead is not None:
            # The audio of a note that starts before the playhead is cut off, so
            # its MIDI would only be a late burst.
            now = int(self._playhead())
            notes = [n for n in notes if n[0] >= now]
        with self._lock:
            self._schedule.add(notes, key=at if key is None else key)

    def move(self, old_at: int, new_at: int) -> None:
        """A bar queued at ``old_at`` now starts at ``new_at``."""
        with self._lock:
            self._schedule.move(old_at, int(new_at) - int(old_at), new_key=int(new_at))

    def heard(self, audio):
        """What plume itself plays of a bar of band audio: none of it when the
        port's instruments are the band."""
        return audio * 0.0 if self.only else audio

    # ------------------------------------------------------------ the sender

    def pump(self, now: float | None = None) -> int:
        """One look: read the playhead, send what is due.  Returns how many sent."""
        now = self._clock() if now is None else now
        if self._playhead is None:
            return 0
        self._stream.observe(int(self._playhead()), now)
        if not self._stream.ready:
            return 0
        upto = self._stream.sample_at(now - self.latency_s - self.offset_s)
        with self._send_lock:
            if self._released:
                return 0
            with self._lock:
                timed = self._schedule.due_timed(upto)
            return self._send([m for _s, m in timed], [s for s, _m in timed])

    def _run(self) -> None:
        while not self._closing.is_set():
            try:
                self.pump()
            except Exception as e:         # noqa: BLE001 - never take the take down
                self._note(f"midi out: {e}")
            self._closing.wait(self._poll_s)

    def _send(self, msgs, samples=None) -> int:
        """Send in order; called holding the send lock.  Once closing, note-ons are
        left out: release ends every note marked sounding, sent or not.  With the
        stream ``samples`` they are for, each message handed to the port is kept in
        the take's file, whether or not the port took it: the file is the band."""
        handle, mido = self._handle, self._mido
        if handle is None or mido is None:
            return 0
        band_file = self.file if samples is not None else None
        n = 0
        for i, (kind, ch, a, b) in enumerate(msgs):
            if kind == "on" and self._closing.is_set():
                continue
            if band_file is not None:
                band_file.add(samples[i], kind, ch, a, b)
            try:
                handle.send(_message(mido, kind, ch, a, b))
                self.sent += 1
                n += 1
            except Exception as e:         # noqa: BLE001 - a port gone mid-take
                self._note(f"midi out: {e}")
                if kind == "off" and not self._released:
                    with self._lock:
                        self._schedule.unsent_off(ch, a)
        return n

    def _note(self, text: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(text)
