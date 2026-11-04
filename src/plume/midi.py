"""Live MIDI input: what a player's keys said, instead of what a mic could tell.

A mic in a room hears the band as loudly as the player, so pitch and timing have
to be guessed.  A MIDI port says exactly which note, when, and how hard.  The band
understands the player through one thing, :class:`~plume.listen.features.Frame`,
so this module's whole job is to produce those from notes on the same hop grid as
:class:`~plume.listen.features.FeatureExtractor`.  The beat tracker, the harmony
tracker, the count-in and the bar summaries are the same code either way.

Three parts:

- :class:`MidiInput` holds the port (``mido`` with ``python-rtmidi``, the optional
  ``midi`` extra, imported only when asked for).  It stamps each message with the
  stream clock on the library's thread, and notices when the port goes away.
- :class:`MidiFrames` turns stamped events into frames and clock beats.
- :class:`MidiMasker` stands in for ``SelfMasker``: a key press cannot be the band
  coming back through the room, so nothing is ever masked.

MIDI input makes no sound and needs no consent of its own.  The session it feeds
is still gated by ``realtime.check_audio_consent``.
"""
from __future__ import annotations

import collections
import importlib
import math
import threading

import numpy as np

from .listen.features import (CHROMA_BASS_HALFLIFE_ST, CHROMA_BASS_MAX_MIDI,
                              CHROMA_BASS_MIN_MIDI, HOP, ONSET_MIN_SEP_S,
                              RMS_FLOOR_DB, Frame)

#: How often the port is checked for having gone away, in seconds.  The library
#: has no disconnect callback, so a missing name in the port list is the signal.
POLL_S = 2.0
#: How long to wait on a port closing, because a driver for a dead device may never
#: return.  It is left to finish on its own after that.
CLOSE_TIMEOUT_S = 1.0
#: The same for a port opening.  A port still opening after this counts as lost, and
#: counts as back if the open ever finishes.
OPEN_TIMEOUT_S = 1.0
#: How long a list of the ports may take before the session starts.  This one is on
#: the main thread, ahead of the cue socket, so nothing else could end the wait.
LIST_TIMEOUT_S = 5.0

#: The channel General MIDI keeps for drums, counted from 0: hits, not pitches.
DRUM_CHANNEL = 9

#: How long a released note keeps counting, fading to nothing.  Real strings ring
#: and the trackers were built on that: a stab that vanished at key-up would leave
#: the bar mostly unvoiced, and the key tracker refuses to commit on that.
RING_S = 0.4

#: Velocity to level.  A note has no level in a room, but the band sets its own
#: level from the player's (``bus_calibration_step``), so this sits where a player
#: on a mic does: about -27 dBFS at a middling velocity.  Velocity 1 stays above
#: ``MindParams.silence_rms_db``, because any note at all is someone playing.
LEVEL_MIN_DB = -42.0
LEVEL_MAX_DB = -12.0

#: A strike, in the units of ``Frame.flux``.  A mic's flux grows with how many notes
#: land together, and the tempo search needs that accent to tell a beat from what
#: falls between beats.  Measured on chords on the beat with single notes between:
#: at 120 and 140 bpm a flat value here read 2/3 of the tempo in most takes, and this
#: slope read the right tempo in all of them.
_FLUX_MIN = 2.0
_FLUX_SPAN = 10.0
_FLUX_MAX_NOTES = 4.0

#: Clock ticks per quarter note, by the MIDI standard.
_TICKS_PER_BEAT = 24
#: Beat gaps averaged for the tempo: enough to calm arrival jitter, few enough to
#: follow a tempo change within a bar.
_CLOCK_GAPS = 4
#: A gap between two ticks longer than this is a pause, not a tempo, so it is kept
#: out of the average.  The slowest clock believed, 30 bpm, ticks every 83 ms.
_CLOCK_PAUSE_S = 0.25
#: No tick for this long and the clock is over: nothing is stating the tempo now,
#: and whatever it last said must not outlive it.
_CLOCK_TIMEOUT_S = 1.0


class MidiError(RuntimeError):
    """A MIDI problem with a sentence the player can act on."""


def _no_library(detail: object) -> MidiError:
    return MidiError("live MIDI needs mido and python-rtmidi: "
                     f"pip install -e '.[midi]' ({detail})")


def _import_mido():
    try:
        return importlib.import_module("mido")
    except ImportError as e:
        raise _no_library(e) from None


def list_ports() -> list[str]:
    """The names of the MIDI input ports, or a :class:`MidiError` if the driver will
    not say within :data:`LIST_TIMEOUT_S`.  Opens none of them."""
    answer: list = []

    def ask() -> None:
        try:
            answer.append(_port_names())
        except MidiError as e:
            answer.append(e)

    asking = threading.Thread(target=ask, name="plume-midi-list", daemon=True)
    asking.start()
    asking.join(LIST_TIMEOUT_S)
    if not answer:
        raise MidiError(f"the MIDI driver did not list its ports in {LIST_TIMEOUT_S:g} "
                        "seconds, so nothing started; unplug the keyboard and try again")
    if isinstance(answer[0], MidiError):
        raise answer[0]
    return answer[0]


def _port_names() -> list[str]:
    mido = _import_mido()
    try:
        return list(mido.get_input_names())
    except ImportError as e:         # mido is there, its rtmidi backend is not
        raise _no_library(e) from None
    except Exception as e:           # noqa: BLE001 - a backend's own error type
        raise MidiError(f"could not list MIDI ports: {e}") from None


def resolve_port(want: str | None) -> str:
    """The one port ``want`` means: an exact name, a substring, or ``auto``."""
    names = list_ports()
    if not names:
        raise MidiError("no MIDI input ports found; is the keyboard plugged in "
                        "and switched on?")
    known = ", ".join(names)
    if want is None or want == "auto":
        if len(names) == 1:
            return names[0]
        raise MidiError(f"more than one MIDI port, so name one with --midi: {known}")
    if want in names:
        return want
    hits = [n for n in names if want.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise MidiError(f"--midi {want} matches more than one port: {', '.join(hits)}")
    raise MidiError(f"no MIDI port matching {want}; there is: {known}")


def velocity_db(velocity: float) -> float:
    """The level a note of this velocity counts as (see LEVEL_MIN_DB)."""
    return LEVEL_MIN_DB + (LEVEL_MAX_DB - LEVEL_MIN_DB) * float(velocity) / 127.0


def _translate(msg) -> tuple | None:
    """A ``mido`` message as ``(kind, a, b, channel)``, or None if it means nothing here."""
    kind, ch = msg.type, getattr(msg, "channel", 0)
    if kind == "note_on":
        if msg.velocity > 0:
            return ("on", msg.note, msg.velocity, ch)
        return ("off", msg.note, 0, ch)
    if kind == "note_off":
        return ("off", msg.note, 0, ch)
    if kind == "control_change":
        if msg.control == 64:
            return ("pedal", msg.value, 0, ch)
        if msg.control in (120, 123):      # all sound off, all notes off
            return ("all_off", 0, 0, ch)
        return None
    if kind == "clock":
        return ("tick", 0, 0, 0)
    if kind in ("start", "stop", "continue"):
        return (kind, 0, 0, 0)
    return None


class MidiInput:
    """One named port, kept open across unplugging, feeding a queue of events.

    ``clock`` gives the stream time now.  ``offset_s`` moves stamps onto the clock
    mic frames are on: a sound reaches the analysis one input latency after it was
    made, a key press does not, and everything downstream (the placement
    compensation above all) was measured against the mic.

    Nothing here may raise into the library's thread or the session: a keyboard
    falling out of a socket costs that keyboard, never the take.  While the port is
    gone the band hears silence, and comes back when the port does.
    """

    def __init__(self, port: str, *, clock, on_change=None, poll_s: float | None = None):
        self.port = port
        self.offset_s = 0.0
        self.connected = False
        #: ``(t_s, kind, a, b, channel)``, oldest first.  Bounded, so a stalled
        #: reader costs old notes rather than memory.
        self.events: collections.deque = collections.deque(maxlen=8192)
        self._clock = clock
        self._on_change = on_change
        self._poll_s = poll_s
        self._handle = None
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._opener: threading.Thread | None = None
        self._gone_since_open = False

    def start(self) -> "MidiInput":
        """Open the port and start watching it.  A failure here is only a port
        that is not back yet."""
        if self._open_bounded():
            self.connected = True
        else:
            self._mark("lost")
        poll = POLL_S if self._poll_s is None else self._poll_s
        if poll > 0.0:
            self._thread = threading.Thread(target=self._watch, args=(poll,),
                                            name="plume-midi", daemon=True)
            self._thread.start()
        return self

    def check(self) -> None:
        """One look at the port list: notice the port going, or coming back."""
        try:
            names = _import_mido().get_input_names()
        except Exception:                  # noqa: BLE001 - not knowing is not gone
            return
        present = self.port in names
        if not present:
            self._gone_since_open = True
        if self.connected and not present:
            self.connected = False
            self._mark("lost")             # first: the close may never come back
            self._close_handle(CLOSE_TIMEOUT_S)
        elif not self.connected and not present and self._handle is not None:
            self._close_handle(CLOSE_TIMEOUT_S)   # a late open, onto a port now gone
        elif not self.connected and present and self._open_bounded():
            self.connected = True
            self._mark("back")

    def close(self, *, timeout: float = CLOSE_TIMEOUT_S) -> None:
        self._closing.set()
        self._close_handle(timeout)

    # ------------------------------------------------------------ internals

    def _open_bounded(self) -> bool:
        """Whether the port is open, waiting on it for a bounded time.  One try at a
        time: a stuck one is waited out, never stacked, and if it finishes late its
        handle is the open port, unless the port went away after that try began."""
        if self._opener is not None and self._opener.is_alive():
            return False
        if self._handle is not None and self._gone_since_open:
            self._close_handle(CLOSE_TIMEOUT_S)
        if self._handle is None:
            self._gone_since_open = False
            self._opener = threading.Thread(target=self._open, name="plume-midi-open",
                                            daemon=True)
            self._opener.start()
            self._opener.join(OPEN_TIMEOUT_S)
        return self._handle is not None

    def _open(self) -> bool:
        try:
            self._handle = _import_mido().open_input(self.port, callback=self._message)
        except Exception:                  # noqa: BLE001 - see the class docstring
            self._handle = None
            return False
        if self._closing.is_set():         # the session ended during that call
            self._drop_handle()
            return False
        return True

    def _close_handle(self, timeout: float) -> None:
        closer = threading.Thread(target=self._drop_handle, daemon=True)
        closer.start()
        closer.join(timeout)

    def _drop_handle(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:              # noqa: BLE001 - it is already gone
                pass

    def _message(self, msg) -> None:
        """Runs on the library's thread."""
        try:
            event = _translate(msg)
            if event is not None:
                self.events.append((self._clock() + self.offset_s, *event))
        except Exception:                  # noqa: BLE001 - see the class docstring
            pass

    def _mark(self, kind: str) -> None:
        try:
            self.events.append((self._clock() + self.offset_s, kind, 0, 0, 0))
            if self._on_change is not None:
                self._on_change(kind)
        except Exception:                  # noqa: BLE001 - see the class docstring
            pass

    def _watch(self, poll: float) -> None:
        while not self._closing.wait(poll):
            self.check()


class MidiFrames:
    """``FeatureExtractor``'s interface, reading a queue of events instead of audio.

    ``process`` takes the same mic blocks, only to count them, so its frames fall
    on the same times.  An event lands on the first frame at or after its stamp,
    which is where a mic's frame would first contain it.  ``log`` receives each
    thing that happened, in stream time, for ``midi.ndjson``.

    A MIDI clock only counts with ``follow_clock``.  Plenty of gear sends clock that
    the player is not playing to, and following it overrides what they play.
    ``on_clock`` hears ``"ignored"`` the first time an unfollowed clock shows up, and
    ``"stopped"`` when a followed one ends.
    """

    def __init__(self, events, *, sr: int = 48000, log=None, follow_clock: bool = False,
                 on_clock=None):
        self._events = events
        self._sr = int(sr)
        self._log = log
        self._follow_clock = bool(follow_clock)
        self._on_clock = on_clock
        self._samples = 0
        self._k = 0
        self._last_onset = -math.inf
        #: (channel, note) -> [velocity, released at or None]
        self._sounding: dict[tuple[int, int], list] = {}
        self._pedal: dict[int, bool] = {}
        self._pedalled: set[tuple[int, int]] = set()
        self._ticks = 0
        self._beat = 0
        self._beat_times: collections.deque = collections.deque(maxlen=_CLOCK_GAPS + 1)
        #: Beats to state, in order; None where the clock ended.
        self._clock_due: list[tuple[float, float, int] | None] = []
        self._clock_stated = False       # the tracker is holding what it was told
        self._clock_ignored = False      # said so already
        self._clock_stopped = False      # after Stop, ticks mean nothing until told
        self._last_tick = -math.inf

    def process(self, block) -> list[Frame]:
        self._samples += int(np.asarray(block).shape[0])
        out = []
        while self._samples >= HOP:
            self._samples -= HOP
            self._k += 1
            out.append(self._frame(self._k * HOP / self._sr))
        return out

    def apply_clock(self, tracker) -> None:
        """Hand any clock beats that have passed to the beat tracker."""
        due, self._clock_due = self._clock_due, []
        for beat in due:
            if beat is None:
                tracker.clock_released()
            else:
                tracker.clock_beat(*beat)

    # ------------------------------------------------------------ internals

    def _write(self, doc: dict) -> None:
        if self._log is not None:
            self._log(doc)

    def _release(self, key: tuple[int, int], t: float) -> None:
        entry = self._sounding.get(key)
        if entry is not None and entry[1] is None:
            entry[1] = t
            self._write({"t_s": round(t, 4), "type": "off", "note": key[1],
                         "channel": key[0]})

    def _release_all(self, t: float, channel: int | None = None) -> None:
        for key in list(self._sounding):
            if channel is None or key[0] == channel:
                self._release(key, t)
                self._pedalled.discard(key)

    def _clock_over(self, t: float) -> None:
        """The clock has gone quiet: let the tracker read the player again."""
        self._last_tick = -math.inf
        self._beat_times.clear()
        if self._clock_stated:
            self._clock_stated = False
            self._clock_due.append(None)
            self._write({"t_s": round(t, 4), "type": "clock_stopped"})
            if self._on_clock is not None:
                self._on_clock("stopped")

    def _tick(self, t: float) -> None:
        if not self._follow_clock:
            if not self._clock_ignored:
                self._clock_ignored = True
                self._write({"t_s": round(t, 4), "type": "clock_ignored"})
                if self._on_clock is not None:
                    self._on_clock("ignored")
            return
        if self._clock_stopped:
            return
        if t - self._last_tick > _CLOCK_PAUSE_S:
            self._beat_times.clear()
        self._last_tick = t
        if self._ticks % _TICKS_PER_BEAT == 0:
            self._beat_times.append(t)
            if len(self._beat_times) >= 3:       # two gaps before a tempo is claimed
                times = self._beat_times
                gap = (times[-1] - times[0]) / (len(times) - 1)
                if gap > 0.0:
                    self._clock_due.append((t, 60.0 / gap, self._beat % 4))
                    self._clock_stated = True
            self._beat += 1
        self._ticks += 1

    def _apply(self, event: tuple) -> float:
        """One event; returns the velocity of a strike, else 0."""
        t, kind, a, b, ch = event
        if kind == "on":
            key = (ch, a)
            self._pedalled.discard(key)
            self._sounding[key] = [b, None]
            self._write({"t_s": round(t, 4), "type": "on", "note": a, "velocity": b,
                         "channel": ch})
            if ch == DRUM_CHANNEL:            # a hit: it has no key-up to wait for
                self._sounding[key][1] = t
            return float(b)
        if kind == "off":
            key = (ch, a)
            if ch == DRUM_CHANNEL:
                pass
            elif self._pedal.get(ch):
                self._pedalled.add(key)
            else:
                self._release(key, t)
        elif kind == "pedal":
            down = a >= 64
            if self._pedal.get(ch) and not down:
                for key in [k for k in self._pedalled if k[0] == ch]:
                    self._release(key, t)
                    self._pedalled.discard(key)
            self._pedal[ch] = down
        elif kind == "all_off":
            self._release_all(t, ch)
        elif kind == "tick":
            self._tick(t)
        elif kind == "start":
            self._ticks = self._beat = 0
            self._beat_times.clear()
            self._clock_stopped = False
            self._last_tick = -math.inf
        elif kind == "stop":
            self._clock_stopped = True
            self._clock_over(t)
        elif kind == "continue":                  # the same place, a new stretch
            self._clock_stopped = False
            self._beat_times.clear()
            self._last_tick = -math.inf
        elif kind in ("lost", "back"):
            # No key-up is coming for anything held when the cable went.
            self._release_all(t)
            self._pedal.clear()
            self._write({"t_s": round(t, 4), "type": kind})
            # Nor any telling where the bar got to.
            self._clock_over(t)
            self._ticks = self._beat = 0
            self._clock_stopped = False
        return 0.0

    def _frame(self, t: float) -> Frame:
        strike = 0.0
        while self._events and self._events[0][0] <= t:
            strike += self._apply(self._events.popleft())
        if t - self._last_tick > _CLOCK_TIMEOUT_S and self._last_tick > -math.inf:
            self._clock_over(t)
            self._ticks = self._beat = 0

        onset = strike > 0.0 and t - self._last_onset >= ONSET_MIN_SEP_S
        strength = min(strike / 127.0, _FLUX_MAX_NOTES)
        flux = 0.0
        if onset:
            self._last_onset = t
            flux = _FLUX_MIN + _FLUX_SPAN * strength

        chroma = np.zeros(12, dtype=np.float32)
        low = np.zeros(12, dtype=np.float32)
        loudest = total = hz = 0.0
        for key, (vel, released) in list(self._sounding.items()):
            weight = vel / 127.0
            if released is not None:
                weight *= 1.0 - (t - released) / RING_S
                if weight <= 0.0:
                    del self._sounding[key]
                    continue
            loudest = max(loudest, weight)
            note = key[1]
            if key[0] == DRUM_CHANNEL:
                continue
            total += weight
            hz += weight * 440.0 * 2.0 ** ((note - 69) / 12.0)
            chroma[note % 12] += weight
            if CHROMA_BASS_MIN_MIDI <= note <= CHROMA_BASS_MAX_MIDI:
                low[note % 12] += weight * 0.5 ** (
                    (note - CHROMA_BASS_MIN_MIDI) / CHROMA_BASS_HALFLIFE_ST)
        if total > 0.0:
            chroma /= total
            low /= total
        return Frame(
            t_s=t,
            rms_db=velocity_db(loudest * 127.0) if loudest > 0.0 else RMS_FLOOR_DB,
            flux=flux,
            onset=onset,
            onset_strength=flux if onset else 0.0,
            chroma=chroma,
            # Audio's centroid sits above the fundamental, and the register
            # estimate halves it to allow for that.
            centroid_hz=2.0 * hz / total if total > 0.0 else 0.0,
            chroma_low=low,
            # Every press clears the tap-start floors at any velocity: a count on
            # a key is never in doubt, unlike a knock across a room.
            attack=1.0 + min(strength, 1.0) if onset else 0.0,
            attack_breadth=1.0 if onset else 0.0,
        )


class MidiMasker:
    """``SelfMasker``'s interface for a source the band cannot leak into."""

    masked_frames = 0
    #: It never hears the room, so it has no reading of the band in the mic.
    coupling_db = None
    coupling_measured = False

    def mask(self, frames, machine_rms_db):  # noqa: ARG002 - the shared signature
        return frames, [False] * len(frames)

    def set_latency_hops(self, hops: int) -> "MidiMasker":  # noqa: ARG002
        return self
