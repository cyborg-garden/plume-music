"""Notice when the input itself has gone wrong, on the analysis thread.

A loose or unplugged cable on a direct input does two things the band should never
pass on.  It can buzz: mains hum, often at full scale, which Hear me would send
straight to the speakers.  And afterwards it can go dead, so plume hears nothing at
all while she thinks she is still playing.

This module only looks.  It runs on the analysis thread, never the audio callback,
and its one way of reaching the callback is ``slot[0]``: a plain float the callback
reads to duck the monitor (see ``realtime.mix_monitor``).  The band keeps playing
either way; what changes is only whether her raw input is heard, plus a short line
for the live page and the decision log (``fault``: None, "buzz" or "silent").

The tests:

* buzz -- the last 80 ms are hum when (1) one line carries a good share of their
  energy, (2) that line sits within 0.3 Hz of a mains frequency or one of its first
  harmonics (50/60, 100/120, ...), and (3) it holds a steady level across the window.
  Level and clipping are not the test.  Hum is a fixed tone that does not decay;
  a pluck has an attack and decays, and an instrument's notes land on a mains
  frequency only by tuning accident.  So a clipping bass, a drop-tuned chug and a
  hot input gain all stay heard, and a quiet steady hum well under full scale is
  still caught.  The monitor ducks after two hum hops in a row (the window has to
  fill, so ~80-110 ms from the onset; the ceiling in ``realtime.mix_monitor``
  holds the speakers at -1 dBFS meanwhile) and comes back after half a second
  without hum.  The player is told once most of the last 100 ms were hum.
* silent -- after she has played, the input stays near silence for at least four
  bars and eight seconds, while the take is not ending.  That is only said, never
  acted on: a dead input and a long rest look the same to the band, and the band
  already has its own idea of what a rest means.
"""
from __future__ import annotations

import collections
import math

import numpy as np

#: Analysis hop, and the window the hum test looks at.
HOP_S = 0.010
WINDOW_S = 0.080
#: Mains and the harmonics a buzz is made of.  Each period is a whole number of
#: samples at 48 kHz, which the demodulation below relies on.
MAINS_HZ = (50.0, 60.0, 100.0, 120.0, 150.0, 200.0, 240.0)
#: How far from mains the line may sit: grids hold well inside this; an instrument's
#: note lands this close to mains only by tuning accident.
MAINS_TOL_HZ = 0.3
#: Share of the window's energy that line must carry.
LINE_SHARE = 0.4
#: Steady: the line's period-by-period level stays within this many dB.
STEADY_DB = 3.0
#: Quieter than this a hum is not worth a word.
HUM_FLOOR_DB = -40.0
#: Hum hops in a row before the monitor ducks; hum hops out of the last
#: SAY_OF before the player is told; the hum-free time that ends it.
DUCK_HOPS = 2
SAY_HOPS, SAY_OF = 8, 10
CLEAN_S = 0.5
#: Silence, and what counts as playing.
SILENT_DB = -50.0
PLAYING_DB = -45.0
PLAYED_S = 0.5
SILENT_BARS = 4.0
SILENT_MIN_S = 8.0

BUZZ_LINE = "Your input is buzzing. Check the cable."
SILENT_LINE = "No sound from your input for a while."
LINES = {"buzz": BUZZ_LINE, "silent": SILENT_LINE}


def monitor_slot() -> np.ndarray:
    """[duck asked (1.0) or not, monitor gain now].

    ``[1]`` is the audio callback's own (``realtime.mix_monitor``); only ``[0]`` is
    written from here.
    """
    return np.array([0.0, 1.0], dtype=np.float64)


class InputGuard:
    def __init__(self, sr: int, slot: np.ndarray | None = None, *, notices: bool = True):
        self.sr = int(sr)
        self.slot = monitor_slot() if slot is None else slot
        #: False on a MIDI take: her notes are not this input, so nothing is said.
        self.notices = notices
        self.hop = max(1, int(round(HOP_S * sr)))
        self.win = int(round(WINDOW_S * sr))
        # One row per mains line: e^(-i 2 pi m t) over the window, and its period.
        self._lines = []
        for m in MAINS_HZ:
            period = int(round(sr / m))
            periods = self.win // period
            if periods < 3 or abs(sr / period - m) > 1e-9:
                continue
            t = np.arange(periods * period) / sr
            self._lines.append((m, period, periods, np.exp(-2j * np.pi * m * t)))
        self._ring = np.zeros(self.win, dtype=np.float64)
        self._filled = 0
        self._pending = np.zeros(0, dtype=np.float64)
        self.bar_s = 2.0              # set by the planner as the tempo is known
        self.ending = False           # set while the take is stopping or ending
        self.winding = False          # set while the band fades out (lead)
        self.buzz = False
        self.silent = False
        self._run = 0
        self._recent: collections.deque = collections.deque(maxlen=SAY_OF)
        self._since_hum = math.inf
        self._quiet_s = 0.0
        self._played_s = 0.0
        self._said: set[str] = set()

    @property
    def fault(self) -> str | None:
        if not self.notices:
            return None
        return "buzz" if self.buzz else "silent" if self.silent else None

    def feed(self, x) -> str | None:
        """Look at more input.  Returns a line to say, the first time a fault is
        standing at the end of what was fed (one set and cleared inside it is not)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._pending.size:
            x = np.concatenate((self._pending, x))
        n_hops = x.size // self.hop
        self._pending = x[n_hops * self.hop:].copy()
        for i in range(n_hops):
            self._hop(x[i * self.hop:(i + 1) * self.hop])
        return self._news()

    # -- one hop

    def _hop(self, h: np.ndarray) -> None:
        n = h.size
        self._ring[:-n] = self._ring[n:]
        self._ring[-n:] = h
        self._filled = min(self.win, self._filled + n)
        db = 10.0 * math.log10(float(np.mean(h * h)) + 1e-12)
        hum = self._filled >= self.win and self._is_hum()
        self._run = self._run + 1 if hum else 0
        self._recent.append(hum)
        self._since_hum = 0.0 if hum else self._since_hum + HOP_S
        clean = self._since_hum >= CLEAN_S
        if sum(self._recent) >= SAY_HOPS:
            self.buzz = True
        elif self.buzz and clean:
            self.buzz = False
            self._said.discard("buzz")
        if self._run >= DUCK_HOPS:
            self.slot[0] = 1.0
        elif clean:
            self.slot[0] = 0.0
        # Clipping or not, any non-hum sound over the line is her playing.
        if not hum and db > PLAYING_DB:
            self._played_s += HOP_S
        self._silence(db)

    def _is_hum(self) -> bool:
        """Whether the window is one steady line sitting on a mains frequency.

        Demodulate at each mains line, one period at a time: a period-long average
        cancels every other harmonic of that line exactly, so what is left is the
        line's own amplitude and phase.  On mains the phase holds still from period
        to period; a note a hertz away turns through a fiftieth of a cycle every
        period, which is plain to see in 80 ms where a spectrum cannot see it.
        """
        w = self._ring
        energy = float(np.mean(w * w))
        if energy <= 10.0 ** (HUM_FLOOR_DB / 10.0):
            return False
        for m, period, periods, osc in self._lines:
            seg = w[-periods * period:]
            z = (seg * osc).reshape(periods, period).mean(axis=1)
            power = np.abs(z) ** 2
            if 2.0 * float(power.mean()) < LINE_SHARE * energy:
                continue
            if 10.0 * math.log10(float(power.max()) / (float(power.min()) + 1e-30)) > STEADY_DB:
                continue
            # Every period's turn, not just their mean: a neighbouring note beating
            # against the line swings the phase back and forth around zero.
            turns = np.angle(z[1:] * np.conj(z[:-1]))
            offsets_hz = turns / (2.0 * np.pi * period / self.sr)
            if float(np.max(np.abs(offsets_hz))) <= MAINS_TOL_HZ:
                return True
        return False

    def _silence(self, db: float) -> None:
        if self.ending or self.winding or not self.notices:
            self._quiet_s = 0.0
            if self.silent:
                self.silent = False
                self._said.discard("silent")
            return
        if db < SILENT_DB:
            self._quiet_s += HOP_S
        elif db > PLAYING_DB:
            self._quiet_s = 0.0
        if self.silent:
            if self._quiet_s == 0.0:
                self.silent = False
                self._said.discard("silent")
                self._played_s = 0.0
            return
        if (self._played_s >= PLAYED_S
                and self._quiet_s >= max(SILENT_BARS * self.bar_s, SILENT_MIN_S)):
            self.silent = True

    def _news(self) -> str | None:
        fault = self.fault
        if fault is None or fault in self._said:
            return None
        self._said.add(fault)
        return LINES[fault]
