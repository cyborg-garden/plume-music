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

* buzz -- the last 80 ms are hum when (1) the lines of a mains grid (50 or 60 Hz and
  its harmonics up to ~500 Hz) carry a good share of their energy, (2) the
  strongest of them sits on its grid, turning evenly, and (3) it holds a steady
  level across the window.  Led by the fundamental, a 50/60 Hz grid may drift up
  to 0.45 Hz (generators and inverters run a little off; beyond 0.3 Hz the drift
  must also be even).  Led by a harmonic, the lines must sit on the grid within half a
  hertz and a second line sharing no factor with the leader must drift in step:
  one held note's partials can sit on grid lines by chance, but they are all
  multiples of that note.  A rectifier's all-even buzz (100/120 Hz and up) counts
  as its own grid.
  Level and clipping are not the test.  Hum is a fixed tone that does not decay;
  a pluck has an attack and decays, and an instrument's notes land on a mains
  line only by tuning accident.  So a clipping bass, a drop-tuned chug and a hot
  input gain all stay heard, and a quiet steady hum well under full scale is
  still caught.  The monitor ducks after two hum hops in a row (the window has to
  fill, so ~80-110 ms from the onset; the ceiling in ``realtime.mix_monitor``
  holds the speakers at -1 dBFS meanwhile) and comes back after half a second
  without hum.  The player is told once most of the last 100 ms, or a third of the
  last second, were hum, or the monitor has been held off for a second.
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
#: Mains grids.  Every line of a grid is demodulated over one period of its
#: fundamental (960 samples at 50 Hz, 800 at 60 Hz), which cancels all the grid's
#: other lines exactly.
MAINS_BASES_HZ = (50.0, 60.0)
#: A full-wave rectifier buzz is all even lines (100 or 120 Hz and up, no 50/60): its
#: own grid, led by its own fundamental, and held to GRID_TOL_HZ (a sharp G2 sits
#: just under 100 Hz).
RECTIFIED_BASES_HZ = (100.0, 120.0)
GRID_TOP_HZ = 500.0
#: How far a fundamental-led buzz may drift, period by period: within GRID_TOL_HZ
#: anything goes (a clipping fault jitters its phase); out to FUNDAMENTAL_TOL_HZ
#: (generators, inverters) the drift must also be even to EVEN_HZ, which two notes
#: beating near mains are not.  The nearest A440 notes (G1 at 49.0 Hz, A#1 58.27,
#: B1 61.74 ...) all sit further off.
GRID_TOL_HZ = 0.3
FUNDAMENTAL_TOL_HZ = 0.45
EVEN_HZ = 0.1
#: A buzz led by a harmonic must sit on the grid itself: each of its lines within this
#: many hertz, period by period.  Off-grid (generator) buzz led by a harmonic is not
#: caught; a sharp low note's partials line up with the upper lines too easily.
HARMONIC_TOL_HZ = 0.5
#: A buzz led by a harmonic needs a second line, at least this strong against the
#: loudest, drifting with it to within this much.
HARMONIC_SECOND = 0.05
HARMONIC_AGREE_HZ = 0.05
#: Share of the window's energy the grid's lines must carry between them.
LINE_SHARE = 0.4
#: Steady: the line's period-by-period level stays within this many dB.
STEADY_DB = 3.0
#: Quieter than this a hum is not worth a word.
HUM_FLOOR_DB = -40.0
#: Hum hops in a row before the monitor ducks; hum hops out of the last
#: SAY_OF before the player is told; the hum-free time that ends it.
DUCK_HOPS = 2
SAY_HOPS, SAY_OF = 8, 10
#: ... or a hum that keeps the monitor off but flickers at the test's edge: this many
#: hum hops out of the last second.
SAY_LONG_HOPS, SAY_LONG_OF = 30, 100
#: ... and, whatever the counts say, a monitor held off this long is always said:
#: one blip holds it off for CLEAN_S, so only a hum that keeps coming back gets here.
SAY_MUTED_S = 1.0
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
        # One block per grid: e^(-i 2 pi k b t) for each of its lines k b, over a
        # whole number of the fundamental's periods.
        self._grids = []
        for b in (*MAINS_BASES_HZ, *RECTIFIED_BASES_HZ):
            period = int(round(sr / b))
            periods = self.win // period
            if periods < 3 or abs(sr / period - b) > 1e-9:
                continue
            ks = np.arange(1, int(GRID_TOP_HZ // b) + 1)
            t = np.arange(periods * period) / sr
            self._grids.append((b, period, periods, ks,
                                np.exp(-2j * np.pi * np.outer(ks * b, t))))
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
        self._long: collections.deque = collections.deque(maxlen=SAY_LONG_OF)
        self._muted_s = 0.0
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
        self._long.append(hum)
        self._muted_s = self._muted_s + HOP_S if self.slot[0] > 0.5 else 0.0
        if (sum(self._recent) >= SAY_HOPS or sum(self._long) >= SAY_LONG_HOPS
                or self._muted_s >= SAY_MUTED_S):
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
        """Whether the window is a steady buzz on a mains grid.

        Demodulate every line of a grid over one period of its fundamental at a time:
        a period-long average cancels all the grid's other lines exactly, so what is
        left is each line's own amplitude and phase.  On the grid the phase holds
        still from period to period; a note a hertz away turns through a fiftieth of
        a cycle every period, which is plain to see in 80 ms where a spectrum cannot.
        """
        w = self._ring
        energy = float(np.mean(w * w))
        if energy <= 10.0 ** (HUM_FLOOR_DB / 10.0):
            return False
        for b, period, periods, ks, osc in self._grids:
            seg = w[-periods * period:]
            z = (osc * seg).reshape(len(ks), periods, period).mean(axis=2)
            power = np.abs(z) ** 2                       # [line, period]
            line_power = power.mean(axis=1)
            if 2.0 * float(line_power.sum()) < LINE_SHARE * energy:
                continue
            i = int(np.argmax(line_power))
            p = power[i]
            if 10.0 * math.log10(float(p.max()) / (float(p.min()) + 1e-30)) > STEADY_DB:
                continue
            # Each line's turn per period, as the grid's own drift (the fundamental's
            # offset in Hz).  Every period's turn must hold, not just their mean: a
            # neighbouring note beating against a line swings its phase about.
            turns = np.angle(z[:, 1:] * np.conj(z[:, :-1]))
            drift = turns / (2.0 * np.pi * period / self.sr) / ks[:, None]
            worst = np.max(np.abs(drift), axis=1)
            k = int(ks[i])
            if k == 1:
                if float(worst[i]) <= GRID_TOL_HZ:
                    return True
                if b in RECTIFIED_BASES_HZ:
                    continue
                # Off the grid (a generator, an inverter) the drift must also be even:
                # two notes beating near mains swing it about.
                if (float(worst[i]) <= FUNDAMENTAL_TOL_HZ
                        and float(np.ptp(drift[i])) <= EVEN_HZ):
                    return True
                continue
            # Lines in absolute hertz: tolerances scaled up by k would let any partial
            # within a few hertz of an upper line through.
            line_worst = worst * ks
            if float(line_worst[i]) > HARMONIC_TOL_HZ:
                continue
            # Led by a harmonic.  A held note's partials sit at multiples of one
            # frequency, so it takes a second grid line sharing no factor with the
            # leader (the fundamental itself, most often), drifting in step.
            mean = drift.mean(axis=1)
            for j in range(len(ks)):
                if (math.gcd(int(ks[j]), k) == 1 and line_power[j] >= HARMONIC_SECOND * line_power[i]
                        and float(line_worst[j]) <= HARMONIC_TOL_HZ
                        and abs(float(mean[j] - mean[i])) <= HARMONIC_AGREE_HZ):
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
