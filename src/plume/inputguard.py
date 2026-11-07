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

* buzz -- a short window is faulty when it is loud and mostly below ~70 Hz: well
  above the level she has been playing at, or near full scale.  Mains hum and a
  cable fault sit there; a plucked string's energy is in its harmonics, so even a
  strum that clips the input is never "mostly below 70 Hz".  Clipping alone is not a
  fault: real takes clip on hard strums when the input gain is hot, and the ceiling
  in ``realtime.mix_monitor`` already keeps that off the speakers.  The monitor ducks
  after two faulty hops in a row (about 25-40 ms from the onset, the low-pass needing
  a window to settle) and comes back after half a second of clean ones; the player is
  told once it has lasted 100 ms, so a single low thump is never called a buzz.
* silent -- after she has played, the input stays near silence for about two bars.
  That is only said, never acted on: a dead input and a rest look the same to the
  band, and the band already has its own idea of what a rest means.
"""
from __future__ import annotations

import collections
import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

#: Analysis hop, and the window each test looks at.  A 70 Hz low-pass needs more than
#: one of its own periods to say anything true about a pick attack.
HOP_S = 0.005
WINDOW_HOPS = 4
#: Near full scale, and how many such samples in a window count as clipping.
HOT_LEVEL = 0.98
HOT_SAMPLES = 8
#: The hum test: energy below LOW_HZ, as a share of all of it.
LOW_HZ = 70.0
LOW_SHARE = 0.6
#: A clipping window needs less of a share: a flat-topped hum spreads into harmonics.
CLIP_LOW_SHARE = 0.3
JUMP_DB = 18.0
#: Before she has played, a hum has to be at least this loud to count.
FIRST_LEVEL_DB = -30.0
#: Quieter than this a hum is not a danger to anyone.
HUM_FLOOR_DB = -40.0
#: Faulty hops in a row before the monitor ducks, before the player is told, and the
#: clean time that ends it.
DUCK_HOPS = 2
BUZZ_SAY_HOPS = 20
CLEAN_S = 0.5
#: Silence, and what counts as playing.
SILENT_DB = -50.0
PLAYING_DB = -45.0
PLAYED_S = 0.5
SILENT_BARS = 2.0
SILENT_MIN_S = 3.0
#: The playing level follows her over about this long.
LEVEL_TAU_S = 2.0

BUZZ_LINE = "Your input is buzzing. Check the cable."
SILENT_LINE = "No sound from your input. Check it is plugged in."
LINES = {"buzz": BUZZ_LINE, "silent": SILENT_LINE}


def monitor_slot() -> np.ndarray:
    """[duck asked (1.0) or not, monitor gain now].

    ``[1]`` is the audio callback's own (``realtime.mix_monitor``); only ``[0]`` is
    written from here.
    """
    return np.array([0.0, 1.0], dtype=np.float64)


class InputGuard:
    def __init__(self, sr: int, slot: np.ndarray | None = None):
        self.sr = int(sr)
        self.slot = monitor_slot() if slot is None else slot
        self.hop = max(1, int(round(HOP_S * sr)))
        self._sos = butter(4, LOW_HZ, btype="low", fs=sr, output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._rest = np.zeros(0, dtype=np.float64)
        self._recent: collections.deque = collections.deque(maxlen=WINDOW_HOPS)
        self._alpha = 1.0 - math.exp(-HOP_S / LEVEL_TAU_S)
        self.level_db: float | None = None   # her recent playing level
        self.bar_s = 2.0                      # set by the planner as the tempo is known
        self.buzz = False
        self.silent = False
        self._bad_run = 0
        self._since_bad = math.inf
        self._quiet_s = 0.0
        self._played_s = 0.0
        self._said: set[str] = set()

    @property
    def fault(self) -> str | None:
        return "buzz" if self.buzz else "silent" if self.silent else None

    def feed(self, x) -> str | None:
        """Look at more input.  Returns a line to say, the first time a fault starts."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._rest.size:
            x = np.concatenate((self._rest, x))
        n_hops = x.size // self.hop
        self._rest = x[n_hops * self.hop:].copy()
        if n_hops == 0:
            return None
        body = x[:n_hops * self.hop]
        low, self._zi = sosfilt(self._sos, body, zi=self._zi)
        hops = body.reshape(n_hops, self.hop)
        lows = low.reshape(n_hops, self.hop)
        e_all = np.mean(hops * hops, axis=1)
        e_low = np.mean(lows * lows, axis=1)
        hot = np.sum(np.abs(hops) >= HOT_LEVEL, axis=1)
        news = None
        for e, el, h in zip(e_all, e_low, hot):
            said = self._hop(float(e), float(el), int(h))
            news = news or said
        return news

    def _hop(self, e: float, e_low: float, hot: int) -> str | None:
        self._recent.append((e, e_low, hot))
        k = len(self._recent)
        e_w = sum(r[0] for r in self._recent) / k
        el_w = sum(r[1] for r in self._recent) / k
        hot_w = sum(r[2] for r in self._recent)
        db = 10.0 * math.log10(e_w + 1e-12)
        share = el_w / (e_w + 1e-12)
        ref = FIRST_LEVEL_DB if self.level_db is None else self.level_db + JUMP_DB
        humming = db >= max(ref, HUM_FLOOR_DB) and share >= LOW_SHARE
        clipped_hum = hot_w >= HOT_SAMPLES and share >= CLIP_LOW_SHARE
        bad = humming or clipped_hum
        self._bad_run = self._bad_run + 1 if bad else 0
        self._since_bad = 0.0 if bad else self._since_bad + HOP_S
        if self._bad_run >= BUZZ_SAY_HOPS:
            self.buzz = True
        elif self.buzz and self._since_bad >= CLEAN_S:
            self.buzz = False
            self._said.discard("buzz")
        clean = self._since_bad >= CLEAN_S
        # Only her playing moves her level, never a fault and never a rest.
        if clean and hot_w == 0 and db > PLAYING_DB:
            self.level_db = db if self.level_db is None else (
                self.level_db + self._alpha * (db - self.level_db))
            self._played_s += HOP_S
        if self._bad_run >= DUCK_HOPS:
            self.slot[0] = 1.0
        elif clean:
            self.slot[0] = 0.0
        self._silence(db)
        return self._news()
    def _silence(self, db: float) -> None:
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
