"""The opinionated accompanist: harmonic commitment for follow mode (v2.6).

WHY.  The HarmonyTracker's chord belief is honest about what it hears — and
against a solo voice what it hears is one note at a time, so the per-bar
belief walks through every triad containing the sung note (a real take's log:
G, E, Em, Gm, C, E, E, C, Em, Am, Fm — one bar each).  The renderer then
re-voices every bar to that walk, and the human hears an accompanist with no
opinion — "it bounces rapidly in pitch and melody; even I, the accompanist,
could adapt to what it plays if it were consistent."  The clock already
learned this lesson (predict, don't react; commit, repair on evidence); this
module is the same posture for harmony.

WHAT.  A :class:`HarmonyGovernor` sits between the tracker's ``HarmonyState``
and everything that consumes it (mind + renderer), in the follow-mode
transports only:

- **Tracking** (before commitment): states pass through untouched — the
  tracker's graceful-degradation ladder (triad / root+fifth / key pedal) is
  unchanged, so a session that never stabilizes never behaves worse than v2.5.
- **Commit**: after ``harmony_commit_bars`` consecutive bars carrying one key
  belief, the governor commits to that key and to the style's top-weighted
  progression in it (the same pool prompted mode composes from), cycling one
  chord per bar from the commit bar.  From then on the human's notes are
  melody over a consistent accompaniment, not per-bar re-harmonization fuel.
- **Modulate**: the tracker keeps listening.  Only when its key belief sits on
  one DIFFERENT key for ``harmony_release_bars`` consecutive bars does the
  governor recommit there — a sung excursion is color; a sustained move is a
  modulation.
- **Unlocked ticks** (no bar grid): the governed chord holds, because a
  free-running grid is exactly when flapping harmony hurts most.

``harmony_commit_bars=0`` disables the governor entirely (the v2.5 behavior).
Pure: no rng, no clock, no device — a function of the observed state stream.
Plain lead mode never sees this module: there the machine composes its own
harmony.  A tap-started lead take does, through plume.bandentry.
"""
from __future__ import annotations

from dataclasses import replace

from .styles_lib import load_style
from .theory import parse_key, roman_to_chord

#: The governed chord is delivered at full confidence: the whole point of
#: commitment is that the renderer voices REAL triads, not hedged root+fifths.
_COMMITTED_CHORD_CONF = 1.0
_COMMITTED_KEY_CONF = 0.9


def progression_symbols(style: str, key_str: str) -> list[str]:
    """The style's top-weighted progression, spelled in ``key_str``.

    Mode-aware pool choice mirrors compose (major pool for major keys, minor
    otherwise, falling back to whichever exists); the TOP-WEIGHTED progression
    rather than a sampled one, because the governor must be deterministic
    given the take.  Empty when the style declares no progressions.
    """
    key = parse_key(key_str)
    pools = load_style(style).get("progressions")
    if isinstance(pools, dict):
        progs = (pools.get("major" if key.mode == "major" else "minor")
                 or pools.get("minor") or pools.get("major"))
    else:  # legacy flat list
        progs = pools
    if not progs:
        return []
    best = max(progs, key=lambda p: float(p.get("weight", 1)))
    return [roman_to_chord(str(n), key).sym for n in best["numerals"]]


class HarmonyGovernor:
    """Feed it (bar, HarmonyState) each planning step; read back the state the
    mind and renderer should believe."""

    def __init__(self, style: str, params):
        self._style = str(style)
        self._commit_bars = int(getattr(params, "harmony_commit_bars", 0))
        self._release_bars = int(getattr(params, "harmony_release_bars", 8))
        self._key: str | None = None       # committed key ("C major")
        self._chords: list[str] = []
        self._commit_bar = 0
        self._cand: str | None = None      # pre-commit candidate + streak
        self._cand_streak = 0
        self._away: str | None = None      # post-commit challenger + streak
        self._away_streak = 0
        self._last_bar: int | None = None
        self._last_chord: str | None = None

    @property
    def committed_key(self) -> str | None:
        return self._key

    def commit(self, key_str: str, bar: int | None) -> None:
        """Commit to a key the player named, rather than one heard."""
        self._commit(key_str, bar)

    def _commit(self, key_str: str, bar: int | None) -> None:
        self._key = key_str
        try:
            self._chords = progression_symbols(self._style, key_str)
        except (ValueError, KeyError, OSError):
            self._chords = []          # unknown style/key: hold key, pass chords
        self._commit_bar = bar if bar is not None else 0
        self._cand = self._away = None
        self._cand_streak = self._away_streak = 0

    def _chord_at(self, bar: int | None) -> str | None:
        if not self._chords:
            return None
        if bar is None:                 # unlocked tick: hold, never advance
            return self._last_chord or self._chords[0]
        self._last_chord = self._chords[(bar - self._commit_bar)
                                        % len(self._chords)]
        return self._last_chord

    def govern(self, bar: int | None, h, *, resting: bool = False):
        """The state to believe for this planning step (None passes through).

        ``resting``: the player sat this bar out.  Chords still advance, but the
        bar says nothing about the key, for or against a change.
        """
        if h is None or self._commit_bars <= 0:
            return h
        new_bar = bar is not None and bar != self._last_bar
        if self._key is None:
            if new_bar and h.key is not None:
                if h.key == self._cand:
                    self._cand_streak += 1
                else:
                    self._cand, self._cand_streak = h.key, 1
                if self._cand_streak >= self._commit_bars:
                    self._commit(h.key, bar)
            if self._key is None:
                if bar is not None:
                    self._last_bar = bar
                return h
        elif new_bar and not resting:
            if h.key is not None and h.key != self._key:
                if h.key == self._away:
                    self._away_streak += 1
                else:
                    self._away, self._away_streak = h.key, 1
                if self._away_streak >= self._release_bars:
                    self._commit(h.key, bar)
            else:
                self._away, self._away_streak = None, 0
        if bar is not None:
            self._last_bar = bar
        chord = self._chord_at(bar)
        if chord is None:               # style without progressions: hold the
            return replace(h, key=self._key,        # key, keep tracker chords
                           key_conf=max(h.key_conf, _COMMITTED_KEY_CONF))
        return replace(h, key=self._key,
                       key_conf=max(h.key_conf, _COMMITTED_KEY_CONF),
                       chord=chord, chord_conf=_COMMITTED_CHORD_CONF)
