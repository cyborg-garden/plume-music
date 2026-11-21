"""A style from how she plays: the evidence, and the rule that scores each style on it.

Tempo says little about a style: seven of the nine bundled styles can be played at 85
bpm.  What tells them apart is in the playing.  Five things are heard of it, every one
counted from her strokes, when she strikes and how, and none from how loud she is, so
the input gain moves a reading only where it loses strokes: near the input's noise
floor, where the quietest strokes no longer stand out of the hiss:

    strokes   strokes per beat (bandentry.StrokeEar)
    swing     where her off-beat eighths fall in the beat: 0.5 straight, 0.67 a triplet
    offbeat   the share of her strokes that fall off the beat
    steady    the share of the gaps between her strokes that are the commonest length
    strum     the share of her strokes that are strummed chords (bandentry.FolkEar)

THE RULE (:func:`score_styles`).  Every style whose tempo range holds the tempo is a
candidate, and so is the tempo's own default (:func:`tempo_default`), which is the
whole answer while nothing has been heard of her: at the count, and on a room mic.
Each style has a band for the things heard that are its own (:data:`PROFILES`).  A
candidate starts at 0 and loses up to 1 for each thing heard that falls outside its
band: nothing inside it, 1 once it is a full :data:`SOFT` step out, in proportion
between.  What a style has no band for costs it nothing, and what was not heard costs
nobody anything.  Then:

- Every style but trap and midwest, which are told by how many strokes she plays, has
  what makes it that style (:data:`NEEDS`): jazz its swing, synthwave a busy, even
  pulse, lofi uneven gaps, house, hip hop and r&b strokes off the beat, folk a strum.
  A style is not a candidate until that has been heard inside its band.  So nothing
  is jazz on a light swing or on none heard, nothing is folk without a strum, and notes
  given no times, a MIDI input that only counts them, choose only between busy and
  sparse.  Given their times, notes are placed in the beat as strokes are.
- A strum counts against every style but folk.
- The best score wins, but scores within :data:`MARGIN` of the best are a near-tie:
  what was heard does not tell those styles apart.  A near-tie goes to the tempo's
  default if it is among them, and otherwise to the one listed first in
  :data:`PROFILES`, the style that asks the most of her playing.
- When every candidate loses a whole point or more (:data:`NO_FIT`), her playing is
  none of these styles' and the tempo's default answers.  Plain playing, a note or a
  chord on each beat, fits none, so it gets what the tempo and key give it.

The bands are few and wide on purpose, and each one says in a word or two what the
style is.  They were written from that description and then adjusted by hand against
270 generated solo parts and looked at, not adjusted, against 40 recorded acoustic
guitar excerpts whose labels map onto three of the styles.  The generated parts are 45
written figures (nine styles by five instruments) at three tempi and two keys each,
so the count on them is how well the rule knows those figures, not a forecast for a
real player.  Some of those figures are the same figure in two styles: straight even
eighths are written for folk, house, synthwave and midwest alike, and a note on each
beat for folk and jazz, so only the tempo and key can tell those apart.  The light
swing that hip hop, r&b and lofi often have is not asked of them, because a player of
those may well play straight: a swing only counts against the styles that are
straight.  With the true tempo and key the rule names the generated part's own style
for 172 of 270 (the rule before this one, on tempo, mode and strokes: 105) and the
recorded excerpt's for 7 of 24 (before: 8).  Moving any one band edge by 15%, or a
swing edge by 0.02, leaves 164 to 174 of 270 and 5 to 7 of 24.  Chord colour
(sevenths against triads) would separate more and is not here: the live chroma does
not carry it.

:class:`RhythmEar` measures swing, offbeat and steady.  Pure, cheap and bounded: it is
fed one bar of stroke times at a time on the planning thread, and keeps two histograms
however long the take.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .listen.features import HOP, WIN

#: A frame's time is the end of its window, and a stroke that peaks in it sits in the
#: middle of that window on average: measured on generated parts with every note on a
#: known grid, the median stroke read 18 to 21 ms after its note.
STROKE_LAG_S = WIN / 2.0 / 48000.0
#: A MIDI note is stamped on the first frame at or after it: half a hop late on average.
NOTE_LAG_S = HOP / 2.0 / 48000.0
#: Strokes are counted by where in the beat they fall, in this many equal bins.  A hop is
#: a 36th of a beat at 160 bpm, so finer bins than this would hold nothing more.
PHASE_BINS = 48
#: On the beat: within an eighth of a beat of it, either side.
_ON_BEAT = 0.125
#: The off-beat eighth falls in these bins: at 0.5 of the beat played straight, at 0.67
#: as a triplet swing.  The window starts past the second sixteenth (0.25) and the middle
#: note of a triplet (0.33), and ends at 0.71, half way from a triplet swing to the last
#: sixteenth (0.75).
_AND_BINS = (20, 34)
#: The last sixteenth of the beat, with room for a late one.  A player's timing spreads
#: her strokes by a few hundredths of a beat, so a late-ish sixteenth lands among the
#: eighths: a dotted figure, on the beat and on the last sixteenth, would read as the
#: hardest swing there is.  Strokes late in the eighths' window are swung eighths only
#: while fewer fall here than there.
_LATE_BINS = (34, 42)
#: The middle note of a triplet, a third of the way through the beat.  Strokes late in
#: the beat are swung eighths only when few fall here: with both it is triplets, and
#: that says nothing of how she would swing an eighth.
_TRIPLET_BINS = (13, 20)
_HARD_SWING = 0.6
#: She is not a metronome and neither is the count she tapped: a tempo taken from four
#: knocks is off by half a percent or so, and a player who holds her own tempo is a
#: tenth of a beat off the bar line by the first bar heard of her, and then slides a few
#: hundredths more every bar.  So her strokes are placed on her own eighths.  The first
#: bar finds her: the strokes within _FIND_WITHIN of a beat (of an eighth, if she plays
#: none on the beat) move the reading to where they fall on average.  Every bar after
#: follows her: the strokes within _FOLLOW_WITHIN of an eighth do the same.  Starting
#: from the bar line keeps on the beat and off it apart, and swung eighths, sixteenths
#: and triplets are all further than that from an eighth and move nothing.
_FIND_WITHIN, _FOLLOW_WITHIN = 0.125, 0.07
#: An input may also run late against the bar lines by more than that, a slow output or
#: a player behind the click, and then nothing is within _FIND_WITHIN of anything.  So a
#: bar that found nothing looks wider, up to a quarter of a beat from a beat, then from an
#: eighth, and there it takes the largest group of strokes that fall together (within
#: _FIND_TOGETHER of one of them), and at least two: one stray stroke that far out is as
#: likely a sixteenth as a late beat.  Past a quarter of a beat late, on the beat and
#: off it cannot be told apart.
_FIND_WIDE, _FIND_TOGETHER = 0.25, 0.06
#: Fewer off-beat eighths than this say nothing about how she swings them.
_SWING_MIN_STROKES = 6
#: The gaps between her strokes are counted by their length in twelfths of a beat, the
#: last bin holding everything from four beats up.
GAP_BINS = 49
#: Fewer gaps than this say nothing about how even they are.
_STEADY_MIN_GAPS = 8


@dataclass(frozen=True)
class Playing:
    """What was heard of how she plays.  None: not heard, or not hers alone to hear."""
    strokes: float | None = None
    swing: float | None = None
    offbeat: float | None = None
    steady: float | None = None
    strum: float | None = None


def _together(off: np.ndarray, within: float, *, long_short: bool = False) -> float | None:
    """Where the group of ``off`` that fall together lies, among those inside ``within``:
    the largest, and the nearest to zero when two are as large.  None unless two strokes
    agree.  ``long_short``: ``off`` is from the beat, and a group counts for less by each
    stroke it would leave early in the beat, between a fifth and under a half of the way
    through: swung eighths run long then short, so of her two groups the one on the
    beat is the one the other follows late."""
    best = None
    for c in off[np.abs(off) <= within]:
        group = off[np.abs(off - c) <= _FIND_TOGETHER]
        early = (int(np.sum(((off - c) % 1.0 > 0.2) & ((off - c) % 1.0 < 0.45)))
                 if long_short else 0)
        key = (early - group.size, abs(float(c)))
        if group.size >= 2 and (best is None or key < best[0]):
            best = (key, float(group.mean()))
    return None if best is None else best[1]


class RhythmEar:
    """Where her strokes fall in the beat and how far apart: how far her eighths swing,
    how many of her strokes are off the beat, and how even the gaps between them are.

    ``bar`` takes the times of the strokes found in one bar of her frames and the bar
    line they were played against.  All of it is counted from when she strikes, never
    from how loud, so none of it moves with the input gain."""

    def __init__(self):
        self._phase = np.zeros(PHASE_BINS, dtype=np.int64)
        self._gaps = np.zeros(GAP_BINS, dtype=np.int64)
        self._last = None            # when her last stroke was, s
        self._lead = 0.0             # how far ahead of the bar line her eighths run, beats
        self._found = False

    def carry_on(self, *, find: bool = False) -> "RhythmEar":
        """A new ear that has heard nothing yet but knows where she is: placed on her
        eighths as this one has her, and with her last stroke for the next gap.  A new
        ear placing her from the bar line alone would find a player who has drifted off
        it a long way into the take on the wrong side of the beat.  ``find``: its first
        bar finds her again from there, as the first bar of a take does from the bar
        line, so a player whose tempo is not quite the band's is followed past where
        following alone loses her (she moves more than _FOLLOW_WITHIN between two of
        the strokes it follows her by)."""
        ear = RhythmEar()
        ear._last, ear._lead = self._last, self._lead
        ear._found = self._found and not find
        return ear

    def bar(self, strokes_at, *, bar_start: float, bar_s: float,
            lag_s: float = STROKE_LAG_S) -> None:
        beat = float(bar_s) / 4.0
        for t in strokes_at:
            if self._last is not None:
                gap = int(round((float(t) - self._last) / beat * 12.0))
                self._gaps[min(max(gap, 0), GAP_BINS - 1)] += 1
            self._last = float(t)
        at = np.array([(float(t) - lag_s - float(bar_start)) / beat
                       for t in strokes_at]) - self._lead
        if not at.size:
            return
        from_beat = (at + 0.5) % 1.0 - 0.5
        from_eighth = ((2.0 * at + 0.5) % 1.0 - 0.5) / 2.0
        moved = None
        if self._found:
            near = np.abs(from_eighth) <= _FOLLOW_WITHIN
            if near.any():
                moved = float(from_eighth[near].mean())
        else:
            for off in (from_beat, from_eighth):
                near = np.abs(off) <= _FIND_WITHIN
                if near.any():
                    moved = float(off[near].mean())
                    break
            else:
                moved = _together(from_beat, _FIND_WIDE, long_short=True)
                if moved is None:
                    moved = _together(from_eighth, _FIND_WIDE)
            self._found = moved is not None
        if moved is not None:
            self._lead += moved
            at = at - moved
        np.add.at(self._phase, ((at % 1.0) * PHASE_BINS).astype(int) % PHASE_BINS, 1)

    @property
    def offbeat(self) -> float | None:
        n = int(self._phase.sum())
        if not n:
            return None
        edge = int(round(_ON_BEAT * PHASE_BINS))
        on = int(self._phase[:edge].sum() + self._phase[PHASE_BINS - edge:].sum())
        return 1.0 - on / n

    @property
    def swing(self) -> float | None:
        lo, hi = _AND_BINS
        ands = self._phase[lo:hi]
        n = int(ands.sum())
        if n < _SWING_MIN_STROKES:
            return None
        # The median, taken inside the bin that holds it.
        upto = np.cumsum(ands)
        k = int(np.searchsorted(upto, n / 2.0))
        before = float(upto[k - 1]) if k else 0.0
        swing = (lo + k + (n / 2.0 - before) / float(ands[k])) / PHASE_BINS
        # Measured from where her strokes on the beat fall, not from the bar line: a
        # swung player's eighths pull the reading of her beat late.
        edge = int(round(_ON_BEAT * PHASE_BINS))
        near = np.concatenate([self._phase[PHASE_BINS - edge:], self._phase[:edge]])
        if near.sum():
            swing -= (float(near @ (np.arange(-edge, edge) + 0.5)) / float(near.sum())
                      / PHASE_BINS)
        if swing >= _HARD_SWING:
            late = int(self._phase[int(_HARD_SWING * PHASE_BINS):hi].sum())
            if 2 * int(self._phase[slice(*_TRIPLET_BINS)].sum()) >= late:
                return None
            if int(self._phase[slice(*_LATE_BINS)].sum()) >= late:
                return None
        return swing

    @property
    def steady(self) -> float | None:
        """The share of the gaps between her strokes that are the commonest length, to
        within a twelfth of a beat: 1 for an even pulse, a half for swung eighths or for
        long and short notes in turn."""
        n = int(self._gaps.sum())
        if n < _STEADY_MIN_GAPS:
            return None
        k = int(np.argmax(self._gaps))
        return float(self._gaps[max(k - 1, 0):k + 2].sum()) / n

    def memory_items(self) -> int:
        return int(self._phase.size + self._gaps.size)


_ANY = float("inf")
#: How far outside a style's band a reading is before it costs that style a whole point.
#: Each is from twice (the strum) to five times what nine readings in ten move between
#: an input and the same one 20 dB down.
SOFT = {"strokes": 0.4, "swing": 0.05, "offbeat": 0.25, "steady": 0.25, "strum": 0.22}
#: Each style's band (low, high) for the things heard that tell it, in the order a
#: near-tie goes: the style that asks most of her playing first.  In words: jazz swings
#: hard; house is straight and off the beat; synthwave is a busy, straight, even pulse;
#: trap is sparse and straight; hip hop and r&b leave room, more off the beat than on
#: it, in uneven gaps, straight or lightly swung; lofi is sparse, mostly on the beat, in
#: uneven gaps, straight or lightly swung; folk is strummed, straight and on the beat;
#: midwest is busy and straight, on and off the beat, and no more even than a picked
#: pattern is, so an even pulse costs it a little and synthwave takes one where both
#: can.  The edges sit between the values plain figures give (quarters one stroke a
#: beat, eighths two; none, half or all of them off the beat; all one gap or long and
#: short in turn), not on them, so a reading that moves a little does not cross one.
PROFILES = {
    "jazz":      {"swing": (0.62, 1.0)},
    "house":     {"swing": (0.0, 0.55), "offbeat": (0.7, 1.0)},
    "synthwave": {"strokes": (1.3, _ANY), "swing": (0.0, 0.55), "steady": (0.7, 1.0)},
    "trap":      {"strokes": (0.0, 0.7), "swing": (0.0, 0.55)},
    "hiphop":    {"strokes": (0.0, 1.7), "swing": (0.0, 0.63), "offbeat": (0.62, 1.0),
                  "steady": (0.0, 0.7)},
    "rnb":       {"strokes": (0.0, 1.7), "swing": (0.0, 0.63), "offbeat": (0.62, 1.0),
                  "steady": (0.0, 0.7)},
    "lofi":      {"strokes": (0.0, 1.3), "swing": (0.0, 0.63), "offbeat": (0.0, 0.62),
                  "steady": (0.0, 0.7)},
    "folk":      {"swing": (0.0, 0.55), "offbeat": (0.0, 0.6), "strum": (0.29, 1.0)},
    "midwest":   {"strokes": (1.3, _ANY), "swing": (0.0, 0.55), "offbeat": (0.3, 0.8),
                  "steady": (0.0, 0.8)},
}
#: The strum band of every style but folk: FolkEar's marks, no evidence of a strum
#: under 0.18 of her strokes and full evidence from 0.40.
NOT_STRUMMED = (0.0, 0.18)
#: What makes each style that style, which must be heard inside its band before the
#: style is a candidate at all.
NEEDS = {"jazz": ("swing",), "house": ("offbeat",), "synthwave": ("strokes", "steady"),
         "hiphop": ("offbeat",), "rnb": ("offbeat",), "lofi": ("steady",),
         "folk": ("strum",)}
#: Scores this close are a near-tie: what was heard does not tell the styles apart, so
#: the tempo's default takes it, or the style that comes first in PROFILES.
MARGIN = 0.15
#: A style that loses this much does not fit what was heard.  When none fits, her
#: playing is not one of these styles' and the tempo's default answers.
NO_FIT = 1.0

#: Strokes a beat from which a player is too busy for trap, the default from 130 to 150.
BUSY_DEFAULT = 0.85

_RANGES: dict = {}


def tempo_range(style: str) -> tuple[float, float]:
    """The style's own tempo range, from its file."""
    if style not in _RANGES:
        from .styles_lib import load_style  # noqa: PLC0415
        lo, hi = load_style(style)["bpm_range"]
        _RANGES[style] = (float(lo), float(hi))
    return _RANGES[style]


def tempo_default(bpm: float, mode: str | None = None, strokes: float | None = None) -> str:
    """The style for a tempo when nothing has been heard of how she plays: at the count,
    on a room mic.  Each band is one style's own range, and a minor key, when it is
    known, turns the band to the style that leaves room.  It is also where a tie goes,
    and where playing that fits no style goes: from 130 to 150, trap's range, a player
    known to be too busy for trap (``strokes`` a beat, :data:`BUSY_DEFAULT`) gets
    midwest, the one other style there."""
    minor = mode == "minor"
    if bpm < 80.0:
        return "rnb" if minor and bpm >= 60.0 else "lofi"
    if bpm < 90.0:
        return "hiphop" if minor else "lofi"
    if bpm < 100.0:
        return "hiphop" if minor else "midwest"
    if bpm < 118.0:
        return "synthwave" if minor else "midwest"
    if bpm < 130.0:
        return "house" if minor else "midwest"
    if bpm <= 150.0:
        return "midwest" if strokes is not None and strokes >= BUSY_DEFAULT else "trap"
    return "midwest"


@dataclass(frozen=True)
class StyleScore:
    """One candidate: 0 fits all that was heard, and each misfit takes away up to 1."""
    style: str
    score: float
    #: What it lost, and on what: (thing heard, cost), the dearest first.
    against: tuple = ()


def _cost(value: float, band: tuple[float, float], soft: float) -> float:
    lo, hi = band
    out = lo - value if value < lo else value - hi if value > hi else 0.0
    return min(1.0, out / soft)


def _against(style: str, heard: Playing) -> list:
    """What ``style`` loses on ``heard``: (thing heard, cost) for each reading outside
    its band."""
    profile = PROFILES[style]
    against = []
    for f in fields(Playing):
        value = getattr(heard, f.name)
        band = profile.get(f.name, NOT_STRUMMED if f.name == "strum" else None)
        if value is not None and band is not None:
            cost = _cost(float(value), band, SOFT[f.name])
            if cost > 0.0:
                against.append((f.name, cost))
    return against


def fit(style: str, playing: Playing | None) -> float:
    """How well ``style`` fits what was heard, as :func:`score_styles` scores it, but
    whatever the tempo and whether or not what makes it that style was heard: 0 fits
    all of it, each misfit takes away up to 1.  The score a style already playing is
    held to when another is weighed against it."""
    return -sum(c for _, c in _against(style, playing or Playing()))


def score_styles(bpm: float, mode: str | None = None,
                 playing: Playing | None = None) -> list[StyleScore]:
    """Every candidate style for this tempo with its score, the answer first and then
    the best of the rest: see the module's docstring for the rule."""
    default = tempo_default(bpm, mode, None if playing is None else playing.strokes)
    heard = playing or Playing()
    out = []
    for style in PROFILES:
        lo, hi = tempo_range(style)
        if style != default and not lo <= bpm <= hi:
            continue
        against = _against(style, heard)
        lost = dict(against)
        if style != default and playing is not None and any(
                getattr(heard, need) is None or lost.get(need, 0.0) > 0.0
                for need in NEEDS.get(style, ())):
            continue
        against.sort(key=lambda a: -a[1])
        out.append(StyleScore(style, -sum(c for _, c in against), tuple(against)))
    # Stable, so equal scores keep the order of PROFILES.
    out.sort(key=lambda s: -s.score)
    close = [s for s in out if s.score >= out[0].score - MARGIN]
    if out[0].score <= -NO_FIT:
        close = [s for s in out if s.style == default]
    first = next((s for s in close if s.style == default), None) or min(
        close, key=lambda s: list(PROFILES).index(s.style))
    out.remove(first)
    out.insert(0, first)
    return out


def explain(scores: list[StyleScore]) -> str:
    """The answer and the best of the rest with their scores and what each lost on, and
    whether the two were a near-tie."""
    def one(s: StyleScore) -> str:
        lost = ", ".join(f"{name} {cost:.2f}" for name, cost in s.against)
        return f"{s.style} {s.score:.2f}" + (f" (against it: {lost})" if lost else "")
    why = " over ".join(one(s) for s in scores[:2])
    if max(s.score for s in scores) <= -NO_FIT:
        return why + ", no style fits: the tempo's default"
    tie = len(scores) > 1 and scores[0].score - scores[1].score <= MARGIN
    return why + (", a near-tie" if tie else "")
