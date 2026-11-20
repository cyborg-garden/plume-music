"""A style from how she plays: the evidence, and the rule that scores each style on it.

Tempo says little about a style: seven of the nine bundled styles can be played at 85
bpm.  What tells them apart is in the playing.  Five things are heard of it, each a
number that does not move with the input gain:

    strokes   strokes per beat (bandentry.StrokeEar)
    swing     where her off-beat eighths fall in the beat: 0.5 straight, 0.67 a triplet
    offbeat   the share of her strokes that fall off the beat
    sounding  the share of the time she sounds, not resting or left to ring out
    strum     the share of her strokes that are strums (bandentry.FolkEar)

THE RULE (:func:`score_styles`).  Every style whose tempo range holds the tempo is a
candidate, and so is the tempo's own default (:func:`tempo_default`), which is the
whole answer while nothing has been heard of her: at the count, and on a room mic.
Each style has a band for each thing heard (:data:`PROFILES`).  A candidate starts at
0 and loses up to 1 for each thing heard that falls outside its band: nothing inside
it, 1 once it is a full :data:`SOFT` step out, in proportion between.  What a style has
no band for costs it nothing, and what was not heard costs nobody anything, with two
exceptions (:data:`NEEDS`): jazz is its swing and folk is its strum, so each loses 1
while its own has not been heard.  Every other style loses as she strums more.  The
best score wins; a tie goes to the tempo's default, then to the order of
:data:`PROFILES`.

The bands were set on 270 generated solo parts (nine styles by five parts by three
tempi by two keys) read over the seven bars heard before the band comes in, and checked
on 40 recorded acoustic guitar excerpts whose labels map onto three of the styles.  The
recordings changed one thing: a real player's swing is less exact than a generated
one's, so the swing bands are wider than the generated parts alone would set them.
With the true tempo the rule names the generated part's own style for 178 of 270 (the
rule before it, on tempo, mode and strokes: 106) and the recorded excerpt's for 10 of
24 (before: 6).  Moving any one band edge by 15% leaves 166 to 182 of 270.  Chord
colour (sevenths against triads) would separate more and is not here: the live chroma
does not carry it.

:class:`RhythmEar` measures swing and offbeat.  Pure, cheap and bounded: it is fed one
bar of stroke times at a time on the planning thread, and keeps one histogram however
long the take.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .listen.features import WIN

#: A frame's time is the end of its window, and a stroke that peaks in it sits in the
#: middle of that window on average: measured on generated parts with every note on a
#: known grid, the median stroke read 18 to 21 ms after its note.
STROKE_LAG_S = WIN / 2.0 / 48000.0
#: Strokes are counted by where in the beat they fall, in this many equal bins.  A hop is
#: a 36th of a beat at 160 bpm, so finer bins than this would hold nothing more.
PHASE_BINS = 48
#: On the beat: within an eighth of a beat of it, either side.
_ON_BEAT = 0.125
#: The off-beat eighth falls in these bins: at 0.5 of the beat played straight, at 0.67
#: as a triplet swing.  The window stops short of the last sixteenth (0.75) and starts
#: past the second (0.25) and the middle note of a triplet (0.33), so a player of
#: sixteenths or of triplets is read by her eighths alone.
_AND_BINS = (20, 35)
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
#: Fewer off-beat eighths than this say nothing about how she swings them.
_SWING_MIN_STROKES = 4


@dataclass(frozen=True)
class Playing:
    """What was heard of how she plays.  None: not heard, or not hers alone to hear."""
    strokes: float | None = None
    swing: float | None = None
    offbeat: float | None = None
    sounding: float | None = None
    strum: float | None = None


class RhythmEar:
    """Where her strokes fall in the beat: how far her eighths swing, and how many of
    her strokes are off the beat.

    ``bar`` takes the times of the strokes found in one bar of her frames and the bar
    line they were played against."""

    def __init__(self):
        self._phase = np.zeros(PHASE_BINS, dtype=np.int64)
        self._lead = 0.0             # how far ahead of the bar line her eighths run, beats
        self._found = False

    def bar(self, strokes_at, *, bar_start: float, bar_s: float) -> None:
        beat = float(bar_s) / 4.0
        at = np.array([(float(t) - STROKE_LAG_S - float(bar_start)) / beat
                       for t in strokes_at]) - self._lead
        if not at.size:
            return
        off = ((2.0 * at + 0.5) % 1.0 - 0.5) / 2.0      # from the nearest eighth
        near = np.abs(off) <= _FOLLOW_WITHIN
        if not self._found:
            off_beat = (at + 0.5) % 1.0 - 0.5           # from the nearest beat
            near = np.abs(off_beat) <= _FIND_WITHIN
            if near.any():
                off = off_beat
            else:
                near = np.abs(off) <= _FIND_WITHIN
        if near.any():
            self._found = True
            moved = float(off[near].mean())
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
        return swing

    def memory_items(self) -> int:
        return int(self._phase.size)


_ANY = float("inf")
#: How far outside a style's band a reading is before it costs that style a whole point.
SOFT = {"strokes": 0.5, "swing": 0.06, "offbeat": 0.25, "sounding": 0.2, "strum": 0.22}
#: Each style's band (low, high) for each thing heard of her playing.  In words:
#: midwest is busy, straight and rings on; lofi is sparse with a light swing; folk is
#: strummed, straight, on the beat and rings on; synthwave is the busiest, straight and
#: unbroken; hip hop swings lightly off the beat with gaps; r&b swings lightly and is
#: mostly off the beat; house is straight, off the beat and clipped; trap is sparse,
#: straight and on the beat; jazz swings hard.  The strum bands are FolkEar's marks: no
#: evidence of a strum under 0.18 of her strokes, full evidence from 0.40.  Folk loses
#: what is missing of it, and every other style, through NOT_STRUMMED, what there is.
NOT_STRUMMED = (0.0, 0.18)
PROFILES = {
    "midwest":   {"strokes": (1.25, _ANY), "swing": (0.0, 0.55), "offbeat": (0.3, 0.8),
                  "sounding": (0.8, 1.0)},
    "lofi":      {"strokes": (0.0, 1.2), "swing": (0.55, 0.60), "offbeat": (0.0, 0.65),
                  "sounding": (0.0, 0.9)},
    "folk":      {"strokes": (0.4, 2.0), "swing": (0.0, 0.55), "offbeat": (0.0, 0.55),
                  "sounding": (0.85, 1.0), "strum": (0.40, 1.0)},
    "synthwave": {"strokes": (1.75, _ANY), "swing": (0.0, 0.55), "sounding": (0.9, 1.0)},
    "hiphop":    {"strokes": (0.3, 1.5), "swing": (0.56, 0.62), "offbeat": (0.5, 1.0),
                  "sounding": (0.0, 0.8)},
    "rnb":       {"strokes": (0.3, 1.25), "swing": (0.56, 0.62), "offbeat": (0.6, 1.0),
                  "sounding": (0.0, 0.9)},
    "house":     {"strokes": (1.0, 2.0), "swing": (0.0, 0.53), "offbeat": (0.8, 1.0),
                  "sounding": (0.0, 0.7)},
    "trap":      {"strokes": (0.0, 0.7), "swing": (0.0, 0.55), "offbeat": (0.0, 0.55),
                  "sounding": (0.0, 0.9)},
    "jazz":      {"strokes": (0.4, 1.8), "swing": (0.60, 1.0)},
}
#: What a style loses while a thing it cannot do without has not been heard.
NEEDS = {("jazz", "swing"): 1.0, ("folk", "strum"): 1.0}

_RANGES: dict = {}


def tempo_range(style: str) -> tuple[float, float]:
    """The style's own tempo range, from its file."""
    if style not in _RANGES:
        from .styles_lib import load_style  # noqa: PLC0415
        lo, hi = load_style(style)["bpm_range"]
        _RANGES[style] = (float(lo), float(hi))
    return _RANGES[style]


def tempo_default(bpm: float, mode: str | None = None) -> str:
    """The style for a tempo when nothing has been heard of how she plays: at the count,
    on a room mic.  Each band is one style's own range, and a minor key, when it is
    known, turns the band to the style that leaves room.  It is also where a tie goes."""
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
        return "trap"
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


def score_styles(bpm: float, mode: str | None = None,
                 playing: Playing | None = None) -> list[StyleScore]:
    """Every candidate style for this tempo with its score, the best first: see the
    module's docstring for the rule."""
    default = tempo_default(bpm, mode)
    heard = playing or Playing()
    out = []
    for style, profile in PROFILES.items():
        lo, hi = tempo_range(style)
        if style != default and not lo <= bpm <= hi:
            continue
        against = []
        for f in fields(Playing):
            value = getattr(heard, f.name)
            if value is None:
                cost = NEEDS.get((style, f.name), 0.0) if playing is not None else 0.0
            else:
                band = profile.get(f.name, NOT_STRUMMED if f.name == "strum" else None)
                cost = _cost(float(value), band, SOFT[f.name]) if band else 0.0
            if cost > 0.0:
                against.append((f.name, cost))
        against.sort(key=lambda a: -a[1])
        out.append(StyleScore(style, -sum(c for _, c in against), tuple(against)))
    # Stable, so equal scores keep the order of PROFILES behind the default.
    out.sort(key=lambda s: (-s.score, s.style != default))
    return out


def explain(scores: list[StyleScore]) -> str:
    """The best two candidates with their scores and what each lost on."""
    def one(s: StyleScore) -> str:
        lost = ", ".join(f"{name} {cost:.2f}" for name, cost in s.against)
        return f"{s.style} {s.score:.2f}" + (f" (against it: {lost})" if lost else "")
    return " over ".join(one(s) for s in scores[:2])
