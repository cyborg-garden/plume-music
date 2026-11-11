"""Her chords: the band plays the chord she means, when it can tell (roadmap step 8).

WHY.  Once the key settles the governor cycles the style's progression one chord
per bar, whatever she plays.  On her takes her bar's pitch content fit the band's
chord about as well as a random chord would on the bars where they disagreed.  A
band that hears you sits on your chord when you sit on it, and plays your
progression when you loop one.

TIMING.  The live loop plans bar N a bar ahead and reads the last bar that has
finished, N-2.  Following a changing progression two bars late would put every
change in the wrong place twice, so a change is followed only once it repeats:
her loop of 2, 4 or 8 bars, seen twice running, predicts bar N from bar N minus the
loop.  A held chord needs no prediction, lag or not.  (ReaLJam, CHI EA 2025: the
best moments were when the agent predicted the chord the player had in mind.)

TRUTH.  A wrong chord is worse than the template, so this commits only when a bar
fits one chord clearly: its triad holds at least :data:`FIT_MIN` of the bar's
pitch energy and beats the next chord by :data:`MARGIN`.  The chords it can name
are the key's seven diatonic ones, and beside them any chord the style borrows
for a degree (the minor key's major V, jazz's iv7 and VI7): never the borrowed
one in place of the key's own.  Anything else, single-note melody included, keeps
the style's own progression.  Each bar's decision says which it was and why
(``decide``'s ``reason``).

ROOM MIC.  A room mic hears the band too, and the band's own chord must not come
back as hers.  Nothing live can say how much of a bar is the band: the self
masker's coupling estimate is a bound for masking, not a measurement.  On the fake
room (speakers 6 dB down) it put the band at under half the mic while she was
silent and the band was all it heard, and at a fifth of it with no bleed at all.
Taking that share of the band's chord out leaves the band's chord standing, and
it read as hers: a held Em from the band's own Cmaj7.  So a follower made with
``band_in_mic`` (any input but a direct one or MIDI) keeps the template and says
why.

Pure: plain numbers in, a plain dict out, no clock, no device, no rng.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .styles_lib import load_style
from .theory import parse_chord, parse_key, roman_to_chord

#: Share of a bar's pitch energy her chord's triad must hold.  Three notes of
#: twelve hold 0.25 by chance, and a melody walking a seven-note scale gives every
#: diatonic triad about 0.43.  Tuned on the synthetic fixtures and her takes.
FIT_MIN = 0.6
#: How far the best chord must beat the next one (score units: fit plus the
#: root's weight).  Neighbouring diatonic triads share two notes, so a bar that
#: sits between them is not a chord we can name.
MARGIN = 0.08
#: Weight of the root's own energy in the score: what tells C from Am from F when
#: all three triads hold a bar of C and E.
ROOT_WEIGHT = 0.25
#: Weight of her lowest notes on the root (the extractor's bass chroma, E1 to C4):
#: a guitar's bass string names the chord when its upper notes sit between two.
#: Bass-aware chord reading is the field's standard step up from plain templates
#: (Mauch and Dixon, ISMIR 2010).
BASS_WEIGHT = 0.25
#: Notes sounding at once (the bar's median, per frame, of exp(entropy) of the
#: chroma) below which a bar is single-note melody, not a chord.  Measured: plucked
#: single notes 1.3-1.6 whether one or four to a bar, plucked triads 3.5-3.7.  A
#: bright or ringing single note reaches 2-3, which is why :data:`BALANCE_MIN` exists.
NOTES_MIN = 2.0
#: Every note of the chord must sound: its weakest triad tone holds at least this
#: share of its strongest.  One bright note carries its own fifth and major third in
#: its overtones (harmonics 3 and 5), so a low C reads as C major, and a lone E as E
#: minor at fit 0.9 with no G in it at all.  Measured on rendered tones: single notes
#: 0.0-0.20 (dull or bright, short or ringing; a very bright low one reaches 0.4),
#: chords of four or five strings 0.30-0.66.  Six strings double the root and fifth
#: and leave one third: 0.16-0.26 on a plain tone.  With a major third that is where
#: a single note sits too, so those voicings (open E, a full G) keep the template on
#: a plain tone.  Not tuned on any take.
BALANCE_MIN = 0.25
#: The same floor when the weakest tone is a minor third.  That is the one chord
#: tone the others' overtones cannot fake (the first harmonic near it is the 19th),
#: so above a trace it is a string of its own.  Measured on rendered tones: single
#: notes and power chords 0.0-0.04, a full Em, F#m or Bm barre 0.20-0.31.  A note
#: outside the chord can still put it there (the third harmonic of the note a fifth
#: below it, the fifth harmonic of the one a major third below), so this floor holds
#: only while neither of those sounds half as loud as the third itself.
MINOR_THIRD_MIN = 0.15
#: Her loop lengths in bars, longest first: an established longer loop knows more
#: than the hold inside it.
LOOP_PERIODS = (8, 4, 2, 1)
#: Bars of her playing kept.
MEMORY_BARS = 32
#: After this many bars with nothing heard from her, the band goes back to its own.
STALE_BARS = 8

#: Same root, same chord underneath, one colour tone more: the gentle motion on a
#: held chord, every other two bars.
_COLOUR = {"maj7": "maj9", "m7": "m9", "7": "9", "": "add9", "m": "madd9"}

_FALLBACK = {
    ("major", True): ("Imaj7", "ii7", "iii7", "IVmaj7", "V7", "vi7", "vii7b5"),
    ("minor", True): ("i7", "ii7b5", "IIImaj7", "iv7", "v7", "VImaj7", "VII7"),
    ("major", False): ("I", "ii", "iii", "IV", "V", "vi", "viidim"),
    ("minor", False): ("i", "iidim", "III", "iv", "v", "VI", "VII"),
}
#: A numeral with a seventh or more on it (sus2 is not one).
_SEVENTH = re.compile(r"(?<!sus)(7|9|11|13)")
_DEGREES = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6}


@dataclass(frozen=True)
class Candidate:
    numeral: str
    sym: str
    root: int
    triad: tuple[int, ...]


@dataclass(frozen=True)
class Reading:
    """One bar of hers: the chord it names (None when unsure), its fit, and the
    best guess even when unsure, for the log."""
    sym: str | None
    fit: float
    best: str | None
    margin: float
    why: str = ""            # when unsure: what made it so


@dataclass(frozen=True)
class HerBar:
    """One bar of hers as heard: chroma (None when nothing had pitch), her bass
    chroma, and how many notes sound at once."""
    chroma: np.ndarray | None
    bass: np.ndarray | None = None
    notes: float | None = None


def _degree(numeral: str) -> int | None:
    if not numeral or numeral[0] == "b":
        return None                  # a borrowed degree is not one of the seven
    i = 0
    while i < len(numeral) and numeral[i] in "iIvV":
        i += 1
    return _DEGREES.get(numeral[:i].lower())


def _triad(chord) -> tuple[int, ...]:
    """Root, third and fifth (the sus tone for a sus chord): what a guitar plays."""
    iv = chord.intervals
    third = next((i for i in iv if i in (3, 4)), next((i for i in iv if i in (2, 5)), None))
    fifth = next((i for i in iv if i in (7, 6, 8)), None)
    return tuple(sorted({chord.root % 12, *((chord.root + i) % 12
                                            for i in (third, fifth) if i is not None)}))


_CAND_CACHE: dict[tuple[str, str], tuple[Candidate, ...]] = {}


def _spell(numeral: str, key) -> Candidate:
    ch = roman_to_chord(numeral, key)
    return Candidate(numeral, ch.sym, ch.root % 12, _triad(ch))


def candidates(key_str: str, style: str) -> tuple[Candidate, ...]:
    """The seven diatonic chords of ``key_str``, spelled as ``style`` spells them,
    then the chords the style borrows.

    A degree the style's own progressions use takes its most-weighted numeral that
    stays in the key, so the band's colour stays (lofi's ii7, not a bare ii); a
    degree it never uses, or only with a borrowed chord, takes the plain diatonic
    seventh, or the triad in a style of triads.  Where the style's most-weighted
    numeral on a degree leaves the key (the minor key's major V, jazz's iv7 and
    VI7), that chord follows the seven: she may play either, and her minor v must
    not be read as the band's V7."""
    hit = _CAND_CACHE.get((key_str, style))
    if hit is not None:
        return hit
    key = parse_key(key_str)
    mode = "major" if key.mode == "major" else "minor"
    try:
        pools = load_style(style).get("progressions")
    except (OSError, KeyError, ValueError):
        pools = None
    if isinstance(pools, dict):
        progs = pools.get(mode) or []
    else:
        progs = pools or []
    votes: dict[int, dict[str, float]] = {}
    sevenths = False
    for p in progs:
        for n in p.get("numerals", ()):
            n = str(n)
            sevenths = sevenths or _SEVENTH.search(n) is not None
            d = _degree(n)
            if d is not None:
                votes.setdefault(d, {})
                votes[d][n] = votes[d].get(n, 0.0) + float(p.get("weight", 1))
    fallback = _FALLBACK[(mode, sevenths or not progs)]
    scale = set(key.scale())
    seven, borrowed = [], []
    for d in range(7):
        own = []                         # the style's chords here, most-weighted first
        for numeral, _ in sorted(votes.get(d, {}).items(),
                                 key=lambda kv: (kv[1], kv[0]), reverse=True):
            try:
                own.append(_spell(numeral, key))
            except ValueError:
                continue
        inside = next((k for k in own if scale.issuperset(k.triad)), None)
        seven.append(inside if inside is not None else _spell(fallback[d], key))
        if own and own[0].sym != seven[-1].sym:
            borrowed.append(own[0])
    res = tuple(seven + borrowed)
    _CAND_CACHE[(key_str, style)] = res
    return res


def read_bar(chroma, cands, bass=None, notes=None) -> Reading | None:
    """Name the chord of one bar's pitch energy, or say it can't.  ``bass``: the
    bar's bass chroma, ``notes``: notes at once, if known.  None: nothing in the
    bar to read."""
    c = np.asarray(chroma, dtype=float)
    total = float(c.sum())
    if not np.isfinite(total) or total <= 1e-9:
        return None
    c = c / total
    lo = np.zeros(12) if bass is None else np.asarray(bass, dtype=float)
    lo = lo / float(lo.sum()) if float(lo.sum()) > 1e-9 else np.zeros(12)
    scored = sorted(((float(c[list(k.triad)].sum()) + ROOT_WEIGHT * float(c[k.root])
                      + BASS_WEIGHT * float(lo[k.root]),
                      float(c[list(k.triad)].sum()), k) for k in cands),
                    key=lambda s: s[0], reverse=True)
    (s1, fit, best), s2 = scored[0], scored[1][0] if len(scored) > 1 else 0.0
    margin = s1 - s2
    tri = sorted((float(c[p]), p) for p in best.triad)
    (weakest, weak_pc), loudest = tri[0], tri[-1][0]
    balance = weakest / loudest if loudest > 0.0 else 0.0
    own_string = ((weak_pc - best.root) % 12 == 3 and
                  2.0 * max(float(c[(weak_pc - 7) % 12]), float(c[(weak_pc - 4) % 12]))
                  < weakest)
    floor = MINOR_THIRD_MIN if own_string else BALANCE_MIN
    if notes is not None and notes < NOTES_MIN:
        why = f"single notes ({notes:.1f} at once)"
    elif balance < floor:
        why = f"one note of {best.sym} barely sounds ({balance:.2f} of the loudest)"
    elif fit < FIT_MIN:
        why = f"fit {fit:.2f} under {FIT_MIN:.2f}"
    elif margin < MARGIN:
        why = f"between {best.sym} and {scored[1][2].sym}"
    else:
        why = ""
    return Reading(None if why else best.sym, round(fit, 3), best.sym, round(margin, 3),
                   why)


def clean_bar(frames) -> HerBar:
    """Her bar from its frames: ``(chroma[, bass chroma])`` each, as heard on an input
    the band cannot reach.  Chroma is None when nothing in the bar has pitch; bass
    is None when no frame carries one."""
    # One array pass, not a loop per frame: this runs on the planner every bar.
    kept = [f for f in frames if float(np.sum(f[0])) > 0.0]
    if not kept:
        return HerBar(None)
    c = np.asarray([f[0] for f in kept], dtype=float)
    c /= c.sum(axis=1, keepdims=True)
    acc = c.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_frame = np.exp(-np.sum(np.where(c > 0.0, c * np.log(c), 0.0), axis=1))
    lows = [f[1] for f in kept if len(f) > 1 and f[1] is not None]
    bass = None
    if lows:
        b = np.asarray(lows, dtype=float)
        low = b[b.sum(axis=1) > 0.0].sum(axis=0) if b.size else np.zeros(12)
        bass = low / float(low.sum()) if float(np.sum(low)) > 1e-9 else None
    return HerBar(acc / float(acc.sum()), bass, float(np.median(per_frame)))


def colour(sym: str, bar: int) -> str:
    """A held chord, with a colour tone on every other two bars."""
    if (int(bar) // 2) % 2 == 0:
        return sym
    try:
        ch = parse_chord(sym)
    except ValueError:
        return sym
    alt = _COLOUR.get(ch.quality)
    return sym if alt is None else sym[: len(sym) - len(ch.quality)] + alt


_REST, _DEAF = "rest", "deaf"


class ChordFollower:
    """Her bars in; the chord the band should play out, with the reason.

    ``hear`` each bar she was heard over (or wasn't), ``decide`` each bar the band
    plans.  Her bars are kept as chroma, so a loop she played before the key was
    known is read in that key the moment it is."""

    def __init__(self, *, band_in_mic: bool = False):
        self._bars: dict[int, object] = {}
        #: The band can reach this input (a room mic): its chord can't be told from hers.
        self.band_in_mic = bool(band_in_mic)

    def hear(self, bar: int, her, *, playing: bool) -> None:
        """``her``: her bar, a :class:`HerBar` from :func:`clean_bar` (or a bare
        chroma).  No chroma while ``playing``: she was heard but nothing had pitch."""
        b = int(bar)
        if not isinstance(her, HerBar):
            her = HerBar(None if her is None else np.asarray(her, dtype=float))
        if not playing:
            self._bars[b] = _REST
        elif her.chroma is None:
            self._bars[b] = _DEAF
        else:
            self._bars[b] = her
        for old in [k for k in self._bars if k <= b - MEMORY_BARS]:
            del self._bars[old]

    def readings(self, key: str, style: str) -> dict[int, Reading | str]:
        cands = candidates(key, style)
        out: dict[int, Reading | str] = {}
        for b, v in sorted(self._bars.items()):
            if isinstance(v, str):
                out[b] = v
            else:
                r = read_bar(v.chroma, cands, v.bass, v.notes)
                out[b] = _REST if r is None else r
        return out

    def decide(self, bar: int, key: str | None, style: str,
               template: str | None) -> dict:
        """The band's chord for ``bar``: hers when a hold or loop is sure, else
        ``template``."""
        def out(chord, source, reason, period=None, last=None):
            r = last if isinstance(last, Reading) else None
            return {"chord": chord, "source": source, "reason": reason,
                    "period": period, "template": template,
                    "heard_bar": heard_bar,
                    "her_chord": None if r is None else r.sym,
                    "her_best": None if r is None else r.best,
                    "her_fit": None if r is None else r.fit}

        heard_bar = max(self._bars) if self._bars else None
        if self.band_in_mic:
            return out(template, "template",
                       "template: on a room mic the band can't tell her chord from its own")
        if key is None or template is None or heard_bar is None:
            return out(template, "template", "template: nothing of hers heard yet")
        try:
            reads = self.readings(key, style)
        except ValueError:
            return out(template, "template", f"template: can't read chords in {key}")
        last = reads.get(heard_bar)
        evidence = [b for b, r in reads.items() if isinstance(r, Reading)]
        if not evidence:
            why = ("nothing she played had a pitch to read" if last == _DEAF
                   else "she has not played a bar yet")
            return out(template, "template", f"template: {why}", last=last)
        if heard_bar - evidence[-1] >= STALE_BARS:
            return out(template, "template",
                       f"template: nothing of hers since bar {evidence[-1]}", last=last)
        recent = evidence[-2:]
        if len(recent) == 2 and all(reads[b].sym is None for b in recent):
            r = reads[recent[-1]]
            return out(template, "template",
                       f"template: bars {recent[0]} and {recent[1]} name no chord "
                       f"({r.why}, best {r.best})", last=last)
        for period in LOOP_PERIODS:
            pattern = self._loop(reads, heard_bar, period)
            if pattern is None:
                continue
            k = (bar - heard_bar + period - 1) // period
            src_bar = bar - k * period
            sym = pattern[src_bar % period]
            if period == 1:
                seen = pattern["seen"]
                return out(colour(sym, bar), "her_hold",
                           f"following her {sym}: bars {seen[0]} and {seen[1]} "
                           f"both read {sym}", period, last)
            seq = " ".join(pattern[(heard_bar - period + 1 + i) % period]
                           for i in range(period))
            return out(sym, "her_loop",
                       f"following her {period}-bar loop {seq}: bar {bar} plays "
                       f"bar {src_bar}'s {sym}", period, last)
        r = reads[evidence[-1]]
        why = (f"her bar {evidence[-1]} reads {r.sym}, no hold or loop yet" if r.sym
               else f"her bar {evidence[-1]} names no chord ({r.why}, best {r.best})")
        return out(template, "template", f"template: {why}", last=last)

    @staticmethod
    def _loop(reads, heard_bar: int, period: int):
        """Her chords by bar mod ``period`` when every place in the loop read the
        same chord its last two times, within three loops back; else None.  Rests are
        skipped, and so is her latest bar if it names no chord: one fill bar.  A
        pattern that is really a shorter loop is left to that one."""
        lo = heard_bar - 3 * period + 1
        seen: dict[int, list[tuple[int, str | None]]] = {}
        newest = True
        for b in range(heard_bar, lo - 1, -1):
            r = reads.get(b)
            if not isinstance(r, Reading):
                continue                # a rest says nothing
            if newest and r.sym is None:
                newest = False
                continue                # one fill bar, the latest, is not a change
            newest = False
            slot = seen.setdefault(b % period, [])
            if len(slot) < 2:
                slot.append((b, r.sym))
        if len(seen) < period:
            return None
        pattern: dict = {}
        for res, slot in seen.items():
            if len(slot) < 2 or slot[0][1] is None or slot[0][1] != slot[1][1]:
                return None
            pattern[res] = slot[0][1]
        if period > 1:
            for sub in (p for p in LOOP_PERIODS if p < period and period % p == 0):
                if all(pattern[r] == pattern[r % sub] for r in range(period)):
                    return None
        else:
            pattern["seen"] = (seen[0][1][0], seen[0][0][0])
        return pattern
