"""When the band joins a tap-started take: drums first, then you, then them.

The order is Juniper's (2026-11-02).  She taps four, the drums start and hold
that tempo, and she plays over them for a while.  Only once plume has heard her
settle in one key do the bass and chords come in, in that key, with the melody
a few bars after them.

Pure: a function of the (bar, harmony, playing) stream, so it is testable
without a device.  Key commitment itself is :class:`HarmonyGovernor`'s job; what
this adds is that bars she sat out are no evidence of a key.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .governor import HarmonyGovernor
from .listen.harmony import HarmonyState
from .theory import parse_key

HARMONY_ROLES = ("bass", "pad", "keys", "shimmer")
MELODY_ROLES = ("lead", "arp")
#: The chords: they wait for the drums and bass to have stated the groove.
CHORD_ROLES = ("pad", "keys", "shimmer")
#: Bars the drums and bass play before the chords join them.
CHORDS_DELAY_BARS = 2
#: Bars after the band before the melody joins: the last part of the entry.
MELODY_DELAY_BARS = 4
#: Pitched drums, which play a note of the key: held back until there is one.
PITCHED_DRUM_ROLES = ("sub",)
#: Bars of rest after which her playing again starts a new phrase.  One bar is a
#: breath, or a room mic missing a bar, and restarting on it every other bar meant a
#: player who rests every other bar never got the band.
PHRASE_REST_BARS = 2
#: Bars on the tracker's key before the band will come in on it rather than on the
#: key it settled earlier.
ENTRY_KEY_BARS = 2
#: When the band is ready to come in but the tracker's key is less than this far ahead
#: of the next key (HarmonyState.key_margin, a difference of profile correlations),
#: it waits once, for her next phrase, and comes in then on whatever the tracker holds.
#: Measured offline on 40 recorded guitar excerpts: 10 of the 25 entries in a wrong
#: key and none of the 14 right ones were under 0.05 at the entry bar, and the tracker
#: was right on 19 of them two bars later.  Entries that wait come in one phrase later;
#: on generated parts, whose key is as clear at the entry as it gets, it changes little.
ENTRY_KEY_MARGIN = 0.05

DRUMS, BAND, FULL = "drums", "band", "full"

#: Detect mode: after this many heard bars with no four running on one key, take the
#: tracker's latest word for it rather than leave the player on drums for the whole take.
DETECT_KEY_FALLBACK_BARS = 16

#: Strokes per beat at or over which a player is busy, and under which one is sparse
#: (StrokeEar).  Calibrated (2026-11-20) on a set of 270 generated solo parts,
#: nine styles by five parts by three tempi by two modes, with the tempo and mode plume
#: heard: with these the rule names the part's own style for 96 of the 262 the band came
#: in on, with strokes at the old marks of 1.0 and 0.5 for 66, and with the flux onsets
#: it used to read for 44.  A busy mark anywhere from 1.5 to 3.0 scores 91 or more, under
#: 1.5 no more than 87; a sparse mark from 0.7 to 1.0 scores the same.  Per style, median
#: strokes a beat over the first seven bars: trap and lofi 0.54, hiphop 0.86, jazz 0.96,
#: folk 0.96, rnb 0.98, house 1.50, midwest 1.75, synthwave 1.96, and 40 recorded
#: acoustic guitar excerpts 1.1 to 1.5 (interquartile).
_BUSY_STROKES_PER_BEAT = 1.55
_SPARSE_STROKES_PER_BEAT = 0.8

#: Folk is a claim about how the guitar is played, never about the tempo alone, so it
#: needs evidence the rest of the rule has no use for (FolkEar).  It answers only at or
#: above this confidence and inside folk's own range.
FOLK_CONFIDENCE = 0.75
FOLK_BPM = (60.0, 132.0)


def detect_style(bpm: float, mode: str | None = None, busy: float | None = None, *,
                 folk: float | None = None) -> str:
    """A style for a player who did not name one: a guess, and the page says so.

    The tempo does most of the work, and each style's own range sets the bands.  The
    mode and how busy the playing is (strokes per beat, StrokeEar) only split a band.
    None means not known yet, or on a room mic not knowable, and counts as neither minor
    nor busy.  The busy and sparse marks are calibrated on generated parts, never on a
    real player (_BUSY_STROKES_PER_BEAT); the tempo bands are not calibrated at all.

    A minor key with room in it is the only sign this rule has of r&b, hip hop or house,
    so those three are claims about tempo and mode and nothing more.  It hears nothing
    that could tell jazz from anything else played fast, so being fast is not enough:
    jazz only answers above 150, where no other style's range reaches, for a player a
    direct input knows is leaving a lot of room, and a sparse player at those tempos is
    still only a guess.  Everyone else fast keeps midwest, and a slow jazz player, or one
    on a room mic, has to name it.

    ``folk`` is FolkEar's confidence that she is strumming folk, None when it has not
    heard enough or cannot hear her alone.  It is the only answer that comes from how she
    plays, so it is asked first, and only a confident one inside folk's range counts.
    """
    if folk is not None and folk >= FOLK_CONFIDENCE and FOLK_BPM[0] <= bpm <= FOLK_BPM[1]:
        return "folk"
    busy_now = busy is not None and busy >= _BUSY_STROKES_PER_BEAT
    minor_with_room = mode == "minor" and not busy_now
    if bpm < 80.0:
        return "rnb" if minor_with_room and bpm >= 60.0 else "lofi"
    if bpm < 90.0:
        if busy_now:
            return "midwest"
        return "hiphop" if minor_with_room else "lofi"
    if bpm < 100.0:
        return "hiphop" if minor_with_room else "midwest"
    if bpm < 118.0:
        return "synthwave" if minor_with_room else "midwest"
    if bpm < 130.0:
        return "house" if minor_with_room else "midwest"
    if bpm <= 150.0:
        return "midwest" if busy is not None and busy >= _SPARSE_STROKES_PER_BEAT else "trap"
    known_sparse = busy is not None and busy < _SPARSE_STROKES_PER_BEAT
    return "jazz" if known_sparse and bpm <= 220.0 else "midwest"


def _mode_of(key: str | None) -> str | None:
    if not key:
        return None
    try:
        return parse_key(key).mode
    except ValueError:
        return None


#: The beat a detected take plays until it knows her style and key.  Juniper,
#: 2026-11-06: "the lofi intro drum beat felt weird and clunky ... it should give a
#: simple beat until it both key and genre deduces".  The count's pick was a guess from
#: the tempo alone, so its swing, its pushed and dragged hits and its 808 were a style
#: she had not chosen yet.  A plain kick, snare and hat was still "too opinionated --
#: it should almost just be like a metronome or a wood block".  So: one recorded
#: cross-stick click on each beat, the one a little stronger, straight, well under the
#: band.  No groove, nothing of any genre; the style's own drums come in with the band.
PLAIN_INTRO_PATTERNS = {
    "click": ("rs:v1.0 ~ ~ ~ rs:v0.6 ~ ~ ~ rs:v0.6 ~ ~ ~ rs:v0.6 ~ ~ ~",),
}
PLAIN_INTRO_VOICES = {"click": "kit_rim"}
PLAIN_INTRO_GAIN_DB = -2.0


def plain_intro_style(name: str) -> dict:
    """``name``'s style with its drums swapped for the plain beat.

    Its harmony and melody stay, unheard (the drums stage holds them back), so the
    engine is a whole one; every drum role it had that the plain beat lacks goes."""
    from .styles_lib import load_style  # noqa: PLC0415
    s = load_style(name)
    keep = HARMONY_ROLES + MELODY_ROLES
    s["swing"] = 0.5
    s["timing_bias_ms"] = {}
    s["drum_patterns"] = {r: list(p) for r, p in PLAIN_INTRO_PATTERNS.items()}
    s.pop("slow_drum_patterns", None)
    s["fills"] = {}
    s["voices"] = {**{r: v for r, v in s["voices"].items() if r in keep},
                   **PLAIN_INTRO_VOICES}
    s["fx_defaults"] = {r: v for r, v in s.get("fx_defaults", {}).items() if r in keep}
    s["role_gain_db"] = {r: v for r, v in (s.get("role_gain_db") or {}).items()
                         if r in keep}
    s.pop("live_only_roles", None)
    return s


#: The click's role.  It stays in the style that takes over from the plain beat
#: (bridged_style), so it can fade out under the drums coming in.
CLICK_ROLE = "click"
#: The click's level in the plain beat: the mix default, under the plain bus gain.
_CLICK_ROLE_DB = -6.0


def bridged_style(name: str) -> dict:
    """``name``'s own style, with the plain beat's click kept as one more role.

    Juniper, 2026-11-07: when the click stopped dead on the entry bar and the style's
    drums came in under it, the beat "came in awkwardly".  So the click carries on
    through the entry, at the level the plain beat had it, and fades out as the
    drums swell in (click_bridge_level).  The plain beat's bus was
    :data:`PLAIN_INTRO_GAIN_DB` under the band's, so the click's own gain takes that
    on, and a level of 1.0 sounds as it did over the intro."""
    from .styles_lib import load_style  # noqa: PLC0415
    s = load_style(name)
    s["drum_patterns"] = {**s["drum_patterns"],
                          CLICK_ROLE: list(PLAIN_INTRO_PATTERNS[CLICK_ROLE])}
    s["voices"] = {**s["voices"], CLICK_ROLE: PLAIN_INTRO_VOICES[CLICK_ROLE]}
    s["role_gain_db"] = {**(s.get("role_gain_db") or {}),
                         CLICK_ROLE: _CLICK_ROLE_DB + PLAIN_INTRO_GAIN_DB}
    return s


def click_bridge_level(bars_in: int, ramp_bars: int) -> float:
    """The click's level ``bars_in`` bars after the band came in: the inverse of the
    drums' swell.  Down a step on the entry bar, a step less each bar, and gone on
    the bar the drums reach full (``ramp_bars``).  With no ramp the drums are full at
    once, and so is the click gone."""
    ramp = int(ramp_bars)
    if ramp <= 0 or bars_in < 0 or bars_in >= ramp:
        return 0.0
    return 1.0 - (int(bars_in) + 1) / float(ramp + 1)


def plain_phrase_feel(bars_in: int, phrase_bars: int) -> float | None:
    """A bar of the plain first phrase after the band came in: how much of the
    style's swing and push it plays (DuetEngine.render_bar's ``entry_feel``), eased
    in from straight over the phrase.  None once the phrase is over: the style's own
    figures and feel from there."""
    n = int(phrase_bars)
    if n <= 0 or bars_in < 0 or bars_in >= n:
        return None
    return int(bars_in) / float(n)


#: A stroke is a peak of the attack at least this share of what the bar's strongest
#: attacks reach (their 99th percentile), and no nearer the last one than _STROKE_SEP
#: hops.  ``Frame.attack`` is a log ratio, so the strokes found barely move with the
#: input level.  ``Frame.onset`` is picked against a fixed flux threshold and does: the
#: same synthetic strum had 72 onsets in seven bars at -6 dBFS and 19 at -30, its strum
#: share went from 0.29 to 0.63, and the guess changed with the gain knob.
_STROKE_PCT, _STROKE_SHARE, _STROKE_SEP = 99.0, 0.5, 4
#: A stroke is a strum when its attack stays above half its peak for this many hops
#: (32 ms at 48 kHz): a pick dragged across the strings.  A picked note, a drum hit or
#: a synth is over in one or two.
_RAKE_HOPS = 3
#: Strum share at which the strum's evidence starts, and where it is full.  The hops
#: counted are one unbroken run around the peak: tremolo picking puts the next pick
#: inside the reading window, so its attack rises again after dipping, and counting from
#: the first hop above half to the last called it a strum (0.88 to 0.96).  Measured that
#: way (2026-11-10) over every seven bars of each take, at input levels up to 30 dB
#: apart: slowly raked synthetic strums 0.22 to 0.57, nineteen in twenty over 0.29; a
#: strum quicker than about 20 ms 0.09 to 0.20, which is not heard as one; fingerpicking
#: 0.00 to 0.08; tremolo picking 0.00 to 0.03.  15 real direct-input takes, which were
#: never strummed folk, read 0.03 in the middle and under 0.35 nineteen times in twenty;
#: one of them has a strummed passage that reads up to 0.61.  Between those levels one
#: take's share moves by 0.13 at most, where with the flux onsets it moved by 0.56.
#: Full sits where those real takes reach it as often as they reached the old mark with
#: the flux onsets, 27 of their 1129 windows, 26 of them in that one passage.  None sits
#: where an acoustic sound in a major key needs a share of 0.30, as it did.
_RAKE_NONE, _RAKE_FULL = 0.18, 0.40
#: Before this much is heard there is no folk evidence either way: four strokes a bar.
#: The strummed takes had 29 to 52 strokes in seven bars.  One real take read 0.50 on 14.
_FOLK_MIN_BARS, _FOLK_MIN_ONSETS = 6, 24
#: What each sign is worth.  Only the strum separates on its own; the acoustic sound
#: (sim.classify_input's two measures) reads "programmed" for every one of those 15
#: takes, a guitar through a pickup, so it only helps, and a major key only helps.  A
#: strum alone never reaches FOLK_CONFIDENCE.  Without the acoustic sound it takes a
#: full strum in a major key: a distorted electric solo measured 0.44, and stays out.
_W_STRUM, _W_ACOUSTIC, _W_MAJOR = 0.55, 0.25, 0.2
#: Attack-breadth histogram bins for the acoustic measure's 75th percentile.
_BREADTH_BINS = 25


def _strokes(att, ok, stroke_at: float, last: int, start: int = 3, *, slow=None,
             last_slow: bool = False, slow_sep: int = 0) -> tuple[list[int], int]:
    """The strokes in ``att`` (a bar's attack with the last bar's tail in front): each
    one's width in hops, and where the last one is.

    A stroke is a peak of the attack at ``stroke_at`` or over, on a frame ``ok`` allows,
    and at least _STROKE_SEP hops after the one before (``last``).  It is read with 3
    frames before it and 6 after, so the last 6 wait for the next bar, and the tail
    carried in front holds them and ``start`` frames before them.  Its width is the
    unbroken run of hops around the peak at half of it or more.

    ``slow`` marks the frames where a stroke would be a slow one (StrokeEar's legato
    note change): next to one of those, it or the one before (``last_slow``), two strokes
    must be ``slow_sep`` hops apart."""
    widths = []
    for k in range(min(start, len(att)), len(att) - 6):
        if not ok[k] or att[k] < stroke_at:
            continue
        slow_here = slow is not None and bool(slow[k])
        if k - last < (slow_sep if slow_here or last_slow else _STROKE_SEP):
            continue
        w = att[k - 3:k + 6]
        if att[k] < float(w[:7].max()):
            continue                 # not the peak: a hop either side is higher
        last, last_slow = k, slow_here
        above = w >= 0.5 * float(att[k])
        lo = hi = 3
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(w) - 1 and above[hi + 1]:
            hi += 1
        widths.append(int(hi - lo + 1))
    return widths, last


class FolkEar:
    """Is she strumming an acoustic guitar?  The evidence for a folk guess.

    Fed one bar of her frames at a time (``Frame`` or anything with ``rms_db``,
    ``attack`` and ``attack_breadth``), only from an input that hears her alone.
    Everything it measures comes from the attack, so how loud she arrives changes
    little.  Memory stays bounded however long the take: counts and a histogram.
    """

    def __init__(self):
        # Here, not per bar: the first bar is read on the planning thread.
        from . import sim  # noqa: PLC0415
        from .listen.beat import BEAT_ATTACK_FLOOR, BEAT_ATTACK_MULT  # noqa: PLC0415
        self._mult, self._floor = BEAT_ATTACK_MULT, BEAT_ATTACK_FLOOR
        self._silent_db, self._broad_at = sim.INPUT_SILENT_FRAME_DB, sim.INPUT_BROAD_BREADTH
        self._strong_max, self._broad_max = (sim.INPUT_BREADTH_STRONG_MAX,
                                             sim.INPUT_BROAD_FRAC_MAX)
        self._tail: list = []            # the last frames, so a strum across a bar line counts
        self._last_stroke = -_STROKE_SEP  # where the last stroke was, counted in the tail
        self._bars = 0
        self._onsets = 0
        self._raked = 0
        self._audible = 0
        self._broad = 0
        self._strong_hist = np.zeros(_BREADTH_BINS, dtype=np.int64)

    def bar(self, frames) -> None:
        frames = [f for f in frames if f is not None]
        if not frames:
            return
        seen = self._tail + frames
        rms = np.array([float(f.rms_db) for f in seen])
        att = np.array([float(getattr(f, "attack", 0.0)) for f in seen])
        br = np.array([float(getattr(f, "attack_breadth", 0.0)) for f in seen])
        audible = rms > self._silent_db
        if not audible.any():
            self._keep_tail(seen, self._last_stroke)
            return
        self._bars += 1
        new = slice(len(self._tail), len(seen))
        self._audible += int(audible[new].sum())
        self._broad += int(((br >= self._broad_at) & audible)[new].sum())
        strong_at = max(self._mult * float(np.median(att[audible])), self._floor)
        strong = (att > strong_at) & audible
        idx = np.minimum((br[new][strong[new]] * _BREADTH_BINS).astype(int), _BREADTH_BINS - 1)
        np.add.at(self._strong_hist, idx, 1)
        stroke_at = max(strong_at, _STROKE_SHARE
                        * float(np.percentile(att[audible], _STROKE_PCT)))
        widths, last = _strokes(att, audible, stroke_at, self._last_stroke)
        self._onsets += len(widths)
        self._raked += sum(int(w >= _RAKE_HOPS) for w in widths)
        self._keep_tail(seen, last)

    def _keep_tail(self, seen: list, last: int) -> None:
        self._tail = seen[-9:]
        self._last_stroke = max(last - (len(seen) - len(self._tail)), -_STROKE_SEP)

    @property
    def strum_share(self) -> float | None:
        if self._bars < _FOLK_MIN_BARS or self._onsets < _FOLK_MIN_ONSETS:
            return None
        return self._raked / self._onsets

    @property
    def acoustic(self) -> bool:
        """sim.classify_input's verdict, from the same two measures over what was heard."""
        n = int(self._strong_hist.sum())
        if not n or not self._audible:
            return False
        p75_bin = int(np.searchsorted(np.cumsum(self._strong_hist), 0.75 * n))
        p75 = (p75_bin + 1) / _BREADTH_BINS
        return p75 <= self._strong_max and self._broad / self._audible <= self._broad_max

    def confidence(self, mode: str | None) -> float | None:
        """0..1 that this is folk, or None before there is enough to say."""
        share = self.strum_share
        if share is None:
            return None
        strum = min(1.0, max(0.0, (share - _RAKE_NONE) / (_RAKE_FULL - _RAKE_NONE)))
        return (_W_STRUM * strum + _W_ACOUSTIC * float(self.acoustic)
                + _W_MAJOR * float(mode == "major"))

    def why(self, mode: str | None) -> str:
        share = self.strum_share
        if share is None:
            return f"folk: not enough heard ({self._bars} bars, {self._onsets} onsets)"
        signs = [f"strummed {share:.0%} of onsets"]
        if self.acoustic:
            signs.append("acoustic sound")
        if mode == "major":
            signs.append("major key")
        return f"folk {self.confidence(mode):.2f}: " + ", ".join(signs)

    def memory_items(self) -> int:
        return len(self._tail) + int(self._strong_hist.size)


#: A stroke is something new that she plays, and there are two ways for a note to be new.
#: A strum or a picked string lifts her level: by _RISE_DB from the lowest of the
#: _RISE_BEFORE hops before the peak to the highest of the three from it.  A legato line
#: (bowed, blown, swelled, a soft synth lead) may not: its next note rises as the last
#: fades, so the level stays put, and a level rise alone read a steady eighth-note line as
#: under one stroke in four.  What does change is the pitch: the chroma _CHROMA_AHEAD hops
#: after the peak is at least _CHROMA_MOVE (half the L1 distance of the two, 0 to 1) from
#: the one just before it.  Vibrato and beating partials stay inside a pitch class, so
#: they move neither.  The chroma is L1-normalised and the rise is in dB, so neither moves
#: with the gain.
_RISE_DB = 1.0
_RISE_BEFORE = 6
_CHROMA_MOVE = 0.4
_CHROMA_AHEAD = 5
#: A legato note change is not over in a hop.  One note crosses into the next over tens of
#: milliseconds, and the crossing holds a new pitch, then often a dip and a lift in level
#: as well: two or three peaks of the attack for one note, and a quarter-note line with a
#: 60 ms crossing read 1.5 to 1.8 strokes a beat.  So next to a stroke that is new by its
#: pitch alone, another must be _LEGATO_SEP hops away (128 ms: sixteenths up to 117 bpm
#: keep every note, and faster legato lines keep every other one, which is busy still).
#: Two strokes that each lift the level keep _STROKE_SEP.
_LEGATO_SEP = 12
#: The new pitch must also sound: _CHROMA_AHEAD hops on, her level no more than this under
#: where it was just before.  A note that stops, cut short or damped, leaves the pitches
#: still ringing, which moves the chroma too, and its edge has an attack of its own.
_CHROMA_HOLD_DB = 6.0
#: Her own level is the 95th percentile of the frames heard from her, and a stroke counts
#: only where she reaches within _SOUNDING_DB of it in the hops from its peak on.  The tail
#: of a ringing note sinks toward the input's hiss, and the attack, a log ratio in every
#: band, rises and falls there as much as anywhere: at the top of the gain knob the hops
#: 20 to 60 dB under her are tail and hiss, never a note she meant.  The same goes for
#: what she barely touches, a string brushed 30 dB under her notes: clear at the top of
#: the knob and under the hiss at the bottom, so counting it would move the count with
#: the knob (it cut the count on 28 of 40 recorded guitar excerpts at 0 dB, by up to 0.6
#: strokes a beat, and on none at -40 dB).  It is relative to her and not a fixed dBFS gate, so
#: it moves with the gain knob as she does.
_SOUNDING_DB = 20.0
_LEVEL_PCT = 95.0
#: Quieter than this is digital silence, not a level.  The level histogram has 1 dB bins
#: from here to 0 dBFS.
_FLOOR_DB = -100.0
_LEVEL_BINS = 101
#: And its attack must reach this share of the bar's strongest (FolkEar's is 0.5): a
#: picked line's softer notes over strings still ringing reach a third of the loud ones'
#: or less, and at 0.5 eighth notes picked thumb and finger read 1.2 to 1.5 a beat.
#: Measured over the first seven bars of 270 generated parts and 40 recorded guitar
#: excerpts: the count at -20 dB within a quarter of the one at 0 dB for 98% of inputs, at
#: -40 dB for 91%; a lead, bass or picked line's count is within 3% of its written notes
#: or under them, by 16% or more one time in ten.
_BUSY_STROKE_SHARE = 0.25


class StrokeEar:
    """How much she plays: strokes per beat, whatever the input gain.

    Fed one bar of her frames at a time (anything with ``rms_db`` and ``attack``, and
    ``chroma`` where there is one).  A stroke is FolkEar's (``_strokes``): a peak of the
    attack, which is a log ratio, over a share of the bar's own strongest attacks, so the
    count barely moves with the gain, where ``Frame.onset``, picked against a fixed flux
    threshold, does: the same inputs read a median 1.75, 1.39 and 1.00 onsets a beat at 0,
    -20 and -40 dB.  Here a stroke must also be new, a rise or a new pitch (_RISE_DB), and
    sound near her own level (_SOUNDING_DB).  Memory stays bounded: a tail of frames and a
    histogram.
    """

    def __init__(self):
        from .listen.beat import BEAT_ATTACK_FLOOR, BEAT_ATTACK_MULT  # noqa: PLC0415
        self._mult, self._floor = BEAT_ATTACK_MULT, BEAT_ATTACK_FLOOR
        self._tail: list = []
        self._last_stroke = -_LEGATO_SEP
        self._last_slow = False
        self._levels = np.zeros(_LEVEL_BINS, dtype=np.int64)
        self.strokes = 0

    def own_level_db(self) -> float | None:
        """Her level so far (dBFS), None before anything audible."""
        n = int(self._levels.sum())
        if not n:
            return None
        k = int(np.searchsorted(np.cumsum(self._levels), _LEVEL_PCT / 100.0 * n))
        return _FLOOR_DB + k + 1.0

    def bar(self, frames) -> None:
        frames = [f for f in frames if f is not None]
        if not frames:
            return
        seen = self._tail + frames
        rms = np.array([float(f.rms_db) for f in seen])
        att = np.array([float(getattr(f, "attack", 0.0)) for f in seen])
        new = rms[len(self._tail):]
        new = new[new > _FLOOR_DB]
        if new.size:
            idx = np.clip((new - _FLOOR_DB).astype(int), 0, _LEVEL_BINS - 1)
            np.add.at(self._levels, idx, 1)
        own = self.own_level_db()
        audible = rms > _FLOOR_DB
        n = len(rms)
        if own is None or not audible.any() or n < _RISE_BEFORE + _CHROMA_AHEAD + 1:
            self._keep_tail(seen, self._last_stroke)
            return
        win = np.lib.stride_tricks.sliding_window_view
        # reach[k]: the loudest of hops k to k + 3 (the last three are never read)
        reach = np.full(n, -np.inf)
        reach[:n - 3] = win(rms, 4).max(axis=1)[:n - 3]
        sounding = audible & (reach > own - _SOUNDING_DB)
        # rise[k] = max(rms[k:k+3]) - min(rms[k-6:k]), for k from _RISE_BEFORE to n - 3
        rise = np.full(n, -np.inf)
        rise[_RISE_BEFORE:n - 2] = (win(rms[_RISE_BEFORE:], 3).max(axis=1)
                                    - win(rms[:n - 3], _RISE_BEFORE).min(axis=1))
        risen = rise >= _RISE_DB
        ok = sounding & (risen | self._moved(seen, rms))
        stroke_at = max(self._mult * float(np.median(att[audible])), self._floor,
                        _BUSY_STROKE_SHARE * float(np.percentile(att[audible], _STROKE_PCT)))
        widths, last = _strokes(att, ok, stroke_at, self._last_stroke, start=_RISE_BEFORE,
                                slow=~risen, last_slow=self._last_slow,
                                slow_sep=_LEGATO_SEP)
        if widths:
            self._last_slow = not bool(risen[last])
        self.strokes += len(widths)
        self._keep_tail(seen, last)

    @staticmethod
    def _moved(seen: list, rms: np.ndarray) -> np.ndarray:
        """moved[k]: the pitch classes _CHROMA_AHEAD hops after k are not the ones two
        hops before it (the hop before k may already hold its attack), and they sound
        (_CHROMA_HOLD_DB).  False where either hop is silent or has no chroma."""
        n = len(seen)
        moved = np.zeros(n, dtype=bool)
        c = np.array([np.zeros(12) if getattr(f, "chroma", None) is None
                      else np.asarray(f.chroma, dtype=float) for f in seen])
        s = c.sum(axis=1)
        c = np.divide(c, s[:, None], out=np.zeros_like(c), where=s[:, None] > 0)
        before, after = c[:n - 2 - _CHROMA_AHEAD], c[2 + _CHROMA_AHEAD:]
        lit = ((s[:n - 2 - _CHROMA_AHEAD] > 0) & (s[2 + _CHROMA_AHEAD:] > 0)
               & (rms[2 + _CHROMA_AHEAD:] >= rms[:n - 2 - _CHROMA_AHEAD] - _CHROMA_HOLD_DB))
        moved[2:n - _CHROMA_AHEAD] = lit & (0.5 * np.abs(before - after).sum(axis=1)
                                            >= _CHROMA_MOVE)
        return moved

    def _keep_tail(self, seen: list, last: int) -> None:
        # The six that wait for this bar's strokes to be read, and the hops before them
        # that a rise is measured from.
        self._tail = seen[-(6 + _RISE_BEFORE):]
        self._last_stroke = max(last - (len(seen) - len(self._tail)), -_LEGATO_SEP)

    def memory_items(self) -> int:
        return len(self._tail) + int(self._levels.size)


class StylePick:
    """Where a detected style comes from over a take.

    Drums need a style the moment the count lands, when only the tempo is known, so it
    picks once there.  It picks once more as the band enters, knowing the key and how
    busy the player has been, and the drums changing then reads as part of the entrance.
    After that it never moves.  A room mic hears the drums as onsets too, so its
    ``trust_busy`` is off, and so is ``hear_technique``: whether she strums is only
    hers to hear on an input that carries nothing else.

    How busy she is comes from her frames when there are any (StrokeEar), so it does not
    move with her input gain, and from the onsets counted for her otherwise: notes on a
    MIDI input, where every one is a stroke.  ``why`` gives every input of the pick.
    """

    def __init__(self, *, key: str | None = None, trust_busy: bool = False,
                 hear_technique: bool = False):
        self._mode = _mode_of(key)
        self._trust_busy = bool(trust_busy)
        self._onsets = 0
        self._beats = 0
        self._stroke_beats = 0
        self.settled = False
        #: How she plays, which only an input hearing her alone can tell (FolkEar).
        self.folk_ear = FolkEar() if hear_technique else None
        self.stroke_ear = StrokeEar() if trust_busy else None
        self.why = ""

    @property
    def wants_frames(self) -> bool:
        """Whether her frames are worth handing over: something here reads them."""
        return self.folk_ear is not None or self.stroke_ear is not None

    def distrust(self) -> None:
        """The input turned out to hold the band: how busy it sounds is no longer hers."""
        self._trust_busy = False

    def heard_bar(self, onsets: int, frames=None) -> None:
        """One bar she played: the onsets counted in it, and her frames, None if they
        were not kept."""
        self._onsets += int(onsets)
        self._beats += 4
        if frames is not None:
            if self.stroke_ear is not None:
                self.stroke_ear.bar(frames)
                self._stroke_beats += 4
            if self.folk_ear is not None:
                self.folk_ear.bar(frames)

    @property
    def busy(self) -> float | None:
        """Strokes (or, with no frames, onsets) per beat over the bars heard, if this
        input can say."""
        if not self._trust_busy or not self._beats:
            return None
        if self._stroke_beats:
            return self.stroke_ear.strokes / self._stroke_beats
        return self._onsets / self._beats

    def at_count(self, bpm: float) -> str:
        return detect_style(bpm, self._mode)

    def at_entry(self, bpm: float, key: str | None) -> str:
        self.settled = True
        mode = _mode_of(key) or self._mode
        folk = None if self.folk_ear is None else self.folk_ear.confidence(mode)
        busy = self.busy
        style = detect_style(bpm, mode, busy, folk=folk)
        self.why = "; ".join([f"{style} from: bpm {bpm:.1f}", f"mode {mode or 'unknown'}",
                              self._busy_why(busy),
                              self.folk_ear.why(mode) if self.folk_ear is not None
                              else "folk: not heard on this input"])
        return style

    def _busy_why(self, busy: float | None) -> str:
        if busy is None:
            return ("busy: unknown, nothing heard yet" if self._trust_busy
                    else "busy: not hers alone to hear on this input")
        what = "strokes" if self._stroke_beats else "onsets"
        word = ("busy" if busy >= _BUSY_STROKES_PER_BEAT else
                "sparse" if busy < _SPARSE_STROKES_PER_BEAT else "neither busy nor sparse")
        return (f"busy {busy:.2f} {what} a beat ({word}: busy from "
                f"{_BUSY_STROKES_PER_BEAT:g}, sparse under {_SPARSE_STROKES_PER_BEAT:g})")


@dataclass(frozen=True)
class EntryState:
    stage: str               # drums | band | full
    held_back: tuple[str, ...]   # roles not allowed to sound yet; the rest play
    key: str | None          # committed key, None until the band is in
    chord: str | None        # chord from the committed progression
    reason: str


_NOTHING_HEARD = HarmonyState(t_s=0.0, key=None, key_conf=0.0, chord=None,
                              chord_conf=0.0, register_low_midi=None,
                              register_high_midi=None)


#: How much of a bar's pitch energy its three strongest notes must hold for the bar
#: to count as someone playing.  Drums barely favour any note: bars measured with
#: nobody playing (a real take, 2026-11-02) held 0.29-0.37, with guitar 0.46-0.76.
TONAL_TOP3 = 0.42


class GuitarEar:
    """Hear a guitar over drums alone.  On a mic the band can reach it only works before
    the band comes in: once bass and chords are in the room, the mic's pitches are mostly
    theirs.  An input the band cannot reach goes on being read under the band."""

    def __init__(self):
        self._heard = np.zeros(12)
        self._since_tonal = 99
        #: This bar alone sounded played, confirmed or not.  A fade needs this: a
        #: player coming back must not wait a bar for confirmation.
        self.last_bar_tonal = False
        #: 1 when this bar confirmed her on the tonal bar just before it, so her
        #: playing began a bar earlier; 0 otherwise.  Across a gap the earlier bar is
        #: as likely a ringing tail, and this bar is where she (re)started.
        self.began_bars_ago = 0

    def bar(self, chromas) -> bool:
        """Is someone playing?  One tonal bar could be a drum ringing near the
        threshold, so it takes two within three bars.  Notes of a tonal bar are
        remembered."""
        frames = [c for c in chromas if float(np.sum(c)) > 0.0]
        mean = np.mean(frames, axis=0) if frames else None
        self.last_bar_tonal = (mean is not None
                               and float(np.sort(mean)[-3:].sum()) >= TONAL_TOP3)
        if not self.last_bar_tonal:
            self._since_tonal += 1
            return False
        confirmed = self._since_tonal <= 2
        self.began_bars_ago = 1 if self._since_tonal == 1 else 0
        self._since_tonal = 1
        self._heard += mean
        return confirmed

    def notes(self, count: int = 5) -> tuple[int, ...]:
        """Her most-played pitch classes, strongest first; empty if none heard."""
        if not np.any(self._heard):
            return ()
        order = np.argsort(self._heard)[::-1][:count]
        return tuple(int(i) for i in order if self._heard[i] > 0.0)


#: How much of a mic must be the band, by :func:`band_share`, for a stretch of it to count
#: as the band being in the mic.  A guess, set against the fake device and made-up speakers
#: and rooms: a clean mic reads up to 0.15 there, held notes on the band's own partials
#: 0.03, and the band through a small speaker and a lively room 0.43 and up, with or
#: without a player over it.  No real headphones, speakers or mic have been measured.
BAND_IN_MIC_SHARE = 0.3
#: Each stretch judged, and how many more of them must say yes than no before the take
#: stops believing the mic: about eight seconds of evidence, so two or three bars.
BAND_IN_MIC_WINDOW_S = 2.0
BAND_IN_MIC_WINDOWS = 4
#: A stretch counts either way only when the band is sounding and the mic is not empty.
BAND_IN_MIC_BAND_FLOOR_DB = -50.0
BAND_IN_MIC_MIC_FLOOR_DB = -60.0
#: Output latency, air and input latency, generously.
BAND_IN_MIC_MAX_DELAY_S = 0.7
#: What a small speaker reproduces and a mic picks up well.  Most of a band's power is
#: bass, which a small speaker never plays, so the bass says nothing about a leak.
BAND_IN_MIC_LOW_HZ = 300.0
BAND_IN_MIC_HIGH_HZ = 4000.0
#: Nothing above :data:`BAND_IN_MIC_HIGH_HZ` is listened to, so both signals are brought
#: down to a rate that just holds it before anything else is done: a fifth of the samples,
#: and so a fifth of the work.
BAND_IN_MIC_RATE_HZ = 9600
#: The filter the band may reach the mic through, in samples at that rate: 107 ms, which
#: holds a speaker's response and the early part of a room.
BAND_IN_MIC_FILTER = 1024
#: Only where the band itself has something to say: this far under its strongest frequency.
BAND_IN_MIC_BAND_RANGE_DB = 30.0


def band_share(mic, band, sr: int, max_delay_s: float = BAND_IN_MIC_MAX_DELAY_S) -> float:
    """How much of ``mic`` is ``band``, come through a speaker and a room, 0 to 1.

    Not a delayed copy: a speaker drops the bass and a room adds its tail, so the band in
    a mic is the band through a short filter nobody knows.  So this lines the two up on
    their best delay, and then asks, frequency by frequency between
    :data:`BAND_IN_MIC_LOW_HZ` and :data:`BAND_IN_MIC_HIGH_HZ`, how much of the mic there
    is explained by the band through any filter up to :data:`BAND_IN_MIC_FILTER` long
    (the coherence of the two, averaged over short pieces of the stretch).  The answer is
    the middle value over the frequencies where the band is sounding.

    The middle, not the mean: a player's notes sit on a few frequencies, however loud, and
    leave the rest to whatever else is in the mic.  So a band under a louder player still
    reads high, and a player holding the band's own notes on a clean mic reads low.

    A stretch holding a sample that is not a number says nothing, and reads 0.
    """
    n = min(len(mic), len(band))
    mic = np.asarray(mic[:n], dtype=np.float64)
    band = np.asarray(band[:n], dtype=np.float64)
    if n == 0 or not (np.all(np.isfinite(mic)) and np.all(np.isfinite(band))):
        return 0.0
    if float(np.sum(mic ** 2)) <= 1e-9 or float(np.sum(band ** 2)) <= 1e-9:
        return 0.0
    step = max(1, int(sr) // BAND_IN_MIC_RATE_HZ)
    if step > 1:
        # A short low-pass and every ``step``-th sample.  Short on purpose: both signals
        # go through the same one, so what it lets fold over is still the band in both.
        from scipy.signal import firwin, upfirdn  # noqa: PLC0415
        low = firwin(61, 1.1 * BAND_IN_MIC_HIGH_HZ, fs=float(sr))
        mic, band = upfirdn(low, mic, 1, step), upfirdn(low, band, 1, step)
    rate = float(sr) / step
    n = len(mic)
    seg, hop = BAND_IN_MIC_FILTER, BAND_IN_MIC_FILTER // 2
    if n < 8 * seg:
        return 0.0
    # The delay, from the phase alone and only in the range a speaker plays: loud bass
    # would otherwise decide it, and the bass may not be in the mic at all.
    from scipy.fft import next_fast_len  # noqa: PLC0415
    size = int(next_fast_len(2 * n, real=True))     # an awkward length costs several times over
    cross = np.fft.rfft(mic, size) * np.conj(np.fft.rfft(band, size))
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    size_of = np.abs(cross)
    keep = (freqs >= BAND_IN_MIC_LOW_HZ) & (freqs <= BAND_IN_MIC_HIGH_HZ) & (size_of > 0.0)
    cross = np.where(keep, cross / np.maximum(size_of, 1e-30), 0.0)
    lag = int(np.argmax(np.abs(np.fft.irfft(cross, size)[:max(1, int(max_delay_s * rate))])))
    mic, band = mic[lag:], band[:n - lag]
    pieces = (len(mic) - seg) // hop + 1
    if pieces < 8:
        return 0.0
    at = np.arange(seg)[None, :] + hop * np.arange(pieces)[:, None]
    window = np.hanning(seg)
    m = np.fft.rfft(mic[at] * window, axis=1)
    b = np.fft.rfft(band[at] * window, axis=1)
    freqs = np.fft.rfftfreq(seg, 1.0 / rate)
    keep = (freqs >= BAND_IN_MIC_LOW_HZ) & (freqs <= BAND_IN_MIC_HIGH_HZ)
    both = np.mean(m * np.conj(b), axis=0)[keep]
    band_power = np.mean(np.abs(b) ** 2, axis=0)[keep]
    mic_power = np.mean(np.abs(m) ** 2, axis=0)[keep]
    explained = np.abs(both) ** 2 / np.maximum(band_power * mic_power, 1e-30)
    sounding = band_power >= float(np.max(band_power)) * 10.0 ** (-BAND_IN_MIC_BAND_RANGE_DB / 10.0)
    if int(np.sum(sounding)) < 8:
        return 0.0
    return min(1.0, float(np.median(explained[sounding])))


class BandInMic:
    """Is the band in a mic the player said it could not reach?

    The safety net under her word: headphones come out, or the yes was wrong, and nothing
    else can see it.  Fed the mic and what the band played, block by block, off the audio
    callback, for as long as the take runs.  Every :data:`BAND_IN_MIC_WINDOW_S` it asks how much of the mic is the band.  A
    stretch that says yes counts one up, one that says no counts one down, and at
    :data:`BAND_IN_MIC_WINDOWS` it latches: one loud bar cannot trip it, and once tripped
    it never goes back, because a take that flips between believing and not is worse than
    either.
    """

    def __init__(self, sr: int):
        self._sr = int(sr)
        self._size = int(BAND_IN_MIC_WINDOW_S * self._sr)
        self._mic: list = []
        self._band: list = []
        self._held = 0
        self._count = 0
        self.latched = False
        #: The band's share of the mic in the stretch that latched it.
        self.share: float | None = None

    def feed(self, mic, band) -> None:
        if self.latched:
            return
        self._mic.append(np.asarray(mic, dtype=np.float32))
        self._band.append(np.asarray(band, dtype=np.float32))
        self._held += len(mic)
        if self._held < self._size:
            return
        mic, band = np.concatenate(self._mic), np.concatenate(self._band)
        self._mic, self._band, self._held = [], [], 0
        for at in range(0, len(mic) - self._size + 1, self._size):
            self._judge(mic[at:at + self._size], band[at:at + self._size])
            if self.latched:
                return

    def _judge(self, mic, band) -> None:
        def db(x):
            return 10.0 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
        if not (np.all(np.isfinite(mic)) and np.all(np.isfinite(band))):
            return                  # a stretch with a sample that is no number is not judged
        if db(band) < BAND_IN_MIC_BAND_FLOOR_DB or db(mic) < BAND_IN_MIC_MIC_FLOOR_DB:
            return                  # nothing to hear, so it says nothing either way
        share = band_share(mic, band, self._sr)
        if share >= BAND_IN_MIC_SHARE:
            self._count += 1
            if self._count >= BAND_IN_MIC_WINDOWS:
                # The share first: another thread reads it as soon as it sees the latch.
                self.share = share
                self.latched = True
        else:
            self._count = max(0, self._count - 1)


class BandEntry:
    def __init__(self, style: str, *, min_drum_bars: int = 8, key_bars: int = 4,
                 melody_delay_bars: int = MELODY_DELAY_BARS, release_bars: int = 8,
                 known_key: str | None = None, after_first_heard_bars: int = 5,
                 key_fallback_bars: int | None = None, phrase_bars: int = 1):
        """``after_first_heard_bars`` counts from when she is confirmed, which is
        about three bars after her first note, so she gets about eight bars.

        ``key_fallback_bars``: the governor wants several bars running on one key, and
        a player who moves around may never give it that.  After this many heard bars,
        commit to the tracker's latest key instead.  None waits for as long as it takes.

        ``phrase_bars``: the band comes in only on the top of one of her groups of this
        many bars, counted from the bar she last started playing in after a rest.  Juniper,
        2026-11-06: "the chords came in not at what felt like the end of a group of
        measures".  It used to come in on whatever bar the key settled, which was bar 8
        of plume's count and bar 6 of hers.  1 enters as soon as it is ready.
        """
        self._min_drum_bars = int(min_drum_bars)
        self._melody_delay = int(melody_delay_bars)
        self._gov = HarmonyGovernor(style, SimpleNamespace(
            harmony_commit_bars=int(key_bars),
            harmony_release_bars=int(release_bars)))
        self._band_bar: int | None = None
        self._heard = 0
        self._known_key = known_key
        self._after_heard = int(after_first_heard_bars)
        self._first_heard: int | None = None
        self._key_fallback_bars = key_fallback_bars
        self._latest_key: str | None = None
        self._phrase_bars = max(1, int(phrase_bars))
        #: The bar she last (re)started playing in: where her groups are counted from.
        #: A phrase is two bars running, so a one-bar flash sets nothing.
        self.phrase_start: int | None = None
        self._run_start: int | None = None
        self._run = 0
        self._rest = PHRASE_REST_BARS
        self._rest_before_run = PHRASE_REST_BARS
        self._ready_since: int | None = None
        self._waited_unsure = False
        self._latest_key_bars = 0
        self.bars_in = 0

    @property
    def band_is_in(self) -> bool:
        return self._band_bar is not None

    def restyle(self, style: str) -> None:
        """The chords come from this style from now on, in the same key."""
        self._gov.restyle(style)

    def chord_at(self, bar: int, harmony) -> str | None:
        """Ask again for a bar already governed, after a :meth:`restyle`."""
        return self._gov.govern(bar, harmony if harmony is not None else _NOTHING_HEARD,
                                resting=True).chord

    def _on_her_phrase(self, bar: int) -> bool:
        return (self._phrase_bars <= 1 or self.phrase_start is None
                or (bar - self.phrase_start) % self._phrase_bars == 0)

    def _phrase_reading(self, heard: int, playing: bool) -> None:
        """Count her phrases from the start of her last run of two or more bars that
        followed a rest of PHRASE_REST_BARS or more (or her first run)."""
        if self._band_bar is not None:
            return
        if not playing:
            self._run = 0
            self._rest += 1
            return
        if self._run == 0:
            self._run_start, self._rest_before_run = heard, self._rest
        self._run += 1
        self._rest = 0
        if self._run == 2 and (self.phrase_start is None
                               or self._rest_before_run >= PHRASE_REST_BARS):
            self.phrase_start = self._run_start

    def update(self, bar: int, harmony, playing: bool, *,
               heard_bar: int | None = None) -> EntryState:
        """``heard_bar``: the bar ``playing`` describes.  The live loop plans a bar
        ahead and reads the last bar that has finished, so it is two bars behind
        ``bar``; None means ``bar`` itself."""
        self._phrase_reading(bar if heard_bar is None else int(heard_bar), playing)
        if playing:
            self._heard += 1
            if self._first_heard is None:
                self._first_heard = bar
        ready = (self._first_heard is not None and bar >= max(
            self._min_drum_bars, self._first_heard + self._after_heard))
        if ready and self._ready_since is None:
            self._ready_since = bar
        # On the top of her phrase, but never more than a phrase late: however her
        # count moves, a ready band is in within phrase_bars bars.
        ready = ready and (self._on_her_phrase(bar)
                           or bar - self._ready_since >= self._phrase_bars)
        if (ready and self._band_bar is None and self._known_key is None
                and not self._waited_unsure and harmony is not None
                and harmony.key is not None and harmony.key_margin < ENTRY_KEY_MARGIN):
            # Two keys are about level: one more of her phrases says which (see
            # ENTRY_KEY_MARGIN).  Once only, so an ambiguous player still gets a band
            # within a phrase.
            self._waited_unsure = True
            self._ready_since = bar
            ready = False
        if self._known_key is not None:
            # She told us the key, so nothing heard can change it.  The band still
            # waits for her to start: drums alone are for playing over.
            if not ready and self._band_bar is None:
                return EntryState(DRUMS, HARMONY_ROLES + MELODY_ROLES + PITCHED_DRUM_ROLES,
                                  None, None,
                                  f"drums: waiting to bring in {self._known_key}")
            if self._gov.committed_key is None:
                self._gov.commit(self._known_key, bar)
            playing, harmony = False, _NOTHING_HEARD
        committed = self._gov.committed_key is not None
        # A bar she sat out says nothing about her key.  Before commitment it
        # never reaches the governor.  Afterwards it has to, because the chord
        # cycle keeps advancing while she rests, but a rest must neither let the
        # tracker's stale belief change key nor interrupt a real key change.
        if playing:
            # A bar the tracker could not name is no evidence either, the same as a
            # rest: one unsure bar must not cost a real key change all its bars.
            unsure = committed and harmony is not None and harmony.key is None
            governed = self._gov.govern(bar, harmony, resting=unsure)
        elif committed:
            governed = self._gov.govern(bar, harmony, resting=True)
        else:
            governed = None
        if playing and harmony is not None and harmony.key is not None:
            self._latest_key_bars = (self._latest_key_bars + 1
                                     if harmony.key == self._latest_key else 1)
            self._latest_key = harmony.key
        if (self._key_fallback_bars is not None and self._gov.committed_key is None
                and self._heard >= self._key_fallback_bars
                and self._latest_key is not None):
            self._gov.commit(self._latest_key, bar)
            governed = self._gov.govern(bar, harmony if harmony is not None else _NOTHING_HEARD,
                                        resting=True)
        if (ready and self._band_bar is None and self._known_key is None
                and self._latest_key is not None
                and self._latest_key_bars >= ENTRY_KEY_BARS
                and self._gov.committed_key not in (None, self._latest_key)):
            # Until the band is in, a commitment is one nobody has heard, so it is
            # only a first guess.  The guess is the first run of bars that agreed,
            # and a player who opens on another chord than her home one (a real
            # take opened on the ii, G# minor, of F# major) makes that run the wrong
            # key.  The tracker's present belief has heard every bar since, and its
            # own hysteresis keeps a passing phrase from moving it, so the band comes
            # in on that, once it has held for ENTRY_KEY_BARS of her bars: one bar of
            # it is a passing chord.  Its latest word, so a rest on the entry bar
            # (likely, now the band waits for her phrase) still counts.  From here on
            # only the release rule can change the key.
            self._gov.commit(self._latest_key, bar)
            governed = self._gov.govern(
                bar, harmony if harmony is not None else _NOTHING_HEARD, resting=True)
        key = self._gov.committed_key
        if key is None or (not ready and self._band_bar is None):
            why = (f"drums: {key} settled, in at the top of her next "
                   f"{self._phrase_bars} bars"
                   if key is not None and not self._on_her_phrase(bar)
                   else f"drums: heard {self._heard} bars, no settled key")
            return EntryState(DRUMS, HARMONY_ROLES + MELODY_ROLES + PITCHED_DRUM_ROLES,
                              None, None, why)
        if self._band_bar is None:
            self._band_bar = bar
        self.bars_in = bar - self._band_bar
        chord = None if governed is None else governed.chord
        # Drums and bass first, then the chords, then the melody: one bar bringing all
        # of it in at once over the click was heard as not coming in smooth.
        if bar - self._band_bar < min(CHORDS_DELAY_BARS, self._melody_delay):
            return EntryState(BAND, CHORD_ROLES + MELODY_ROLES, key, chord,
                              f"band: drums and bass in {key}")
        if bar - self._band_bar < self._melody_delay:
            return EntryState(BAND, MELODY_ROLES, key, chord,
                              f"band: bass and chords in {key}")
        return EntryState(FULL, (), key, chord, f"full: melody over {key}")


#: The shape of the song once the band is in: (name, bars, density).  Density
#: picks how busy the hats, keys and melody are; the kick and snare never change,
#: so the song grows without the groove being rewritten under her.
SONG_ARC = (("verse", 16, 0.45), ("lift", 16, 0.85), ("verse", 8, 0.5),
            ("break", 4, 0.3), ("lift", 16, 0.9), ("verse", 16, 0.45),
            ("break", 4, 0.3), ("lift", 24, 0.9))
#: Roles sitting out a section.  A break drops the kick and snare, so the band
#: coming back reads as an event; the high pad is saved for the lifts.
_SECTION_RESTS = {"verse": ("shimmer",), "break": ("kick", "snare", "shimmer", "lead"),
                  "lift": ()}


def entry_ramp_db(bars_in: int, *, under_db: float, ramp_bars: int) -> float:
    """A part's level ``bars_in`` bars after it came in: ``under_db`` on its first
    bar, rising in a straight line to 0 dB after ``ramp_bars`` bars.  It steps at the
    bar lines: a few dB a bar, under the rest of the band."""
    if ramp_bars <= 0 or bars_in >= ramp_bars:
        return 0.0
    return float(under_db) * (1.0 - max(0, int(bars_in)) / float(ramp_bars))


class EntryRamp:
    """Each part's level as the band comes in, counted from that part's own first bar:
    the drums and bass, then the chords, then the melody, each swell in.  What was
    already playing over the intro stays where it was."""

    def __init__(self, *, under_db: float, ramp_bars: int,
                 last_join_bars: int = MELODY_DELAY_BARS):
        self._under, self._bars = float(under_db), int(ramp_bars)
        self._last_join = int(last_join_bars)
        self._joined: dict[str, int] = {}

    def levels(self, bars_in: int, roles, *, intro=()) -> dict[str, float]:
        """Only the entry swells.  A part first heard after the last one the entry
        brings in (the outro's crash, a lift's high pad) is not entering the band, so
        it plays at full."""
        out = {}
        for r in roles:
            if r in intro:
                out[r] = 1.0
                continue
            first = self._joined.setdefault(r, int(bars_in))
            if first > self._last_join:
                out[r] = 1.0
                continue
            db = entry_ramp_db(int(bars_in) - first, under_db=self._under,
                               ramp_bars=self._bars)
            out[r] = 10.0 ** (db / 20.0)
        return out


def follow_band_settled(bars_heard_in: int, *, trusted: bool, ramp_bars: int,
                        last_join_bars: int = MELODY_DELAY_BARS) -> bool:
    """May the follow learn her level from a bar heard ``bars_heard_in`` bars after the
    band came in?  A room mic hears the band as well as her, and while the band is
    still swelling in it would learn a quiet normal, then read her as louder once the
    whole band lands and push the band up.  So it waits for every part of the entry,
    the melody last, to reach full."""
    return trusted or int(bars_heard_in) >= int(last_join_bars) + int(ramp_bars)


class PhraseHold:
    """The band's density, changed only at the top of a phrase or a new section.

    Density picks the hat grid and the bass figure.  Following her moves it a little
    every bar, and over a threshold that rewrote the groove for one bar mid-phrase,
    which she heard as the band changing tempo."""

    def __init__(self, bars: int = 4):
        self._bars = max(1, int(bars))
        self._held: float | None = None
        self._section: str | None = None
        self._recent: list[float] = []

    def value(self, bars_in: int, section: str, density: float) -> float:
        """At the top of a phrase, the median of the phrase just played (its own
        ``bars`` readings), so one odd bar does not set the next phrase.  A new section
        takes its own density at once."""
        if section != self._section:
            self._recent = []
        if self._held is None or section != self._section:
            self._held, self._section = float(density), section
        elif int(bars_in) % self._bars == 0 and self._recent:
            self._held = float(np.median(self._recent))
        self._recent = [*self._recent, float(density)][-self._bars:]
        return self._held


#: Each section's own density in the arc (its first time through).
_ARC_DENSITY = {name: d for name, _, d in reversed(SONG_ARC)}
#: Parts that sit out as she plays softer than her normal: the first group when the
#: band has come down to meet her, the second when it has come well down.  The kick,
#: snare and bass never do: the groove stays, lighter.  Names a style has no part for
#: are ignored.
FOLLOW_THIN = (("shimmer", "lead", "arp", "openhat"), ("clap", "keys", "crackle"))
#: How far the band's level has leaned with her, in dB of its follow gain, before the
#: song's shape gives way: softer, much softer, louder.  Her LEVEL only: fewer notes at
#: full level is not playing softer, and only thins the band's own patterns (the
#: follow's density).  A lean releases only half way back to neutral, so one hovering
#: at an edge does not flip the song.
YIELD_SOFTER_DB = -2.0
YIELD_MUCH_SOFTER_DB = -4.5
YIELD_LOUDER_DB = 1.5
#: The kit, played lighter while the band has come down to meet her: dB on each drum
#: part, softer and much softer.  The groove keeps its shape, and her guitar is not
#: fighting the loudest thing in the band.
DRUM_ROLES = ("kick", "snare", "hat", "openhat", "clap", "foot", "tick")
FOLLOW_DRUM_DB = (-2.0, -4.0)
_LEANS = {-2: "much softer", -1: "softer", 0: "neutral", 1: "louder"}


@dataclass(frozen=True)
class YieldState:
    section: str
    density: float
    resting: tuple[str, ...]
    lean: str
    reason: str
    drums: float = 1.0           # linear level on each of :data:`DRUM_ROLES`


class PlanYield:
    """The song's shape gives way to her.

    The arc (:data:`SONG_ARC`) is a default, not a cage.  While the band's level has
    come down to meet her (:class:`BandFollow`), a lift is played as a verse and the
    lighter parts sit out; further down, more of them; while it has come up with her,
    a verse or a break is played as a lift.  It reads the follow's own level, which
    already waited out a single odd bar, so one quiet bar changes nothing here either.

    A section or a part changes only on a phrase line (every ``phrase_bars`` from the
    band's entry): a change inside a phrase was heard as the band changing tempo (see
    :class:`PhraseHold`).  It changes for a lean held through the whole phrase before
    the line, and not for one that flipped way at the line before as well: a player
    answering herself phrase by phrase, soft then loud, is the song's own shape, and a
    band following each phrase a phrase late would be full on her soft phrases and
    thin on her loud ones.  The drums' level follows her lean on any bar."""

    def __init__(self, phrase_bars: int = 4):
        self._phrase = max(1, int(phrase_bars))
        self._lean = 0
        self._applied: int | None = None    # the lean the section and parts follow
        self._heard: list[int] = []         # her lean on each bar since the last line
        self._changed = False               # the lean flipped way at the last line

    def _read(self, gain_db: float) -> int:
        def past(edge, scale=1.0):
            return gain_db <= edge * scale if edge < 0 else gain_db >= edge * scale
        raw = (-2 if past(YIELD_MUCH_SOFTER_DB) else -1 if past(YIELD_SOFTER_DB)
               else 1 if past(YIELD_LOUDER_DB) else 0)
        cur = self._lean
        if raw == cur or (raw < cur < 0) or (0 < cur < raw):
            return raw                      # further the same way: at once
        # Back towards neutral (or across it): only once past the release.
        edges = {-2: YIELD_MUCH_SOFTER_DB, -1: YIELD_SOFTER_DB, 1: YIELD_LOUDER_DB}
        if cur != 0 and past(edges[cur], 0.5):
            return cur
        if cur == -2 and raw == 0 and past(YIELD_SOFTER_DB, 0.5):
            return -1
        return raw

    def bar(self, section: str, density: float, resting, *, gain_db: float,
            density_x: float = 1.0, bars_in: int | None = None) -> YieldState:
        """This bar's section, density and resting parts, given the song's own and the
        band's lean.  ``density_x`` is the follow's density, which thins the band's
        patterns elsewhere and never moves the song here.  ``bars_in``: bars since the
        band came in, for the phrase lines; None applies a new lean at once."""
        self._lean = self._read(float(gain_db))
        if self._applied is None or bars_in is None:
            self._applied = self._lean
        elif int(bars_in) % self._phrase == 0:
            # The whole phrase, but for its first bar: her lean takes a bar to show.
            # One way all through, at the mildest it went (softer, then much softer:
            # softer).
            held = self._heard[-self._phrase:][1:] if len(self._heard) >= self._phrase else []
            signs = {(x > 0) - (x < 0) for x in held}
            whole = bool(held) and len(signs) == 1
            new = (max(held) if held and held[0] < 0 else min(held)) if whole else None
            sign = (lambda x: (x > 0) - (x < 0))
            flip = whole and sign(new) != sign(self._applied)
            if whole and new != self._applied and not (flip and self._changed):
                self._applied, self._changed = new, flip
            else:
                self._changed = False
            self._heard = []
        if bars_in is not None:
            self._heard.append(self._lean)
        lean, shape = self._lean, self._applied
        resting = tuple(resting)
        played, notes = section, []
        drums = (10.0 ** (FOLLOW_DRUM_DB[0 if lean == -1 else 1] / 20.0)
                 if lean < 0 else 1.0)
        if shape < 0:
            if section == "lift":
                played = "verse"
            thin = FOLLOW_THIN[0] + (FOLLOW_THIN[1] if shape <= -2 else ())
            if played != section:
                density, resting = _ARC_DENSITY[played], _SECTION_RESTS[played]
            resting = (*resting, *(r for r in thin if r not in resting))
            notes.append("lighter parts resting")
        elif shape > 0 and section in ("verse", "break"):
            played = "lift"
            density, resting = _ARC_DENSITY[played], _SECTION_RESTS[played]
        if played != section:
            notes.insert(0, f"{section} played as {played}")
        if drums < 1.0:
            notes.append("drums lighter")
        if shape != lean:
            notes.append("the song changes at the next phrase line")
        reason = (f"yield: {_LEANS[lean]}" + (f", {', '.join(notes)}" if notes else "")
                  if lean or shape else "yield: the song's own shape")
        return YieldState(played, float(density), resting, _LEANS[lean], reason,
                          round(drums, 4))


def yield_to(plan_yield: "PlanYield", follow: "BandFollow", section: str, density: float,
             resting, *, bars_in: int) -> YieldState:
    """The planner's yield for one bar.  On an input that hears only her, the song
    yields to her settled level (:attr:`BandFollow.yield_db`).  On a room mic it never
    does, since fewer parts there would read as her softer; a mic found to hold the band
    partway through hands back whatever had yielded on the next phrase line."""
    gain = follow.yield_db if follow.trusted else 0.0
    return plan_yield.bar(section, density, resting, gain_db=gain,
                          density_x=follow.density, bars_in=bars_in)


def song_section(bars_in: int) -> tuple[str, float, tuple[str, ...]]:
    """(section name, density, roles resting) ``bars_in`` bars after the band came
    in.  The arc repeats, so a long take keeps moving."""
    total = sum(bars for _, bars, _ in SONG_ARC)
    at = max(0, int(bars_in)) % total
    for name, bars, density in SONG_ARC:
        if at < bars:
            return name, density, _SECTION_RESTS[name]
        at -= bars
    raise AssertionError("unreachable: at is reduced modulo the arc's length")


# ------------------------------------------------------------ following the player
#
# Juniper, 2026-11-06: "the band should follow the player but just with the confidence
# waiting thing so that if i just make like a little mistake it doesnt assume thats the
# whole direction we're going."  Before this the band's level never moved and its
# density was the arc's, by bar count, whatever she played.

#: The band follows.  "lead", where plume steers the song itself, is on the roadmap;
#: it would be another entry here, with the arc at full depth instead of following.
FOLLOW = "follow"
FOLLOW_MODES = (FOLLOW,)

#: How long a new way of playing has to hold before the band answers.  Juniper,
#: 2026-11-06, on a take at 60 bpm: "it did not get quieter when i was playing quieter
#: tho."  Three whole bars, read two bars late and answered a decibel a bar, was
#: sixteen seconds after she went soft: not following.
#:
#: Her level is read every half bar, and a clear change is answered sooner than a
#: slight one.  A clear change, :data:`FOLLOW_CLEAR_DB` from what the band follows
#: or her coming back to her normal, holds a bar and a half: three readings, where one
#: odd bar is two, so it can never be a move, and one odd hit sits in a single half.
#: At a fast tempo a bar and a half goes by too quickly to be a choice, so it is never
#: under :data:`FOLLOW_HOLD_MIN_S`, and never over two bars.
#:
#: A slight change, past the deadband but not clear, holds three bars as it always
#: did, even at a slow tempo: her own level wanders by that much from bar to bar (5-7
#: dB on that take, at 60 bpm), and two bars of it moved the band twice where she
#: had not changed.  A band answering every wander is not following either.
FOLLOW_HOLD_BARS = 1.5
FOLLOW_HOLD_MIN_S = 3.0
FOLLOW_HOLD_MAX_BARS = 2.0
FOLLOW_STEADY_HOLD_BARS = 3.0
FOLLOW_CLEAR_DB = 8.0
#: A room mic hears the band as well as the player, so it is less sure what it heard.
FOLLOW_HOLD_BARS_ROOM = 2.5
FOLLOW_HOLD_MIN_S_ROOM = 6.0
FOLLOW_HOLD_MAX_BARS_ROOM = 4.0
FOLLOW_STEADY_HOLD_BARS_ROOM = 4.0
#: Played bars it listens to before it knows how this player normally plays.
FOLLOW_LEARN_BARS = 4
#: How far a bar has to be from what the band already follows to count as different.
#: 3 dB is mind.py's energy_up_db; half an octave of busyness is about 40% more or
#: fewer notes.  Inside these a bar is more of the same.
FOLLOW_LEVEL_DEADBAND_DB = 3.0
FOLLOW_BUSY_DEADBAND = 0.5
#: Notes per beat added to both sides of the busyness ratio, so a sparse player's
#: second note is not "twice as busy".  mind.py's phrase floor.
_BUSY_FLOOR = 0.25
#: The band moves half as far as she did: it answers rather than mirrors (the idea
#: behind mind.py's delayed recovery), and comes down to find a soft player without
#: vanishing under her.  Past :data:`FOLLOW_GAIN_BIG_DB` of drop it answers with
#: :data:`FOLLOW_GAIN_PER_DB_BIG` of each further decibel: she went 14 dB down on
#: the 2026-11-06 take, and half of that, capped at 6, was too small to hear as an
#: answer under her.
FOLLOW_GAIN_PER_DB = 0.5
FOLLOW_GAIN_BIG_DB = 6.0
FOLLOW_GAIN_PER_DB_BIG = 0.75
#: Up is bounded tighter than down: a band that lifts with her must not bury her.
FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_MAX_DB = -10.0, 3.0
#: Density, as a multiple of the song's own.  Softer playing thins the band as well as
#: quietening it; busier playing fills it in.
FOLLOW_DENSITY_PER_DB = 0.04
FOLLOW_DENSITY_PER_OCTAVE = 0.25
FOLLOW_DENSITY_MIN, FOLLOW_DENSITY_MAX = 0.6, 1.3
#: A room mic never moves the band as far.
_ROOM_GAIN_DB = (-4.0, 2.0)
_ROOM_DENSITY = (0.75, 1.15)
#: The most it moves in one bar, so an answer is a glide.  The engine glides a gain
#: change across the bar as well, so even this is never heard as a step.  A decibel a
#: bar was too slow to hear at all: at 60 bpm the answer to a soft passage took
#: longer than the passage.  Going back towards her normal it moves faster still, so
#: when she comes back up the band is not still on its way down.
FOLLOW_GAIN_STEP_DB = 3.0
FOLLOW_GAIN_RETURN_STEP_DB = 4.0
FOLLOW_DENSITY_STEP = 0.1
#: Notes per beat past which more band buries the player: mind.py's density_high.
#: Over it the band never fills in, however busy that is for them.
FOLLOW_TORRENT = 2.5
#: Bars a "busier" or "sparser" cue holds the band's density where the cue left it.
#: After that following moves again, under the cue's own scale, which stays.
FOLLOW_CUE_HOLD_BARS = 16
#: How much of the arc's swing survives while the band follows.  The arc stays for
#: its sections (breaks, the high pad saved for lifts, the moments the take page
#: shows), but its density by bar count alone is the band leading, so following
#: keeps half of it and lets her playing do the rest.
FOLLOW_ARC_DEPTH = 0.5
_ARC_MEAN = sum(bars * d for _, bars, d in SONG_ARC) / sum(bars for _, bars, _ in SONG_ARC)
#: Bars of her recent readings that can re-centre the band's lean, and the share of
#: them that must sit plainly to one side of it (see ``_Patience._drift``).  Four bars,
#: three quarters: a soft passage that comes back near her normal for one bar is
#: still a soft passage, and two odd bars in four are not a move.  After it moves the
#: band it waits for four new bars, so it never re-centres every bar.
FOLLOW_DRIFT_BARS = 4.0
FOLLOW_DRIFT_SHARE = 0.75
#: A second move straight back the other way, each within this many bars of the last
#: (about a phrase, counted from one move to the next),
#: settles on her average over :data:`FOLLOW_DRIFT_BARS` instead (see
#: ``_Patience._patience``): she is swinging, not going anywhere.  One swell and one
#: drop is answered like any drop.
FOLLOW_REVERSAL_BARS = 4.5
#: Bars of her latest playing that, all plainly one way, end a settled swing: longer
#: than any leg of a swing (see FOLLOW_REVERSAL_BARS).
FOLLOW_SWING_RELEASE_BARS = 6.0
#: Her normal is the median of her last :data:`FOLLOW_NORMAL_BARS` bars played at her
#: normal: a bar the band is leaning on (a passage it is following, or a swing) never
#: moves it, so a passage is followed for as long as it lasts and she comes back to the
#: same normal.  Short enough to follow her normal as it drifts slowly over a take (a
#: few dB over a few minutes).  A median, so one odd bar among her first does not set it.
FOLLOW_NORMAL_BARS = 32
#: Bars a lean has to hold, steady, before that level is her normal.  A lean she has
#: held since the band came in, louder than her first bars, is an intro that was simply
#: soft: a section.  Any other (a chorus, a soft passage, a loud intro) only after
#: longer than any passage the band follows.
FOLLOW_ABSORB_LOUDER_BARS = 16
FOLLOW_ABSORB_SOFTER_BARS = 64
#: Bars over which a lean becomes her normal, a step a bar: a phrase.
FOLLOW_ABSORB_RAMP_BARS = 4
#: Bars at her own normal after which a lean is a passage, not her opening: longer
#: than an intro, shorter than a verse.  A soft intro longer than this is a verse
#: (her louder playing after it is a chorus, and only after a long while her normal).
FOLLOW_SETTLED_BARS = 12
#: After a passage becomes her normal, bars back at her old normal that put her normal
#: back at once, so the band never reads her own level as louder or softer than normal;
#: and bars for which the old normal is remembered.
FOLLOW_RECENTRE_BARS = 2
FOLLOW_RECENTRE_MEMORY_BARS = 64


@dataclass(frozen=True)
class FollowState:
    gain_db: float               # added to the band's level
    density: float               # multiplies the band's density
    moved: tuple[str, ...]       # what changed this bar: softer, louder, busier, sparser
    reason: str


def _half_bars_over(seconds: float, bar_s: float | None) -> float:
    return (math.ceil(2.0 * seconds / float(bar_s) - 1e-9) / 2.0
            if bar_s is not None and bar_s > 0.0 else 0.0)


def follow_hold_bars(bar_s: float | None, *, trusted: bool) -> float:
    """Bars a clear change must hold, at bars of ``bar_s`` seconds.

    In half bars, the unit her level is read in: :data:`FOLLOW_HOLD_BARS`, raised to
    cover :data:`FOLLOW_HOLD_MIN_S` and capped at :data:`FOLLOW_HOLD_MAX_BARS` (the
    room mic's own).  No tempo known means the bars alone."""
    hold, min_s, most = ((FOLLOW_HOLD_BARS, FOLLOW_HOLD_MIN_S, FOLLOW_HOLD_MAX_BARS)
                         if trusted else
                         (FOLLOW_HOLD_BARS_ROOM, FOLLOW_HOLD_MIN_S_ROOM,
                          FOLLOW_HOLD_MAX_BARS_ROOM))
    return min(max(hold, _half_bars_over(min_s, bar_s)), most)


class _Patience:
    """One reading's patience: commit only to what has held long enough.

    Each reading is compared with what is already committed, and carries how much of a
    bar it covers.  A reading inside the deadband is more of the same and ends any run,
    and one on the other side starts a new one.  So a mistake moves nothing, and cannot
    be half of a real move either: the same rule the key governor keeps.

    A clear reading (``clear`` or more from what is committed, or back inside the
    deadband of her normal while the band leans) also counts towards a shorter run,
    ``quick``; one that is not clear ends that run.
    """

    def __init__(self, hold: float, deadband: float, *, quick: float | None = None,
                 clear: float | None = None):
        self.hold, self._deadband = float(hold), float(deadband)
        self.quick = self.hold if quick is None else float(quick)
        self._clear = clear
        self.value = 0.0
        self._run: list[float] = []
        self._run_bars = 0.0
        self._fast: list[float] = []
        self._fast_bars = 0.0
        self._side = 0
        #: Her last :data:`FOLLOW_DRIFT_BARS` of readings, ``(reading, bars)``.
        self._recent: list[tuple[float, float]] = []
        #: The way the last move went, and the bars heard since it.
        self._last_side = 0
        self._since = 0.0
        #: Settled on her average after two quick reversals (see FOLLOW_REVERSAL_BARS).
        self._averaged = False
        #: Her last :data:`FOLLOW_SWING_RELEASE_BARS` of readings, for ``_release``.
        self._latest: list[tuple[float, float]] = []
        #: Quick reversals in a row, and whether the last move was a whole one.
        self._reversals = 0
        self._last_big = False

    def _reset(self) -> None:
        self._run, self._run_bars, self._fast, self._fast_bars = [], 0.0, [], 0.0

    def _is_clear(self, reading: float, off: float) -> bool:
        if self._clear is None:
            return False
        # Back towards her normal by more than half the lean is her coming back.
        return (abs(off) >= self._clear
                or (self.value != 0.0 and abs(reading) <= 0.5 * abs(self.value)))

    def heard(self, reading: float, bars: float = 1.0) -> int:
        """-1 or +1 when this reading commits a move that way, else 0."""
        self._recent.append((float(reading), float(bars)))
        while sum(b for _, b in self._recent[1:]) >= FOLLOW_DRIFT_BARS - 1e-9:
            self._recent.pop(0)
        self._latest.append((float(reading), float(bars)))
        while sum(b for _, b in self._latest[1:]) >= FOLLOW_SWING_RELEASE_BARS - 1e-9:
            self._latest.pop(0)
        quick = self._since <= FOLLOW_REVERSAL_BARS + 1e-9
        self._since += float(bars)
        was = self.value
        step = self._release() or self._patience(reading, bars) or self._drift()
        if step:
            # Only a whole move to or from her normal counts as one leg of a swing, not
            # a nudge within a lean.
            big = self._leg(was, self.value)
            reversal = step == -self._last_side and quick and big and self._last_big
            self._reversals = self._reversals + 1 if reversal else 0
            self._last_side, self._since, self._last_big = step, 0.0, big
        return step

    def _leg(self, was: float, now: float) -> bool:
        """A move from her normal to a lean, or back: one leg of a swing."""
        return (abs(now - was) > self._deadband
                and min(abs(was), abs(now)) <= self._deadband)

    def _release(self) -> int:
        """A settled swing ends once her last :data:`FOLLOW_SWING_RELEASE_BARS` all sit
        plainly (half the deadband or more) one way of it: longer than any leg of a
        swing, so she has stopped swinging, and the band follows where she went."""
        if not self._averaged:
            return 0
        got = sum(b for _, b in self._latest)
        if got < FOLLOW_SWING_RELEASE_BARS - 1e-9:
            return 0
        edge = 0.5 * self._deadband
        sides = {(1 if r > self.value else -1) if abs(r - self.value) > edge else 0
                 for r, _ in self._latest}
        if len(sides) != 1 or 0 in sides:
            return 0
        tail = self._latest[-max(1, round(len(self._latest) / 2)):]
        value = sum(r * b for r, b in tail) / sum(b for _, b in tail)
        self.value = 0.0 if abs(value) <= self._deadband else value
        self._averaged = False
        self._recent = list(tail)
        self._reset()
        self._side = 0
        return sides.pop()

    def _drift(self) -> int:
        """Where she has sat over the last :data:`FOLLOW_DRIFT_BARS`, when that is
        plainly not where the band is.

        A real player's level wanders a few dB from bar to bar.  A bar that wanders
        back inside the deadband ends a run, so a soft passage that wanders can keep
        starting its patience over and never commit, or commit to its first shallow
        bars and stay there.  Here, when most of the recent bars (by
        :data:`FOLLOW_DRIFT_SHARE`) sit on one side of what is committed and their median
        is more than half the deadband from it, that median becomes the lean."""
        total = sum(b for _, b in self._recent)
        if total < FOLLOW_DRIFT_BARS - 1e-9 or self._averaged:
            return 0                    # a settled swing ends only by _release
        edge = 0.5 * self._deadband
        ordered = sorted(self._recent)
        acc, lower, upper = 0.0, None, None
        for r, b in ordered:
            acc += b
            if lower is None and acc >= 0.5 * total - 1e-9:
                lower = r
            if acc > 0.5 * total + 1e-9:
                upper = r
                break
        median = 0.5 * (lower + (lower if upper is None else upper))
        off = median - self.value
        if abs(off) <= edge:
            return 0
        side = 1 if off > 0 else -1
        past = sum(b for r, b in self._recent if side * (r - self.value) > 0.0)
        if past < FOLLOW_DRIFT_SHARE * total - 1e-9:
            return 0
        value = 0.0 if abs(median) <= self._deadband else median
        if value == self.value:
            return 0
        self.value = value
        self._reset()
        self._side = 0
        self._recent = []
        self._averaged = False
        return side

    def _patience(self, reading: float, bars: float) -> int:
        if self._averaged:
            return 0
        was = self.value
        off = reading - self.value
        side = 0 if abs(off) <= self._deadband else (1 if off > 0 else -1)
        if side == 0:
            self._reset()
            self._side = 0
            return 0
        if side != self._side:
            self._reset()
            self._side = side
        self._run.append(reading)
        self._run_bars += float(bars)
        if self._is_clear(reading, off):
            self._fast.append(reading)
            self._fast_bars += float(bars)
        else:
            self._fast, self._fast_bars = [], 0.0
        if self._fast and self._fast_bars >= self.quick - 1e-9:
            held, held_bars = self._fast, self._fast_bars
        elif self._run_bars >= self.hold - 1e-9:
            held, held_bars = self._run, self._run_bars
        else:
            return 0
        # Where she is now: the run's last ``quick`` worth, not where it began.
        held = held[-max(1, round(self.quick * len(held) / held_bars)):]
        self.value = sum(held) / len(held)
        if (side == -self._last_side and self._since <= FOLLOW_REVERSAL_BARS + 1e-9
                and self._reversals >= 1 and self._last_big and self._leg(was, self.value)):
            # Back the other way within a few bars, for the second time running: two
            # soft bars and two loud ones, say.  Chasing each would put the band, two
            # bars behind, at its fullest on her soft bars.  It settles on where she
            # has been instead.
            total = sum(b for _, b in self._recent)
            self.value = sum(r * b for r, b in self._recent) / total
            # From here a run of a phrase or less is more of the same back and forth;
            # only her last FOLLOW_SWING_RELEASE_BARS all one way (``_release``) end it.
            self._averaged = True
        if abs(self.value) <= self._deadband:
            self.value = 0.0          # back inside normal is normal, not a lean
        self._reset()
        self._side = 0
        return side

    @property
    def settled(self) -> bool:
        """Not leaning, and nothing building towards a lean."""
        return self.value == 0.0 and not self._run


class BandFollow:
    """The band's level and density, following how she plays.

    Feed it each bar's level (dB) and busyness (onsets per beat), and whether she
    played.  Both are read against her own normal, learned from her first played bars:
    nobody is soft in absolute terms.  A bar she sat out is no evidence of anything, so
    it neither moves the band nor breaks a run.

    ``parts``, when given, are her level over each half bar heard since the last call
    (None for a half she sat out), and the level is followed from those instead of the
    whole bar: the planner reads a half bar as soon as it has been played, so the band
    finds her sooner.  ``bar_s`` is the bar's length, which the hold is counted against
    (see :func:`follow_hold_bars`).  The bar's own level still teaches her normal.

    ``trusted`` is False on a room mic, where the band is in the signal.  Then it
    ignores busyness (the drums are onsets too), waits longer, moves less far, learns
    her normal only once the band is in, and takes its own lean back off what it hears,
    so a band that lifted cannot read as a player who lifted.
    """

    def __init__(self, *, trusted: bool, mode: str = FOLLOW):
        if mode not in FOLLOW_MODES:
            raise ValueError(f"band mode {mode!r}: only {FOLLOW!r} exists, and plume "
                             f"leading is on the roadmap")
        self.mode = mode
        self.trusted = bool(trusted)
        steady = FOLLOW_STEADY_HOLD_BARS if self.trusted else FOLLOW_STEADY_HOLD_BARS_ROOM
        self._level = _Patience(steady, FOLLOW_LEVEL_DEADBAND_DB,
                                quick=follow_hold_bars(None, trusted=self.trusted),
                                clear=FOLLOW_CLEAR_DB)
        self._busy = _Patience(steady, FOLLOW_BUSY_DEADBAND)
        self._gain_lim = ((FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_MAX_DB) if self.trusted
                          else _ROOM_GAIN_DB)
        self._dens_lim = ((FOLLOW_DENSITY_MIN, FOLLOW_DENSITY_MAX) if self.trusted
                          else _ROOM_DENSITY)
        self._normal_db: float | None = None
        self._normal_busy: float | None = None
        self._learned = 0
        self._torrent = False
        self.gain_db = 0.0
        self.density = 1.0
        self._cue_hold = 0
        #: Her played bars' levels and busyness, oldest first (see FOLLOW_NORMAL_BARS).
        self._levels: list[float] = []
        self._busies: list[float] = []
        self._held = 0                # bars she has held the band's lean (see _absorb)
        self._ramp: tuple[int, float, float] | None = None   # steps left, step, target
        self._at_normal = 0           # bars in a row at her normal, once it is known
        self._settled = False         # she has played FOLLOW_SETTLED_BARS at her normal
        self._prior: float | None = None   # her normal before a lean became it
        self._prior_age = 0
        self._prior_near = 0

    @property
    def hold_bars(self) -> float:
        """Bars a clear change holds before the band answers, at this tempo."""
        return self._level.quick

    def shape(self, arc_density: float) -> float:
        """The song arc's density, keeping only part of its swing while following."""
        return _ARC_MEAN + FOLLOW_ARC_DEPTH * (float(arc_density) - _ARC_MEAN)

    def _learn(self, db: float, busy: float | None) -> None:
        """One more played bar into her normal (see :data:`FOLLOW_NORMAL_BARS`): only a
        bar at her normal, once it is known."""
        known = self._learned >= FOLLOW_LEARN_BARS
        level_at_normal = self._level.value == 0.0 and not self._level._averaged
        if not known or level_at_normal:
            self._levels.append(float(db))
        if busy is not None and (not known or self._busy.value == 0.0):
            self._busies.append(float(busy))
        self._learned += 1
        del self._levels[:-FOLLOW_NORMAL_BARS]
        del self._busies[:-FOLLOW_NORMAL_BARS]
        self._normal_db = float(np.median(self._levels))
        self._normal_busy = float(np.median(self._busies)) if self._busies else None

    def bar(self, rms_db: float | None, onsets_per_beat: float | None, *,
            playing: bool, band_in: bool = True, parts=None,
            bar_s: float | None = None, settled: bool = True) -> FollowState:
        """This bar's lean.  ``band_in`` False: drums alone, so only listen.
        ``settled`` False: the band is in but still swelling in (see
        follow_band_settled), so it listens without learning or moving."""
        busy = None if onsets_per_beat is None or not self.trusted else float(onsets_per_beat)
        if bar_s is not None:
            self._level.quick = follow_hold_bars(bar_s, trusted=self.trusted)
        if not band_in:
            # A room mic over drums alone hears another mix than it will with the band
            # in, so only a trusted input learns from these bars.
            if self.trusted and playing and rms_db is not None:
                self._learn(float(rms_db), busy)
            return FollowState(0.0, 1.0, (), "follow: listening until the band is in")
        if not settled:
            # The same as drums alone for a room mic: it hears a mix that is still
            # changing.  A trusted input still learns her.
            if self.trusted and playing and rms_db is not None:
                self._learn(float(rms_db), busy)
            return FollowState(0.0, 1.0, (),
                               "follow: listening while the band comes in")
        moved: list[str] = []
        gain_was, dens_was = self._targets()
        known = self._learned >= FOLLOW_LEARN_BARS
        # Off a room mic, take back what the band's own lean put into it.
        own = 0.0 if self.trusted else self.gain_db
        if known:
            if parts is not None:
                readings = [(float(p) - own, 0.5) for p in parts if p is not None]
            elif playing and rms_db is not None:
                readings = [(float(rms_db) - own, 1.0)]
            else:
                readings = []
            for db, bars in readings:
                step = self._level.heard(db - self._normal_db, bars)
                if step:
                    moved.append("louder" if step > 0 else "softer")
        if playing and rms_db is not None:
            if known and busy is not None and self._normal_busy is not None:
                self._torrent = busy > FOLLOW_TORRENT
                ratio = math.log2((busy + _BUSY_FLOOR) / (self._normal_busy + _BUSY_FLOOR))
                step = self._busy.heard(ratio)
                if step:
                    moved.append("busier" if step > 0 else "sparser")
            self._learn(float(rms_db) - own, busy)
            if known:
                self._absorb(float(rms_db) - own, bool(moved))
        gain_t, dens_t = self._targets()
        # A move is reported only when the band's own target went that way: at a limit,
        # or thinned by the torrent cap on a "busier" bar, the log and the after-take
        # recap would otherwise claim a change nobody heard.  Her level drives density
        # too, so a softer bar at the gain cap still counts when the band thins.
        up = {"louder": gain_t > gain_was or dens_t > dens_was,
              "softer": gain_t < gain_was or dens_t < dens_was,
              "busier": dens_t > dens_was, "sparser": dens_t < dens_was}
        moved = [m for m in moved if up[m]]
        back = abs(gain_t) < abs(self.gain_db)
        self.gain_db = _toward(self.gain_db, gain_t, FOLLOW_GAIN_RETURN_STEP_DB if back
                               else FOLLOW_GAIN_STEP_DB)
        if self._cue_hold > 0:
            self._cue_hold -= 1
        else:
            self.density = _toward(self.density, dens_t, FOLLOW_DENSITY_STEP)
        return self._state(tuple(moved))

    def _reseed(self, db: float) -> None:
        self._levels = [float(db)] * FOLLOW_LEARN_BARS
        self._normal_db = float(db)
        lvl = self._level
        lvl.value, lvl._averaged, lvl._recent, lvl._latest = 0.0, False, [], []
        lvl._reset()
        lvl._side = 0
        self._held = 0

    def _absorb(self, db: float, moved: bool) -> None:
        """A lean held long enough becomes her normal (:data:`FOLLOW_ABSORB_LOUDER_BARS`,
        :data:`FOLLOW_ABSORB_SOFTER_BARS`); and back at the normal it replaced, that is
        her normal again at once (:data:`FOLLOW_RECENTRE_BARS`).  Becoming her normal
        takes :data:`FOLLOW_ABSORB_RAMP_BARS`, a step a bar, so the band eases back to
        its own level instead of jumping there."""
        if self._ramp:
            left, step, target = self._ramp
            if left <= 1:
                self._ramp = None
                self._reseed(target)
            else:
                self._ramp = (left - 1, step, target)
                self._shift(step)
            return
        if self._prior is not None:
            self._prior_age += 1
            near = abs(db - self._prior) <= FOLLOW_LEVEL_DEADBAND_DB
            self._prior_near = self._prior_near + 1 if near else 0
            if self._prior_near >= FOLLOW_RECENTRE_BARS:
                self._reseed(self._prior)
                self._prior = None
                return
            if self._prior_age > FOLLOW_RECENTRE_MEMORY_BARS:
                self._prior = None
        lean = self._level.value
        if lean == 0.0 and not self._level._averaged:
            self._at_normal += 1
            self._settled = self._settled or self._at_normal >= FOLLOW_SETTLED_BARS
        else:
            self._at_normal = 0
        if lean == 0.0 or self._level._averaged or moved:
            self._held = 0
            return
        if abs(db - self._normal_db - lean) <= FOLLOW_LEVEL_DEADBAND_DB:
            self._held += 1
        opening = lean > 0 and not self._settled
        limit = FOLLOW_ABSORB_LOUDER_BARS if opening else FOLLOW_ABSORB_SOFTER_BARS
        if self._held >= limit:
            # An opening that was off is forgotten; a passage's old normal is kept, to
            # come back to.
            self._prior = None if opening else self._normal_db
            self._prior_age, self._prior_near = 0, 0
            step = lean / FOLLOW_ABSORB_RAMP_BARS
            self._ramp = (FOLLOW_ABSORB_RAMP_BARS - 1, step, self._normal_db + lean)
            self._shift(step)

    def _shift(self, step: float) -> None:
        """Her normal ``step`` dB towards a lean, and the lean that much less."""
        self._normal_db += step
        self._levels = [self._normal_db] * FOLLOW_LEARN_BARS
        lvl = self._level
        lvl.value -= step
        lvl._recent, lvl._latest = [], []
        lvl._reset()
        lvl._side = 0

    @property
    def yield_db(self) -> float:
        """The band's level lean as the song's shape should read it: none while she is
        swinging between soft and loud (the follow has settled on her average), whose
        dip is the swing's and not a quieter player's."""
        return 0.0 if self._level._averaged else self.gain_db

    def _targets(self) -> tuple[float, float]:
        lvl, busy = self._level.value, self._busy.value
        gain = FOLLOW_GAIN_PER_DB * lvl
        if lvl < -FOLLOW_GAIN_BIG_DB:
            gain = -(FOLLOW_GAIN_PER_DB * FOLLOW_GAIN_BIG_DB
                     + FOLLOW_GAIN_PER_DB_BIG * (-lvl - FOLLOW_GAIN_BIG_DB))
        gain = _clip(gain, *self._gain_lim)
        dens = 1.0 + FOLLOW_DENSITY_PER_DB * lvl + FOLLOW_DENSITY_PER_OCTAVE * busy
        hi = min(1.0, self._dens_lim[1]) if self._torrent else self._dens_lim[1]
        return gain, _clip(dens, self._dens_lim[0], hi)

    def cue(self, value: str) -> FollowState:
        """A "busier" or "sparser" cue is the player's own word, and beats following.

        Any lean against it goes at once, and the density holds there for
        :data:`FOLLOW_CUE_HOLD_BARS`.  A lean the same way as the cue stays."""
        if value not in ("busier", "sparser"):
            return self._state(())
        self.density = (min(self.density, 1.0) if value == "sparser"
                        else max(self.density, 1.0))
        self._cue_hold = FOLLOW_CUE_HOLD_BARS
        return self._state(())

    def _state(self, moved: tuple[str, ...]) -> FollowState:
        why = (f"follow: level {self.gain_db:+.1f}dB density x{self.density:.2f}"
               + (f", {' and '.join(moved)} for {self.hold_bars:g} bars" if moved else ""))
        return FollowState(round(self.gain_db, 3), round(self.density, 4), moved, why)


def swell_bound(heard: float, spared: float, rest: float) -> float:
    """``heard`` unless it is over ``rest``, and then no further over than ``spared``:
    never under ``rest``, never over ``heard``."""
    return heard if heard <= rest else min(heard, max(spared, rest))


class FollowTwice:
    """The band's level and density, following her twice over.

    One :class:`BandFollow` hears everything, as the band always has.  A second hears
    the same with ``spared`` in place of each bar's readings: the bars in which the
    input clipped out of character for her (a knock on the instrument, a crackling
    cable) left out.  The band does what the first says, except that where the first
    has it over its resting level or density, it goes no further over than the second
    (:func:`swell_bound`).

    So on an input that hears only her, leaving a clip out can only take away a swell,
    or a filling out, that hearing everything would have made: on every bar the band's
    level lies between resting and what hearing everything gives, and so does its
    density.  ``yield_db`` is bounded the same way, and is always 0 or ``gain_db``.

    That is said for a direct input only.  A room mic's follower takes its own lean
    back off what it hears, so a swell held back would be taken off bars the band never
    played it in: on an untrusted input there is no second follower, nothing is held,
    and this is :class:`BandFollow` and no more.  The same on a MIDI take, where no
    clip is ever marked.  Nothing here is in the audio callback: it is the planner's, a
    bar at a time."""

    def __init__(self, *, trusted: bool, mode: str = FOLLOW):
        self.heard = BandFollow(trusted=trusted, mode=mode)
        #: The second follower: the first's twin from the first bar anything is left
        #: out of.  Until then the two have heard the same, and there is only one.
        self.spared: BandFollow | None = None
        self.mode, self.trusted = self.heard.mode, self.heard.trusted
        self.gain_db, self.density = 0.0, 1.0

    hold_bars = property(lambda self: self.heard.hold_bars)

    def shape(self, arc_density: float) -> float:
        return self.heard.shape(arc_density)

    @property
    def held_back(self) -> tuple[float, float] | None:
        """(dB, density) of a swell hearing everything would have made and the band
        is not making; None when the band is doing just what hearing everything says."""
        if self.gain_db == self.heard.gain_db and self.density == self.heard.density:
            return None
        return (round(self.heard.gain_db - self.gain_db, 3),
                round(self.heard.density - self.density, 4))

    @property
    def yield_db(self) -> float:
        other = self.spared or self.heard
        return swell_bound(self.heard.yield_db, other.yield_db, 0.0)

    def _settle(self) -> None:
        other = self.spared or self.heard
        self.gain_db = swell_bound(self.heard.gain_db, other.gain_db, 0.0)
        self.density = swell_bound(self.heard.density, other.density, 1.0)

    def distrust(self) -> None:
        """The input turns out to hold the band: from here a room mic's follower, new,
        as the take would have made it, and no second one.  It never goes back."""
        self.heard = BandFollow(trusted=False, mode=self.mode)
        self.spared, self.trusted = None, False
        self._settle()

    def cue(self, value: str) -> FollowState:
        state = self.heard.cue(value)
        if self.spared is not None:
            self.spared.cue(value)
        self._settle()
        return self._state(state, state.moved)

    def bar(self, rms_db: float | None, onsets_per_beat: float | None, *,
            parts=None, spared=None, **kw) -> FollowState:
        """This bar's lean.  As :meth:`BandFollow.bar`, with ``spared``: the same
        ``(rms_db, onsets_per_beat, parts)`` with the clipped bars left out, or None
        when nothing was."""
        was = (self.gain_db, self.density, self.held_back)
        whole = (rms_db, onsets_per_beat, parts)
        db, busy, sparts = whole if spared is None or not self.trusted else spared
        if self.spared is None and (db, busy, sparts) != whole:
            self.spared = copy.deepcopy(self.heard)
        state = self.heard.bar(rms_db, onsets_per_beat, parts=parts, **kw)
        other = state if self.spared is None else self.spared.bar(db, busy, parts=sparts, **kw)
        self._settle()
        if was[2] is None and self.held_back is None:
            return self._state(state, state.moved)
        # While a swell is held back, a move is one the band made: either reading
        # called it, and the level or the density went that way.
        up = {"louder": self.gain_db > was[0] or self.density > was[1],
              "softer": self.gain_db < was[0] or self.density < was[1],
              "busier": self.density > was[1], "sparser": self.density < was[1]}
        called = state.moved + tuple(m for m in other.moved if m not in state.moved)
        return self._state(state, tuple(m for m in called if up[m]))

    def _state(self, state: FollowState, moved: tuple[str, ...]) -> FollowState:
        held = self.held_back
        if held is None and moved == state.moved:
            return state
        why = (f"follow: level {self.gain_db:+.1f}dB density x{self.density:.2f}"
               + (f", {' and '.join(moved)} for {self.hold_bars:g} bars" if moved else ""))
        if held is not None:
            why += (f"; a swell of {held[0]:+.1f}dB x{self.heard.density:.2f} held back:"
                    f" the input clipped as she had not been playing")
        return FollowState(round(self.gain_db, 3), round(self.density, 4), moved, why)


def _toward(now: float, target: float, step: float) -> float:
    return now + max(-step, min(step, target - now))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


OUTRO_SIGNAL_BARS = 4
OUTRO_FADE_BARS = 6


def outro(seconds_in: float, wind_down_s: float | None, bar_s: float):
    """Where a timed song is in its ending, or None before it.

    (bars into the outro, gain offset in dB, whether the take ends).  It signals
    first, with a crash and the drums thinning out, so the player can wind down
    with the band; only then does it fade.
    """
    if wind_down_s is None or seconds_in < wind_down_s:
        return None
    bars = int((seconds_in - wind_down_s) / bar_s)
    fading = max(0, bars - OUTRO_SIGNAL_BARS + 1)
    gain = -36.0 * min(fading, OUTRO_FADE_BARS) / OUTRO_FADE_BARS
    return bars, gain, fading >= OUTRO_FADE_BARS


#: How long after the player stops a direct or MIDI take should be over.
END_AFTER_S = 20.0
#: Roughly how long a fade should take, whatever the tempo.  In whole bars, so this
#: is only ever as close as the bar lets it be.
_FADE_TARGET_S = 6.0
_FADE_MIN_BARS, _FADE_MAX_BARS = 2, 6
#: What reading costs before anything can react.  Bars are read two bars late and a
#: player stops somewhere inside one, so a stop is noticed one to two bars after it
#: happens.  Waiting in whole bars can add up to one more, so a take lands a little over.
_READ_LAG_BARS = 1.5
#: A rest shorter than this never dips the band, at any tempo, however late that makes the
#: ending at a slow one.  Bars are read late, so a dip for a rest would land after she is back.
MIN_REST_S = 6.0


class SilenceEnding:
    """Nobody playing: wait, then fade out and end (Juniper, 2026-11-02).

    Times are bar starts in stream seconds.  A bar of playing during the fade
    brings the band straight back, so a rest costs at most a short dip.  Bars are read
    two bars late, so the take stays open ``listen_bars`` past the fade: a return
    in the last bars of the fade is still heard.

    ``end_after_s``: for an input that really can tell the player stopped (direct,
    MIDI), size the wait and fade so the take ends about that long after they stop,
    given ``bar_s`` on each update.  The wait never drops below :data:`MIN_REST_S` or two
    bars, and reading and fading take six more, so below about 90 bpm it ends later than
    asked: that is the price of a phrase rest not ducking a slow song.  Until they have
    played a bar it keeps ``silence_s`` and ``fade_bars``: they have only just knocked, and
    their hands are still on the way back to the strings.  A room mic leaves it unset
    and keeps those numbers throughout, because it cannot hear a player stop over the
    band.
    """

    def __init__(self, *, silence_s: float = 30.0, fade_bars: int = 4,
                 floor_db: float = -36.0, listen_bars: int = 2,
                 end_after_s: float | None = None):
        self._silence_s = float(silence_s)
        self._fade_bars = int(fade_bars)
        self._floor_db = float(floor_db)
        self._listen_bars = int(listen_bars)
        self._end_after_s = None if end_after_s is None else float(end_after_s)
        self._last_heard: float | None = None
        self._played = False
        self._fade = 0
        self._bars_of_fade = self._fade_bars
        #: A fade that will run to its end whatever is heard (see long_wait).
        self._finish = False

    def long_wait(self) -> None:
        """From here it cannot tell the player stopped after all, so it waits as a room
        mic's ending does.  A fade already running was begun on an input that could
        tell, so it is kept and runs to its end: the band is in the mic now, the mic
        will not go silent again, and "not silent" must not bring the band back."""
        self._end_after_s = None
        self._finish = self._fade > 0

    def _timing(self, bar_s: float | None) -> tuple[float, int]:
        """(seconds to wait, bars to fade over)."""
        if self._end_after_s is None or bar_s is None or not self._played:
            return self._silence_s, self._fade_bars
        fade = max(_FADE_MIN_BARS, min(_FADE_MAX_BARS, round(_FADE_TARGET_S / bar_s)))
        spent = (fade + self._listen_bars + _READ_LAG_BARS) * bar_s
        return max(self._end_after_s - spent, MIN_REST_S, 2.0 * bar_s), fade

    def update(self, t_bar: float, playing: bool,
               bar_s: float | None = None) -> tuple[float, bool]:
        """(gain offset in dB for this bar, whether the take should end)."""
        playing = bool(playing) and not self._finish
        self._played = self._played or bool(playing)
        if playing or self._last_heard is None:
            self._last_heard = float(t_bar)
            self._fade = 0
            if playing:
                return 0.0, False
        if self._fade == 0:           # once it is fading, a tempo cue cannot reshape it
            wait, self._bars_of_fade = self._timing(bar_s)
            # 1e-6: a wait of one bar must not lose to rounding in the clock's sums.
            if t_bar - self._last_heard < wait - 1e-6:
                return 0.0, False
        self._fade += 1
        gain = self._floor_db * min(self._fade, self._bars_of_fade) / self._bars_of_fade
        return gain, self._fade >= self._bars_of_fade + self._listen_bars


def count_start(taps, soft_hits, *, tol_frac: float = 0.15,
                max_back: int = 12) -> list[float]:
    """The whole count, including knocks too quiet to have confirmed it.

    A count is confirmed by its four clearest knocks, which need not be its
    first four: on 2026-11-02 the first three of a count were quiet, the band
    counted from the fourth, and the drums came in three beats late.  So walk
    back along the confirmed spacing through quieter hits, until a beat is
    missing.  The first knock found is where she started counting.
    """
    knocks = sorted(float(t) for t in taps)
    gaps = sorted(b - a for a, b in zip(knocks, knocks[1:]))
    period = gaps[len(gaps) // 2]
    soft = sorted(float(t) for t in soft_hits)
    anchor = knocks[0]
    for beats_back in range(1, max_back + 1):
        # Measured from the confirmed knock each time: stepping from each soft hit
        # would let stray room noise walk the grid away.
        want = anchor - beats_back * period
        near = [t for t in soft if abs(t - want) <= tol_frac * period]
        if not near:
            break
        knocks.insert(0, want)     # the grid's time: a hit only proves the beat
    return knocks


def tap_tempo(taps) -> tuple[float, float]:
    """(bpm, next beat time) from a confirmed count-in's tap times.

    Median interval, so one rushed first knock does not set the tempo.
    """
    ts = sorted(float(t) for t in taps)
    if len(ts) < 2:
        raise ValueError(f"tap_tempo needs at least 2 taps, got {len(ts)}")
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    n = len(gaps)
    period = gaps[n // 2] if n % 2 else 0.5 * (gaps[n // 2 - 1] + gaps[n // 2])
    return 60.0 / period, ts[-1] + period


def first_bar_time(next_beat_s: float, bar_s: float, playhead_s: float,
                   comp_s: float, *, margin_s: float = 0.15) -> float:
    """The first downbeat still far enough ahead to place, on the tapped grid.

    Placing early by ``comp_s`` at a fast tempo can land the next beat in the
    past.  Rather than clip it, or start on a beat that puts the "one" somewhere
    else, come in a whole bar later.
    """
    t = float(next_beat_s)
    while t - comp_s < playhead_s + margin_s:
        t += bar_s
    return t
