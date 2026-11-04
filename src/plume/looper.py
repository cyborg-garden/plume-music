"""The smart looper: "make a loop of this", as a typed ``plume cue``.

You ask, you play, and plume does the fiddly part a loop pedal leaves to your
foot: it finds where the phrase really starts and ends, snaps that to the bar,
closes the join so it cannot click, and either fits the loop to the tempo that
is running or, if none is, lets the loop set it.

Everything down to :class:`Looper` is pure.  :class:`Looper` adds only the state
one session needs and takes one lock; it opens no device, starts no thread, and
keeps audio in memory only.  ``realtime.lead`` feeds it.

**What this deliberately does not do.**

- *No pitch-preserving time-stretch.*  A stretch good enough for a sustained
  guitar means a phase-locked vocoder or a library such as Rubber Band, and a
  plain vocoder smears the very attacks a loop lives on.  So a loop is fitted to
  its bars by padding or trimming its end, which cannot move its pitch, and only
  within :data:`MAX_FIT_RATIO`.  Past that it is refused, by name, not bent.
- *No MIDI.*  A loop is audio.  On a ``--midi`` take it is cut from the mic,
  which is still open, and the notes play no part in it.
- *No bleed removal.*  The loop is the raw mic.  Over speakers the band and any
  earlier layers are in that mic and end up in the loop.  Detection cannot save
  this: :class:`SelfMasker <plume.realtime.SelfMasker>` only blanks features, and
  on a real room it takes off a few dB.  Headphones or a direct input.
"""
from __future__ import annotations

import dataclasses
import math
import re
import threading

import numpy as np
from scipy.ndimage import (maximum_filter1d, median_filter, minimum_filter1d,
                           uniform_filter1d)

#: Samples per level reading.  Same hop as the feature extractor.
LEVEL_HOP = 512
#: Below this the capture is digital silence, whatever the fader says.
SILENCE_DB = -80.0
#: A sound must stand this far above its own room before it counts as playing.
PHRASE_CONTRAST_DB = 12.0
#: Sound ends this far under the phrase's peak (or just over the room).
GATE_BELOW_PEAK_DB = 35.0
GATE_ABOVE_FLOOR_DB = 10.0
#: A new note: this much rise over the last few readings, at least this far apart.
ONSET_RISE_DB = 6.0
ONSET_MIN_SEP_S = 0.06
#: Less sound than this in a whole take is a rustle, not a note: a plucked note
#: stays over the gate for far longer, and nothing plays a loop of a rustle.
STRAY_MAX_S = 0.12
#: A sound this far under the take's peak and that short, ahead of the phrase, is
#: hands finding the strings rather than a first note, however soon the phrase
#: follows.  Any louder and it might be the first note, so it stays.
STRAY_UNDER_PEAK_DB = 12.0
#: Quiet this long separates one sound from the next.
STRAY_GAP_S = 0.1
#: A take with no quiet in it at all has no phrase to read, only a level, and
#: below this it is the room.  Anything raised from there is a loud hiss.
NO_CONTRAST_MIN_DB = -45.0
#: How far over the level at the cue a take has to get before its loudness alone can
#: count as playing.  Steady noise is loud enough for that but never rises.
CUE_RISE_DB = 10.0
#: Sound this loud is playing even with no rise: a chord already ringing at the cue.
#: A room gets nowhere near it.
PLAINLY_PLAYING_DB = -25.0

#: At the time limit, how to tell a held note or chord from steady noise, since neither
#: has a rise or a quiet in it: analysis window, shortest hold, the part of the spectrum
#: read (under the low end hum lives in, over the top of what a chord puts out), how far
#: over its neighbours a bin must stand to count as a pitch, and how much of the sound
#: has to be in such bins.  Hiss and rumble put almost nothing there, and a tone 7 dB over
#: noise puts about 80% there.
HELD_WINDOW = 8192
HELD_MIN_S = 2.0
HELD_BAND_HZ = (70.0, 5000.0)
HELD_PEAK_OVER_NEIGHBOURS = 10.0
HELD_NEIGHBOURS = 31
HELD_MIN_SHARE = 0.5

#: The limit passed and nothing counted as playing.  All that is known is what it says:
#: a quiet melody and a noisy room look the same here.
NO_CLEAR_START = ("no clear start above the room, so no loop: next time play louder, or say "
                  "\"that's the loop\" before the limit")

#: A note this close to a bar line (in beats) is ON that line.
ON_THE_LINE_BEATS = 0.25
#: A phrase may start this early (in beats) and still belong to the coming bar.
EARLY_START_BEATS = 0.125

#: The most a loop may miss its bars by and still be fitted.  At 96 bpm, 2% of
#: four bars is 200 ms at the very end, which is about as far as "that was in
#: time" stretches.  Bigger needs a real time-stretch, which plume does not have.
MAX_FIT_RATIO = 0.02
#: Bar counts tried when none is named.  Whole doublings only: allowing 3 and 6
#: reads quarter notes at 72 as dotted quarters at 108, the same 3/2 alias that
#: has already fooled the beat tracker on sparse takes.
TEMPO_BAR_COUNTS = (1, 2, 4, 8, 16)
TEMPO_RANGE_BPM = (45.0, 170.0)        # the same range tap-start accepts
#: Onsets further than this from an eighth-note grid (0 on it, 1 halfway) do not
#: show a tempo.  One note on the grid is no evidence, since the first note is on
#: it by construction and any one other note is near some reading, so a take with
#: few notes has to sit tighter.
TEMPO_MAX_GRID_ERROR = 0.3
#: Fewer notes than this show no tempo unless the bars are named.  Three notes
#: leave one note to judge the fit by, and that is no fit at all.
TEMPO_MIN_NOTES = 4
#: The limit by note count, measured on random gaps of 0.2 to 1.5 s so that about
#: 1 random take in 20 or fewer gets a tempo.  At 15 ms of human timing this refuses about a third
#: of steady 4-note takes and a seventh of 5-note ones.  That is the right way
#: round: a refusal says "name the bars", and a wrong tempo starts the band.
TEMPO_GRID_LIMITS = {4: 0.1, 5: 0.15, 6: 0.2, 7: 0.2, 8: 0.25, 9: 0.25}
#: Where octave-ambiguous takes are pulled to, and how hard.
TEMPO_PRIOR_BPM = 100.0
TEMPO_PRIOR_WEIGHT = 1.0
#: Preferring the closing downbeat, which is real evidence, over a guessed gap.
TEMPO_GUESSED_GAP_PENALTY = 0.35

#: Length of the crossfade that closes the loop (20 ms).
JOIN_SAMPLES = 960
#: Fade on a bare head and on a layer coming in or going out (about 5 ms).
DECLICK_SAMPLES = 256
HEAD_FADE_SAMPLES = 64
#: Every layer is brought to this peak, so a loop's level does not depend on how
#: far you were from the mic.
LOOP_PEAK_DB = -12.0
LOOP_MAX_BOOST_DB = 40.0

#: With no tempo, this long without sound closes the take.  Long enough for the
#: rests in a slow phrase; say "that's the loop" to close sooner.
NO_TEMPO_HOLD_S = 2.5
#: How long a take with only a stray sound or two in it keeps listening.  The cue
#: comes first, then your hands find the guitar, and that makes noise.  Say
#: "that's the loop" to keep one note sooner.
LONE_SOUND_HOLD_S = 8.0
#: One take is never longer than this, and its memory is bounded by it.
MAX_TAKE_S = 64.0
#: The most bars a take could hold, at the fastest tempo plume accepts.
MAX_NAMED_BARS = int(MAX_TAKE_S * TEMPO_RANGE_BPM[1] / 240.0)
#: Digits read from a spoken number.  Past this it is too many bars whatever it says.
_MAX_BAR_DIGITS = 6
MAX_LAYERS = 6
#: How much of the mic to keep from before the cue, for the join.
PRE_ROLL_SAMPLES = JOIN_SAMPLES
#: The self-masker's speaker-to-mic estimate above which the loop has the band in it.
BLEED_WARN_DB = -40.0
#: How often a take is re-read for its phrase (samples).
CHECK_EVERY_SAMPLES = 4800


class LoopRefused(ValueError):
    """The loop cannot honestly be made.  The message says why, for the player."""


def db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


# ------------------------------------------------------------------ auto-trim

@dataclasses.dataclass(frozen=True)
class Phrase:
    """Where the playing is, in samples from the start of the audio."""
    first: int
    last_onset: int
    end: int
    onsets: tuple[int, ...]
    sounding_s: float = math.inf


def level_db(audio: np.ndarray, hop: int = LEVEL_HOP) -> np.ndarray:
    """RMS level per hop, dBFS.  A trailing part-hop is left for next time."""
    n = audio.size // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    blocks = np.asarray(audio[:n * hop], dtype=np.float64).reshape(n, hop)
    power = np.mean(blocks * blocks, axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _onset_hops(above: np.ndarray, levels: np.ndarray, sep: int) -> list[int]:
    onsets = [int(above[0])]
    for i in above[1:]:
        i = int(i)
        if i - onsets[-1] < sep:
            continue
        if levels[i] - float(levels[max(0, i - 3):i].min()) >= ONSET_RISE_DB:
            onsets.append(i)
    return onsets


def _without_lead_strays(above: np.ndarray, levels: np.ndarray, peak: float,
                         sr: int, hop: int) -> np.ndarray:
    """``above`` from the first sound that could be a note."""
    gap = max(1, int(round(STRAY_GAP_S * sr / hop)))
    start = 0
    for s in np.flatnonzero(np.diff(above) >= gap):
        head = above[start:s + 1]
        if (head.size * hop / sr >= STRAY_MAX_S
                or float(levels[head].max()) > peak - STRAY_UNDER_PEAK_DB):
            break
        start = int(s) + 1
    return above[start:]


def _phrase_hops(levels: np.ndarray, sr: int, *,
                 contrast_db: float = PHRASE_CONTRAST_DB,
                 hop: int = LEVEL_HOP,
                 lead_gap_s: float | None = None,
                 lead_notes: int = 2) -> tuple[int, int, list[int], float] | None:
    """(first hop, end hop, onset hops, seconds sounding), or None if nothing here
    is playing.

    Every threshold is relative to this take: its own peak and its own room.
    Turning a player down must not delete their notes (DREAM.md, v2).  The one
    exception is a take with no quiet in it, which has nothing to be relative to.

    ``lead_gap_s`` is the silence that ends a phrase.  Fewer than ``lead_notes``
    notes (or a rustle) followed by that much silence and then more sound was not
    the phrase, it was someone getting ready.
    """
    if levels.size == 0:
        return None
    peak = float(levels.max())
    floor = float(np.percentile(levels, 5))
    if peak < SILENCE_DB or peak - floor < contrast_db:
        return None
    if peak - floor < PHRASE_CONTRAST_DB and peak < NO_CONTRAST_MIN_DB:
        return None
    gate = max(peak - GATE_BELOW_PEAK_DB, floor + GATE_ABOVE_FLOOR_DB)
    gate = min(gate, peak - 6.0)
    above = np.flatnonzero(levels > gate)
    if above.size == 0:
        return None
    sep = max(1, int(round(ONSET_MIN_SEP_S * sr / hop)))
    if lead_gap_s is not None:
        gap = max(1, int(round(lead_gap_s * sr / hop)))
        start = 0
        for s in np.flatnonzero(np.diff(above) >= gap):
            head = above[start:s + 1]
            if (len(_onset_hops(head, levels, sep)) >= lead_notes
                    and head.size * hop / sr >= STRAY_MAX_S):
                break
            start = int(s) + 1
        above = above[start:]
        above = _without_lead_strays(above, levels, peak, sr, hop)
    first, end = int(above[0]), int(above[-1]) + 1
    return first, end, _onset_hops(above, levels, sep), above.size * hop / sr


def is_held_pitch(audio: np.ndarray, sr: int) -> bool:
    """Whether the loudest stretch of ``audio`` is a sustained note or chord.

    Reads only the windows near the take's loudest, and the median of them, so a clink
    or a voice in passing leaves no mark.  A hum is a held pitch too: this can't tell
    a fridge from a chord.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n = audio.size // HELD_WINDOW
    if n == 0:
        return False
    frames = audio[:n * HELD_WINDOW].reshape(n, HELD_WINDOW)
    power = np.mean(frames * frames, axis=1)
    loud = frames[power >= 0.25 * float(power.max())]     # within 6 dB of the loudest
    if loud.shape[0] * HELD_WINDOW < HELD_MIN_S * sr or float(power.max()) <= 0.0:
        return False
    spectra = np.abs(np.fft.rfft(loud * np.hanning(HELD_WINDOW), axis=1)) ** 2
    hz = np.fft.rfftfreq(HELD_WINDOW, 1.0 / sr)
    band = np.median(spectra, axis=0)[(hz >= HELD_BAND_HZ[0]) & (hz <= HELD_BAND_HZ[1])]
    neighbours = median_filter(band, size=HELD_NEIGHBOURS, mode="nearest")
    peaks = band > HELD_PEAK_OVER_NEIGHBOURS * neighbours
    return float(band[peaks].sum()) >= HELD_MIN_SHARE * float(band.sum()) > 0.0


def _sharpen(audio: np.ndarray, hop_i: int, hop: int = LEVEL_HOP) -> int:
    """The sample where an attack noticed at ``hop_i`` actually begins."""
    lo = max(0, (hop_i - 1) * hop)
    region = np.abs(audio[lo:(hop_i + 2) * hop])
    if region.size == 0:
        return lo
    return lo + int(np.argmax(region >= 0.1 * float(region.max())))


def find_phrase(audio: np.ndarray, sr: int, *,
                contrast_db: float = PHRASE_CONTRAST_DB,
                lead_gap_s: float | None = None,
                lead_notes: int = 2) -> Phrase | None:
    """Where the phrase really starts and ends.  None if nobody played.

    ``contrast_db=0`` accepts a take with no quiet in it at all, which is what a
    player closing a sustained take by hand means, provided it is louder than a
    room (:data:`NO_CONTRAST_MIN_DB`).
    """
    audio = np.asarray(audio, dtype=np.float32)
    found = _phrase_hops(level_db(audio), sr, contrast_db=contrast_db,
                         lead_gap_s=lead_gap_s, lead_notes=lead_notes)
    if found is None:
        return None
    _, end, onsets, sounding = found
    at = tuple(_sharpen(audio, i) for i in onsets)
    return Phrase(first=at[0], last_onset=at[-1],
                  end=min(audio.size, end * LEVEL_HOP), onsets=at,
                  sounding_s=sounding)


def bars_held(last_onset_bars: float, *, beats_per_bar: int = 4,
              tol_beats: float = ON_THE_LINE_BEATS) -> int:
    """How many bars a phrase fills, from where its last note fell.

    A last note on a bar line closes the phrase rather than opening a new bar:
    landing on the next "one" is how a player says the phrase is over.
    """
    pos = max(0.0, float(last_onset_bars))
    nearest = round(pos)
    if nearest >= 1 and abs(pos - nearest) * beats_per_bar <= tol_beats:
        return int(nearest)
    return max(1, math.ceil(pos))


# ------------------------------------------------------- the loop sets the tempo

@dataclasses.dataclass(frozen=True)
class TempoReading:
    bpm: float
    bars: int
    span_s: float


def _grid_error(onsets_s: np.ndarray, eighth_s: float) -> float:
    pos = onsets_s / eighth_s
    return float(np.mean(np.abs(pos - np.round(pos))) * 2.0)


def infer_tempo(phrase: Phrase, sr: int, *, bars: int | None = None,
                beats_per_bar: int = 4) -> TempoReading:
    """A tempo from the take alone.  Raises :class:`LoopRefused` if it shows none.

    The loop's length is the strong clue, so a tempo is only ever a whole number
    of bars across the phrase, and there are two readings of how long that is:
    the last note closes it (real evidence), or one more gap would have.  Each is
    scored by how well the notes sit on eighths.  Doubling and halving fit equally
    well, so a mild pull towards a middling tempo decides.  It can be wrong, which
    is why naming the bars ("make a 4 bar loop") skips the question.

    Unnamed, the first note does not count towards the fit (it is on the grid by
    construction) and the fit has to be tighter the fewer notes there are, so
    that a few coughs after the cue do not start the band.
    """
    ats = np.asarray(phrase.onsets, dtype=np.float64)
    rel = (ats - ats[0]) / float(sr)
    if rel.size < 2:          # naming the bars can't help: one sound has no length
        raise LoopRefused("one sound with nothing running shows no tempo, even if you say "
                          "how many bars: count the band in first, then make the loop, or "
                          "play at least two notes and say how many bars")
    if bars is None and rel.size < TEMPO_MIN_NOTES:
        raise LoopRefused(f"only {rel.size} notes in that take, which shows no tempo: "
                          "say how many bars, like \"make a 4 bar loop\"")
    gaps = np.diff(rel)
    spans = [(float(rel[-1]), 0.0),
             (float(rel[-1] + np.median(gaps)), TEMPO_GUESSED_GAP_PENALTY)]
    counts = TEMPO_BAR_COUNTS if bars is None else (int(bars),)
    best: tuple[float, float, TempoReading] | None = None
    lo, hi = TEMPO_RANGE_BPM
    in_range = False
    for span, penalty in spans:
        for n in counts:
            bpm = 60.0 * n * beats_per_bar / span
            if not lo <= bpm <= hi:
                continue
            in_range = True
            # the closing note is the top of the next pass, not a note in this one
            kept = rel if penalty else rel[:-1]
            err = _grid_error(kept if bars else kept[1:], 30.0 / bpm)
            prior = TEMPO_PRIOR_WEIGHT * math.log2(bpm / TEMPO_PRIOR_BPM) ** 2
            score = err + prior + penalty
            if best is None or score < best[0]:
                best = (score, err, TempoReading(bpm=bpm, bars=n, span_s=span))
    if not in_range:
        raise LoopRefused(
            f"that take would be outside {lo:.0f} to {hi:.0f} bpm at every bar "
            "count, so plume will not set a tempo from it")
    limit = (TEMPO_MAX_GRID_ERROR if bars
             else TEMPO_GRID_LIMITS.get(rel.size, TEMPO_MAX_GRID_ERROR))
    if best is None or best[1] > limit:
        need = ("count it in first" if bars
                else "say how many bars, like \"make a 4 bar loop\", or count it in first")
        raise LoopRefused(f"those notes do not sit on a steady beat plume can find: {need}")
    return best[2]


# --------------------------------------------------------------------- fitting

def fit_to_bars(audio: np.ndarray, bars: int, bpm: float, sr: int, *,
                beats_per_bar: int = 4,
                max_ratio: float = MAX_FIT_RATIO) -> np.ndarray:
    """Exactly ``bars`` bars long, by padding or trimming the end.

    Never resamples: the pitch cannot move and neither can a note.  A miss bigger
    than ``max_ratio`` would need a real time-stretch, so it is refused.
    """
    target = int(round(int(bars) * beats_per_bar * 60.0 / float(bpm) * sr))
    have = int(audio.size)
    miss = abs(have / target - 1.0) if target else 1.0
    if miss > max_ratio:
        raise LoopRefused(
            f"that take is {miss * 100.0:.1f}% away from {bars} bar(s) at "
            f"{bpm:.1f} bpm, more than the {max_ratio * 100.0:.0f}% plume will fit, "
            "and plume has no time-stretch that keeps pitch")
    if have >= target:
        return np.array(audio[:target], dtype=np.float32)
    out = np.zeros(target, dtype=np.float32)
    out[:have] = audio
    return out


# ------------------------------------------------------------------------ join

def close_join(body: np.ndarray, pre: np.ndarray,
               fade: int = JOIN_SAMPLES) -> np.ndarray:
    """Close a loop so that its last sample runs straight into its first.

    ``pre`` is what the mic heard just before ``body`` began.  The end of the loop
    crossfades into it, so the loop ends on the sample that really came before its
    first one.  That is a join with nothing to click, and it leaves the attack at
    the top of the loop untouched, which fading the head would soften.  Nothing
    before the top (the take started loud) leaves a real jump, so only then does
    the head get a very short fade.
    """
    out = np.array(body, dtype=np.float32)
    fade = min(int(fade), out.size // 2)
    if fade <= 0:
        return out
    before = np.zeros(fade, dtype=np.float32)
    pre = np.asarray(pre, dtype=np.float32)[-fade:]
    if pre.size:
        before[-pre.size:] = pre
    theta = np.linspace(0.0, 0.5 * math.pi, fade, dtype=np.float64)
    tail = out[-fade:]
    out[-fade:] = (tail * np.cos(theta) + before * np.sin(theta)).astype(np.float32)
    step = float(np.abs(np.diff(body[:fade])).max()) if fade > 1 else 0.0
    if abs(float(out[0]) - float(out[-1])) > 2.0 * step + 1e-4:
        n = min(HEAD_FADE_SAMPLES, out.size)
        out[:n] *= np.sin(np.linspace(0.0, 0.5 * math.pi, n)).astype(np.float32)
    return out


def normalize(audio: np.ndarray, *, peak_db: float = LOOP_PEAK_DB,
              max_boost_db: float = LOOP_MAX_BOOST_DB) -> np.ndarray:
    """To one peak, whatever the mic level.  Boost is capped so noise stays noise."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0.0:
        return np.array(audio, dtype=np.float32)
    gain = min(db_to_lin(peak_db) / peak, db_to_lin(max_boost_db))
    return (audio * gain).astype(np.float32)


# ---------------------------------------------------------------------- layers

@dataclasses.dataclass
class LoopLayer:
    """One loop.  ``anchor_bar`` is the session bar its first bar was played on,
    which is how it keeps the place in the bar and the phrase it was played in."""
    name: str
    audio: np.ndarray
    bars: int
    bpm: float
    anchor_bar: int = 0
    muted: bool = False
    band_in_mic: bool = False
    entered: bool = False       # has reached the top of its cycle
    sounding: bool = False      # was audible at the end of the last bar
    retiring: bool = False      # a new take of this name is on its way in

    def bar_slice(self, bar: int, n: int) -> np.ndarray:
        at = (int(bar) - self.anchor_bar) % self.bars
        start = int(round(at * self.audio.size / self.bars))
        idx = (start + np.arange(int(n))) % self.audio.size
        return self.audio[idx].copy()


class LoopStack:
    """The layers, by name.  Changes land on the next bar planned, faded so that
    they cannot click, and a layer first comes in at the top of its own cycle."""

    def __init__(self) -> None:
        self.layers: list[LoopLayer] = []
        self._leaving: list[LoopLayer] = []

    # -- names ----------------------------------------------------------

    def names(self) -> list[str]:
        return [x.name for x in self.layers if not x.retiring]

    def latest(self) -> str:
        names = self.names()
        if not names:
            raise LoopRefused("there is no loop yet: say \"make a loop of this\"")
        return names[-1]

    def layer(self, name: str) -> LoopLayer:
        for x in self.layers:
            if x.name == name and not x.retiring:
                return x
        names = self.names()
        raise LoopRefused(f"no loop called {name!r}"
                          + (f": there is {', '.join(names)}" if names
                             else ": there are none yet"))

    def next_name(self) -> str:
        taken = {x.name for x in self.layers}
        i = 1
        while f"loop{i}" in taken:
            i += 1
        return f"loop{i}"

    # -- changes --------------------------------------------------------

    def add(self, layer: LoopLayer) -> None:
        """A name that is already here is a retake: the old take plays on until
        the new one reaches the top of its cycle, so there is no hole."""
        for x in self.layers:
            if x.name == layer.name:
                x.retiring = True
        self.layers.append(layer)

    def mute(self, name: str) -> None:
        self.layer(name)
        for x in self.layers:
            if x.name == name:
                x.muted = True

    def unmute(self, name: str) -> None:
        self.layer(name).muted = False

    def swap(self, out_name: str, in_name: str) -> None:
        self.layer(out_name)
        self.layer(in_name)
        self.mute(out_name)
        self.unmute(in_name)

    def drop(self, name: str) -> None:
        self.layer(name)
        self._leaving += [x for x in self.layers if x.name == name]
        self.layers = [x for x in self.layers if x.name != name]

    # -- sound ----------------------------------------------------------

    @property
    def fading(self) -> bool:
        return bool(self._leaving)

    def _render(self, x: LoopLayer, bar: int, n: int, want: bool) -> np.ndarray | None:
        """This layer's bar, or None if it is silent throughout."""
        was = x.sounding
        x.sounding = want
        if not want and not was:
            return None
        s = x.bar_slice(bar, n)
        k = min(DECLICK_SAMPLES, n)
        ramp = np.sin(np.linspace(0.0, 0.5 * math.pi, k)).astype(np.float32)
        if want and not was:
            s[:k] *= ramp
        elif was and not want:
            s[:k] *= ramp[::-1]
            s[k:] = 0.0
        return s

    def mix_bar(self, bar: int, n: int, *, stop: bool = False) -> np.ndarray:
        """``n`` samples of every layer for this bar.  ``stop`` fades them all."""
        out = np.zeros(int(n), dtype=np.float32)
        arrivals = set()
        for x in self.layers:
            if not x.retiring and not x.entered and not x.muted and (
                    (int(bar) - x.anchor_bar) % x.bars == 0):
                x.entered = True
                arrivals.add(x.name)
        for x in self.layers:
            if x.retiring and x.name in arrivals:
                self._leaving.append(x)
        self.layers = [x for x in self.layers
                       if not (x.retiring and x.name in arrivals)]
        for x in self.layers:
            want = x.entered and not x.muted and not stop
            s = self._render(x, bar, int(n), want)
            if s is not None:
                out += s
        for x in self._leaving:
            s = self._render(x, bar, int(n), False)
            if s is not None:
                out += s
        self._leaving = []
        return out

    def status(self) -> list[dict]:
        return [{"name": x.name, "bars": x.bars, "bpm": round(x.bpm, 2),
                 "muted": x.muted, "sounding": x.sounding,
                 "anchor_bar": x.anchor_bar, "band_in_mic": x.band_in_mic}
                for x in self.layers if not x.retiring]


# ----------------------------------------------------------------------- words

_BAR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "twelve": 12, "sixteen": 16}
_NAME_NOISE = {"the", "a", "an", "my", "that", "this", "loop", "loops", "layer",
               "layers", "new", "another", "all", "looper", "looping", "called",
               "named", "please", "now", "turn", "recording", "of", "every", "each",
               "both"}
_CLOSE = ("that's the loop", "thats the loop", "that is the loop", "close the loop",
          "close loop", "end the loop", "end loop", "finish the loop", "loop done")
_CANCEL = ("cancel the loop", "cancel loop", "scrap the loop", "forget the loop",
           "cancel the take")
#: These mean every loop to a mute or a drop, so no loop may be called one of them.
_ALL_LOOPS_WORDS = ("all", "every", "each", "both")
_LIST = ("list loops", "list the loops", "list layers", "show loops",
         "show the loops", "what loops", "which loops")
_RECORD_RE = re.compile(
    r"\b(?:make|record|cut|capture|grab|start|begin)\s+(?:me\s+)?(?:(?:a|an|another|the|new)\s+)*"
    r"(?P<mid>.*?)\s*\b(?:loop|layer)\b\s*(?P<tail>.*)$")
_TAIL_RE = re.compile(
    r"^(?:(?:of|from|out\s+of|with)\s+(?:this|that|it)(?:\s+[a-z0-9']+)?\s*)?"
    r"(?:(?:called|named|as)\s+(?:the\s+)?(?P<name>[a-z0-9']+))?$")
_BAR_COUNT = r"(\d+|" + "|".join(_BAR_WORDS) + r")\s+bars?\b"
_BARS_RE = re.compile(r"\b" + _BAR_COUNT)
_TAIL_BARS_RE = re.compile(r"\b(?:(?:of|for|over)\s+)?" + _BAR_COUNT + r"(?:\s+long\b)?\s*")
_LOOP_WORD = r"\b(?:loops?\d*|layers?\d*|looper|looping)\b"
_STOP_RE = re.compile(r"\b(?:stop|end|finish|kill|quit|close)\s+(?P<name>.*"
                      + _LOOP_WORD + r".*)$")
_STOP_AFTER_RE = re.compile(r"^(?P<name>.*" + _LOOP_WORD + r".*?)\s+(?:stop|off)$")
_START_RE = re.compile(r"\b(?:start|begin)\s+looping\b")
_NEXT_BARS_RE = re.compile(r"\bloop\s+(?:for\s+|the\s+next\s+)?" + _BAR_COUNT)
_PAST_RE = re.compile(r"\bloop\s+(?:the\s+|that\s+)?(?:last|previous)\b")
_NUMBERED_RE = re.compile(r"\b(?:loop|layer)\s*(\d+|" + "|".join(_BAR_WORDS) + r")\b")
_ONE_NAMED_RE = re.compile(_NUMBERED_RE.pattern + r"|\b(?:called|named)\b")
_SWAP_RE = re.compile(r"\bswap\s+(?P<a>.+?)\s+(?:for|with)\s+(?P<b>.+)$")


def _bar_count(said: str) -> int:
    return _BAR_WORDS.get(said) or int(said[:_MAX_BAR_DIGITS])


def _target(words: str) -> str | None:
    # "loop 1" is how anyone says the loop plume named "loop1"
    words = _NUMBERED_RE.sub(lambda m: f"loop{_bar_count(m[1])}", words)
    kept = [w for w in words.split() if w not in _NAME_NOISE]
    return "-".join(kept) or None


def parse_loop_cue(text) -> dict | None:
    """``{"op": ..., "text": ...}`` plus ``bars``/``name``/``other`` when said.

    None means this is not about a loop and belongs to the band.  A cue has to say
    "loop" or "layer" to be one, so this cannot take "drop the bass" or "mute" away
    from the form and texture cues that already mean something, and a sentence
    that asks for nothing this knows is left alone too.  Two kinds are never left
    alone, because the band would read them as something else: stopping a loop
    (to the band, "stop" ends the song) and asking for one in words this cannot
    follow (``op`` is ``"unclear"``, so the player is told).  A loop cue the band
    would hear as ending the song is one of those.  Never raises.
    """
    if not isinstance(text, str):
        return None
    words = " ".join(re.findall(r"[a-z0-9']+", text.lower()))
    if not re.search(_LOOP_WORD, words):
        return None

    def out(op: str, **extra) -> dict:
        d = {"op": op, "text": text}
        d.update({k: v for k, v in extra.items() if v is not None})
        return d

    for phrases, op in ((_CLOSE, "close"), (_CANCEL, "cancel"), (_LIST, "list")):
        if op == "close" and _ONE_NAMED_RE.search(words):
            continue        # "end loop 2" is about that loop, not the take
        if any(re.search(rf"\b{p}\b", words) for p in phrases):
            return out(op)
    swap = _SWAP_RE.search(words)
    if swap:
        a, b = _target(swap["a"]), _target(swap["b"])
        return out("swap", name=a, other=b) if a and b else None
    verb = re.search(r"\b(unmute|bring back|restore|turn on|mute|silence|pause|turn off"
                     r"|drop|delete|remove)\b", words)
    stop = _STOP_RE.search(words) or _STOP_AFTER_RE.search(words)
    if stop and not verb:
        return out("stop", name=_target(stop["name"]))
    if verb:
        op = {"unmute": "unmute", "bring back": "unmute", "restore": "unmute",
              "turn on": "unmute", "mute": "mute", "silence": "mute",
              "pause": "mute", "turn off": "mute"}.get(verb[1], "drop")
        rest = words[verb.end():]
        name = _target(rest)
        every = name is None and any(w in _ALL_LOOPS_WORDS for w in rest.split())
        return out(op, name=name, every=every or None)
    if re.search(r"\bloop (?:this|that|it)\b", words):
        said = _BARS_RE.search(words)
        return out("record", bars=_bar_count(said[1]) if said else None)
    if _START_RE.search(words):
        return out("record")
    rec = _RECORD_RE.search(words)
    if rec is None:
        said = _NEXT_BARS_RE.search(words)
        if said:
            return out("record", bars=_bar_count(said[1]))
        if _PAST_RE.search(words):
            return out("unclear")
        # Left to the band this would end the song, which nobody asking about a
        # loop means.  Imported here because the band's words are not the looper's.
        from .mind import parse_cue  # noqa: PLC0415
        return out("unclear") if parse_cue(text)["kind"] == "end" else None
    mid, rest = rec["mid"], rec["tail"].strip()
    bars = None
    said = _BARS_RE.search(mid)
    if said:
        bars = _bar_count(said[1])
        mid = mid[:said.start()] + mid[said.end():]
    else:
        said = _TAIL_BARS_RE.search(rest)
        if said:
            bars = _bar_count(said[1])
            rest = (rest[:said.start()] + rest[said.end():]).strip()
    tail = _TAIL_RE.match(rest)
    if tail is None:
        return out("unclear")
    if tail["name"] in _ALL_LOOPS_WORDS:
        return out("unclear")
    name = tail["name"] or _target(mid)
    return out("record", bars=bars, name=name)


# ------------------------------------------------------------ one session's state

@dataclasses.dataclass(frozen=True)
class LoopTempo:
    """A tempo a loop has set, for a transport that had none."""
    bpm: float
    next_line_s: float      # where the phrase would come round, input-stream seconds


@dataclasses.dataclass
class _Take:
    name: str | None
    bars: int | None
    start: int              # input-stream sample of the first sample captured
    pre: np.ndarray
    chunks: list[np.ndarray]
    size: int = 0
    closing: bool = False
    levels: list[float] = dataclasses.field(default_factory=list)
    unread: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    since_check: int = 0
    first: int | None = None
    room_db: float | None = None   # the mic just before the cue, if there was any
    loudest_db: float = -math.inf


class Looper:
    """The looper for one session.  Called from three threads, safe from all:
    the cue socket (:meth:`handle_cue`), analysis (:meth:`feed`) and the planner
    (:meth:`note_bar`, :meth:`mix_bar`, :meth:`held_bpm`).  The audio callback
    never touches it.

    Sample positions are in the input stream.  A bar placed at output sample ``p``
    is heard and played along to, so it comes back through the mic at about
    ``p + roundtrip_samples``, and that is the grid a take is cut against.
    """

    def __init__(self, sr: int, *, roundtrip_samples: int = 0, bleed_probe=None):
        self.sr = int(sr)
        self.roundtrip_samples = int(roundtrip_samples)
        self.bleed_probe = bleed_probe
        self.stack = LoopStack()
        self._lock = threading.Lock()
        self._pos = 0
        self._pre = np.zeros(0, dtype=np.float32)
        self._take: _Take | None = None
        self._lines: list[tuple[int, int, float]] = []   # (bar, input sample, bpm)
        self._tempo: LoopTempo | None = None
        self._events: list[str] = []     # for the planner's decision log
        self._notes: list[str] = []      # for the player's terminal
        self._last_event: str | None = None
        self._gain = 1.0
        self._room = 1.0                 # what _give_way left the last bar at

    # -- the socket's side ----------------------------------------------

    def handle_cue(self, text) -> dict | None:
        """The reply to a loop cue, or None if it is not one."""
        cue = parse_loop_cue(text)
        if cue is None:
            return None
        with self._lock:
            try:
                reply = self._act(cue)
            except LoopRefused as e:
                return {"ok": False, "op": cue["op"], "error": str(e)}
        return {"ok": True, "op": cue["op"], **reply}

    def _act(self, cue: dict) -> dict:
        op, name = cue["op"], cue.get("name")
        if op == "unclear":
            raise LoopRefused(f"could not follow {cue['text']!r}, so no take has "
                              "started: try \"make a 4 bar loop\" or \"make a loop "
                              "of this\"")
        if op == "stop":
            return self._stop(name)
        if op == "record":
            return self._arm(name, cue.get("bars"))
        if op == "close":
            if self._take is None:
                raise LoopRefused("no take is being recorded")
            self._take.closing = True
            return {"said": "closing the take"}
        if op == "cancel":
            if self._take is None:
                raise LoopRefused("no take is being recorded")
            self._take = None
            return {"said": "take cancelled"}
        if op == "list":
            said = ", ".join(self.stack.names()) or "no loops"
            if self._last_event:
                said += f" (last: {self._last_event})"
            return {"said": said, **self._status()}
        if op == "swap":
            self.stack.swap(name, cue["other"])
            return {"said": f"swapping {name} for {cue['other']} on the next bar"}
        if name is None:
            name = self.stack.latest()
        change = {"mute": self.stack.mute, "unmute": self.stack.unmute,
                  "drop": self.stack.drop}[op]
        verb = {"mute": "muting", "unmute": "bringing back", "drop": "dropping"}[op]
        if cue.get("every"):
            names = self.stack.names()
            for x in names:
                change(x)
            return {"said": f"{verb} {', '.join(names)} on the next bar"}
        change(name)
        return {"said": f"{verb} {name} on the next bar", "name": name}

    def _stop(self, name: str | None) -> dict:
        """Stopping a loop mutes it, which can be undone.  A take in progress
        is what "stop" means while one is being recorded."""
        if name is not None:
            self.stack.mute(name)
            return {"said": f"muting {name} on the next bar", "name": name}
        if self._take is not None:
            self._take.closing = True
            return {"said": "closing the take"}
        names = self.stack.names()
        if not names:
            raise LoopRefused("there is no loop to stop")
        for x in names:
            self.stack.mute(x)
        return {"said": f"muting {', '.join(names)} on the next bar"}

    def _arm(self, name: str | None, bars: int | None) -> dict:
        if self._take is not None:
            raise LoopRefused("already recording a take: say \"that's the loop\" or "
                              "\"cancel the loop\" first")
        retake = name is not None and name in self.stack.names()
        if len(self.stack.names()) >= MAX_LAYERS and not retake:
            raise LoopRefused(f"{MAX_LAYERS} loops is the most plume will hold: "
                              "drop one first")
        if bars is not None:
            if bars < 1:
                raise LoopRefused("a loop needs at least one bar")
            if bars > MAX_NAMED_BARS:
                raise LoopRefused(f"{MAX_NAMED_BARS} bars is the most one take can hold")
            if self._lines and bars * self._bar_s() > MAX_TAKE_S:
                raise LoopRefused(f"{bars} bars is longer than the "
                                  f"{MAX_TAKE_S:.0f} seconds one take can hold")
        pre = self._pre.copy()
        room_db = None
        if pre.size:
            power = float(np.mean(np.square(pre, dtype=np.float64)))
            room_db = max(SILENCE_DB, 10.0 * math.log10(max(power, 1e-12)))
        self._take = _Take(name=name, bars=bars, start=self._pos, pre=pre, chunks=[],
                           room_db=room_db)
        length = f"{bars} bar loop" if bars else "loop"
        return {"said": f"recording a {length}: play when you are ready",
                "name": name, "bars": bars}

    # -- the analysis thread's side -------------------------------------

    @property
    def capturing(self) -> bool:
        with self._lock:
            return self._take is not None

    def feed(self, block: np.ndarray) -> None:
        """Mic audio, in order, from the start of the stream."""
        block = np.asarray(block, dtype=np.float32)
        with self._lock:
            self._pos += block.size
            joined = np.concatenate((self._pre, block)) if self._pre.size else block
            self._pre = joined[-PRE_ROLL_SAMPLES:].copy()
            t = self._take
            if t is None:
                return
            t.chunks.append(block.copy())
            t.size += block.size
            t.since_check += block.size
            if t.closing:
                self._finish(contrast_db=0.0, why="closed by hand")
            elif t.size >= int(MAX_TAKE_S * self.sr):
                # A held chord that never rose has no other way to be kept.
                held = self._rose(t) or is_held_pitch(np.concatenate(t.chunks), self.sr)
                self._finish(contrast_db=0.0 if held else PHRASE_CONTRAST_DB,
                             why=f"reached the {MAX_TAKE_S:.0f} second limit",
                             nothing=(NO_CLEAR_START if t.loudest_db > SILENCE_DB else None))
            elif self._take_is_over(t):
                self._finish(contrast_db=self._contrast_for(t), why="phrase ended")

    def _contrast_for(self, t: _Take) -> float:
        """A length named on a running grid ends on its bar line, so it needs no quiet
        to show where the phrase ends: held chords have none.  Something still has to
        have started, or a noisy room would close the take before the player does."""
        named = t.bars is not None and self._lines
        return 0.0 if named and self._rose(t) else PHRASE_CONTRAST_DB

    @staticmethod
    def _rose(t: _Take) -> bool:
        """Whether this take has got clearly louder than the mic was at the cue, or is
        too loud to be a room at all."""
        if t.loudest_db >= PLAINLY_PLAYING_DB:
            return True
        base = t.room_db if t.room_db is not None else (t.levels[0] if t.levels else None)
        return base is not None and t.loudest_db - base >= CUE_RISE_DB

    def _take_is_over(self, t: _Take) -> bool:
        """Cheap: a named length ends on its bar line, otherwise on silence."""
        if t.first is not None and t.bars is not None and self._lines:
            if self._pos >= self._line(self._start_bar(t.first) + t.bars):
                return True
        if t.since_check < CHECK_EVERY_SAMPLES:
            return False
        fresh = np.concatenate((t.unread, *t.chunks[-self._unread_blocks(t):]))
        t.since_check = 0
        usable = fresh.size - fresh.size % LEVEL_HOP
        heard = level_db(fresh[:usable])
        t.levels.extend(heard.tolist())
        if heard.size:
            t.loudest_db = max(t.loudest_db, float(heard.max()))
        t.unread = fresh[usable:].copy()
        gap_s, notes = self._lead_in(t)
        found = _phrase_hops(np.asarray(t.levels), self.sr,
                             contrast_db=self._contrast_for(t),
                             lead_gap_s=gap_s, lead_notes=notes)
        if found is None or found[3] < STRAY_MAX_S:
            t.first = None
            return False
        t.first = t.start + found[0] * LEVEL_HOP
        if t.bars is not None and self._lines:
            return False
        hold = gap_s if len(found[2]) >= notes else LONE_SOUND_HOLD_S
        heard_to = found[1] * LEVEL_HOP
        return (len(t.levels) * LEVEL_HOP - heard_to) / self.sr >= hold

    def _lead_in(self, t: _Take) -> tuple[float, int]:
        """(the silence that ends a phrase, the notes that make one).  With no
        tempo and no bars named, anything short of a tempo's worth of notes cannot
        become a loop at all, so there is nothing to gain by stopping for it."""
        if self._lines:
            return self._bar_s(), 2
        return NO_TEMPO_HOLD_S, 2 if t.bars is not None else TEMPO_MIN_NOTES

    def _unread_blocks(self, t: _Take) -> int:
        """How many of the newest chunks arrived since the last check."""
        want, got, k = t.since_check, 0, 0
        while got < want and k < len(t.chunks):
            k += 1
            got += t.chunks[-k].size
        return max(1, k)

    def _finish(self, *, contrast_db: float, why: str, nothing: str | None = None) -> None:
        t, self._take = self._take, None
        audio = np.concatenate(t.chunks) if t.chunks else np.zeros(0, dtype=np.float32)
        gap_s, notes = self._lead_in(t)
        phrase = find_phrase(audio, self.sr, contrast_db=contrast_db,
                             lead_gap_s=gap_s, lead_notes=notes)
        if phrase is None or phrase.sounding_s < STRAY_MAX_S:
            self._event(f"loop not made ({why}): {nothing or 'heard nothing to loop'}")
            return
        try:
            layer = (self._cut_on_grid(t, audio, phrase) if self._lines
                     else self._cut_free(t, audio, phrase))
        except LoopRefused as e:
            self._event(f"loop not made ({why}): {e}")
            return
        layer.band_in_mic = self._band_in_mic()
        self.stack.add(layer)
        note = (f"loop {layer.name!r} made ({why}): {layer.bars} bar(s) at "
                f"{layer.bpm:.1f} bpm, comes in at the top of its cycle")
        if layer.band_in_mic:
            note += ("; the speakers are in the mic, so this loop has the band in "
                     "it: use headphones or a direct input")
        self._event(note)

    def _event(self, text: str) -> None:
        self._events = [*self._events[-31:], text]
        self._notes = [*self._notes[-31:], text]
        self._last_event = text

    def _band_in_mic(self) -> bool:
        if self.bleed_probe is None:
            return False
        try:
            measured = self.bleed_probe()   # None: there has been no evidence yet
            return measured is not None and float(measured) > BLEED_WARN_DB
        except Exception:                # noqa: BLE001 - only ever a warning
            return False

    def _cut_on_grid(self, t: _Take, audio: np.ndarray, phrase: Phrase) -> LoopLayer:
        """A tempo is running: keep the phrase where it was played in the bar."""
        bpm = self._lines[-1][2]
        b0 = self._start_bar(t.start + phrase.first)
        line0 = self._line(b0)
        bar_n = self._bar_s() * self.sr
        bars = t.bars or bars_held((t.start + phrase.last_onset - line0) / bar_n)
        natural = self._line(b0 + bars) - line0
        at = line0 - t.start
        body = np.zeros(natural, dtype=np.float32)
        lo, hi = max(0, at), min(audio.size, at + natural)
        if hi > lo:
            body[lo - at:hi - at] = audio[lo:hi]
            if lo > at:
                # the take began after this bar line: do not start on a cliff
                k = min(DECLICK_SAMPLES, hi - lo)
                body[lo - at:lo - at + k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
        loop = fit_to_bars(body, bars, bpm, self.sr)
        if at < 0:      # the bar line came before the take: silence, not the pre-roll
            pre = np.zeros(0, dtype=np.float32)
        else:
            pre = audio[max(0, at - JOIN_SAMPLES):at]
            if pre.size < JOIN_SAMPLES and t.pre.size:
                pre = np.concatenate((t.pre, pre))
        return LoopLayer(name=t.name or self.stack.next_name(),
                         audio=normalize(close_join(loop, pre)), bars=bars, bpm=bpm,
                         anchor_bar=b0)

    def _cut_free(self, t: _Take, audio: np.ndarray, phrase: Phrase) -> LoopLayer:
        """No tempo is running: the phrase's top is the "one" and the loop sets it."""
        tempo = infer_tempo(phrase, self.sr, bars=t.bars)
        n = int(round(tempo.bars * 4 * 60.0 / tempo.bpm * self.sr))
        body = np.zeros(n, dtype=np.float32)
        got = audio[phrase.first:phrase.first + n]
        body[:got.size] = got
        pre = audio[max(0, phrase.first - JOIN_SAMPLES):phrase.first]
        if pre.size < JOIN_SAMPLES and t.pre.size:
            pre = np.concatenate((t.pre, pre))
        self._tempo = LoopTempo(
            bpm=tempo.bpm,
            next_line_s=(t.start + phrase.first + n) / self.sr)
        return LoopLayer(name=t.name or self.stack.next_name(),
                         audio=normalize(close_join(body, pre)), bars=tempo.bars,
                         bpm=tempo.bpm, anchor_bar=0)

    def take_tempo(self) -> LoopTempo | None:
        """The tempo a loop has just set, handed over once."""
        with self._lock:
            tempo, self._tempo = self._tempo, None
            return tempo

    # -- the planner's side ---------------------------------------------

    def _bar_s(self) -> float:
        return 4 * 60.0 / self._lines[-1][2]

    def _line(self, bar: int) -> int:
        """Input sample of a bar line, reckoned forward past what is planned."""
        for b, pos, _ in reversed(self._lines):
            if b == bar:
                return pos
            if b < bar:
                return pos + int(round((bar - b) * self._bar_s() * self.sr))
        b, pos, bpm = self._lines[0]
        return pos - int(round((b - bar) * 4 * 60.0 / bpm * self.sr))

    def _bar_for(self, pos: int) -> int:
        """The bar whose line is at or before ``pos``, never before the first."""
        bar, line = self._lines[0][:2]
        for b, p, _ in self._lines:
            if p > pos:
                return bar
            bar, line = b, p
        return bar + int((pos - line) // (self._bar_s() * self.sr))

    def _start_bar(self, first: int) -> int:
        """The bar a phrase belongs to; a hair early still means the coming one."""
        beat_n = self._bar_s() / 4.0 * self.sr
        return self._bar_for(first + int(EARLY_START_BEATS * beat_n))

    def note_bar(self, bar: int, write_sample: int, bpm: float) -> None:
        """A bar has been planned at this output sample, at this tempo."""
        with self._lock:
            self._lines.append((int(bar), int(write_sample) + self.roundtrip_samples,
                                float(bpm)))
            del self._lines[:-512]

    def held_bpm(self) -> float | None:
        """While any loop exists the band holds its tempo: a loop is a fixed
        length of audio, and a clock that moved under it would pull them apart."""
        with self._lock:
            return self.stack.layers[0].bpm if self.stack.layers else None

    def mix_bar(self, bar: int, n: int, *, gain_db: float | None = None,
                stop: bool = False, under: np.ndarray | None = None) -> np.ndarray:
        """The loops for one bar.  ``gain_db`` is ramped across it, not stepped, and
        None keeps the level already reached: a tail asking for a bar of 0 dB would
        swell back up out of a fade.  ``under`` is the band's (n, 2) audio for this
        bar, which the loops give way to."""
        with self._lock:
            out = self.stack.mix_bar(bar, n, stop=stop)
        target = self._gain if gain_db is None else db_to_lin(gain_db)
        if target != self._gain:
            out *= np.linspace(self._gain, target, out.size, dtype=np.float32)
            self._gain = target
        elif target != 1.0:
            out *= target
        return self._give_way(out, under)

    def _give_way(self, loops: np.ndarray, band: np.ndarray | None) -> np.ndarray:
        """The loops, dipped for a moment wherever they would take the sum past full
        scale.  Each layer is already at :data:`LOOP_PEAK_DB`, so it is their sum and
        the band's peaks that get there.  The band is never touched, so this removes
        none of the band's own clipping.  Within about 5 ms of a peak where the band alone
        is past full scale the loops stay as they are, since dipping them right on the
        peak saves nothing.  The band alone is often under full scale at the edges of
        that stretch, so there the loops do add some clipping of their own.

        The dip looks ahead within the bar, which makes it a slope that is low enough
        by the time the peak arrives.  It cannot see into the next bar, so the first
        :data:`HEAD_FADE_SAMPLES` of a bar are a ramp from wherever the last one ended.
        """
        if band is None:
            return loops * self._room if self._room != 1.0 else loops
        n = loops.size
        peak = np.zeros(n, dtype=np.float64)
        m = min(n, band.shape[0])
        peak[:m] = np.abs(band[:m]).max(axis=1)
        mine = np.abs(loops)
        reach = 2 * DECLICK_SAMPLES + 1
        over = (peak + mine > 1.0) & (mine > 0.0)
        hot = peak >= 1.0
        if hot.any():
            over &= maximum_filter1d(hot.astype(np.uint8), size=reach, mode="nearest") == 0
        if self._room == 1.0 and not over.any():
            return loops
        need = np.ones(n, dtype=np.float64)
        need[over] = (1.0 - peak[over]) / mine[over]
        gain = minimum_filter1d(need, size=reach, mode="nearest")
        gain = np.minimum(uniform_filter1d(gain, size=reach, mode="nearest"), 1.0)
        k = min(HEAD_FADE_SAMPLES, n)
        gain[:k] = np.linspace(self._room, gain[k - 1], k)
        self._room = float(gain[-1])
        return (loops * gain).astype(np.float32)

    # -- reporting ------------------------------------------------------

    def pop_events(self) -> list[str]:
        with self._lock:
            out, self._events = self._events, []
            return out

    def pop_notes(self) -> list[str]:
        with self._lock:
            out, self._notes = self._notes, []
            return out

    def status(self) -> dict:
        with self._lock:
            return self._status()

    def _status(self) -> dict:
        t = self._take
        return {"recording": None if t is None else (t.name or True),
                "layers": self.stack.status(), "last_event": self._last_event}

    @property
    def active(self) -> bool:
        with self._lock:
            return (self._take is not None or bool(self.stack.layers)
                    or self.stack.fading)
