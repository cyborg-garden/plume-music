"""Fixture humans for v1 behavior tests (SPEC-V1 section 9).

``make_human(kind, *, bpm=96, bars=24, key="A minor", seed=1)`` synthesizes a
"human guitarist" as mono float32 @ 48 kHz using only v0 modules: pluck strums
on chord tones (theory.voice_chord over a 4-chord diatonic cycle), occasional
808sub bass notes, humanized via groove.apply_groove.

Kinds
-----
- ``steady``        : constant bpm, one strum per beat.
- ``drift``         : bpm ramps linearly +6% across the middle 8 bars, then
                      holds at 1.06x for the remaining bars.
- ``stops``         : exactly 2 silent bars mid-take (hard palm-mute), resumes.
- ``long_stop``     : 8 silent bars centered, then resumes.
- ``busy_middle``   : onset density triples (1 -> 3 onsets/beat) for 4 bars.
- ``rubato_intro``  : first 8 s genuinely pulse-free (irregular strums, IOI
                      cv > 0.4), then the full steady grid offset by 8 s.
- ``high_register`` : strums voiced entirely above C5 (midi 72), no bass.

Truth metadata (``make_human_with_truth``) returns ``(audio, truth)`` with:
- ``bpm_curve``   : list of (t_s, bpm) sampled at every beat start (plus a
                    final entry at the end of the last bar). The t values ARE
                    the true beat times.
- ``bar_times_s`` : true start time of each bar (len == bars).
- ``silent_bars`` : 0-based indices of fully silent bars (exact zeros).
- ``key``         : the key string as given (parse_key-compatible).
Extra (non-contractual but handy) keys: ``busy_bars``, ``intro_end_s``,
``bars``, ``bpm_base``, ``sr``.

Determinism: outputs are a pure function of (kind, bpm, bars, key, seed) —
one np.random.default_rng(seed), no wall-clock, no global state.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.groove import GrooveParams, apply_groove  # noqa: E402
from plume.theory import (  # noqa: E402
    Key, midi_to_hz, parse_key, roman_to_chord, spell, voice_chord,
)
from plume.voices import render_note  # noqa: E402

SR = 48000

KINDS = (
    "steady", "drift", "stops", "long_stop",
    "busy_middle", "rubato_intro", "high_register",
)

_INTRO_S = 8.0            # rubato_intro: pulse-free span before the grid
_DRIFT_PCT = 0.06         # drift: +6% over the middle 8 bars
_STRUM_STAGGER_S = 0.009  # per-string delay inside a strum
_FADE_S = 0.025           # palm-mute fade into a silent span

_GROOVE = GrooveParams(
    swing=0.55, timing_bias_ms={}, humanize_ms=3.0, humanize_vel=0.05,
)


def _check_bars(kind: str, bars: int) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown fixture kind: {kind!r} (have: {list(KINDS)})")
    need = {"long_stop": 12, "drift": 8, "stops": 8, "busy_middle": 8}.get(kind, 1)
    if bars < need:
        raise ValueError(f"kind {kind!r} needs bars >= {need}, got {bars}")


def _silent_bars(kind: str, bars: int) -> list[int]:
    if kind == "stops":
        mid = bars // 2
        return [mid - 1, mid]
    if kind == "long_stop":
        s = (bars - 8) // 2
        return list(range(s, s + 8))
    return []


def _busy_bars(kind: str, bars: int) -> list[int]:
    if kind != "busy_middle":
        return []
    s = max(0, bars // 2 - 2)
    return list(range(s, min(bars, s + 4)))


def _beat_grid(kind: str, bpm: float, bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat bpm values (len bars*4) and beat times (len bars*4 + 1)."""
    n_beats = bars * 4
    if kind == "drift":
        s = max(0, (bars - 8) // 2) * 4
        e = min(n_beats, s + 32)              # ramp spans the middle 8 bars
        k = np.arange(n_beats, dtype=np.float64)
        bpm_beats = np.interp(k, [s, e], [bpm, bpm * (1.0 + _DRIFT_PCT)])
    else:
        bpm_beats = np.full(n_beats, float(bpm))
    beat_times = np.concatenate([[0.0], np.cumsum(60.0 / bpm_beats)])
    return bpm_beats, beat_times


def _progression(key_str: str):
    key = parse_key(key_str)
    numerals = ["I", "vi", "IV", "V"] if key.mode == "major" else ["i", "bVI", "bIII", "bVII"]
    return [roman_to_chord(n, key) for n in numerals]


def _add(buf: np.ndarray, y: np.ndarray, t_s: float) -> None:
    s0 = int(round(t_s * SR))
    if s0 < 0:
        y = y[-s0:]
        s0 = 0
    n = min(y.shape[0], buf.shape[0] - s0)
    if n > 0:
        buf[s0:s0 + n] += y[:n].astype(np.float64)


def _strum(buf: np.ndarray, tones: list[int], t_s: float, velocity: float,
           rng: np.random.Generator, stagger_s: float = _STRUM_STAGGER_S) -> None:
    for i, m in enumerate(tones):
        nt = t_s + stagger_s * i + float(rng.uniform(0.0, 0.003))
        y = render_note("pluck", sr=SR, dur_s=0.4, freq_hz=midi_to_hz(m),
                        velocity=velocity * (0.94 ** i), rng=rng, params={})
        _add(buf, y, nt)


def _grooved_onset(f: float, bar_idx: int, bpm_bar: float, beat_times: np.ndarray,
                   offset: float, role: str, velocity: float,
                   rng: np.random.Generator) -> tuple[float, float]:
    """Grid time of beat-fraction ``f`` in bar ``bar_idx`` + groove delta."""
    idx = bar_idx * 4 + int(f)
    fp = f - int(f)
    t_grid = offset + beat_times[idx] + fp * (beat_times[idx + 1] - beat_times[idx])
    t_local, vel = apply_groove(f, role=role, bpm=bpm_bar, g=_GROOVE,
                                rng=rng, velocity=velocity)
    delta = t_local - f * (60.0 / bpm_bar)
    return max(0.0, t_grid + delta), vel


def _render_intro(buf: np.ndarray, chords, low_midi: int,
                  rng: np.random.Generator) -> None:
    """Pulse-free strums in [0, _INTRO_S): irregular IOIs, cv > 0.4.

    IOI categories (short/long/medium) are cycled in rng-shuffled triples so
    the mixture is guaranteed bimodal (cv ~0.5) without a repeating period.
    """
    ranges = [(0.22, 0.38), (0.85, 1.35), (0.50, 0.70)]
    t = 0.4
    queue: list[int] = []
    i_ev = 0
    while t < _INTRO_S - 0.8:
        tones = voice_chord(chords[(i_ev // 4) % 4], low_midi=low_midi)
        _strum(buf, tones, t, float(rng.uniform(0.45, 0.7)), rng, stagger_s=0.012)
        if not queue:
            queue = [int(x) for x in rng.permutation(3)]
        lo, hi = ranges[queue.pop()]
        t += float(rng.uniform(lo, hi))
        i_ev += 1


def make_human_with_truth(kind: str, *, bpm: float = 96, bars: int = 24,
                          key: str = "A minor", seed: int = 1
                          ) -> tuple[np.ndarray, dict]:
    """Synthesize the fixture and its ground-truth metadata. See module doc."""
    _check_bars(kind, bars)
    rng = np.random.default_rng(seed)
    chords = _progression(key)
    silent = set(_silent_bars(kind, bars))
    busy = set(_busy_bars(kind, bars))
    bpm_beats, beat_times = _beat_grid(kind, float(bpm), bars)
    offset = _INTRO_S if kind == "rubato_intro" else 0.0
    low_midi = 76 if kind == "high_register" else 57
    include_bass = kind != "high_register"

    end_s = offset + float(beat_times[-1])
    buf = np.zeros(int(round(end_s * SR)), dtype=np.float64)

    if kind == "rubato_intro":
        _render_intro(buf, chords, low_midi, rng)

    base_pattern = [0.0, 1.0, 2.0, 3.0]                 # 1 onset/beat
    busy_pattern = list(np.arange(0.0, 3.0, 0.25))      # 12 onsets/bar = 3/beat

    for b in range(bars):
        if b in silent:
            continue
        chord = chords[b % 4]
        tones = voice_chord(chord, low_midi=low_midi)
        bpm_bar = float(bpm_beats[b * 4])
        pattern = busy_pattern if b in busy else base_pattern
        for f in pattern:
            onset, vel = _grooved_onset(f, b, bpm_bar, beat_times, offset,
                                        "pluck", 0.6, rng)
            _strum(buf, tones, onset, vel, rng)
        if include_bass:
            bass_midi = 36 + (chord.root % 12)
            for f, base_vel in ([(0.0, 0.85), (2.0, 0.7)] if b % 2 else [(0.0, 0.85)]):
                onset, vel = _grooved_onset(f, b, bpm_bar, beat_times, offset,
                                            "bass", base_vel, rng)
                y = render_note("808sub", sr=SR, dur_s=1.75 * 60.0 / bpm_bar,
                                freq_hz=midi_to_hz(bass_midi), velocity=vel,
                                rng=rng, params={})
                _add(buf, y, onset)

    # enforce silence: palm-mute fade into each silent span, exact zeros inside
    if silent:
        run = sorted(silent)
        spans = [[run[0], run[0]]]
        for b in run[1:]:
            if b == spans[-1][1] + 1:
                spans[-1][1] = b
            else:
                spans.append([b, b])
        fade_n = int(round(_FADE_S * SR))
        for b0, b1 in spans:
            s0 = int(round((offset + beat_times[4 * b0]) * SR))
            s1 = min(int(round((offset + beat_times[4 * (b1 + 1)]) * SR)), buf.shape[0])
            a = max(0, s0 - fade_n)
            if s0 > a:
                u = np.linspace(0.0, 1.0, s0 - a, endpoint=False)
                buf[a:s0] *= 0.5 * (1.0 + np.cos(np.pi * u))
            buf[s0:s1] = 0.0

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 0.0:
        buf *= 0.85 / peak

    n_beats = bars * 4
    truth = {
        "bpm_curve": [(float(offset + beat_times[i]), float(bpm_beats[i]))
                      for i in range(n_beats)]
                     + [(float(offset + beat_times[-1]), float(bpm_beats[-1]))],
        "bar_times_s": [float(offset + beat_times[4 * b]) for b in range(bars)],
        "silent_bars": sorted(silent),
        "key": key,
        "busy_bars": sorted(busy),
        "intro_end_s": offset,
        "bars": bars,
        "bpm_base": float(bpm),
        "sr": SR,
    }
    return buf.astype(np.float32), truth


def make_human(kind: str, *, bpm: float = 96, bars: int = 24,
               key: str = "A minor", seed: int = 1) -> np.ndarray:
    """Audio only; see make_human_with_truth."""
    return make_human_with_truth(kind, bpm=bpm, bars=bars, key=key, seed=seed)[0]


# ===========================================================================
# v2: humans who JOIN a machine that is already playing (SPEC-V2 section 6)
# ===========================================================================
#
# ``make_human`` fixtures are the v1 chair: the human owns the clock and the
# machine earns its way in, so their grid IS the truth.  A joiner is the other
# chair.  The machine's grid exists first and is the reference frame; every
# joiner is generated ONTO that grid with its deviation applied, so the
# deviation is a property of the rendered audio and not a label attached to it.
#
# SIGN CONVENTION, stated once because everything downstream depends on it:
# ``mean_offset_ms`` is POSITIVE when the human is AHEAD of (earlier in time
# than) the machine grid.  "Rushing +30ms" therefore means each onset is
# rendered 30ms EARLIER than the machine beat it belongs to.  SPEC-V2 section 6
# words dragging as "-35ms behind", i.e. 35ms LATER; both readings agree with
# this one sign rule.
#
# The deviations are deliberately built to be recoverable by a real onset
# detector, not merely present in the metadata: the strum stagger is 4ms rather
# than the 9ms ``make_human`` uses (a 27ms smear across a triad is comparable
# to the deviation being measured), swing is OFF (a swung 16th moves onsets by
# ~30ms at 120bpm, which is exactly the rushing signal), and the timing jitter
# is a zero-mean 4ms gaussian so the MEAN offset survives while individual
# onsets still land where a human's would.

JOINER_KINDS = (
    "on_grid", "rushing", "dragging", "late_join",
    "wrong_key", "far_key", "takes_over", "drops_out",
)

# nominal signed deviation from the machine grid, ms; POSITIVE = AHEAD/early
_JOINER_OFFSET_MS = {"rushing": 30.0, "dragging": -35.0}
# semitones the human's key sits away from the machine's
_JOINER_TRANSPOSE = {"wrong_key": 1, "far_key": 6}

_LATE_JOIN_BAR = 8          # late_join: silent bars [0, 8), enters on bar 8
_DROPOUT_START_BAR = 8      # drops_out: plays 8 bars ...
_DROPOUT_LEN_BARS = 12      # ... then 12 silent bars
_TAKEOVER_MAX_DENSITY = 4   # takes_over: ramps 1 -> 4 onsets/beat

_JOINER_STAGGER_S = 0.004   # per-string delay inside a strum (see note above)
_JOINER_LOW_MIDI = 57       # A3 — same register as the v1 fixtures
_JOINER_MIN_BARS = {
    "late_join": _LATE_JOIN_BAR + 4,
    "drops_out": _DROPOUT_START_BAR + _DROPOUT_LEN_BARS,
    "takes_over": 8,
}

# Straight, not swung: see the module note above.  humanize_ms is 4.0 (a real
# player's jitter) and stays zero-mean, so ~1 onset/beat over 24 bars gives the
# mean offset a standard error of 4/sqrt(96) = 0.4ms — well inside "a few ms".
_JOINER_GROOVE = GrooveParams(
    swing=0.5, timing_bias_ms={}, humanize_ms=4.0, humanize_vel=0.05,
)

# 808sub with a slow gate release.  The preset's 30ms release (tau 10ms) on a
# still-audible low sine is a step, and a step is broadband: it lands in the
# spectral flux as a full-sized ONSET at the note's END.  Measured in isolation
# — eight 0.875s 808sub notes, one per second — the preset yields SIXTEEN
# detected onsets (one per attack at +11ms, one per release at +885ms) and
# ``release_ms=250`` yields eight.  That matters here because a phantom onset
# 875ms after the beat is a phantom BEAT to anything measuring the human's
# timing.  (Not fixed in ``voices.py``: this is a fixture-side choice about
# what the simulated human plays, and voices.py is not mine to change.)
_JOINER_BASS_PARAMS = {"release_ms": 250.0}


def _check_joiner(kind: str, bars: int, machine_bpm: float) -> None:
    if kind not in JOINER_KINDS:
        raise ValueError(
            f"unknown joiner kind: {kind!r} (have: {list(JOINER_KINDS)})")
    need = _JOINER_MIN_BARS.get(kind, 1)
    if bars < need:
        raise ValueError(f"joiner {kind!r} needs bars >= {need}, got {bars}")
    if machine_bpm <= 0.0:
        raise ValueError(f"machine_bpm must be positive, got {machine_bpm}")


def _joiner_key(kind: str, machine_key: str) -> str:
    """The key the HUMAN plays in, given the machine's."""
    shift = _JOINER_TRANSPOSE.get(kind, 0)
    if shift == 0:
        return machine_key
    mk = parse_key(machine_key)
    hk = Key(tonic=(mk.tonic + shift) % 12, mode=mk.mode)
    return f"{spell(hk.tonic, hk)} {hk.mode}"


def _joiner_silent_bars(kind: str, bars: int) -> list[int]:
    if kind == "late_join":
        return list(range(min(_LATE_JOIN_BAR, bars)))
    if kind == "drops_out":
        return list(range(_DROPOUT_START_BAR,
                          min(bars, _DROPOUT_START_BAR + _DROPOUT_LEN_BARS)))
    return []


def _joiner_density(kind: str, bars: int, silent: set[int]) -> list[float]:
    """Onsets per beat, per bar.  0.0 in silent bars."""
    if kind == "takes_over":
        start = max(1, bars // 4)
        span = max(1, bars // 2)
        # floor(x + 0.5), not round(): python/numpy round half to EVEN, which
        # would collapse the 2.5 rung back down to 2 and flatten the ramp.
        vals = [float(int(np.floor(
            1.0 + (_TAKEOVER_MAX_DENSITY - 1.0)
            * min(1.0, max(0.0, (b - start) / span)) + 0.5)))
            for b in range(bars)]
    else:
        vals = [1.0] * bars
    return [0.0 if b in silent else vals[b] for b in range(bars)]


def _density_pattern(density: float) -> list[float]:
    """Beat fractions within a 4-beat bar for a given onsets/beat."""
    d = int(round(density))
    if d <= 0:
        return []
    return [k / d for k in range(4 * d)]


def _apply_silence_spans(buf: np.ndarray, silent: list[int],
                         bar_times: np.ndarray) -> None:
    """Palm-mute fade into each silent run, exact zeros inside it.

    Deliberately a separate copy of ``make_human_with_truth``'s inline block
    rather than a shared extraction: that block is byte-load-bearing for other
    agents' fixtures and is not mine to move.
    """
    if not silent:
        return
    run = sorted(silent)
    spans = [[run[0], run[0]]]
    for b in run[1:]:
        if b == spans[-1][1] + 1:
            spans[-1][1] = b
        else:
            spans.append([b, b])
    fade_n = int(round(_FADE_S * SR))
    for b0, b1 in spans:
        s0 = int(round(bar_times[b0] * SR))
        s1 = min(int(round(bar_times[b1 + 1] * SR)), buf.shape[0])
        a = max(0, s0 - fade_n)
        if s0 > a:
            u = np.linspace(0.0, 1.0, s0 - a, endpoint=False)
            buf[a:s0] *= 0.5 * (1.0 + np.cos(np.pi * u))
        buf[s0:s1] = 0.0


def make_joiner(kind: str, *, machine_bpm: float = 120.0,
                machine_key: str = "A minor", bars: int = 24,
                seed: int = 1) -> tuple[np.ndarray, dict]:
    """A human joining a machine that is already playing (SPEC-V2 section 6).

    The machine's grid is the reference frame: constant ``machine_bpm``,
    4 beats to the bar, bar 0 starting at t=0, spanning ``bars`` bars.  The
    human is rendered onto that grid with the kind's deviation applied.

    Kinds
    -----
    - ``on_grid``    : in time and in key — nothing to negotiate.
    - ``rushing``    : every onset ~30ms AHEAD of the machine grid.
    - ``dragging``   : every onset ~35ms BEHIND it.
    - ``late_join``  : silent for 8 bars, then enters exactly on the bar-8
                       boundary, on grid and in key.
    - ``wrong_key``  : a semitone above ``machine_key`` (conformable).
    - ``far_key``    : a tritone away (not conformable — the machine should
                       hold rather than guess).
    - ``takes_over`` : density ramps 1 -> 4 onsets/beat across the take.
    - ``drops_out``  : plays 8 bars, then 12 silent bars, then resumes if
                       ``bars`` allows.

    Returns ``(audio, truth)``.  ``truth`` carries the contract keys from
    SPEC-V2 section 6:

    - ``onset_times_s``  : every rendered attack time (s), sorted.  These are
      the ACTUAL times including jitter, not the nominal grid.
    - ``mean_offset_ms`` : measured mean of ``(grid_time - onset_time)``, i.e.
      POSITIVE = the human is ahead of the machine.  This is the realized
      value including jitter, so a test can compare an audio measurement
      against it directly rather than against the nominal.
    - ``key``            : the key the HUMAN is playing (parse_key-compatible).
    - ``join_bar``       : first bar in which the human plays.
    - ``density_curve``  : onsets/beat per bar (len ``bars``), 0.0 when silent.
    - ``silent_bars``    : 0-based indices of exactly-silent bars.
    - ``machine_bpm``    : the machine's tempo.

    Extra (non-contractual) keys: ``machine_key``, ``machine_beat_times_s``,
    ``machine_bar_times_s``, ``grid_times_s`` (the nominal machine-grid time
    each onset was derived from, index-aligned with ``onset_times_s``),
    ``nominal_offset_ms``, ``kind``, ``bars``, ``sr``.

    Deterministic: a pure function of (kind, machine_bpm, machine_key, bars,
    seed) — one ``np.random.default_rng(seed)``, no wall-clock, no globals.
    """
    _check_joiner(kind, bars, machine_bpm)
    rng = np.random.default_rng(seed)

    beat_s = 60.0 / float(machine_bpm)
    n_beats = bars * 4
    machine_beats = np.arange(n_beats + 1, dtype=np.float64) * beat_s
    machine_bars = machine_beats[::4]

    human_key = _joiner_key(kind, machine_key)
    chords = _progression(human_key)
    silent = _joiner_silent_bars(kind, bars)
    silent_set = set(silent)
    density = _joiner_density(kind, bars, silent_set)
    offset_s = _JOINER_OFFSET_MS.get(kind, 0.0) / 1000.0   # + = ahead => earlier

    end_s = float(machine_beats[-1])
    buf = np.zeros(int(round(end_s * SR)), dtype=np.float64)

    onsets: list[float] = []
    grids: list[float] = []

    for b in range(bars):
        if b in silent_set:
            continue
        chord = chords[b % 4]
        tones = voice_chord(chord, low_midi=_JOINER_LOW_MIDI)
        at: dict[float, float] = {}
        for f in _density_pattern(density[b]):
            t_grid = float(machine_bars[b]) + f * beat_s
            t_local, vel = apply_groove(f, role="pluck", bpm=float(machine_bpm),
                                        g=_JOINER_GROOVE, rng=rng, velocity=0.6)
            jitter = t_local - f * beat_s          # groove delta only
            t_on = t_grid + jitter - offset_s
            _strum(buf, tones, t_on, vel, rng, stagger_s=_JOINER_STAGGER_S)
            onsets.append(t_on)
            grids.append(t_grid)
            at[f] = t_on
        # bass: root in octave 2 on beats 1 and 3 — the key evidence the
        # harmony tracker's chroma_low term actually leans on (features.py
        # CHROMA_BASS_*).  Held at 2/bar even under takes_over: it is the
        # STRUMS that ramp, and a 4/beat 808 would just be a drone.
        #
        # It lands on the SAME jittered instant as the strum at that beat, not
        # on an independently jittered one.  Two draws 4ms apart would be one
        # musical event but can be two spectral-flux peaks (the 808's 8ms
        # attack rises after the pluck's), which double-counts the onset and
        # biases any mean-offset measurement made from the audio.  One hand,
        # one attack time.
        bass_midi = 36 + (chord.root % 12)
        for f, base_vel in ((0.0, 0.85), (2.0, 0.7)):
            t_on = at.get(f)
            if t_on is None:
                continue
            vel = float(np.clip(base_vel + rng.normal(0.0, 1.0)
                                * _JOINER_GROOVE.humanize_vel, 0.05, 1.0))
            y = render_note("808sub", sr=SR, dur_s=1.75 * beat_s,
                            freq_hz=midi_to_hz(bass_midi), velocity=vel,
                            rng=rng, params=_JOINER_BASS_PARAMS)
            _add(buf, y, t_on)

    _apply_silence_spans(buf, silent, machine_bars)

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 0.0:
        buf *= 0.85 / peak

    order = np.argsort(np.asarray(onsets, dtype=np.float64)) if onsets \
        else np.zeros(0, dtype=np.int64)
    onset_arr = np.asarray(onsets, dtype=np.float64)[order]
    grid_arr = np.asarray(grids, dtype=np.float64)[order]
    mean_offset_ms = (float(np.mean(grid_arr - onset_arr) * 1000.0)
                      if onset_arr.size else 0.0)
    play_bars = [b for b in range(bars) if b not in silent_set]

    truth = {
        "onset_times_s": [float(t) for t in onset_arr],
        "mean_offset_ms": mean_offset_ms,
        "key": human_key,
        "join_bar": int(play_bars[0]) if play_bars else -1,
        "density_curve": density,
        "silent_bars": silent,
        "machine_bpm": float(machine_bpm),
        # extras
        "machine_key": machine_key,
        "machine_beat_times_s": [float(t) for t in machine_beats[:n_beats]],
        "machine_bar_times_s": [float(t) for t in machine_bars[:bars]],
        "grid_times_s": [float(t) for t in grid_arr],
        "nominal_offset_ms": float(_JOINER_OFFSET_MS.get(kind, 0.0)),
        "kind": kind,
        "bars": bars,
        "sr": SR,
    }
    return buf.astype(np.float32), truth
