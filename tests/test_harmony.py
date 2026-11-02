"""Tests for listen/harmony.py — key/chord belief (SPEC-V1 section 3).

Frames are duck-typed (harmony only reads t_s / rms_db / chroma /
centroid_hz), so these tests use a local stub with the exact Frame field set
from SPEC-V1 section 1 and do not depend on listen/features.py build order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from plume.listen.harmony import HarmonyState, HarmonyTracker
from plume.theory import parse_key

HOP_S = 512 / 48000  # frame hop from SPEC-V1 section 1


@dataclass(frozen=True)
class FrameStub:
    t_s: float
    rms_db: float = -20.0
    flux: float = 0.0
    onset: bool = False
    onset_strength: float = 0.0
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    centroid_hz: float = 880.0


def chroma_from_pcs(pcs, noise=None) -> np.ndarray:
    c = np.zeros(12, dtype=np.float64)
    for pc in pcs:
        c[pc % 12] = 1.0
    if noise is not None:
        c = c + noise
    s = c.sum()
    return (c / s).astype(np.float32)


def triad_pcs(root: int, minor: bool = False) -> tuple[int, int, int]:
    third = 3 if minor else 4
    return (root % 12, (root + third) % 12, (root + 7) % 12)


def make_frames(chromas, t0: float = 0.0, rms_db: float = -20.0,
                centroid_hz: float = 880.0) -> list[FrameStub]:
    return [
        FrameStub(t_s=t0 + (i + 1) * HOP_S, rms_db=rms_db, chroma=c,
                  centroid_hz=centroid_hz)
        for i, c in enumerate(chromas)
    ]


def silent_frames(n: int, t0: float = 0.0) -> list[FrameStub]:
    return [
        FrameStub(t_s=t0 + (i + 1) * HOP_S, rms_db=-90.0,
                  chroma=np.zeros(12, dtype=np.float32), centroid_hz=0.0)
        for i in range(n)
    ]


def chord_loop_chromas(chord_specs, seconds_each: float, total_s: float,
                       noise_rng=None, noise_amt: float = 0.0) -> list[np.ndarray]:
    """Cycle (root, minor) chord specs, seconds_each per chord, for total_s."""
    n_total = int(round(total_s / HOP_S))
    n_each = max(1, int(round(seconds_each / HOP_S)))
    out = []
    for i in range(n_total):
        root, minor = chord_specs[(i // n_each) % len(chord_specs)]
        noise = noise_rng.random(12) * noise_amt if noise_rng is not None else None
        out.append(chroma_from_pcs(triad_pcs(root, minor), noise=noise))
    return out


# --------------------------------------------------------------- chord tests


@pytest.mark.parametrize(
    "root,minor,expected", [(0, False, "C"), (9, True, "Am"), (7, False, "G")]
)
def test_pure_triad_chord_within_5_frames(root, minor, expected):
    tr = HarmonyTracker()
    frames = make_frames([chroma_from_pcs(triad_pcs(root, minor))] * 20)
    first_correct = None
    for i, f in enumerate(frames):
        state = tr.process([f])
        if state.chord == expected and first_correct is None:
            first_correct = i
    assert first_correct is not None
    assert first_correct <= 4, f"chord emitted at frame {first_correct}, want <5"
    assert state.chord_conf > 0.5


def test_chord_stable_under_10pct_noise():
    rng = np.random.default_rng(42)
    tr = HarmonyTracker()
    clean = chroma_from_pcs(triad_pcs(0))
    chromas = []
    for _ in range(120):
        noisy = clean + 0.10 * rng.random(12).astype(np.float32)
        chromas.append((noisy / noisy.sum()).astype(np.float32))
    for i, f in enumerate(make_frames(chromas)):
        state = tr.process([f])
        if i >= 6:
            assert state.chord == "C", f"chord unstable at frame {i}: {state.chord}"


def test_silence_chord_none():
    tr = HarmonyTracker()
    state = tr.process(silent_frames(50))
    assert state.chord is None
    assert state.key is None
    assert state.register_low_midi is None
    assert state.register_high_midi is None


def test_voiced_then_silence_clears_chord():
    tr = HarmonyTracker()
    tr.process(make_frames([chroma_from_pcs(triad_pcs(0))] * 90))
    assert tr.process([]).chord == "C"
    state = tr.process(silent_frames(50, t0=90 * HOP_S))
    assert state.chord is None


def test_chord_spelled_flat_in_believed_f_minor():
    tr = HarmonyTracker()
    # establish F minor belief: Fm-weighted natural-minor loop, 10s
    loop = chord_loop_chromas(
        [(5, True), (5, True), (10, True), (0, True)], 0.5, 10.0
    )
    tr.process(make_frames(loop))
    # now a Bb-minor triad: root pc 10 must spell as "Bbm" (flats), not "A#m"
    state = tr.process(
        make_frames([chroma_from_pcs(triad_pcs(10, True))] * 90, t0=10.0)
    )
    assert state.chord == "Bbm"


# ----------------------------------------------------------------- key tests


def test_key_from_f_minor_chord_loop():
    # Fm (x2) - Bbm - Cm: the full F natural-minor pc set, tonic-weighted.
    # SPEC-V1 allows the relative (Ab major) as long as we assert the tonic
    # pitch class is in {5, 8} — documented here: KK correlation on natural
    # minor material can prefer the relative major; both share the pc set.
    tr = HarmonyTracker()
    loop = chord_loop_chromas(
        [(5, True), (5, True), (10, True), (0, True)], 0.5, 12.0
    )
    state = tr.process(make_frames(loop))
    assert state.key is not None
    k = parse_key(state.key)
    assert k.tonic in (5, 8), f"key {state.key}: tonic pc {k.tonic} not F/Ab"
    if k.tonic == 5:
        assert k.mode == "minor"
    else:
        assert k.mode == "major"
    assert 0.0 < state.key_conf <= 1.0


def test_key_hysteresis_no_early_flip_then_decay_flip():
    tr = HarmonyTracker()
    fm_loop = chord_loop_chromas(
        [(5, True), (5, True), (10, True), (0, True)], 0.5, 20.0
    )
    state = tr.process(make_frames(fm_loop))
    assert state.key is not None
    start_tonic = parse_key(state.key).tonic
    assert start_tonic in (5, 8)

    # E major material (maximally distant): E - A - B, E-weighted
    e_specs = [(4, False), (4, False), (9, False), (11, False)]

    # 2.5s of new material — inside the 3s hysteresis window: must NOT flip
    early = chord_loop_chromas(e_specs, 0.5, 2.5)
    state = tr.process(make_frames(early, t0=20.0))
    assert state.key is not None
    assert parse_key(state.key).tonic == start_tonic, "flipped inside hysteresis window"

    # 30s total of new key material — old belief decayed (half-life 15s): flips
    rest = chord_loop_chromas(e_specs, 0.5, 27.5)
    state = tr.process(make_frames(rest, t0=22.5))
    assert state.key is not None
    k = parse_key(state.key)
    # accept E major or its relative C# minor (documented, same pc set)
    assert k.tonic in (4, 1), f"belief did not flip: {state.key}"


# ------------------------------------------------------------- register tests


def test_register_low_vs_high_synthetic_spectra():
    # A2 (midi 45, 110Hz): chroma one-hot pc 9, centroid ~2x fundamental
    lo_tr = HarmonyTracker()
    lo = lo_tr.process(
        make_frames([chroma_from_pcs([9])] * 200, centroid_hz=220.0)
    )
    # A5 (midi 81, 880Hz)
    hi_tr = HarmonyTracker()
    hi = hi_tr.process(
        make_frames([chroma_from_pcs([9])] * 200, centroid_hz=1760.0)
    )
    assert lo.register_low_midi is not None and hi.register_low_midi is not None
    assert lo.register_high_midi < hi.register_low_midi
    assert lo.register_low_midi - 3 <= 45 <= lo.register_high_midi + 3
    assert hi.register_low_midi - 3 <= 81 <= hi.register_high_midi + 3


def test_register_none_after_sustained_silence():
    tr = HarmonyTracker()
    state = tr.process(make_frames([chroma_from_pcs([0, 4, 7])] * 90))
    assert state.register_low_midi is not None
    # > 2s of silence -> register reports None (content no longer active)
    n_silent = int(2.5 / HOP_S)
    state = tr.process(silent_frames(n_silent, t0=90 * HOP_S))
    assert state.register_low_midi is None
    assert state.register_high_midi is None


# ---------------------------------------------------------------- misc tests


def test_determinism():
    rng = np.random.default_rng(7)
    clean = chroma_from_pcs(triad_pcs(2, True))
    chromas = []
    for _ in range(150):
        noisy = clean + 0.1 * rng.random(12).astype(np.float32)
        chromas.append((noisy / noisy.sum()).astype(np.float32))
    frames = make_frames(chromas)

    def run() -> list[HarmonyState]:
        tr = HarmonyTracker()
        return [tr.process([f]) for f in frames]

    a, b = run(), run()
    for sa, sb in zip(a, b):
        assert sa == sb


def test_empty_process_returns_current_state():
    tr = HarmonyTracker()
    s0 = tr.process([])
    assert s0.chord is None and s0.key is None
    tr.process(make_frames([chroma_from_pcs(triad_pcs(0))] * 20))
    s1 = tr.process([])
    s2 = tr.process([])
    assert s1 == s2
    assert s1.chord == "C"


def test_state_time_is_last_frame_time():
    tr = HarmonyTracker()
    frames = make_frames([chroma_from_pcs(triad_pcs(0))] * 10)
    state = tr.process(frames)
    assert state.t_s == pytest.approx(frames[-1].t_s)


# ------------------------------------------- key belief on real audio (v1.4)
#
# Everything above deliberately uses chroma stubs.  The v1.4 key amendments
# are about information the stubs cannot carry — REGISTER (Frame.chroma_low)
# and SUSTAIN — so this section drives the real feature extractor with
# synthesized audio.  That is the point: the P1 defect was invisible to
# pitch-class stubs by construction, because a stub chroma has no register.

import sys                                                    # noqa: E402
from pathlib import Path                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume.listen.features import FeatureExtractor             # noqa: E402
from plume.listen.harmony import (                             # noqa: E402
    _KEY_CONF_FLOOR, _COLL_DIST, _key_name, _switch_margin,
)

SR = 48000


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def sine_audio(midis, secs=8.0, amps=None, gain_db=0.0, sr=SR):
    """Steady sum-of-sines at the given MIDI pitches, peak ~0.25 before gain."""
    t = np.arange(int(secs * sr)) / sr
    amps = amps if amps is not None else [1.0] * len(midis)
    x = sum(a * np.sin(2.0 * np.pi * _hz(m) * t) for m, a in zip(midis, amps))
    x = 0.25 * x / np.max(np.abs(x)) * 10.0 ** (gain_db / 20.0)
    return x.astype(np.float32)


def click_audio(bpm=120.0, bars=17, peak_db=-21.0, hz=1000.0, decay=90.0,
                click_s=0.03, sr=SR):
    """An UNPITCHED-in-the-musical-sense input: sparse percussive hits."""
    spb = 60.0 / bpm
    x = np.zeros(int(bars * 4 * spb * sr), dtype=np.float32)
    n = int(click_s * sr)
    t = np.arange(n) / sr
    w = (np.exp(-t * decay) * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)
    amp = 10.0 ** (peak_db / 20.0)
    for b in range(bars * 4):
        i = int(round(b * spb * sr))
        x[i:i + n] += (amp * (1.0 if b % 4 == 0 else 0.6) * w).astype(np.float32)
    return x


def track_audio(x, sr=SR):
    """Run the full front end and return (final state, [(t, key, conf), ...])."""
    fe, tr = FeatureExtractor(sr=sr), HarmonyTracker()
    trace, last, state = [], None, None
    for i in range(0, x.size, 8192):
        for f in fe.process(x[i:i + 8192]):
            state = tr.process([f])
            if state.key != last:
                trace.append((state.t_s, state.key, state.key_conf))
                last = state.key
    return state, trace


# ---- P1: the key follows the bass, not the pitch-class mass

def test_rootless_extended_voicing_is_keyed_off_its_bass_note():
    """THE P1 MINIMAL REPRO, made permanent.

    C2 + E3 G3 B3 D4 is a Cmaj9.  Before v1.4 the chord came out right ("C")
    and the KEY came out a fifth away ("G major"), because chroma is flat
    across registers and the collection {C,D,E,G,B} scores higher against the
    G-major profile than the C-major one.  Doubling the C2 loudly used to be
    the only way to get "C major" — that difference is the whole bug.
    """
    state, _ = track_audio(sine_audio([36, 52, 55, 59, 62]))
    assert state.chord == "C"
    assert state.key is not None
    assert parse_key(state.key) == parse_key("C major"), (
        f"rootless Cmaj9 keyed as {state.key!r}, want C major")


def test_a_loud_root_does_not_change_the_answer():
    """The guard against over-correcting: the reading that was already right
    must stay right.  Quadrupling the C2's amplitude is the same harmony."""
    state, _ = track_audio(
        sine_audio([36, 52, 55, 59, 62], amps=[4.0, 1.0, 1.0, 1.0, 1.0]))
    assert parse_key(state.key) == parse_key("C major")


@pytest.mark.parametrize("midi,pc", [(57, 9), (45, 9), (69, 9), (60, 0)])
def test_a_pure_tone_keys_to_its_own_pitch_class(midi, pc):
    """220Hz -> A.  A lone tone names no mode in particular, but it must never
    name a tonic other than itself (the register weighting must not invent a
    root a fifth or a third away)."""
    state, _ = track_audio(sine_audio([midi]))
    assert state.key is not None
    assert parse_key(state.key).tonic == pc, f"{midi} -> {state.key!r}"


def test_a_bass_root_under_an_upper_triad_beats_the_triad():
    """F2 under a C-major triad is F, not C: the bass decides the tonic."""
    state, _ = track_audio(sine_audio([41, 60, 64, 67]))
    assert state.key is not None
    assert parse_key(state.key).tonic == 5, (
        f"F2 + C-major triad keyed as {state.key!r}, want an F tonic")


# ---- P2: unpitched material must not produce a confident key

@pytest.mark.parametrize("peak_db", [-21.0, -41.0])
def test_a_click_track_never_names_a_key(peak_db):
    """A bare click track used to name a key at conf ~0.65 and the sideman
    comped on it.  A 1kHz click IS a tone in any single frame — what separates
    it from music is SUSTAIN, so the gate is on tonal duty, not on spectrum.
    """
    state, trace = track_audio(click_audio(peak_db=peak_db))
    assert state.key is None, f"click named {state.key!r} at {state.key_conf}"
    assert state.key_conf == 0.0
    named = [e for e in trace if e[1] is not None]
    assert not named, f"click named a key mid-take: {named[:3]}"


def test_a_kick_only_pattern_never_names_a_key():
    """Kick drums DO put energy in the bass band, so the low-register weighting
    must not become a way for percussion to vote on the key."""
    state, trace = track_audio(
        click_audio(hz=55.0, decay=14.0, click_s=0.18, peak_db=-12.0))
    assert state.key is None, f"kick pattern named {state.key!r}"
    assert not [e for e in trace if e[1] is not None]


def test_sustained_tone_at_the_same_level_as_the_clicks_does_name_a_key():
    """The other half of the duty gate: it must not be a level gate in
    disguise.  Same peak level as the click track, held instead of struck."""
    state, _ = track_audio(sine_audio([45, 52, 57], secs=14.0, gain_db=-21.0))
    assert state.key is not None
    assert state.key_conf >= _KEY_CONF_FLOOR


# ---- P2: early belief is provisional, and says so

def test_key_confidence_starts_low_and_grows_with_evidence():
    """Belief used to be asserted at full confidence on ~2s of audio."""
    x = sine_audio([36, 52, 55, 59, 62], secs=30.0)
    state, trace = track_audio(x)
    first_t, first_key, first_conf = [e for e in trace if e[1]][0]
    assert first_t >= 1.0, "a key was named before there was any evidence"
    assert first_conf < 0.5, f"first declaration asserted at conf {first_conf}"
    assert state.key_conf > first_conf + 0.15, (
        f"confidence never grew: {first_conf:.2f} -> {state.key_conf:.2f}")


def test_first_key_declaration_time_depends_on_the_material():
    """The old symptom was a CONSTANT: the first ``key_change`` fired at
    t=2.112s on every input alike, because the only gate was an evidence
    clock that every voiced frame advances at the same rate.  Gating on ramped
    confidence instead makes the moment a property of how strongly the input
    states its key.  Measured spread before: 0.085s across these three inputs
    (2.112-2.197); after: ~1s."""
    times = []
    for midis in ([36, 48, 52, 55], [36, 52, 55, 59, 62], [45, 52, 57],
                  [41, 60, 64, 67]):
        _, trace = track_audio(sine_audio(midis, secs=12.0))
        times.append([e for e in trace if e[1]][0][0])
    assert max(times) - min(times) > 0.5, (
        f"first-declaration time is still effectively a constant: "
        f"{[round(t, 3) for t in times]}")


@pytest.mark.parametrize("gain_db", [0.0, -20.0, -40.0])
def test_key_belief_is_gain_invariant(gain_db):
    """The same take recorded quieter must reach the same key."""
    state, _ = track_audio(
        sine_audio([36, 52, 55, 59, 62], secs=12.0, gain_db=gain_db))
    assert parse_key(state.key) == parse_key("C major"), (
        f"{gain_db} dB -> {state.key!r}")


# ---- P2: closely-related-key stickiness

def test_collection_distance_grades_the_switch_margin():
    c_major, a_minor = 0, 12 + 9
    g_major, f_major = 7, 5
    e_major = 4
    assert _COLL_DIST[c_major, a_minor] == 0          # relative: same 7 pcs
    assert _COLL_DIST[c_major, g_major] == 2          # one accidental
    assert _COLL_DIST[c_major, f_major] == 2
    assert _COLL_DIST[c_major, e_major] > 2           # a real modulation
    assert (_switch_margin(c_major, a_minor)
            > _switch_margin(c_major, g_major)
            > _switch_margin(c_major, e_major))


def test_zero_collection_distance_means_exactly_the_relative_pairs():
    """The v1.2 relative-key rule is now the distance==0 case of the v1.4 one.
    That is only a faithful generalisation if the two agree exactly, so the
    equivalence is asserted rather than assumed (the explicit `_is_relative`
    helper was removed when `_COLL_DIST` subsumed it)."""
    for a in range(24):
        for b in range(24):
            if a == b:
                continue
            same_mode = (a < 12) == (b < 12)
            if same_mode:
                relative = False
            else:
                maj, mnr = (a, b - 12) if a < 12 else (b, a - 12)
                relative = (maj + 9) % 12 == mnr
            assert bool(_COLL_DIST[a, b] == 0) is relative, (
                f"{_key_name(a)} vs {_key_name(b)}: dist {_COLL_DIST[a, b]}, "
                f"relative={relative}")


def test_a_neighbour_key_is_sticky_but_a_real_modulation_still_lands():
    """Both halves of the rule, at durations that actually discriminate.

    On this scenario the belief follows the move to F major after ~28s of F
    material under the ordinary 0.05 margin and after ~49s under the
    neighbour margin, so 35s separates the two and 65s is past both.  The
    second half is the important one: because _KEY_NEIGHBOR_STICKY is set
    below the ~0.19 correlation gap that saturated single-key material can
    produce, stickiness can never become a permanent block.
    """
    f_specs = [(5, False), (10, False), (5, False), (0, False)]
    tr = HarmonyTracker()
    c_loop = chord_loop_chromas(
        [(0, False), (5, False), (7, False), (0, False)], 0.5, 20.0)
    tr.process(make_frames(c_loop))
    assert parse_key(tr.process([]).key).tonic in (0, 9)

    state = tr.process(
        make_frames(chord_loop_chromas(f_specs, 0.5, 35.0), t0=20.0))
    assert parse_key(state.key).tonic in (0, 9), (
        f"a subdominant-heavy passage moved the key to {state.key!r}")

    state = tr.process(
        make_frames(chord_loop_chromas(f_specs, 0.5, 30.0), t0=55.0))
    assert parse_key(state.key).tonic in (5, 2), (
        f"a 65-second modulation never landed: {state.key!r}")
