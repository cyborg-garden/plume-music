"""The live band's balance: the drums must not bury the rest of the band.

The owner (2026-11-06): "the drums are audible but the synths and stuff get a bit
buried."  Measured offline on DuetEngine, the voices that carry the harmony and the
tune (pad, keys, lead, arp, shimmer) sat 8-10.5 dB (K-weighted loudness) under the
drums in house, lofi, synthwave and trap.  Each style's ``role_gain_db`` now brings
them within a few dB.  Solo voices are rendered from the same seeded take with
every other voice at -300 dB, so the notes are the ones the full mix plays.

Offline only: nothing here opens an audio device.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from plume.duet import _ROLE_GAIN_DB, DuetEngine
from plume.listen.beat import BeatState
from plume.listen.harmony import HarmonyState
from plume.mind import BarPlan

SR = 48000
BARS = 8
CHORDS = ["Am", "F", "C", "G"]
STYLES = ["hiphop", "house", "jazz", "lofi", "midwest", "rnb", "synthwave", "trap"]
#: How far under the drums the harmony and tune may sit.  Before: -10.5 (trap) to -7.8 (lofi).
#: After: -5.0 at worst (house) over this take.
MOST_UNDER_DRUMS_DB = -7.0

# ITU-R BS.1770 K-weighting at 48 kHz.
_K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285],
       [1.0, -1.69065929318241, 0.73248077421585])
_K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _loudness(y):
    z = lfilter(*_K2, lfilter(*_K1, y, axis=0), axis=0)
    return -0.691 + 10.0 * np.log10(float(np.sum(np.mean(z ** 2, axis=0))) + 1e-20)


def _take(style, only):
    eng = DuetEngine(style=style, seed=11, sr=SR, fade_in_bars=0.0, fade_out_bars=0.0)
    roles = [r for r in eng.roles if r != "crash"]
    table = eng._style.setdefault("role_gain_db", {})
    for r in roles:
        table[r] = table.get(r, _ROLE_GAIN_DB.get(r, -6.0)) if r in only(eng, r) else -300.0
    lo, hi = eng._style["bpm_range"]
    bpm = (lo + hi) / 2.0
    iv = 60.0 / bpm
    out = []
    for bar in range(BARS):
        t = bar * 4 * iv
        plan = BarPlan(bar=bar, state="comping", roles={r: 1.0 for r in roles}, gain_db=0.0,
                       density=0.75, register_ceiling_midi=None, reason="test")
        beat = BeatState(t_s=t, bpm=bpm, confidence=0.9, locked=True, bar=bar, beat_in_bar=0,
                         phase_beats=0.0, next_beat_t_s=t + iv, next_downbeat_t_s=t + 4 * iv,
                         beat_interval_s=iv)
        harm = HarmonyState(t_s=t, key="A minor", key_conf=0.9, chord=CHORDS[bar % 4],
                            chord_conf=0.9, register_low_midi=None, register_high_midi=None)
        out.append(eng.render_bar(plan, beat, harm))
    return np.concatenate(out).astype(np.float64)


def _drums(eng, r):
    return {r} if r in eng._grids else set()


def _harmony_and_tune(eng, r):
    return set() if r in eng._grids or r in ("bass", "crackle") else {r}


@pytest.mark.parametrize("style", STYLES)
def test_the_drums_do_not_bury_the_harmony_and_the_tune(style):
    drums = _loudness(_take(style, _drums))
    rest = _loudness(_take(style, _harmony_and_tune))
    assert rest - drums >= MOST_UNDER_DRUMS_DB, f"{style}: {rest - drums:.1f} dB under the drums"


# The midwest 808 under the kick.  Juniper, 2026-11-06: "the synth that was doing bum
# bum pause bum bum sounds a little corny ... it stands out and doesnt blend."  It is
# the only code-made voice in midwest that plays short hits.  It was never tuned (only
# a role called "kick" was), so it boomed at 45 Hz whatever the key, and it sat as loud
# as the bass.

def _rms_db(y):
    return 10.0 * np.log10(float(np.mean(y ** 2)) + 1e-20)


def test_the_midwest_808_plays_the_bass_note():
    eng = DuetEngine(style="midwest", seed=11, sr=SR, fade_in_bars=0.0)
    bpm, iv = 95.0, 60.0 / 95.0
    for bar, chord in enumerate(["G#m", "C#m", "E", "D#"]):
        plan = BarPlan(bar=bar, state="comping", roles={"sub": 1.0, "bass": 1.0},
                       gain_db=0.0, density=0.75, register_ceiling_midi=None, reason="t")
        beat = BeatState(t_s=bar * 4 * iv, bpm=bpm, confidence=0.9, locked=True, bar=bar,
                         beat_in_bar=0, phase_beats=0.0, next_beat_t_s=0.0,
                         next_downbeat_t_s=0.0, beat_interval_s=iv)
        harm = HarmonyState(t_s=0.0, key="G# minor", key_conf=0.9, chord=chord,
                            chord_conf=0.9, register_low_midi=None, register_high_midi=None)
        eng.render_bar(plan, beat, harm)
        subs = {e["midi"] % 12 for e in eng.last_events if e["role"] == "sub"}
        bass = {e["midi"] % 12 for e in eng.last_events if e["role"] == "bass"}
        assert subs and subs <= bass, f"{chord}: 808 on {subs}, bass on {bass}"


def test_the_midwest_808_sits_under_the_bass():
    sub = _rms_db(_take("midwest", lambda eng, r: {"sub"}))
    bass = _rms_db(_take("midwest", lambda eng, r: {"bass"}))
    assert sub <= bass - 4.0, f"808 {sub:.1f} dB, bass {bass:.1f} dB"


def test_an_808_kick_is_not_tuned_to_a_key_nobody_has_heard():
    """lofi, trap and synthwave: over drums alone the 808 was pitched to the fallback
    tonic, a note in a key nobody had heard (and a room mic hears it)."""
    from plume.duet import _low_octave_hz
    for key, want in ((None, None), ("D minor", _low_octave_hz(2))):
        eng = DuetEngine(style="lofi", seed=11, sr=SR, fade_in_bars=0.0)
        iv = 60.0 / 80.0
        plan = BarPlan(bar=0, state="comping", roles={"kick": 1.0}, gain_db=0.0,
                       density=0.75, register_ceiling_midi=None, reason="t")
        beat = BeatState(t_s=0.0, bpm=80.0, confidence=0.9, locked=True, bar=0,
                         beat_in_bar=0, phase_beats=0.0, next_beat_t_s=0.0,
                         next_downbeat_t_s=0.0, beat_interval_s=iv)
        harm = HarmonyState(t_s=0.0, key=key, key_conf=0.9 if key else 0.0, chord=None,
                            chord_conf=0.0, register_low_midi=None, register_high_midi=None)
        eng.render_bar(plan, beat, harm)
        kicks = {e["freq_hz"] for e in eng.last_events if e["role"] == "kick"}
        assert kicks == {want}, key
