"""The recorded kit: what the midwest style's drums are made of."""
import numpy as np
import pytest

from plume import samplekit, voices
from plume.duet import DuetEngine
from plume.styles_lib import load_style


def rng():
    return np.random.default_rng(1)


@pytest.mark.parametrize("voice", samplekit.voice_names())
def test_every_kit_voice_renders_a_hit_near_the_start(voice):
    y = voices.render_note(voice, sr=48000, dur_s=0.1, freq_hz=None, velocity=0.8,
                           rng=rng(), params={})
    assert y.dtype == np.float32 and y.size > 4800
    assert np.flatnonzero(np.abs(y) > 0.1 * np.abs(y).max())[0] < 96, "hit is late"


def test_velocity_changes_the_recording_as_well_as_the_level():
    soft = voices.render_note("kit_snare", sr=48000, dur_s=0.1, freq_hz=None,
                              velocity=0.15, rng=rng(), params={})
    hard = voices.render_note("kit_snare", sr=48000, dur_s=0.1, freq_hz=None,
                              velocity=0.9, rng=rng(), params={})
    assert np.abs(soft).max() < np.abs(hard).max()
    n = min(soft.size, hard.size)
    a, b = soft[:n] / np.abs(soft).max(), hard[:n] / np.abs(hard).max()
    assert np.abs(a - b).mean() > 0.005


def test_a_repeated_hit_does_not_repeat_one_recording():
    r = rng()
    hits = {samplekit.hit("kit_kick", 0.8, r, 48000).tobytes() for _ in range(12)}
    assert len(hits) > 1


def test_same_seed_same_hits():
    assert (samplekit.hit("kit_hat", 0.5, rng(), 48000).tobytes()
            == samplekit.hit("kit_hat", 0.5, rng(), 48000).tobytes())


def test_resampling_keeps_the_length_right():
    y = samplekit.hit("kit_kick", 0.8, rng(), 44100)
    assert y.size == pytest.approx(0.8 * 44100, rel=0.01)


def test_midwest_style_uses_only_recorded_drums_and_is_playable():
    style = load_style("midwest")
    for role in ("kick", "snare", "hat", "openhat"):
        assert style["voices"][role] in samplekit.voice_names(), role
    assert DuetEngine(style="midwest", seed=0, sr=48000) is not None


@pytest.mark.parametrize("voice,midi", [("fingerbass", 45), ("wurli", 60), ("vibes", 79)])
def test_tuned_voices_land_on_the_right_pitch(voice, midi):
    hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    y = voices.render_note(voice, sr=48000, dur_s=0.8, freq_hz=hz, velocity=0.7,
                           rng=rng(), params={}).astype(np.float64)
    seg = y[2400:26400] * np.hanning(24000)
    spec = np.abs(np.fft.rfft(seg, 192000))
    f = np.fft.rfftfreq(192000, 1 / 48000)
    near = (f > hz * 0.94) & (f < hz * 1.06)
    peak = f[near][np.argmax(spec[near])]
    assert abs(12 * np.log2(peak / hz)) < 0.5, (voice, peak, hz)
    assert spec[near].max() > 0.1 * spec.max(), "the note is not the strong partial"


def test_a_note_fades_out_after_its_gate():
    y = voices.render_note("wurli", sr=48000, dur_s=0.2, freq_hz=261.6, velocity=0.7,
                           rng=rng(), params={})
    assert y.size <= int((0.2 + 0.5) * 48000) + 1
    assert abs(y[-1]) < 1e-3


def _melody(seed, bars=32, key="C major", chord="Cmaj7"):
    from plume.mind import BarPlan
    from plume import realtime
    d = DuetEngine(style="midwest", seed=seed, sr=48000, key_fallback=key)
    out = []
    for b in range(bars):
        t = b * 2.4
        plan = BarPlan(bar=b, state="comping", roles={"lead": 1.0}, gain_db=0.0,
                       density=0.6, register_ceiling_midi=None, reason="t")
        d.render_bar(plan, realtime.machine_beat_state(b, t, 100.0),
                     realtime.machine_harmony_state(t, key, chord))
        out.append([e["midi"] for e in d.last_events if e["role"] == "lead"])
    return out


def test_the_melody_moves_rests_and_comes_back():
    bars = _melody(1)
    played = [b for b in bars if b]
    assert len({m for b in played for m in b}) >= 3, "still one repeated note"
    assert all(not bars[i] for i in range(4, 8)), "should sit out to leave room"
    assert not bars[3], "each phrase rests its last bar"
    assert bars[0] == bars[16 - 16] and bars[0] == _melody(1)[0]


def test_melody_stays_in_key_and_in_register():
    for key, pcs in (("C major", {0, 2, 4, 7, 9, 11}), ("A minor", {9, 0, 2, 4, 7, 5})):
        notes = {m for b in _melody(2, key=key) for m in b}
        assert notes and all(72 <= m <= 86 for m in notes)
        assert {m % 12 for m in notes} <= pcs, (key, notes)


def test_a_lift_plays_through_and_echoes_while_a_verse_leaves_room():
    from plume.mind import BarPlan
    from plume import realtime

    def bars(density):
        d = DuetEngine(style="midwest", seed=2, sr=48000, key_fallback="F major")
        out = []
        for b in range(8):
            t = b * 2.4
            plan = BarPlan(bar=b, state="comping", roles={"lead": 1.0}, gain_db=0.0,
                           density=density, register_ceiling_midi=None, reason="t")
            d.render_bar(plan, realtime.machine_beat_state(b, t, 100.0),
                         realtime.machine_harmony_state(t, "F major", "Fmaj7"))
            out.append([e for e in d.last_events if e["role"] == "lead"])
        return out
    verse, lift = bars(0.45), bars(0.85)
    assert not verse[5] and lift[5], "a lift keeps playing where a verse rests"
    assert sum(map(len, lift)) > sum(map(len, verse))
    assert min(e["velocity"] for e in verse[0]) < 0.35, "the echoes are quieter"


def test_shimmer_sits_an_octave_above_the_chord():
    from plume.mind import BarPlan
    from plume import realtime
    d = DuetEngine(style="midwest", seed=2, sr=48000, key_fallback="F major")
    plan = BarPlan(bar=0, state="comping", roles={"shimmer": 1.0, "pad": 1.0},
                   gain_db=0.0, density=0.85, register_ceiling_midi=None, reason="t")
    d.render_bar(plan, realtime.machine_beat_state(0, 0.0, 100.0),
                 realtime.machine_harmony_state(0.0, "F major", "Fmaj7"))
    pad = [e["midi"] for e in d.last_events if e["role"] == "pad"]
    high = [e["midi"] for e in d.last_events if e["role"] == "shimmer"]
    assert len(high) == 2 and min(high) > max(pad) - 12 and max(high) > max(pad)


def test_every_drum_bar_is_the_whole_groove_and_the_808_doubles_the_kick():
    from plume.mind import BarPlan
    from plume import realtime
    d = DuetEngine(style="midwest", seed=2, sr=48000, key_fallback="F major")
    roles = {r: 1.0 for r in ("kick", "sub", "snare", "clap", "hat", "openhat")}
    plan = BarPlan(bar=0, state="comping", roles=roles, gain_db=0.0, density=0.55,
                   register_ceiling_midi=None, reason="t")
    d.render_bar(plan, realtime.machine_beat_state(0, 0.0, 90.0),
                 realtime.machine_harmony_state(0.0, None, None))
    spb = 60.0 / 90.0

    def beats(role):
        return [round(e["onset_s"] / spb * 2) / 2
                for e in d.last_events if e["role"] == role]
    assert beats("kick") == [0.0, 1.5, 2.5]
    assert beats("sub") == beats("kick")
    assert beats("snare") == [2.0] and beats("clap") == [2.0]
    assert len(beats("hat")) == 8


def test_a_slow_count_gets_a_backbeat_and_a_fast_one_stays_half_time():
    from plume.mind import BarPlan
    from plume import realtime

    def snares(bpm):
        d = DuetEngine(style="midwest", seed=2, sr=48000, key_fallback="F major")
        plan = BarPlan(bar=0, state="comping",
                       roles={r: 1.0 for r in ("kick", "snare", "clap", "bass")},
                       gain_db=0.0, density=0.85, register_ceiling_midi=None, reason="t")
        d.render_bar(plan, realtime.machine_beat_state(0, 0.0, bpm),
                     realtime.machine_harmony_state(0.0, "F major", "Fmaj7"))
        spb = 60.0 / bpm
        return ([round(e["onset_s"] / spb) for e in d.last_events
                 if e["role"] == "snare" and e["velocity"] > 0.5],
                [round(e["onset_s"] / spb * 2) / 2 for e in d.last_events if e["role"] == "kick"],
                [round(e["onset_s"] / spb * 2) / 2 for e in d.last_events if e["role"] == "bass"])
    slow, fast = snares(51.0), snares(102.0)
    assert slow[0] == [1, 3] and fast[0] == [2]
    assert slow[1] == [0.0, 2.0, 2.5] == slow[2]   # 1.75 swings and rounds up
    assert fast[1] == [0.0, 1.5, 2.5] == fast[2]


def test_the_melody_walks_her_notes_when_it_has_heard_them():
    from plume.mind import BarPlan
    from plume import realtime
    d = DuetEngine(style="midwest", seed=3, sr=48000, key_fallback="F major")
    d.heard_notes = (5, 0, 7)     # F, C, G; never an off-key note
    played = set()
    for b in range(24):
        plan = BarPlan(bar=b, state="comping", roles={"lead": 1.0}, gain_db=0.0,
                       density=0.85, register_ceiling_midi=None, reason="t")
        d.render_bar(plan, realtime.machine_beat_state(b, b * 2.4, 100.0),
                     realtime.machine_harmony_state(b * 2.4, "F major", "Fmaj7"))
        played |= {e["midi"] % 12 for e in d.last_events if e["role"] == "lead"}
    chord_tones = {5, 9, 0, 4}    # beat-one notes land on the chord
    assert played and played <= {5, 0, 7} | chord_tones
    assert {5, 0, 7} & played
