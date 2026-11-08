"""Playing with nothing named: the key and the style come from the take.

The rule is a guess by design, and these pin the guess, not whether it is a good one.
Nothing here has heard a real guitar.  Everything live runs on the fake device.
"""
import argparse
import json

import numpy as np
import pytest
from conftest import needs_laptop_speed  # noqa: E402
from conftest import fake_audio, patched_module, playhead_rate, short_socket_path

from plume import cli, governor, realtime
from plume import roles as roles_mod
from plume.bandentry import (DETECT_KEY_FALLBACK_BARS, FOLK_CONFIDENCE, BandEntry,
                              FolkEar, StylePick, detect_style)
from plume.listen.harmony import HarmonyState
from plume.watch import Listener

SR = realtime.SR
RIG = [
    {"name": "Laptop Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Laptop Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4},
]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


# ------------------------------------------------------------------ the route

def _args(**kw):
    return argparse.Namespace(**{"prompt": None, "session": None, "detect": False, **kw})


def test_detect_needs_neither_a_prompt_nor_a_session():
    assert cli._session_or_prompt(_args(detect=True), what="lead") == (None, None)


def test_detect_ignores_a_session_lying_in_the_folder(tmp_path, monkeypatch):
    (tmp_path / "stray.plume.json").write_text("{not even a session")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PLUME_SESSION", str(tmp_path / "stray.plume.json"))
    assert cli._session_or_prompt(_args(detect=True), what="lead") == (None, None)
    # and without it that file is exactly what gets picked up
    with pytest.raises(cli.CliError, match="session file invalid"):
        cli._session_or_prompt(_args(), what="lead")


def test_no_prompt_and_no_detect_is_still_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PLUME_SESSION", raising=False)
    with pytest.raises(cli.CliError, match="no session found"):
        cli._session_or_prompt(_args(), what="lead")


def test_detect_keeps_a_prompt_when_there_is_one():
    assert cli._session_or_prompt(_args(detect=True, prompt="in F major"),
                                  what="lead") == (None, "in F major")


@pytest.mark.parametrize("argv,says", [
    (["lead", "--detect"], "--tap-start"),
    (["lead", "--detect", "--tap-start", "--session", "x.plume.json"], "--session"),
    (["lead", "--detect", "--tap-start", "--style", "trap"], "--style"),
])
def test_detect_refuses_what_it_cannot_honour(argv, says, capsys, monkeypatch):
    # Before any hardware: if a refusal ever got this far the test would fail on it.
    monkeypatch.setattr(realtime, "lead", lambda **kw: pytest.fail("reached the transport"))
    monkeypatch.setattr(realtime, "check_device", lambda *a: pytest.fail("probed a device"))
    assert cli.main(argv) != 0
    assert says in capsys.readouterr().err


def test_the_cli_hands_detect_to_the_transport_with_nothing_to_compose_from(monkeypatch, tmp_path):
    (tmp_path / "stray.plume.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda *a: None)
    assert cli.main(["lead", "--tap-start", "--detect"]) == 0
    assert seen["detect"] is True and seen["tap_start"] is True
    assert seen["session"] is None and seen["prompt"] is None


def test_a_take_without_detect_does_not_pass_it_on(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda *a: None)
    assert cli.main(["lead", "--tap-start", "lofi"]) == 0
    assert "detect" not in seen and seen["style"] == "lofi"


@pytest.mark.parametrize("kw", [{"session": object(), "tap_start": True}, {"prompt": "lofi"}])
def test_the_transport_refuses_too_before_touching_a_device(kw, monkeypatch):
    monkeypatch.setattr(realtime, "_import_sounddevice", lambda: pytest.fail("reached audio"))
    with pytest.raises(realtime.RealtimeError, match="--detect"):
        realtime.lead(detect=True, **kw)


# ------------------------------------------------------------------- the rule

@pytest.mark.parametrize("bpm,mode,busy,style", [
    (60.0, None, None, "lofi"),
    (60.0, "major", 0.2, "lofi"),
    (59.9, "minor", 0.2, "lofi"),
    (50.0, "minor", None, "lofi"),
    (60.0, "minor", 0.2, "rnb"),
    (79.9, "minor", None, "rnb"),
    (79.9, "minor", 3.0, "lofi"),
    (80.0, None, None, "lofi"),
    (85.0, "major", 0.99, "lofi"),
    (85.0, "major", 1.0, "midwest"),
    (85.0, "minor", 0.2, "hiphop"),
    (89.9, "minor", 2.0, "midwest"),
    (90.0, None, None, "midwest"),
    (95.0, "major", 0.2, "midwest"),
    (95.0, "minor", 0.2, "hiphop"),
    (95.0, "minor", 1.0, "midwest"),
    (100.0, "major", 0.2, "midwest"),
    (100.0, "minor", 0.2, "synthwave"),
    (100.0, "minor", None, "synthwave"),
    (100.0, "minor", 1.0, "midwest"),
    (117.9, "minor", 0.0, "synthwave"),
    (118.0, "minor", 0.0, "house"),
    (124.0, None, None, "midwest"),
    (124.0, "minor", 1.0, "midwest"),
    (129.9, "minor", None, "house"),
    (129.9, "major", 0.0, "midwest"),
    (130.0, None, None, "trap"),
    (140.0, "minor", 0.49, "trap"),
    (140.0, "minor", 0.5, "midwest"),
    (150.0, "major", 0.1, "trap"),
    (150.1, "major", 0.1, "jazz"),
    (170.0, "minor", 0.49, "jazz"),
    (220.0, "major", 0.2, "jazz"),
    # Fast and not known to be sparse is the older answer: nothing there says jazz.
    (170.0, None, None, "midwest"),
    (170.0, "major", None, "midwest"),
    (170.0, "minor", 0.5, "midwest"),
    (170.0, "major", 2.0, "midwest"),
    (170.0, "minor", 3.0, "midwest"),
    (220.1, "major", 0.2, "midwest"),
])
def test_the_rule(bpm, mode, busy, style):
    assert detect_style(bpm, mode, busy) == style


def test_a_room_mic_never_says_how_busy_anyone_is():
    pick = StylePick(trust_busy=False)
    for _ in range(8):
        pick.heard_bar(16)
    assert pick.busy is None
    assert pick.at_entry(85.0, "C major") == "lofi"
    assert pick.at_entry(140.0, "C major") == "trap"


def test_a_direct_input_does():
    pick = StylePick(trust_busy=True)
    assert pick.busy is None          # no bars heard: not the same as not busy
    pick.heard_bar(8)
    pick.heard_bar(0)
    assert pick.busy == pytest.approx(1.0)
    assert pick.at_entry(85.0, "C major") == "midwest"
    assert pick.at_entry(140.0, "C major") == "midwest"


def test_a_key_she_named_informs_the_first_pick():
    assert StylePick().at_count(100.0) == "midwest"
    assert StylePick(key="A minor").at_count(100.0) == "synthwave"


def _every_guess():
    seen = {}
    for bpm in np.arange(40.0, 260.0, 0.5):
        for mode in (None, "major", "minor"):
            for busy in (None, 0.2, 0.7, 2.0):
                for folk in (None, 0.5, 0.9):
                    seen.setdefault(detect_style(float(bpm), mode, busy, folk=folk),
                                    set()).add(float(bpm))
    return seen


def test_every_bundled_style_can_be_guessed_at_somewhere_in_its_own_range():
    """Fair means reachable: no style is shipped that detect can never pick."""
    from plume.styles_lib import list_styles, load_style
    seen = _every_guess()
    assert set(seen) == set(list_styles())
    for style, tempi in seen.items():
        lo, hi = load_style(style)["bpm_range"]
        assert any(lo <= t <= hi for t in tempi), style


@pytest.mark.parametrize("style", ["jazz", "house", "hiphop", "rnb", "folk"])
def test_a_new_genre_is_never_guessed_outside_its_own_range(style):
    """lofi and midwest stay the catch-alls past every range, as they always were.  The
    four new styles are claims about a tempo, so they never answer outside it."""
    from plume.styles_lib import load_style
    lo, hi = load_style(style)["bpm_range"]
    tempi = _every_guess()[style]
    assert lo <= min(tempi) and max(tempi) <= hi


def test_a_room_mic_never_guesses_jazz():
    """Jazz needs a player known to be sparse, and a room mic cannot know that."""
    pick = StylePick(trust_busy=False)
    pick.heard_bar(0)
    assert pick.at_count(170.0) == "midwest"
    assert pick.at_entry(170.0, "C major") == "midwest"


def test_a_fast_player_on_a_direct_input_is_not_handed_jazz_for_being_fast():
    pick = StylePick(trust_busy=True)
    pick.heard_bar(16)
    assert pick.at_entry(170.0, "A minor") == "midwest"


def test_a_settled_pick_says_so():
    pick = StylePick()
    assert not pick.settled
    pick.at_count(100.0)
    assert not pick.settled
    pick.at_entry(100.0, "A minor")
    assert pick.settled


# --------------------------------------------------------------------- folk
#
# Folk is a claim about how the guitar is played, never about tempo alone.  The one
# signal measured to separate is the strum: a pick dragged across the strings keeps the
# attack up for 30 ms or more, where a picked note, a drum or a synth is over in one or
# two hops.  Measured on synthetic strums and picking (tests/folk_fixtures.py) and, locally
# and never committed, on 15 real direct-input takes: those takes' strum share was
# 0.09-0.29, a slowly raked strum's 0.49-0.64.

def test_folk_answers_only_above_its_confidence_and_inside_its_range():
    assert detect_style(96.0, "major", 0.5, folk=FOLK_CONFIDENCE) == "folk"
    assert detect_style(96.0, "major", 0.5, folk=FOLK_CONFIDENCE - 0.01) == "midwest"
    assert detect_style(96.0, "major", 0.5) == "midwest"
    assert detect_style(150.0, "major", 0.1, folk=1.0) == "trap"
    assert detect_style(59.0, "major", 0.1, folk=1.0) == "lofi"


def _ear(audio, *, bars=None):
    """``audio`` through the real feature extractor, a bar at a time, into a FolkEar."""
    from plume.listen.features import FeatureExtractor
    fe = FeatureExtractor(sr=SR)
    frames = []
    for i in range(0, len(audio), 4800):
        frames.extend(fe.process(audio[i:i + 4800]))
    ear = FolkEar()
    per = max(1, len(frames) // (bars or 12))
    for i in range(0, len(frames), per):
        ear.bar(frames[i:i + per])
    return ear


@pytest.fixture(scope="module")
def takes():
    import folk_fixtures as ff
    return {
        "strum": ff.make_take("strum"),
        "strum_minor": ff.make_take("strum", key="E minor", numerals=("i", "VII", "VI", "VII")),
        "strum_slow": ff.make_take("strum", bpm=66.0),
        "strum_fast_rake": ff.make_take("strum", rake_ms=(8.0, 18.0)),
        "pick": ff.make_take("pick"),
    }


def test_a_slowly_strummed_major_guitar_is_heard_as_folk(takes):
    ear = _ear(takes["strum"])
    assert ear.confidence("major") >= FOLK_CONFIDENCE
    assert "strummed" in ear.why("major")


def test_a_strummed_minor_folk_tune_needs_the_acoustic_sound_as_well(takes):
    ear = _ear(takes["strum_minor"])
    assert ear.acoustic
    assert ear.confidence("minor") >= FOLK_CONFIDENCE


@pytest.mark.parametrize("name", ["pick", "strum_fast_rake"])
def test_what_it_cannot_hear_as_a_strum_is_not_folk(takes, name):
    """Fingerpicking, however folky, is not claimed: nothing measured tells it from a
    twinkly midwest line.  Nor is a strum so quick it lands inside two hops."""
    assert _ear(takes[name]).confidence("major") < FOLK_CONFIDENCE


def test_too_little_heard_is_no_evidence(takes):
    ear = _ear(takes["strum"][: SR * 3], bars=2)
    assert ear.confidence("major") is None


def test_a_direct_input_strumming_gets_folk_at_the_entry(takes):
    from plume.listen.features import FeatureExtractor
    fe = FeatureExtractor(sr=SR)
    frames = []
    for i in range(0, len(takes["strum"]), 4800):
        frames.extend(fe.process(takes["strum"][i:i + 4800]))
    pick = StylePick(trust_busy=True, hear_technique=True)
    per = len(frames) // 12
    for i in range(12):
        bar = frames[i * per:(i + 1) * per]
        pick.heard_bar(sum(f.onset for f in bar), frames=bar)
    assert pick.at_count(96.0) == "midwest"      # nothing heard yet at the count
    assert pick.at_entry(96.0, "G major") == "folk"


def test_a_room_mic_never_guesses_folk(takes):
    """It hears plume's own click and drums as well as her, so her strum is not hers alone."""
    pick = StylePick(trust_busy=False, hear_technique=False)
    from plume.listen.features import FeatureExtractor
    fe = FeatureExtractor(sr=SR)
    frames = fe.process(takes["strum"])
    pick.heard_bar(40, frames=frames)
    assert pick.at_entry(96.0, "G major") == "midwest"


def test_the_folk_ear_keeps_a_bounded_memory(takes):
    from plume.listen.features import FeatureExtractor
    fe = FeatureExtractor(sr=SR)
    frames = fe.process(takes["strum"])
    ear = FolkEar()
    for _ in range(50):
        ear.bar(frames[:400])
    assert ear.memory_items() < 200


def test_folk_s_range_is_the_style_s_own():
    from plume.bandentry import FOLK_BPM
    from plume.styles_lib import load_style
    assert list(FOLK_BPM) == load_style("folk")["bpm_range"]


# ---------------------------------------------------- a key that never settles

def _heard(key):
    return HarmonyState(t_s=0.0, key=key, key_conf=0.5, chord=None, chord_conf=0.0,
                        register_low_midi=None, register_high_midi=None)


def _wandering(entry, bars, *, keys=("C major", "G major", "D major")):
    """Never the same key two bars running, so the governor never commits."""
    out = []
    for bar in range(bars):
        out.append(entry.update(bar, _heard(keys[bar % len(keys)]), True))
    return out


def test_without_a_fallback_a_wandering_player_stays_on_drums():
    assert all(s.stage == "drums" for s in _wandering(BandEntry("lofi"), 60))


def test_after_sixteen_heard_bars_the_latest_key_brings_the_band_in():
    states = _wandering(BandEntry("lofi", key_fallback_bars=DETECT_KEY_FALLBACK_BARS), 20)
    assert DETECT_KEY_FALLBACK_BARS == 16
    assert all(s.stage == "drums" for s in states[:15])
    assert states[15].stage == "band"
    # bar 15 of the wander, and the band has a chord to play at once
    assert states[15].key == "C major" and states[15].chord is not None
    assert states[16].key == "C major"


def test_bars_she_sat_out_do_not_count_towards_it():
    entry = BandEntry("lofi", key_fallback_bars=16)
    for bar in range(40):
        state = entry.update(bar, _heard("C major"), False)
    assert state.stage == "drums"


def test_nothing_to_fall_back_on_means_it_keeps_waiting():
    entry = BandEntry("lofi", key_fallback_bars=16)
    for bar in range(40):
        state = entry.update(bar, _heard(None), True)
    assert state.stage == "drums"


def test_a_key_that_settles_is_not_overruled_by_the_fallback():
    entry = BandEntry("lofi", key_fallback_bars=16)
    for bar in range(12):
        state = entry.update(bar, _heard("A minor"), True)
    assert state.key == "A minor"
    for bar in range(12, 18):         # past sixteen heard, short of a real key change
        state = entry.update(bar, _heard("G major"), True)
    assert state.key == "A minor"


# ------------------------------------------------------------------ restyling

def test_a_restyled_entry_plays_the_new_progression_on_the_same_cycle():
    entry = BandEntry("midwest", known_key="A minor")
    for bar in range(10):
        state = entry.update(bar, _heard(None), bar >= 2)
    assert state.stage == "band"
    midwest = governor.progression_symbols("midwest", "A minor")
    synth = governor.progression_symbols("synthwave", "A minor")
    assert midwest != synth, "these two have to differ for this to prove anything"
    assert state.chord in midwest
    entry.restyle("synthwave")
    chord = entry.chord_at(9, None)
    assert chord == synth[(9 - entry._gov._commit_bar) % len(synth)]
    assert entry.update(10, _heard(None), True).chord == synth[
        (10 - entry._gov._commit_bar) % len(synth)]


def test_restyling_before_a_key_is_known_commits_to_nothing():
    entry = BandEntry("midwest")
    entry.restyle("trap")
    assert entry.update(0, _heard("A minor"), False).key is None
    assert entry._gov.committed_key is None


# ------------------------------------------------------- the real transport, no device

def _strummed_take(bpm, *, seconds=70.0, strums_per_beat=2.0, busier_from_s=None):
    """Four knocks, then C major struck ``strums_per_beat`` a beat until ``seconds``,
    and six a beat from ``busier_from_s``."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int((seconds + 20.0) * SR)).astype(np.float32) * 3e-4
    period = 60.0 / bpm
    for i in range(4):
        at = int((2.0 + i * period) * SR)
        n = int(0.004 * SR)
        x[at:at + n] += rng.standard_normal(n).astype(np.float32) * 0.8
    t = np.arange(int(period * SR)) / SR
    env = np.exp(-t * 6.0)
    chord = sum(np.sin(2 * np.pi * f * t) / (1 + 0.3 * j)
                for j, f in enumerate((130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 523.25)))
    chord = (chord * env * 0.1).astype(np.float32)
    at = 2.0 + 12 * period      # drums alone for two bars first
    while at + 2.0 < seconds:
        i = int(at * SR)
        x[i:i + chord.shape[0]] += chord
        busier = busier_from_s is not None and at >= busier_from_s
        at += period / (6.0 if busier else strums_per_beat)
    return x


def _rows(rec):
    rows = [json.loads(line) for line in (rec / "decisions.ndjson").read_text().splitlines()]
    return [r for r in rows if isinstance(r.get("bar"), int)]


def _detect_take(tmp_path, human, *, name="rec", prompt=None, rate_x=12.0, **kw):
    route = {"direct": True, "input_device": "audio interface", "input_channel": 1}
    route.update(kw.pop("route", {}))
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=rate_x)):
        realtime.lead(prompt=prompt, seed=0, tap_start=True, detect=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / name),
                      status=False, socket_path=short_socket_path(name), **route, **kw)
    return tmp_path / name


def test_a_busy_player_at_85_starts_lofi_and_the_band_arrives_as_midwest(tmp_path, consent):
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.5)
    rows = _rows(rec)
    drums = [r for r in rows if r["entry"] == "drums"]
    band = [r for r in rows if r["entry"] != "drums"]
    assert drums and band, "the band never came in"
    assert {r["style"] for r in drums} == {"lofi"}
    assert {r["style"] for r in band} == {"midwest"}
    assert all(r["style_guess"] and r["key_guess"] for r in rows)
    # midwest's own instruments: its drums and bass from the first bar it is in, its
    # keys once the chords join (bandentry.CHORDS_DELAY_BARS later)
    from plume.bandentry import CHORDS_DELAY_BARS
    assert "bass" in band[0]["roles"] and "keys" not in drums[-1]["roles"]
    assert "keys" in band[CHORDS_DELAY_BARS]["roles"]
    # and in a key it heard
    assert band[0]["key"] in ("C major", "A minor")
    prog = governor.progression_symbols("midwest", band[0]["key"])
    assert band[0]["chord"] in prog and band[1]["chord"] in prog
    # There is no bar gap: every bar follows the last, and none of them went quiet.
    ts = [r["t_s"] for r in rows]
    bar_s = 240.0 / rows[0]["bpm"]
    assert all(b - a == pytest.approx(bar_s, abs=0.01) for a, b in zip(ts, ts[1:]))
    import soundfile as sf
    machine, _ = sf.read(str(rec / "machine.wav"), dtype="float32")
    change = int(band[0]["t_s"] * SR)
    for beat in range(-8, 8):
        at = change + int(beat * bar_s / 4 * SR)
        seg = machine[at:at + int(bar_s / 4 * SR)]
        assert seg.size and float(np.abs(seg).max()) > 1e-3, f"silence at beat {beat}"


def test_a_change_of_style_changes_the_chords_in_that_very_bar(tmp_path, consent):
    """Am at the count, from synthwave's own cycle, would be a wrong chord in midwest."""
    rec = _detect_take(tmp_path, _strummed_take(100.0), prompt="in A minor", max_minutes=0.5)
    rows = _rows(rec)
    drums = [r for r in rows if r["entry"] == "drums"]
    band = [r for r in rows if r["entry"] != "drums"]
    assert {r["style"] for r in drums} == {"synthwave"}, "the count's pick, heard from bar 0"
    assert {r["style"] for r in band} == {"midwest"}
    midwest = governor.progression_symbols("midwest", "A minor")
    # A named key commits on the bar the band comes in, so the cycle starts there.
    assert [r["chord"] for r in band[:4]] == midwest


def test_a_style_that_stays_keeps_its_own_chords(tmp_path, consent):
    """The count's pick has to reach the chords, not just the drums."""
    rec = _detect_take(tmp_path, _strummed_take(100.0, strums_per_beat=0.5), prompt="in A minor",
                       max_minutes=0.5)
    band = [r for r in _rows(rec) if r["entry"] != "drums"]
    assert band and {r["style"] for r in band} == {"synthwave"}
    assert [r["chord"] for r in band[:4]] == governor.progression_symbols("synthwave", "A minor")


@needs_laptop_speed
def test_the_style_never_moves_once_the_band_is_in(tmp_path, consent):
    """Sparse until the band is in, then busy enough that a second look would say midwest."""
    human = _strummed_take(85.0, seconds=110.0, strums_per_beat=0.5, busier_from_s=38.0)
    rec = _detect_take(tmp_path, human, max_minutes=0.5)
    rows = _rows(rec)
    assert rows[-1]["t_s"] > 80.0, "too short to have been tempted"
    band = [r for r in rows if r["entry"] != "drums"]
    assert band and band[0]["t_s"] < 38.0
    assert {r["style"] for r in rows} == {"lofi"}


def test_a_fast_count_gets_its_style_before_the_first_bar(tmp_path, consent):
    rec = _detect_take(tmp_path, _strummed_take(140.0, seconds=8.0), max_minutes=0.1)
    rows = _rows(rec)
    assert rows and rows[0]["style"] == "trap"
    # the drums stage's plain click (test_plain_intro), with the band held back
    assert "click" in rows[0]["roles"] and "keys" not in rows[0]["roles"]


def test_the_old_engine_rings_out_under_the_new_one(tmp_path, consent, monkeypatch):
    """A marker for a tail, since what a real tail holds is up to the style: a steady level
    no band plays.  It has to reach the bar the band came in on, on the bar's own clock."""
    import soundfile as sf

    import plume.duet as duet_mod
    tail_n = SR // 10

    class Marked(duet_mod.DuetEngine):
        def flush_tail(self):
            super().flush_tail()
            return np.full((tail_n, 2), 0.25, dtype=np.float32)

    monkeypatch.setattr(duet_mod, "DuetEngine", Marked)
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.5)
    band = [r for r in _rows(rec) if r["entry"] != "drums"]
    assert band and band[0]["style"] == "midwest"
    machine, _ = sf.read(str(rec / "machine.wav"), dtype="float32")
    comp = int(0.020 * SR)               # the fake device's reported round trip
    at = int(band[0]["t_s"] * SR) - comp
    inside = float(np.mean(machine[at + 200:at + tail_n - 200]))
    before = float(np.mean(machine[at - tail_n:at - 200]))
    assert inside - before == pytest.approx(0.25, abs=0.05)


class _Wandering:
    """A tracker that never believes one key for two bars running."""

    KEYS = ("C major", "G major", "D major")
    BAR_S = 240.0 / 115.0

    def __init__(self):
        self._last = None

    def process(self, frames):
        if frames:
            at = int(frames[-1].t_s / self.BAR_S) % len(self.KEYS)
            self._last = _heard(self.KEYS[at])
        return self._last


@needs_laptop_speed
@pytest.mark.parametrize("detect,comes_in", [(True, True), (False, False)])
def test_detect_cannot_strand_her_on_drums(tmp_path, consent, monkeypatch, detect, comes_in):
    import plume.listen.harmony as harmony_mod
    monkeypatch.setattr(harmony_mod, "HarmonyTracker", _Wandering)
    with patched_module("sounddevice", fake_audio(_strummed_take(115.0), devices=RIG, rate_x=12.0)):
        realtime.lead(prompt="midwest", seed=0, tap_start=True, detect=detect,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("wander"), max_minutes=0.8,
                      direct=True, input_device="audio interface", input_channel=1)
    rows = _rows(tmp_path / "rec")
    heard = sum(r["playing"] for r in rows)
    assert heard >= 20, "not enough bars to prove anything"
    band = [r for r in rows if r["entry"] != "drums"]
    assert bool(band) is comes_in


def test_the_same_playing_on_a_room_mic_stays_lofi(tmp_path, consent):
    """The onsets are all there.  A room mic just isn't allowed to believe them."""
    # A room mic masks the band out of what it hears, the costliest listening there is,
    # so this is the take a starved runner leaves behind its playhead.
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.5,
                       rate_x=playhead_rate(12.0),
                       route={"direct": False, "input_device": "audio interface",
                              "input_channel": 1})
    rows = _rows(rec)
    assert {r["style"] for r in rows} == {"lofi"}, "room mode took a reading it can't trust"
    assert any(r["entry"] != "drums" for r in rows), "the band never came in"


def _folk_take(bpm, *, bars=28, kind="strum"):
    """Four knocks on the body, two bars of drums alone, then a strummed G major."""
    import folk_fixtures as ff
    rng = np.random.default_rng(4)
    period = 60.0 / bpm
    lead_in = 2.0 + 12 * period
    guitar = ff.make_take(kind, bpm=bpm, bars=bars)
    x = rng.standard_normal(int(lead_in * SR) + guitar.shape[0]).astype(np.float32) * 3e-4
    for i in range(4):
        at = int((2.0 + i * period) * SR)
        n = int(0.004 * SR)
        x[at:at + n] += rng.standard_normal(n).astype(np.float32) * 0.8
    x[int(lead_in * SR):] += guitar
    return x


def test_a_strummed_folk_tune_on_a_direct_input_brings_in_the_folk_band(tmp_path, consent):
    rec = _detect_take(tmp_path, _folk_take(96.0), max_minutes=0.6)
    rows = _rows(rec)
    band = [r for r in rows if r["entry"] != "drums"]
    assert band, "the band never came in"
    assert {r["style"] for r in band} == {"folk"}
    assert all(r["style_guess"] for r in rows)
    assert "strummed" in band[0]["style_why"]


def test_the_same_tune_fingerpicked_is_not_called_folk(tmp_path, consent):
    rec = _detect_take(tmp_path, _folk_take(96.0, kind="pick"), max_minutes=0.6)
    band = [r for r in _rows(rec) if r["entry"] != "drums"]
    assert band, "the band never came in"
    assert "folk" not in {r["style"] for r in band}


def test_a_style_she_named_is_not_a_guess(tmp_path, consent):
    rec = _detect_take(tmp_path, _strummed_take(85.0), prompt="trap", max_minutes=0.3)
    rows = _rows(rec)
    assert {r["style"] for r in rows} == {"trap"}
    assert not any(r["style_guess"] for r in rows)
    assert all(r["key_guess"] for r in rows)
    doc = json.loads((rec / "session.json").read_text())
    assert doc["style"] == "trap" and doc["detect"] is True


def test_a_key_she_named_is_not_a_guess(tmp_path, consent):
    rec = _detect_take(tmp_path, _strummed_take(85.0), prompt="in A minor", max_minutes=0.4)
    rows = _rows(rec)
    assert all(r["style_guess"] and not r["key_guess"] for r in rows)
    band = [r for r in rows if r["entry"] != "drums"]
    assert band and {r["key"] for r in band} == {"A minor"}


def test_what_the_take_says_about_itself(tmp_path, consent):
    rec = _detect_take(tmp_path, np.zeros(SR * 3, dtype=np.float32), max_minutes=0.02)
    doc = json.loads((rec / "session.json").read_text())
    assert doc["detect"] is True and doc["monitor"] is False
    # no placeholder passed off as what is playing
    assert doc["style"] == "detect" and doc["session"] is None


def test_a_take_without_detect_says_that_too(tmp_path, consent):
    with patched_module("sounddevice", fake_audio(np.zeros(SR * 3, dtype=np.float32), devices=RIG)):
        realtime.lead(prompt="midwest", seed=0, tap_start=True, monitor=True, status=False,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      socket_path=short_socket_path("nodet"), max_minutes=0.02)
    doc = json.loads((tmp_path / "rec" / "session.json").read_text())
    assert doc["detect"] is False and doc["monitor"] is True and doc["style"] == "midwest"


def test_the_whole_way_through_the_cli(tmp_path, consent, monkeypatch):
    """No prompt, no session, a stray session file in the folder, and a real argv."""
    (tmp_path / "stray.plume.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    rec = tmp_path / "rec"
    with patched_module("sounddevice", fake_audio(np.zeros(SR * 3, dtype=np.float32), devices=RIG)):
        code = cli.main(["lead", "--tap-start", "--detect", "--record", str(rec),
                         "--device", "audio interface", "--input-channel", "1", "--direct",
                         "--socket", short_socket_path("clidet"), "--max-minutes", "0.02"])
    assert code == 0
    assert json.loads((rec / "session.json").read_text())["detect"] is True


def test_the_feed_passes_the_guesses_on(tmp_path, consent):
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.3)
    docs = Listener(rec, every=1).poll()
    assert docs
    assert all(d["style_guess"] is True and d["key_guess"] is True for d in docs)
    assert docs[0]["style"] == "lofi"


def test_an_older_take_has_nothing_to_pass_on(tmp_path):
    import soundfile as sf
    (tmp_path / "decisions.ndjson").write_text(
        json.dumps({"bar": 0, "t_s": 0.0, "bpm": 120.0}) + "\n")
    for name in ("human.wav", "machine.wav"):
        sf.write(str(tmp_path / name), np.zeros(SR * 3, dtype=np.float32), SR)
    doc = Listener(tmp_path, every=1).poll()[0]
    assert doc["style"] is None and doc["style_guess"] is None and doc["key_guess"] is None
