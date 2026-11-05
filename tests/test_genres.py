"""jazz, house, hip hop and r&b: each has to be itself, offline and with the band live.

These numbers say a genre has the bones it was built with.  They do not say it sounds
good.  Nobody has listened to these yet; that is for a person, not a test.
"""
import dataclasses
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from conftest import fake_audio, patched_module, short_socket_path  # noqa: E402
from plume import duet as duet_mod  # noqa: E402
from plume import figures, realtime  # noqa: E402
from plume import fx as fx_mod  # noqa: E402
from plume import mind as mind_mod  # noqa: E402
from plume import roles as roles_mod  # noqa: E402
from plume.compose import PromptSpec, compose, describe_prompt, parse_prompt  # noqa: E402
from plume.duet import DuetEngine  # noqa: E402
from plume.engine import compile_events, render  # noqa: E402
from plume.governor import progression_symbols  # noqa: E402
from plume.sim import simulate  # noqa: E402
from plume.styles_lib import list_styles, load_style  # noqa: E402
from plume.theory import midi_to_hz, parse_chord, parse_key, roman_to_chord  # noqa: E402
from plume.voices import render_note  # noqa: E402

GENRES = ["jazz", "house", "hiphop", "rnb"]
OLDER = ["lofi", "midwest", "synthwave", "trap"]
SR = 48000
PEAK_LIMIT_DB = -0.9      # the offline limiter's ceiling is -1 dB, with a little to spare
CODE_VOICES = ["housekick", "boomkick", "offhat", "roundbass", "housestab"]
#: Recorded because code made them sound fake.
SAMPLED_VOICES = ["brush", "upright", "piano"]
NEW_VOICES = CODE_VOICES + SAMPLED_VOICES
#: The lowest and highest note each sampled voice is asked for in these styles, live or offline.
SAMPLED_RANGES = {"upright": (28, 55), "piano": (45, 107), "wurli": (48, 91)}


# ------------------------------------------------------------------ prompts

@pytest.mark.parametrize("text,style", [
    ("r&b", "rnb"), ("R & B at 70 bpm", "rnb"), ("rnb", "rnb"), ("neo soul", "rnb"),
    ("neo-soul in F minor", "rnb"), ("hip hop", "hiphop"), ("hip-hop", "hiphop"),
    ("hiphop", "hiphop"), ("boom bap", "hiphop"), ("boom-bap beat", "hiphop"),
    ("house", "house"), ("deep house at 124", "house"), ("a house beat", "house"),
    ("the house groove", "house"), ("jazz", "jazz"), ("something jazzy", "jazz"),
    ("house music", "house"), ("house in A minor", "house"), ("house at 124", "house"),
    ("house, 3 movements", "house"), ("house 124 bpm", "house"), ("play some house", "house"),
    ("make me some house at 124", "house"), ("chill house", "house"),
    ("dark house track", "house"), ("tech-house", "house"), ("acid  house", "house"),
    # the word means the genre unless it plainly means something else
    ("i want some house", "house"), ("can you play some house", "house"),
    ("house please", "house"), ("124 bpm house", "house"), ("C minor house", "house"),
    ("french house", "house"), ("classic house", "house"), ("piano house", "house"),
    ("house for coding", "house"), ("good house music", "house"), ("my house music", "house"),
    ("house in a minor", "house"), ("a house in C minor", "house"), ("chill house", "house"),
    ("house catchy and fast", "house"), ("house online", "house"),
    # a prompt about the music keeps the genre, whatever words come after the name
    ("deep house in the key of F minor", "house"), ("house key of C minor", "house"),
    ("deep house key of F minor", "house"), ("house in the key of C minor", "house"),
    ("a house in the key of C minor", "house"), ("house in the style of chicago", "house"),
    ("deep house at night", "house"), ("tech house on the radio", "house"),
    ("deep  house on", "house"), ("deep-house party", "house"), ("acid house band", "house"),
    ("play house next", "house"), ("play house at night", "house"),
    ("play some house on repeat", "house"), ("some house in the morning", "house"),
    ("an old house beat", "house"), ("play some old house music", "house"), ("old house", "house"),
    ("about a deep house beat", "house"), ("a song about a deep house", "house"),
    ("in the house music style", "house"), ("empty house music", "house"),
    # plain requests that a longer list of building phrases used to swallow
    ("house on repeat", "house"), ("house on vinyl", "house"), ("house at night", "house"),
    ("house in the club", "house"), ("an old house in C minor", "house"),
    ("in the house at 124 bpm", "house"), ("house is the vibe", "house"),
])
def test_aliases_reach_their_style(text, style):
    assert parse_prompt(text).styles == [style]


@pytest.mark.parametrize("text", [
    "play me something for my house", "burn the house down", "a house of cards",
    "house party", "a haunted house", "in a coffee house", "full house",
    "something for grandma's house", "the white house", "the opera house",
    "open house vibe", "house sitting", "house arrest", "the  house", "a house", "the house",
    "my house", "our house", "house-party", "the house, please", "powerhouse", "housewife",
    "my house keys", "some coffee house", "play a haunted house",
])
def test_house_the_building_is_not_house_the_genre(text):
    assert parse_prompt(text).styles == []


@pytest.mark.parametrize("text,styles,bpm", [
    ("chill trap beat over lofi ambient synthwave at 120 bpm and 3 major movements",
     ["trap", "lofi", "synthwave"], 120.0),
    ("lofi midwest emo in F major", ["midwest", "lofi"], None),
    ("dark lofi at 88 bpm", ["lofi"], 88.0),
    # Words that used to be ignored must not start changing tempo or seed.
    ("lofi hip hop", ["lofi"], None),
    ("trap house", ["trap"], None),
    ("chill jazz synthwave", ["synthwave"], None),
    ("emo house party", ["midwest"], None),
])
def test_a_prompt_naming_an_older_style_means_what_it_always_meant(text, styles, bpm):
    spec = parse_prompt(text)
    assert spec.styles == styles and spec.bpm == bpm


@pytest.mark.parametrize("text,style", [
    # a word right before another genre colours it, and an adjective never leads
    ("jazzy house", "house"), ("jazzy hip hop", "hiphop"), ("hip hop jazz", "jazz"),
    ("house jazz", "jazz"), ("jazz house", "house"), ("hip hop house", "house"),
    ("neo soul hip hop", "hiphop"), ("something jazzy for a house beat", "house"),
    # apart, the first one named is what was asked for
    ("hip hop with jazz chords", "hiphop"), ("jazz with a boom bap beat", "jazz"),
    ("neo soul and hip hop", "rnb"),
])
def test_two_new_genres_make_one_recipe_at_its_own_tempo(text, style):
    """compose has no blend for these, so a second name could only drag the tempo out of
    the range of the recipe that plays."""
    spec = parse_prompt(text)
    assert spec.styles == [style]
    lo, hi = load_style(style)["bpm_range"]
    assert lo <= compose(spec, seed=1).bpm <= hi


@pytest.mark.parametrize("text,kept,dropped", [
    ("jazzy house", ["house"], "jazzy"),
    ("jazz house", ["house"], "jazz"),
    ("hip hop house", ["house"], "hip"),
    ("hip hop jazz", ["jazz"], "hop"),
    ("neo soul and hip hop", ["rnb"], "hip"),
])
def test_the_genre_that_adds_nothing_is_not_reported_as_understood(text, kept, dropped):
    said = describe_prompt(text)
    assert said["understood"]["style"] == kept and dropped in said["ignored"]


def test_a_genre_word_the_prompt_set_aside_is_reported_as_ignored():
    said = describe_prompt("trap jazz")
    assert said["understood"]["style"] == ["trap"] and "jazz" in said["ignored"]


@pytest.mark.parametrize("text,word", [
    ("jazz, dark house beat", "dark"), ("jazz, chill house", "chill"),
    ("trap, dark house music", "dark"), ("jazz, deep house at 124 bpm", "124"),
])
def test_a_word_that_was_applied_is_never_reported_as_not_understood(text, word):
    said = describe_prompt(text)
    assert word not in said["ignored"] and "house" in said["ignored"]
    if word in {"dark", "chill"}:
        assert said["understood"]["mood"] == [word]


@pytest.mark.parametrize("cue,value", [
    ("go boom bap", "hiphop"), ("make it hip hop", "hiphop"), ("try neo soul", "rnb"),
    ("more r&b", "rnb"), ("make it house", "house"), ("more jazzy", "jazz"),
    ("more soul", "soul"),
])
def test_cues_use_the_same_names(cue, value):
    parsed = mind_mod.parse_cue(cue)
    assert (parsed["kind"], parsed["value"]) == ("style", value)


def test_the_page_can_offer_all_of_them_and_they_parse_back():
    from plume import guitake
    assert set(GENRES) <= set(list_styles())
    for name in GENRES:
        assert guitake.canonical_style(name) == name
    assert guitake.canonical_style("R&B") == "rnb"
    assert guitake.canonical_style("hip hop") == "hiphop"


# ------------------------------------------------------------------- voices

def _note(preset, hz=None, dur=0.5, vel=0.8, seed=0):
    return render_note(preset, sr=SR, dur_s=dur, freq_hz=hz, velocity=vel,
                       rng=np.random.default_rng(seed), params={})


@pytest.mark.parametrize("preset", NEW_VOICES)
def test_new_voice_is_finite_bounded_and_reproducible(preset):
    y = _note(preset, 110.0)
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert 0.05 < np.abs(y).max() <= 1.0
    assert np.array_equal(y, _note(preset, 110.0))


@pytest.mark.parametrize("preset", NEW_VOICES)
def test_new_voice_goes_quiet_when_left_alone(preset):
    y = _note(preset, 110.0, dur=0.2)
    tail = y[int(0.8 * y.size):]
    assert np.abs(tail).max() < 0.5 * np.abs(y).max()


@pytest.mark.parametrize("preset,hz", [("roundbass", 41.2), ("roundbass", 110.0)])
def test_a_pitched_voice_is_at_the_pitch_it_was_asked_for(preset, hz):
    y = _note(preset, hz, dur=1.0)[: SR]
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size), n=SR * 2))
    freqs = np.fft.rfftfreq(SR * 2, 1 / SR)
    band = (freqs > 0.8 * hz) & (freqs < 1.2 * hz)
    assert abs(freqs[band][np.argmax(spectrum[band])] - hz) < 0.02 * hz
    above = (freqs > 1.2 * hz)
    assert spectrum[band].max() > 0.5 * spectrum[above].max()


def test_a_harder_hit_on_the_piano_is_brighter():
    def centroid(y):
        s = np.abs(np.fft.rfft(y))
        return float((s * np.fft.rfftfreq(y.size, 1 / SR)).sum() / s.sum())
    assert (centroid(_note("piano", 261.6, vel=1.0).astype(np.float64))
            > centroid(_note("piano", 261.6, vel=0.2).astype(np.float64)))


def test_the_house_kick_is_shorter_than_the_boom_bap_kick_and_both_beat_an_808():
    def length_s(preset):
        y = np.abs(_note(preset, 50.0, dur=0.01))
        return np.flatnonzero(y > 0.1 * y.max())[-1] / SR
    assert length_s("housekick") < length_s("boomkick") < length_s("808kick")


# ------------------------------------------------------------------- figures

def test_a_walking_bar_starts_on_its_root_and_leans_into_the_next():
    rng = np.random.default_rng(0)
    for root, next_root in [(2, 7), (7, 0), (0, 9), (9, 2), (5, 5), (11, 10)]:
        for _ in range(40):
            line = figures.walking_line(root, parse_chord("Dm7").intervals, next_root, rng)
            assert len(line) == 4 and line[0] % 12 == root
            assert all(28 <= m <= 60 for m in line)
            assert line[2] != line[3], "a repeated note is not a walk"
            if next_root != root:
                assert min((line[3] - next_root) % 12, (next_root - line[3]) % 12) == 1


def test_a_walking_line_copes_with_a_bare_root_and_fifth():
    line = figures.walking_line(0, (0, 7), 7, np.random.default_rng(0))
    assert line[0] % 12 == 0 and len(set(line)) >= 3


@pytest.mark.parametrize("sym,want", [
    ("Dm7", {5, 0, 4}),       # F, C, and the ninth E
    ("G7", {11, 5, 9}),
    ("Cmaj7", {4, 11, 2}),
])
def test_a_shell_is_third_seventh_and_ninth_and_never_the_root(sym, want):
    notes = figures.rootless_voicing(parse_chord(sym), shell=True)
    assert {m % 12 for m in notes} == want
    assert parse_chord(sym).root not in {m % 12 for m in notes}


def test_a_half_diminished_shell_keeps_out_the_ninth():
    notes = figures.rootless_voicing(parse_chord("Dm7b5"), shell=True)
    assert {m % 12 for m in notes} == {5, 0}


def test_a_neo_soul_voicing_keeps_every_colour_tone():
    notes = figures.rootless_voicing(parse_chord("Dm11"))
    assert {m % 12 for m in notes} == {5, 9, 0, 4, 7}    # third, fifth, seventh, 9th, 11th
    assert notes == sorted(notes) and notes[-1] - notes[0] <= 16


def test_eleventh_chords_are_spelled_from_a_numeral():
    assert roman_to_chord("ii11", parse_key("C major")).sym == "Dm11"


# ------------------------------------------------------------------- recipes

@pytest.mark.parametrize("name,lo,hi", [("house", 118, 130), ("hiphop", 80, 100),
                                        ("rnb", 60, 90)])
def test_tempo_ranges(name, lo, hi):
    assert load_style(name)["bpm_range"] == [lo, hi]


def test_jazz_has_a_wide_range_and_its_turnarounds_in_both_modes():
    style = load_style("jazz")
    lo, hi = style["bpm_range"]
    assert hi - lo >= 100
    tops = {m: max(style["progressions"][m], key=lambda p: p["weight"])["numerals"]
            for m in ("major", "minor")}
    assert tops["major"][:3] == ["ii7", "V7", "Imaj7"]
    assert tops["minor"][:3] == ["ii7b5", "V7", "i7"]
    # no chord twice in a row, so the bass always knows which bar it is in
    for symbols in (progression_symbols("jazz", "C major"), progression_symbols("jazz", "A minor")):
        assert len(set(symbols)) == len(symbols)


def test_rnb_harmony_reaches_ninths_and_elevenths_where_it_is_played_live():
    qualities = {parse_chord(s).quality for s in progression_symbols("rnb", "C major")}
    assert {"m11", "9", "maj9", "m9"} <= qualities


def test_house_progression_is_built_of_minor_ninths():
    qualities = {parse_chord(s).quality for s in progression_symbols("house", "A minor")}
    assert "m9" in qualities


def test_each_new_voice_is_what_it_says_and_every_recipe_voice_exists():
    from plume import samplekit, voices
    for name in CODE_VOICES:
        assert name in voices.PRESETS and name not in samplekit.voice_names()
        assert name not in samplekit._tuned_manifest()
    assert "brush" in samplekit.voice_names()
    assert {"upright", "piano"} <= set(samplekit._tuned_manifest())
    for style in GENRES:
        for preset in load_style(style)["voices"].values():
            assert preset in voices.PRESETS


def _cents_off(preset, midi, vel):
    """The strongest of the first few partials, in cents from where it should be."""
    hz = midi_to_hz(midi)
    y = _note(preset, hz, dur=1.0, vel=vel).astype(np.float64)
    seg = y[int(0.05 * SR):int(0.6 * SR)]
    n = 2 ** 20
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    found = []
    for k in (1, 2, 3, 4):
        if hz * k > 0.4 * SR:
            break
        band = (freqs > hz * k * 2 ** (-1 / 24)) & (freqs < hz * k * 2 ** (1 / 24))
        at = int(np.argmax(spec[band]))
        found.append((spec[band][at], 1200.0 * np.log2(freqs[band][at] / (hz * k))))
    return max(found)[1]


@pytest.mark.parametrize("preset", sorted(SAMPLED_RANGES))
@pytest.mark.parametrize("vel", [0.3, 0.9])
def test_a_sampled_voice_is_in_tune_across_the_notes_it_plays(preset, vel):
    lo, hi = SAMPLED_RANGES[preset]
    off = {m: _cents_off(preset, m, vel) for m in range(lo, hi + 1)}
    assert max(abs(c) for c in off.values()) < 25.0, off


@pytest.mark.parametrize("preset", sorted(SAMPLED_RANGES))
def test_a_sampled_note_starts_and_ends_without_a_click(preset):
    lo, hi = SAMPLED_RANGES[preset]
    for midi in range(lo, hi + 1, 3):
        for dur in (0.1, 0.4, 2.0, 6.0):
            y = _note(preset, midi_to_hz(midi), dur=dur, vel=0.9)
            assert np.isfinite(y).all() and np.abs(y).max() > 0.05
            assert abs(y[0]) < 0.1 * np.abs(y).max(), (midi, dur)
            assert abs(y[-1]) < 2e-3, (midi, dur)
            end = max(16, int(0.01 * y.size))
            assert np.abs(np.diff(y[-end:])).max() < 0.05, (midi, dur)


def test_a_brush_hit_starts_at_once_ends_quietly_and_follows_velocity():
    y = _note("brush", vel=0.9)
    assert np.flatnonzero(np.abs(y) > 0.1 * np.abs(y).max())[0] < 0.01 * SR
    assert abs(y[-1]) < 1e-3
    assert np.abs(_note("brush", vel=0.1)).max() < np.abs(y).max()


# The top of the piano really does thin out, so its neighbours are held below that.
@pytest.mark.parametrize("preset,lo,hi", [("upright", 28, 55), ("piano", 45, 89), ("wurli", 48, 91)])
@pytest.mark.parametrize("vel", [0.3, 0.9])
def test_a_line_does_not_jump_in_level_from_one_note_to_the_next(preset, lo, hi, vel):
    def body(midi, seed):
        y = _note(preset, midi_to_hz(midi), dur=0.5, vel=vel, seed=seed)
        return _rms_db(y[:int(0.3 * SR)])
    level = {m: [body(m, seed) for seed in range(8)] for m in range(lo, hi + 1)}
    means = {m: float(np.mean(v)) for m, v in level.items()}
    steps = {m: abs(means[m + 1] - means[m]) for m in range(lo, hi)}
    assert max(steps.values()) < 2.0, steps
    takes = {m: float(np.max(v) - np.min(v)) for m, v in level.items()}
    assert max(takes.values()) < 3.0, takes


@pytest.mark.parametrize("preset,hz", [("upright", 82.4), ("wurli", 261.6), ("piano", 261.6)])
def test_a_repeated_sampled_note_is_not_stuck_on_one_take_or_level(preset, hz):
    takes = {_note(preset, hz, vel=0.9, seed=seed).tobytes() for seed in range(8)}
    if preset == "upright":
        assert len(takes) > 1
    soft, hard = _note(preset, hz, vel=0.3), _note(preset, hz, vel=0.9)
    assert np.abs(soft).max() < np.abs(hard).max()


@pytest.mark.parametrize("style", ["jazz", "hiphop", "rnb"])
def test_a_first_live_bar_is_quick_even_with_nothing_loaded_yet(style):
    """A sample reads from disk on first use, on the planner, which has the bar before to do it in."""
    from plume import samplekit
    engine = _engine(style)
    bpm = max(load_style(style)["bpm_range"])
    plan, harm = _bar(style, _band_roles(engine), 0)
    samplekit._load.cache_clear()
    t = time.perf_counter()
    engine.render_bar(plan, SimpleNamespace(bpm=float(bpm)), harm)
    assert time.perf_counter() - t < 0.5 * 240.0 / bpm


def test_the_sampled_voices_load_inside_a_small_budget():
    from plume import samplekit
    root = pathlib.Path(figures.__file__).parent / "samples"
    new = [p for v in ("upright", "piano") for p in (root / "tuned").glob(f"{v}_*.flac")]
    new += list((root / "kit").glob("brush_*.flac"))
    assert sum(p.stat().st_size for p in new) < 6_000_000
    samplekit._load.cache_clear()
    total = 0
    for voice in ("upright", "piano"):
        for entry in samplekit._tuned_manifest()[voice]:
            total += samplekit._load(entry["file"], SR, samplekit._TUNED).nbytes
    assert total < 80_000_000


@pytest.mark.parametrize("style,role,lo,hi", [
    ("jazz", "bass", -22.0, -13.0), ("jazz", "pad", -30.0, -20.0), ("jazz", "snare", -46.0, -33.0),
    ("hiphop", "pad", -28.0, -18.0), ("rnb", "keys", -25.0, -15.0), ("rnb", "lead", -30.0, -20.0),
])
def test_a_sampled_part_sits_where_the_code_voice_it_replaced_sat(renders, style, role, lo, hi):
    y, _ = sf.read(renders[style].stems / f"{role}.wav")
    assert lo <= _rms_db(y) <= hi
    live, _ = _plain_take(style, {role: 1.0}, bars=8)
    assert lo - 1.0 <= _rms_db(live) <= hi + 1.0
    assert np.abs(y).max() < 1.0


# ------------------------------------------------------------------ offline

def _session(style, *, humanize=False):
    session = compose(PromptSpec(styles=[style], name=style, movements=1), seed=5)
    if not humanize:
        session.humanize_ms = 0.0
    return session


def _hot_bars(session):
    """The bar window of the section with the highest energy."""
    at, best = 0, None
    for mv in session.movements:
        for sec in mv.sections:
            if best is None or sec.energy > best[0]:
                best = (sec.energy, at, at + sec.bars)
            at += sec.bars
    return best[1], best[2]


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    out = {}
    folder = tmp_path_factory.mktemp("genres")
    for style in GENRES + OLDER:
        session = _session(style)
        bars = _hot_bars(session)
        stems = folder / f"{style}_stems"
        render(session, out_path=str(folder / f"{style}.wav"), bars=bars, stems_dir=str(stems))
        y, sr = sf.read(folder / f"{style}.wav")
        assert sr == SR
        out[style] = SimpleNamespace(y=y, session=session, stems=stems)
    return out


def _rms_db(y):
    return 20.0 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)


@pytest.mark.parametrize("style", GENRES)
def test_a_render_is_clean_and_within_its_limits(renders, style):
    y = renders[style].y
    assert np.isfinite(y).all()
    assert 20.0 * np.log10(np.abs(y).max()) <= PEAK_LIMIT_DB


@pytest.mark.parametrize("style", GENRES)
def test_a_render_sits_with_the_older_styles_in_level(renders, style):
    older = [_rms_db(renders[s].y) for s in OLDER]
    assert min(older) - 4.0 <= _rms_db(renders[style].y) <= max(older) + 1.0


@pytest.mark.parametrize("style", GENRES)
def test_a_composed_tempo_is_in_the_genres_range(renders, style):
    lo, hi = load_style(style)["bpm_range"]
    assert lo <= renders[style].session.bpm <= hi


def _events(style, track):
    """(beat in bar, seconds late of the straight grid slot) for a track's hot bars."""
    session = _session(style)
    a, b = _hot_bars(session)
    bar_s = 240.0 / session.bpm
    return session, [ev for ev in compile_events(session)
                     if ev.track == track and a * bar_s - 0.1 <= ev.t_s < b * bar_s - 0.1], a * bar_s


def _beats(session, evs, t0):
    spb = 60.0 / session.bpm
    # A hit pushed ahead of the bar line belongs to that bar line, not to beat 3.99.
    return [((ev.t_s - t0) / spb + 0.05) % 4.0 - 0.05 for ev in evs]


def test_house_is_four_on_the_floor_with_the_clap_on_two_and_four(renders):
    session, kicks, t0 = _events("house", "kick")
    assert {abs(round(b, 1)) for b in _beats(session, kicks, t0)} == {0.0, 1.0, 2.0, 3.0}
    session, claps, t0 = _events("house", "snare")
    hits = {round(b) for b in _beats(session, claps, t0)}
    assert {1, 3} <= hits


def test_house_hat_sits_on_the_offbeat(renders):
    session, hats, t0 = _events("house", "hat")
    fracs = {round(b % 1.0, 1) for b in _beats(session, hats, t0)}
    assert fracs == {0.5}


def _onsets_s(y, sr=SR):
    """Where the level jumps by 6 dB or more in 8 ms.  A level threshold alone would
    keep firing on a cymbal's ring."""
    hop = sr // 500
    window = int(0.012 * sr)           # longer than half a cycle of a kick's tail
    power = np.convolve(y ** 2, np.ones(window) / window, mode="same")[::hop]
    db = 10.0 * np.log10(power + 1e-12)
    floor = db.max() - 40.0
    rise = db[4:] - db[:-4]
    out = []
    for i in np.flatnonzero((rise > 6.0) & (db[4:] > floor)):
        t = (i + 4) * hop / sr
        if not out or t - out[-1] > 0.07:
            out.append(float(t))
    return np.array(out)


def test_a_measured_house_kick_keeps_the_beat(renders):
    r = renders["house"]
    kick, _ = sf.read(r.stems / "kick.wav")
    gaps = np.diff(_onsets_s(kick.mean(axis=1)))
    assert np.median(gaps) == pytest.approx(60.0 / r.session.bpm, rel=0.02)
    assert np.percentile(gaps, 90) < 1.1 * 60.0 / r.session.bpm


def test_a_measured_jazz_ride_swings_two_to_one(renders):
    r = renders["jazz"]
    ride, _ = sf.read(r.stems / "hat.wav")
    spb = 60.0 / r.session.bpm
    fracs = (_onsets_s(ride.mean(axis=1)) / spb) % 1.0
    skips = [f for f in fracs if 0.3 < f < 0.9]
    assert skips, "no skip notes heard"
    assert np.median(skips) == pytest.approx(2.0 / 3.0, abs=0.05)
    # and none of the straight eighth, which is what a stiff ride would play
    assert not [f for f in skips if abs(f - 0.5) < 0.05]


def test_jazz_walks_in_quarters_and_its_kick_is_feathered(renders):
    session, bass, t0 = _events("jazz", "bass")
    assert {abs(round(b, 1)) for b in _beats(session, bass, t0)} == {0.0, 1.0, 2.0, 3.0}
    assert len({ev.freq_hz for ev in bass}) >= 6
    kick, _ = sf.read(renders["jazz"].stems / "kick.wav")
    ride, _ = sf.read(renders["jazz"].stems / "hat.wav")
    assert _rms_db(kick) < _rms_db(ride)
    assert max(ev.velocity for ev in compile_events(session) if ev.track == "kick") < 0.55


@pytest.mark.parametrize("style,late_ms", [("hiphop", 26.0), ("rnb", 34.0)])
def test_the_snare_is_lazy(style, late_ms):
    session, snares, t0 = _events(style, "snare")
    spb = 60.0 / session.bpm
    backbeats = [ev for ev in snares if abs(((ev.t_s - t0) / spb) % 2.0 - 1.0) < 0.15]
    assert backbeats
    late = [1000.0 * (((ev.t_s - t0) / spb) % 2.0 - 1.0) * spb for ev in backbeats]
    assert np.median(late) == pytest.approx(late_ms, abs=2.0)


def test_rnb_drags_harder_than_hip_hop_relative_to_each_kick():
    def lag(style):
        bias = load_style(style)["timing_bias_ms"]
        return bias["snare"] - bias["kick"]
    assert lag("rnb") >= lag("hiphop") >= 25.0


def test_hip_hop_hats_swing(renders):
    session, hats, t0 = _events("hiphop", "hat")
    spb = 60.0 / session.bpm
    # subtract the hats' own lateness: what is left is the swing
    bias = load_style("hiphop")["timing_bias_ms"]["hat"] / 1000.0 / spb
    pos = [(((ev.t_s - t0) / spb) - bias) % 0.5 for ev in hats]
    late = [p for p in pos if 0.2 < p < 0.45]
    assert late, "the busiest hat grid should carry sixteenths"
    expect = 0.25 + (0.6 - 0.5) * 2.0 * 0.25
    assert np.median(late) == pytest.approx(expect, abs=0.01)


def test_hip_hop_kick_is_boom_bap_not_four_on_the_floor(renders):
    session, kicks, t0 = _events("hiphop", "kick")
    assert {abs(round(b, 1)) for b in _beats(session, kicks, t0)} == {0.0, 1.8, 2.5}


def _spectrum_bands(y):
    mono = y.mean(axis=1)
    power = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1 / SR)
    edges = np.geomspace(40.0, 16000.0, 13)
    bands = np.array([power[(freqs >= a) & (freqs < b)].sum() for a, b in zip(edges, edges[1:])])
    return 10.0 * np.log10(bands / bands.sum() + 1e-12)


def test_every_pair_of_genres_has_a_different_shape_of_sound(renders):
    shapes = {s: _spectrum_bands(renders[s].y) for s in GENRES}
    for i, a in enumerate(GENRES):
        for b in GENRES[i + 1:]:
            assert np.mean(np.abs(shapes[a] - shapes[b])) > 2.0, (a, b)


def test_every_pair_of_genres_has_a_different_kick_grid():
    grids = {}
    for style in GENRES:
        session, kicks, t0 = _events(style, "kick")
        grids[style] = frozenset(abs(round(b, 1)) for b in _beats(session, kicks, t0))
    jazz, house = grids["jazz"], grids["house"]
    assert jazz == house == frozenset({0.0, 1.0, 2.0, 3.0})   # same grid: see below
    assert grids["hiphop"] != grids["rnb"] != house
    # jazz and house share four beats and differ by a long way in how hard they play them
    def strongest(style):
        session, kicks, _ = _events(style, "kick")
        return np.median([ev.velocity for ev in kicks])
    assert strongest("jazz") < 0.5 * strongest("house")


def test_the_crash_never_plays_as_a_track_offline():
    for style in GENRES:
        assert "crash" not in {t.name for t in _session(style).tracks}


@pytest.mark.parametrize("style", GENRES)
def test_the_last_section_still_has_a_voice(style):
    session = _session(style)
    assert session.movements[-1].sections[-1].plays


# --------------------------------------------------------------------- live

def _bar(style, roles, bar, density=0.9, *, chord="auto", key=None):
    key = key or f"C {load_style(style)['default_mode']}"
    symbols = progression_symbols(style, key)
    return (SimpleNamespace(bar=bar, state="comping", roles=roles, gain_db=0.0, density=density,
                            register_ceiling_midi=None, reason="", announce=None,
                            announce_kind=""),
            SimpleNamespace(key=key,
                            chord=symbols[bar % len(symbols)] if chord == "auto" else chord,
                            chord_conf=1.0 if chord == "auto" else 0.0))


def _engine(style, seed=2):
    key = f"C {load_style(style)['default_mode']}"
    return DuetEngine(style=style, seed=seed, sr=SR, key_fallback=key,
                      fade_in_bars=0.0, fade_out_bars=0.0, texture_xfade_beats=0.0)


def _band_roles(engine):
    """What a tap-started take asks for once the band is in: every role but the crash."""
    return {r: 1.0 for r in engine.roles if r != "crash"}


def _play(style, bars=4, density=0.9, roles=None, bpm=None):
    engine = _engine(style)
    style_doc = load_style(style)
    bpm = bpm or sum(style_doc["bpm_range"]) / 2.0
    all_events, audio = [], []
    for b in range(bars):
        plan, harm = _bar(style, roles or _band_roles(engine), b, density)
        audio.append(engine.render_bar(plan, SimpleNamespace(bpm=bpm), harm))
        all_events.append([dict(e, bar=b) for e in engine.last_events])
    return all_events, np.concatenate(audio), bpm


def _beats_of(events, role, bpm):
    return [e["onset_s"] * bpm / 60.0 for e in events if e["role"] == role]


def _plain_take(style, roles, *, bars=2, bpm=None, **style_changes):
    """A render with no timing or velocity jitter, so two renders can be compared."""
    engine = _engine(style)
    engine._style = {**engine._style, **style_changes}
    engine._groove = dataclasses.replace(engine._groove, humanize_ms=0.0, humanize_vel=0.0)
    bpm = bpm or sum(load_style(style)["bpm_range"]) / 2.0
    audio = []
    for b in range(bars):
        plan, harm = _bar(style, roles, b)
        audio.append(engine.render_bar(plan, SimpleNamespace(bpm=bpm), harm))
    return np.concatenate(audio).astype(np.float64), bpm


def test_live_house_bass_pumps_with_the_kick():
    roles = {"kick": 1.0, "bass": 1.0}
    pumped, bpm = _plain_take("house", roles)
    flat, _ = _plain_take("house", roles, live_fx={})
    alone, _ = _plain_take("house", {"bass": 1.0}, live_fx={})
    assert np.abs(alone).max() > 0.0
    spb = int(SR * 60.0 / bpm)
    taken = np.abs(flat - pumped).mean(axis=1)
    bass = np.abs(alone).mean(axis=1)
    # how much of the bass is gone just after a kick, and just before the next one
    def lost(a, b):
        return taken[a:b].mean() / max(bass[a:b].mean(), 1e-9)
    beats = range(1, 7)
    early = np.median([lost(b * spb + spb // 2, b * spb + spb // 2 + spb // 8) for b in beats])
    late = np.median([lost(b * spb + 7 * spb // 8, (b + 1) * spb) for b in beats])
    assert early > 1.5 * late > 0.0
    assert early > 0.07      # half a beat after a kick the dip has mostly recovered


def test_live_house_stabs_pump_as_well():
    roles = {"kick": 1.0, "pad": 1.0}
    pumped, _ = _plain_take("house", roles)
    flat, _ = _plain_take("house", roles, live_fx={})
    assert _rms_db(pumped) < _rms_db(flat)


def test_live_rnb_keys_are_chorused_into_stereo():
    wide, _ = _plain_take("rnb", {"keys": 1.0})
    mono, _ = _plain_take("rnb", {"keys": 1.0}, live_fx={})
    assert np.abs(mono[:, 0] - mono[:, 1]).max() < 1e-6
    assert np.sqrt(np.mean((wide[:, 0] - wide[:, 1]) ** 2)) > 0.1 * np.sqrt(np.mean(wide ** 2))
    assert abs(_rms_db(wide) - _rms_db(mono)) < 0.3


@pytest.mark.parametrize("midi", range(48, 73, 3))
def test_the_live_chorus_loses_no_pitch_and_no_peak_room(midi):
    """A note can catch a dip of the delay for a moment, with its peak at the limit, but as
    the wobble moves on no pitch stays down."""
    y = _note(load_style("rnb")["voices"]["keys"], midi_to_hz(midi), dur=0.8).astype(np.float64)
    wides = [duet_mod._chorus(y, SR, 0.4, at_s) for at_s in np.arange(0.0, 16.0, 0.37)]
    losses = [_rms_db(wide) - _rms_db(y) for wide in wides]
    assert np.mean(losses) > -1.5 and min(losses) > -5.0
    assert max(np.abs(wide).max() for wide in wides) <= 2.0 ** 0.5 * np.abs(y).max() + 1e-9


def test_the_live_chorus_wobble_moves_on_from_note_to_note():
    """It used to start from one phase, so a repeated note came out the same every time."""
    y = _note(load_style("rnb")["voices"]["keys"], midi_to_hz(60), dur=0.8).astype(np.float64)
    first = duet_mod._chorus(y, SR, 0.4, 0.0)
    assert np.array_equal(first, duet_mod._chorus(y, SR, 0.4, 0.0))
    assert np.abs(first - duet_mod._chorus(y, SR, 0.4, 1.3)).max() > 0.01


def test_a_live_keys_note_gets_the_time_it_plays_at(monkeypatch):
    seen = []
    real = duet_mod._chorus
    monkeypatch.setattr(duet_mod, "_chorus", lambda y, sr, depth, at_s: (
        seen.append(at_s), real(y, sr, depth, at_s))[1])
    engine = _engine("rnb")
    engine._groove = dataclasses.replace(engine._groove, humanize_ms=0.0, humanize_vel=0.0)
    for b in (3, 4):
        plan, harm = _bar("rnb", {"keys": 1.0}, b, density=0.3)
        engine.render_bar(plan, SimpleNamespace(bpm=75.0), harm)
    bar_s = 240.0 / 75.0
    assert seen and min(seen) >= 3 * bar_s and max(seen) >= 4 * bar_s


def test_the_chorus_starts_where_it_always_did_unless_told_otherwise():
    x = _note("pad", 220.0, dur=0.3)
    assert np.array_equal(fx_mod.chorus(x, SR, 0.5), fx_mod.chorus(x, SR, 0.5, phase_s=0.0))


def _crest_db(y):
    return 20.0 * np.log10(np.abs(y).max() / np.sqrt(np.mean(y ** 2)))


@pytest.mark.parametrize("role", ["kick", "snare", "bass"])
def test_live_hip_hop_is_saturated_at_the_same_level(role):
    dusty, _ = _plain_take("hiphop", {role: 1.0})
    clean, _ = _plain_take("hiphop", {role: 1.0}, live_fx={})
    assert np.abs(dusty).max() < 0.98 * np.abs(clean).max()
    assert _crest_db(dusty) < _crest_db(clean)
    assert abs(_rms_db(dusty) - _rms_db(clean)) < 0.3


def test_saturating_a_note_keeps_its_level_and_stays_in_its_peak():
    for preset, hz in (("boomkick", 50.0), ("roundbass", 55.0), ("kit_snare", None)):
        y = _note(preset, hz).astype(np.float64)
        driven = duet_mod._saturate(y, 0.25)
        assert abs(_rms_db(driven) - _rms_db(y)) < 0.1, preset
        assert np.abs(driven).max() <= np.abs(y).max() + 1e-9, preset
    silence = np.zeros(100)
    assert np.array_equal(duet_mod._saturate(silence, 0.25), silence)


def test_older_styles_keep_their_live_sound():
    """They have never had these effects live and nobody has asked for them to change."""
    for style in OLDER:
        assert "live_fx" not in load_style(style)


@pytest.mark.parametrize("style", GENRES)
def test_live_fx_only_asks_for_what_the_band_can_do(style):
    doc = load_style(style)
    for role, fx in doc.get("live_fx", {}).items():
        assert role in doc["voices"] and set(fx) <= {"duck", "chorus", "drive"}


@pytest.mark.parametrize("style", GENRES)
def test_the_live_band_never_reads_the_master_chain(style):
    """PLAYING.md says so: reverb, delay and the master chain are only in plume render."""
    roles = {"kick": 1.0, "bass": 1.0, "pad": 1.0}
    a, _ = _plain_take(style, roles)
    b, _ = _plain_take(style, roles, master={"reverb": 1.0, "drive": 1.0, "lofi": 1.0})
    assert np.array_equal(a, b)


def test_live_jazz_kick_leans_ahead_as_it_does_offline():
    doc = load_style("jazz")
    assert doc["timing_bias_ms"]["kick"] == doc["timing_bias_ms"]["kit_kick"] < 0
    engine = _engine("jazz")
    engine._groove = dataclasses.replace(engine._groove, humanize_ms=0.0)
    bpm = 120.0
    plan, harm = _bar("jazz", _band_roles(engine), 1)
    engine.render_bar(plan, SimpleNamespace(bpm=bpm), harm)
    kicks = [e["onset_s"] for e in engine.last_events if e["role"] == "kick"]
    late = [1000.0 * (t - round(t * 2.0) / 2.0) for t in kicks if t > 0.1]
    assert late and all(ms == pytest.approx(-4.0, abs=0.1) for ms in late)


def test_the_live_band_voices_the_new_roles():
    assert "foot" in DuetEngine(style="jazz", seed=0).roles
    assert "tick" in DuetEngine(style="house", seed=0).roles


@pytest.mark.parametrize("style", GENRES)
@pytest.mark.parametrize("density", [0.3, 0.45, 0.9])
def test_a_live_band_is_finite_and_near_the_older_styles(style, density):
    _, y, _ = _play(style, bars=4, density=density)
    assert np.isfinite(y).all() and y.shape[1] == 2
    older = [_rms_db(_play(s, bars=4, density=density)[1]) for s in ("midwest", "synthwave")]
    assert min(older) - 5.0 <= _rms_db(y) <= max(older) + 1.0


@pytest.mark.parametrize("style", GENRES)
def test_a_live_bar_is_quick_to_render(style):
    """The planner renders a bar while the one before it plays."""
    engine = _engine(style)
    bpm = max(load_style(style)["bpm_range"])
    worst = 0.0
    for b in range(4):
        plan, harm = _bar(style, _band_roles(engine), b)
        t = time.perf_counter()
        engine.render_bar(plan, SimpleNamespace(bpm=float(bpm)), harm)
        worst = max(worst, time.perf_counter() - t)
    assert worst < 0.5 * 240.0 / bpm


def test_live_jazz_rides_in_triplets_and_keeps_the_foot_on_two_and_four():
    events, _, bpm = _play("jazz", bars=2, density=0.9)
    ride = _beats_of(events[1], "hat", bpm)
    skips = [b % 1.0 for b in ride if 0.2 < b % 1.0 < 0.9]
    # within the groove's small jitter
    assert skips and np.median(skips) == pytest.approx(2.0 / 3.0, abs=0.05)
    foot = _beats_of(events[1], "foot", bpm)
    assert [round(b) for b in foot] == [1, 3]


def test_live_jazz_walks_into_each_next_chord_of_its_progression():
    events, _, _ = _play("jazz", bars=8, density=0.5)
    symbols = progression_symbols("jazz", "C major")
    for b in range(7):
        bass = [e for e in events[b] if e["role"] == "bass"]
        assert len(bass) == 4
        assert bass[0]["midi"] % 12 == parse_chord(symbols[b % 4]).root
        next_root = parse_chord(symbols[(b + 1) % 4]).root
        assert min((bass[3]["midi"] - next_root) % 12, (next_root - bass[3]["midi"]) % 12) == 1


def test_live_jazz_comps_without_the_root_on_the_triplet_grid():
    events, _, bpm = _play("jazz", bars=8, density=0.9)
    symbols = progression_symbols("jazz", "C major")
    for b in range(8):
        comp = [e for e in events[b] if e["role"] == "pad"]
        assert comp
        assert parse_chord(symbols[b % 4]).root not in {e["midi"] % 12 for e in comp}
        for e in comp:
            at = e["onset_s"] * bpm / 60.0
            assert min(abs(at * 3 - round(at * 3)), 1) < 0.2


def test_live_house_holds_four_on_the_floor_at_every_density():
    for density in (0.45, 0.55, 0.9):
        events, _, bpm = _play("house", bars=1, density=density)
        kicks = _beats_of(events[0], "kick", bpm)
        assert [round(b) for b in kicks] == [0, 1, 2, 3]
        assert [round(b) for b in _beats_of(events[0], "snare", bpm)] == [1, 3]
        assert {round(b % 1.0, 1) for b in _beats_of(events[0], "hat", bpm)} == {0.5}


def _ninth_and_shell(sym):
    chord = parse_chord(sym)
    third = next(iv for iv in chord.intervals if iv in (3, 4))
    seventh = next(iv for iv in chord.intervals if iv in (10, 11))
    return {(chord.root + iv) % 12 for iv in (third, seventh)}, (chord.root + 14) % 12


def test_live_house_stabs_play_the_third_seventh_and_ninth_of_each_minor_ninth():
    symbols = progression_symbols("house", "A minor")
    engine = _engine("house")
    for b, sym in enumerate(symbols):
        plan, harm = _bar("house", _band_roles(engine), b)
        harm = SimpleNamespace(key="A minor", chord=sym, chord_conf=1.0)
        engine.render_bar(plan, SimpleNamespace(bpm=124.0), harm)
        stab = {e["midi"] % 12 for e in engine.last_events if e["role"] == "pad"}
        shell, ninth = _ninth_and_shell(sym)
        assert shell <= stab, sym
        if parse_chord(sym).quality == "m9":
            assert ninth in stab, sym


def test_offline_house_stabs_keep_the_ninth_too():
    session = _session("house")
    chords = session.movements[0].sections[0].chords
    assert any(parse_chord(c).quality == "m9" for c in chords)
    bar_s = 240.0 / session.bpm
    for b, sym in enumerate(chords):
        notes = {int(round(69 + 12 * np.log2(ev.freq_hz / 440.0))) % 12
                 for ev in compile_events(session) if ev.track == "pad"
                 and b * bar_s - 0.1 <= (ev.t_s % (len(chords) * bar_s)) < (b + 1) * bar_s - 0.1}
        if notes and parse_chord(sym).quality == "m9":
            assert _ninth_and_shell(sym)[1] in notes, sym


def test_live_house_bass_stays_off_the_kick():
    events, _, bpm = _play("house", bars=1)
    for b in _beats_of(events[0], "bass", bpm):
        assert abs(b % 1.0) > 0.2


def test_live_hip_hop_snare_is_late_and_its_bass_locks_to_the_kick():
    events, _, bpm = _play("hiphop", bars=4)
    for bar in events:
        late = [(b - round(b)) * 60.0 / bpm * 1000.0
                for b in _beats_of(bar, "snare", bpm) if abs(b - 1) < 0.3 or abs(b - 3) < 0.3]
        assert late and np.median(late) > 15.0
    kicks = {round(b * 4) for b in _beats_of(events[0], "kick", bpm)}
    bass = {round(b * 4) for b in _beats_of(events[0], "bass", bpm)}
    assert bass == kicks


def test_live_rnb_keys_play_the_upper_tones_of_its_extended_chords():
    events, _, _ = _play("rnb", bars=4)
    fm11 = next(s for s in progression_symbols("rnb", "C minor") if s.endswith("m11"))
    bar_i = progression_symbols("rnb", "C minor").index(fm11)
    keys = {e["midi"] % 12 for e in events[bar_i] if e["role"] == "keys"}
    chord = parse_chord(fm11)
    assert keys == {(chord.root + iv) % 12 for iv in chord.intervals[1:]}


@pytest.mark.parametrize("style", GENRES)
def test_the_jam_path_roles_sound_and_double_nothing(style):
    """A plain jam plans kick, snare, hat, bass, pad and keys, whatever the style calls
    its parts, so a style's defining sounds have to live on those roles."""
    roles = dict(mind_mod._ROLES["comping"])
    events, y, _ = _play(style, bars=2, roles=roles)
    assert np.isfinite(y).all()
    names = {e["role"] for e in events[1]}
    assert {"kick", "snare", "hat", "bass", "pad"} <= names
    if style != "rnb":
        assert "keys" not in names     # would only double the pad


def test_the_jam_path_keeps_the_signature_of_house_and_jazz():
    roles = dict(mind_mod._ROLES["comping"])
    events, _, bpm = _play("house", bars=1, roles=roles)
    assert {e["preset"] for e in events[0] if e["role"] == "snare"} == {"clap"}
    assert {round(b % 1.0, 1) for b in _beats_of(events[0], "hat", bpm)} == {0.5}
    events, _, _ = _play("jazz", bars=1, roles=roles)
    assert {e["preset"] for e in events[0] if e["role"] == "hat"} == {"kit_ride"}


@pytest.mark.parametrize("style", GENRES)
def test_a_style_with_no_chord_belief_still_plays(style):
    engine = _engine(style)
    plan, harm = _bar(style, _band_roles(engine), 0, chord=None)
    audio = engine.render_bar(plan, SimpleNamespace(bpm=100.0), harm)
    assert np.isfinite(audio).all() and np.abs(audio).max() > 0.01
    assert any(e["role"] == "bass" for e in engine.last_events)


@pytest.mark.parametrize("style", GENRES)
def test_a_live_bar_stays_the_same_on_the_same_seed(style):
    a = _play(style, bars=2)[1]
    b = _play(style, bars=2)[1]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("style", GENRES)
def test_the_offline_duet_runs_on_each_genre(tmp_path, style):
    from test_bandentry import _take
    wav = tmp_path / "in.wav"
    sf.write(wav, _take(seconds=20.0), SR, subtype="PCM_24")
    out = tmp_path / "duet.wav"
    simulate(str(wav), style=style, out_path=str(out), seed=1)
    y, _ = sf.read(out)
    assert np.isfinite(y).all() and 20.0 * np.log10(np.abs(y).max()) <= 0.0


# ------------------------------------------- a whole take, on a fake device

@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


def _take_rows(rec):
    return [json.loads(line) for line in (rec / "decisions.ndjson").read_text().splitlines()
            if '"entry"' in line]


def _fake_take(tmp_path, name, *, bpm=75.0, seconds=60.0, **kw):
    from test_bandentry import _take
    fake = fake_audio(_take(bpm=bpm, seconds=seconds, room_s=20.0), devices=RIG, rate_x=12.0)
    with patched_module("sounddevice", fake):
        realtime.lead(seed=0, tap_start=True, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / name), status=False,
                      socket_path=short_socket_path(name), max_minutes=0.7, direct=True,
                      input_device="audio interface", input_channel=1, **kw)
    return tmp_path / name


@pytest.mark.parametrize("prompt,style", [("jazz in C major", "jazz"),
                                          ("house in C major", "house"),
                                          ("boom bap in C major", "hiphop"),
                                          ("neo soul in C major", "rnb")])
def test_a_named_genre_plays_a_whole_tap_started_take(tmp_path, consent, prompt, style):
    rec = _fake_take(tmp_path, style, prompt=prompt)
    rows = _take_rows(rec)
    assert rows and {r["style"] for r in rows} == {style}
    assert not any(r.get("style_guess") for r in rows)
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert in_band, "the band never came in"
    symbols = set(progression_symbols(style, "C major"))
    assert {r["chord"] for r in in_band} <= symbols
    y, _ = sf.read(rec / "machine.wav")
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_a_guessed_genre_says_it_is_a_guess(tmp_path, consent):
    rec = _fake_take(tmp_path, "guess", bpm=85.0, detect=True, prompt="in A minor")
    rows = _take_rows(rec)
    assert rows and {r["style"] for r in rows} == {"hiphop"}
    assert all(r["style_guess"] for r in rows)
