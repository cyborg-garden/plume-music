"""The band keeps listening for the whole take, and the song yields to her.

A player who gets quieter and sparser should hear the band do the same, and a band
that is still listening, not one sticking with what it found at the start.  Three
things make that so, and each is tested here:

- her level is read the way a real player plays: it wanders a few dB from bar to bar,
  and a soft passage that wanders still brings the band down to meet her;
- the song's own shape gives way: a lift she plays softly under is played as a verse,
  with fewer parts, and a verse she drives is played as a lift;
- her chords are read from what a guitar really plays (two-note shapes and power
  chords), and a loop that does not read clean every time is still followed.

All pure except the last test, which drives the real transport on a fake device.
"""
import numpy as np
import pytest
from conftest import needs_laptop_speed  # noqa: E402

from plume import herchords as hc
from plume.bandentry import (FOLLOW_LEARN_BARS, SONG_ARC, BandFollow, PlanYield,
                             song_section)

NORMAL_DB, NORMAL_BUSY = -27.0, 2.8
#: Her bar-to-bar wander, in dB, as a real player's level moves: never the same twice,
#: sometimes back near her normal for a bar.  Mean -5.7 dB.
SOFT_WANDER = (-5, -2, -7, -5, -9, -2, -6, -9, -5, -5, -4, -8, -6, -4, -8, -3, -7, -6)
#: Going soft by degrees: a few bars a little under her normal, then well under it,
#: wandering 3 dB either side of -7.5.  Every other bar sits inside the deadband of
#: the shallow lean the first bars commit.
SOFT_BY_DEGREES = (-4.5, -4, -4) + (-9, -6) * 8
#: The same wander around her normal.
NORMAL_WANDER = (1, -2, 2, -1, 0, -2, 1, 2, -1, 0, -2, 1)


def learned():
    f = BandFollow(trusted=True)
    for _ in range(FOLLOW_LEARN_BARS):
        f.bar(NORMAL_DB, NORMAL_BUSY, playing=True)
    return f


def wander(f, offsets, busy=NORMAL_BUSY):
    return [f.bar(NORMAL_DB + d, busy, playing=True) for d in offsets]


# ------------------------------------------------------------ her level, as it really moves

def test_a_wandering_soft_passage_still_brings_the_band_down_to_meet_her():
    """A bar back near her normal starts the patience over, so a soft passage that
    wanders has to be caught some other way too."""
    f = learned()
    states = wander(f, SOFT_WANDER, busy=1.0)
    # Half of her mean drop, as for a steady drop (test_follow_faster), within a dB.
    assert states[-1].gain_db <= -2.0
    assert states[-1].density <= 0.8
    # ... and it got there while she was still soft, not at the end.
    assert states[11].gain_db <= -2.0


def test_going_soft_by_degrees_the_band_follows_her_all_the_way_down():
    """The band committed to her first, shallow drop, and every later bar that
    wandered back inside the deadband of that started its patience over: it stayed at
    -2 dB however long she sat 7 dB under her normal."""
    f = learned()
    states = wander(f, SOFT_BY_DEGREES, busy=1.0)
    # Her mean is -7.5: half of the first 6 dB and three quarters of the rest.
    assert states[-1].gain_db <= -3.5
    assert states[10].gain_db <= -3.0       # while she is still soft, not at the end
    assert states[-1].density <= 0.8


def test_after_a_wandering_soft_passage_the_band_comes_back_up_with_her():
    f = learned()
    wander(f, SOFT_WANDER, busy=1.0)
    back = wander(f, NORMAL_WANDER)
    assert back[-1].gain_db >= -0.5 and back[-1].density >= 0.95
    first = next(i for i, s in enumerate(back) if s.gain_db >= -0.5)
    assert first <= 6


def test_a_player_who_wanders_around_her_normal_moves_nothing():
    f = learned()
    assert all(s.gain_db == 0.0 and s.density == 1.0 for s in wander(f, NORMAL_WANDER * 3))


# ------------------------------------------------------------ the song yields to her

def _band(states, *, start=16):
    """The band's section, density and parts per bar: the song's shape, the yield, and
    the follow, in the planner's order."""
    y = PlanYield()
    out = []
    for i, s in enumerate(states):
        section, density, resting = song_section(start + i)
        got = y.bar(section, density, resting, gain_db=s.gain_db, density_x=s.density)
        out.append(got)
    return out


ALL_PARTS = ("kick", "snare", "hat", "openhat", "clap", "bass", "sub", "keys", "pad",
             "shimmer", "lead", "arp", "crackle")


def parts(got):
    return len([r for r in ALL_PARTS if r not in got.resting])


def test_the_song_holds_its_shape_while_she_does_not_depart():
    y = PlanYield()
    for bars_in in range(sum(b for _, b, _ in SONG_ARC)):
        section, density, resting = song_section(bars_in)
        got = y.bar(section, density, resting, gain_db=0.0, density_x=1.0)
        assert (got.section, got.density, got.resting) == (section, density, resting)
        assert got.drums == 1.0


def test_a_lift_she_plays_softly_under_becomes_a_verse_with_fewer_parts():
    lift = song_section(16)
    assert lift[0] == "lift"
    got = PlanYield().bar(*lift, gain_db=-3.0, density_x=0.8)
    assert got.section == "verse" and got.density < lift[1]
    assert parts(got) < parts(PlanYield().bar(*lift, gain_db=0.0, density_x=1.0)) - 1
    assert "kick" not in got.resting and "snare" not in got.resting    # the groove stays
    assert got.drums < 1.0                                             # ... lighter
    assert "softer" in got.reason


def test_much_softer_thins_the_band_further_but_never_drops_the_groove():
    verse = song_section(0)
    soft = PlanYield().bar(*verse, gain_db=-3.0, density_x=0.8)
    softer = PlanYield().bar(*verse, gain_db=-6.0, density_x=0.65)
    assert parts(softer) < parts(soft) and softer.drums < soft.drums < 1.0
    for r in ("kick", "snare", "bass"):
        assert r not in softer.resting


def test_a_verse_she_drives_becomes_a_lift():
    verse = song_section(0)
    got = PlanYield().bar(*verse, gain_db=2.0, density_x=1.2)
    assert got.section == "lift" and got.density > verse[1]


def test_a_lean_hovering_at_the_edge_does_not_flip_the_song_every_bar():
    y = PlanYield()
    lift = song_section(16)
    seq = [y.bar(*lift, gain_db=g, density_x=1.0).section
           for g in (-2.6, -1.8, -2.4, -1.6, -2.2, -1.9, -2.5, -1.7)]
    flips = sum(a != b for a, b in zip(seq, seq[1:]))
    assert flips <= 1, seq


def test_one_quiet_bar_changes_nothing_the_band_plays():
    f = learned()
    states = (wander(f, NORMAL_WANDER[:4]) + wander(f, [-12], busy=0.3)
              + wander(f, NORMAL_WANDER))
    neutral = _band([type(states[0])(0.0, 1.0, (), "")] * len(states))
    assert [(g.section, g.density, g.resting) for g in _band(states)] == \
           [(g.section, g.density, g.resting) for g in neutral]


def test_a_quiet_sparse_stretch_takes_parts_out_within_a_few_bars_and_back():
    """Scripted: her normal, eight soft sparse bars (8 dB down, a third of the notes),
    her normal again.  The band thins by at least two parts and 2 dB within three bars
    of her going soft, and is whole again within four of her coming back."""
    f = learned()
    states = (wander(f, [0] * 6) + wander(f, [-8] * 8, busy=1.0) + wander(f, [0] * 10))
    band = _band(states, start=16)                  # inside the first lift
    whole = _band([type(states[0])(0.0, 1.0, (), "")] * len(states), start=16)
    assert [parts(g) for g in band[:6]] == [parts(g) for g in whole[:6]]
    soft = range(6, 14)
    assert next(i for i in soft if parts(band[i]) <= parts(whole[i]) - 2) <= 6 + 2
    assert next(i for i in soft if states[i].gain_db <= -2.0) <= 6 + 2
    back = range(14, len(states))
    assert next(i for i in back if parts(band[i]) == parts(whole[i])) <= 14 + 4
    assert parts(band[-1]) == parts(whole[-1]) and states[-1].gain_db == 0.0


# ------------------------------------------------------------ her chords, as a guitar plays them

KEY, STYLE = "D major", "midwest"


def _chroma(weights: dict) -> np.ndarray:
    c = np.zeros(12)
    for pc, w in weights.items():
        c[pc % 12] += w
    return c / c.sum()


@pytest.mark.parametrize("root,fifth,sym", [(7, 2, "G"), (11, 6, "Bm"), (9, 4, "A"),
                                             (2, 9, "D")])
def test_a_power_chord_names_the_keys_own_chord_on_its_root(root, fifth, sym):
    cands = hc.candidates(KEY, STYLE)
    r = hc.read_bar(_chroma({root: 1.0, fifth: 0.6}), cands, notes=3.5)
    assert r.sym is not None and r.sym.startswith(sym), r
    assert r.sym in {c.sym for c in cands}


def test_two_strings_root_and_third_name_the_chord():
    r = hc.read_bar(_chroma({7: 1.0, 11: 0.7}), hc.candidates(KEY, STYLE), notes=3.0)
    assert r.sym is not None and r.sym.startswith("G"), r


def test_a_triad_is_enough_and_needs_no_seventh():
    r = hc.read_bar(_chroma({2: 1.0, 6: 0.8, 9: 0.9}), hc.candidates(KEY, STYLE))
    assert r.sym is not None and r.sym.startswith("D") and r.fit >= hc.FIT_MIN, r


def test_a_very_bright_lone_low_note_is_still_not_a_chord():
    """Twenty harmonics rolling off slowly: its own fifth reaches 0.37 of its root in
    the ear's chroma.  Still one note."""
    from test_herchords import _bars_of
    sr, bar_s = 48000, 2.0
    t = np.arange(int(bar_s * sr)) / sr
    for midi in (40, 45, 47):
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        one = sum(np.sin(2 * np.pi * f0 * h * t) / h ** 0.7 for h in range(1, 21)
                  if f0 * h < sr / 2) * np.exp(-t * 1.2)
        f = hc.ChordFollower()
        for heard, frames in enumerate(_bars_of(np.tile(0.05 * one, 10), bar_s)):
            f.hear(heard, hc.clean_bar(frames), playing=True)
            d = f.decide(heard + 2, KEY, STYLE, "Em7")
            assert d["source"] == "template", (midi, heard, d)


G5, D5, B5, A5 = ({7: 1.0, 2: 0.6}, {2: 1.0, 9: 0.6}, {11: 1.0, 6: 0.6}, {9: 1.0, 4: 0.6})
FUZZ = {0: 0.3, 5: 0.2, 10: 0.25}       # a bar that names nothing: between, or noise


def _follow(bars, *, lag=2):
    f = hc.ChordFollower()
    out = {}
    tmpl = ("Em7", "A7", "Dmaj7", "Dmaj7")
    for heard, w in enumerate(bars):
        f.hear(heard, None if w is None else hc.HerBar(_chroma(w), None, 3.5),
               playing=w is not None)
        out[heard + lag] = f.decide(heard + lag, KEY, STYLE, tmpl[(heard + lag) % 4])
    return out


def test_a_loop_that_misreads_once_each_time_round_is_still_followed():
    """Her four-bar loop of power chords, with one bar each time round that names
    nothing (a strum between two chords).  The strict loop never sees every place
    twice alike; the vote follows it, and plays her chord in time."""
    loop = (G5, D5, B5, A5)
    names = ("G", "D", "Bm", "A")
    bars = []
    for i in range(32):
        bars.append(FUZZ if i % 4 == (i // 4) % 4 else loop[i % 4])
    d = _follow(bars)
    later = [b for b in sorted(d) if 16 <= b < 34]
    followed = [b for b in later if d[b]["source"] == "her_loop"]
    assert len(followed) >= len(later) * 3 // 4, [d[b]["reason"] for b in later]
    for b in followed:
        assert d[b]["chord"].startswith(names[b % 4]), (b, d[b])


def test_one_contrary_bar_in_a_followed_loop_does_not_flip_the_band():
    loop = (G5, D5, B5, A5)
    bars = [loop[i % 4] for i in range(20)] + [A5] + [loop[i % 4] for i in range(21, 32)]
    d = _follow(bars)
    for b in range(22, 34):
        assert d[b]["source"] == "her_loop", (b, d[b])
        assert d[b]["chord"].startswith(("G", "D", "Bm", "A")[b % 4]), (b, d[b])


def test_bars_that_name_nothing_keep_the_template_without_flicker():
    d = _follow([FUZZ] * 16)
    assert all(x["source"] == "template" and x["chord"] == x["template"] for x in d.values())


def test_she_changes_her_loop_and_the_band_follows_the_new_one():
    """Listening the whole take long, not only at the start."""
    first = [(G5, D5, B5, A5)[i % 4] for i in range(24)]
    second = [(B5, G5, D5, A5)[i % 4] for i in range(24, 56)]
    d = _follow(first + second)
    names = ("Bm", "G", "D", "A")
    late = [b for b in range(44, 58)]
    assert all(d[b]["source"] == "her_loop" and d[b]["chord"].startswith(names[b % 4])
               for b in late), [(b, d[b]["chord"], d[b]["reason"]) for b in late]


# ------------------------------------------------------------ the real transport, no device

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    from plume import realtime
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


@needs_laptop_speed
def test_a_direct_take_thins_out_when_she_plays_softly_and_fills_back_in(tmp_path, consent):
    """The scripted quiet stretch on the fake device: 12 dB softer from 50 s to 80 s,
    then her normal again.  The band's level and parts come down within a few bars of
    her and go back up after her."""
    import json

    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take

    from plume import realtime
    from plume import roles as roles_mod

    bpm, soft_at, back_at = 115.0, 50.0, 80.0
    human = _take(bpm=bpm, seconds=130.0)
    human[int(soft_at * 48000):int(back_at * 48000)] *= 0.25
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=12.0)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("thin"), max_minutes=1.5,
                      input_device="audio interface", input_channel=1)
    rows = [json.loads(x) for x in (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int) and r.get("entry") == "full"
            and r["fade_db"] == 0.0]
    bar_s = 240.0 / bpm

    def n_parts(r):
        return len([v for v in r["roles"].values() if v > 0.05])

    before = [r for r in rows if soft_at - 6 * bar_s < r["t_s"] < soft_at]
    soft = [r for r in rows if soft_at + 4 * bar_s <= r["t_s"] < back_at]
    after = [r for r in rows if back_at + 5 * bar_s <= r["t_s"] < 120.0]
    assert before and soft and after, "the take did not reach every stretch"
    assert max(r["gain_db"] for r in soft) <= min(r["gain_db"] for r in before) - 2.0
    assert max(n_parts(r) for r in soft) <= min(n_parts(r) for r in before) - 1
    assert all(r["yield"]["lean"] in ("softer", "much softer") for r in soft)
    assert min(r["gain_db"] for r in after) > max(r["gain_db"] for r in soft)
    assert max(n_parts(r) for r in after) > max(n_parts(r) for r in soft)
