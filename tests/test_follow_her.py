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
from conftest import needs_laptop_speed, playhead_rate  # noqa: E402

from plume import herchords as hc
from plume.bandentry import (FOLLOW_LEARN_BARS, FOLLOW_NORMAL_SEED_BARS, SONG_ARC,
                             BandFollow, PlanYield, song_section, yield_to)

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
        got = y.bar(section, density, resting, gain_db=s.gain_db, density_x=s.density,
                    bars_in=start + i)
        out.append(got)
    return out


def _arc_run(offsets, busy=None, *, start=0, trusted=None):
    """Her bars (dB from her normal) through the follow and the yield, from ``start``
    bars after the band came in: ``(follow state, yield state, the arc's own)``, as the
    planner does it (:func:`plume.bandentry.yield_to`).  ``trusted``: per bar, whether
    the input still hears only her (a mic found to hold the band turns False)."""
    f = learned()
    y = PlanYield()
    out = []
    for i, d in enumerate(offsets):
        if trusted is not None and not trusted[i]:
            f.trusted = False
        s = f.bar(NORMAL_DB + d, NORMAL_BUSY if busy is None else busy[i], playing=True)
        own = song_section(start + i)
        out.append((s, yield_to(y, f, *own, bars_in=start + i), own))
    return out


def test_a_loud_sparse_player_keeps_a_full_band_and_its_sections():
    """Whole-note strumming at full level is not playing softer.  Fewer notes thin the
    band's own patterns, but the parts, the drums and the song's sections stay: only
    her level takes the band down."""
    for level in (0.0, 3.0):
        run = _arc_run(NORMAL_STRETCH + [level] * 32,
                       [NORMAL_BUSY] * len(NORMAL_STRETCH) + [0.7] * 32, start=12)
        for s, got, own in run:
            assert (got.section, got.resting) == (own[0], own[2]), (level, got)
            assert got.drums == 1.0 and got.lean in ("neutral", "louder"), (level, got)
        assert run[-1][0].density < 1.0          # ... but the patterns are thinner


#: Bars of her own normal before a passage: long enough to count as settled.
SETTLED = [0] * 8
#: A verse and more at her normal: what a passage after it is measured against.
NORMAL_STRETCH = [0] * 32


@pytest.mark.parametrize("offset", [4.0, -4.0, -9.0])
def test_an_offset_held_since_her_opening_bars_becomes_her_normal(offset):
    """Her first bars set her normal, and they can simply be off: a player warming up
    softly, or loudly, and then playing at her real level from there on.  An offset she
    has held since the band came in is her normal, at any depth, and the song's sections
    come back with it."""
    run = _arc_run([offset] * 112)
    own = sum(got.section == arc[0] for _, got, arc in run)
    assert own >= 0.8 * len(run), own
    assert run[-1][0].gain_db == 0.0 and run[-1][1].lean == "neutral"
    assert all(got.drums == 1.0 for _, got, _ in run[40:])


@pytest.mark.parametrize("depth", [-5.0, -6.5, -7.5, -9.0])
def test_a_long_soft_passage_after_her_normal_stays_followed_to_its_end(depth):
    """A passage is not a new normal.  After 32 bars at her normal, a soft passage of
    two song sections is followed for its whole length, whatever its depth, and when
    she comes back the band comes back to its own level, not past it."""
    n = len(NORMAL_STRETCH)
    run = _arc_run(NORMAL_STRETCH + [depth] * 32 + [0] * 16, start=0)
    soft = run[n + 4:n + 32]
    assert all(s.gain_db <= -2.0 for s, _, _ in soft), [s.gain_db for s, _, _ in soft]
    assert all(got.lean in ("softer", "much softer") for _, got, _ in soft)
    back = run[n + 32:]
    assert all(s.gain_db <= 0.5 for s, _, _ in back), [s.gain_db for s, _, _ in back]
    assert all(got.lean != "louder" for _, got, _ in back)
    assert all(got.section == arc[0] for _, got, arc in back[8:])


def test_a_swell_then_a_drop_is_answered_like_a_sudden_drop():
    """She plays louder for a few bars, then drops well under her normal.  That is one
    swell and one drop, not a player swinging back and forth: the band answers the
    drop as quickly and as deeply as it answers a sudden one."""
    sudden = _arc_run(SETTLED + [-8] * 14)
    swell = _arc_run(SETTLED[:5] + [4, 6, 6] + [-8] * 14)
    drop_at = len(SETTLED)
    target = sudden[-1][0].gain_db
    assert target <= -3.0
    reached = next(i for i in range(drop_at, len(swell)) if swell[i][0].gain_db <= target + 0.5)
    assert reached - drop_at <= 3, [s.gain_db for s, _, _ in swell]
    assert swell[-1][0].gain_db == pytest.approx(target, abs=0.1)
    assert swell[-1][1].lean == sudden[-1][1].lean


def test_a_noisy_drop_after_a_swell_is_not_held_at_the_average():
    """Real playing wanders: after a swell she drops 5-9 dB, with a bar now and then
    back near her normal.  The band must not lock onto the average of her loud bars
    and her soft ones: within four bars of the drop it has come down past -2 dB."""
    noisy = [-5, -2, -6, -5, -9, -2, -5, -9, -5, -5, -4, -8]
    run = _arc_run(SETTLED[:5] + [4, 6, 6] + noisy)
    drop_at = len(SETTLED)
    reached = next(i for i in range(drop_at, len(run)) if run[i][0].gain_db <= -2.0)
    assert reached - drop_at <= 4, [s.gain_db for s, _, _ in run]


def test_a_player_swinging_two_soft_two_loud_keeps_the_songs_shape():
    """Her level swings every two bars: the band does not pump with it, and an average
    dip that comes only from the swing does not hold the song down either.  Its lifts
    come, with every part, as written."""
    run = _arc_run(SETTLED + [-8, -8, 0, 0] * 16, start=0)
    late = run[24:]
    assert all(got.section == arc[0] and got.resting == arc[2] for _, got, arc in late)
    assert all(got.drums == 1.0 for _, got, _ in late)
    assert any(got.section == "lift" for _, got, _ in late)


def test_when_the_mic_turns_out_to_hold_the_band_the_song_comes_back_on_a_phrase_line():
    """A mic believed to hear only her can turn out to hear the band too, and from
    then on the song no longer yields.  Whatever had yielded comes back on the next
    phrase line, not on the bar the input was found out."""
    trusted = [True] * 22 + [False] * 10
    run = _arc_run([0] * 6 + [-8] * 26, start=8, trusted=trusted)
    assert run[21][1].resting != run[21][2][2]          # it had yielded by then
    for i in range(22, len(run)):
        if (8 + i) % 4:
            assert run[i][1].resting == run[i - 1][1].resting, i
    assert run[-1][1].resting == run[-1][2][2] and run[-1][1].drums == 1.0


def test_a_take_played_at_one_level_from_its_first_bars_is_her_normal():
    """Well under her first bars from the start, and for the whole take: the band
    comes down with her at first, but a take played at one level is her normal, and
    nothing leans for the rest of it, either way."""
    run = _arc_run([-12] * 60, [1.0] * 60)
    assert run[8][0].gain_db <= -5.0                       # followed at first
    assert run[-1][0].gain_db == 0.0 and run[-1][0].density == 1.0
    assert all(got.lean == "neutral" for _, got, _ in run[-16:])


def _opening(offsets):
    """Her bars from her very first (no learned normal first): the follow learns from
    the first of them, as it does over the drums."""
    f = BandFollow(trusted=True)
    y = PlanYield()
    out = []
    for i, d in enumerate(offsets):
        s = f.bar(NORMAL_DB + d, NORMAL_BUSY, playing=True)
        if i < FOLLOW_LEARN_BARS:
            continue
        bi = i - FOLLOW_LEARN_BARS
        own = song_section(bi)
        out.append((s, yield_to(y, f, *own, bars_in=bi), own))
    return out


@pytest.mark.parametrize("intro,depth", [(8, -6.0), (8, -10.0), (24, -6.0), (8, 6.0)])
def test_an_intro_off_her_real_level_gives_way_to_it(intro, depth):
    """A soft (or loud) intro, then her real level for the rest of the take.  The band
    reads her real level against the intro for a while (about as long as the intro and
    the bars its first bars count for), then it is her normal: no permanent lift or
    duck, and the song plays as written from there on."""
    run = _opening([depth] * intro + [0.0] * 72)
    leaning = [i for i, (_, got, _) in enumerate(run) if got.lean != "neutral"]
    assert leaning and leaning[-1] <= 2 * intro + FOLLOW_NORMAL_SEED_BARS + 4, leaning[-3:]
    tail = run[-24:]
    assert all(s.gain_db == 0.0 for s, _, _ in tail)
    assert all((got.section, got.resting, got.drums) == (arc[0], arc[2], 1.0)
               for _, got, arc in tail)


@pytest.mark.parametrize("depth", [-7.0, -12.0])
def test_a_soft_first_verse_then_her_chorus(depth):
    """Her normal over the drums, a soft first verse from the band's entry, then her
    chorus at her real level: the verse is followed, and the chorus ends up as her
    normal with the song as written, not lifted for the rest of the take."""
    run = _opening([0.0] * 4 + [depth] * 24 + [0.0] * 40)
    verse = run[4:16]
    assert all(s.gain_db <= -2.0 for s, _, _ in verse), [s.gain_db for s, _, _ in verse]
    tail = run[-16:]
    assert all(s.gain_db == 0.0 and got.lean == "neutral" for s, got, _ in tail)


def test_a_player_who_never_returns_to_her_opening_level():
    """Softer than her opening bars, then a little louder than that for good: that is
    where she plays, and nothing leans for the rest of the take."""
    run = _opening([-8.0] * 8 + [-4.0] * 56)
    assert all(s.gain_db == 0.0 and got.lean == "neutral" for s, got, _ in run[-20:])


def test_a_noisy_opening_then_a_soft_passage():
    """Her first bars wander a few dB, she settles, then plays a soft passage: the
    passage is followed, and the band is back with her after it."""
    noisy = [3, -2, 4, 0, -4, 3, -3, 4, 0, -2, 4, -3]
    run = _opening(noisy + [0] * 20 + [-7] * 24 + [0] * 16)
    soft = run[8 + 20 + 4:8 + 20 + 24]
    assert all(got.lean in ("softer", "much softer") for _, got, _ in soft)
    assert all(s.gain_db == 0.0 for s, _, _ in run[-6:])


@pytest.mark.parametrize("leg", [2, 3, 4])
def test_soft_and_loud_phrases_in_turn_neither_pump_nor_flip_the_parts(leg):
    """Call and response: ``leg`` soft bars, ``leg`` at her normal, over and over.  The
    band's level settles instead of pumping, and the song's parts are as written on
    her loud bars and her soft ones alike, never thin on her loud ones."""
    run = _arc_run(NORMAL_STRETCH + ([-8] * leg + [0] * leg) * 12, start=0)
    # Its first time round, a phrase of her soft can't yet be told from the start of a
    # soft verse; from the second it is a swing.
    late = run[len(NORMAL_STRETCH) + 16:]
    gains = [s.gain_db for s, _, _ in late]
    assert max(gains) - min(gains) <= 1.5, gains
    assert all((got.section, got.resting, got.drums) == (arc[0], arc[2], 1.0)
               for _, got, arc in late)


def test_a_verse_and_chorus_eight_bars_each_are_followed_at_most_a_phrase_late():
    """Eight soft bars, eight at her normal, in turn: a verse and a chorus.  The parts
    follow each change within a phrase of it."""
    n = len(NORMAL_STRETCH)
    run = _arc_run(NORMAL_STRETCH + ([-8] * 8 + [0] * 8) * 4, start=0)
    for k in range(4):
        soft, back = n + 16 * k, n + 16 * k + 8
        thin = [i for i in range(soft, soft + 16) if run[i][1].resting != run[i][2][2]]
        assert thin and thin[0] - soft <= 4 + 4, (k, thin)          # a phrase, plus her lag
        full = [i for i in range(back, min(back + 16, len(run)))
                if run[i][1].resting == run[i][2][2]]
        assert full and full[0] - back <= 4 + 4, (k, full)


def test_a_lean_that_flips_every_phrase_does_not_flip_the_parts_every_phrase():
    """Her lean held through a phrase, then the other way through the next, and so on:
    the song changes on a line only if it did not just flip the other way, so the
    parts never turn over phrase by phrase, a phrase late each time."""
    y = PlanYield(4)
    changes, last = 0, None
    for b in range(4, 52):
        gain = -3.0 if (b // 4) % 2 else 0.0
        got = y.bar(*song_section(0), gain_db=gain, bars_in=b)
        if last is not None and got.resting != last:
            changes += 1
            assert b % 4 == 0
        last = got.resting
    assert 0 < changes <= 6, changes          # 12 lines, at most every other one


def test_one_dip_and_back_is_not_a_swing():
    """One soft dip and back: the band comes back with her at once, not to an
    average of the dip and her normal."""
    f = learned()
    wander(f, [0] * 8 + [-8] * 3)
    back = wander(f, [0] * 6)
    assert next(i for i, s in enumerate(back) if s.gain_db == 0.0) <= 2


def test_a_passage_whose_depth_wobbles_is_still_a_passage():
    """Well under her normal, sometimes a little less so: never back at her normal, so
    never a swing.  The song yields for its whole length."""
    run = _arc_run(NORMAL_STRETCH + [-14, -14, -6, -6] * 6, start=0)
    soft = run[len(NORMAL_STRETCH) + 4:]
    assert all(got.lean in ("softer", "much softer") for _, got, _ in soft)
    assert all(s.gain_db <= -2.0 for s, _, _ in soft)


def test_a_swinging_player_who_then_goes_soft_is_followed():
    """She swings soft and loud, then goes soft and stays there: the band stops
    treating it as a swing and follows her down within a bar or two of a phrase."""
    run = _arc_run(NORMAL_STRETCH + [-8, -8, 0, 0] * 6 + [-10] * 16, start=0)
    start = len(NORMAL_STRETCH) + 24
    down = next(i for i in range(start, len(run)) if run[i][0].gain_db <= -4.0)
    assert down - start <= 8, [s.gain_db for s, _, _ in run[start:]]


def test_the_songs_sections_and_parts_change_only_on_phrase_lines():
    """A section or a part changing in the middle of a phrase was heard as the band
    changing tempo: what yields waits for the next phrase line.  The drums' level may
    still move on any bar."""
    run = _arc_run([0] * 4 + [-8] * 12 + [0] * 12, [NORMAL_BUSY] * 4 + [1.0] * 12
                   + [NORMAL_BUSY] * 12, start=13)
    moved = 0
    for i in range(1, len(run)):
        a, b = run[i - 1][1], run[i][1]
        if (a.section, a.resting) != (b.section, b.resting):
            moved += 1
            assert (13 + i) % 4 == 0, (13 + i, a, b)
    assert moved >= 2                      # it did yield, and come back


def test_two_soft_bars_then_two_loud_ones_do_not_make_the_band_pump():
    """A player alternating two soft bars and two loud ones: the band two bars behind
    would be fullest on her soft bars and thinnest on her loud ones.  It settles
    instead: its level steadies and the song does not flip back and forth."""
    run = _arc_run([-8, -8, 0, 0] * 10, start=12)
    late = run[12:]
    gains = [s.gain_db for s, _, _ in late]
    assert max(gains) - min(gains) <= 1.5, gains
    flips = sum((a[1].section, a[1].resting) != (b[1].section, b[1].resting)
                for a, b in zip(late, late[1:]) if a[2] == b[2])
    assert flips <= 1


def test_one_soft_bar_in_two_is_not_a_soft_passage():
    """Every other bar soft is her phrasing, not a quieter player."""
    f = learned()
    assert all(s.gain_db == 0.0 for s in wander(f, [0, -8] * 10))


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
    # The parts change on the line after a whole phrase of her soft: within two lines.
    assert next(i for i in soft if parts(band[i]) <= parts(whole[i]) - 2) <= 6 + 8
    assert next(i for i in soft if states[i].gain_db <= -2.0) <= 6 + 2
    back = range(14, len(states))
    assert next(i for i in back if parts(band[i]) == parts(whole[i])) <= 14 + 8
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


def test_a_strummed_chord_with_passing_notes_is_still_named():
    """A G strummed with a passing A and C#: the triad holds 0.57 of the bar, clearly
    ahead of any other chord."""
    c = _chroma({7: 1.0, 11: 0.75, 2: 0.85, 9: 1.0, 1: 0.95})
    r = hc.read_bar(c, hc.candidates(KEY, STYLE), notes=4.0)
    assert 0.55 <= r.fit < 0.6, r
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
    # Parts change on a phrase line after a whole phrase of her soft: by the second
    # line after she went soft.
    soft = [r for r in rows if soft_at + 9 * bar_s <= r["t_s"] < back_at]
    after = [r for r in rows if back_at + 9 * bar_s <= r["t_s"] < 120.0]
    assert before and soft and after, "the take did not reach every stretch"
    assert max(r["gain_db"] for r in soft) <= min(r["gain_db"] for r in before) - 2.0
    assert max(n_parts(r) for r in soft) <= min(n_parts(r) for r in before) - 1
    assert all(r["yield"]["lean"] in ("softer", "much softer") for r in soft)
    assert min(r["gain_db"] for r in after) > max(r["gain_db"] for r in soft)
    assert max(n_parts(r) for r in after) > max(n_parts(r) for r in soft)


def test_on_the_real_transport_the_song_yields_and_the_ending_does_not(tmp_path, consent):
    """The planner's own path, on the fake device, at a pace a shared runner keeps up
    with: she plays softly from 45 s, and the song ends on a timer.  The band comes
    down to meet her (lighter drums, the yield logged), and the ending's own shape is
    never yielded or ducked."""
    import json

    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take

    from plume import realtime
    from plume import roles as roles_mod

    bpm, soft_at, seconds = 115.0, 45.0, 100.0
    rate = playhead_rate(12.0)
    human = _take(bpm=bpm, seconds=seconds)
    human[int(soft_at * 48000):] *= 0.25            # 12 dB softer from here on
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=rate)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("yield"),
                      max_minutes=(seconds / rate + 8.0) / 60.0, wind_down_s=68.0,
                      input_device="audio interface", input_channel=1)
    rows = [json.loads(x) for x in (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int) and r.get("entry") == "full"]
    bar_s = 240.0 / bpm
    before = [r for r in rows if r["t_s"] < soft_at and r["section"] != "outro"]
    soft = [r for r in rows if soft_at + 5 * bar_s <= r["t_s"] and r["section"] != "outro"]
    ending = [r for r in rows if r["section"] == "outro"]
    assert before and soft and ending, "the take did not reach every stretch"
    assert all(r["yield"]["lean"] == "neutral" for r in before)
    assert all(r["yield"]["lean"] in ("softer", "much softer") for r in soft)
    assert max(r["roles"]["kick"] for r in soft) < min(r["roles"]["kick"] for r in before)
    assert max(r["gain_db"] for r in soft) < min(r["gain_db"] for r in before)
    for r in ending:                    # the ending is the song's own, whatever she does
        assert "yield" not in r and r["roles"].get("kick", 1.0) == 1.0, r
