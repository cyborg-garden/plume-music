"""A bump or a crackle on the input does not make the band swell.

A knock on the instrument's body or a bad cable puts a short burst at full scale on the
input.  Averaged into the bar it reads as a bar many dB louder, and a few of them would
lift and fill the band for no musical reason.  So the band follows her twice: once
hearing everything, as it always has, and once with the bars left out in which the
input clipped as she had not been playing.  It does what the first says, but goes no
further over its resting level and density than the second.

The guarantee tested here, over many players: on every bar the band's level lies
between its resting level and what hearing everything gives, and so does its density.
Leaving a clip out can take a swell away.  It cannot make the band quieter, sparser,
louder or busier than it would have been.

Synthetic signals only, and the listening is the live path's own: FeatureExtractor
frames, ``ClipWatch``, the readings the planner makes and ``FollowTwice``.  The last
test drives the real transport on a fake device.
"""
import numpy as np
import pytest

from plume import realtime
from plume.bandentry import BandFollow, FollowState, FollowTwice, swell_bound
from plume.listen.features import FeatureExtractor

SR, BPM, BLOCK = 48000, 150.0, 2048
BAR_S = 240.0 / BPM
SILENCE_RMS_DB = -45.0
#: Open chord shapes, and the same two octaves down, where a bass or a baritone plays.
CHORD = (130.81, 164.81, 196.0, 261.63, 329.63, 392.0)
LOW = (32.70, 41.20, 49.0, 65.41, 73.42, 82.41)
BARS, LEAN = 24, (12, 20)


def strikes(bars=BARS, *, peak, per_bar=4, decay=3.0, notes=CHORD, noise=False,
            accent=None, seed=7):
    """``bars`` of one sound struck ``per_bar`` times a bar over a little hiss, each
    strike peaking at ``peak`` and dying away at ``decay`` per second.  ``noise``: a
    drum or a slap, not a pitch.  ``accent``: (which strike of the bar, its peak as a
    function of the bar)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(min(1.5, 8.0 / decay) * SR)) / SR
    if noise:
        one = rng.standard_normal(t.size)
    else:
        one = sum(np.sin(2 * np.pi * f * t) / (1 + 0.3 * j) for j, f in enumerate(notes))
    one = one * np.exp(-t * decay) * np.minimum(1.0, t / 0.003)
    one = (one / np.abs(one).max()).astype(np.float32)
    x = rng.standard_normal(int(bars * BAR_S * SR)).astype(np.float32) * 1e-4
    for k in range(int(bars * per_bar)):
        i = int(k * BAR_S / per_bar * SR)
        seg = one[: x.size - i]
        amp = accent[1](k // per_bar) if accent and k % per_bar == accent[0] else peak
        x[i:i + seg.size] += amp * seg
    return x


def louder(x, from_bar, db, *, to_bar=None):
    y = x.copy()
    a = int(from_bar * BAR_S * SR)
    b = y.size if to_bar is None else int(to_bar * BAR_S * SR)
    y[a:b] *= 10.0 ** (db / 20.0)
    return y


def thump(x, at_s, *, seconds=0.2, hz=40.0, release_s=0.015):
    """A knock on the body: a low boom far past full scale, so the input clips it flat
    from rail to rail for ``seconds``, then damped."""
    y = x.copy()
    t = np.arange(int((seconds + 10 * release_s) * SR)) / SR
    env = np.where(t < seconds, 1.0, np.exp(-(t - seconds) / release_s))
    boom = 6.0 * np.sin(2 * np.pi * hz * t) * env * np.minimum(1.0, t / 0.01)
    i = int(at_s * SR)
    y[i:i + t.size] += boom.astype(np.float32)[: y.size - i]
    return np.clip(y, -1.0, 1.0)


def crackle(x, at_s, *, clicks, over_s):
    """A bad cable: scattered clicks, each a millisecond or a few at full scale."""
    y = x.copy()
    rng = np.random.default_rng(5)
    for off in np.sort(rng.uniform(0.0, over_s, clicks)):
        i = int((at_s + off) * SR)
        y[i:i + int(rng.integers(48, 193))] = rng.choice((-1.0, 1.0))
    return y


def follow(x, bars=BARS):
    """Bar by bar, what the band does with ``x`` (``FollowTwice``, fed as the planner
    feeds it) beside what it always did: one plain ``BandFollow`` on every reading as
    it is, with no clip looked at.  Rows of ``(now, ever)`` FollowStates."""
    x = np.clip(x, -1.0, 1.0)
    extractor, clips = FeatureExtractor(sr=SR), realtime.ClipWatch()
    events, clipped = [], []
    for i in range(0, x.size, BLOCK):
        clipped.extend(clips.process(x[i:i + BLOCK]))
        events.extend((f.t_s, f.rms_db, f.onset) for f in extractor.process(x[i:i + BLOCK]))
    twice, plain = FollowTwice(trusted=True), BandFollow(trusted=True)
    ref_db, fed, rows = None, None, []
    for n in range(bars - 1):
        w0, w1, nxt = n * BAR_S, (n + 1) * BAR_S, (n + 2) * BAR_S
        evs = [e for e in events if w0 <= e[0] < w1]
        later = [e for e in events if w1 <= e[0] < nxt]
        kw = realtime.bar_features_dict(n, evs, silence_rms_db=SILENCE_RMS_DB, ref_db=ref_db)
        if not kw["silent"]:
            ref_db = kw["rms_db"] if ref_db is None else 0.5 * kw["rms_db"] + 0.5 * ref_db
        halves = dict(fed_until=fed, heard_until=w1 + 0.5 * BAR_S + 0.1,
                      silence_rms_db=SILENCE_RMS_DB, ref_db=ref_db)
        parts, fed = realtime.follow_parts(evs, later, w0, w1, nxt, **halves)
        spared_parts, _ = realtime.follow_parts(evs, later, w0, w1, nxt, clipped=clipped,
                                                **halves)
        same = dict(playing=not kw["silent"], parts=parts, bar_s=BAR_S)
        now = twice.bar(kw["rms_db"], kw["onset_density"], **same,
                        spared=(*realtime.follow_bar(kw, clipped, w0, w1), spared_parts))
        rows.append((now, plain.bar(kw["rms_db"], kw["onset_density"], **same)))
        assert twice.yield_db in (0.0, twice.gain_db), n
    follow.twice = twice
    return rows, clipped


def bounded(rows):
    """THE guarantee, on every bar: the band's level and density between resting and
    what hearing everything gives.  Returns the bars on which a swell was held back."""
    held = []
    for n, (now, ever) in enumerate(rows):
        assert min(ever.gain_db, 0.0) <= now.gain_db <= max(ever.gain_db, 0.0), (n, now, ever)
        assert min(ever.density, 1.0) <= now.density <= max(ever.density, 1.0), (n, now, ever)
        if (now.gain_db, now.density) != (ever.gain_db, ever.density):
            held.append(n)
    return held


def still(states):
    return all(s.gain_db == 0.0 and s.density == 1.0 and not s.moved for s in states)


def lean_accent(to, was=0.25):
    return lambda bar: to if LEAN[0] <= bar < LEAN[1] else was


# ------------------------------------------------------------ a quiet player, and bursts

@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
def test_knocks_over_a_couple_of_bars_do_not_lift_or_fill_the_band(notes):
    """A few knocks running would be a louder, busier passage, were they her."""
    x = strikes(peak=0.15, notes=notes)
    for off in (0.5, 1.3, 2.1, 2.9, 3.7):
        x = thump(x, LEAN[0] * BAR_S + off)
    rows, clipped = follow(x)
    assert clipped and bounded(rows)
    assert still(now for now, _ in rows) and follow.twice.spared is not None
    assert max(ever.gain_db for _, ever in rows) >= 1.5, "the knocks alone lift nothing"


def test_crackle_over_a_few_bars_does_not_lift_the_band():
    x = crackle(strikes(peak=0.15), LEAN[0] * BAR_S + 0.2, clicks=16, over_s=4 * BAR_S)
    rows, clipped = follow(x)
    assert clipped and bounded(rows)
    assert still(now for now, _ in rows)
    assert max(ever.gain_db for _, ever in rows) >= 1.0, "the crackle alone lifts nothing"


def test_a_swell_held_back_says_so():
    x = strikes(peak=0.15)
    for off in (0.5, 1.3, 2.1, 2.9, 3.7):
        x = thump(x, LEAN[0] * BAR_S + off)
    rows, _ = follow(x)
    n = bounded(rows)[0]
    now, ever = rows[n]
    assert "louder" in ever.moved or ever.gain_db > 0.0
    assert now.moved == () and "held back" in now.reason and "held back" not in ever.reason
    # and a move it logs is one the band made
    was = FollowState(0.0, 1.0, (), "")
    for now, _ in rows:
        assert "louder" not in now.moved or (now.gain_db > was.gain_db
                                             or now.density > was.density)
        assert "softer" not in now.moved or (now.gain_db < was.gain_db
                                             or now.density < was.density)
        was = now


def test_a_real_lean_in_is_followed_as_ever_through_a_knock():
    """A knock in a passage she really plays louder takes nothing away that she did not
    lose to the knock itself: the band still lifts, and never past where it would."""
    lean = louder(strikes(peak=0.15), LEAN[0], 6.0)
    clean, _ = follow(lean)
    assert [now for now, _ in clean] == [ever for _, ever in clean]
    rows, clipped = follow(thump(lean, (LEAN[0] + 4) * BAR_S + 0.5))
    assert clipped
    bounded(rows)
    assert clean[-1][0].gain_db >= 2.0
    assert rows[-1][0].gain_db == pytest.approx(clean[-1][0].gain_db, abs=0.25)


# ------------------------------------------------------------ a player mistaken for a burst

@pytest.mark.parametrize("decay, on", [(8.0, 2), (8.0, 3), (20.0, 3)])
def test_one_accent_a_bar_leaping_to_full_scale_never_brings_the_band_down(decay, on):
    """Soft strikes on every beat and one accent a bar, which clips for eight bars.
    With those bars left out, only the soft halves heard early would speak, and alone
    they say she went softer.  The band does what it always did."""
    rows, clipped = follow(strikes(peak=0.06, decay=decay, accent=(on, lean_accent(1.3))))
    assert clipped
    bounded(rows)
    assert min(now.gain_db for now, _ in rows) >= min(ever.gain_db for _, ever in rows)


@pytest.mark.parametrize("pattern, down_db", [((1, 0), -4.0), ((1, 0), -8.0),
                                              ((1, 1, 0), -6.0)])
def test_clipped_bars_between_softer_ones_never_bring_the_band_down(pattern, down_db):
    """Full-scale bars and softer bars in turn: with the loud ones left out the soft
    ones would speak alone."""
    x = strikes(peak=0.2)
    for bar in range(LEAN[0], LEAN[1] + 2):
        loud = pattern[(bar - LEAN[0]) % len(pattern)]
        x = louder(x, bar, 16.0 if loud else down_db, to_bar=bar + 1)
    rows, clipped = follow(x)
    assert clipped
    bounded(rows)


@pytest.mark.parametrize("sound", [dict(decay=3.0), dict(decay=20.0),
                                   dict(per_bar=8, decay=30.0, noise=True)],
                         ids=["ringing", "staccato", "percussion"])
def test_a_quiet_player_leaping_to_full_scale_is_lifted_late_or_not_at_all_never_down(sound):
    x = louder(strikes(peak=0.12, **sound), LEAN[0], 20.0, to_bar=LEAN[1])
    rows, clipped = follow(x)
    assert clipped
    bounded(rows)
    assert min(now.gain_db for now, _ in rows[LEAN[0]:LEAN[1]]) >= 0.0


# ------------------------------------------------------------ a player who plays hot

@pytest.mark.parametrize("sound, up_db", [
    (dict(peak=0.6, decay=20.0), 6.0),
    (dict(peak=0.6, per_bar=16, decay=40.0, noise=True), 6.0),
    (dict(peak=0.55, decay=5.0, notes=LOW), 6.0),
    (dict(peak=1.4, decay=3.0), 6.0),
    (dict(peak=0.6, per_bar=1, decay=3.0), 6.0),
    (dict(peak=0.32, decay=3.0), 11.0),
], ids=["staccato", "percussion in sixteenths", "low register", "always clipping",
        "one strike a bar", "only just hot"])
def test_a_player_whose_own_strikes_clip_is_followed_exactly_as_ever(sound, up_db):
    x = louder(strikes(**sound), LEAN[0], up_db, to_bar=LEAN[1])
    assert np.abs(x).max() >= 1.0, "the lean never reached full scale"
    rows, clipped = follow(x)
    assert not clipped and bounded(rows) == []
    assert all(now == ever for now, ever in rows)
    assert follow.twice.spared is None, "with no clip marked there is one follower"
    assert any("louder" in now.moved for now, _ in rows) or sound.get("per_bar") == 1


# ------------------------------------------------------------ whoever plays

@pytest.mark.parametrize("seed", range(20))
def test_whoever_plays_the_band_is_between_resting_and_what_it_always_did(seed):
    """Players drawn at random: quiet, hot and hovering between; ringing and dry; dense
    and sparse; an accent that leaps; a lean that clips; loud bars between softer ones;
    knocks; crackle."""
    rng = np.random.default_rng(100 + seed)
    bars = 20
    peak = float(rng.choice([0.06, 0.1, 0.15, 0.25, 0.3, 0.35, 0.6]))
    sound = dict(peak=peak, per_bar=int(rng.choice([1, 2, 4, 8, 16])),
                 decay=float(rng.choice([3.0, 8.0, 20.0, 40.0])),
                 noise=bool(rng.integers(2)), notes=CHORD if rng.integers(2) else LOW,
                 seed=seed)
    if rng.integers(2):
        to, was = float(rng.choice([0.5, 1.3, 2.0])), float(rng.choice([0.25, peak]))
        sound["accent"] = (int(rng.integers(sound["per_bar"])),
                           lambda bar: to if 10 <= bar < 16 else was)
    x = strikes(bars, **sound)
    if seed % 4 == 0:                              # hovering about the threshold
        for n, p in enumerate(rng.uniform(0.25, 0.35, bars)):
            x = louder(x, n, 20.0 * np.log10(p / peak), to_bar=n + 1)
    shape = int(rng.integers(4))
    if shape == 1:                                 # a lean that clips
        x = louder(x, 10, float(rng.choice([6.0, 12.0, 20.0])), to_bar=16)
    elif shape == 2:                               # loud bars between softer ones
        for bar in range(10, 18):
            up = (bar - 10) % int(rng.choice([2, 3])) == 0
            x = louder(x, bar, 20.0 if up else float(rng.choice([-4.0, -8.0])),
                       to_bar=bar + 1)
    for _ in range(int(rng.integers(0, 5))):
        x = thump(x, float(rng.uniform(1.0, bars * BAR_S - 1.0)),
                  seconds=float(rng.choice([0.05, 0.15, 0.3, 0.6])))
    if rng.integers(2):
        x = crackle(x, float(rng.uniform(1.0, bars * BAR_S - 8.0)),
                    clicks=int(rng.integers(2, 13)), over_s=float(rng.uniform(0.5, 7.0)))
    bounded(follow(x, bars)[0])


@pytest.mark.parametrize("heard, spared, rest, want", [
    (3.0, 0.0, 0.0, 0.0), (3.0, 1.5, 0.0, 1.5), (3.0, -4.0, 0.0, 0.0), (1.5, 3.0, 0.0, 1.5),
    (-4.0, 0.0, 0.0, -4.0), (-4.0, -9.0, 0.0, -4.0), (0.0, 3.0, 0.0, 0.0),
    (1.3, 0.7, 1.0, 1.0), (1.3, 1.1, 1.0, 1.1), (0.7, 1.3, 1.0, 0.7)])
def test_the_bound_takes_a_swell_away_and_nothing_else(heard, spared, rest, want):
    assert swell_bound(heard, spared, rest) == want


def test_follow_twice_with_nothing_spared_is_band_follow():
    """Fed the same twice, on any player: state for state what one follower says."""
    rng = np.random.default_rng(3)
    twice, plain = FollowTwice(trusted=True), BandFollow(trusted=True)
    for n in range(60):
        db, busy = -27.0 + float(rng.normal(0, 5)), 2.8 + float(rng.normal(0, 1))
        kw = dict(playing=n % 9 != 0, parts=(db - 1.0, db + 1.0), bar_s=2.0)
        a = twice.bar(db, busy, spared=(db, busy, kw["parts"]) if n % 2 else None, **kw)
        assert a == plain.bar(db, busy, **kw) and twice.held_back is None
        if n == 30:
            assert twice.cue("sparser") == plain.cue("sparser")
    assert (twice.gain_db, twice.density, twice.yield_db, twice.hold_bars) == (
        plain.gain_db, plain.density, plain.yield_db, plain.hold_bars)
    assert isinstance(a, FollowState)


NORMAL = (-27.0, 1.0)


def _swelling(twice, *, bars=6, plain=None):
    """``twice`` after her normal and then ``bars`` of knocks: hearing everything reads
    them 12 dB up, and they are left out of the second reading.  ``plain`` hears
    everything alongside."""
    def bar(db, spared):
        kw = dict(playing=True, parts=(db, db), bar_s=2.0)
        if plain is not None:
            plain.bar(db, NORMAL[1], **kw)
        return twice.bar(db, NORMAL[1], spared=spared, **kw)
    for _ in range(8):
        bar(NORMAL[0], None)
    for _ in range(bars):
        state = bar(NORMAL[0] + 12.0, (None, None, (None, None)))
    return state


def test_a_swell_held_back_does_not_lift_the_songs_shape_either():
    """The song's shape reads ``yield_db``: a verse must not become a lift for knocks."""
    twice = FollowTwice(trusted=True)
    _swelling(twice)
    assert twice.heard.gain_db > 1.0 and twice.heard.yield_db == twice.heard.gain_db
    assert (twice.gain_db, twice.yield_db) == (0.0, 0.0) and twice.held_back is not None


def test_yield_is_the_bands_lean_or_nothing_whatever_is_left_out():
    """As it is for one follower: the lean, or none while she swings soft and loud."""
    rng = np.random.default_rng(11)
    for take in range(20):
        twice = FollowTwice(trusted=True)
        for n in range(60):
            db = NORMAL[0] + float(rng.choice([-9.0, -4.0, 0.0, 0.0, 5.0, 12.0])) + float(
                rng.normal(0, 1.5))
            kw = dict(playing=True, parts=(db, db), bar_s=2.0)
            out = n > 8 and rng.random() < 0.4
            twice.bar(db, NORMAL[1], spared=(None, None, (None, None)) if out else None, **kw)
            assert twice.yield_db in (0.0, twice.gain_db), (take, n)
            assert min(twice.heard.gain_db, 0.0) <= twice.gain_db <= max(twice.heard.gain_db, 0.0)


def test_her_own_word_reaches_both_followers():
    """A "sparser" cue beats following, and holds for some bars.  The second follower
    is told too, or it would go on filling out and be no bound on the first when the
    hold ends."""
    twice, alone = FollowTwice(trusted=True), BandFollow(trusted=True)
    _swelling(twice, bars=1)                       # a second follower now exists
    for db, busy in [(NORMAL[0], 2.2)] * 8:        # and she plays busier, to both
        kw = dict(playing=True, parts=(db, db), bar_s=2.0)
        twice.bar(db, busy, spared=(db, busy, kw["parts"]), **kw)
    assert twice.spared.density > 1.0 and twice.density > 1.0
    twice.cue("sparser")
    assert twice.heard.density == twice.spared.density == twice.density == 1.0
    kw = dict(playing=True, parts=(NORMAL[0],) * 2, bar_s=2.0)
    twice.bar(NORMAL[0], 2.2, spared=(NORMAL[0], 2.2, kw["parts"]), **kw)
    assert twice.spared.density == 1.0, "the cue holds in the second follower too"
    assert alone.cue("sparser") is not None


def test_on_a_room_mic_there_is_one_follower_and_nothing_is_ever_held():
    """A room mic's follower takes its own lean back off what it hears.  A swell held
    back would be taken off bars the band never played it in, so there it is main's
    follower and no more, whatever is said to have clipped."""
    twice, plain = FollowTwice(trusted=False), BandFollow(trusted=False)
    rng = np.random.default_rng(4)
    for n in range(80):
        db, busy = NORMAL[0] + float(rng.normal(0, 6)), NORMAL[1]
        kw = dict(playing=True, bar_s=2.0, band_in=n > 3)
        out = (None, None, None) if n % 3 == 0 else None
        assert twice.bar(db, busy, spared=out, **kw) == plain.bar(db, busy, **kw), n
        assert twice.spared is None and twice.held_back is None
        assert (twice.gain_db, twice.density, twice.yield_db) == (
            plain.gain_db, plain.density, plain.yield_db)
    assert any(s != 0.0 for s in (plain.gain_db, plain.density - 1.0)) or plain._learned


def test_a_take_that_stops_believing_the_input_is_a_room_mics_from_that_bar_on():
    """Mid-swell, with one held back: the second follower goes, nothing is held, and
    every bar from there is what a new room mic's follower says.  It never goes back."""
    twice = FollowTwice(trusted=True)
    _swelling(twice)
    assert twice.spared is not None and twice.held_back is not None
    twice.distrust()
    room = BandFollow(trusted=False)
    assert (twice.trusted, twice.spared, twice.held_back) == (False, None, None)
    assert (twice.gain_db, twice.density, twice.yield_db) == (0.0, 1.0, 0.0)
    rng = np.random.default_rng(6)
    for n in range(40):
        db = NORMAL[0] + float(rng.normal(0, 6))
        kw = dict(playing=True, bar_s=2.0)
        assert twice.bar(db, NORMAL[1], spared=(None, None, None), **kw) == room.bar(
            db, NORMAL[1], **kw), n
        assert twice.spared is None and twice.held_back is None


# ------------------------------------------------------------ whose clip it is

def tone(seconds, peak):
    t = np.arange(int(seconds * SR)) / SR
    return (peak * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _heard(watch, peak, seconds=12.0):
    """``watch`` after ``seconds`` of a player whose playing peaks at ``peak``."""
    watch.process(np.clip(tone(seconds, peak), -1.0, 1.0))
    return watch


BURST = np.ones(realtime.HOP, dtype=np.float32)


def test_clip_watch_names_the_hops_at_full_scale_however_the_stream_is_cut():
    x = np.zeros(8 * realtime.HOP, dtype=np.float32)
    x[3 * realtime.HOP + 7] = -1.0
    x[5 * realtime.HOP - 1] = 0.995
    x[6 * realtime.HOP] = 0.9                      # loud, not clipped
    want = [12.0 + 4 * realtime.HOP / SR, 12.0 + 5 * realtime.HOP / SR]
    for block in (64, 700, x.size):
        watch = _heard(realtime.ClipWatch(), 0.15)
        got = [t for i in range(0, x.size, block) for t in watch.process(x[i:i + block])]
        assert got == pytest.approx(want), block


@pytest.mark.parametrize("peak, marked", [(0.1, True), (0.28, True), (0.32, False),
                                          (0.6, False), (1.3, False)])
def test_a_clip_is_marked_only_when_full_scale_is_out_of_character(peak, marked):
    watch = _heard(realtime.ClipWatch(), peak)
    assert watch.typical_peak == pytest.approx(min(peak, 1.0), rel=1e-3)
    assert bool(watch.process(BURST)) is marked


def test_before_she_has_been_heard_for_long_nothing_is_marked():
    watch = _heard(realtime.ClipWatch(), 0.1, seconds=8.0)
    assert watch.typical_peak is None and watch.process(BURST) == []
    assert _heard(watch, 0.1, seconds=2.2).process(BURST)


def test_a_rest_is_not_her_playing_softly():
    """A player who plays near full scale and rests is the same player after it."""
    watch = _heard(realtime.ClipWatch(), 0.6)
    watch.process(np.zeros(40 * SR, dtype=np.float32))
    watch.process(tone(40.0, 0.005))               # the hiss of a rest, too
    assert watch.typical_peak == pytest.approx(0.6, rel=1e-3)
    assert watch.process(BURST) == []


def test_bumps_in_a_rest_are_not_her_playing_loudly():
    """Nine knocks while she sits out for half a minute: full scale is still not hers."""
    watch = _heard(realtime.ClipWatch(), 0.15, seconds=30.0)
    rest = np.zeros(36 * SR, dtype=np.float32)
    for k in range(9):
        rest = thump(rest, 1.0 + 4.0 * k, seconds=0.25)
    assert len(watch.process(rest)) > 9
    assert watch.typical_peak == pytest.approx(0.15, rel=1e-3)


def test_a_few_bursts_while_she_plays_do_not_make_full_scale_her_own():
    """Bursts in a third of the last half minute, then a run of four, and a clip is
    still not hers."""
    watch = _heard(realtime.ClipWatch(), 0.15, seconds=30.0)
    x = tone(2.0, 0.15)
    for i in range(19):
        watch.process(thump(x, 1.0, seconds=0.1) if i % 3 == 0 or i >= 15 else x)
    assert watch.typical_peak == pytest.approx(0.15, rel=1e-3)
    assert watch.process(BURST)


def test_a_player_who_clips_most_of_the_time_plays_that_loud():
    watch = _heard(realtime.ClipWatch(), 0.15, seconds=30.0)
    _heard(watch, 1.3, seconds=16.0)
    assert watch.typical_peak == pytest.approx(1.0)
    assert watch.process(BURST) == []


def test_a_clip_is_in_the_frames_that_hold_it_and_a_moment_after():
    clipped = [10.0]
    assert realtime.clip_in(clipped, 9.0, 10.01) and realtime.clip_in(clipped, 10.0, 11.0)
    assert realtime.clip_in(clipped, 10.05, 11.0)      # still in the window, or ringing
    assert not realtime.clip_in(clipped, 9.0, 10.0)    # a strike on the bar line is the
    assert not realtime.clip_in(clipped, 10.2, 11.0)   # new bar's, not the old one's
    assert not realtime.clip_in([], 9.0, 11.0)


def test_the_spared_reading_of_a_bar_is_the_bar_or_nothing():
    kw = {"rms_db": -20.0, "onset_density": 1.5}
    assert realtime.follow_bar(kw, [5.0], 0.0, 2.0) == (-20.0, 1.5)
    assert realtime.follow_bar(kw, [1.0], 0.0, 2.0) == (None, None)
    assert realtime.follow_bar(None, [], 0.0, 2.0) == (None, None)
    evs = [(t, -20.0, False) for t in np.arange(0.0, 3.0, 0.02).tolist()]
    same = dict(fed_until=None, heard_until=3.0, silence_rms_db=SILENCE_RMS_DB, ref_db=None)
    assert realtime.follow_parts(evs, [], 0.0, 3.0, 6.0, **same)[0] == pytest.approx(
        [-20.0, -20.0])
    assert realtime.follow_parts(evs, [], 0.0, 3.0, 6.0, clipped=[2.9], **same)[0] == [
        None, None]


# ------------------------------------------------------------ the real transport, no device

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


def _stop_once_planned(sock, decisions, until_s, *, wall_s=150.0):
    """Stop the take once the planner has logged a bar past ``until_s`` of it, however
    long that takes the machine: the take ends by where it is, not by the clock.  (The
    fake device can run ahead of a planner on a busy machine, so it is the planner's
    place that counts.)  Returns the thread and a list holding True once it got there."""
    import json
    import threading

    from conftest import wait_until
    got = []

    def there():
        try:
            last = decisions.read_text().splitlines()[-1]
            return float(json.loads(last).get("t_s") or 0.0) >= until_s
        except (OSError, IndexError, ValueError):
            return False

    def run():
        got.append(wait_until(there, timeout=wall_s, poll=0.05))
        try:
            realtime.send_stop(socket_path=sock)
        except realtime.RealtimeError:
            pass
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, got


def test_on_the_real_transport_thumps_and_crackle_do_not_lift_the_band(tmp_path, consent):
    """The planner's own path, on the fake device: a steady player whose input is
    knocked and crackles once the band is in.  The band never leans louder."""
    import json

    from conftest import fake_audio, patched_module, playhead_rate, short_socket_path
    from test_bandentry import _take

    from plume import roles as roles_mod

    bpm, seconds = 115.0, 100.0
    bar_s = 240.0 / bpm
    human = _take(bpm=bpm, seconds=seconds)
    hits = (50.3, 52.9, 61.2, 70.4, 71.1)
    for at in hits:
        human = thump(human, at, seconds=0.25)
    human = crackle(human, 56.0, clicks=8, over_s=4.0)
    sock = short_socket_path("thump")
    decisions = tmp_path / "rec" / "decisions.ndjson"
    stopper, got = _stop_once_planned(sock, decisions, hits[-1] + 6 * bar_s)
    with patched_module("sounddevice",
                        fake_audio(human, devices=RIG, rate_x=playhead_rate(6.0))):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=sock, max_minutes=3.0,
                      input_device="audio interface", input_channel=1)
    stopper.join(timeout=5.0)
    if got != [True]:
        pytest.fail("timed out: the machine never played the take past its last knock")
    rows = [json.loads(x) for x in decisions.read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int) and r.get("follow")]
    assert [r for r in rows if r["t_s"] > hits[-1] + 4 * bar_s], rows[-1]["t_s"]
    assert all("louder" not in (r["follow"].get("moved") or ()) for r in rows)
    assert max(r["follow"]["gain_db"] for r in rows) == 0.0
    assert any("held_back" in r["follow"] for r in rows), "nothing was ever held back"


def test_only_an_input_that_hears_only_her_is_watched_for_clips(consent, monkeypatch):
    """A room mic's take builds no ClipWatch, so its follower is the one it always was."""
    from conftest import fake_audio, patched_module, short_socket_path

    from plume import roles as roles_mod
    made = []

    class Spy(realtime.ClipWatch):
        def __init__(self, *a, **kw):
            made.append(self)
            super().__init__(*a, **kw)
    monkeypatch.setattr(realtime, "ClipWatch", Spy)
    for direct, watched in ((True, True), (False, False)):
        del made[:]
        with patched_module("sounddevice", fake_audio(np.ones(SR, dtype=np.float32),
                                                      devices=RIG)):
            realtime.lead(prompt="lofi", seed=0, tap_start=True, direct=direct,
                          role=roles_mod.RoleParams(role="lead"), status=False,
                          socket_path=short_socket_path("w-spy"), max_minutes=0.01,
                          **({"input_device": "audio interface", "input_channel": 1}
                             if direct else {}))
        assert bool(made) is watched, direct
