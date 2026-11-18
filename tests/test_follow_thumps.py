"""A bump or a crackle on the input is not her playing louder.

A knock on the instrument's body or a bad cable puts a short burst at full scale on the
input.  Averaged into the bar it reads as a bar many dB louder, and the band would
swell for no musical reason.  For a player whose own playing sits far below full scale,
a bar with such a burst in it gives the band no evidence at all: what she played under
it is not known.  A player whose strikes reach full scale by leaning in is read exactly
as she always was: her clips are her playing.

So every reading the band gets is the one it always got, or nothing.  That is the
property tested here, over many players, and it is what makes a move the wrong way
impossible.

Synthetic signals only, and the listening is the live path's own: FeatureExtractor
frames, ``ClipWatch``, and the readings the planner feeds BandFollow.  The last test
drives the real transport on a fake device.
"""
import numpy as np
import pytest

from plume import realtime
from plume.bandentry import FOLLOW_LEARN_BARS, BandFollow
from plume.listen.features import FeatureExtractor

SR, BPM, BLOCK = 48000, 100.0, 256
BAR_S = 240.0 / BPM
SILENCE_RMS_DB = -45.0
#: Open chord shapes, and the same two octaves down, where a bass or a baritone plays.
CHORD = (130.81, 164.81, 196.0, 261.63, 329.63, 392.0)
LOW = (32.70, 41.20, 49.0, 65.41, 73.42, 82.41)


def playing(bars, notes=CHORD, *, level=0.05, per_bar=4, sr=SR):
    """``bars`` of a chord struck ``per_bar`` times a bar and left to ring, over a
    little hiss."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(bars * BAR_S * sr)).astype(np.float32) * 1e-4
    t = np.arange(int(1.2 * sr)) / sr
    strike = sum(np.sin(2 * np.pi * f * t) / (1 + 0.3 * j) for j, f in enumerate(notes))
    strike = (strike * np.exp(-t * 3.0) * np.minimum(1.0, t / 0.004)).astype(np.float32)
    for k in range(int(bars * per_bar)):
        i = int(k * BAR_S / per_bar * sr)
        seg = strike[: x.size - i]
        x[i:i + seg.size] += level * seg
    return x


def louder(x, from_bar, db, *, to_bar=None):
    y = x.copy()
    a = int(from_bar * BAR_S * SR)
    b = y.size if to_bar is None else int(to_bar * BAR_S * SR)
    y[a:b] *= 10.0 ** (db / 20.0)
    return y


def thump(x, at_s, *, seconds=0.3, hz=40.0, release_s=0.015):
    """A knock on the body: a low boom far past full scale, so the input clips it flat
    from rail to rail for ``seconds``, then damped."""
    y = x.copy()
    t = np.arange(int((seconds + 10 * release_s) * SR)) / SR
    env = np.where(t < seconds, 1.0, np.exp(-(t - seconds) / release_s))
    boom = 6.0 * np.sin(2 * np.pi * hz * t) * env * np.minimum(1.0, t / 0.01)
    i = int(at_s * SR)
    y[i:i + t.size] += boom.astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def crackle(x, at_s, *, clicks=4, over_s=BAR_S):
    """A bad cable: scattered clicks, each a millisecond or a few at full scale."""
    y = x.copy()
    rng = np.random.default_rng(5)
    for off in np.sort(rng.uniform(0.0, over_s, clicks)):
        i = int((at_s + off) * SR)
        n = int(rng.integers(48, 193))
        y[i:i + n] = rng.choice((-1.0, 1.0))
    return y


def listen(x):
    """Her frames, and the clips marked as out of character for her, as live."""
    extractor, clips = FeatureExtractor(sr=SR), realtime.ClipWatch()
    events, clipped = [], []
    for i in range(0, x.size, BLOCK):
        block = x[i:i + BLOCK]
        clipped.extend(clips.process(block))
        events.extend((f.t_s, f.rms_db, f.onset) for f in extractor.process(block))
    return events, clipped


def follow(x, *, bars, heard=None, watch=True):
    """What the planner feeds BandFollow from ``x``, a bar at a time: the half-bar
    readings, the bar's own level and note count, and the band's lean after it.
    ``watch`` False: with no clip ever marked, which is every reading as it was before
    clips were looked at.  ``heard``: ``listen(x)``, to listen only once."""
    events, clipped = heard or listen(x)
    if not watch:
        clipped = []
    band = BandFollow(trusted=True)
    ref_db, fed, out = None, None, []
    for n in range(bars - 1):
        w0, w1, nxt = n * BAR_S, (n + 1) * BAR_S, (n + 2) * BAR_S
        evs = [e for e in events if w0 <= e[0] < w1]
        later = [e for e in events if w1 <= e[0] < nxt]
        kw = realtime.bar_features_dict(n, evs, silence_rms_db=SILENCE_RMS_DB, ref_db=ref_db)
        if not kw["silent"]:
            ref_db = kw["rms_db"] if ref_db is None else 0.5 * kw["rms_db"] + 0.5 * ref_db
        parts, fed = realtime.follow_parts(
            evs, later, w0, w1, nxt, fed_until=fed, heard_until=w1 + 0.5 * BAR_S + 0.1,
            silence_rms_db=SILENCE_RMS_DB, ref_db=ref_db, clipped=clipped)
        bar_db, busy = realtime.follow_bar(kw, clipped, w0, w1)
        state = band.bar(bar_db, busy, playing=not kw["silent"], parts=parts, bar_s=BAR_S)
        out.append({"parts": parts, "bar_db": bar_db, "busy": busy,
                    "gain_db": state.gain_db, "density": state.density,
                    "moved": state.moved})
    return out


def both(x, *, bars):
    """``follow`` with clips watched, and as it always was."""
    heard = listen(x)
    return (follow(x, bars=bars, heard=heard),
            follow(x, bars=bars, heard=heard, watch=False))


def assert_as_ever_or_nothing(now, ever):
    """THE property: every level and note count the band is given is the one it was
    always given for that stretch, or nothing.  Returns how many were nothing."""
    nothing = 0
    for n, (a, b) in enumerate(zip(now, ever, strict=True)):
        assert len(a["parts"]) == len(b["parts"]), n
        for got, was in zip([*a["parts"], a["bar_db"], a["busy"]],
                            [*b["parts"], b["bar_db"], b["busy"]]):
            assert got is None or got == was, (n, a, b)
            nothing += got is None and was is not None
    return nothing


def still(rows):
    """The band never moved."""
    return all(not r["moved"] and r["gain_db"] == 0.0 and r["density"] == 1.0
               for r in rows)


BARS = FOLLOW_LEARN_BARS + 12
AT_BAR = FOLLOW_LEARN_BARS + 3
#: Mid-bar, between strikes, as a knock lands when a hand comes off the strings.
AT_S = AT_BAR * BAR_S + 0.8


# ------------------------------------------------------------ a quiet player, and bursts

@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
@pytest.mark.parametrize("seconds", [0.15, 0.3, 0.6])
def test_a_bar_with_a_thump_in_it_says_nothing_and_every_other_bar_reads_as_ever(
        notes, seconds):
    x = playing(BARS, notes)
    clean = follow(x, bars=BARS)
    hit, ever = both(thump(x, AT_S, seconds=seconds), bars=BARS)
    assert_as_ever_or_nothing(hit, ever)
    assert [n for n, r in enumerate(hit) if r["bar_db"] is None] == [AT_BAR]
    assert hit[AT_BAR]["busy"] is None
    for n, (a, b) in enumerate(zip(hit, clean)):
        if n != AT_BAR:
            assert (a["bar_db"], a["busy"]) == (b["bar_db"], b["busy"]), n
        # Both halves of the bar, wherever in it the thump fell: the early half is
        # read a bar sooner, with the bar before.
        gone = [p is None for p in a["parts"]]
        assert gone == {AT_BAR - 1: [False, True], AT_BAR: [True, False]}.get(
            n, [False] * len(gone))
    assert still(hit)


def test_the_thump_alone_would_have_read_many_db_louder():
    """The burst these tests reject is a real one: averaged in, it is the loudest thing
    in its bar by far."""
    _, ever = both(thump(playing(BARS), AT_S), bars=BARS)
    clean = follow(playing(BARS), bars=BARS)
    assert ever[AT_BAR]["bar_db"] - clean[AT_BAR]["bar_db"] > 15.0


def test_a_thump_at_the_very_end_of_a_bar_takes_the_next_bar_with_it():
    """Its ring is in the first frames of the bar after."""
    x = thump(playing(BARS), (AT_BAR + 1) * BAR_S - 0.11, seconds=0.1)
    hit, ever = both(x, bars=BARS)
    assert_as_ever_or_nothing(hit, ever)
    assert [n for n, r in enumerate(hit) if r["bar_db"] is None] == [AT_BAR, AT_BAR + 1]
    assert still(hit)


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
@pytest.mark.parametrize("at", [(0.8, 2.0, 3.3), (0.8, 1.9, 3.1, 4.4, 5.5)])
def test_knocks_over_a_couple_of_bars_do_not_lift_or_fill_the_band(notes, at):
    """One loud bar moves nothing anyway.  A few knocks running would be a louder,
    busier passage, were they her."""
    x = playing(BARS, notes)
    for off in at:
        x = thump(x, AT_BAR * BAR_S + off, seconds=0.2)
    hit, ever = both(x, bars=BARS)
    assert_as_ever_or_nothing(hit, ever)
    assert still(hit) and not still(ever)


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
def test_crackle_over_a_few_bars_is_not_her_playing_louder(notes):
    x = crackle(playing(BARS, notes), AT_BAR * BAR_S + 0.2, clicks=12, over_s=3 * BAR_S)
    hit, ever = both(x, bars=BARS)
    assert_as_ever_or_nothing(hit, ever)
    assert still(hit) and not still(ever)


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
def test_a_real_lean_in_is_still_followed(notes):
    """Six dB louder and held: the band lifts with her, on low notes as on any others."""
    x = louder(playing(BARS, notes), AT_BAR, 6.0)
    rows, ever = both(x, bars=BARS)
    assert rows == ever
    assert any("louder" in r["moved"] for r in rows[AT_BAR:AT_BAR + 4])
    assert rows[-1]["gain_db"] >= 2.0
    before, after = rows[AT_BAR - 1]["bar_db"], rows[AT_BAR + 1]["bar_db"]
    assert after - before == pytest.approx(6.0, abs=0.3)


def test_a_lean_in_is_followed_through_a_thump():
    lean = louder(playing(BARS), AT_BAR, 6.0)
    clean = follow(lean, bars=BARS)
    rows, ever = both(thump(lean, AT_S + BAR_S), bars=BARS)
    assert_as_ever_or_nothing(rows, ever)
    assert any("louder" in r["moved"] for r in rows[AT_BAR:AT_BAR + 5])
    assert not any(set(r["moved"]) & {"softer", "sparser"} for r in rows)
    assert rows[-1]["gain_db"] == pytest.approx(clean[-1]["gain_db"], abs=0.25)


def test_a_thump_while_she_sits_out_is_not_her_playing():
    x = playing(BARS)
    a, b = int(AT_BAR * BAR_S * SR), int((AT_BAR + 2) * BAR_S * SR)
    x[a - int(0.3 * SR):b] *= 0.001
    rows, ever = both(thump(x, AT_S), bars=BARS)
    assert_as_ever_or_nothing(rows, ever)
    assert all(not r["moved"] for r in rows[:AT_BAR + 2])
    assert rows[AT_BAR]["bar_db"] is None


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
    t0 = 12.0
    want = [t0 + 4 * realtime.HOP / SR, t0 + 5 * realtime.HOP / SR]
    for block in (64, 256, 700, x.size):
        watch = _heard(realtime.ClipWatch(), 0.15)
        got = [t for i in range(0, x.size, block) for t in watch.process(x[i:i + block])]
        assert got == pytest.approx(want), block


@pytest.mark.parametrize("peak, marked", [(0.1, True), (0.25, True), (0.28, True),
                                          (0.32, False), (0.35, False), (0.6, False),
                                          (1.3, False)])
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
    assert watch.process(BURST)


def test_a_few_bursts_while_she_plays_do_not_make_full_scale_her_own():
    """Bursts in a third of the last half minute, and a clip is still not hers."""
    watch = _heard(realtime.ClipWatch(), 0.15, seconds=30.0)
    x = tone(2.0, 0.15)
    for i in range(15):
        watch.process(thump(x, 1.0, seconds=0.1) if i % 3 == 0 else x)
    assert watch.typical_peak == pytest.approx(0.15, rel=1e-3)
    for _ in range(4):                             # and then a run of them, eight seconds
        watch.process(thump(x, 1.0, seconds=0.1))
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


def test_follow_bar_hands_on_the_bar_as_it_was_read_or_nothing():
    kw = {"rms_db": -20.0, "onset_density": 1.5}
    assert realtime.follow_bar(kw, [], 0.0, 2.0) == (-20.0, 1.5)
    assert realtime.follow_bar(kw, [5.0], 0.0, 2.0) == (-20.0, 1.5)
    assert realtime.follow_bar(kw, [1.0], 0.0, 2.0) == (None, None)
    assert realtime.follow_bar(None, [], 0.0, 2.0) == (None, None)


# ------------------------------------------------------------ a player who plays hot

def strikes(bars, *, peak, per_bar, decay, notes=CHORD, noise=False, accents=None,
            accent_peak=None, seed=7):
    """``bars`` of one sound struck ``per_bar`` times a bar, each strike peaking at
    ``peak`` and dying away at ``decay`` per second.  ``noise``: a drum or a slap, not
    a pitch.  ``accents``: (every how many strikes, how many dB up).  ``accent_peak``:
    instead of the dB, the accent's peak in a given bar, as a function of the bar."""
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
        amp = peak
        if accents and k % accents[0] == 0:
            amp = (accent_peak(k // per_bar) if accent_peak
                   else peak * 10.0 ** (accents[1] / 20.0))
        x[i:i + seg.size] += amp * seg
    return x


HOT_BARS, LEAN, SOFT = 28, (10, 18), (20, 28)

#: Sounds a player makes whose strikes reach full scale when she leans in, or always.
HOT = {
    "ringing chord": dict(peak=0.5, per_bar=4, decay=3.0),
    "strum": dict(peak=0.6, per_bar=4, decay=8.0),
    "staccato": dict(peak=0.6, per_bar=4, decay=20.0),
    "percussion in eighths": dict(peak=0.6, per_bar=8, decay=30.0, noise=True),
    "percussion in sixteenths": dict(peak=0.6, per_bar=16, decay=40.0, noise=True),
    "accents ten dB up": dict(peak=0.35, per_bar=8, decay=30.0, noise=True,
                              accents=(4, 10.0)),
    "low register": dict(peak=0.55, per_bar=4, decay=5.0, notes=LOW),
    "always clipping, ringing": dict(peak=1.4, per_bar=4, decay=3.0),
    "always clipping, staccato": dict(peak=1.4, per_bar=4, decay=20.0),
    "always clipping, percussion": dict(peak=1.4, per_bar=8, decay=30.0, noise=True),
    "one strike a bar": dict(peak=0.6, per_bar=1, decay=3.0),
    "only just hot": dict(peak=0.32, per_bar=4, decay=3.0),
}


def hot_take(up_db=6.0, **sound):
    """Her normal, louder from bar 10 to 18, eight dB softer from bar 20, through an
    input that clips at full scale."""
    x = strikes(HOT_BARS, **sound)
    x = louder(x, LEAN[0], up_db, to_bar=LEAN[1])
    x = louder(x, SOFT[0], -8.0, to_bar=SOFT[1])
    return np.clip(x, -1.0, 1.0)


@pytest.mark.parametrize("name", list(HOT))
def test_a_player_whose_own_strikes_clip_is_read_as_she_always_was(name):
    """Level, note count and every move of the band: the same as with no clip marked."""
    x = hot_take(up_db=11.0 if name == "only just hot" else 6.0, **HOT[name])
    lean = x[int(LEAN[0] * BAR_S * SR):int(LEAN[1] * BAR_S * SR)]
    assert np.abs(lean).max() >= 1.0, "the lean never reached full scale"
    now, ever = both(x, bars=HOT_BARS)
    assert now == ever


def test_a_hot_players_lean_in_lifts_the_band_and_her_soft_passage_brings_it_down():
    rows = follow(hot_take(**HOT["staccato"]), bars=HOT_BARS)
    moves = [m for r in rows for m in r["moved"]]
    assert "louder" in [m for r in rows[LEAN[0]:LEAN[1]] for m in r["moved"]]
    assert "softer" in [m for r in rows[SOFT[0]:] for m in r["moved"]]
    assert moves.index("louder") < moves.index("softer")


# ------------------------------------------------------------ in between

def held(rows, ever):
    """Under her lean the band was never under where it stood, and every reading it got
    was one it always got, or nothing."""
    assert_as_ever_or_nothing(rows, ever)
    lean = rows[LEAN[0]:LEAN[1]]
    assert min(r["gain_db"] for r in lean) >= 0.0
    assert min(r["density"] for r in lean) >= 1.0
    return lean


@pytest.mark.parametrize("between_db", [-3.0, -6.0, -8.0, -12.0, -18.0])
@pytest.mark.parametrize("per_bar, decay, every, to", [(8, 40.0, 2, 1.3), (8, 60.0, 2, 1.3),
                                                       (16, 60.0, 4, 2.0)])
def test_a_quiet_players_accents_leaping_to_full_scale_never_bring_the_band_down(
        between_db, per_bar, decay, every, to):
    """Her accents clip for eight bars and the notes between stay as soft as they were.
    Reading the bar from the notes between would read her softer at her loudest."""
    x = strikes(HOT_BARS, peak=0.25 * 10.0 ** (between_db / 20.0), per_bar=per_bar,
                decay=decay, noise=True, accents=(every, None),
                accent_peak=lambda bar: to if LEAN[0] <= bar < LEAN[1] else 0.25)
    lean = held(*both(np.clip(x, -1.0, 1.0), bars=HOT_BARS))
    assert not [r["moved"] for r in lean if set(r["moved"]) & {"softer", "sparser"}]


@pytest.mark.parametrize("sound", ["ringing chord", "staccato", "percussion in eighths"])
@pytest.mark.parametrize("peak, up_db", [(0.12, 20.0), (0.25, 13.0)])
def test_a_quiet_player_leaping_to_full_scale_never_brings_the_band_down(sound, peak, up_db):
    x = strikes(HOT_BARS, **{**HOT[sound], "peak": peak})
    held(*both(np.clip(louder(x, LEAN[0], up_db, to_bar=LEAN[1]), -1.0, 1.0), bars=HOT_BARS))


def test_bursts_on_a_sparse_players_strikes_do_not_move_the_band():
    """Two strikes a bar, and a knock lands on five of them running: each takes her
    strike with it."""
    x = strikes(BARS, peak=0.15, per_bar=2, decay=3.0)
    for k in range(5):
        x = thump(x, (AT_BAR + k) * BAR_S - 0.02, seconds=0.2)
    rows, ever = both(x, bars=BARS)
    assert_as_ever_or_nothing(rows, ever)
    assert still(rows) and not still(ever)


@pytest.mark.parametrize("seed", range(24))
def test_whoever_plays_every_reading_is_the_one_it_always_was_or_nothing(seed):
    """Players drawn at random: quiet, hot and hovering between, ringing and dry, dense
    and sparse, with accents that leap, a lean that clips, knocks and crackle."""
    rng = np.random.default_rng(100 + seed)
    bars = 20
    peak = float(rng.choice([0.08, 0.15, 0.25, 0.3, 0.35, 0.6]))
    drift = rng.uniform(0.25, 0.35, bars) if seed % 4 == 0 else None
    sound = dict(peak=peak, per_bar=int(rng.choice([1, 2, 4, 8, 16])),
                 decay=float(rng.choice([3.0, 8.0, 20.0, 40.0, 60.0])),
                 noise=bool(rng.integers(2)), notes=CHORD if rng.integers(2) else LOW,
                 seed=seed)
    if rng.integers(2):
        to = float(rng.choice([0.5, 1.3, 2.0]))
        base = peak * 10.0 ** (float(rng.choice([3.0, 6.0, 12.0, 18.0])) / 20.0)
        sound.update(accents=(int(rng.choice([2, 4])), None),
                     accent_peak=lambda bar: to if 10 <= bar < 16 else base)
    x = strikes(bars, **sound)
    if drift is not None:
        for n in range(bars):
            x[int(n * BAR_S * SR):int((n + 1) * BAR_S * SR)] *= drift[n] / peak
    if rng.integers(2):
        x = louder(x, 10, float(rng.choice([6.0, 12.0, 20.0])), to_bar=16)
    for _ in range(int(rng.integers(0, 5))):
        x = thump(x, float(rng.uniform(1.0, bars * BAR_S - 1.0)),
                  seconds=float(rng.choice([0.05, 0.15, 0.3, 0.6])))
    if rng.integers(2):
        x = crackle(x, float(rng.uniform(1.0, bars * BAR_S - 8.0)),
                    clicks=int(rng.integers(2, 13)), over_s=float(rng.uniform(0.5, 7.0)))
    now, ever = both(np.clip(x, -1.0, 1.0), bars=bars)
    assert_as_ever_or_nothing(now, ever)


def test_the_random_players_do_get_bars_that_say_nothing():
    """The property above is not passing for want of a marked clip."""
    x = playing(BARS)
    for k in range(4):
        x = thump(x, (AT_BAR + 2 * k) * BAR_S + 0.5)
    assert assert_as_ever_or_nothing(*both(x, bars=BARS)) >= 4 * 4


# ------------------------------------------------------------ the real transport, no device

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


def test_on_the_real_transport_thumps_and_crackle_do_not_lift_the_band(tmp_path, consent):
    """The planner's own path, on the fake device: a steady player whose input is
    knocked and crackles once the band is in.  The band never leans louder."""
    import json

    from conftest import fake_audio, patched_module, playhead_rate, short_socket_path
    from test_bandentry import _take

    from plume import roles as roles_mod

    bpm, seconds = 115.0, 100.0
    rate = playhead_rate(12.0)
    human = _take(bpm=bpm, seconds=seconds)
    hits = (50.3, 52.9, 61.2, 70.4, 71.1)
    for at in hits:
        human = thump(human, at, seconds=0.25)
    human = crackle(human, 56.0, clicks=8, over_s=4.0)
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=rate)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("thump"),
                      max_minutes=(seconds / rate + 8.0) / 60.0,
                      input_device="audio interface", input_channel=1)
    rows = [json.loads(x) for x in (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int) and r.get("follow")]
    heard = [r for r in rows if r["t_s"] > hits[-1] + 4 * 240.0 / bpm]
    assert heard, "the take did not run past the last knock"
    assert all("louder" not in (r["follow"].get("moved") or ()) for r in rows)
    assert max(r["follow"]["gain_db"] for r in rows) == 0.0
