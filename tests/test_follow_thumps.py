"""A bump or a crackle on the input is not her playing louder.

A knock on the instrument's body or a bad cable puts a short burst at full scale on the
input.  Averaged into the bar it reads as a bar many dB louder, and the band would
swell for no musical reason.  The frames a full-scale burst touches are left out of the
level the band follows, and everything she plays, however low or loud, still counts.

Synthetic signals only, and the listening is the live path's own: FeatureExtractor
frames, ``ClipWatch``, and the half-bar readings the planner feeds BandFollow.  The
last test drives the real transport on a fake device.
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


def follow(x, *, bars):
    """What the planner feeds BandFollow from ``x``, a bar at a time: the half-bar
    readings, the bar's own level and the band's lean after it."""
    extractor, clips = FeatureExtractor(sr=SR), realtime.ClipWatch()
    events, clipped = [], []
    for i in range(0, x.size, BLOCK):
        block = x[i:i + BLOCK]
        clipped.extend(clips.process(block))
        events.extend((f.t_s, f.rms_db, f.onset) for f in extractor.process(block))
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
        bar_db, busy = realtime.follow_bar(kw, evs, clipped, silence_rms_db=SILENCE_RMS_DB,
                                           ref_db=ref_db)
        state = band.bar(bar_db, busy, playing=not kw["silent"],
                         parts=parts, bar_s=BAR_S)
        out.append({"parts": parts, "bar_db": bar_db, "gain_db": state.gain_db,
                    "moved": state.moved})
    return out


BARS = FOLLOW_LEARN_BARS + 12
AT_BAR = FOLLOW_LEARN_BARS + 3
#: Mid-bar, between strikes, as a knock lands when a hand comes off the strings.
AT_S = AT_BAR * BAR_S + 0.8


def _assert_reads_the_same(clean, hit):
    """Every bar reads within half a dB of the same take without the burst, every half
    bar (a burst is twice as much of one) within 1.25 dB, and the band does nothing it
    would not have done."""
    for was, now in zip(clean, hit):
        assert len(was["parts"]) == len(now["parts"])
        assert abs(was["bar_db"] - now["bar_db"]) <= 0.5, (was, now)
        for a, b in zip(was["parts"], now["parts"]):
            assert abs(a - b) <= 1.25, (was, now)
    assert all(not r["moved"] and r["gain_db"] == 0.0 for r in hit)


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
@pytest.mark.parametrize("seconds", [0.15, 0.3, 0.6])
def test_a_thump_on_the_body_is_not_her_playing_louder(notes, seconds):
    """The longest lands over one of her strikes, which is gone with it."""
    x = playing(BARS, notes)
    clean = follow(x, bars=BARS)
    hit = follow(thump(x, AT_S, seconds=seconds), bars=BARS)
    _assert_reads_the_same(clean, hit)


def test_the_thump_alone_would_have_read_many_db_louder():
    """The burst these tests reject is a real one: averaged in, it is the loudest thing
    in its half bar by far."""
    x = thump(playing(BARS), AT_S)
    a, b = int((AT_S - 0.8) * SR), int((AT_S - 0.8 + 0.5 * BAR_S) * SR)
    with_it = 10 * np.log10(np.mean(x[a:b].astype(np.float64) ** 2))
    without = 10 * np.log10(np.mean(playing(BARS)[a:b].astype(np.float64) ** 2))
    assert with_it - without > 10.0


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
@pytest.mark.parametrize("at", [(0.8, 2.0, 3.3), (0.8, 1.9, 3.1, 4.4, 5.5)])
def test_knocks_over_a_couple_of_bars_do_not_lift_or_fill_the_band(notes, at):
    """One loud bar moves nothing anyway.  A few knocks running would be a louder,
    busier passage, were they her."""
    x = playing(BARS, notes)
    for off in at:
        x = thump(x, AT_BAR * BAR_S + off, seconds=0.2)
    rows = follow(x, bars=BARS)
    assert all(not r["moved"] and r["gain_db"] == 0.0 for r in rows)


def test_a_bar_of_crackle_reads_as_the_bar_she_played():
    x = playing(BARS)
    _assert_reads_the_same(follow(x, bars=BARS),
                           follow(crackle(x, AT_BAR * BAR_S + 0.2), bars=BARS))


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
def test_crackle_over_a_few_bars_is_not_her_playing_louder(notes):
    x = crackle(playing(BARS, notes), AT_BAR * BAR_S + 0.2, clicks=12, over_s=3 * BAR_S)
    rows = follow(x, bars=BARS)
    assert all(not r["moved"] and r["gain_db"] == 0.0 for r in rows)


@pytest.mark.parametrize("notes", [CHORD, LOW], ids=["guitar", "low instrument"])
def test_a_real_lean_in_is_still_followed(notes):
    """Six dB louder and held: the band lifts with her, on low notes as on any others."""
    x = louder(playing(BARS, notes), AT_BAR, 6.0)
    rows = follow(x, bars=BARS)
    assert any("louder" in r["moved"] for r in rows[AT_BAR:AT_BAR + 4])
    assert rows[-1]["gain_db"] >= 2.0
    before, after = rows[AT_BAR - 1]["bar_db"], rows[AT_BAR + 1]["bar_db"]
    assert after - before == pytest.approx(6.0, abs=0.3)


def test_a_lean_in_is_followed_through_a_thump():
    x = thump(louder(playing(BARS), AT_BAR, 6.0), AT_S + BAR_S)
    clean = follow(louder(playing(BARS), AT_BAR, 6.0), bars=BARS)
    rows = follow(x, bars=BARS)
    assert [r["moved"] for r in rows] == [r["moved"] for r in clean]
    assert rows[-1]["gain_db"] == pytest.approx(clean[-1]["gain_db"], abs=0.25)


def test_playing_hard_enough_to_clip_now_and_then_is_still_her_playing_louder():
    """Her loudest strikes touching full scale are still her: only the clipped moments
    are left out, and what rings on is as loud as she played it."""
    x = np.clip(louder(playing(BARS, level=0.12), AT_BAR, 10.0), -1.0, 1.0)
    assert np.abs(x).max() >= 1.0
    rows = follow(x, bars=BARS)
    assert any("louder" in r["moved"] for r in rows[AT_BAR:AT_BAR + 4])
    assert rows[-1]["gain_db"] >= 2.0


def test_a_half_bar_that_is_mostly_clipped_says_nothing():
    """Nothing to go on is no evidence, not a quiet bar."""
    x = playing(BARS)
    a = int(AT_BAR * BAR_S * SR)
    x[a:a + int(0.5 * BAR_S * SR)] = 1.0
    rows = follow(x, bars=BARS)
    assert None in [p for r in rows[AT_BAR - 1:AT_BAR + 1] for p in r["parts"]]
    assert all(not r["moved"] for r in rows)


def test_a_thump_while_she_sits_out_is_not_her_playing():
    x = playing(BARS)
    a, b = int(AT_BAR * BAR_S * SR), int((AT_BAR + 2) * BAR_S * SR)
    x[a - int(0.3 * SR):b] *= 0.001
    rows = follow(thump(x, AT_S), bars=BARS)
    assert all(not r["moved"] for r in rows[:AT_BAR + 2])
    assert rows[AT_BAR]["bar_db"] is None


def test_clip_watch_names_the_hops_at_full_scale_however_the_stream_is_cut():
    x = np.zeros(8 * realtime.HOP, dtype=np.float32)
    x[3 * realtime.HOP + 7] = -1.0
    x[5 * realtime.HOP - 1] = 0.995
    x[6 * realtime.HOP] = 0.9                      # loud, not clipped
    want = [4 * realtime.HOP / SR, 5 * realtime.HOP / SR]
    for block in (64, 256, 700, x.size):
        watch = realtime.ClipWatch()
        got = [t for i in range(0, x.size, block) for t in watch.process(x[i:i + block])]
        assert got == pytest.approx(want), block


def test_with_nothing_clipped_the_readings_are_the_ones_they_always_were():
    x = playing(6)
    extractor = FeatureExtractor(sr=SR)
    events = [(f.t_s, f.rms_db, f.onset) for f in extractor.process(x)]
    evs = [e for e in events if BAR_S <= e[0] < 2 * BAR_S]
    kw = realtime.bar_features_dict(1, evs, silence_rms_db=SILENCE_RMS_DB, ref_db=None)
    assert realtime.follow_bar(kw, evs, [], silence_rms_db=SILENCE_RMS_DB,
                               ref_db=None) == (kw["rms_db"], kw["onset_density"])
    a, _ = realtime.follow_parts(evs, [], BAR_S, 2 * BAR_S, 3 * BAR_S, fed_until=None,
                                 heard_until=2 * BAR_S, silence_rms_db=SILENCE_RMS_DB,
                                 ref_db=None)
    b, _ = realtime.follow_parts(evs, [], BAR_S, 2 * BAR_S, 3 * BAR_S, fed_until=None,
                                 heard_until=2 * BAR_S, silence_rms_db=SILENCE_RMS_DB,
                                 ref_db=None, clipped=[])
    assert a == b and None not in a


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
