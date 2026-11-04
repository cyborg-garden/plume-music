"""The smart looper's pure half: trim, tempo, fit, join, layers, the words.

Every input is synthesized here.  Loop audio lives in memory and is never
written anywhere, and no test opens a device (the live half is
tests/test_looper_live.py, on the fake device).
"""
from __future__ import annotations

import numpy as np
import pytest

from plume import looper as lp

SR = 48000


def pluck(freq=220.0, dur_s=0.35, gain=0.5):
    t = np.arange(int(dur_s * SR)) / SR
    return (gain * np.exp(-t * 7.0) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def take(beats, bpm, *, lead_s=0.5, tail_s=3.0, noise_db=-75.0, freq=220.0,
         gain=0.5, seed=1):
    """Plucks at ``beats`` over a little room noise; returns (audio, lead samples)."""
    beat_s = 60.0 / bpm
    n = int((lead_s + max(beats) * beat_s + tail_s) * SR)
    x = np.zeros(n, dtype=np.float32)
    for b in beats:
        at = int(round((lead_s + b * beat_s) * SR))
        p = pluck(freq, min(0.35, beat_s * 0.9), gain)
        x[at:at + p.size] += p[:max(0, n - at)]
    amp = 10.0 ** (noise_db / 20.0)
    x += (amp * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)
    return x, int(lead_s * SR)


# ---------------------------------------------------------------- auto-trim

class TestFindPhrase:
    def test_finds_where_the_playing_starts_and_ends(self):
        x, lead = take([0, 1, 2, 3], 120.0)
        ph = lp.find_phrase(x, SR)
        assert ph is not None
        assert abs(ph.first - lead) < 0.005 * SR
        assert len(ph.onsets) == 4
        assert abs(ph.last_onset - (lead + 3 * 0.5 * SR)) < 0.015 * SR
        assert ph.end > ph.last_onset
        assert ph.end < x.size - SR       # the room is not part of the phrase

    def test_silence_is_no_phrase(self):
        assert lp.find_phrase(np.zeros(SR, dtype=np.float32), SR) is None

    def test_room_noise_alone_is_no_phrase(self):
        rng = np.random.default_rng(3)
        x = (1e-3 * rng.standard_normal(2 * SR)).astype(np.float32)
        assert lp.find_phrase(x, SR) is None

    def test_a_quieter_player_gets_the_same_phrase(self):
        """It listens to the playing, not to the fader (DREAM.md, v2)."""
        x, _ = take([0, 1, 2, 3], 120.0)
        loud = lp.find_phrase(x, SR)
        quiet = lp.find_phrase(x * 10.0 ** (-24.0 / 20.0), SR)
        assert quiet is not None
        assert abs(quiet.first - loud.first) <= lp.LEVEL_HOP
        assert len(quiet.onsets) == len(loud.onsets)

    def test_a_note_already_ringing_only_counts_when_forced(self):
        t = np.arange(2 * SR) / SR
        x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        assert lp.find_phrase(x, SR) is None
        ph = lp.find_phrase(x, SR, contrast_db=0.0)
        assert ph is not None and ph.first < lp.LEVEL_HOP


class TestBarsHeld:
    @pytest.mark.parametrize("pos,want", [
        (0.0, 1),      # one note is a one-bar loop
        (0.75, 1),     # beat 4 of the first bar
        (1.5, 2),
        (3.75, 4),
        (4.0, 4),      # landing on the next "one" closes the phrase
        (4.04, 4),     # a little late on it
        (3.96, 4),     # a little early on it
        (4.2, 5),      # far enough past the line to be a real note in bar five
    ])
    def test_counts_bars_from_the_last_note(self, pos, want):
        assert lp.bars_held(pos) == want


# ------------------------------------------------------------ tempo from take

class TestInferTempo:
    def run(self, beats, bpm, **kw):
        x, _ = take(beats, bpm)
        ph = lp.find_phrase(x, SR)
        return lp.infer_tempo(ph, SR, **kw)

    @pytest.mark.parametrize("bpm", [72.0, 96.0, 120.0, 140.0])
    def test_quarters_with_a_closing_downbeat(self, bpm):
        got = self.run(list(range(17)), bpm)
        assert got.bars == 4
        assert got.bpm == pytest.approx(bpm, rel=0.015)

    @pytest.mark.parametrize("bpm", [80.0, 110.0])
    def test_eighths_that_stop_before_the_line(self, bpm):
        got = self.run([i * 0.5 for i in range(16)], bpm)
        assert got.bars == 2
        assert got.bpm == pytest.approx(bpm, rel=0.015)

    def test_saying_the_bars_settles_a_sparse_take(self):
        """A sparse take is the one that has fooled the tracker before."""
        got = self.run([0, 3, 8], 75.0, bars=2)
        assert got.bars == 2
        assert got.bpm == pytest.approx(75.0, rel=0.02)

    def test_a_tempo_outside_the_range_is_refused(self):
        with pytest.raises(lp.LoopRefused):
            self.run([0, 1, 2, 3, 4], 300.0, bars=1)

    def test_too_few_notes_without_a_bar_count_is_refused(self):
        with pytest.raises(lp.LoopRefused, match="bar"):
            self.run([0, 1], 90.0)


class TestSparseTakes:
    """A take with no clear beat is refused: the claim, counted."""

    @staticmethod
    def shows_tempo(rel_s):
        at = (np.asarray(rel_s) * SR).astype(int)
        ph = lp.Phrase(first=int(at[0]), last_onset=int(at[-1]), end=int(at[-1]) + 1,
                       onsets=tuple(int(a) for a in at))
        try:
            lp.infer_tempo(ph, SR)
            return True
        except lp.LoopRefused:
            return False

    @pytest.mark.parametrize("gap_s", [(0.2, 1.5), (0.1, 0.6), (0.5, 2.0)])
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 10])
    def test_random_onsets_rarely_show_a_tempo(self, n, gap_s):
        rng = np.random.default_rng(n)
        hits = sum(self.shows_tempo(np.cumsum(np.r_[0.0, rng.uniform(*gap_s, n - 1)]))
                   for _ in range(1000))
        assert hits / 1000 <= 0.06

    def test_three_notes_are_never_enough(self):
        for rel in ([0, 1, 2], [0, 0.5, 1.0], [0, 0.83, 2.11]):
            assert not self.shows_tempo(rel)

    @pytest.mark.parametrize("beats,want", [
        (list(range(5)), 0.80), (list(range(9)), 0.80), (list(range(17)), 0.95),
        ([i * 0.5 for i in range(16)], 0.80),
    ])
    def test_a_steady_phrase_still_shows_one(self, beats, want):
        """Fifteen milliseconds of human timing, which is a loose take."""
        rng = np.random.default_rng(11)
        hits, total = 0, 0
        for bpm in (70.0, 90.0, 110.0, 130.0):
            for _ in range(150):
                t = np.asarray(beats) * 60.0 / bpm
                hits += self.shows_tempo(t - t[0] + rng.normal(0.0, 0.015, t.size)
                                         + 0.1)
                total += 1
        assert hits / total >= want


# ------------------------------------------------------------------- fitting

class TestFit:
    def test_exact_length(self):
        x = np.ones(96000, dtype=np.float32)
        out = lp.fit_to_bars(x, 1, 120.0, SR)
        assert out.size == 96000

    @pytest.mark.parametrize("have", [94500, 97500])
    def test_a_small_miss_is_padded_or_trimmed_never_resampled(self, have):
        t = np.arange(have) / SR
        x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = lp.fit_to_bars(x, 1, 120.0, SR)
        assert out.size == 96000
        k = min(have, 96000)
        assert np.array_equal(out[:k], x[:k])    # the pitch cannot have moved

    @pytest.mark.parametrize("have", [90000, 102000])
    def test_a_large_miss_is_refused_and_says_why(self, have):
        with pytest.raises(lp.LoopRefused, match="stretch"):
            lp.fit_to_bars(np.ones(have, dtype=np.float32), 1, 120.0, SR)


# ---------------------------------------------------------------------- join

def wrap_jump(loop):
    return float(abs(loop[0] - loop[-1]))


class TestJoin:
    def test_a_tone_joined_anywhere_has_no_click(self):
        t = np.arange(SR * 2) / SR
        x = (0.5 * np.sin(2 * np.pi * 233.7 * t)).astype(np.float32)
        at, n = 4001, 48000
        naive = x[at:at + n]
        assert wrap_jump(naive) > 0.05
        out = lp.close_join(naive, x[at - lp.JOIN_SAMPLES:at])
        assert out.size == n
        step = np.abs(np.diff(x)).max()
        assert wrap_jump(out) <= 1.5 * step
        assert np.abs(np.diff(out)).max() <= 1.5 * step

    def test_the_attack_at_the_top_of_the_loop_is_untouched(self):
        p = pluck(gain=0.5)
        body = np.zeros(48000, dtype=np.float32)
        body[:p.size] = p
        pre = np.zeros(lp.JOIN_SAMPLES, dtype=np.float32)
        out = lp.close_join(body, pre)
        assert np.array_equal(out[:p.size], p)

    def test_no_material_before_the_top_still_cannot_click(self):
        x = np.full(48000, 0.4, dtype=np.float32)
        out = lp.close_join(x, np.zeros(0, dtype=np.float32))
        assert wrap_jump(out) < 0.02
        assert out[lp.JOIN_SAMPLES] == pytest.approx(0.4)

    def test_normalize_is_level_independent_and_capped(self):
        x = pluck(gain=0.05)
        a = lp.normalize(x)
        b = lp.normalize(x * 0.1)
        assert np.abs(a).max() == pytest.approx(lp.db_to_lin(lp.LOOP_PEAK_DB), rel=1e-3)
        assert np.abs(b).max() == pytest.approx(np.abs(a).max(), rel=1e-3)
        faint = x * 1e-4
        gain = np.abs(lp.normalize(faint)).max() / np.abs(faint).max()
        assert gain == pytest.approx(lp.db_to_lin(lp.LOOP_MAX_BOOST_DB), rel=1e-3)


# -------------------------------------------------------------------- layers

def layer(name, *, bars=1, level=0.5, anchor=0, n=1000):
    # each bar its own constant level, so a test can see which bar it got
    audio = np.concatenate([np.full(n, level * (i + 1) / bars, dtype=np.float32)
                            for i in range(bars)])
    return lp.LoopLayer(name=name, audio=audio, bars=bars, bpm=120.0,
                        anchor_bar=anchor)


class TestStack:
    def test_a_layer_waits_for_the_top_of_its_cycle(self):
        s = lp.LoopStack()
        s.add(layer("a", bars=4, anchor=2))
        assert not s.mix_bar(3, 1000).any()   # (3-2)%4 == 1: mid-phrase, wait
        assert s.mix_bar(6, 1000).any()       # (6-2)%4 == 0: phrase top

    def test_it_plays_the_bar_it_was_recorded_against(self):
        s = lp.LoopStack()
        s.add(layer("a", bars=2, level=0.4))
        assert s.mix_bar(0, 1000)[-1] == pytest.approx(0.2)
        assert s.mix_bar(1, 1000)[-1] == pytest.approx(0.4)
        assert s.mix_bar(2, 1000)[-1] == pytest.approx(0.2)

    def test_mute_and_back_with_no_click(self):
        s = lp.LoopStack()
        s.add(layer("a"))
        s.mix_bar(0, 1000)
        s.mute("a")
        off = s.mix_bar(1, 1000)
        assert np.abs(np.diff(off)).max() < 0.01  # faded rather than cut
        assert off[-1] == 0.0
        assert not s.mix_bar(2, 1000).any()
        s.unmute("a")
        on = s.mix_bar(3, 1000)
        assert on[0] == 0.0 and on[-1] == pytest.approx(0.5)
        assert np.abs(np.diff(on)).max() < 0.01

    def test_swap_trades_one_for_the_other_on_the_same_bar(self):
        s = lp.LoopStack()
        s.add(layer("a", level=0.5))
        s.add(layer("b", level=0.25))
        s.mute("b")
        s.mix_bar(0, 1000)
        s.swap("a", "b")
        out = s.mix_bar(1, 1000)
        assert out[-1] == pytest.approx(0.25)
        assert s.layer("a").muted and not s.layer("b").muted

    def test_drop_forgets_it_after_fading_it(self):
        s = lp.LoopStack()
        s.add(layer("a"))
        s.mix_bar(0, 1000)
        s.drop("a")
        assert s.names() == []
        tail = s.mix_bar(1, 1000)
        assert tail[0] > 0.0 and tail[-1] == 0.0
        assert not s.mix_bar(2, 1000).any()

    def test_an_unknown_name_is_refused_with_the_names_that_exist(self):
        s = lp.LoopStack()
        s.add(layer("bass"))
        with pytest.raises(lp.LoopRefused, match="bass"):
            s.mute("keys")

    def test_retaking_a_name_keeps_the_old_one_until_the_new_one_lands(self):
        s = lp.LoopStack()
        s.add(layer("a", level=0.5))
        s.mix_bar(0, 1000)
        s.add(layer("a", bars=2, level=0.2, anchor=2))
        assert s.mix_bar(3, 1000)[-1] == pytest.approx(0.5)  # still the old take
        assert s.mix_bar(4, 1000)[-1] == pytest.approx(0.1)  # new take, top of its cycle
        assert len(s.status()) == 1

    def test_wrapping_never_reads_past_the_loop(self):
        s = lp.LoopStack()
        s.add(layer("a", n=1000))
        assert s.mix_bar(0, 1001).size == 1001


# -------------------------------------------------------------------- words

class TestParse:
    @pytest.mark.parametrize("text,want", [
        ("make a loop of this", {"op": "record"}),
        ("Make a loop of this!", {"op": "record"}),
        ("loop this", {"op": "record"}),
        ("record a loop", {"op": "record"}),
        ("make a 4 bar loop of this", {"op": "record", "bars": 4}),
        ("make a two-bar loop", {"op": "record", "bars": 2}),
        ("make a loop called bass", {"op": "record", "name": "bass"}),
        ("make a bass loop", {"op": "record", "name": "bass"}),
        ("make an 8 bar keys loop", {"op": "record", "bars": 8, "name": "keys"}),
        ("that's the loop", {"op": "close"}),
        ("close the loop", {"op": "close"}),
        ("cancel the loop", {"op": "cancel"}),
        ("mute the bass loop", {"op": "mute", "name": "bass"}),
        ("mute loop2", {"op": "mute", "name": "loop2"}),
        ("mute the loop", {"op": "mute"}),
        ("unmute the bass loop", {"op": "unmute", "name": "bass"}),
        ("bring back the bass loop", {"op": "unmute", "name": "bass"}),
        ("drop the bass loop", {"op": "drop", "name": "bass"}),
        ("delete layer keys", {"op": "drop", "name": "keys"}),
        ("swap the bass loop for the keys loop",
         {"op": "swap", "name": "bass", "other": "keys"}),
        ("swap loop1 with loop2", {"op": "swap", "name": "loop1", "other": "loop2"}),
        ("list loops", {"op": "list"}),
        ("stop the loop", {"op": "stop"}),
        ("stop looping", {"op": "stop"}),
        ("stop the loops", {"op": "stop"}),
        ("stop all the loops", {"op": "stop"}),
        ("stop the looper", {"op": "stop"}),
        ("finish looping", {"op": "stop"}),
        ("end looping", {"op": "stop"}),
        ("stop the bass loop", {"op": "stop", "name": "bass"}),
        ("make a loop of 4 bars", {"op": "record", "bars": 4}),
        ("make a loop 4 bars long", {"op": "record", "bars": 4}),
        ("make a loop for two bars", {"op": "record", "bars": 2}),
        ("make a loop of this riff", {"op": "record"}),
        ("make a loop of this riff called bass", {"op": "record", "name": "bass"}),
        ("make a loop of 4 bars called bass",
         {"op": "record", "bars": 4, "name": "bass"}),
        ("make a loop of this for 4 bars", {"op": "record", "bars": 4}),
        ("loop this for 4 bars", {"op": "record", "bars": 4}),
        ("loop the next 4 bars", {"op": "record", "bars": 4}),
        ("mute loop 1", {"op": "mute", "name": "loop1"}),
        ("mute loop one", {"op": "mute", "name": "loop1"}),
        ("unmute layer 2", {"op": "unmute", "name": "loop2"}),
        ("swap loop 1 for loop 2", {"op": "swap", "name": "loop1", "other": "loop2"}),
        ("make a loop of the chorus when we get there", {"op": "unclear"}),
        ("stop loop 1", {"op": "stop", "name": "loop1"}),
        ("stop loop1", {"op": "stop", "name": "loop1"}),
        ("stop layer 2", {"op": "stop", "name": "loop2"}),
        ("end loop 2", {"op": "stop", "name": "loop2"}),
        ("end loop2", {"op": "stop", "name": "loop2"}),
        ("close loop 2", {"op": "stop", "name": "loop2"}),
        ("stop the loop called bass", {"op": "stop", "name": "bass"}),
        ("loop stop", {"op": "stop"}),
        ("loop 1 stop", {"op": "stop", "name": "loop1"}),
        ("loop off", {"op": "stop"}),
        ("turn the bass loop off", {"op": "stop", "name": "bass"}),
        ("turn off the loop", {"op": "mute"}),
        ("pause loop 1", {"op": "mute", "name": "loop1"}),
        ("turn on loop 1", {"op": "unmute", "name": "loop1"}),
        ("mute the loop called bass", {"op": "mute", "name": "bass"}),
        ("mute loop 1 please", {"op": "mute", "name": "loop1"}),
        ("start a loop", {"op": "record"}),
        ("start looping", {"op": "record"}),
        ("begin a 4 bar loop called bass", {"op": "record", "bars": 4, "name": "bass"}),
        ("loop 4 bars", {"op": "record", "bars": 4}),
        ("loop for two bars", {"op": "record", "bars": 2}),
        ("make a loop 4 bars long called bass",
         {"op": "record", "bars": 4, "name": "bass"}),
        ("make a loop out of this", {"op": "record"}),
        ("make a loop with this", {"op": "record"}),
        ("loop the last 4 bars", {"op": "unclear"}),
        ("loop the last chorus", {"op": "unclear"}),
        ("let the loop finish", {"op": "unclear"}),
    ])
    def test_loop_words(self, text, want):
        got = lp.parse_loop_cue(text)
        assert got is not None
        for k, v in want.items():
            assert got[k] == v
        assert set(got) - set(want) <= {"text"}

    @pytest.mark.parametrize("text", [
        "darker", "drop the bass", "mute", "go to the chorus", "take it out",
        "drop to half time", "go to the loop", "", None, 7, "swap the loop",
    ])
    def test_everything_else_is_left_for_the_band(self, text):
        assert lp.parse_loop_cue(text) is None

    @pytest.mark.parametrize("text", [
        "stop the loop", "stop looping", "stop the loops", "stop the looper",
        "finish looping", "stop loop1", "stop loop 1", "stop layer 2", "loop stop",
        "loop 1 stop", "stop the loop called bass", "stop the bass layer",
        "loop the last chorus", "let the loop finish", "loop the ending",
    ])
    def test_stopping_a_loop_cannot_end_the_song(self, text):
        """Left to the band, each of these reads as the form-ending cue."""
        from plume.mind import parse_cue
        assert parse_cue(text)["kind"] == "end"
        assert lp.parse_loop_cue(text) is not None


# ----------------------------------------------- the session half, no threads

def room(n, db=-65.0, seed=5):
    amp = 10.0 ** (db / 20.0)
    return (amp * np.random.default_rng(seed).standard_normal(int(n))).astype(np.float32)


def rustle(n, at_s, *, db=-40.0, dur_s=0.05, seed=6):
    x = np.zeros(int(n), dtype=np.float32)
    k = int(dur_s * SR)
    amp = 10.0 ** (db / 20.0)
    x[int(at_s * SR):int(at_s * SR) + k] = (
        amp * np.random.default_rng(seed).standard_normal(k)).astype(np.float32)
    return x


def play(lo, audio, block=256):
    for i in range(0, audio.size, block):
        lo.feed(audio[i:i + block])


class TestLooperRunningTempo:
    """120 bpm: a bar is exactly 96000 samples, and bar b starts at b * 96000."""

    def grid(self, lo, bars, bpm=120.0):
        bar_n = int(round(4 * 60.0 / bpm * SR))
        for b in bars:
            lo.note_bar(b, b * bar_n, bpm)

    def test_the_phrase_lands_on_the_bar_it_was_played_in(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(8))
        play(lo, np.zeros(96000, dtype=np.float32))
        assert lo.handle_cue("make a loop of this")["ok"]
        x = np.zeros(96000 * 2 + 96000 * 2, dtype=np.float32)
        # a pickup on beat 3 of bar 1, then bar 2's downbeat
        for at in (48000, 96000, 120000):
            p = pluck()
            x[at:at + p.size] += p
        play(lo, x)
        assert not lo.capturing
        st = lo.status()["layers"]
        assert [(l["name"], l["bars"]) for l in st] == [("loop1", 2)]
        out = np.concatenate([lo.mix_bar(b, 96000) for b in (2, 3, 4)])
        # bar 2 is mid-cycle for a 2-bar loop anchored at bar 1: it waits
        assert not out[:96000].any()
        attack = np.flatnonzero(np.abs(out[96000:96000 * 2]) > 0.01)[0]
        assert attack == pytest.approx(48000, abs=64)  # still on beat 3
        assert lo.held_bpm() == 120.0

    def test_the_round_trip_moves_the_grid(self):
        lo = lp.Looper(SR, roundtrip_samples=4800)
        self.grid(lo, range(8))
        assert lo.handle_cue("make a loop of this")["ok"]
        x = np.zeros(int(96000 + (lp.LONE_SOUND_HOLD_S + 1.0) * SR), dtype=np.float32)
        p = pluck()
        x[96000 + 4800:96000 + 4800 + p.size] = p  # heard bar 1, played on it
        play(lo, x)
        out = lo.mix_bar(1, 96000)
        attack = np.flatnonzero(np.abs(out) > 0.01)[0]
        assert attack < 64

    def test_saying_the_bars_closes_on_the_line_without_waiting_for_silence(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(8))
        assert lo.handle_cue("make a 2 bar loop")["bars"] == 2
        t = np.arange(96000 * 4) / SR
        x = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.55 + 0.45 * np.sign(
            np.sin(2 * np.pi * 2 * t)))).astype(np.float32)
        play(lo, x[:96000 * 2 - 2048])
        assert lo.capturing
        play(lo, x[96000 * 2 - 2048:96000 * 2 + 1024])
        assert not lo.capturing
        assert lo.status()["layers"][0]["bars"] == 2

    def test_tempo_moved_too_far_during_the_take_is_refused(self):
        lo = lp.Looper(SR)
        lo.note_bar(0, 0, 120.0)
        lo.note_bar(1, 96000, 120.0)
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        x = np.zeros(96000 * 2, dtype=np.float32)
        p = pluck()
        x[:p.size] = p
        lo.note_bar(2, 192000, 110.0)      # the band has conceded 8%
        play(lo, x)
        assert not lo.capturing
        assert lo.status()["layers"] == []
        assert any("stretch" in e for e in lo.pop_events())

    def test_a_refusal_can_be_read_back(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(4))
        lo.handle_cue("make a loop")
        play(lo, np.zeros(SR // 2, dtype=np.float32))
        lo.handle_cue("that's the loop")
        play(lo, np.zeros(1024, dtype=np.float32))
        assert "heard nothing" in lo.handle_cue("list loops")["said"]

    def test_only_one_take_at_a_time(self):
        lo = lp.Looper(SR)
        assert lo.handle_cue("make a loop")["ok"]
        assert lo.handle_cue("make a loop")["ok"] is False

    def test_silence_then_close_heard_nothing(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(4))
        lo.handle_cue("make a loop")
        play(lo, np.zeros(SR // 2, dtype=np.float32))
        assert lo.handle_cue("that's the loop")["ok"]
        play(lo, np.zeros(1024, dtype=np.float32))
        assert not lo.capturing
        assert lo.status()["layers"] == []
        assert any("heard nothing" in e for e in lo.pop_events())

    def test_a_take_never_grows_without_bound(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(4))
        lo.handle_cue("make a loop")
        t = np.arange(SR) / SR
        second = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for _ in range(int(lp.MAX_TAKE_S) + 2):
            lo.feed(second)
        assert not lo.capturing

    def rustle_then_phrase(self, *, phrase_at=6.0, bpm=120.0, rustle_at=0.5,
                           rustle_db=-40.0):
        """The room, a 50 ms rustle half a second after the cue, then the phrase."""
        beat = 60.0 / bpm
        n = int((phrase_at + 8 * beat + 6.0) * SR)
        x = room(n)
        x += rustle(n, rustle_at, db=rustle_db)
        for i in range(8):
            at = int((phrase_at + i * beat) * SR)
            p = pluck()
            x[at:at + p.size] += p[:max(0, n - at)]
        return x

    def test_a_stray_sound_before_the_first_note_does_not_lose_the_phrase(self):
        """Cue, put your hands on the guitar, play."""
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        x = self.rustle_then_phrase()
        play(lo, x[:int(5.9 * SR)])
        assert lo.capturing                 # the rustle did not close it
        play(lo, x[int(5.9 * SR):])
        assert not lo.capturing
        got = lo.status()["layers"][0]
        assert got["anchor_bar"] == 3 and got["bars"] == 2

    def test_a_stray_sound_is_not_in_the_loop(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        play(lo, self.rustle_then_phrase())
        loop = lo.stack.layers[0].audio
        attack = np.flatnonzero(np.abs(loop) > 0.05)[0]
        assert attack == pytest.approx(0, abs=64)     # the phrase, on its bar line

    @pytest.mark.parametrize("before_s", [1.0, 0.5, 0.15])
    def test_a_stray_sound_just_before_the_first_note_is_not_the_phrase(self, before_s):
        """Hands find the strings, then play: too soon for the wait to help."""
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        play(lo, self.rustle_then_phrase(rustle_at=6.0 - before_s))
        got = lo.status()["layers"][0]
        assert got["anchor_bar"] == 3 and got["bars"] == 2

    def test_several_stray_sounds_before_the_phrase_are_not_the_phrase(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        x = self.rustle_then_phrase()
        for at_s in (1.5, 3.0, 4.5, 5.5):
            x += rustle(x.size, at_s)
        play(lo, x)
        got = lo.status()["layers"][0]
        assert got["anchor_bar"] == 3 and got["bars"] == 2

    def test_a_short_loud_first_note_is_still_the_first_note(self):
        """Only a sound well under the playing can be a stray one."""
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        x = self.rustle_then_phrase(rustle_at=5.5, rustle_db=-16.0)
        play(lo, x)
        assert lo.status()["layers"][0]["anchor_bar"] == 2

    def test_a_stray_sound_does_not_start_the_clock_on_a_named_length(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a 2 bar loop")
        play(lo, self.rustle_then_phrase())
        got = lo.status()["layers"][0]
        assert got["anchor_bar"] == 3 and got["bars"] == 2

    def test_a_knock_that_rings_then_a_long_gap_is_not_the_phrase(self):
        """Longer than a rustle, so only the wait saves it."""
        lo = lp.Looper(SR)
        self.grid(lo, range(40))
        lo.handle_cue("make a loop of this")
        n = int(14.0 * SR)
        x = room(n)
        p = pluck(gain=0.1)
        x[SR // 2:SR // 2 + p.size] += p
        for i in range(8):
            at = int((6.0 + i * 0.5) * SR)
            q = pluck()
            x[at:at + q.size] += q
        play(lo, x)
        got = lo.status()["layers"][0]
        assert got["anchor_bar"] == 3 and got["bars"] == 2

    def test_a_stray_sound_alone_never_becomes_a_loop(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(80))
        lo.handle_cue("make a loop of this")
        x = room(int((lp.MAX_TAKE_S + 1.0) * SR))
        x += rustle(x.size, 0.5)
        play(lo, x)
        assert not lo.capturing
        assert lo.status()["layers"] == []
        assert any("no clear start" in e for e in lo.pop_events())

    def test_a_stray_sound_alone_closed_by_hand_is_not_a_loop(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(8))
        lo.handle_cue("make a loop of this")
        x = room(2 * SR)
        x += rustle(x.size, 0.5)
        play(lo, x)
        lo.handle_cue("that's the loop")
        play(lo, room(1024))
        assert lo.status()["layers"] == []

    @pytest.mark.parametrize("how", ["by hand", "at the limit"])
    def test_room_noise_alone_is_not_a_loop(self, how):
        """Walking away after the cue must not leave a loud hiss on repeat."""
        lo = lp.Looper(SR)
        self.grid(lo, range(80))
        lo.handle_cue("make a loop of this")
        if how == "by hand":
            play(lo, room(3 * SR))
            lo.handle_cue("that's the loop")
            play(lo, room(1024))
        else:
            play(lo, room(int((lp.MAX_TAKE_S + 1.0) * SR)))
        assert not lo.capturing
        assert lo.status()["layers"] == []
        want = "heard nothing" if how == "by hand" else "no clear start"
        assert any(want in e for e in lo.pop_events())

    def test_a_held_note_with_no_quiet_in_it_is_still_a_loop_by_hand(self):
        lo = lp.Looper(SR)
        self.grid(lo, range(8))
        lo.handle_cue("make a 1 bar loop")
        t = np.arange(SR) / SR
        lo.feed((0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        lo.handle_cue("that's the loop")
        lo.feed(np.zeros(1024, dtype=np.float32))
        assert lo.status()["layers"]

    def test_stop_the_loop_mutes_every_loop(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a"))
        lo.stack.add(layer("b"))
        reply = lo.handle_cue("stop looping")
        assert reply["ok"] and "a" in reply["said"] and "b" in reply["said"]
        assert all(x["muted"] for x in lo.status()["layers"])
        assert lo.handle_cue("bring back the a loop")["ok"]    # nothing was lost

    def test_stop_the_named_loop_mutes_only_that_one(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a"))
        lo.stack.add(layer("b"))
        assert lo.handle_cue("stop the b loop")["ok"]
        assert [x["muted"] for x in lo.status()["layers"]] == [False, True]

    @pytest.mark.parametrize("text", ["stop loop 2", "stop loop2", "stop layer 2",
                                      "end loop 2", "loop 2 stop"])
    def test_stopping_one_numbered_loop_leaves_the_others(self, text):
        lo = lp.Looper(SR)
        lo.stack.add(layer("loop1"))
        lo.stack.add(layer("loop2"))
        assert lo.handle_cue(text)["ok"]
        assert [x["muted"] for x in lo.status()["layers"]] == [False, True]

    def test_stopping_a_numbered_loop_does_not_close_the_take(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("loop1"))
        lo.handle_cue("make a loop")
        assert lo.handle_cue("end loop 1")["ok"]
        assert lo.capturing and lo.status()["layers"][0]["muted"]

    def test_stop_the_loop_during_a_take_closes_the_take(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop")
        reply = lo.handle_cue("stop the loop")
        assert reply["ok"] and "closing" in reply["said"]

    def test_stop_the_loop_with_nothing_to_stop_says_so(self):
        reply = lp.Looper(SR).handle_cue("stop the loop")
        assert reply["ok"] is False and "no loop" in reply["error"]

    def test_a_loop_cue_it_cannot_follow_is_refused_and_starts_nothing(self):
        lo = lp.Looper(SR)
        reply = lo.handle_cue("make a loop of the chorus when we get there")
        assert reply["ok"] is False and "make a 4 bar loop" in reply["error"]
        assert not lo.capturing

    def test_mute_loop_1_finds_the_auto_named_loop(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("loop1"))
        assert lo.handle_cue("mute loop 1")["ok"]
        assert lo.status()["layers"][0]["muted"]

    def test_too_many_layers_is_refused(self):
        lo = lp.Looper(SR)
        for i in range(lp.MAX_LAYERS):
            lo.stack.add(layer(f"l{i}"))
        assert lo.handle_cue("make a loop")["ok"] is False

    def test_a_cue_about_nothing_comes_back_refused_not_raised(self):
        lo = lp.Looper(SR)
        reply = lo.handle_cue("mute the bass loop")
        assert reply["ok"] is False and "no loop" in reply["error"].lower()

    def test_not_a_loop_cue_returns_none(self):
        assert lp.Looper(SR).handle_cue("darker") is None

    def test_mix_follows_the_ending_fade_without_steps(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a", n=1000))
        first = lo.mix_bar(0, 1000, gain_db=0.0)
        faded = lo.mix_bar(1, 1000, gain_db=-12.0)
        assert first[-1] == pytest.approx(0.5, rel=0.01)
        assert faded[0] == pytest.approx(0.5, rel=0.01)   # picks up where it was
        assert faded[-1] == pytest.approx(0.5 * lp.db_to_lin(-12.0), rel=0.01)


class TestLooperSetsTheTempo:
    def test_with_no_grid_the_loop_proposes_one(self):
        lo = lp.Looper(SR)
        assert lo.handle_cue("make a loop of this")["ok"]
        x, lead = take(list(range(17)), 96.0, tail_s=lp.NO_TEMPO_HOLD_S + 1.5)
        play(lo, x)
        assert not lo.capturing
        tempo = lo.take_tempo()
        assert tempo is not None
        assert tempo.bpm == pytest.approx(96.0, rel=0.015)
        assert tempo.next_line_s == pytest.approx(lead / SR + 16 * 60.0 / 96.0,
                                                  rel=0.02)
        assert lo.take_tempo() is None         # handed over once
        got = lo.status()["layers"][0]
        assert got["bars"] == 4 and got["anchor_bar"] == 0
        assert lo.held_bpm() == pytest.approx(tempo.bpm)

    def test_the_length_is_whole_bars_at_that_tempo(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        x, _ = take(list(range(9)), 110.0, tail_s=lp.NO_TEMPO_HOLD_S + 1.5)
        play(lo, x)
        tempo = lo.take_tempo()
        lay = lo.stack.layers[0]
        want = int(round(lay.bars * 4 * 60.0 / tempo.bpm * SR))
        assert lay.audio.size == want

    def test_a_take_that_shows_no_tempo_is_refused_and_says_what_to_do(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        x, _ = take([0], 90.0, tail_s=lp.LONE_SOUND_HOLD_S + 1.5)
        play(lo, x)
        assert not lo.capturing
        assert lo.take_tempo() is None
        assert any("bar" in e for e in lo.pop_events())

    def test_a_stray_sound_does_not_cost_the_phrase_with_no_tempo_either(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        beat = 60.0 / 96.0
        n = int((6.0 + 16 * beat + lp.NO_TEMPO_HOLD_S + 1.5) * SR)
        x = room(n)
        x += rustle(n, 0.5)
        for i in range(17):
            at = int((6.0 + i * beat) * SR)
            p = pluck(dur_s=0.3)
            x[at:at + p.size] += p
        play(lo, x)
        assert not lo.capturing
        tempo = lo.take_tempo()
        assert tempo is not None and tempo.bpm == pytest.approx(96.0, rel=0.015)
        assert tempo.next_line_s == pytest.approx(6.0 + 16 * beat, abs=0.02)

    def test_a_stray_sound_just_before_the_first_note_costs_nothing(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        beat = 60.0 / 96.0
        n = int((6.0 + 16 * beat + lp.NO_TEMPO_HOLD_S + 1.5) * SR)
        x = room(n)
        x += rustle(n, 5.5)
        for i in range(17):
            at = int((6.0 + i * beat) * SR)
            p = pluck(dur_s=0.3)
            x[at:at + p.size] += p
        play(lo, x)
        tempo = lo.take_tempo()
        assert tempo is not None and tempo.bpm == pytest.approx(96.0, rel=0.015)
        assert tempo.next_line_s == pytest.approx(6.0 + 16 * beat, abs=0.02)

    def test_a_few_stray_sounds_wait_for_the_phrase(self):
        """Two coughs are not a take: no tempo can come of them, so keep listening."""
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        beat = 60.0 / 96.0
        n = int((7.0 + 16 * beat + lp.NO_TEMPO_HOLD_S + 1.5) * SR)
        x = room(n)
        for at_s in (0.5, 1.3):
            p = pluck(gain=0.05)
            x[int(at_s * SR):int(at_s * SR) + p.size] += p
        for i in range(17):
            at = int((7.0 + i * beat) * SR)
            p = pluck(dur_s=0.3)
            x[at:at + p.size] += p
        play(lo, x)
        tempo = lo.take_tempo()
        assert tempo is not None and tempo.bpm == pytest.approx(96.0, rel=0.015)

    @pytest.mark.parametrize("stop", ["that's the loop", "limit"])
    def test_room_noise_sets_no_tempo(self, stop):
        lo = lp.Looper(SR)
        lo.handle_cue("make a 2 bar loop")
        if stop == "limit":
            play(lo, room(int((lp.MAX_TAKE_S + 1.0) * SR)))
        else:
            play(lo, room(3 * SR))
            lo.handle_cue(stop)
            play(lo, room(1024))
        assert lo.take_tempo() is None and lo.status()["layers"] == []

    def test_three_uneven_sounds_set_no_tempo(self):
        """Coughs or laughter after the cue must not start the band."""
        lo = lp.Looper(SR)
        lo.handle_cue("make a loop of this")
        n = int((3.11 + lp.LONE_SOUND_HOLD_S + 1.5) * SR)
        x = room(n)
        for at_s in (1.0, 1.83, 3.11):
            p = pluck()
            x[int(at_s * SR):int(at_s * SR) + p.size] += p
        play(lo, x)
        assert not lo.capturing
        assert lo.take_tempo() is None and lo.status()["layers"] == []

    def test_the_join_of_a_free_take_is_clean(self):
        lo = lp.Looper(SR)
        lo.handle_cue("make a 2 bar loop")
        x, _ = take(list(range(9)), 100.0, tail_s=lp.NO_TEMPO_HOLD_S + 1.5)
        play(lo, x)
        a = lo.stack.layers[0].audio
        assert abs(a[0] - a[-1]) < 0.01


def grid_of(lo, bars=64, bpm=120.0):
    bar_n = int(round(4 * 60.0 / bpm * SR))
    for b in range(bars):
        lo.note_bar(b, b * bar_n, bpm)


class TestANamedLengthOverLegato:
    """A sustained take has no quiet in it.  It still ends on the bar line it named."""

    def closes_at(self, lo, x, block=256):
        for i in range(0, x.size, block):
            lo.feed(x[i:i + block])
            if not lo.capturing:
                return (i + block) / SR
        return None

    def test_a_held_tone_closes_on_its_bar_line(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        play(lo, room(256))
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        t = np.arange(10 * SR) / SR
        held = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        assert self.closes_at(lo, held) == pytest.approx(2.0, abs=0.02)
        assert [(x["name"], x["bars"]) for x in lo.status()["layers"]] == [("loop1", 1)]
        assert "phrase ended" in lo.status()["last_event"]

    def test_strums_that_never_decay_far_close_on_their_bar_line(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        play(lo, room(256))
        assert lo.handle_cue("make a 2 bar loop")["ok"]
        t = np.arange(int(0.25 * SR)) / SR
        strum = (0.3 * np.exp(-2.0 * t) * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        assert self.closes_at(lo, np.tile(strum, 40)) == pytest.approx(4.0, abs=0.02)
        assert lo.status()["layers"][0]["bars"] == 2

    def test_the_next_take_is_not_refused_any_more(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        play(lo, room(256))
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        t = np.arange(3 * SR) / SR
        play(lo, (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32))
        assert lo.handle_cue("make a 1 bar loop")["ok"]

    def test_a_quiet_room_still_does_not_close_it(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, room(6 * SR, db=-60.0))
        assert lo.capturing and not lo.status()["layers"]

    @pytest.mark.parametrize("db", [-44.0, -40.0, -35.0, -30.0])
    def test_steady_noise_never_starts_a_named_take(self, db):
        """Louder than a quiet room, and nobody has played: this is still waiting."""
        lo = lp.Looper(SR)
        grid_of(lo)
        play(lo, room(2048, db=db))
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, room(8 * SR, db=db))
        assert lo.capturing and not lo.status()["layers"]

    def test_a_loud_chord_already_ringing_at_the_cue_still_closes(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        t = np.arange(10 * SR) / SR
        held = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        play(lo, held[:2048])
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, held[2048:])
        assert not lo.capturing and [x["bars"] for x in lo.status()["layers"]] == [1]

    def test_noise_with_no_mic_before_the_cue_does_not_start_one_either(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, room(8 * SR, db=-40.0))
        assert lo.capturing and not lo.status()["layers"]

    def test_the_player_still_gets_their_bars_over_a_noisy_room(self):
        lo = lp.Looper(SR)
        grid_of(lo)
        play(lo, room(2048, db=-40.0))
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        wait = room(6 * SR, db=-40.0)
        play(lo, wait)
        t = np.arange(10 * SR) / SR
        held = room(t.size, db=-40.0) + (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        at = self.closes_at(lo, held)
        assert at is not None and at < 4.1, "the bar it starts in, and one more at most"
        assert [x["bars"] for x in lo.status()["layers"]] == [1]

    def test_walking_away_from_a_noisy_room_leaves_no_loop_at_the_limit(self):
        lo = lp.Looper(SR)
        grid_of(lo, bars=64)
        play(lo, room(2048, db=-40.0))
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, room(int((lp.MAX_TAKE_S + 1.0) * SR), db=-40.0))
        assert not lo.capturing and lo.status()["layers"] == []

    def test_with_no_bar_grid_a_held_tone_waits_for_the_player(self):
        """There is no bar line to close on: the docs say this."""
        lo = lp.Looper(SR)
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        t = np.arange(6 * SR) / SR
        play(lo, (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32))
        assert lo.capturing


def held_chord(n, db, *, freqs=(220.0,), seed=9):
    """A steady chord with a few harmonics, at ``db`` overall."""
    t = np.arange(int(n)) / SR
    x = sum(sum(np.sin(2 * np.pi * f * k * t + k) / k for k in (1, 2, 3)) for f in freqs)
    return (x * 10.0 ** (db / 20.0) / float(np.sqrt(np.mean(x * x)))).astype(np.float32)


def coloured_noise(n, db, slope, seed=11):
    """Noise whose power falls as 1/f**slope: 0 is hiss, 2 and 3 are a room's rumble."""
    x = np.fft.rfft(np.random.default_rng(seed).standard_normal(int(n)))
    x *= np.maximum(np.fft.rfftfreq(int(n), 1.0 / SR), 20.0) ** (-slope / 2.0)
    y = np.fft.irfft(x, int(n))
    return (y * 10.0 ** (db / 20.0) / float(np.sqrt(np.mean(y * y)))).astype(np.float32)


class TestTheTimeLimit:
    """64 seconds in, nothing has counted as a rise.  What was it?"""
    LONG = int((lp.MAX_TAKE_S + 1.0) * SR)

    def run_take(self, before, during):
        lo = lp.Looper(SR)
        grid_of(lo, bars=64)
        play(lo, before)
        assert lo.handle_cue("make a 1 bar loop")["ok"]
        play(lo, during)
        assert not lo.capturing
        return lo

    @pytest.mark.parametrize("db", [-35.0, -30.0, -27.0])
    @pytest.mark.parametrize("freqs", [(220.0,), (110.0, 138.6, 164.8, 220.0)])
    def test_a_chord_already_ringing_at_the_cue_is_kept(self, db, freqs):
        x = held_chord(self.LONG + 2048, db, freqs=freqs)
        lo = self.run_take(x[:2048], x[2048:])
        assert [y["bars"] for y in lo.status()["layers"]] == [1]
        assert "made" in lo.status()["last_event"]      # a beating chord can close sooner

    @pytest.mark.parametrize("chord_db,noise_db", [(-35.0, -42.0), (-32.0, -40.0)])
    def test_a_chord_entering_over_noise_without_a_rise_is_kept(self, chord_db, noise_db):
        x = room(self.LONG, db=noise_db)
        x[5 * SR:] += held_chord(x.size - 5 * SR, chord_db)
        lo = self.run_take(room(2048, db=noise_db), x)
        assert [y["bars"] for y in lo.status()["layers"]] == [1]

    @pytest.mark.parametrize("slope", [0.0, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("db", [-46.0, -40.0, -30.0])
    def test_noise_of_any_colour_is_not_a_held_pitch(self, slope, db):
        assert not lp.is_held_pitch(coloured_noise(self.LONG, db, slope), SR)

    @pytest.mark.parametrize("slope,db", [(0.0, -46.0), (0.0, -40.0), (0.0, -35.0), (0.0, -30.0),
                                          (1.0, -46.0), (1.0, -40.0), (1.0, -35.0)])
    def test_quiet_steady_noise_is_not_kept_and_says_what_it_was(self, slope, db):
        n = self.LONG + 2048
        x = coloured_noise(n, db, slope)
        lo = self.run_take(x[:2048], x[2048:])
        assert lo.status()["layers"] == []
        said = lo.status()["last_event"]
        assert "no clear start" in said and "heard nothing" not in said

    def test_a_quiet_melody_is_not_called_noise(self):
        """It can't be told from a room here, so the message claims neither."""
        x = room(self.LONG, db=-46.0)
        for i, at in enumerate(range(2 * SR, x.size - SR, SR // 2)):
            note = pluck(freq=220.0 + 40.0 * (i % 5), gain=0.004)
            x[at:at + note.size] += note
        lo = self.run_take(room(2048, db=-46.0), x)
        said = lo.status()["last_event"]
        assert lo.status()["layers"] == [] and "loop not made" in said
        assert "noise" not in said and "no playing" not in said
        assert "play louder" in said and "that's the loop" in said

    @pytest.mark.parametrize("cue", ["make a loop of this", "make a 2 bar loop"])
    def test_a_chord_with_nothing_running_says_what_would_work(self, cue):
        """Naming the bars can't rescue one sound, so it must not be the only advice."""
        lo = lp.Looper(SR)
        assert lo.handle_cue(cue)["ok"]
        play(lo, held_chord(self.LONG, -20.0))
        said = lo.status()["last_event"]
        assert lo.status()["layers"] == [] and "loop not made" in said
        assert "count the band in" in said

    def test_a_take_with_the_bars_named_is_not_told_to_name_them(self):
        ats = tuple(int(s * SR) for s in (0.0, 0.125, 0.375, 0.625, 0.875, 2.0))
        phrase = lp.Phrase(first=0, last_onset=ats[-1], end=ats[-1], onsets=ats)
        with pytest.raises(lp.LoopRefused, match="steady beat") as refused:
            lp.infer_tempo(phrase, SR, bars=1)
        assert "how many bars" not in str(refused.value)

    def test_a_clink_in_steady_noise_is_not_a_held_note(self):
        x = room(self.LONG, db=-40.0)
        x[20 * SR:20 * SR + SR // 4] += pluck(gain=0.03, dur_s=0.25)
        assert not lp.is_held_pitch(x, SR)

    def test_digital_silence_still_heard_nothing(self):
        lo = self.run_take(np.zeros(2048, dtype=np.float32),
                           np.zeros(self.LONG, dtype=np.float32))
        assert "heard nothing" in lo.status()["last_event"]


class TestAFadeThatEndsWithLoopsInIt:
    def test_the_last_tail_stays_at_the_faded_level(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a", n=4000))
        lo.mix_bar(0, 4000)
        lo.mix_bar(1, 4000, gain_db=-60.0)
        before = float(np.abs(lo.mix_bar(2, 4000, gain_db=-60.0)).max())
        tail = lo.mix_bar(3, lp.DECLICK_SAMPLES, stop=True)
        assert float(np.abs(tail).max()) <= before * 1.01

    def test_a_tail_at_full_level_is_still_at_full_level(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a", n=4000))
        lo.mix_bar(0, 4000)
        tail = lo.mix_bar(1, lp.DECLICK_SAMPLES, stop=True)
        assert tail[0] == pytest.approx(0.5, abs=0.01) and abs(tail[-1]) < 0.01

    def test_with_no_level_named_a_bar_keeps_the_one_it_has(self):
        lo = lp.Looper(SR)
        lo.stack.add(layer("a", n=4000))
        lo.mix_bar(0, 4000, gain_db=-20.0)
        assert float(np.abs(lo.mix_bar(1, 4000)).max()) == pytest.approx(0.05, rel=0.01)


class TestAllLoops:
    def three(self):
        lo = lp.Looper(SR)
        for name in ("bass", "keys", "lead"):
            lo.stack.add(layer(name))
        return lo

    @pytest.mark.parametrize("said", ["mute all loops", "mute all the loops",
                                      "silence all layers"])
    def test_mute_all_mutes_every_one(self, said):
        lo = self.three()
        reply = lo.handle_cue(said)
        assert reply["ok"] and all(x in reply["said"] for x in ("bass", "keys", "lead"))
        assert all(x["muted"] for x in lo.status()["layers"])

    def test_bring_back_all_brings_back_every_one(self):
        lo = self.three()
        lo.handle_cue("mute all loops")
        assert lo.handle_cue("unmute all loops")["ok"]
        assert not any(x["muted"] for x in lo.status()["layers"])

    def test_drop_all_drops_every_one(self):
        lo = self.three()
        assert lo.handle_cue("drop all loops")["ok"]
        assert lo.status()["layers"] == []

    def test_all_with_nothing_there_says_so(self):
        reply = lp.Looper(SR).handle_cue("mute all loops")
        assert reply["ok"] is False and "no loop" in reply["error"]

    def test_a_name_still_means_only_that_one(self):
        lo = self.three()
        lo.handle_cue("mute the bass loop")
        assert [x["muted"] for x in lo.status()["layers"]] == [True, False, False]

    def test_no_name_still_means_the_latest(self):
        lo = self.three()
        lo.handle_cue("mute the loop")
        assert [x["muted"] for x in lo.status()["layers"]] == [False, False, True]


class TestAnAbsurdBarCount:
    @pytest.mark.parametrize("bars", ["9" * 400, "9" * 5000, "1000"])
    @pytest.mark.parametrize("grid", [False, True])
    def test_it_is_refused_and_costs_nothing(self, bars, grid):
        lo = lp.Looper(SR)
        if grid:
            grid_of(lo)
        reply = lo.handle_cue(f"make a {bars} bar loop")
        assert reply["ok"] is False and "bars" in reply["error"] and not lo.capturing

    def test_the_longest_count_that_can_fit_is_taken_and_never_kills_the_feed(self):
        lo = lp.Looper(SR)
        assert lo.handle_cue(f"make a {lp.MAX_NAMED_BARS} bar loop")["ok"]
        x, _ = take([0, 1, 2, 3], 120.0, tail_s=lp.NO_TEMPO_HOLD_S + 1.0)
        play(lo, x)            # refused or made, it must not raise on this thread
        assert not lo.capturing
        assert lo.handle_cue(f"make a {lp.MAX_NAMED_BARS + 1} bar loop")["ok"] is False

    def test_the_parser_never_raises_on_one(self):
        assert lp.parse_loop_cue("mute loop " + "9" * 5000)["op"] == "mute"


class TestLoopsLeaveRoomForTheBand:
    N = 48000

    def band(self, level):
        t = np.arange(self.N) / SR
        wave = level * np.sin(2 * np.pi * 110.0 * t)
        return np.stack((wave, wave), axis=1).astype(np.float32)

    def stack_of(self, count, level=0.5):
        lo = lp.Looper(SR)
        t = np.arange(self.N) / SR
        for i in range(count):
            audio = (level * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
            lo.stack.add(lp.LoopLayer(name=f"l{i}", audio=audio, bars=1, bpm=120.0,
                                      anchor_bar=0))
        lo.mix_bar(0, self.N)     # all in, past the fade
        return lo

    def test_six_loops_alone_stay_inside_full_scale(self):
        lo = self.stack_of(6)
        out = lo.mix_bar(1, self.N, under=np.zeros((self.N, 2), dtype=np.float32))
        assert float(np.abs(out).max()) <= 1.0 + 1e-6

    def test_loops_never_push_a_quiet_band_past_full_scale(self):
        lo = self.stack_of(3)
        band = self.band(0.6)
        lo.mix_bar(1, self.N, under=band)
        out = lo.mix_bar(2, self.N, under=band)
        assert float(np.abs(band + out[:, None]).max()) <= 1.0 + 1e-6

    def kicks(self, level, *, every=12000, length=480):
        """Short hits with a decay, the shape that really goes past full scale."""
        band = np.zeros((self.N, 2), dtype=np.float32)
        decay = np.exp(-np.arange(length) / (length / 3.0)).astype(np.float32)
        for at in range(1000, self.N - length, every):
            band[at:at + length] = (level * decay)[:, None]
        return band

    def test_a_loop_is_not_gated_where_the_band_clips_anyway(self):
        """Dipping there costs the loop and saves nothing."""
        lo = self.stack_of(1, level=0.25)
        lo.stack.layers[0].audio[:] = 0.25
        band = self.kicks(1.8)
        lo.mix_bar(1, self.N, under=band)
        out = lo.mix_bar(2, self.N, under=band)
        clipping = np.abs(band).max(axis=1) >= 1.0
        assert clipping.any()
        assert float(np.abs(out[clipping]).min()) == pytest.approx(0.25, abs=1e-5)

    def test_past_a_clipped_peak_loops_still_add_no_clipping(self):
        lo = self.stack_of(1, level=0.25)
        lo.stack.layers[0].audio[:] = 0.25
        band = self.kicks(1.8)
        lo.mix_bar(1, self.N, under=band)
        out = lo.mix_bar(2, self.N, under=band)
        peak = np.abs(band).max(axis=1)
        reach = 2 * lp.DECLICK_SAMPLES + 1
        near = lp.maximum_filter1d((peak >= 1.0).astype(np.uint8), size=reach) > 0
        assert not (np.abs(band[:, 0] + out)[~near] > 1.0 + 1e-6).any()

    def test_a_band_just_under_full_scale_still_gets_its_dip(self):
        lo = self.stack_of(1, level=0.25)
        lo.stack.layers[0].audio[:] = 0.25
        band = self.kicks(0.95)
        lo.mix_bar(1, self.N, under=band)
        out = lo.mix_bar(2, self.N, under=band)
        assert float(np.abs(band[:, 0] + out).max()) <= 1.0 + 1e-6
        assert float(np.abs(out).min()) < 0.25

    def test_a_dipped_level_carries_on_with_no_band_to_give_way_to(self):
        """A tail has no band under it, and must start where the last bar ended."""
        lo = self.stack_of(1, level=0.25)
        lo.stack.layers[0].audio[:] = 0.25
        band = np.zeros((self.N, 2), dtype=np.float32)
        band[-1000:] = 0.95
        last = lo.mix_bar(1, self.N, under=band)
        tail = lo.mix_bar(2, self.N)
        assert float(last[-1]) < 0.1
        assert float(tail[0]) == pytest.approx(float(last[-1]), abs=0.01)

    def test_with_room_to_spare_nothing_changes(self):
        lo = self.stack_of(1, level=0.25)
        out = lo.mix_bar(1, self.N, under=self.band(0.3))
        plain = self.stack_of(1, level=0.25).mix_bar(1, self.N)
        assert np.array_equal(out, plain)

    def test_giving_way_never_steps(self):
        """Clicks are the other way to ruin it: the dip is a slope, also across bars."""
        lo = self.stack_of(1, level=0.25)
        steady = np.full(self.N, 0.25, dtype=np.float32)
        lo.stack.layers[0].audio = steady
        hit = np.zeros((self.N, 2), dtype=np.float32)
        hit[:400] = 0.95            # right at the top of the bar
        hit[20000:20400] = 0.95
        quiet = np.zeros((self.N, 2), dtype=np.float32)
        out = np.concatenate([lo.mix_bar(b, self.N, under=u)
                              for b, u in ((1, quiet), (2, hit), (3, quiet))])
        assert float(np.abs(np.diff(out)).max()) <= 0.25 / lp.HEAD_FADE_SAMPLES * 1.5
        assert float(out[self.N * 2 + self.N // 2]) == pytest.approx(0.25)

    def test_a_band_of_another_length_does_not_matter(self):
        lo = self.stack_of(1)
        assert lo.mix_bar(1, self.N, under=np.zeros((self.N - 3, 2), np.float32)).size == self.N
        assert lo.mix_bar(2, self.N, under=np.zeros((self.N + 3, 2), np.float32)).size == self.N

    def test_silence_in_a_loop_over_a_loud_band_stays_silence(self):
        lo = self.stack_of(1)
        lo.stack.layers[0].audio[:] = 0.0
        lo.mix_bar(1, self.N)
        with np.errstate(invalid="raise", divide="raise"):
            out = lo.mix_bar(2, self.N, under=self.band(1.5))
        assert np.isfinite(out).all() and not out.any()


@pytest.mark.parametrize("said", ["drop all of the loops", "mute every loop", "mute both loops",
                                  "mute all of the loops", "unmute each loop"])
def test_every_way_of_saying_all_the_loops_reaches_all_of_them(said):
    cue = lp.parse_loop_cue(said)
    assert cue["every"] is True and "name" not in cue


@pytest.mark.parametrize("word", ["both", "every", "each", "all"])
def test_a_loop_cannot_be_called_what_means_every_loop(word):
    assert lp.parse_loop_cue(f"make a loop called {word}")["op"] == "unclear"
    lo = lp.Looper(SR)
    assert lo.handle_cue(f"record a loop named {word}")["ok"] is False
    assert not lo.capturing


def test_mute_both_loops_with_three_mutes_all_and_says_so():
    lo = lp.Looper(SR)
    for name in "abc":
        lo.stack.add(layer(name))
    said = lo.handle_cue("mute both loops")["said"]
    assert all(name in said for name in "abc")
    assert [y["muted"] for y in lo.status()["layers"]] == [True] * 3
