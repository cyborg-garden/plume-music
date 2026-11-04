"""Live MIDI input, proven on a fake port and a fake audio device only.

Nothing here opens a real port or a real speaker, so none of it says a real
keyboard works: it says that the right notes at the right times reach the same
beat and harmony code the microphone feeds.
"""
import collections
import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from conftest import (FakeMidiMessage as Msg, FakeMidiPort, FakeMido, fake_audio, patched_module,
                      short_socket_path, wait_until)

from plume import cli, midi, mind as mind_mod, realtime
from plume.bandentry import GuitarEar
from plume.listen.beat import BeatTracker
from plume.listen.features import HOP, CountInDetector
from plume.listen.harmony import HarmonyTracker
from plume.watch import Listener

SR = realtime.SR
C, F, G = (60, 64, 67), (53, 57, 60), (55, 59, 62)


def on(t, note, vel=90, ch=0):
    return (t, "on", note, vel, ch)


def off(t, note, ch=0):
    return (t, "off", note, 0, ch)


def frames_of(events, seconds, *, log=None, block=512):
    """Run an event list through MidiFrames for a stretch of stream time."""
    source = midi.MidiFrames(collections.deque(sorted(events, key=lambda e: e[0])),
                             sr=SR, log=log)
    out = []
    for _ in range(int(seconds * SR / block)):
        out.extend(source.process(np.zeros(block, dtype=np.float32)))
    return out


def chords(pattern, *, bpm=120.0, bars=16, hold=0.9, vel=90):
    """One chord per bar, struck on each beat, sounding for ``hold`` of a beat."""
    beat = 60.0 / bpm
    out = []
    for bar in range(bars):
        notes = pattern[bar % len(pattern)]
        for b in range(4):
            t = (bar * 4 + b) * beat + 0.5
            for n in notes:
                out += [on(t, n, vel), off(t + hold * beat, n)]
    return out


# ---------------------------------------------------------------- notes to frames

class TestFrames:
    def test_a_note_is_an_onset_on_the_first_frame_at_or_after_it(self):
        frames = frames_of([on(1.0, 60)], 1.2)
        onsets = [f for f in frames if f.onset]
        assert len(onsets) == 1
        assert 1.0 <= onsets[0].t_s < 1.0 + HOP / SR + 1e-9
        assert onsets[0].onset_strength > 0.0

    def test_chroma_is_exactly_the_notes_held(self):
        events = [on(0.5, n) for n in C]
        f = frames_of(events, 1.0)[-1]
        assert f.chroma.shape == (12,)
        assert abs(float(f.chroma.sum()) - 1.0) < 1e-5
        assert {int(i) for i in np.flatnonzero(f.chroma)} == {0, 4, 7}

    def test_louder_playing_reads_louder_and_stays_above_plume_silence(self):
        soft = frames_of([on(0.1, 60, 1)], 0.5)[-1].rms_db
        loud = frames_of([on(0.1, 60, 127)], 0.5)[-1].rms_db
        assert mind_mod.MindParams().silence_rms_db < soft < loud < 0.0

    def test_a_note_off_rings_briefly_then_goes_silent(self):
        frames = frames_of([on(0.1, 60), off(0.2, 60)], 2.0)
        during_ring = next(f for f in frames if f.t_s > 0.25)
        assert during_ring.chroma[0] > 0.0
        last = frames[-1]
        assert last.rms_db == realtime.RMS_FLOOR_DB
        assert float(last.chroma.sum()) == 0.0

    def test_low_notes_carry_bass_evidence_and_high_notes_do_not(self):
        low = frames_of([on(0.1, 36)], 0.5)[-1]
        high = frames_of([on(0.1, 84)], 0.5)[-1]
        assert low.chroma_low[0] > 0.2
        assert float(high.chroma_low.sum()) == 0.0

    def test_a_rolled_chord_is_one_onset(self):
        frames = frames_of([on(0.5, 60), on(0.515, 64), on(0.53, 67)], 1.0)
        assert sum(f.onset for f in frames) == 1

    def test_framing_does_not_depend_on_the_blocking(self):
        events = [on(0.3, 60), off(0.5, 60), on(0.7, 64)]
        a = frames_of(events, 1.0, block=256)
        b = frames_of(events, 1.0, block=1024)
        n = min(len(a), len(b))
        assert [f.t_s for f in a[:n]] == [f.t_s for f in b[:n]]
        assert [f.onset for f in a[:n]] == [f.onset for f in b[:n]]

    def test_the_sustain_pedal_keeps_a_released_note_sounding(self):
        events = [(0.1, "pedal", 127, 0, 0), on(0.2, 60), off(0.3, 60)]
        assert frames_of(events, 3.0)[-1].chroma[0] > 0.0
        events.append((1.0, "pedal", 0, 0, 0))
        assert float(frames_of(events, 3.0)[-1].chroma.sum()) == 0.0

    def test_all_notes_off_really_is(self):
        events = [on(0.1, 60), on(0.1, 64), (0.5, "all_off", 0, 0, 0)]
        assert float(frames_of(events, 2.0)[-1].chroma.sum()) == 0.0

    def test_a_drum_channel_note_is_a_hit_with_no_pitch(self):
        frames = frames_of([on(0.5, 36, 100, ch=9)], 1.0)
        assert sum(f.onset for f in frames) == 1
        assert all(float(f.chroma.sum()) == 0.0 for f in frames)
        assert max(f.rms_db for f in frames) > mind_mod.MindParams().silence_rms_db


# ---------------------------------------------- the same harmony and beat code

class TestUnderstanding:
    def test_harmony_reads_the_key_and_the_chords(self):
        tracker = HarmonyTracker()
        state = tracker.process(frames_of(chords([C, F, G, C]), 34.0))
        assert state.key == "C major"
        # Each chord, alone, on the same code path.
        for notes, name in ((F, "F"), (G, "G")):
            state = HarmonyTracker().process(
                frames_of(chords([notes], bars=2), 3.0))
            assert state.chord == name

    def test_short_stabs_still_read_as_a_key(self):
        """A stab is as much harmony as a held chord.  With no ring, the voiced
        share of the bar would fall below what the key tracker needs."""
        state = HarmonyTracker().process(
            frames_of(chords([C, F, G, C], hold=0.15), 34.0))
        assert state.key == "C major"

    @pytest.mark.parametrize("bpm", [72.0, 96.0, 120.0, 140.0])
    def test_the_beat_tracker_locks_on_the_player(self, bpm):
        tracker = BeatTracker()
        state = tracker.process(frames_of(chords([C, F], bpm=bpm, bars=24), 40.0))
        assert state.locked
        assert state.bpm == pytest.approx(bpm, rel=0.03)

    def test_it_locks_through_human_timing(self):
        rng = np.random.default_rng(7)
        events = [(e[0] + float(rng.normal(0, 0.008)) if e[1] == "on" else e[0], *e[1:])
                  for e in chords([C, F], bpm=100.0, bars=24)]
        state = BeatTracker().process(frames_of(events, 40.0))
        assert state.locked
        assert state.bpm == pytest.approx(100.0, rel=0.03)

    @pytest.mark.parametrize("bpm", [120.0, 140.0])
    def test_chords_on_the_beat_with_notes_between_read_as_the_beat(self, bpm):
        """How a chord outweighs a single note is what stops a 2/3-tempo reading."""
        rng, beat, events = np.random.default_rng(3), 60.0 / bpm, []
        for i in range(int(40.0 / beat * 2)):
            t = 0.5 + i * beat / 2.0 + float(rng.normal(0, 0.01))
            if i % 2 == 0:
                notes, vel = (C if (i // 8) % 2 == 0 else F), 90
            elif rng.random() > 0.3:
                notes, vel = (72,), 60
            else:
                continue
            events += [e for n in notes for e in (on(t, n, vel), off(t + beat * 0.4, n))]
        state = BeatTracker().process(frames_of(events, 40.0))
        assert state.locked
        assert state.bpm == pytest.approx(bpm, rel=0.03)

    def test_acoustic_mode_reads_it_too(self):
        state = BeatTracker(acoustic=True).process(
            frames_of(chords([C, F], bars=24), 40.0))
        assert state.locked
        assert state.bpm == pytest.approx(120.0, rel=0.03)

    @pytest.mark.parametrize("vel", [20, 110])
    def test_four_even_key_presses_are_a_tap_start_count(self, vel):
        """The detectors lead --tap-start builds, with its own settings."""
        detector = CountInDetector(sr=SR, floor=realtime.TAP_START_KNOCK_FLOOR,
                                   waiting=True, even_frac=0.15,
                                   bpm_range=(45.0, 170.0))
        events = [on(1.0 + i * 0.6, 60, vel) for i in range(4)]
        events += [off(1.1 + i * 0.6, 60) for i in range(4)]
        taps = detector.process(frames_of(events, 3.0))
        assert len(taps) == 4
        assert taps[1] - taps[0] == pytest.approx(0.6, abs=0.02)

    def test_the_guitar_ear_hears_a_keyboard(self):
        ear = GuitarEar()
        frames = frames_of(chords([C], bars=4), 8.0)
        bars = [[f.chroma for f in frames if 2.0 * b <= f.t_s < 2.0 * (b + 1)]
                for b in range(3)]
        assert [ear.bar(b) for b in bars][-1] is True
        assert set(ear.notes(3)) == {0, 4, 7}

    def test_a_bar_of_playing_and_a_bar_of_nothing(self):
        params = mind_mod.MindParams()
        frames = frames_of(chords([C], bars=1), 8.0)
        def bar(lo, hi):
            return realtime.bar_features_dict(
                0, [(f.t_s, f.rms_db, f.onset) for f in frames if lo <= f.t_s < hi],
                silence_rms_db=params.silence_rms_db)
        assert not bar(0.5, 2.5)["silent"]
        assert bar(5.0, 7.0)["silent"]

    def test_the_mic_masker_is_not_applied_to_keys(self):
        masker = midi.MidiMasker()
        frames = frames_of(chords([C], bars=1), 2.0)
        out, flags = masker.mask(frames, [0.0] * len(frames))
        assert out == frames and not any(flags)
        assert masker.set_latency_hops(5) is masker


# ------------------------------------------------------------------- MIDI clock

def clock_ticks(n, bpm, *, start=None, t0=0.5):
    tick = 60.0 / bpm / 24.0
    events = [] if start is None else [(start, "start", 0, 0, 0)]
    events += [(t0 + i * tick, "tick", 0, 0, 0) for i in range(n)]
    return events


def run_clock(tracker, events, seconds, *, follow=True, **kw):
    source = midi.MidiFrames(collections.deque(sorted(events, key=lambda e: e[0])), sr=SR,
                             follow_clock=follow, **kw)
    state = None
    for _ in range(int(seconds * SR / 512)):
        frames = source.process(np.zeros(512, dtype=np.float32))
        source.apply_clock(tracker)
        state = tracker.process(frames)
    return state, source


class TestClock:
    def test_clock_sets_the_tempo_with_no_notes_at_all(self):
        state, _ = run_clock(BeatTracker(), clock_ticks(24 * 12, 97.0), 8.0)
        assert state.locked
        assert state.bpm == pytest.approx(97.0, rel=0.01)

    def test_clock_follows_a_tempo_change_and_keeps_counting_bars(self):
        tracker = BeatTracker()
        first = clock_ticks(24 * 16, 120.0)
        end = first[-1][0]
        second = [(end + (i + 1) * 60.0 / 90.0 / 24.0, "tick", 0, 0, 0)
                  for i in range(24 * 16)]
        state_a, source = run_clock(tracker, first, end + 0.05)
        bar_a = state_a.bar
        q = source._events                        # the same stream, carrying on
        q.extend(second)
        for _ in range(int((second[-1][0] - end) * SR / 512)):
            frames = source.process(np.zeros(512, dtype=np.float32))
            source.apply_clock(tracker)
            state = tracker.process(frames)
        assert state.bpm == pytest.approx(90.0, rel=0.015)
        assert state.bar > bar_a                  # a bar reset would silence jam

    @pytest.mark.parametrize("nudge", [-0.2, 0.0, 0.2])
    def test_one_bar_for_every_four_clock_beats(self, nudge):
        """Wherever the tracker's own grid happened to be when the clock came in."""
        tracker = BeatTracker()
        state, source = run_clock(tracker, chords([C], bars=8), 12.0)
        assert state.locked
        bar = state.bar
        source._events.extend(clock_ticks(24 * 40, 120.0, t0=12.0 + nudge + 0.25))
        for _ in range(int(19.0 * SR / 512)):
            frames = source.process(np.zeros(512, dtype=np.float32))
            source.apply_clock(tracker)
            state = tracker.process(frames)
        assert state.bar - bar == pytest.approx(19.0 / 2.0, abs=1.5)

    def test_start_puts_the_downbeat_on_the_first_tick(self):
        state, _ = run_clock(BeatTracker(), clock_ticks(24 * 9, 120.0, start=0.45), 5.0)
        assert state.locked
        # ticks begin at 0.5 and beat 1 returns every 2 s
        assert (state.next_downbeat_t_s - 0.5) % 2.0 == pytest.approx(0.0, abs=0.03) \
            or (state.next_downbeat_t_s - 0.5) % 2.0 == pytest.approx(2.0, abs=0.03)

    def test_a_clock_arriving_after_a_lock_never_moves_the_bar_backwards(self):
        tracker = BeatTracker()
        notes = chords([C, F], bars=20)
        state, source = run_clock(tracker, notes, 30.0)
        assert state.locked and state.bar > 2
        bar = state.bar
        source._events.extend(clock_ticks(24 * 8, 120.0, t0=30.1))
        for _ in range(int(5.0 * SR / 512)):
            frames = source.process(np.zeros(512, dtype=np.float32))
            source.apply_clock(tracker)
            state = tracker.process(frames)
        assert state.bar >= bar

    def test_absurd_clock_is_ignored(self):
        tracker = BeatTracker()
        tracker.clock_beat(1.0, 9000.0, 0)
        assert not tracker.process([]).locked

    def test_a_clock_nobody_asked_to_follow_changes_nothing(self):
        """A keyboard or a DAW can send clock the player is not playing to."""
        notes = chords([C, F], bpm=90.0, bars=26)
        alone, _ = run_clock(BeatTracker(), notes, 70.0, follow=False)
        state, _ = run_clock(BeatTracker(), notes + clock_ticks(24 * 140, 120.0), 70.0,
                             follow=False)
        assert state.bpm == pytest.approx(90.0, rel=0.02)
        assert state.bpm == alone.bpm
        assert state.next_beat_t_s == alone.next_beat_t_s

    def test_an_ignored_clock_is_mentioned_once(self):
        told, logged = [], []
        run_clock(BeatTracker(), clock_ticks(24 * 4, 120.0), 2.0, follow=False,
                  on_clock=told.append, log=logged.append)
        assert told == ["ignored"]
        assert [d["type"] for d in logged] == ["clock_ignored"]

    @pytest.mark.parametrize("stop", [False, True])
    def test_a_clock_that_ends_hands_the_tempo_back_to_the_player(self, stop):
        """Stop may be sent, and a device may well keep ticking after it."""
        events = clock_ticks(24 * 16, 120.0)
        if stop:
            events += clock_ticks(24 * 140, 120.0, t0=8.5)
            events.append((8.45, "stop", 0, 0, 0))
        notes = [(e[0] + 8.5, *e[1:]) for e in chords([C, F], bpm=90.0, bars=23)]
        told = []
        state, _ = run_clock(BeatTracker(), events + notes, 68.0, on_clock=told.append)
        assert state.bpm == pytest.approx(90.0, rel=0.02)
        assert told == ["stopped"]

    def test_a_count_in_outlives_an_old_clock(self):
        tracker = BeatTracker()
        tracker.clock_beat(1.0, 120.0, 0)
        for i in range(4):
            tracker.tap(2.0 + i * 0.75)
        tracker.clock_released()
        assert tracker.process([]).bpm == pytest.approx(80.0)
        assert tracker._tap_anchored

    @pytest.mark.parametrize("gap", [
        [(9.0, "lost", 0, 0, 0), (14.0, "back", 0, 0, 0)],
        [],
        [(9.0, "stop", 0, 0, 0), (14.0, "continue", 0, 0, 0)],
    ], ids=["cable", "silent", "stop-continue"])
    @pytest.mark.parametrize("pause", [0.6, 1.5, 5.0])
    def test_a_pause_in_the_clock_is_not_a_tempo(self, gap, pause):
        gap = [(9.0 + (e[0] - 9.0) * pause / 5.0, *e[1:]) for e in gap]
        events = clock_ticks(24 * 17, 120.0) + clock_ticks(24 * 16, 120.0, t0=9.0 + pause)
        stated = RecordingTracker()
        run_clock(stated, events + gap, 18.0 + pause)
        assert stated.beats[-1][0] > 9.0 + pause
        assert {round(b[1]) for b in stated.beats} == {120}

    @pytest.mark.parametrize("gap,resume,place", [
        ([(8.6, "lost", 0, 0, 0), (9.0, "back", 0, 0, 0)], 9.0, 2),
        ([(8.6, "stop", 0, 0, 0), (9.0, "continue", 0, 0, 0)], 9.0, 3),
        ([(8.6, "stop", 0, 0, 0), (13.0, "continue", 0, 0, 0)], 13.5, 3),
        ([], 13.5, 2),
        ([(8.6, "stop", 0, 0, 0), (8.8, "lost", 0, 0, 0), (9.0, "back", 0, 0, 0)], 9.5, 2),
    ], ids=["cable", "stop-continue", "long-stop-continue", "silence", "stop-then-cable"])
    def test_where_the_bar_is_after_a_gap(self, gap, resume, place):
        """A cable or a silence loses the place, so the first tick is beat one, as it
        is when a clock arrives with no Start.  Continue means carry on from here,
        however long the Stop lasted."""
        first = clock_ticks(24 * 4 + 12, 120.0, t0=6.25)      # stops mid beat 1 of bar 2
        second = clock_ticks(24 * 16, 120.0, t0=resume)
        stated = RecordingTracker()
        run_clock(stated, first + gap + second, resume + 8.0)
        after = [b for b in stated.beats if b[0] >= resume]
        # Two gaps before a tempo is claimed, so the third beat heard is the first
        # stated: index 2 of a new count, or index 7 of the count that carried on.
        assert after[0][2] == place


class RecordingTracker:
    """What a MidiFrames told the beat tracker."""

    def __init__(self):
        self.beats, self.released = [], 0

    def clock_beat(self, t_s, bpm, beat_in_bar):
        self.beats.append((t_s, bpm, beat_in_bar))

    def clock_released(self):
        self.released += 1

    def process(self, _frames):
        return None


# -------------------------------------------------------------- ports, fake only

class TestPorts:
    def test_listing(self):
        with patched_module("mido", FakeMido(["A", "B"])):
            assert midi.list_ports() == ["A", "B"]

    def test_no_library_says_how_to_get_it(self):
        with patched_module("mido", None):
            with pytest.raises(midi.MidiError, match=r"\.\[midi\]"):
                midi.list_ports()

    def test_a_missing_backend_is_a_clean_error_too(self):
        class NoBackend:
            def get_input_names(self):
                raise ImportError("No module named 'rtmidi'")
        with patched_module("mido", NoBackend()):
            with pytest.raises(midi.MidiError, match=r"\.\[midi\]"):
                midi.list_ports()

    def test_auto_takes_the_only_port(self):
        with patched_module("mido", FakeMido(["Only"])):
            assert midi.resolve_port("auto") == "Only"

    def test_auto_refuses_to_guess(self):
        with patched_module("mido", FakeMido(["A", "B"])):
            with pytest.raises(midi.MidiError, match="A.*B"):
                midi.resolve_port("auto")

    def test_a_name_matches_a_substring_ignoring_case(self):
        with patched_module("mido", FakeMido(["Studio Keys", "Pads"])):
            assert midi.resolve_port("keys") == "Studio Keys"

    def test_an_exact_name_beats_a_substring(self):
        with patched_module("mido", FakeMido(["Keys", "Keys 2"])):
            assert midi.resolve_port("Keys") == "Keys"

    def test_ambiguous_and_absent_names_are_refused(self):
        with patched_module("mido", FakeMido(["Keys 1", "Keys 2"])):
            with pytest.raises(midi.MidiError, match="Keys 1"):
                midi.resolve_port("keys")
            with pytest.raises(midi.MidiError, match="nope"):
                midi.resolve_port("nope")

    def test_a_driver_that_never_lists_is_refused_in_bounded_time(self, monkeypatch):
        monkeypatch.setattr(midi, "LIST_TIMEOUT_S", 0.05)
        let_go = threading.Event()

        class Stuck(FakeMido):
            def get_input_names(self):
                let_go.wait(20.0)
                return []
        with patched_module("mido", Stuck()):
            t0 = time.monotonic()
            try:
                with pytest.raises(midi.MidiError, match="did not list"):
                    midi.resolve_port("auto")
            finally:
                let_go.set()
        assert time.monotonic() - t0 < 5.0

    def test_a_listing_error_still_reads_as_itself(self):
        with patched_module("mido", FakeMido(fail_list=True)):
            with pytest.raises(midi.MidiError, match="backend trouble"):
                midi.list_ports()

    def test_no_ports_at_all(self):
        with patched_module("mido", FakeMido([])):
            with pytest.raises(midi.MidiError, match="no MIDI input"):
                midi.resolve_port("auto")

    def test_importing_plume_never_imports_the_library(self):
        code = ("import sys, plume.realtime, plume.watch, plume.cli, plume.midi;"
                "sys.exit(1 if 'mido' in sys.modules or 'rtmidi' in sys.modules else 0)")
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def make_input(fake, **kw):
    now = kw.pop("now", None) or [1.0]
    with patched_module("mido", fake):
        source = midi.MidiInput("Fake Keys", clock=lambda: now[0], poll_s=0.0, **kw)
        source.start()
    return source, now


class TestInput:
    def test_messages_become_stamped_events(self):
        fake = FakeMido()
        source, now = make_input(fake)
        fake.port.send(Msg("note_on", note=60, velocity=90))
        now[0] = 1.5
        fake.port.send(Msg("note_on", note=60, velocity=0))
        fake.port.send(Msg("note_off", note=62, velocity=30, channel=3))
        fake.port.send(Msg("control_change", control=64, value=127))
        fake.port.send(Msg("control_change", control=123, value=0))
        fake.port.send(Msg("clock"))
        fake.port.send(Msg("start"))
        fake.port.send(Msg("active_sensing"))
        assert list(source.events) == [
            (1.0, "on", 60, 90, 0), (1.5, "off", 60, 0, 0), (1.5, "off", 62, 0, 3),
            (1.5, "pedal", 127, 0, 0), (1.5, "all_off", 0, 0, 0),
            (1.5, "tick", 0, 0, 0), (1.5, "start", 0, 0, 0)]
        source.close()

    def test_the_transport_messages_are_read(self):
        fake = FakeMido()
        source, _ = make_input(fake)
        fake.port.send(Msg("stop"))
        fake.port.send(Msg("continue"))
        assert [e[1] for e in source.events] == ["stop", "continue"]
        source.close()

    def test_the_offset_moves_stamps_onto_the_mic_clock(self):
        fake = FakeMido()
        source, _ = make_input(fake)
        source.offset_s = 0.25
        fake.port.send(Msg("note_on", note=60, velocity=90))
        assert source.events[0][0] == pytest.approx(1.25)
        source.close()

    def test_nonsense_from_the_library_never_raises_into_its_thread(self):
        fake = FakeMido()
        source, _ = make_input(fake)
        fake.port.callback(object())
        fake.port.callback(Msg("note_on"))         # no note, no velocity
        assert not source.events
        source.close()

    def test_a_clock_that_raises_never_raises_into_the_librarys_thread(self):
        fake = FakeMido()
        with patched_module("mido", fake):
            source = midi.MidiInput("Fake Keys", clock=lambda: 1 / 0, poll_s=0.0)
            source.start()
            fake.port.send(Msg("note_on", note=60, velocity=90))
        assert not source.events
        source.close()

    def test_unplugging_is_reported_and_replugging_recovers(self):
        fake = FakeMido()
        changes = []
        source, _ = make_input(fake, on_change=changes.append)
        first = fake.port
        with patched_module("mido", fake):
            source.check()
            assert source.connected and not changes
            fake.names.clear()
            source.check()
            assert not source.connected and first.closed
            source.check()                         # says it once
            fake.names.append("Fake Keys")
            source.check()
        assert source.connected and fake.port is not first
        assert changes == ["lost", "back"]
        assert [e[1] for e in source.events] == ["lost", "back"]
        source.close()

    def test_a_listing_failure_is_not_a_disconnection(self):
        fake = FakeMido()
        source, _ = make_input(fake)
        fake.fail_list = True
        with patched_module("mido", fake):
            source.check()
        assert source.connected
        source.close()

    def test_a_dead_port_that_raises_on_close_is_survived(self):
        fake = FakeMido(fail_close=True)
        source, _ = make_input(fake)
        fake.names.clear()
        with patched_module("mido", fake):
            source.check()
        assert not source.connected
        source.close()

    def test_a_port_that_vanished_before_start_is_retried(self):
        fake = FakeMido(fail_open=True)
        changes = []
        source, _ = make_input(fake, on_change=changes.append)
        assert not source.connected
        fake.fail_open = False
        with patched_module("mido", fake):
            source.check()
        assert source.connected and changes == ["lost", "back"]
        source.close()

    def test_a_close_that_hangs_cannot_hang_the_ending(self):
        fake = FakeMido()
        source, _ = make_input(fake)
        fake.port.close = lambda: time.sleep(30)
        t0 = time.monotonic()
        source.close(timeout=0.1)
        assert time.monotonic() - t0 < 5.0

    def test_a_close_that_hangs_on_unplug_still_reports_it_and_reconnects(self, monkeypatch):
        """The driver for a dead device may never return, and that is when it closes."""
        monkeypatch.setattr(midi, "CLOSE_TIMEOUT_S", 0.05)
        fake = FakeMido()
        changes, release = [], threading.Event()
        source, _ = make_input(fake, on_change=changes.append)
        seen_at_close = []

        def hang():
            seen_at_close.append(list(changes))
            release.wait(30.0)
        fake.port.close = hang
        fake.names.clear()
        t0 = time.monotonic()
        with patched_module("mido", fake):
            source.check()
            assert time.monotonic() - t0 < 5.0
            assert changes == ["lost"] and [e[1] for e in source.events] == ["lost"]
            assert wait_until(lambda: seen_at_close == [["lost"]])  # told before it hung
            fake.names.append("Fake Keys")
            source.check()
        release.set()
        assert source.connected and changes == ["lost", "back"]
        source.close()

    def test_held_notes_do_not_stick_when_the_cable_goes(self):
        events = [on(0.1, 60), (0.5, "lost", 0, 0, 0)]
        logged = []
        frames = frames_of(events, 2.0, log=logged.append)
        assert float(frames[-1].chroma.sum()) == 0.0
        assert [d["type"] for d in logged] == ["on", "off", "lost"]

    def test_the_log_has_what_a_listener_needs(self):
        logged = []
        frames_of([on(0.1, 60, 64, ch=2), off(0.2, 60, ch=2)], 0.5, log=logged.append)
        assert logged[0]["type"] == "on" and logged[0]["note"] == 60
        assert logged[0]["velocity"] == 64 and logged[0]["channel"] == 2
        assert logged[0]["t_s"] == pytest.approx(0.1)
        assert logged[1]["type"] == "off" and logged[1]["t_s"] == pytest.approx(0.2)


@pytest.mark.usefixtures("fake_device_consent")
class TestClockIsAskedFor:
    def analysed(self, port_kw):
        """Clock at 120 through the live wiring, with no notes."""
        class Timeline:
            playhead = 0
        source, frames, masker = realtime._midi_listening(
            "Fake Keys", Timeline(), None, **port_kw)
        source.events.extend(clock_ticks(24 * 12, 120.0))
        tracker, state = BeatTracker(), None
        for _ in range(int(6.0 * SR / 512)):
            _f, _m, state, _h = realtime.analysis_step(
                extractor=frames, hop_rms=realtime.HopRms(), masker=masker,
                tracker=tracker, harmony=HarmonyTracker(),
                human_block=np.zeros(512, dtype=np.float32),
                machine_block=np.zeros(512, dtype=np.float32))
        return state

    def test_the_live_wiring_ignores_a_clock_and_says_so(self, capsys):
        state = self.analysed({})
        assert not state.locked and state.bpm == 0.0
        err = capsys.readouterr().err
        assert err.count("--midi-clock") == 1

    def test_the_live_wiring_follows_it_when_told(self, capsys):
        state = self.analysed({"follow_clock": True})
        assert state.locked and state.bpm == pytest.approx(120.0, rel=0.01)
        assert "--midi-clock" not in capsys.readouterr().err

    def test_the_end_of_a_followed_clock_is_said(self, capsys):
        class Timeline:
            playhead = 0
        source, frames, _m = realtime._midi_listening(
            "Fake Keys", Timeline(), None, follow_clock=True)
        source.events.extend(clock_ticks(24 * 4, 120.0))
        for _ in range(int(4.0 * SR / 512)):
            frames.process(np.zeros(512, dtype=np.float32))
        assert "clock stopped" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", [False, True])
    def test_jam_hands_the_choice_to_the_listening(self, flag, tmp_path, plume_registry,
                                                   monkeypatch):
        seen, real = [], realtime._midi_listening

        def listening(*a, **kw):
            seen.append(kw.get("follow_clock"))
            return real(*a, **kw)
        monkeypatch.setattr(realtime, "_midi_listening", listening)
        sock = short_socket_path("midi-flag")
        sd, mido = fake_audio(None, rate_x=12.0), FakeMido()
        stopper = stop_at(sock, lambda: (sd.stream.blocks * 256 / SR) if sd.stream else 0.0, 2.0)
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), socket_path=sock,
                         midi="auto", midi_clock=flag)
        stopper.join(timeout=5.0)
        assert seen == [flag]

    @pytest.mark.parametrize("flag", [False, True])
    def test_lead_hands_the_choice_to_the_listening(self, flag, tmp_path, plume_registry,
                                                    monkeypatch):
        seen, real = [], realtime._midi_listening

        def listening(*a, **kw):
            seen.append(kw.get("follow_clock"))
            return real(*a, **kw)
        monkeypatch.setattr(realtime, "_midi_listening", listening)
        sock = short_socket_path("midi-lead-flag")
        sd, mido = fake_audio(None, rate_x=12.0), FakeMido()
        stopper = stop_at(sock, lambda: (sd.stream.blocks * 256 / SR) if sd.stream else 0.0, 2.0)
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.lead(prompt="lofi at 90 bpm", seed=0, socket_path=sock, status=False,
                          midi="auto", midi_clock=flag)
        stopper.join(timeout=5.0)
        assert seen == [flag]

    @pytest.mark.parametrize("flag", [False, True])
    def test_jam_taking_the_lead_keeps_the_choice(self, flag, monkeypatch):
        seen = []
        monkeypatch.setattr(realtime, "lead", lambda **kw: seen.append(kw))
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                     lead_in="lofi at 90 bpm", midi="auto", midi_clock=flag)
        assert seen[0]["midi"] == "auto" and seen[0]["midi_clock"] is flag


# ----------------------------------- the real transports, fake device, fake port

class Player:
    """Stands in for the player, pressing keys from inside the fake device's pump.

    Running on the pump keeps a fast-pumped stream honest: a separate thread polling
    the clock lands notes tens of milliseconds late in stream time, which a test
    then blames on plume.  ``script(t)`` runs on every block.
    """

    def __init__(self, sd, mido, *, bpm=110.0, seq=(C, F, G, C), script=None):
        self.sd, self.mido, self.script = sd, mido, script
        self.beat, self.seq = 60.0 / bpm, seq
        self.i = 0
        self.up: tuple[float, tuple] | None = None
        original = sd.Stream

        def Stream(*, blocksize, callback, **kw):  # noqa: N802 - sd's name
            def pumped(indata, outdata, frames, time_info, status):
                callback(indata, outdata, frames, time_info, status)
                self.advance(self.t())
            return original(blocksize=blocksize, callback=pumped, **kw)

        sd.Stream = Stream

    def t(self) -> float:
        s = self.sd.stream
        return 0.0 if s is None else s.blocks * 256 / SR

    def advance(self, t: float) -> None:
        if self.script is not None:
            self.script(t)
        port = self.mido.port
        if port is None or port.closed:
            return
        if self.up is not None and t >= self.up[0]:
            for n in self.up[1]:
                port.send(Msg("note_off", note=n, velocity=0))
            self.up = None
        if t >= 1.0 + self.i * self.beat:
            notes = self.seq[(self.i // 4) % len(self.seq)]
            for n in notes:
                port.send(Msg("note_on", note=n, velocity=90))
            self.up = (t + 0.8 * self.beat, notes)
            self.i += 1


def stop_at(sock, stream_t, until):
    def run():
        wait_until(lambda: stream_t() >= until, timeout=120.0)
        try:
            realtime.send_stop(socket_path=sock)
        except realtime.RealtimeError:
            pass
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def read_rows(rec):
    return [json.loads(x) for x in (rec / "decisions.ndjson").read_text().splitlines()]


def read_midi(rec):
    return [json.loads(x) for x in (rec / "midi.ndjson").read_text().splitlines()]


#: These two run a whole minute of a take at the speed it is really played.  At 6x
#: on a busy machine the analysis falls more than the input ring behind and the band
#: never sounds, a failure that reads as a musical regression and is only CPU.
REAL_TIME = 1.0


@pytest.mark.usefixtures("fake_device_consent")
class TestTransports:
    def test_jam_follows_keys_with_a_silent_mic_and_survives_an_unplug(
            self, tmp_path, plume_registry, monkeypatch):
        """The mic hears nothing at all, so everything the band knows came by MIDI."""
        rec, sock = tmp_path / "take", short_socket_path("midi-jam")
        sd, mido = fake_audio(None, rate_x=REAL_TIME), FakeMido()
        monkeypatch.setattr(midi, "POLL_S", 0.05)

        def script(t):
            mido.names[:] = [] if 34.0 < t < 42.0 else ["Fake Keys"]   # the cable

        player = Player(sd, mido, script=script)
        stopper = stop_at(sock, player.t, 60.0)
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                         record_dir=str(rec), socket_path=sock, midi="auto")
        stopper.join(timeout=5.0)

        assert sd.stream.sounded()
        logged = read_midi(rec)
        kinds = [d["type"] for d in logged]
        assert kinds[0] == "port" and "on" in kinds and "off" in kinds
        assert "lost" in kinds and "back" in kinds
        assert kinds.index("lost") < kinds.index("back")
        after = [d for d in logged[kinds.index("back"):] if d["type"] == "on"]
        assert after                               # notes flow again
        assert all(p.closed for p in mido.ports)
        rows = read_rows(rec)
        played = [r for r in rows if r.get("bpm") and abs(float(r["bpm"]) - 110.0) < 5.0]
        assert played and played[-1]["key"] == "C major"
        manifest = json.loads((rec / "session.json").read_text())
        assert manifest["midi_port"] == "Fake Keys"
        assert (rec / "ending.json").exists()

    def test_lead_tap_start_counts_off_keys_and_hears_the_key_with_a_silent_mic(
            self, tmp_path, plume_registry):
        rec, sock = tmp_path / "take", short_socket_path("midi-lead")
        sd, mido = fake_audio(None, rate_x=REAL_TIME), FakeMido()
        player = Player(sd, mido)
        stopper = stop_at(sock, player.t, 60.0)
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.lead(prompt="lofi at 90 bpm", seed=0, record_dir=str(rec),
                          socket_path=sock, status=False, tap_start=True, midi="auto")
        stopper.join(timeout=5.0)

        assert sd.stream.sounded()
        rows = [r for r in read_rows(rec) if "bpm" in r]
        assert rows and rows[0]["bpm"] == pytest.approx(110.0, rel=0.015)
        assert any(r.get("entry") not in (None, "drums") for r in rows)
        assert rows[-1]["key"] == "C major"
        assert all(p.closed for p in mido.ports)

    def test_a_bad_port_is_refused_before_a_stream_opens(self, tmp_path, plume_registry):
        sd, mido = fake_audio(None), FakeMido(["Keys"])
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            with pytest.raises(realtime.RealtimeError, match="nope"):
                realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                             socket_path=short_socket_path("midi-bad"), midi="nope")
            with pytest.raises(realtime.RealtimeError, match="nope"):
                realtime.lead(prompt="lofi", seed=0, status=False,
                              socket_path=short_socket_path("midi-bad"), midi="nope")
        assert not sd.streams and not mido.ports

    def test_without_the_flag_nothing_touches_the_library(self, tmp_path, plume_registry):
        """Set to None so any import of it would raise."""
        sock = short_socket_path("midi-none")
        sd = fake_audio(None, rate_x=12.0)
        stopper = stop_at(sock, lambda: (sd.stream.blocks * 256 / SR) if sd.stream else 0.0, 3.0)
        with patched_module("sounddevice", sd), patched_module("mido", None):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), socket_path=sock)
        stopper.join(timeout=5.0)
        assert not (tmp_path / "midi.ndjson").exists()

    def test_a_loud_band_cannot_mask_a_key_press(self):
        """Nothing in the live wiring may treat keys as the band coming back."""
        class Timeline:
            playhead = 0
        source, frames, masker = realtime._midi_listening(
            "Fake Keys", Timeline(), None, status=False)
        source.events.extend(on(1.0 + i * 0.5, 60, 90) for i in range(8))
        t = np.arange(SR * 6) / SR
        loud = (0.8 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        hop_rms = realtime.HopRms()
        tracker, harmony, seen = BeatTracker(), HarmonyTracker(), []
        for start in range(0, loud.size, 4096):
            block = loud[start:start + 4096]
            out, flags, _b, _h = realtime.analysis_step(
                extractor=frames, hop_rms=hop_rms, masker=masker, tracker=tracker,
                harmony=harmony, human_block=np.zeros(block.size, dtype=np.float32),
                machine_block=block)
            assert not any(flags)
            seen.extend(f for f in out if f.onset)
        assert len(seen) == 8


# ----------------------------------------------------------------- plume watch

def recorded(tmp_path, midi_docs=None, *, bars=24, mic=0.0):
    rows, n = [], int(2.0 * SR)
    for b in range(bars):
        rows.append({"bar": b, "t_s": b * 2.0, "bpm": 120.0, "key": "C major",
                     "chord": "C", "section": "verse", "entry": "full"})
    noise = mic * np.sin(2 * np.pi * 330.0 * np.arange(n * bars) / SR)
    sf.write(str(tmp_path / "human.wav"), noise.astype("float32"), SR)
    sf.write(str(tmp_path / "machine.wav"), np.zeros(n * bars, dtype="float32"), SR)
    (tmp_path / "decisions.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if midi_docs is not None:
        (tmp_path / "midi.ndjson").write_text(
            '{"t_s": 0.0, "type": "port", "port": "Fake Keys"}\n'
            + "".join(json.dumps(d) + "\n" for d in midi_docs))
    return tmp_path


def played(start, stop, notes=C, *, every=0.5, vel=90):
    out, t = [], float(start)
    while t < stop:
        for n in notes:
            out.append({"t_s": t, "type": "on", "note": n, "velocity": vel, "channel": 0})
            out.append({"t_s": t + 0.4, "type": "off", "note": n, "channel": 0})
        t += every
    return sorted(out, key=lambda d: d["t_s"])


class TestWatch:
    def test_a_mic_take_is_marked_mic(self, tmp_path):
        docs = Listener(recorded(tmp_path, mic=0.1), every=4).poll()
        assert docs and all(d["you_source"] == "mic" for d in docs)

    def test_a_midi_take_reports_the_notes_played_not_what_the_mic_heard(self, tmp_path):
        """The mic is full of a loud 330 Hz (E); the player played F A C."""
        d = recorded(tmp_path, played(0.0, 48.0, F), mic=0.3)
        docs = Listener(d, every=4).poll()
        assert docs and all(x["you_source"] == "midi" for x in docs)
        assert set(docs[0]["you_notes"]) == {"F", "A", "C"}
        assert docs[0]["you_busy"] == pytest.approx(1.0, abs=0.05)
        assert docs[0]["you_db"] == pytest.approx(midi.velocity_db(90), abs=0.5)

    def test_it_knows_the_player_stopped_even_with_a_loud_room(self, tmp_path):
        """The case the docs tell you not to trust on a mic."""
        d = recorded(tmp_path, played(0.0, 16.0) + played(32.0, 48.0), mic=0.3)
        changes = [c for x in Listener(d, every=4).poll() for c in x["changes"]]
        assert "you:stopped" in changes and "you:back" in changes
        assert changes.index("you:stopped") < changes.index("you:back")

    def test_a_long_held_chord_counts_as_playing(self, tmp_path):
        held = [{"t_s": 0.1, "type": "on", "note": n, "velocity": 90, "channel": 0}
                for n in C]
        docs = Listener(recorded(tmp_path, held), every=4).poll()
        assert all(set(x["you_notes"]) == {"C", "E", "G"} for x in docs)
        assert not any("you:stopped" in x["changes"] for x in docs)

    def test_drum_pads_are_busy_but_not_notes(self, tmp_path):
        hits = [{"t_s": i * 0.5, "type": "on", "note": 36, "velocity": 90, "channel": 9}
                for i in range(90)]
        docs = Listener(recorded(tmp_path, hits), every=4).poll()
        assert docs[0]["you_notes"] == [] and docs[0]["you_busy"] > 0.9

    def test_a_pulled_cable_is_named_as_that(self, tmp_path):
        logged = played(0.0, 16.0) + [{"t_s": 17.0, "type": "lost"},
                                      {"t_s": 33.0, "type": "back"}] + played(33.0, 48.0)
        changes = [c for x in Listener(recorded(tmp_path, logged), every=4).poll()
                   for c in x["changes"]]
        assert "midi:lost" in changes and "midi:back" in changes

    def test_notes_before_the_first_row_are_not_counted_in_it(self, tmp_path):
        """Twenty seconds of playing before the first bar, then the same again."""
        logged = played(-20.0, 48.0, every=0.5)
        docs = Listener(recorded(tmp_path, logged), every=4).poll()
        assert docs[0]["you_busy"] == pytest.approx(1.0, abs=0.05)
        assert not any("you:sparser" in x["changes"] for x in docs)

    def test_a_count_in_is_not_counted_in_the_first_line(self, tmp_path):
        count = [{"t_s": -2.0 + i * 0.5, "type": "on", "note": 60, "velocity": 40,
                  "channel": 0} for i in range(4)]
        docs = Listener(recorded(tmp_path, count + played(0.0, 48.0)), every=4).poll()
        assert docs[0]["you_busy"] == pytest.approx(1.0, abs=0.05)
        assert docs[0]["you_db"] == pytest.approx(midi.velocity_db(90), abs=0.5)

    def test_a_chord_struck_before_the_group_is_still_sounding_in_it(self, tmp_path):
        held = [{"t_s": -1.0, "type": "on", "note": n, "velocity": 90, "channel": 0} for n in C]
        held += [{"t_s": 1.0, "type": "off", "note": n, "channel": 0} for n in C]
        doc = Listener(recorded(tmp_path, held), every=4).poll()[0]
        assert doc["you_busy"] == 0.0
        assert set(doc["you_notes"]) == {"C", "E", "G"}
        assert doc["you_db"] == pytest.approx(midi.velocity_db(90), abs=0.5)

    def test_a_half_written_line_is_read_on_the_next_poll(self, tmp_path):
        d = recorded(tmp_path, played(0.0, 8.0), bars=4)
        with (d / "midi.ndjson").open("a") as f:
            f.write('{"t_s": 7.9, "type"')
        assert Listener(d, every=4).poll()


# ---------------------------------------------------------------------- the CLI

class TestCli:
    def test_the_verb_lists_ports(self, capsys):
        with patched_module("mido", FakeMido(["Keys", "Pads"])):
            assert cli.main(["midi", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["ports"] == ["Keys", "Pads"]

    def test_the_verb_without_the_extra_says_what_to_install(self, capsys):
        with patched_module("mido", None):
            assert cli.main(["midi"]) != 0
        assert ".[midi]" in capsys.readouterr().err

    @pytest.mark.parametrize("verb,extra", [("jam", []), ("lead", ["lofi at 90 bpm"])])
    def test_live_verbs_pass_the_port_only_when_asked(self, verb, extra, monkeypatch):
        seen = []
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, verb, lambda **kw: seen.append(kw))
        assert cli.main([verb, *extra, "--midi", "keys"]) == 0
        assert cli.main([verb, *extra]) == 0
        assert seen[0]["midi"] == "keys"
        assert "midi" not in seen[1]

    def test_voice_passes_it_on_to_the_take(self, monkeypatch):
        from plume import voice
        seen = {}
        monkeypatch.setattr(realtime, "check_audio_consent", lambda *_a, **_k: None)
        monkeypatch.setattr(voice, "run", lambda **kw: seen.update(kw))
        monkeypatch.setitem(sys.modules, "sounddevice", fake_audio(None))
        with patched_module("mido", FakeMido(["Studio Keys"])):
            assert cli.main(["voice", "lofi", "--midi", "keys"]) == 0
        assert seen["extra_args"] == ["--midi", "Studio Keys"]
        with patched_module("mido", FakeMido(["Studio Keys"])):
            assert cli.main(["voice", "lofi", "--midi", "keys", "--midi-clock"]) == 0
        assert seen["extra_args"] == ["--midi", "Studio Keys", "--midi-clock"]

    @pytest.mark.parametrize("fake,says", [
        (FakeMido(["Keys"]), "no MIDI port matching nope"),
        (FakeMido(["A", "B"]), "more than one"),
        (None, ".[midi]"),
    ], ids=["no-match", "ambiguous", "no-library"])
    def test_voice_says_what_is_wrong_with_the_port_itself(self, fake, says, monkeypatch, capsys):
        """It used to say so only in the band's log, after loading whisper."""
        from plume import voice
        ran = []
        monkeypatch.setattr(realtime, "check_audio_consent", lambda *_a, **_k: None)
        monkeypatch.setattr(voice, "run", lambda **kw: ran.append(kw))
        want = "auto" if fake is not None and len(fake.names) > 1 else "nope"
        monkeypatch.setitem(sys.modules, "sounddevice", fake_audio(None))
        with patched_module("mido", fake):
            assert cli.main(["voice", "lofi", "--midi", want]) != 0
        assert not ran
        assert says in capsys.readouterr().err

    @pytest.mark.parametrize("verb,extra", [("jam", []), ("lead", ["lofi at 90 bpm"])])
    def test_following_a_clock_is_asked_for(self, verb, extra, monkeypatch):
        seen = []
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, verb, lambda **kw: seen.append(kw))
        assert cli.main([verb, *extra, "--midi", "keys", "--midi-clock"]) == 0
        assert cli.main([verb, *extra, "--midi", "keys"]) == 0
        assert seen[0]["midi_clock"] is True
        assert "midi_clock" not in seen[1]

    @pytest.mark.parametrize("verb,extra", [("jam", []), ("lead", ["lofi"]), ("voice", ["lofi"])])
    def test_a_clock_with_no_port_is_refused(self, verb, extra, monkeypatch, capsys):
        from plume import voice
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, "check_audio_consent", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, verb if verb != "voice" else "jam", lambda **_k: None)
        monkeypatch.setattr(voice, "run", lambda **_k: None)
        assert cli.main([verb, *extra, "--midi-clock"]) != 0
        assert "--midi" in capsys.readouterr().err

    @pytest.mark.parametrize("verb,extra", [("jam", []), ("lead", ["lofi"]),
                                             ("voice", ["lofi"])])
    def test_midi_does_not_get_past_the_audio_consent_gate(self, verb, extra, monkeypatch,
                                                           capsys):
        """Keys in do not change the rule about sound out.  The fake device is what
        gets this as far as the consent gate: with no audio library at all the
        machine check refuses first, and a refusal for that reason proves nothing."""
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
        # Past the gate is where a session would start, so a broken gate fails here
        # rather than running one.
        from plume import voice
        started = []
        for module, name in ((realtime, "jam"), (realtime, "lead"), (voice, "run")):
            monkeypatch.setattr(module, name, lambda **kw: started.append(kw))
        sd, mido = fake_audio(None), FakeMido()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main([verb, *extra, "--midi", "auto"]) != 0
        assert "refusing to play live audio" in capsys.readouterr().err
        assert not started and not sd.streams and not mido.ports


class StuckMido(FakeMido):
    """A driver for a dead device: opening never comes back until it is let go."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.let_go = threading.Event()
        self.opens = 0

    def open_input(self, name, callback=None):
        self.opens += 1
        self.let_go.wait(20.0)
        return super().open_input(name, callback=callback)


class LateMido(StuckMido):
    """Its open is already inside the driver, so it lands whatever the port does next."""

    def open_input(self, name, callback=None):
        self.opens += 1
        self.let_go.wait(20.0)
        self.let_go.clear()
        self.ports.append(FakeMidiPort(name, callback))
        return self.ports[-1]


class TestOpenIsBounded:
    @pytest.fixture(autouse=True)
    def quick(self, monkeypatch):
        monkeypatch.setattr(midi, "OPEN_TIMEOUT_S", 0.05)

    def start(self, fake, told=None):
        with patched_module("mido", fake):
            source = midi.MidiInput("Fake Keys", clock=lambda: 1.0, poll_s=0.0,
                                    on_change=told)
            t0 = time.monotonic()
            source.start()
        return source, time.monotonic() - t0

    def test_a_stuck_port_does_not_hold_up_the_session(self):
        fake = StuckMido()
        told = []
        source, took = self.start(fake, told.append)
        try:
            assert took < 5.0 and source.connected is False and told == ["lost"]
        finally:
            fake.let_go.set()
            source.close()

    def test_it_is_not_opened_again_while_the_first_try_is_stuck(self):
        fake = StuckMido()
        source, _ = self.start(fake)
        try:
            with patched_module("mido", fake):
                source.check()
                source.check()
            assert fake.opens == 1 and source.connected is False
        finally:
            fake.let_go.set()
            source.close()

    def test_a_late_open_counts_and_never_opens_a_second_port(self):
        fake = StuckMido()
        told = []
        source, _ = self.start(fake, told.append)
        fake.let_go.set()
        assert wait_until(lambda: fake.port is not None)
        with patched_module("mido", fake):
            assert wait_until(lambda: source.check() or source.connected)
            source.check()
        assert told == ["lost", "back"] and fake.opens == 1
        fake.port.send(Msg("note_on", note=60, velocity=90))
        assert source.events[-1][1] == "on"
        source.close()
        assert fake.port.closed

    def test_a_port_that_opens_after_the_session_ended_is_closed(self):
        fake = StuckMido()
        source, _ = self.start(fake)
        t0 = time.monotonic()
        source.close()
        assert time.monotonic() - t0 < 5.0
        with patched_module("mido", fake):
            fake.let_go.set()
            assert wait_until(lambda: fake.port is not None and fake.port.closed)

    def test_a_late_open_onto_a_port_now_gone_is_dropped(self):
        """It looks connected and is not: the band would hear silence for good."""
        fake = LateMido()
        told = []
        source, _ = self.start(fake, told.append)
        fake.names = []                     # the cable comes out while it is stuck
        with patched_module("mido", fake):
            source.check()
            fake.let_go.set()
            assert wait_until(lambda: not source._opener.is_alive())
            source.check()
            assert fake.ports[0].closed and source._handle is None
            fake.names = ["Fake Keys"]      # and goes back in
            fake.let_go.set()
            assert wait_until(lambda: source.check() or source.connected)
        assert told == ["lost", "back"] and fake.opens == 2
        assert source._handle is fake.ports[1] and not fake.ports[1].closed
        source.close()

    def test_a_late_open_is_not_trusted_if_the_port_came_and_went_before_it_landed(self):
        """The cable is back before the next look, so that look never sees it gone."""
        fake = LateMido()
        told = []
        source, _ = self.start(fake, told.append)
        fake.names = []
        with patched_module("mido", fake):
            source.check()
            fake.names = ["Fake Keys"]
            fake.let_go.set()
            assert wait_until(lambda: not source._opener.is_alive())
            fake.let_go.set()
            assert wait_until(lambda: source.check() or source.connected)
        assert fake.opens == 2 and fake.ports[0].closed and source._handle is fake.ports[1]
        source.close()

    def test_a_port_that_refuses_is_still_only_a_port_that_is_not_back_yet(self):
        fake = FakeMido(fail_open=True)
        told = []
        source, _ = self.start(fake, told.append)
        assert source.connected is False and told == ["lost"]
        fake.fail_open = False
        with patched_module("mido", fake):
            assert wait_until(lambda: source.check() or source.connected)
        assert told == ["lost", "back"]
        source.close()
