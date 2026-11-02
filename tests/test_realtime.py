"""Tests for plume.realtime (SPEC-V1 section 7) — CI-safe smoke tests.

No audio device in CI, so these cover the parts that are pure: ring buffers,
the output schedule, scheduler math (revision deadline = downbeat - 1 beat),
per-hop RMS, schedule-aware self-masking, bar summarization, and the clean
no-device error paths (sounddevice absent / deviceless / mismatched).  The
musical logic itself lives in the sim-tested modules.
"""
from __future__ import annotations

import ast
import importlib
import math
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plume import realtime  # noqa: E402
from plume.listen.features import HOP, Frame, FeatureExtractor  # noqa: E402
from plume.realtime import (  # noqa: E402
    HopRms,
    RealtimeError,
    RingBuffer,
    SelfMasker,
    TimelineBuffer,
    bar_features_dict,
    revision_deadline,
)


def _realtime_source() -> str:
    """``realtime.py`` as text, for the source-reading guards below."""
    return pathlib.Path(realtime.__file__).read_text(encoding="utf-8")


def _func_ast(*names: str) -> ast.FunctionDef:
    """The AST of a (possibly nested) function, found by NAME on disk.

    Deliberately not ``inspect.getsource``.  That resolves a live code
    object's ``co_firstlineno`` against whatever the file says NOW, so if the
    module is edited while a process has it imported it silently returns a
    DIFFERENT function's body — which is exactly how these guards first
    "failed": ``inspect.getsource(realtime.lead)`` handed back
    ``_resolve_lead_session``.  A guard that reads source has to read it the
    same way a reader would.
    """
    node: ast.AST = ast.parse(_realtime_source())
    for name in names:
        node = next(
            (n for n in ast.walk(node)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == name),
            None)
        assert node is not None, f"realtime.py has no function named {name!r}"
    return node


def _frame(t_s=0.0, rms_db=-20.0, onset=False, strength=3.0):
    return Frame(
        t_s=t_s, rms_db=rms_db, flux=strength if onset else 0.1,
        onset=onset, onset_strength=strength if onset else 0.0,
        chroma=np.zeros(12, dtype=np.float32), centroid_hz=440.0,
    )


# ---------------------------------------------------------------- RingBuffer

class TestRingBuffer:
    def test_roundtrip(self):
        rb = RingBuffer(16)
        x = np.arange(10, dtype=np.float32)
        assert rb.write(x) == 10
        assert rb.available == 10
        out = rb.read()
        np.testing.assert_array_equal(out, x)
        assert rb.available == 0

    def test_wraparound_preserves_order(self):
        rb = RingBuffer(8)
        rb.write(np.arange(6, dtype=np.float32))
        rb.read(4)  # consume 4, leaving 2; next write wraps
        rb.write(np.arange(10, 15, dtype=np.float32))
        out = rb.read()
        np.testing.assert_array_equal(
            out, np.array([4, 5, 10, 11, 12, 13, 14], dtype=np.float32))

    def test_overflow_drops_and_counts(self):
        rb = RingBuffer(8)
        wrote = rb.write(np.ones(12, dtype=np.float32))
        assert wrote == 8
        assert rb.dropped == 4
        assert rb.space == 0
        # the accepted samples are intact
        np.testing.assert_array_equal(rb.read(), np.ones(8, dtype=np.float32))

    def test_read_caps_at_available(self):
        rb = RingBuffer(8)
        rb.write(np.ones(3, dtype=np.float32))
        assert rb.read(100).shape[0] == 3
        assert rb.read(100).shape[0] == 0

    def test_two_channel(self):
        rb = RingBuffer(8, channels=2)
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        assert rb.write(x) == 5
        out = rb.read()
        assert out.shape == (5, 2)
        np.testing.assert_array_equal(out, x)

    def test_mono_accepts_column_vector(self):
        rb = RingBuffer(8)
        rb.write(np.ones((4, 1), dtype=np.float32))
        assert rb.available == 4

    def test_bad_args(self):
        with pytest.raises(ValueError):
            RingBuffer(0)
        with pytest.raises(ValueError):
            RingBuffer(8, channels=0)

    def test_many_interleaved_ops_keep_fifo_order(self):
        rb = RingBuffer(32)
        expect: list[float] = []
        got: list[float] = []
        k = 0
        for step in range(50):
            n = (step * 7) % 5 + 1
            x = np.arange(k, k + n, dtype=np.float32)
            wrote = rb.write(x)
            expect.extend(x[:wrote].tolist())
            k += n
            got.extend(rb.read((step * 3) % 4).tolist())
        got.extend(rb.read().tolist())
        assert got == expect


# ------------------------------------------------------------ TimelineBuffer

class TestTimelineBuffer:
    def test_reads_silence_by_default(self):
        tl = TimelineBuffer(64)
        out = np.ones((16, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out == 0.0)
        assert tl.playhead == 16

    def test_write_then_read_at_position(self):
        tl = TimelineBuffer(64)
        tl.write_at(8, np.ones((4, 2), dtype=np.float32))
        out = np.empty((16, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out[:8] == 0.0)
        assert np.all(out[8:12] == 1.0)
        assert np.all(out[12:] == 0.0)

    def test_write_behind_playhead_is_clamped(self):
        tl = TimelineBuffer(64)
        out = np.empty((10, 2), dtype=np.float32)
        tl.read_into(out)  # playhead = 10
        audio = np.ones((8, 2), dtype=np.float32)
        wrote = tl.write_at(6, audio)   # samples 6..14, only 10..14 land
        assert wrote == 4
        out2 = np.empty((8, 2), dtype=np.float32)
        tl.read_into(out2)
        assert np.all(out2[:4] == 1.0)
        assert np.all(out2[4:] == 0.0)

    def test_consumed_region_is_zeroed_for_reuse(self):
        cap = 32
        tl = TimelineBuffer(cap)
        tl.write_at(0, np.ones((cap, 2), dtype=np.float32))
        out = np.empty((cap, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out == 1.0)
        # the second lap of the ring must be clean silence
        tl.read_into(out)
        assert np.all(out == 0.0)

    def test_add_mixes_overlapping_writes(self):
        tl = TimelineBuffer(64)
        tl.write_at(4, np.ones((8, 2), dtype=np.float32))
        tl.write_at(8, np.ones((8, 2), dtype=np.float32))
        out = np.empty((20, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out[8:12] == 2.0)
        assert np.all(out[4:8] == 1.0)

    def test_revision_clear_then_rewrite(self):
        tl = TimelineBuffer(64)
        tl.write_at(10, np.full((8, 2), 0.5, dtype=np.float32))
        tl.clear_region(10, 8)
        tl.write_at(12, np.full((8, 2), 0.25, dtype=np.float32))
        out = np.empty((24, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out[10:12] == 0.0)      # old placement gone
        assert np.all(out[12:20] == 0.25)     # revision in place

    def test_mono_write_upmixes(self):
        tl = TimelineBuffer(64)
        tl.write_at(0, np.ones(4, dtype=np.float32))
        out = np.empty((4, 2), dtype=np.float32)
        tl.read_into(out)
        assert np.all(out == 1.0)

    def test_write_clips_to_capacity_window(self):
        tl = TimelineBuffer(16)
        wrote = tl.write_at(8, np.ones((16, 2), dtype=np.float32))
        assert wrote == 8   # only up to playhead + capacity
        assert tl.end == 16

    def test_end_tracks_scheduled_material(self):
        tl = TimelineBuffer(64)
        assert tl.end == 0
        tl.write_at(10, np.ones((5, 2), dtype=np.float32))
        assert tl.end == 15


# ------------------------------------------------------------------ HopRms

class TestHopRms:
    def test_known_levels(self):
        hr = HopRms(hop=4)
        vals = hr.process(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32))
        assert len(vals) == 2
        assert vals[0] == pytest.approx(0.0, abs=1e-6)     # full-scale = 0 dBFS
        assert vals[1] == pytest.approx(-90.0)             # silence floors

    def test_blocking_invariance(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(HOP * 5).astype(np.float32) * 0.1
        a = HopRms()
        one = a.process(x)
        b = HopRms()
        two: list[float] = []
        for i in range(0, x.size, 777):
            two.extend(b.process(x[i:i + 777]))
        assert one == pytest.approx(two)

    def test_lockstep_with_feature_extractor(self):
        # One RMS value per completed HOP — the same cadence FeatureExtractor
        # emits frames at, so machine RMS aligns with frames index-for-index.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(HOP * 7 + 123).astype(np.float32) * 0.1
        fx = FeatureExtractor()
        hr = HopRms()
        n_frames = 0
        n_rms = 0
        for i in range(0, x.size, 1000):
            block = x[i:i + 1000]
            n_frames += len(fx.process(block))
            n_rms += len(hr.process(block))
        assert n_frames == n_rms


# ---------------------------------------------------------------- SelfMasker

class TestSelfMasker:
    def test_loud_machine_suppresses_onsets(self):
        m = SelfMasker()
        m.human_rms_db = -20.0
        frames = [_frame(onset=True), _frame(onset=True)]
        out, flags = m.mask(frames, [-10.0, -12.0])   # machine well above -26
        assert flags == [True, True]
        assert all(not f.onset for f in out)
        assert all(f.onset_strength == 0.0 for f in out)
        assert m.masked_frames == 2

    def test_quiet_machine_passes_through(self):
        m = SelfMasker()
        m.human_rms_db = -20.0
        frames = [_frame(onset=True, t_s=0.1)]
        out, flags = m.mask(frames, [-80.0])
        assert flags == [False]
        assert out[0].onset is True
        assert out[0].onset_strength > 0.0
        assert m.masked_frames == 0

    def test_margin_is_6db(self):
        # v1.6: the LEVEL rule in isolation.  With the transient rule left on
        # (the default) the machine's first audible frame is a rise out of
        # silence — which it is, and which is masked on purpose — so the
        # margin cannot be read off a two-frame stream unless it is the only
        # rule running.  See TestSelfMaskerTransients for the other half.
        m = SelfMasker(transients=False)
        m.human_rms_db = -20.0
        _, flags = m.mask([_frame(), _frame()], [-25.9, -26.1])
        assert flags == [True, False]

    def test_silence_never_masks(self):
        # Before anyone plays, human estimate is at the floor; digital silence
        # on the machine bus must not mask the room.
        m = SelfMasker()
        _, flags = m.mask([_frame(onset=True)], [-90.0])
        assert flags == [False]

    def test_human_estimate_tracks_unmasked_frames_only(self):
        m = SelfMasker(human_alpha=0.5)
        m.human_rms_db = -40.0
        m.mask([_frame(rms_db=-20.0)], [-90.0])       # unmasked: EMA moves
        assert m.human_rms_db == pytest.approx(-30.0)
        before = m.human_rms_db
        m.mask([_frame(rms_db=0.0)], [0.0])           # masked: EMA frozen
        assert m.human_rms_db == before

    def test_missing_machine_values_treated_as_silence(self):
        m = SelfMasker()
        m.human_rms_db = -20.0
        out, flags = m.mask([_frame(onset=True), _frame(onset=True)], [-10.0])
        assert flags == [True, False]
        assert out[1].onset is True


# --------------------------------------------------------- scheduler math

class TestSchedulerMath:
    def test_revision_deadline_is_one_beat_before_downbeat(self):
        # 120 bpm -> beat = 0.5s; downbeat at 10.0 -> revisable until 9.5
        assert revision_deadline(10.0, 0.5) == pytest.approx(9.5)

    def test_revision_deadline_inf_before_lock(self):
        assert revision_deadline(math.inf, 0.5) == math.inf
        assert revision_deadline(math.inf, math.inf) == math.inf


class TestBarFeatures:
    def test_summary_math(self):
        events = [(0.0, -20.0, True), (0.1, -20.0, False),
                  (0.2, -20.0, True), (0.3, -20.0, False)]
        d = bar_features_dict(3, events, silence_rms_db=-45.0,
                              register_low_midi=52.0, register_high_midi=76.0)
        assert d["bar"] == 3
        assert d["rms_db"] == pytest.approx(-20.0, abs=1e-6)
        assert d["onset_density"] == pytest.approx(2 / 4)
        assert d["silent"] is False
        assert d["register_low_midi"] == 52.0
        assert d["register_high_midi"] == 76.0

    def test_rms_combines_in_power_domain(self):
        events = [(0.0, 0.0, False), (0.1, -90.0, False)]
        d = bar_features_dict(0, events, silence_rms_db=-45.0)
        # mean power of full-scale and silence is ~half power = -3.01 dB
        assert d["rms_db"] == pytest.approx(-3.01, abs=0.05)

    def test_empty_bar_is_silent(self):
        d = bar_features_dict(5, [], silence_rms_db=-45.0)
        assert d["silent"] is True
        assert d["rms_db"] == -90.0
        assert d["onset_density"] == 0.0

    def test_quiet_bar_below_threshold_is_silent(self):
        events = [(0.0, -60.0, False)]
        d = bar_features_dict(1, events, silence_rms_db=-45.0)
        assert d["silent"] is True


# ----------------------------------------------------------- device errors

def _fake_sd(devices, query_error=None):
    mod = types.SimpleNamespace()
    if query_error is not None:
        def q():
            raise query_error
    else:
        def q():
            return devices
    mod.query_devices = q
    return mod


class TestDeviceErrors:
    def test_module_imports_without_sounddevice(self, monkeypatch):
        """Importing plume.realtime must never touch the audio stack.

        The re-import is restored afterwards.  Leaving the fresh module in
        ``sys.modules`` leaves a SECOND ``RealtimeError`` class behind, and
        every later ``pytest.raises(realtime.RealtimeError)`` in another file
        then fails depending on collection order — a test-isolation landmine
        that has nothing to do with the behavior under test.
        """
        monkeypatch.setitem(sys.modules, "plume.realtime", realtime)
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        del sys.modules["plume.realtime"]
        mod = importlib.import_module("plume.realtime")
        assert hasattr(mod, "jam")

    def test_missing_sounddevice_is_clean(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        msg = str(ei.value)
        assert "sounddevice" in msg
        assert "plume sim" in msg          # points at the offline twin

    def test_jam_errors_before_touching_siblings(self, monkeypatch):
        # With sounddevice absent, jam() must fail cleanly without importing
        # mind/duet (which may not exist while builders run concurrently).
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        with pytest.raises(RealtimeError):
            realtime.jam(style="lofi", seed=0, params=None)

    def test_no_devices_at_all(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd([]))
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        assert "no audio" in str(ei.value)

    def test_output_only_reports_missing_input(self, monkeypatch):
        devs = [{"name": "Speakers", "max_input_channels": 0,
                 "max_output_channels": 2}]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        assert "input" in str(ei.value)

    def test_input_only_reports_missing_output(self, monkeypatch):
        devs = [{"name": "Mic", "max_input_channels": 1,
                 "max_output_channels": 0}]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        assert "output" in str(ei.value)

    def test_named_device_not_found_lists_devices(self, monkeypatch):
        devs = [{"name": "Built-in", "max_input_channels": 1,
                 "max_output_channels": 2}]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device("UMC404")
        msg = str(ei.value)
        assert "UMC404" in msg and "Built-in" in msg

    def test_named_device_matches_substring(self, monkeypatch):
        devs = [
            {"name": "Built-in Microphone", "max_input_channels": 1,
             "max_output_channels": 0},
            {"name": "Built-in Output", "max_input_channels": 0,
             "max_output_channels": 2},
        ]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        # `check_device` also carries the audio-consent gate, and pytest is not
        # a tty.  This test is about DEVICE resolution, so consent is granted
        # explicitly rather than the gate being weakened (see
        # tests/test_audio_consent.py, which owns that contract).
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
        realtime.check_device("built-in")   # must not raise

    def test_query_failure_is_clean(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "sounddevice",
            _fake_sd(None, query_error=OSError("PortAudio not initialized")))
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        assert "device query failed" in str(ei.value)

    def test_healthy_duplex_passes(self, monkeypatch):
        devs = [{"name": "Duplex", "max_input_channels": 2,
                 "max_output_channels": 2}]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
        realtime.check_device()   # must not raise

    def test_consent_is_still_checked_after_the_device_resolves(self, monkeypatch):
        """The v2.1 reorder must not have DROPPED the consent gate.

        A healthy duplex device with consent refused still refuses — the device
        check running first only changes WHICH error a box that cannot play
        hears, never whether a box that can play is allowed to.
        """
        devs = [{"name": "Duplex", "max_input_channels": 2,
                 "max_output_channels": 2}]
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "0")
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        assert "refusing to play live audio" in str(ei.value)

    def test_a_box_that_cannot_play_is_told_the_truth_first(self, monkeypatch):
        """The dead-end fix, asserted.

        With NO device and NO consent, the user must hear about the device.
        Telling them to set PLUME_ALLOW_AUDIO=1 first sends them down a path
        that cannot work: setting it only reveals the real error one line
        later.
        """
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd([]))
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        with pytest.raises(RealtimeError) as ei:
            realtime.check_device()
        msg = str(ei.value)
        assert "no audio" in msg
        assert realtime.AUDIO_CONSENT_ENV not in msg


# =====================================================================
# v1.3 — the defects the first session with a human and a guitar exposed
# in this file's half of the system.  Everything below is unit level and
# headless; the live-path versions live in test_session_lifecycle.py,
# driven by the fake audio device in tests/conftest.py.
# =====================================================================

import io          # noqa: E402
import json        # noqa: E402
import os          # noqa: E402
import threading   # noqa: E402
import time        # noqa: E402

import soundfile as sf   # noqa: E402

from plume.realtime import (  # noqa: E402
    HardExit,
    SessionRecorder,
    StatusLine,
    _abort_stream,
    _drain_timeline,
    session_manifest,
)


def _wait_true(predicate, seconds=1.0):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class _Pipe(io.StringIO):
    """A stream that is honestly not a terminal (which is what a pipe is)."""

    def isatty(self):
        return False


class _Tty(io.StringIO):
    def isatty(self):
        return True


# ------------------------------------------------- P2-G: the status readout

class TestStatusLine:
    r"""A status display a pipe cannot read is not a status display.

    MEASURED: a six-minute session printed status continuously with ``\r``
    and ``end=""``.  Piped to a file that is ONE line, unterminated, growing
    without bound — ``tail -f`` shows nothing, ``readline()`` blocks forever,
    and every log collector and agent subprocess sees a session that never
    said a word.  The pretty in-place display is right for a terminal and
    only for a terminal.
    """

    def test_a_terminal_still_gets_the_pretty_in_place_line(self):
        out = _Tty()
        s = StatusLine(out)
        assert s.tty is True
        s.update("bar 1")
        s.update("bar 2")
        assert out.getvalue() == "\rbar 1\rbar 2"
        assert "\n" not in out.getvalue()

    def test_a_pipe_gets_one_newline_terminated_line_per_update(self):
        out = _Pipe()
        s = StatusLine(out)
        assert s.tty is False
        s.update("[listening]  80.0 bpm")
        s.update("[bar 4]      80.1 bpm")
        text = out.getvalue()
        assert text.endswith("\n")
        assert "\r" not in text, "carriage returns leaked into a pipe"
        assert text.splitlines() == ["[listening]  80.0 bpm",
                                     "[bar 4]      80.1 bpm"]

    def test_a_pipe_is_readable_line_by_line_while_it_is_written(self):
        """The actual requirement: an observer can consume it incrementally."""
        out = _Pipe()
        s = StatusLine(out)
        seen = []
        for i in range(5):
            s.update(f"bar {i}")
            seen.append(len(out.getvalue().splitlines()))
        assert seen == [1, 2, 3, 4, 5], "lines did not appear one at a time"

    def test_close_terminates_an_in_place_line_only_on_a_terminal(self):
        tty, pipe = _Tty(), _Pipe()
        a, b = StatusLine(tty), StatusLine(pipe)
        a.update("x")
        b.update("x")
        a.close()
        b.close()
        assert tty.getvalue() == "\rx\n"
        assert pipe.getvalue() == "x\n", "a pipe got a spurious second newline"

    def test_close_before_any_update_writes_nothing(self):
        out = _Tty()
        StatusLine(out).close()
        assert out.getvalue() == ""

    def test_a_broken_pipe_disables_the_readout_and_never_raises(self):
        """The reader went away.  The band plays on."""

        class _Broken(io.StringIO):
            def isatty(self):
                return False

            def write(self, _s):
                raise OSError("broken pipe")

        s = StatusLine(_Broken())
        s.update("x")            # must not raise
        assert s.enabled is False
        s.update("y")
        assert s.updates == 0

    def test_disabled_writes_nothing(self):
        out = _Pipe()
        s = StatusLine(out, enabled=False)
        s.update("x")
        s.close()
        assert out.getvalue() == ""

    def test_a_stream_with_no_isatty_is_treated_as_a_pipe(self):
        class _Bare:
            def __init__(self):
                self.text = ""

            def write(self, s):
                self.text += s

            def flush(self):
                pass

        bare = _Bare()
        s = StatusLine(bare)
        assert s.tty is False
        s.update("x")
        assert bare.text == "x\n"


# ------------------------------------------------ P0-A: records, as they go

def _read_ndjson(path):
    """Parse a decisions log the way a live observer has to: a line at a
    time, tolerating a last line the writer has not finished."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.endswith("\n"):
                break
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


class TestSessionRecorder:
    """``--record DIR`` wrote NOTHING for six minutes and lost the whole take
    to the ``kill -9`` that was the only way to end the session.  A take is
    now written continuously or not at all."""

    def test_the_manifest_exists_before_the_first_bar(self, tmp_path):
        d = tmp_path / "take"
        r = SessionRecorder(d)
        r.open(session_manifest(kind="jam", style="lofi", role="follow", seed=3))
        try:
            doc = json.loads((d / "session.json").read_text())
        finally:
            r.finish()
        assert doc["pid"] == os.getpid()
        assert doc["kind"] == "jam" and doc["style"] == "lofi"
        assert doc["streaming"] is True
        assert doc["stop_command"] == f"plume stop --pid {os.getpid()}"
        assert doc["files"]["decisions"] == "decisions.ndjson"

    def test_every_decision_is_readable_before_the_session_ends(self, tmp_path):
        d = tmp_path / "take"
        r = SessionRecorder(d).open({"kind": "jam"})
        try:
            for bar in range(3):
                r.decision({"bar": bar, "state": "playing"})
                # read it back through a SECOND handle, as `tail -f` would
                assert len(_read_ndjson(d / "decisions.ndjson")) == bar + 1
        finally:
            r.finish()
        assert [x["bar"] for x in _read_ndjson(d / "decisions.ndjson")] == [0, 1, 2]

    def test_a_kill_mid_session_leaves_valid_parseable_lines(self, tmp_path):
        """A ``kill -9`` costs at most the line being written, never the take."""
        d = tmp_path / "take"
        r = SessionRecorder(d).open({"kind": "jam"})
        for bar in range(20):
            r.decision({"bar": bar, "reason": "held"})
        # no finish(), no close(): this is where the process was killed
        rows = _read_ndjson(d / "decisions.ndjson")
        assert len(rows) == 20
        assert all(isinstance(x["bar"], int) for x in rows)
        r.close()

    def test_the_wavs_grow_while_the_session_is_running(self, tmp_path):
        d = tmp_path / "take"
        r = SessionRecorder(d, flush_every_s=0.0).open({"kind": "jam"})
        sizes = []
        try:
            for _ in range(3):
                x = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
                r.audio(x, 0.5 * x)
                sizes.append(sf.info(str(d / "human.wav")).frames)
        finally:
            r.finish()
        assert sizes == [4800, 9600, 14400], sizes
        assert sf.info(str(d / "machine.wav")).frames == 14400

    def test_a_killed_wav_is_still_valid_and_readable(self, tmp_path):
        """libsndfile updates the header on flush, so a partial file plays."""
        d = tmp_path / "take"
        r = SessionRecorder(d, flush_every_s=0.0).open({"kind": "jam"})
        x = np.full(2400, 0.25, dtype=np.float32)
        r.audio(x, -x)
        # killed here — nothing was closed
        y, sr = sf.read(str(d / "human.wav"), dtype="float32")
        assert sr == realtime.SR
        assert y.shape[0] == 2400
        assert float(np.abs(y - 0.25).max()) < 1e-3
        r.close()

    def test_memory_does_not_grow_with_the_session(self, tmp_path):
        """A forty-minute take must not be forty minutes of float32 in RAM."""
        d = tmp_path / "take"
        r = SessionRecorder(d).open({"kind": "jam"})
        try:
            for _ in range(50):
                r.audio(np.zeros(4800, dtype=np.float32),
                        np.zeros(4800, dtype=np.float32))
            held = [v for v in vars(r).values()
                    if isinstance(v, list) and v and isinstance(v[0], np.ndarray)]
        finally:
            r.finish()
        assert held == [], "the recorder is accumulating audio in memory again"
        assert r.frames_written == 50 * 4800

    def test_ending_json_is_written_last_and_says_why(self, tmp_path):
        d = tmp_path / "take"
        r = SessionRecorder(d).open({"kind": "jam"})
        assert not (d / "ending.json").exists()
        r.finish({"reason": realtime.END_STOPPED, "capped": False, "bars": 4})
        doc = json.loads((d / "ending.json").read_text())
        assert doc["reason"] == realtime.END_STOPPED and doc["bars"] == 4

    def test_finish_is_idempotent(self, tmp_path):
        r = SessionRecorder(tmp_path / "take").open({"kind": "jam"})
        r.finish({"reason": "a"})
        r.finish({"reason": "b"})       # must not raise, must not rewrite
        doc = json.loads((tmp_path / "take" / "ending.json").read_text())
        assert doc["reason"] == "a"

    def test_writing_after_close_is_a_no_op_not_a_crash(self, tmp_path):
        r = SessionRecorder(tmp_path / "take").open({"kind": "jam"})
        r.finish()
        r.decision({"bar": 1})          # a planner thread that has not noticed
        r.audio(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
        assert r.decisions_written == 0

    def test_a_recorder_that_cannot_write_never_takes_the_band_down(
            self, tmp_path, monkeypatch):
        """Recording is bookkeeping.  Bookkeeping must not stop the music."""
        r = SessionRecorder(tmp_path / "take")
        real_open = open

        def _no(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", _no)
        r.open({"kind": "jam"})         # must not raise
        monkeypatch.setattr("builtins.open", real_open)
        r.decision({"bar": 0})          # must not raise
        r.finish({"reason": "x"})
        assert r.errors, "a failure was swallowed without being counted"

    def test_it_is_safe_from_two_threads_at_once(self, tmp_path):
        """The planner writes decisions while analysis writes audio."""
        d = tmp_path / "take"
        r = SessionRecorder(d, flush_every_s=0.0).open({"kind": "jam"})
        stop = threading.Event()

        def audio():
            while not stop.is_set():
                r.audio(np.zeros(256, dtype=np.float32),
                        np.zeros(256, dtype=np.float32))

        t = threading.Thread(target=audio, daemon=True)
        t.start()
        try:
            for bar in range(200):
                r.decision({"bar": bar})
        finally:
            stop.set()
            t.join(timeout=5.0)
            r.finish()
        rows = _read_ndjson(d / "decisions.ndjson")
        assert [x["bar"] for x in rows] == list(range(200))
        assert not r.errors, r.errors

    def test_emergency_close_never_blocks_on_a_held_lock(self, tmp_path):
        """The watchdog's cleanup must not wait on a thread that is stuck —
        waiting on the wedge is not a way out of the wedge."""
        r = SessionRecorder(tmp_path / "take").open({"kind": "jam"})
        held = threading.Event()
        release = threading.Event()

        def holder():
            with r._lock:
                held.set()
                release.wait(5.0)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        assert held.wait(2.0)
        t0 = time.monotonic()
        got = r.emergency_close({"reason": "x"}, timeout=0.2)
        elapsed = time.monotonic() - t0
        release.set()
        t.join(timeout=5.0)
        assert got is False
        assert elapsed < 1.0, f"emergency close blocked for {elapsed:.1f}s"
        r.close()


class TestSessionManifest:
    def test_it_is_json_serializable_including_params_dataclasses(self):
        from plume.mind import MindParams
        from plume.roles import RoleParams

        doc = session_manifest(kind="lead", style="trap", role="lead", seed=1,
                               params=MindParams(), role_params=RoleParams())
        text = json.dumps(doc)          # must not raise
        assert "entry_conf" in text
        assert doc["params"]["entry_conf"] == MindParams().entry_conf

    def test_it_names_the_files_an_observer_should_tail(self):
        doc = session_manifest(kind="jam", style="lofi", role="follow", seed=0)
        assert set(doc["files"]) == {"decisions", "human", "machine", "ending"}


# ------------------------------------------------- P0-B: the ending's clock

class TestHardExit:
    """The watchdog that makes ``plume stop`` true.

    It exists because the two obvious mechanisms failed at once: a Python
    signal handler cannot run while the main thread is inside a blocking C
    call, and ``Pa_StopStream`` is documented to be exactly such a call.  So
    the last resort has to be an ordinary thread calling ``os._exit``.
    """

    def test_it_ends_the_process_when_the_ending_overruns(self):
        exits, out = [], io.StringIO()
        h = HardExit(exit_fn=exits.append, stream=out)
        h.arm("stopped", seconds=0.05)
        assert _wait_true(lambda: h.fired, 3.0), "the watchdog never fired"
        assert exits == [realtime.EXIT_WEDGED]
        assert "did not finish" in out.getvalue()

    def test_disarming_before_the_deadline_lets_the_process_live(self):
        exits = []
        h = HardExit(exit_fn=exits.append, stream=io.StringIO())
        h.arm("stopped", seconds=0.4)
        h.disarm()
        time.sleep(0.6)
        assert exits == [] and h.fired is False

    def test_it_is_inert_until_it_is_armed(self):
        exits = []
        h = HardExit(exit_fn=exits.append, stream=io.StringIO())
        assert h.armed is False and h.remaining_s is None
        time.sleep(0.2)
        assert exits == []

    def test_a_second_signal_shortens_the_fuse_but_nothing_extends_it(self):
        """The impatient player must be able to pull the deadline in."""
        exits = []
        h = HardExit(exit_fn=exits.append, stream=io.StringIO())
        h.arm("ending", seconds=30.0)
        first = h.remaining_s
        h.arm("ending", seconds=60.0)         # ignored
        assert h.remaining_s <= first
        h.arm("second Ctrl-C", seconds=0.05)  # honoured
        assert _wait_true(lambda: h.fired, 3.0)
        assert exits == [realtime.EXIT_WEDGED]
        assert h.reason == "second Ctrl-C"

    def test_the_take_is_closed_before_the_process_goes(self, tmp_path):
        """``os._exit`` runs no atexit and flushes no buffers, so the cleanup
        has to happen first — and it has to be the recorder's, not a
        ``finally`` on a main thread that is not running."""
        d = tmp_path / "take"
        r = SessionRecorder(d).open({"kind": "jam"})
        r.decision({"bar": 0})
        order, exits = [], []

        def cleanup(_reason):
            order.append("cleanup")
            r.finish({"reason": realtime.END_WEDGED, "bars": 1})

        h = HardExit(exit_fn=lambda c: (order.append("exit"), exits.append(c)),
                     on_exit=cleanup, stream=io.StringIO())
        h.arm("wedged", seconds=0.05)
        assert _wait_true(lambda: h.fired, 3.0)
        assert order == ["cleanup", "exit"]
        doc = json.loads((d / "ending.json").read_text())
        assert doc["reason"] == realtime.END_WEDGED

    def test_a_failing_cleanup_still_ends_the_process(self):
        """Nothing in the cleanup may buy the wedge another minute."""
        exits = []

        def boom(_reason):
            raise RuntimeError("cleanup exploded")

        h = HardExit(exit_fn=exits.append, on_exit=boom, stream=io.StringIO())
        h.arm("wedged", seconds=0.05)
        assert _wait_true(lambda: h.fired, 3.0)
        assert exits == [realtime.EXIT_WEDGED]

    def test_the_exit_code_is_distinguishable_from_a_clean_end(self):
        assert realtime.EXIT_WEDGED not in (0, 1, 130, 143)

    def test_the_budget_contains_every_bounded_step_of_the_ending(self):
        """A deadline shorter than the ending it bounds destroys good takes —
        the mistake a previous round already made with SIGTERM."""
        parts = (realtime.PLANNER_JOIN_S + realtime.TAIL_DRAIN_MAX_S
                 + realtime.ANALYSIS_JOIN_S + realtime.CUE_CLOSE_S)
        assert realtime.ENDING_HARD_DEADLINE_S >= parts
        assert realtime.ENDING_HARD_DEADLINE_S - parts >= 1.0
        # ...and `plume stop`'s budget still outlasts the whole ending.
        assert realtime.DEFAULT_STOP_TIMEOUT > realtime.ENDING_HARD_DEADLINE_S


class TestAbortStream:
    """``Pa_StopStream`` waits for pending buffers; ``Pa_AbortStream`` does not."""

    def test_abort_is_used_when_the_stream_has_one(self):
        class _S:
            def __init__(self):
                self.aborted = False

            def abort(self):
                self.aborted = True

        s = _S()
        assert _abort_stream(s) is True and s.aborted is True

    def test_a_stream_without_abort_is_left_alone(self):
        assert _abort_stream(object()) is False

    def test_an_abort_that_raises_is_not_fatal(self):
        class _S:
            def abort(self):
                raise RuntimeError("PortAudio said no")

        assert _abort_stream(_S()) is False


class TestDrainTimeline:
    def test_it_returns_as_soon_as_the_schedule_has_played(self):
        tl = TimelineBuffer(realtime.SR, channels=2)
        tl.write_at(0, np.ones((100, 2), dtype=np.float32))
        tl.read_into(np.zeros((100, 2), dtype=np.float32))
        t0 = time.monotonic()
        assert _drain_timeline(tl, max_s=5.0) is True
        assert time.monotonic() - t0 < 0.5

    def test_a_playhead_that_stops_advancing_does_not_hang_the_ending(self):
        """The device died mid-ending.  Waiting buys silence, so it is capped."""
        tl = TimelineBuffer(realtime.SR, channels=2)
        tl.write_at(0, np.ones((4800, 2), dtype=np.float32))
        t0 = time.monotonic()
        assert _drain_timeline(tl, max_s=0.2) is False
        assert time.monotonic() - t0 < 2.0

    def test_the_budget_is_derived_from_what_is_left_to_play(self):
        """A fixed cap either clips a slow bar or waits out a dead device.

        Half a second of schedule left must not be given the same patience as
        a six-second bar at 40 bpm — and neither may exceed the ceiling the
        total ending budget is built from.
        """
        short = TimelineBuffer(realtime.SR * 30, channels=2)
        short.write_at(0, np.ones((realtime.SR // 2, 2), dtype=np.float32))
        t0 = time.monotonic()
        assert _drain_timeline(short) is False       # playhead never moves
        short_wait = time.monotonic() - t0

        slow = TimelineBuffer(realtime.SR * 30, channels=2)
        slow.write_at(0, np.ones((realtime.SR * 6, 2), dtype=np.float32))
        t0 = time.monotonic()
        assert _drain_timeline(slow) is False
        slow_wait = time.monotonic() - t0

        assert slow_wait > short_wait + 3.0, (
            f"a 6 s bar got no more patience than a 0.5 s one "
            f"({slow_wait:.1f}s vs {short_wait:.1f}s)")
        # ...and a slow bar really is given more time than it needs to play.
        assert slow_wait > 6.0
        assert slow_wait <= realtime.TAIL_DRAIN_MAX_S + 1.0


# =====================================================================
# v1.6 — self-masking that is actually in the signal path.
#
# The measured failure this section exists for.  Juniper played an acoustic
# guitar in CGCFCE (open tuning: heavy sustain, weak transients) into the
# laptop mic while plume came out of the laptop speakers.  plume's tempo
# estimate walked 167.9 / 163.0 / 169.2 / 165.4 / 80.6 / 80.3 / 121.0 bpm at
# confidence 0.41-0.62, and it "only plays like two random measures here and
# there".  That is precisely the condition SelfMasker exists for, and it was
# inoperative in four separate ways at once — each of which has a test below,
# and none of which a test of SelfMasker on its own could see:
#
#   1. the flag never reached the tracker (it cleared `Frame.onset`, which the
#      tracker has not read since v1.5, and both call sites dropped `flags`);
#   2. the machine BUS was compared against a MIC estimate — two domains with
#      a speaker, a room and a microphone in between;
#   3. the mask was aligned to when the sound was SENT, not when it was HEARD;
#   4. the flag gated only phase correction, while the tempo estimate's
#      autocorrelation window took every masked frame's envelope anyway.
# =====================================================================

def _aframe(t_s=0.0, rms_db=-20.0, onset=False, strength=3.0, attack=4.0):
    """A frame with the v1.5 ``attack`` field the real extractor emits."""
    return Frame(
        t_s=t_s, rms_db=rms_db, flux=strength if onset else 0.1,
        onset=onset, onset_strength=strength if onset else 0.0,
        chroma=np.zeros(12, dtype=np.float32), centroid_hz=440.0,
        attack=attack, attack_breadth=0.5,
    )


class TestSelfMaskerReachesTheTracker:
    """Defect 1 and defect 4: the flag was computed and then thrown away, and
    even delivered it would only have reached half of the tracker."""

    def test_a_masked_frame_carries_the_flag_the_tracker_reads(self):
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        out, flags = m.mask([_aframe(onset=True)], [-10.0])
        assert flags == [True]
        assert out[0].self_masked is True, (
            "BeatTracker picks its own onsets off Frame.attack since v1.5, so "
            "clearing Frame.onset suppresses nothing on a real stream")

    def test_an_unmasked_frame_is_not_flagged(self):
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        out, _ = m.mask([_aframe(onset=True)], [-90.0])
        assert out[0].self_masked is False

    def test_the_envelope_of_a_masked_frame_is_blanked(self):
        # Defect 4.  BeatTracker._step appends every frame's envelope to the
        # autocorrelation window BEFORE it consults the mask, so a mask that
        # only gates phase leaves the machine's own pulse driving the period.
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        out, _ = m.mask([_aframe(onset=True, attack=9.0)], [-10.0])
        assert out[0].attack == 0.0
        assert out[0].flux == 0.0

    def test_an_unmasked_frames_envelope_is_untouched(self):
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        out, _ = m.mask([_aframe(attack=9.0)], [-90.0])
        assert out[0].attack == 9.0

    def test_the_analysis_step_hands_the_flags_to_the_tracker(self):
        """The wiring itself, which is where the defect actually lived."""
        seen = {}

        class SpyTracker:
            def process(self, frames, masked=None):
                seen["masked"] = masked
                seen["self_masked"] = [f.self_masked for f in frames]
                return "beat"

        class NullHarmony:
            def process(self, frames):      # noqa: ARG002
                return "harm"

        class OneFrame:
            def process(self, block):       # noqa: ARG002
                return [_aframe(onset=True)]

        class LoudBus:
            def process(self, block):       # noqa: ARG002
                return [-10.0]

        masker = realtime.SelfMasker()
        masker.human_rms_db = -20.0
        frames, flags, beat, harm = realtime.analysis_step(
            extractor=OneFrame(), hop_rms=LoudBus(), masker=masker,
            tracker=SpyTracker(), harmony=NullHarmony(),
            human_block=np.zeros(8, dtype=np.float32),
            machine_block=np.zeros(8, dtype=np.float32))
        assert flags == [True]
        assert seen["masked"] == [True], "flags were dropped on the floor again"
        assert seen["self_masked"] == [True]
        assert (beat, harm) == ("beat", "harm")
        assert frames[0].self_masked is True


class TestSelfMaskerDomains:
    """Defect 2: bus dBFS is not mic dBFS, and the gap is the room."""

    def test_coupling_starts_at_zero_so_the_v1_rule_is_the_starting_point(self):
        m = realtime.SelfMasker()
        assert m.coupling_db == pytest.approx(realtime.SELF_MASK_COUPLING_INIT_DB)

    def test_coupling_converges_down_to_the_real_path_gain(self):
        # mic = human + machine*g.  While the human is quiet, (mic - bus) IS g.
        m = realtime.SelfMasker(transients=False)
        for i in range(400):
            # machine at -20 dBFS on the bus, arriving at -45 in the mic
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-45.0)], [-20.0])
        assert m.coupling_db == pytest.approx(-25.0, abs=1.0)

    def test_a_quiet_bleed_no_longer_masks_a_loud_human(self):
        # The whole point: a bus hotter than the mic used to mask every frame
        # regardless of whether anything was audible in the room — which also
        # hollowed out the human's bar summary, because shared["events"] reads
        # the masked frames.
        m = realtime.SelfMasker(transients=False)
        for i in range(400):        # learn the room: 30 dB of attenuation
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-50.0)], [-20.0])
        m.human_rms_db = -20.0      # now the human plays, loudly
        _, flags = m.mask([_aframe(rms_db=-20.0)], [-20.0])
        assert flags == [False]

    def test_coupling_creeps_back_up_when_the_speakers_are_turned_up(self):
        m = realtime.SelfMasker(transients=False)
        m.coupling_db = -40.0
        for i in range(400):        # 400 frames ~ 4.3 s, creep is 2 dB/s
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-10.0)], [-20.0])
        assert m.coupling_db > -35.0

    def test_coupling_is_clamped_at_the_bottom(self):
        m = realtime.SelfMasker(transients=False)
        for i in range(50):
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-90.0)], [-10.0])
        assert m.coupling_db >= realtime.SELF_MASK_COUPLING_MIN_DB


class TestSelfMaskerAlignment:
    """Defect 3: the mask must line up with when the sound came BACK."""

    def test_the_bus_is_delayed_by_the_round_trip(self):
        m = realtime.SelfMasker(latency_hops=3, transients=False)
        m.human_rms_db = -20.0
        # three frames of loud bus: that sound has not reached the mic yet
        _, flags = m.mask([_aframe(), _aframe(), _aframe()], [0.0, 0.0, 0.0])
        assert flags == [False, False, False]
        # the fourth frame is where the first block's sound actually arrives
        _, flags = m.mask([_aframe()], [-90.0])
        assert flags == [True]

    def test_zero_latency_is_the_old_immediate_behaviour(self):
        m = realtime.SelfMasker(latency_hops=0, transients=False)
        m.human_rms_db = -20.0
        _, flags = m.mask([_aframe()], [0.0])
        assert flags == [True]

    def test_surplus_bus_values_are_kept_not_dropped(self):
        # They belong to frames the extractor has not emitted yet.  Dropping
        # them (the pre-v1.6 ``if i < len(...)``) would shift the alignment by
        # however many were dropped, permanently and silently.
        m = realtime.SelfMasker(transients=False)
        m.human_rms_db = -20.0
        _, flags = m.mask([_aframe()], [0.0, 0.0, 0.0])
        assert flags == [True]
        _, flags = m.mask([_aframe(), _aframe()], [])
        assert flags == [True, True]

    def test_set_latency_hops_refuses_once_masking_has_started(self):
        m = realtime.SelfMasker()
        m.mask([_aframe()], [-90.0])
        with pytest.raises(RealtimeError):
            m.set_latency_hops(4)

    def test_chunking_invariance(self):
        # Same stream, one call or a hundred: identical flags.  The masker is
        # fed whatever the ring happened to hold, so this is not academic.
        rng = np.random.default_rng(11)
        frames = [_aframe(t_s=i * 0.0107, rms_db=float(rng.uniform(-40, -10)),
                          attack=float(rng.uniform(0, 8)))
                  for i in range(200)]
        bus = [float(rng.uniform(-70, -5)) for _ in range(200)]

        one = realtime.SelfMasker(latency_hops=2)
        _, flags_one = one.mask(frames, bus)

        many = realtime.SelfMasker(latency_hops=2)
        flags_many: list[bool] = []
        i = 0
        for step in (1, 7, 3, 40, 11, 60, 78):
            _, fl = many.mask(frames[i:i + step], bus[i:i + step])
            flags_many.extend(fl)
            i += step
        _, fl = many.mask(frames[i:], bus[i:])
        flags_many.extend(fl)
        assert flags_one == flags_many
        assert one.coupling_db == pytest.approx(many.coupling_db)


class TestSelfMaskerTransients:
    """The rule a level comparison cannot supply.

    ``BeatTracker`` picks onsets off ``Frame.attack``, a banded LOG RATIO and
    therefore level-independent by design.  So "is the machine louder than the
    human" is the wrong question: plume's own drums win the attack competition
    against a sustaining open-tuned guitar from 12 dB BELOW it (measured, in
    the closed loop below).  The frames that matter are the ones our own
    attack lands in, whatever its level.
    """

    def test_a_bus_rise_masks_and_a_sustained_bus_does_not(self):
        m = realtime.SelfMasker(hold_s=0.0)
        m.human_rms_db = 0.0        # level rule cannot fire (human is huge)
        m.mask([_aframe(t_s=i * 0.0107) for i in range(20)], [-50.0] * 20)
        settled = m.machine_transients      # the machine coming up out of
        # silence is itself a rise, and is masked on purpose
        _, flags = m.mask([_aframe(t_s=0.3)], [-30.0])   # 20 dB: our own hit
        assert flags == [True]
        assert m.machine_transients == settled + 1
        # ...and holding that level is NOT an attack.  A machine pad, or the
        # long body of a chord, must not gate the human out: only the frames
        # our own transient lands in do.
        _, flags = m.mask([_aframe(t_s=0.31 + i * 0.0107) for i in range(40)],
                          [-30.0] * 40)
        assert not any(flags[20:])

    def test_a_hit_masks_about_its_own_length_not_the_whole_bar(self):
        # A drum hit at 96 bpm: ~90 ms of sound in a 625 ms beat.  The masked
        # share must stay in that neighbourhood, not swallow the beat.
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        beat_frames, hit_frames = 58, 9
        for i in range(beat_frames * 8):
            phase = i % beat_frames
            bus = (-18.0 - 3.0 * phase) if phase < hit_frames else -85.0
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-20.0)], [bus])
        assert m.masked_fraction < 3.0 * hit_frames / beat_frames

    def test_masking_stays_a_small_share_of_the_stream(self):
        # A mask that covers most of the stream is not a mask, it is a gate,
        # and a tracker left with nothing is not better off.
        m = realtime.SelfMasker()
        m.human_rms_db = -20.0
        rng = np.random.default_rng(5)
        for i in range(600):
            phase = i % 25              # a machine hit every 25 frames
            bus = -18.0 - 2.0 * phase if phase < 9 else -70.0
            m.mask([_aframe(t_s=i * 0.0107,
                            rms_db=float(rng.uniform(-25, -15)))], [bus])
        assert 0.0 < m.masked_fraction < 0.5, m.masked_fraction

    def test_nothing_is_masked_while_the_machine_is_silent(self):
        m = realtime.SelfMasker()
        for i in range(200):
            m.mask([_aframe(t_s=i * 0.0107, onset=True)], [-90.0])
        assert m.masked_frames == 0
        assert m.masked_fraction == 0.0

    def test_masked_fraction_is_zero_before_any_frame(self):
        assert realtime.SelfMasker().masked_fraction == 0.0


class TestStreamLatencyHops:
    def test_a_sounddevice_style_pair_is_summed(self):
        class S:
            latency = (0.010, 0.012)
        assert realtime.stream_latency_hops(S()) == 2   # 22 ms at 48k/512

    def test_a_scalar_latency_is_accepted(self):
        class S:
            latency = 0.0213
        assert realtime.stream_latency_hops(S()) == 2

    def test_a_stream_that_will_not_say_falls_back(self):
        class S:
            pass
        assert realtime.stream_latency_hops(S()) == int(round(
            realtime.SELF_MASK_DEFAULT_LATENCY_S * realtime.SR / HOP))

    def test_an_absurd_latency_is_clamped(self):
        class S:
            latency = (5.0, 5.0)
        assert realtime.stream_latency_hops(S()) == int(round(
            realtime.SELF_MASK_MAX_LATENCY_S * realtime.SR / HOP))

    def test_garbage_is_not_an_exception(self):
        class S:
            latency = "nope"
        assert realtime.stream_latency_hops(S()) >= 0


# --------------------------------------------- the closed loop, end to end

def _machine_drums(bpm: float, dur_s: float, *, seed: int = 3) -> np.ndarray:
    """A drum-ish machine bus: hard transients on every beat, like plume's."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * realtime.SR)
    buf = np.zeros(n, dtype=np.float64)
    beat = 60.0 / float(bpm)
    hit_n = int(0.09 * realtime.SR)
    t = np.arange(hit_n) / realtime.SR
    kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 28)
    snare = rng.standard_normal(hit_n) * np.exp(-t * 40) * 0.6
    k = 0
    while k * beat < dur_s:
        at = int(k * beat * realtime.SR)
        y = kick if k % 2 == 0 else (kick * 0.4 + snare)
        take = min(hit_n, n - at)
        if take > 0:
            buf[at:at + take] += y[:take]
        k += 1
    return (buf / max(1e-9, float(np.abs(buf).max())) * 0.35).astype(np.float32)


def _closed_loop(human, machine, *, bleed_db, delay_s=0.020, masker=None,
                 acoustic=False):
    """Run THE LIVE ANALYSIS PATH over a mic signal that contains the
    machine's own output, delayed and attenuated, the way a laptop mic does.

    Drives :func:`realtime.analysis_step` — the function ``jam`` and ``lead``
    actually call — rather than a re-implementation of it, because the defect
    this exists for was in the *wiring*, not in ``SelfMasker``: every unit test
    of the class passed while the flag it computed went nowhere.

    ``bleed_db`` of -200 means headphones: nothing of the machine reaches the
    mic at all.  ``acoustic`` selects the tracker mode ``test_acoustic.py``
    uses for sustained open-tuned guitar.
    """
    from plume.listen.beat import BeatTracker
    from plume.listen.harmony import HarmonyTracker

    n = max(human.size, machine.size)
    h = np.zeros(n, dtype=np.float32)
    h[:human.size] = human
    m = np.zeros(n, dtype=np.float32)
    m[:machine.size] = machine
    mic = h.copy()
    delay = int(delay_s * realtime.SR)
    if delay < n:
        mic[delay:] += m[: n - delay] * (10.0 ** (bleed_db / 20.0))

    extractor = FeatureExtractor(sr=realtime.SR)
    hop_rms = HopRms()
    masker = realtime.SelfMasker(latency_hops=2) if masker is None else masker
    tracker, harmony = BeatTracker(acoustic=acoustic), HarmonyTracker()
    seen = []
    for i in range(0, n, 4096):
        _f, _fl, beat, _h = realtime.analysis_step(
            extractor=extractor, hop_rms=hop_rms, masker=masker,
            tracker=tracker, harmony=harmony,
            human_block=mic[i:i + 4096], machine_block=m[i:i + 4096])
        if beat is not None:
            seen.append(beat)
    late = [b for b in seen if b.t_s > seen[-1].t_s * 0.5]
    return float(np.median([b.bpm for b in late])), seen[-1], masker


#: The tempi differ on purpose: with both at the same tempo a tracker fooled
#: by its own output gives the right answer for the wrong reason and the test
#: cannot tell.  Here, locking to 96 means plume followed itself.
HUMAN_BPM = 138.0
MACHINE_BPM = 96.0


@pytest.fixture(scope="module")
def material():
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures_gen import make_human_with_truth
    human, _truth = make_human_with_truth("steady", bpm=HUMAN_BPM, bars=12,
                                          seed=1)
    human = (human / max(1e-9, float(np.abs(human).max()))
             * 0.3).astype(np.float32)
    return human, _machine_drums(MACHINE_BPM, human.size / realtime.SR)


class TestClosedLoopSelfMasking:
    """plume hearing itself through the speakers, at a DIFFERENT tempo."""

    HUMAN_BPM = HUMAN_BPM
    MACHINE_BPM = MACHINE_BPM

    def test_the_tracker_locks_to_the_human_not_to_plume(self, material):
        human, machine = material
        bpm, last, masker = _closed_loop(human, machine, bleed_db=-6.0)
        assert bpm == pytest.approx(self.HUMAN_BPM, abs=4.0), (
            f"locked to {bpm:.1f} bpm with the human at {self.HUMAN_BPM} and "
            f"plume at {self.MACHINE_BPM} — plume followed its own drums "
            f"(masked {masker.masked_fraction:.0%} of frames)")
        assert last.locked

    def test_it_holds_across_the_range_of_room_bleed(self, material):
        human, machine = material
        for bleed_db in (-3.0, -12.0, -20.0):
            bpm, _last, _m = _closed_loop(human, machine, bleed_db=bleed_db)
            assert bpm == pytest.approx(self.HUMAN_BPM, abs=4.0), (
                f"bleed {bleed_db} dB: locked to {bpm:.1f} bpm")

    def test_without_the_mask_it_locks_to_the_machine(self, material):
        """The sentinel: proof that the two tests above can actually fail.

        This is the shipped pre-v1.6 behaviour — the flag computed and
        discarded — and it is what Juniper's session did.  If this ever starts
        passing, the closed loop has stopped reproducing the defect and the
        tests above have stopped meaning anything.
        """
        human, machine = material
        off = realtime.SelfMasker(latency_hops=2, transients=False,
                                  blank_envelope=False, adapt_coupling=False,
                                  coupling_db=-90.0)
        bpm, _last, _m = _closed_loop(human, machine, bleed_db=-6.0, masker=off)
        assert bpm == pytest.approx(self.MACHINE_BPM, abs=4.0), (
            f"expected the unmasked tracker to follow plume's own "
            f"{self.MACHINE_BPM} bpm drums; got {bpm:.1f}")

    def test_the_mask_costs_nothing_when_the_machine_is_silent(self, material):
        human, machine = material
        bpm, last, masker = _closed_loop(human, np.zeros_like(machine),
                                         bleed_db=-6.0)
        assert masker.masked_frames == 0
        assert bpm == pytest.approx(self.HUMAN_BPM, abs=2.0)
        assert last.confidence > 0.8

    def test_the_same_tempo_case_is_not_destabilised(self, material):
        human, _machine = material
        same = _machine_drums(self.HUMAN_BPM, human.size / realtime.SR)
        bpm, last, _m = _closed_loop(human, same, bleed_db=-6.0)
        assert bpm == pytest.approx(self.HUMAN_BPM, abs=2.0)
        assert last.locked


# ------------------- v1.7: the mask must not become the metronome ----------

#: v1.6 behaviour: the transient rule gated on the raw bus, i.e. on nothing.
_UNGATED = dict(transient_gate_db=1e9)

#: The acoustic fixtures need this many bars before the defect shows.  At 16
#: bars and fewer the ungated masker gives the RIGHT answer, which is the
#: whole reason four rounds of tests missed this: the harm is cumulative —
#: periodic blanking has to persist long enough to dominate the tracker's
#: autocorrelation window.  Juniper played for six minutes.
_ACOUSTIC_BARS = 20


@pytest.fixture(scope="module")
def acoustic_material():
    """Sustained open-tuned acoustic guitar + machine drums at a DIFFERENT
    tempo.  This is the material from the room, not the programmed corpus
    every earlier masking test used."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_acoustic import make_acoustic

    out = {}
    for kind, truth in (("fingerpick", 84.0), ("strum", 72.0)):
        audio, _doc = make_acoustic(kind, bpm=truth, bars=_ACOUSTIC_BARS,
                                    seed=7)
        audio = np.asarray(audio, dtype=np.float32)
        audio = (audio / max(1e-9, float(np.abs(audio).max()))
                 * 0.3).astype(np.float32)
        out[kind] = (audio, truth,
                     _machine_drums(MACHINE_BPM, audio.size / realtime.SR))
    return out


class TestTheMaskDoesNotStampTheMachinesTempoOnTheHuman:
    """The v1.6 fix, applied to the material it was meant for, made it worse.

    A masked frame is a BLANKED frame (``attack``/``flux`` -> 0), and the
    transient rule fired on any rise of the machine's own bus — a fact about
    what we SENT.  So it fired on headphones too, where there is nothing in
    the mic to mask, and the blanks landed at the machine's own beat rate.  To
    an autocorrelation a periodic hole is not missing evidence, it IS a
    period: the mask taught the tracker the machine's tempo on a signal that
    did not contain it.

    These run with NO BLEED AT ALL.  Any influence of the machine on the
    reading is the mask's doing and nothing else.
    """

    @pytest.mark.parametrize("kind", ["fingerpick", "strum"])
    def test_on_headphones_the_mask_leaves_the_human_s_tempo_alone(
            self, acoustic_material, kind):
        # v2.7: on ``fingerpick`` this is now held by TWO mechanisms — the
        # transient gate, and SPEC-V1 amendment 48's settled-grid guard.  The
        # gate's own contribution is pinned at frame level by
        # test_the_gate_is_what_stops_the_bus_being_blanked.
        human, truth, machine = acoustic_material[kind]
        bpm, _last, m = _closed_loop(human, machine, bleed_db=-200.0,
                                     acoustic=True)
        assert bpm == pytest.approx(truth, abs=4.0), (
            f"{kind}: nothing of the machine reached the mic, and the tracker "
            f"still read {bpm:.1f} bpm against a true {truth} with plume "
            f"playing {MACHINE_BPM} — the mask put it there "
            f"(masked {m.masked_fraction:.0%} of frames)")

    @pytest.mark.parametrize("kind", ["fingerpick", "strum"])
    def test_the_gate_is_what_stops_the_bus_being_blanked(self,
                                                          acoustic_material,
                                                          kind):
        """THE SENTINEL, at the layer the gate actually acts on.

        The gate decides which frames get BLANKED.  Everything downstream of
        that — what the autocorrelation makes of a periodic hole, what the
        tracker then reports — is a second mechanism that can change
        underneath this file, and in v2.7 one did (see the test below).  So
        the load-bearing claim is pinned here first, where nothing else can
        rescue it: with the gate ungated the masker blanks 16% of frames on
        both fixtures, against 1.7% (fingerpick) and 0.5% (strum) gated.

        On headphones there is nothing in the mic to mask, so the gated
        numbers are the honest floor and the 16% is pure fabrication.
        """
        human, _truth, machine = acoustic_material[kind]
        _b, _l, gated = _closed_loop(human, machine, bleed_db=-200.0,
                                     acoustic=True)
        _b, _l, ungated = _closed_loop(
            human, machine, bleed_db=-200.0, acoustic=True,
            masker=realtime.SelfMasker(latency_hops=2, **_UNGATED))
        assert ungated.masked_fraction > 0.10, (
            f"{kind}: the UNGATED masker blanked only "
            f"{ungated.masked_fraction:.0%} of frames — it is no longer "
            f"reproducing the v1.6 defect, so this file's subject is gone")
        assert ungated.masked_fraction > gated.masked_fraction * 4.0, (
            f"{kind}: gated {gated.masked_fraction:.1%} vs ungated "
            f"{ungated.masked_fraction:.1%} — the transient gate is no longer "
            f"what decides whether the bus gets blanked")

    @pytest.mark.parametrize("kind", ["strum"])
    def test_the_ungated_transient_rule_is_what_did_it(self, acoustic_material,
                                                       kind):
        """The end-to-end half: same audio, same tracker, gate off -> wrong
        answer.  ``strum`` reads 96.15 against a true 72 — plume's own tempo,
        exactly, on a signal that never carried it, and a 4/3 of the truth.

        Stated as a DIFFERENCE rather than as "it reads exactly 96", because
        what has to stay true is that the gate is load-bearing, not that the
        broken version fails in one particular way.

        **v2.7: ``fingerpick`` left this parametrisation, and why that is a
        re-derivation and not a relaxation.**  It used to read 96.2 against a
        true 84 and now reads 84.22 — the blanking still happens (16% of
        frames, pinned by the test above), but SPEC-V1 amendment 48 stopped a
        SETTLED grid yielding to a non-octave alias in one second, and 96.2/84
        is 1.145.  The fingerpick fixture settles on 84 before the periodic
        holes accumulate, so the tracker now refuses them; the strum fixture
        does not settle first, so 96.2/72 = 4/3 still captures it.  That is
        defence in depth, not a fixed gate: the mask still fabricates a
        machine-rate periodicity, and the frame-level sentinel above is where
        that claim now lives.  If amendment 48 is ever reverted, put
        ``fingerpick`` back in this list — it will fail again.

        If THIS ever stops failing too, ``test_on_headphones...`` has lost its
        end-to-end proof and both should be re-derived rather than relaxed.
        """
        human, truth, machine = acoustic_material[kind]
        bpm, _last, m = _closed_loop(
            human, machine, bleed_db=-200.0, acoustic=True,
            masker=realtime.SelfMasker(latency_hops=2, **_UNGATED))
        assert abs(bpm - truth) > 4.0, (
            f"{kind}: the ungated mask read {bpm:.1f} against a true {truth} "
            f"— i.e. it did no harm, so the gate this file is about is not "
            f"load-bearing here any more (masked {m.masked_fraction:.0%})")

    def test_with_the_room_in_the_mic_the_mask_still_earns_its_place(
            self, acoustic_material):
        """The gate must not have turned the mask off.  On fingerpick through
        speakers the mask is the difference between right and wrong."""
        human, truth, machine = acoustic_material["fingerpick"]
        on, _l, m = _closed_loop(human, machine, bleed_db=-6.0, acoustic=True)
        off, _l2, _m2 = _closed_loop(
            human, machine, bleed_db=-6.0, acoustic=True,
            masker=realtime.SelfMasker(latency_hops=2, transients=False,
                                       blank_envelope=False,
                                       adapt_coupling=False,
                                       coupling_db=-90.0))
        assert on == pytest.approx(truth, abs=4.0), (
            f"masked, through speakers, the tracker read {on:.1f} not {truth}")
        assert off == pytest.approx(MACHINE_BPM, abs=4.0), (
            f"unmasked it should follow plume's own drums; got {off:.1f} — "
            f"the test no longer proves the mask does anything")
        assert m.masked_frames > 0

    def test_the_gate_is_above_the_measured_competition_threshold(self):
        """plume's drums win the attack competition against a sustaining
        guitar even 12 dB below it (measured, see SELF_MASK_RISE_DB's note).
        A gate at or under 12 would stop masking sound that still fools the
        tracker."""
        assert realtime.SELF_MASK_TRANSIENT_GATE_DB > 12.0
        assert (realtime.SELF_MASK_TRANSIENT_GATE_DB
                > realtime.SELF_MASK_MARGIN_DB), (
            "the transient gate must be looser than the level rule's margin, "
            "or the level rule could mask a frame the transient rule has "
            "already judged inaudible")


class TestTheTransientGateIsAboutTheMicNotTheBus:
    """Unit statements of the same rule, without a tracker in the way."""

    def _drive(self, *, bus_db, human_db, coupling_db, n=400):
        m = realtime.SelfMasker(coupling_db=coupling_db, adapt_coupling=False)
        m.human_rms_db = human_db
        for i in range(n):
            phase = i % 25                      # a machine hit every 25 frames
            bus = bus_db if phase < 6 else -70.0
            m.mask([_aframe(t_s=i * 0.0107, rms_db=human_db)], [bus])
        return m

    def test_a_transient_the_mic_cannot_hear_does_not_mask(self):
        # bus is loud, but the room swallows it: 40 dB under the human.
        m = self._drive(bus_db=-10.0, human_db=-20.0, coupling_db=-50.0)
        assert m.masked_frames == 0, (
            f"masked {m.masked_fraction:.0%} of the stream for sound that "
            f"arrives 40 dB under the player")

    def test_a_transient_the_mic_can_hear_still_masks(self):
        # same bus, an untreated room: the machine lands right on the human.
        m = self._drive(bus_db=-10.0, human_db=-20.0, coupling_db=-10.0)
        assert m.masked_frames > 0

    def test_the_gate_is_a_level_comparison_not_a_bus_threshold(self):
        """Identical bus, identical coupling, louder human -> less masking.
        Under v1.6 the human's level could not affect the transient rule at
        all, which is precisely how headphones and a live room became the
        same case."""
        quiet = self._drive(bus_db=-20.0, human_db=-24.0, coupling_db=-6.0)
        loud = self._drive(bus_db=-20.0, human_db=0.0, coupling_db=-6.0)
        assert quiet.masked_frames > loud.masked_frames


# ------------------- v1.8: the coupling floor (the mask must converge OFF) --

def _closed_loop_states(human, machine, *, bleed_db, delay_s=0.020,
                        masker=None, acoustic=False):
    """Like :func:`_closed_loop` but returns EVERY beat state, because the
    v1.8 defect is invisible to a median: the harm is a MINORITY of samples
    reading the machine's tempo, growing with take length.  A median over the
    late half read 72.0 while 92% of one 20-second window read 96."""
    from plume.listen.beat import BeatTracker
    from plume.listen.harmony import HarmonyTracker

    n = max(human.size, machine.size)
    h = np.zeros(n, dtype=np.float32)
    h[:human.size] = human
    m = np.zeros(n, dtype=np.float32)
    m[:machine.size] = machine
    mic = h.copy()
    delay = int(delay_s * realtime.SR)
    if delay < n and bleed_db > -150.0:      # <= -150 means headphones exactly
        mic[delay:] += m[: n - delay] * (10.0 ** (bleed_db / 20.0))

    extractor = FeatureExtractor(sr=realtime.SR)
    hop_rms = HopRms()
    masker = realtime.SelfMasker(latency_hops=2) if masker is None else masker
    tracker, harmony = BeatTracker(acoustic=acoustic), HarmonyTracker()
    seen = []
    for i in range(0, n, 4096):
        _f, _fl, beat, _h = realtime.analysis_step(
            extractor=extractor, hop_rms=hop_rms, masker=masker,
            tracker=tracker, harmony=harmony,
            human_block=mic[i:i + 4096], machine_block=m[i:i + 4096])
        if beat is not None:
            seen.append(beat)
    return seen, masker


def _worst_window_machine_share(seen, *, tol=2.0, window_s=20.0, step_s=5.0):
    """The metric the defect was measured with: over every ``window_s`` span,
    the share of tempo samples within ``tol`` of the MACHINE's bpm — worst
    window returned.  Assert on this, never on a median (method lesson: two
    rounds of defects survived medians)."""
    ts = np.array([b.t_s for b in seen])
    bpm = np.array([b.bpm for b in seen])
    is_machine = np.abs(bpm - MACHINE_BPM) < tol
    worst = 0.0
    t0 = 0.0
    end = float(ts[-1]) if ts.size else 0.0
    while t0 + window_s <= end:
        sel = (ts >= t0) & (ts < t0 + window_s)
        if int(sel.sum()) > 10:
            worst = max(worst, float(is_machine[sel].mean()))
        t0 += step_s
    return worst


#: Long-take fixture lengths.  48 bars is where the measured share hit 14.7%
#: (2.9% at 20 bars — which is why every earlier round missed it), 96 is the
#: six-minutes-of-playing regime Juniper actually plays.
_COUPLING_CASES = [("fingerpick", 84.0, 48), ("strum", 72.0, 48),
                   ("strum", 72.0, 96)]


@pytest.fixture(scope="module")
def coupling_material():
    """Zero-bleed long takes: sustained acoustic + machine drums, 48/96 bars."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_acoustic import make_acoustic

    out = {}
    for kind, truth, bars in _COUPLING_CASES:
        audio, _doc = make_acoustic(kind, bpm=truth, bars=bars, seed=7)
        audio = np.asarray(audio, dtype=np.float32)
        audio = (audio / max(1e-9, float(np.abs(audio).max()))
                 * 0.3).astype(np.float32)
        out[(kind, bars)] = (audio, truth,
                             _machine_drums(MACHINE_BPM,
                                            audio.size / realtime.SR))
    return out


class TestTheCouplingFloor:
    """v1.8: silence in the mic must not be required for honesty.

    ``SelfMasker.coupling_db`` was a leaky MINIMUM of (mic - bus): its only
    real observations are the human's gaps, and a sustained open-tuned guitar
    never leaves one.  Measured on the zero-bleed strum fixture, the estimate
    sat at -5..-11 dB for the whole take — indistinguishable from a real
    -6 dB room — so ~11% of frames were blanked at the MACHINE's beat rate,
    and the share of tempo samples reading the machine's 96 GREW with take
    length: 2.9% at 20 bars, 7.1% at 24, 7.8% at 32, 14.7% at 48 — and 0.0%
    at every length with the mask off.  The 20-bar fixtures the earlier
    rounds used sat inside the warm-up; the harm is cumulative.

    The fix estimates coupling by regression of mic power on bus power over
    an EWMA window — the machine's own inter-hit silences anchor the
    baseline, so no human gap is needed.  The invariant: with zero bleed the
    mask must converge to NEVER masking, at every take length.
    """

    @pytest.mark.parametrize(("kind", "truth", "bars"), _COUPLING_CASES)
    def test_zero_bleed_worst_window_machine_share_is_under_1pct(
            self, coupling_material, kind, truth, bars):
        human, _truth, machine = coupling_material[(kind, bars)]
        seen, masker = _closed_loop_states(human, machine, bleed_db=-200.0,
                                           acoustic=True)
        worst = _worst_window_machine_share(seen)
        assert worst < 0.01, (
            f"{kind}/{bars} bars, zero bleed: {worst:.1%} of one window read "
            f"the machine's {MACHINE_BPM} bpm (true {truth}) — the coupling "
            f"estimate is masking a machine the mic cannot hear "
            f"(coupling {masker.coupling_db:.1f} dB, "
            f"masked {masker.masked_fraction:.1%})")

    def test_the_mask_converges_to_never_masking_on_headphones(
            self, coupling_material):
        """The invariant itself, stated on the masker rather than the tracker:
        after the estimator's warm-up, a zero-bleed take masks NOTHING."""
        human, _truth, machine = coupling_material[("strum", 48)]
        seen, masker = _closed_loop_states(human, machine, bleed_db=-200.0,
                                           acoustic=True)
        assert masker.coupling_db <= -50.0, (
            f"48 bars of zero bleed and the coupling estimate still reads "
            f"{masker.coupling_db:.1f} dB — it never converged")
        # Whole-take fraction: the warm-up may mask a few frames at the top
        # (worst-case-first is correct there), but over 48 bars that must
        # amortize to almost nothing — 11% was the broken steady state.
        assert masker.masked_fraction < 0.01, (
            f"masked {masker.masked_fraction:.1%} of a take with no bleed")

    def test_differential_sentinel_mask_on_equals_mask_off_at_48_bars(
            self, coupling_material):
        """Mask on vs mask off, same zero-bleed audio, 48 bars: the tempo
        trajectory must be the same.  Any divergence is the mask's own doing —
        there is nothing of the machine in the mic to mask."""
        human, truth, machine = coupling_material[("strum", 48)]
        on_seen, _m = _closed_loop_states(human, machine, bleed_db=-200.0,
                                          acoustic=True)
        off = realtime.SelfMasker(latency_hops=2, transients=False,
                                  blank_envelope=False, adapt_coupling=False,
                                  coupling_db=-90.0)
        off_seen, _m2 = _closed_loop_states(human, machine, bleed_db=-200.0,
                                            acoustic=True, masker=off)
        w_on = _worst_window_machine_share(on_seen)
        w_off = _worst_window_machine_share(off_seen)
        assert w_off == 0.0, "the sentinel's baseline is broken"
        assert w_on <= w_off + 0.01, (
            f"mask on stamped {w_on:.1%} of a window with the machine's tempo "
            f"where mask off stamped {w_off:.1%}")
        on_bpm = float(np.median([b.bpm for b in on_seen
                                  if b.t_s > on_seen[-1].t_s * 0.5]))
        assert on_bpm == pytest.approx(truth, abs=4.0)

    def test_the_speakers_case_still_masks_at_48_bars(self, coupling_material):
        """The other half of the invariant: the regression must not have
        turned the mask off in a real room.  Same 48-bar strum, -6 dB bleed:
        the masker must still be doing SOMETHING (and the earlier bleed-matrix
        tests prove the something is the right thing on the material masking
        can recover)."""
        human, _truth, machine = coupling_material[("strum", 48)]
        _seen, masker = _closed_loop_states(human, machine, bleed_db=-6.0,
                                            acoustic=True)
        assert masker.masked_frames > 0, (
            "with real -6 dB bleed the mask never fired at all — the "
            "regression has become an off switch")
        assert masker.coupling_db > -30.0, (
            f"coupling read {masker.coupling_db:.1f} dB in a -6 dB room")


class TestCouplingRegressionUnit:
    """The v1.8 estimator in isolation: what evidence moves it, what cannot."""

    def _spiky_bus(self, i, *, loud=-20.0, period=25, width=6):
        return loud if (i % period) < width else -85.0

    def test_uncorrelated_mic_drives_the_estimate_down_and_the_mask_off(self):
        # A busy human (level wandering slowly around -25, the shape a
        # sustained guitar's envelope actually has) and a spiky machine bus
        # the mic never sees.  The leaky minimum alone stuck at ~-11 dB here
        # (measured) — high enough to mask ~11% of frames.  The regression
        # must read "the mic does not move with the bus", fall to the floor,
        # and the mask must stop firing entirely.
        m = realtime.SelfMasker()

        def human_db(i):
            return -25.0 + 4.0 * math.sin(i / 50.0)

        for i in range(2000):
            m.mask([_aframe(t_s=i * 0.0107, rms_db=human_db(i))],
                   [self._spiky_bus(i)])
        # -40 is 22 dB below the level at which the transient gate can fire
        # against this human; the closed-loop fixtures (real audio) converge
        # all the way to the -60 floor and own that stronger claim.
        assert m.coupling_db <= -40.0, (
            f"20 s of zero-bleed evidence and coupling still reads "
            f"{m.coupling_db:.1f} dB")
        settled = m.masked_frames
        for i in range(2000, 3000):
            m.mask([_aframe(t_s=i * 0.0107, rms_db=human_db(i))],
                   [self._spiky_bus(i)])
        assert m.masked_frames == settled, (
            "the estimate is low and the mask is still firing")

    def test_a_real_room_is_read_close_to_its_true_gain(self):
        # mic power = human power + g^2 * bus power, g = -12 dB: the textbook
        # case the regression must nail without the human ever going quiet.
        rng = np.random.default_rng(4)
        g2 = 10.0 ** (-12.0 / 10.0)
        m = realtime.SelfMasker()
        for i in range(2000):
            bus_db = self._spiky_bus(i)
            h_pow = 10.0 ** (float(rng.uniform(-30, -24)) / 10.0)
            mic_pow = h_pow + g2 * (10.0 ** (bus_db / 10.0))
            m.mask([_aframe(t_s=i * 0.0107,
                            rms_db=10.0 * math.log10(mic_pow))], [bus_db])
        assert m.coupling_db == pytest.approx(-12.0, abs=3.0)

    def test_a_constant_bus_leaves_the_leaky_minimum_in_charge(self):
        # A flat pad's slope is noise, not a measurement: the regression must
        # stand aside (this is the pre-v1.8 estimator, byte for byte).
        m = realtime.SelfMasker(transients=False)
        for i in range(400):
            m.mask([_aframe(t_s=i * 0.0107, rms_db=-45.0)], [-20.0])
        assert m._regression_coupling_db() is None
        assert m.coupling_db == pytest.approx(-25.0, abs=1.0)

    def test_turning_the_speakers_up_is_still_followed(self):
        # The regression pinned the estimate at the floor (headphones)…
        rng = np.random.default_rng(5)
        m = realtime.SelfMasker()
        for i in range(1500):
            m.mask([_aframe(t_s=i * 0.0107,
                            rms_db=float(rng.uniform(-30, -24)))],
                   [self._spiky_bus(i)])
        assert m.coupling_db <= -50.0
        # …then the player unplugs the headphones: the machine lands in the
        # mic at -6 dB.  The estimate must come back up (creep-bounded).
        g2 = 10.0 ** (-6.0 / 10.0)
        for i in range(1500, 6000):
            bus_db = self._spiky_bus(i)
            h_pow = 10.0 ** (float(rng.uniform(-30, -24)) / 10.0)
            mic_pow = h_pow + g2 * (10.0 ** (bus_db / 10.0))
            m.mask([_aframe(t_s=i * 0.0107,
                            rms_db=10.0 * math.log10(mic_pow))], [bus_db])
        assert m.coupling_db > -15.0, (
            f"the room changed and the estimate stayed at "
            f"{m.coupling_db:.1f} dB — a stale 'quiet room' is pinning the "
            f"mask off")

    def test_regression_states_are_chunking_invariant(self):
        rng = np.random.default_rng(6)
        frames = [_aframe(t_s=i * 0.0107, rms_db=float(rng.uniform(-40, -10)),
                          attack=float(rng.uniform(0, 8)))
                  for i in range(300)]
        bus = [self._spiky_bus(i, loud=float(rng.uniform(-25, -15)))
               for i in range(300)]
        one = realtime.SelfMasker(latency_hops=2)
        _, f_one = one.mask(frames, bus)
        many = realtime.SelfMasker(latency_hops=2)
        f_many: list[bool] = []
        i = 0
        for step in (1, 13, 5, 100, 31, 150):
            _, fl = many.mask(frames[i:i + step], bus[i:i + step])
            f_many.extend(fl)
            i += step
        assert f_one == f_many
        assert one.coupling_db == pytest.approx(many.coupling_db)


# ------------------------------------------------- v1.7: the bounded history

class TestUnlockedEventHistoryIsBounded:
    """The planner's per-frame history must not grow with the session.

    ``shared["events"]`` is trimmed at a bar boundary — and a bar boundary
    only exists once the beat tracker has LOCKED.  Juniper's six-minute take
    never locked once (the postmortem's finding 1: every status sample read
    `listening`), so the trim never ran and the list grew for the whole
    session: ~34k tuples at six minutes, ~225k at forty.  That is the same
    accumulate-until-the-end shape :class:`SessionRecorder` was written to
    remove, left behind in the planner, and it is worst in exactly the
    condition the machine is hardest to use in.
    """

    def test_it_drops_what_no_bar_summary_could_ever_ask_for(self):
        events = [(float(i) * 0.01, -20.0, False) for i in range(10_000)]
        now = events[-1][0]
        kept = realtime.trim_events(events, now)
        assert kept, "everything was dropped"
        assert kept[0][0] >= now - realtime.EVENT_KEEP_S
        assert kept[-1] == events[-1], "the newest event was dropped"

    def test_it_keeps_two_bars_of_the_slowest_plausible_tempo(self):
        """A bar summary reaches back to ``next_downbeat - 2*bar_len``, so the
        window has to cover two bars or a trim could change a decision."""
        two_bars_at_16bpm = 2 * 4 * (60.0 / 16.0)
        assert realtime.EVENT_KEEP_S >= two_bars_at_16bpm

    def test_a_fresh_history_is_untouched(self):
        events = [(float(i) * 0.01, -20.0, i % 8 == 0) for i in range(100)]
        assert realtime.trim_events(events, events[-1][0]) == events

    def test_it_is_bounded_however_long_the_session_runs(self):
        """The property that matters: feed it an hour, one frame at a time,
        with the planner's own guard, and it never exceeds the window."""
        events: list = []
        dt = HOP / float(realtime.SR)
        for i in range(int(3600 / dt)):
            events.append((i * dt, -20.0, False))
            if len(events) > realtime.EVENT_TRIM_AT:
                events = realtime.trim_events(events, i * dt)
        assert len(events) <= realtime.EVENT_TRIM_AT + 1, (
            f"grew to {len(events)} over an hour")

    def test_the_planner_actually_calls_it(self):
        """Non-vacuity guard.  The pure function above can be perfect while
        the planner never reaches it — which is precisely the shape of the
        defect it exists for.  Read ``jam``'s own source and require the call
        inside the not-locked branch.
        """
        planner = _func_ast("jam", "planner_loop")
        branch = next(
            n for n in ast.walk(planner)
            if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.op, ast.Not)
            and "locked" in ast.dump(n.test.operand))
        called = {c.func.id for c in ast.walk(branch)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "trim_events" in called, (
            "the unlocked branch of the planner does not trim its event "
            "history — a session that never locks grows without bound")


# ------------------------------------------- v1.7: no key written over itself

class TestNoDecisionRecordWritesOverItself:
    """A dict literal with the same key twice keeps only the last one.

    ``lead``'s decision record had ``human_offset_ms`` in it twice: the
    negotiator's sticky reading, and thirteen lines later the raw per-bar
    measurement.  Python silently kept the second, so the first was computed
    and discarded on every bar of every take — with the key present and a
    plausible number in it, which is why nothing caught it.  This is the guard
    that cannot rot: it reads the module's own source.
    """

    def test_no_dict_literal_in_realtime_repeats_a_key(self):
        tree = ast.parse(_realtime_source())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            dupes = {k for k in keys if keys.count(k) > 1}
            if dupes:
                offenders.append((node.lineno, sorted(dupes)))
        assert not offenders, (
            "dict literals with a repeated key (the later value silently "
            f"wins): {offenders}")

    def test_the_lead_record_still_carries_both_offsets(self):
        """Both numbers survive, under names that cannot collide."""
        lead = _func_ast("lead")
        keys = [k.value for n in ast.walk(lead) if isinstance(n, ast.Dict)
                for k in n.keys if isinstance(k, ast.Constant)
                and isinstance(k.value, str)]
        assert keys.count("human_offset_ms") == 1, (
            "the raw per-bar measurement is missing, or written twice again")
        assert keys.count("role_human_offset_ms") == 1, (
            "the negotiator's sticky reading is gone — it is the signal "
            "tempo_policy='hold' exists to publish")


# =====================================================================
# v1.8 — the unlocked-tick plans must SOUND, the status line must not lie,
# and the machine bus must actually sit where the plan says (relative to
# the human).  The reported live symptom: status said "listening" for six
# minutes while the human heard intermittent playing — the bars placed
# during brief locks kept sounding after the lock fell, the unlocked plans
# the mind intended to sound (wobble holds, the beat-loss fade) were logged
# and never rendered, and no surface admitted either.
# =====================================================================

from plume.realtime import (  # noqa: E402
    apply_calibration,
    bus_calibration_step,
    live_status_state,
)


class TestLiveStatusState:
    """The pure mapping behind the readout — the observability contract."""

    def test_locked_names_the_bar(self):
        assert live_status_state(True, 12, True) == "bar 12"
        assert live_status_state(True, 12, False) == "bar 12"

    def test_unlocked_and_sounding_says_playing(self):
        assert live_status_state(False, -1, True) == "playing (no grid)"

    def test_unlocked_and_silent_says_listening(self):
        assert live_status_state(False, -1, False) == "listening"

    def test_the_jam_status_loop_actually_uses_it(self):
        """Non-vacuity: the pure function can be perfect while the transport
        keeps its hardcoded 'listening' — which was the defect."""
        jam = _func_ast("jam")
        called = {c.func.id for c in ast.walk(jam)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "live_status_state" in called, (
            "jam's status loop no longer derives its state word from what is "
            "scheduled — the readout can lie about audible playing again")


class TestUnlockedTickRendersOnTheLivePath:
    """FIX 2: the mind intends sound on the unlocked path (wobble-hold bars,
    the beat-loss fade — SPEC-V1), sim has rendered those since v1.2, and the
    live planner logged them, set ``planned_bar``, and never rendered a
    sample.  The AST guards pin the live branch to sim's exact call shape;
    the fake-device tests below prove audio actually reaches the bus.
    """

    def _unlocked_branch(self):
        planner = _func_ast("jam", "planner_loop")
        return next(
            n for n in ast.walk(planner)
            if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.op, ast.Not)
            and "locked" in ast.dump(n.test.operand))

    def test_the_unlocked_branch_renders_as_a_continuation(self):
        branch = self._unlocked_branch()
        renders = [c for c in ast.walk(branch)
                   if isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "render_bar"]
        assert renders, (
            "the unlocked branch of jam's planner never calls render_bar — "
            "unlocked-tick plans are logged but never sound (the reported "
            "live symptom)")
        for call in renders:
            kw = {k.arg: k.value for k in call.keywords}
            assert "continuation" in kw and getattr(
                kw["continuation"], "value", None) is True, (
                "an unlocked-tick bar is a held continuation (sim passes "
                "continuation=True); without it the fade rolls back and the "
                "entrance restarts on every wobble bar")

    def test_the_unlocked_branch_gates_on_the_sounding_vocabulary(self):
        """It must select exactly the plans sim's unlocked tick selects."""
        branch = self._unlocked_branch()
        assert "_SOUNDING_STATES" in ast.dump(branch), (
            "the unlocked render is not gated on _SOUNDING_STATES — either "
            "silent plans render (audible garbage) or the shared vocabulary "
            "was dropped and the twins can diverge")


# ---------------------------------------------- FIX 3: the calibration math

class TestBusCalibrationMath:
    RENDERED = 10.0 ** (-20.0 / 10.0)      # a bar rendered at -20 dB RMS

    def test_a_full_solve_targets_the_human_reference(self):
        # Rendered bar at -20 dB RMS, human at -25, plan wants -6 under the
        # human: the bus must move by (-25 + -6) - (-20) = -11 dB.
        db, base = bus_calibration_step(None, None, self.RENDERED, -25.0,
                                        -6.0, resolve=True)
        assert db == pytest.approx(-11.0, abs=1e-9)
        assert base == (pytest.approx(-11.0), -25.0)

    def test_no_human_reference_means_no_move(self):
        assert bus_calibration_step(None, None, 1e-3, None, -6.0,
                                    resolve=True) == (None, None)
        db, base = bus_calibration_step(-4.0, (-4.0, -25.0), 1e-3, None,
                                        -6.0, resolve=False)
        assert (db, base) == (-4.0, (-4.0, -25.0))

    def test_a_silent_render_on_a_resolve_bar_means_no_move(self):
        db, base = bus_calibration_step(-4.0, None, 0.0, -25.0, -6.0,
                                        resolve=True)
        assert (db, base) == (-4.0, None)

    def test_between_solves_only_the_human_moves_the_target(self):
        """The regression this design exists to prevent: a thinned bar
        renders QUIET ON PURPOSE, and a per-bar re-solve would boost it right
        back up.  With the human unmoved, the bar's own rendered level —
        however quiet — must change nothing."""
        base = (-11.0, -25.0)
        thin_ms = 10.0 ** (-38.0 / 10.0)       # a sparse thinned bar
        db, base2 = bus_calibration_step(-11.0, base, thin_ms, -25.0, -3.0,
                                         resolve=False)
        assert db == -11.0, (
            f"a quiet thinned bar moved the calibration to {db:+.1f} — the "
            f"bus is cancelling the mind's own dynamics")
        assert base2 == base

    def test_the_human_stepping_up_is_tracked_and_smoothed(self):
        base = (-11.0, -25.0)
        # human steps +12 dB -> target -11 + 12 = +1, approached by alpha
        db, _ = bus_calibration_step(-11.0, base, self.RENDERED, -13.0,
                                     -6.0, resolve=False)
        assert db == pytest.approx(-11.0 + realtime.LIVE_CAL_ALPHA * 12.0)
        for _ in range(12):
            db, _ = bus_calibration_step(db, base, self.RENDERED, -13.0,
                                         -6.0, resolve=False)
        assert db == pytest.approx(1.0, abs=0.05)

    def test_the_applied_gain_is_clamped_sane(self):
        low, _ = bus_calibration_step(None, None, self.RENDERED, -80.0,
                                      -6.0, resolve=True)
        high, _ = bus_calibration_step(None, None, self.RENDERED, 20.0,
                                       0.0, resolve=True)
        assert low == realtime.LIVE_CAL_MIN_DB
        assert high == realtime.LIVE_CAL_MAX_DB
        # ...and the tracking path clamps too
        db, _ = bus_calibration_step(-11.0, (-11.0, -25.0), self.RENDERED,
                                     40.0, -6.0, resolve=False)
        assert db <= realtime.LIVE_CAL_MAX_DB

    def test_apply_calibration_glides_without_a_step(self):
        bar = np.ones((1000, 2), dtype=np.float32)
        out = apply_calibration(bar, 1.0, 0.5)
        assert out[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert out[-1, 0] == pytest.approx(0.5, abs=1e-3)
        # continuity: the next bar starts exactly where this one ended
        nxt = apply_calibration(bar, 0.5, 0.25)
        assert nxt[0, 0] == pytest.approx(out[-1, 0], abs=1e-3)

    def test_apply_calibration_identity_is_free_and_mono_safe(self):
        bar = np.ones((8, 2), dtype=np.float32)
        assert apply_calibration(bar, 1.0, 1.0) is bar
        mono = np.ones(8, dtype=np.float32)
        out = apply_calibration(mono, 0.5, 0.5)
        assert out.shape == mono.shape
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_the_planner_logs_the_applied_gain_per_bar(self):
        """Observability: the calibration is the one number that decides how
        loud the machine is against the player, and it used to leave no
        trace.  Every decision record on the jam path carries ``cal_db``."""
        for path in ("_plan_and_place", "planner_loop"):
            fn = _func_ast("jam", path)
            consts = [n.value for n in ast.walk(fn)
                      if isinstance(n, ast.Constant) and n.value == "cal_db"]
            assert consts, (
                f"jam's {path} does not put cal_db in its decision record — "
                f"the applied calibration left no trace again")


# ------------------------------- the live paths, driven end to end (fake IO)

from conftest import fake_audio, patched_module, short_socket_path, wait_until  # noqa: E402


class _GridBeatTracker:
    """A stand-in BeatTracker emitting a controlled grid.

    ``locked=None`` free-runs unlocked forever (with a real, frozen bpm — the
    state a lock loss leaves behind); ``locked=True`` walks a clean locked
    grid so the boundary planner runs on schedule.  The point is to drive the
    TRANSPORT's planner deterministically; the real tracker has its own tests.
    """

    def __init__(self, *, bpm=120.0, locked=True, **_kw):
        self.bpm = float(bpm)
        self.locked = bool(locked)
        self._t = 0.0

    def tap(self, *_a):                                # pragma: no cover
        pass

    def process(self, frames, masked=None):  # noqa: ARG002 - contract
        from plume.listen.beat import BeatState

        if frames:
            self._t = float(frames[-1].t_s)
        interval = 60.0 / self.bpm
        bar_len = 4.0 * interval
        bar = int(self._t / bar_len)
        if not self.locked:
            return BeatState(
                t_s=self._t, bpm=self.bpm, confidence=0.35, locked=False,
                bar=-1, beat_in_bar=0, phase_beats=0.0,
                next_beat_t_s=math.inf, next_downbeat_t_s=math.inf,
                beat_interval_s=interval)
        return BeatState(
            t_s=self._t, bpm=self.bpm, confidence=0.95, locked=True,
            bar=bar, beat_in_bar=int((self._t / interval) % 4),
            phase_beats=(self._t % bar_len) / interval,
            next_beat_t_s=(int(self._t / interval) + 1) * interval,
            next_downbeat_t_s=(bar + 1) * bar_len,
            beat_interval_s=interval)


def _stub_mind_class(state, *, roles=None, gain_db=0.0):
    """A Mind whose every plan is ``state`` — the transport does the rest."""
    from plume.mind import BarPlan

    plan_roles = ({"pad": 1.0, "bass": 1.0} if roles is None else roles)

    class _StubMind:
        def __init__(self, *_a, **_kw):
            pass

        def plan_bar(self, beat, harmony, human):  # noqa: ARG002
            bar = beat.bar + 1 if beat.bar >= 0 else 0
            return BarPlan(bar=bar, state=state,
                           roles=dict(plan_roles), gain_db=float(gain_db),
                           density=0.5, register_ceiling_midi=None,
                           reason=f"stub: always {state}")

    return _StubMind


class _CapturingStatus(realtime.StatusLine):
    """Swapped in for the real StatusLine so a test can read the readout."""

    texts: list[str] = []

    def __init__(self, *_a, **_kw):
        super().__init__(stream=io.StringIO(), tty=False)

    def update(self, text):
        type(self).texts.append(text)
        super().update(text)


def _run_jam_stubbed(monkeypatch, tmp_path, *, human, mind_cls, tracker_kw,
                     stop_after_blocks=900, rate_x=12.0):
    """Run the REAL jam transport over the fake device with a stub mind and a
    stub tracker; returns (fake_device, decision_rows, status_texts)."""
    import plume.listen.beat as beat_mod
    import plume.mind as mind_mod

    reg = tmp_path / "registry"
    reg.mkdir()
    monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(reg))
    monkeypatch.setattr(mind_mod, "Mind", mind_cls)
    monkeypatch.setattr(
        beat_mod, "BeatTracker",
        lambda *a, **kw: _GridBeatTracker(**tracker_kw))
    _CapturingStatus.texts = []
    monkeypatch.setattr(realtime, "StatusLine", _CapturingStatus)

    sock = short_socket_path("rt-stub")
    rec = tmp_path / "take"
    fake = fake_audio(human, rate_x=rate_x)

    def stop_later():
        wait_until(lambda: fake.stream is not None
                   and fake.stream.blocks >= stop_after_blocks, timeout=60.0)
        try:
            realtime.send_stop(socket_path=sock)
        except realtime.RealtimeError:      # pragma: no cover - already down
            pass

    stopper = threading.Thread(target=stop_later, daemon=True)
    stopper.start()
    from plume.mind import MindParams
    with patched_module("sounddevice", fake):
        realtime.jam(style="lofi", seed=0, params=MindParams(),
                     record_dir=str(rec), socket_path=sock)
    stopper.join(timeout=10.0)
    rows = _read_ndjson(rec / "decisions.ndjson")
    return fake, rows, list(_CapturingStatus.texts)


def _machine_channel_rms_db(fake, lo_frac, hi_frac):
    out = fake.stream.output
    seg = out[int(out.shape[0] * lo_frac): int(out.shape[0] * hi_frac)]
    mono = seg.mean(axis=1).astype(np.float64)
    p = float(np.mean(mono * mono)) if mono.size else 0.0
    return 10.0 * math.log10(max(p, 1e-12))


class TestUnlockedRenderingEndToEnd:
    """The live path, through the fake device: what the decision log says
    sounded must be what the bus carried, and the status must match both."""

    def test_a_sounding_unlocked_plan_reaches_the_bus_and_the_status(
            self, monkeypatch, tmp_path):
        rng = np.random.default_rng(0)
        human = (rng.standard_normal(realtime.SR * 4) * 0.03).astype(np.float32)
        fake, rows, texts = _run_jam_stubbed(
            monkeypatch, tmp_path, human=human,
            mind_cls=_stub_mind_class("comping"),
            tracker_kw={"locked": False, "bpm": 120.0})
        assert fake.stream.sounded(), (
            "the mind planned sound on every unlocked tick and the bus is "
            "silent — unlocked-tick plans still never render on the live path")
        sounded_rows = [r for r in rows if r.get("sounding")]
        assert sounded_rows, (
            "no decision record admits the unlocked audio that sounded")
        assert any("playing (no grid)" in t for t in texts), (
            f"audio sounded while unlocked and no status line ever said so "
            f"(states seen: {sorted({t.split(']')[0] for t in texts if ']' in t})})")

    def test_a_silent_unlocked_plan_stays_silent_everywhere(
            self, monkeypatch, tmp_path):
        rng = np.random.default_rng(1)
        human = (rng.standard_normal(realtime.SR * 4) * 0.03).astype(np.float32)
        fake, rows, texts = _run_jam_stubbed(
            monkeypatch, tmp_path, human=human,
            mind_cls=_stub_mind_class("listening", roles={}),
            tracker_kw={"locked": False, "bpm": 120.0},
            stop_after_blocks=600)
        assert not fake.stream.sounded(), (
            "the mind planned silence and the bus carried audio")
        assert not any(r.get("sounding") for r in rows)
        assert not any("playing" in t for t in texts), (
            "the status claims playing over a silent bus — the readout lies "
            "in the other direction now")
        assert any("listening" in t for t in texts)


class TestLiveBusCalibrationEndToEnd:
    """FIX 3 through the fake device: the human arrives at two different
    levels and the machine bus must sit at ``plan.gain_db`` relative to EACH,
    which no fixed factor (and no absence of one) can do."""

    GAIN_DB = -6.0

    def test_the_machine_tracks_the_human_at_the_specified_offset(
            self, monkeypatch, tmp_path):
        rng = np.random.default_rng(2)
        half_s = 24
        quiet = (rng.standard_normal(realtime.SR * half_s)
                 * 10.0 ** (-26.0 / 20.0)).astype(np.float32)
        loud = (rng.standard_normal(realtime.SR * half_s)
                * 10.0 ** (-14.0 / 20.0)).astype(np.float32)
        human = np.concatenate([quiet, loud])
        fake, rows, _texts = _run_jam_stubbed(
            monkeypatch, tmp_path, human=human,
            mind_cls=_stub_mind_class("comping", gain_db=self.GAIN_DB),
            tracker_kw={"locked": True, "bpm": 120.0},
            stop_after_blocks=int(realtime.SR * (2 * half_s - 4) / 256),
            rate_x=10.0)
        assert fake.stream.sounded()
        # cal_db must be in the log, and must MOVE when the human's level does
        cals = [r["cal_db"] for r in rows if r.get("cal_db") is not None]
        assert cals, "no decision record carries the applied calibration"
        assert max(cals) - min(cals) > 6.0, (
            f"the human moved 12 dB and the applied calibration moved "
            f"{max(cals) - min(cals):.1f} dB — the bus is not tracking")
        # and the audio itself: the machine's level must follow the human's
        # 12 dB step (measured over the settled tail of each half).
        quiet_db = _machine_channel_rms_db(fake, 0.30, 0.48)
        loud_db = _machine_channel_rms_db(fake, 0.80, 0.98)
        step = loud_db - quiet_db
        assert step == pytest.approx(12.0, abs=4.0), (
            f"human stepped +12 dB, machine stepped {step:+.1f} dB")

    def test_the_bus_sits_at_the_planned_offset_from_the_human(
            self, monkeypatch, tmp_path):
        """Absolute placement, not just tracking: rendered machine RMS lands
        at human_ref + gain_db (within the tolerance the EMA and the noise
        floor of a short take allow)."""
        rng = np.random.default_rng(3)
        human = (rng.standard_normal(realtime.SR * 30)
                 * 10.0 ** (-20.0 / 20.0)).astype(np.float32)
        fake, rows, _texts = _run_jam_stubbed(
            monkeypatch, tmp_path, human=human,
            mind_cls=_stub_mind_class("comping", gain_db=self.GAIN_DB),
            tracker_kw={"locked": True, "bpm": 120.0},
            stop_after_blocks=int(realtime.SR * 26 / 256),
            rate_x=10.0)
        machine_db = _machine_channel_rms_db(fake, 0.5, 0.95)
        # the human's own RMS as the transport measured it: white noise at
        # -20 dBFS peak-scaled — measure it directly for the target
        p = float(np.mean(human.astype(np.float64) ** 2))
        human_db = 10.0 * math.log10(p)
        assert machine_db == pytest.approx(human_db + self.GAIN_DB, abs=4.0), (
            f"machine at {machine_db:.1f} dB against a human at "
            f"{human_db:.1f} dB — offset {machine_db - human_db:+.1f}, "
            f"planned {self.GAIN_DB:+.1f}")
