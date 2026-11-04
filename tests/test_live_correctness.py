"""Live-path correctness, driven through the real transports on the fake device.

Nothing here opens a real device: ``patched_module`` swaps ``sounddevice`` for
the stand-in in conftest.py, and that stand-in plays into a numpy buffer.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import fake_audio, patched_module, short_socket_path, wait_until
from plume import mind as mind_mod
from plume import realtime
from plume import roles as roles_mod
from plume.compose import PromptSpec, compose
from plume.duet import DuetEngine
from plume.listen import beat as beat_mod
from plume.score import Movement, SectionSpec
from test_bandentry import _take

SR = realtime.SR


@pytest.fixture(autouse=True)
def _private(tmp_path, monkeypatch, plume_registry):
    """No developer calibration file and no developer session registry."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))


def strum(bpm: float, seconds: float) -> np.ndarray:
    """A loud chord on every beat: unmistakably somebody playing, all the way."""
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(int(0.25 * SR)) / SR
    hit = sum(np.sin(2 * np.pi * f * t) for f in (130.81, 196.0, 329.63))
    hit = (hit * np.exp(-t * 6.0) * 0.3).astype(np.float32)
    at = 0.0
    while int(at * SR) + hit.shape[0] < x.shape[0]:
        i = int(at * SR)
        x[i:i + hit.shape[0]] += hit
        at += 60.0 / bpm
    return x


def two_section_session(*, bpm: float = 150.0):
    """A map whose section lengths are not multiples of their chord cycles."""
    sess = compose(PromptSpec(styles=["lofi"], bpm=bpm, key="C major",
                              movements=1, mood=[], name="sections"), seed=7)
    plays = dict(sess.movements[0].sections[0].plays)
    sess.movements = [Movement(name="m", sections=[
        SectionSpec(name="a", bars=3, energy=0.6, plays=plays,
                    chords=["C", "F", "G", "Am"]),
        SectionSpec(name="b", bars=6, energy=0.6, plays=plays,
                    chords=["Dm", "G", "Em", "A"]),
    ])]
    return sess


def run_lead(sess, *, tmp_path, human=None, rate_x=12.0, seconds=2.0, **kw):
    """Run plain lead for ``seconds`` of wall clock; return (fake, decisions)."""
    rec = tmp_path / "rec"
    sock = short_socket_path("corr")
    fake = fake_audio(human, rate_x=rate_x)

    def stop_later():
        assert wait_until(lambda: Path(sock).exists(), timeout=30.0)
        time.sleep(seconds)
        realtime.send_stop(socket_path=sock)

    stopper = threading.Thread(target=stop_later, daemon=True)
    stopper.start()
    with patched_module("sounddevice", fake):
        realtime.lead(session=sess, seed=0,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec), socket_path=sock, status=False,
                      max_minutes=0, **kw)
    stopper.join(timeout=5.0)
    rows = [json.loads(line) for line in
            (rec / realtime.REC_DECISIONS).read_text().splitlines()]
    return fake, [r for r in rows if isinstance(r.get("bar"), int)]


@pytest.mark.usefixtures("fake_device_consent")
class TestPlainLeadHearsTheBarThatFinished:
    """A bar is planned a whole bar ahead of the playhead, so the bar just
    before it has barely begun.  Plain lead read that one, at real speed, and
    so judged the player on a few milliseconds of audio.  Only real speed shows
    it: a device pumped faster fills the window before the planner wakes.
    """

    def test_at_real_speed_the_window_holds_a_whole_bar(self, tmp_path,
                                                        monkeypatch):
        spans: list[int] = []
        silent: list[bool] = []
        real = realtime.bar_features_dict

        def spy(bar, events, **kw):
            spans.append(len(events))
            out = real(bar, events, **kw)
            silent.append(out["silent"])
            return out

        monkeypatch.setattr(realtime, "bar_features_dict", spy)
        sess = compose(PromptSpec(styles=["lofi"], bpm=200.0, key="C major",
                                  movements=1, mood=[], name="w"), seed=7)
        _, rows = run_lead(sess, tmp_path=tmp_path, human=strum(200.0, 30.0),
                           rate_x=1.0, seconds=7.0)
        frames_per_bar = 1.2 * SR / realtime.HOP
        assert len(spans) >= 3, (
            f"only {len(spans)} bars had any frames at all, of {len(rows)}")
        assert min(spans[1:]) > 0.8 * frames_per_bar, (
            f"frames per bar {spans}, a whole bar is {frames_per_bar:.0f}")
        assert not any(silent), f"a player on every beat read as silent: {silent}"

    def test_a_tempo_change_does_not_slide_the_window(self, tmp_path,
                                                      monkeypatch):
        """The window is the bar that was played, at the tempo it was played."""
        windows: list[tuple[float, float]] = []
        units: list[float] = []
        real = roles_mod.onset_phase_offset_ms

        def spy(onsets, start, end, unit_s, **kw):
            windows.append((start, end))
            units.append(unit_s)
            return real(onsets, start, end, unit_s, **kw)

        monkeypatch.setattr(roles_mod, "onset_phase_offset_ms", spy)
        sess = compose(PromptSpec(styles=["lofi"], bpm=180.0, key="C major",
                                  movements=1, mood=[], name="w"), seed=7)
        rec = tmp_path / "rec"
        sock = short_socket_path("tempo")

        def cue_then_stop():
            assert wait_until(lambda: Path(sock).exists(), timeout=30.0)
            time.sleep(1.2)
            realtime.send_cue("slow down", socket_path=sock)
            time.sleep(1.8)
            realtime.send_stop(socket_path=sock)

        t = threading.Thread(target=cue_then_stop, daemon=True)
        t.start()
        fake = fake_audio(strum(180.0, 30.0), rate_x=6.0)
        with patched_module("sounddevice", fake):
            realtime.lead(session=sess, seed=0,
                          role=roles_mod.RoleParams(role="lead"),
                          record_dir=str(rec), socket_path=sock, status=False,
                          max_minutes=0)
        t.join(timeout=5.0)
        rows = [json.loads(line) for line in
                (rec / realtime.REC_DECISIONS).read_text().splitlines()]
        starts = {r["bar"]: r["t_s"] for r in rows if isinstance(r.get("bar"), int)}
        assert len({r["bpm"] for r in rows if "bpm" in r}) > 1, (
            "the tempo never moved, so this test would prove nothing")
        checked = 0
        for bar, t_bar in sorted(starts.items()):
            if bar < 2:
                continue
            for w_start, w_end in windows:
                if abs(w_end - starts[bar - 1]) < 1e-3:
                    assert w_start == pytest.approx(starts[bar - 2], abs=1e-3)
                    checked += 1
        assert checked >= 2
        lengths = [end - start for start, end in windows]
        assert max(lengths) > 1.01 * min(lengths), (
            "every measured bar had one length, so the subdivision check below "
            "would prove nothing")
        for length, unit in zip(lengths, units):
            assert unit == pytest.approx(length / 16.0, abs=1e-6), (
                "the sixteenth must come from the bar that was measured")


def run_with_cue(sess, *, tmp_path, human, cue, rate_x, cue_after, stop_after, **kw):
    """Plain lead with one cue sent mid-take; return the decision rows."""
    rec = tmp_path / "rec"
    sock = short_socket_path("cue")

    def cue_then_stop():
        assert wait_until(lambda: Path(sock).exists(), timeout=30.0)
        time.sleep(cue_after)
        realtime.send_cue(cue, at="now", socket_path=sock)
        time.sleep(stop_after)
        realtime.send_stop(socket_path=sock)

    t = threading.Thread(target=cue_then_stop, daemon=True)
    t.start()
    with patched_module("sounddevice", fake_audio(human, rate_x=rate_x)):
        realtime.lead(session=sess, seed=0,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec), socket_path=sock, status=False,
                      max_minutes=0, **kw)
    t.join(timeout=5.0)
    rows = [json.loads(line) for line in
            (rec / realtime.REC_DECISIONS).read_text().splitlines()]
    return rec, [r for r in rows if isinstance(r.get("bar"), int)]


@pytest.mark.usefixtures("fake_device_consent")
class TestTheMachinesRitardIsNotThePlayers:
    """The ending suite listens for the PLAYER slowing down.  A "ritard" cue
    slows the machine's own clock, so handing the form the machine's grid made
    the band's obedience read as the player's intent, and the take ended.

    The real tracker rarely locks on this fixture, and an unlocked tracker keeps
    the detector off whichever beat it is given, so the tests script a locked
    tracker: the only condition under which the difference can show.
    """

    def long_map(self):
        sess = compose(PromptSpec(styles=["lofi"], bpm=200.0, key="C major",
                                  movements=1, mood=[], name="r"), seed=7)
        plays = dict(sess.movements[0].sections[0].plays)
        sess.movements = [Movement(name="m", sections=[
            SectionSpec(name="a", bars=64, energy=0.6, plays=plays,
                        chords=["C", "F", "G", "Am"])])]
        return sess

    def script_tracker(self, monkeypatch, bpm_at):
        real = beat_mod.BeatTracker.process

        def locked(self, frames, **kw):
            st = real(self, frames, **kw)
            return dataclasses.replace(st, locked=True, bpm=bpm_at(st.t_s))

        monkeypatch.setattr(beat_mod.BeatTracker, "process", locked)

    def test_a_ritard_cue_does_not_end_a_take_with_a_player(self, tmp_path,
                                                            monkeypatch):
        self.script_tracker(monkeypatch, lambda t: 200.0)
        seen: list[int] = []
        heard: list[bool] = []
        real = mind_mod.Mind.plan_form

        def spy(self, role, beat, harmony, human, session, cue=None):
            seen.append(beat.bar)
            heard.append(human is not None and not human.silent)
            return real(self, role, beat, harmony, human, session, cue)

        monkeypatch.setattr(mind_mod.Mind, "plan_form", spy)
        _, rows = run_with_cue(self.long_map(), tmp_path=tmp_path,
                               human=strum(200.0, 60.0), cue="ritard",
                               rate_x=4.0, cue_after=0.3, stop_after=3.0)
        bpms = [r["bpm"] for r in rows]
        ended = [r for r in rows if r["form"]["action"] == "end"]
        assert not ended, f"{ended[0]['form']['reason']}; bpm {bpms}"
        assert min(bpms) < 170.0, f"the ritard never ran: {bpms}"
        assert sum(heard) > 3, f"no player was heard, so no ritard could read: {heard}"
        assert seen == [r["bar"] for r in rows], (
            "the form must count the machine's bars, not the tracker's")

    def test_the_players_own_ritard_still_ends_the_take(self, tmp_path,
                                                        monkeypatch):
        self.script_tracker(monkeypatch, lambda t: max(90.0, 200.0 * 0.98 ** t))
        rec = tmp_path / "rec"
        with patched_module("sounddevice",
                            fake_audio(strum(200.0, 60.0), rate_x=4.0)):
            realtime.lead(session=self.long_map(), seed=0,
                          role=roles_mod.RoleParams(role="lead"),
                          record_dir=str(rec), socket_path=short_socket_path("rit"),
                          status=False, max_minutes=0.4)
        rows = [json.loads(line) for line in
                (rec / realtime.REC_DECISIONS).read_text().splitlines()]
        assert [r for r in rows if isinstance(r.get("bar"), int)
                and r["form"]["action"] == "end"], (
            "a player slowing down no longer reaches the ending suite")


@pytest.mark.usefixtures("fake_device_consent")
class TestChordsFollowTheSection:
    """``SectionSpec.chords`` cycle from the top of the section.  ``sim`` does
    that; live indexed them by the session's bar number, so a section whose
    predecessor was not a multiple of the cycle started on the wrong chord."""

    def test_a_section_starts_on_its_first_chord(self, tmp_path):
        _, rows = run_lead(two_section_session(), tmp_path=tmp_path,
                           rate_x=12.0, seconds=2.0)
        by_bar = {r["bar"]: r for r in rows}
        assert {0, 1, 2, 3, 4} <= set(by_bar), sorted(by_bar)
        assert [by_bar[b]["form"]["section_idx"] for b in range(5)] == [0, 0, 0, 1, 1]
        assert [by_bar[b]["chord"] for b in range(5)] == ["C", "F", "G", "Dm", "G"]


    def test_a_re_entered_section_restarts_its_chords(self, tmp_path):
        """The section index does not change here, only the form's action says
        the section began again, and three bars do not fill a four-chord cycle."""
        sess = two_section_session()
        sess.movements[0].sections = sess.movements[0].sections[:1]
        _, rows = run_lead(sess, tmp_path=tmp_path, rate_x=12.0, seconds=2.0)
        by_bar = {r["bar"]: r for r in rows}
        assert {0, 1, 2, 3, 4} <= set(by_bar), sorted(by_bar)
        assert [by_bar[b]["form"]["section_idx"] for b in range(5)] == [0] * 5
        assert by_bar[3]["form"]["action"] in ("advance", "jump"), by_bar[3]["form"]
        assert [by_bar[b]["chord"] for b in range(5)] == ["C", "F", "G", "C", "F"]


@pytest.mark.usefixtures("fake_device_consent")
class TestFromBar:
    """``--from-bar N`` means "start N bars into the map"."""

    def test_it_starts_there_and_it_starts_now(self, tmp_path):
        sess = two_section_session()
        fake, rows = run_lead(sess, tmp_path=tmp_path, rate_x=12.0, seconds=1.5,
                              from_bar=5)
        first = rows[0]
        assert first["bar"] == 5
        assert first["t_s"] == pytest.approx(0.0, abs=1e-6), (
            "the first bar should sound straight away, not N bars from now")
        assert first["form"]["section_idx"] == 1, "it started at the wrong section"
        assert first["chord"] == "Em", "bar 5 is the third bar of b"
        out = fake.stream.output
        assert np.abs(out[:int(1.6 * SR)]).max() > 1e-3, (
            "the opening bar is silent")

    @pytest.mark.parametrize("from_bar", [0, 1, 3, 8, 9, 100])
    def test_seek_lands_inside_the_map(self, from_bar):
        sess = two_section_session()
        m = mind_mod.Mind(mind_mod.MindParams())
        idx, pos = m.seek_form(sess, from_bar)
        assert (idx, pos) == {0: (0, 0), 1: (0, 1), 3: (1, 0), 8: (1, 5),
                              9: (1, 5), 100: (1, 5)}[from_bar]

    def test_the_first_form_plan_keeps_the_seek(self):
        sess = two_section_session()
        m = mind_mod.Mind(mind_mod.MindParams())
        m.seek_form(sess, 5)
        beat = realtime.machine_beat_state(5, 0.0, 120.0)
        harm = realtime.machine_harmony_state(0.0, None, None)
        plan = m.plan_form(None, beat, harm, None, sess)
        assert plan.section_idx == 1
        assert plan.bars_left_in_section == 3

    def test_the_machine_clock_can_be_numbered_from_any_bar(self):
        c = realtime.MachineClock(120.0, t0=0.0, first_bar=5)
        assert c.commit() == (5, 0.0, 120.0)
        assert c.bar == 6


    def test_tap_start_from_a_bar_does_not_hear_the_knocks(self, tmp_path,
                                                           monkeypatch):
        """The window before the band's first bar holds the knocks, whichever
        bar the band starts on.  The first bar read as playing is the bar after.
        """
        from_bar = 3
        read: list[int] = []
        real = realtime.bar_features_dict

        def spy(bar, events, **kw):
            read.append(bar)
            return real(bar, events, **kw)

        monkeypatch.setattr(realtime, "bar_features_dict", spy)
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
        sock = short_socket_path("tapf")
        rec = tmp_path / "rec"
        with patched_module("sounddevice",
                            fake_audio(_take(seconds=20.0, room_s=30.0), rate_x=12.0)):
            realtime.lead(prompt="lofi", seed=0, tap_start=True, from_bar=from_bar,
                          role=roles_mod.RoleParams(role="lead"),
                          record_dir=str(rec), socket_path=sock, status=False,
                          max_minutes=0.3)
        rows = [json.loads(line) for line in
                (rec / realtime.REC_DECISIONS).read_text().splitlines()]
        bars = [r["bar"] for r in rows if isinstance(r.get("bar"), int)]
        assert bars[0] == from_bar and len(bars) > 3, bars
        assert read, "no bar was ever read as playing, so this proves nothing"
        assert min(read) == from_bar + 1, read


@pytest.mark.usefixtures("fake_device_consent")
class TestJamEndsWhileUnlocked:
    """A take that never locks is common: six minutes of open-tuned guitar did
    exactly that.  Its planner only watched for the graceful ending on the
    locked path, so it sat out the whole join timeout, and flushed the tail
    after the stream had closed, where nobody could hear it."""

    def test_the_tail_is_flushed_while_the_stream_is_still_open(self, monkeypatch):
        fake = fake_audio(np.zeros(SR, dtype=np.float32), rate_x=12.0)
        at_flush: list[bool] = []
        real = DuetEngine.flush_tail

        def spy(self):
            at_flush.append(fake.stream.aborted or fake.stream.closed)
            return real(self)

        monkeypatch.setattr(DuetEngine, "flush_tail", spy)
        sock = short_socket_path("unlk")
        stopped: list[float] = []

        def stop_later():
            assert wait_until(lambda: Path(sock).exists(), timeout=30.0)
            time.sleep(0.5)
            realtime.send_stop(socket_path=sock)
            stopped.append(time.monotonic())

        t = threading.Thread(target=stop_later, daemon=True)
        t.start()
        with patched_module("sounddevice", fake):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                         socket_path=sock, max_minutes=0)
        done = time.monotonic()
        t.join(timeout=5.0)
        assert at_flush == [False], "the tail was flushed into a dead stream"
        assert done - stopped[0] < realtime.PLANNER_JOIN_S, (
            "stopping an unlocked jam sat out the planner's join timeout")


def test_a_dead_worker_fails_the_suite(tmp_path):
    """The instrument behind every threaded test here: an exception on a worker
    thread has to fail a run, or a test with a dead planner passes in silence.
    """
    probe = tmp_path / "test_probe.py"
    probe.write_text(textwrap.dedent("""\
        import threading

        def test_worker_dies():
            t = threading.Thread(target=lambda: 1 / 0)
            t.start()
            t.join()
    """))
    config = Path(__file__).resolve().parents[1] / "pyproject.toml"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-c", str(config), "--rootdir", str(tmp_path), str(probe)],
        capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert proc.returncode != 0, proc.stdout
    assert "ZeroDivisionError" in proc.stdout
