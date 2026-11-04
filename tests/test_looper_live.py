"""The looper through the real lead transport and the real cue socket.

On the FAKE device only (tests/conftest.py), so nothing here makes a sound or
says anything about how a loop sounds in a room.  The human is a synthesized
tone at a frequency the band does not sit on, which lets a test measure that
tone in the machine's output and so see the loop by itself, after the human has
stopped.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from conftest import fake_audio, patched_module, short_socket_path, wait_until
from plume import cli, realtime
from plume import looper as lp
from plume import roles as roles_mod
from plume.compose import PromptSpec, compose

SR = realtime.SR
TONE_HZ = 3137.0
STOP = "(stop)"
SAFETY_S = 300.0    # wall clock; only a wedged take ever reaches it
#: The worst a bar may be written into the past.  Over this, bars have gone missing
#: from the output, and whatever else a test sees is not about the looper.
MAX_LATE_S = 0.25
#: The most mic audio the fake device may run ahead of the looper hearing it.
MAX_UNREAD_S = 0.25


CUES_DUE: list = [[]]    # the bars of the lines cue_when has not said yet


def notes(times_s, *, total_s, gain=0.4, hz=TONE_HZ):
    """Tone plucks at these times over silence."""
    x = np.zeros(int(total_s * SR), dtype=np.float32)
    t = np.arange(int(0.3 * SR)) / SR
    p = (gain * np.exp(-t * 6.0) * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    for at in times_s:
        i = int(at * SR)
        x[i:i + p.size] += p
    return x


def held(t0_s, t1_s, *, total_s, gain=0.3):
    """One tone with no decay, so a loop of it is at full level at any instant."""
    x = np.zeros(int(total_s * SR), dtype=np.float32)
    i, j = int(t0_s * SR), int(t1_s * SR)
    t = np.arange(j - i) / SR
    x[i:j] = (gain * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    return x


def envelope(out, at, *, hz=TONE_HZ, n=96):
    """The tone's level around sample ``at`` of the machine's output."""
    x = out[max(0, at - n // 2):at + n // 2].mean(axis=1)
    ref = np.exp(-2j * np.pi * hz * np.arange(x.size) / SR)
    return 2.0 * float(np.abs((x * ref).mean())) if x.size else 0.0


def attack_after(out, line_s, *, hz=TONE_HZ):
    """Seconds from this line to where the tone first reaches half its level."""
    x = out[int((line_s - 0.02) * SR):int((line_s + 0.5) * SR)].mean(axis=1)
    ref = np.exp(-2j * np.pi * hz * np.arange(x.size) / SR)
    env = np.abs(np.convolve(x * ref, np.ones(96) / 96, mode="same"))
    return np.flatnonzero(env > 0.5 * env.max())[0] / SR - 0.02


def tone_db(out, t0_s, t1_s):
    """Loudest the test tone gets in this stretch of the machine's output.

    Read in short windows: over a long one, notes half a period out of phase with
    each other cancel, and a loud loop reads as silence.
    """
    x = out[int(t0_s * SR):int(t1_s * SR)].mean(axis=1)
    n = 1024
    if x.size < n:
        return -200.0
    x = x[:x.size - x.size % n].reshape(-1, n)
    ref = np.exp(-2j * np.pi * TONE_HZ * np.arange(n) / SR)
    amp = 2.0 * float(np.abs((x * ref).mean(axis=1)).max())
    return 20.0 * np.log10(max(amp, 1e-10))


def rows_of(rec: Path):
    lines = (rec / "decisions.ndjson").read_text().splitlines()
    return [r for r in map(json.loads, lines) if isinstance(r.get("bar"), int)]


def cue_when(sock, script):
    """A thread that plays ``[(bar or None, text)]`` into the live socket.

    The session cap is wall-clock and the fake device runs fast, so a test ends
    its take with a ``STOP`` line when the bars it needs have been planned.  The
    fake device waits for a line that is due (:class:`PlannerGate`), so that on a
    busy machine a cue is still said on the bar the test means.
    """
    replies = {}
    due = [bar for bar, _ in script]     # a thread left over must not touch the next
    CUES_DUE[0] = due

    def run():
        if not wait_until(lambda: Path(sock).exists(), timeout=30.0):
            due.clear()
            return
        for bar, text in script:
            if bar is not None:
                def reached(b=bar):
                    try:
                        got = realtime.send_status(socket_path=sock).get("bar")
                    except realtime.RealtimeError:
                        return False
                    return got is not None and got >= b
                wait_until(reached, timeout=30.0, poll=0.005)
            try:
                if text == STOP:
                    realtime.send_stop(socket_path=sock)
                    due.clear()
                    return
                replies[text] = realtime.send_cue(text, socket_path=sock)
            except realtime.RealtimeError as e:
                replies[text] = {"ok": False, "error": str(e)}
            due.pop(0)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, replies


@pytest.fixture
def live_env(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    return tmp_path


class PlannerGate:
    """Holds the fake device while it would outrun the threads under test: a bar
    being planned, a tempo not yet picked up, mic audio unread, a cue that is due.

    A fast device outruns them on a busy machine: at a load of 100 a bar takes a
    second, the bars land in the past, and the output goes silent while the rows
    still say "sounding".  Analysis falls behind the same way, and a take then
    closes tens of seconds after it should.  A real device at real time has
    seconds to spare, so this removes a race the tests made up.
    """

    def __init__(self):
        self._planning = threading.Event()
        self.late_s = 0.0
        self.stream_samples = lambda: 0
        self._fed = 0
        self._bar = -1
        self._patches = []

    def busy(self):
        due = CUES_DUE[0][0] if CUES_DUE[0] else -1
        return ((due is None or 0 <= due <= self._bar) or self._planning.is_set()
                or self.stream_samples() - self._fed > MAX_UNREAD_S * SR)

    def __enter__(self):
        gate = self
        real_commit = realtime.MachineClock.commit
        real_write_at = realtime.TimelineBuffer.write_at
        real_feed = lp.Looper.feed
        real_take_tempo = lp.Looper.take_tempo

        def feed(looper, block):
            gate._fed += block.size
            return real_feed(looper, block)

        def commit(clock, *a, **kw):
            gate._planning.set()
            got = real_commit(clock, *a, **kw)
            gate._bar = got[0]
            return got

        def take_tempo(looper):
            tempo = real_take_tempo(looper)
            if tempo is not None:
                gate._planning.set()
            return tempo

        def write_at(timeline, at, audio, **kw):
            gate.late_s = max(gate.late_s, (timeline._play - int(at)) / SR)
            gate._planning.clear()
            return real_write_at(timeline, at, audio, **kw)

        self._patches = [mock.patch.object(realtime.MachineClock, "commit", commit),
                         mock.patch.object(realtime.TimelineBuffer, "write_at", write_at),
                         mock.patch.object(lp.Looper, "feed", feed),
                         mock.patch.object(lp.Looper, "take_tempo", take_tempo)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._planning.clear()
        return False


def run_lead(human, *, rec, sock, max_s=SAFETY_S, **kw):
    device = kw.pop("device_kw", {})
    gate = PlannerGate()
    fake = fake_audio(human, rate_x=8.0, warmup_s=4.0, hold=gate.busy, **device)
    gate.stream_samples = lambda: (fake.stream.blocks * fake.stream._n
                                   if fake.stream is not None else 0)
    lead_kw = {"prompt": "lofi"} if kw.get("tap_start") else {
        "session": compose(PromptSpec(styles=["lofi"], bpm=120.0, key="C major",
                                      movements=1, mood=[], name="looper"), seed=7)}
    with patched_module("sounddevice", fake), gate:
        realtime.lead(seed=0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec), socket_path=sock, status=False,
                      max_minutes=max_s / 60.0, **lead_kw, **kw)
    assert gate.late_s <= MAX_LATE_S, (
        f"the planner fell behind the fake device by {gate.late_s:.2f} s")
    return fake.stream.output


# ------------------------------------------------- with the band's tempo running

@pytest.mark.usefixtures("fake_device_consent")
class TestRunningTempo:
    """120 bpm: a bar is 2 s.  The phrase starts on bar 3, at 6 s, and is 2 bars."""

    PHRASE = [6.02 + 0.5 * i for i in range(7)]

    def test_make_mute_bring_back_drop(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopa")
        th, replies = cue_when(sock, [
            (None, "make a loop called bass"),
            (9, "mute the bass loop"),
            (12, "unmute the bass loop"),
            (15, "drop the bass loop"),
            (20, STOP),
        ])
        out = run_lead(notes(self.PHRASE, total_s=80.0), rec=rec, sock=sock)
        th.join(timeout=10.0)
        assert replies["make a loop called bass"]["ok"] is True
        rows = rows_of(rec)
        loops = [r for r in rows if "loop" in r]
        made = [e for r in loops for e in r["loop"]["events"]]
        assert any("'bass' made" in e and "2 bar" in e for e in made), made
        assert not any("speakers" in e for e in made), made

        # Comes in at the top of its own cycle: 2-bar loop anchored on bar 3.
        by_bar = {r["bar"]: r for r in rows}
        first = next(r["bar"] for r in rows if any(
            x["sounding"] for x in r.get("loop", {}).get("layers", [])))
        assert (first - 3) % 2 == 0, first
        assert rows[first]["loop"]["layers"][0]["anchor_bar"] == 3

        # The tone is in the output with the human long gone, on the bar it was
        # played against, and absent before anyone played it.
        assert tone_db(out, 0.0, 5.5) < -35.0     # only the band, faintly
        t = by_bar[first]["t_s"]
        heard = tone_db(out, t, t + 4.0)
        assert heard > -20.0, heard

        # Bars are planned ahead, so each change lands on a later bar than asked.
        muted = next(r["bar"] for r in loops
                     if r["bar"] >= 9 and not r["loop"]["layers"][0]["sounding"])
        back = next(r["bar"] for r in loops
                    if r["bar"] > muted and r["loop"]["layers"]
                    and r["loop"]["layers"][0]["sounding"])
        tm = by_bar[muted]["t_s"] + 0.01
        assert tone_db(out, tm, by_bar[back]["t_s"]) < heard - 20.0
        tb = by_bar[back]["t_s"]
        assert tone_db(out, tb, tb + 2.0) > heard - 3.0
        gone = next(r["bar"] for r in rows
                    if r["bar"] > back and not r.get("loop", {}).get("layers"))
        tg = by_bar[gone]["t_s"] + 0.01
        assert tone_db(out, tg, tg + 4.0) < heard - 20.0

    def test_the_join_lands_on_the_grid_with_no_gap_between_passes(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopb")
        th, _ = cue_when(sock, [(None, "make a 2 bar loop"), (14, STOP)])
        out = run_lead(notes(self.PHRASE, total_s=80.0), rec=rec, sock=sock)
        th.join(timeout=10.0)
        rows = rows_of(rec)
        first = next(r for r in rows if any(
            x["sounding"] for x in r.get("loop", {}).get("layers", [])))
        # Each of two passes starts with the phrase's first note right on its
        # bar line, within the 10 ms the round trip is quantized to.
        for pass_i in range(2):
            line = first["t_s"] + pass_i * 4.0
            x = out[int((line - 0.02) * SR):int((line + 0.05) * SR)].mean(axis=1)
            t = np.arange(x.size) / SR
            ref = np.exp(-2j * np.pi * TONE_HZ * t)
            env = np.abs(np.convolve(x * ref, np.ones(96) / 96, mode="same"))
            at = np.flatnonzero(env > 0.5 * env.max())[0] / SR - 0.02
            assert abs(at) < 0.015, at

    def test_a_loop_holds_the_tempo_and_says_so(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopc")
        th, _ = cue_when(sock, [(None, "make a 2 bar loop"), (11, "faster"), (24, STOP)])
        run_lead(notes(self.PHRASE, total_s=80.0), rec=rec, sock=sock)
        th.join(timeout=10.0)
        rows = rows_of(rec)
        held = [r for r in rows if r.get("loop", {}).get("layers")]
        assert held
        assert all(r["bpm"] == pytest.approx(120.0, abs=0.01) for r in held)
        notes_ = [n for r in rows for n in (r["cue_effects"] or [])]
        assert any("NOT applied yet" in n for n in notes_), notes_

    def test_no_new_audio_file_appears(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopd")
        th, _ = cue_when(sock, [(None, "make a 2 bar loop"), (10, STOP)])
        run_lead(notes(self.PHRASE, total_s=80.0), rec=rec, sock=sock)
        th.join(timeout=10.0)
        assert {p.name for p in rec.iterdir() if p.suffix == ".wav"} == {
            "human.wav", "machine.wav"}


# --------------------------------------------------- with no tempo running at all

class TestLoopSetsTheTempo:
    BPM = 96.0

    def human(self):
        beat = 60.0 / self.BPM
        return notes([4.0 + i * beat for i in range(17)], total_s=120.0)

    def test_the_band_follows_the_loop(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loope")
        th, replies = cue_when(sock, [(None, "make a loop of this"), (4, STOP)])
        out = run_lead(self.human(), rec=rec, sock=sock, tap_start=True)
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        rows = rows_of(rec)
        assert rows, "the loop never started the band"
        assert all(r["bpm"] == pytest.approx(self.BPM, rel=0.015) for r in rows)
        assert rows[0]["t_s"] > 14.0                 # after the phrase, not during
        assert rows[0]["entry"] == "drums"
        assert rows[0]["loop"]["layers"][0]["sounding"] is True
        # No band had played, so nothing can be known about speakers in the mic.
        assert rows[0]["loop"]["layers"][0]["band_in_mic"] is False
        # bar 0 is the top of the loop, so its first note is on the bar line
        t0 = rows[0]["t_s"]
        assert tone_db(out, t0, t0 + 0.3) > -20.0

    def test_a_take_with_no_tempo_leaves_it_waiting_for_a_count(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopf")
        th, _ = cue_when(sock, [(None, "make a loop of this")])
        human = notes([4.0], total_s=60.0)
        out = run_lead(human, rec=rec, sock=sock, max_s=8.0, tap_start=True)
        th.join(timeout=10.0)
        assert rows_of(rec) == []
        assert not out.any()      # silent, and it stayed that way


# ------------------------------------------- what lead does with the loop's audio

class TestLeadWiring:
    def test_the_silence_ending_fades_the_loop_with_the_band(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopn")
        beat = 60.0 / 96.0
        human = notes([4.0 + i * beat for i in range(17)], total_s=120.0)
        th, replies = cue_when(sock, [(None, "make a loop of this")])
        out = run_lead(human, rec=rec, sock=sock, tap_start=True)
        th.join(timeout=10.0)
        rows = rows_of(rec)
        assert rows, replies
        quietest = min(rows, key=lambda r: r["gain_db"])
        assert rows[0]["gain_db"] - quietest["gain_db"] > 30.0
        assert quietest["loop"]["layers"][0]["sounding"] is True
        bar_s = 4.0 * beat
        full = tone_db(out, rows[0]["t_s"], rows[0]["t_s"] + bar_s)
        faded = tone_db(out, quietest["t_s"], quietest["t_s"] + bar_s)
        assert full > -20.0, full
        assert faded < full - 20.0, (full, faded)

    def test_stopping_fades_a_loop_rather_than_cutting_it(self, live_env):
        rec, sock = live_env / "rec", short_socket_path("loopo")
        th, _ = cue_when(sock, [(None, "make a 1 bar loop"), (12, STOP)])
        out = run_lead(held(5.98, 9.0, total_s=80.0), rec=rec, sock=sock)
        th.join(timeout=10.0)
        rows = rows_of(rec)
        assert rows[-1]["loop"]["layers"][0]["sounding"] is True
        # The last planned bar runs out here, in the middle of a held note.
        end = int(round((rows[-1]["t_s"] + 2.0) * SR))
        steady = envelope(out, end - 2000)
        assert steady > 0.1, steady
        assert envelope(out, end + 64) > 0.5 * steady        # still on its way down
        # Well down before the fade's last sample: full level up to there is a cut.
        assert envelope(out, end + lp.DECLICK_SAMPLES * 3 // 4) < 0.6 * steady
        assert envelope(out, end + lp.DECLICK_SAMPLES + 64) < 0.2 * steady

    def test_a_later_take_lands_on_a_tap_start_grid(self, live_env, monkeypatch):
        """The band sounds early by the measured round trip, so that is the amount
        a take played to it comes back late.  A different figure from the stream's
        own, so reading the wrong one puts the phrase off its bar."""
        monkeypatch.setattr("plume.calibrate.load_roundtrip_ms",
                            lambda *_a: 300.0)
        rec, sock = live_env / "rec", short_socket_path("loopp")
        beat = 60.0 / 96.0
        bar_s, hz = 4.0 * beat, 2093.0
        phrase = 34.0                               # a bar line: see below
        human = (notes([4.0 + i * beat for i in range(17)], total_s=120.0)
                 + notes([phrase + i * beat for i in range(8)], total_s=120.0, hz=hz))
        th, _ = cue_when(sock, [(None, "make a loop of this"),
                                (3, "make a 2 bar loop called top"), (13, STOP)])
        out = run_lead(human, rec=rec, sock=sock, tap_start=True)
        th.join(timeout=10.0)
        rows = rows_of(rec)
        by_bar = {r["bar"]: r for r in rows}
        top = next(x for r in rows for x in r.get("loop", {}).get("layers", [])
                   if x["name"] == "top")
        # The loop sets the grid, so the phrase is on it only if the band started
        # a whole number of bars before it, which this tempo and first bar give.
        anchor = (phrase - rows[0]["t_s"]) / bar_s
        assert anchor == pytest.approx(round(anchor), abs=0.01), rows[0]["t_s"]
        assert top["anchor_bar"] == round(anchor)
        heard = int(round(0.3 * SR))
        first = next(r["bar"] for r in rows if any(
            x["name"] == "top" and x["sounding"]
            for x in r.get("loop", {}).get("layers", [])))
        line = (by_bar[first]["t_s"] * SR - heard) / SR
        assert abs(attack_after(out, line, hz=hz)) < 0.015


# ----------------------------------------------------------------- bleed warning

class TestBleed:
    def run(self, live_env, sock_name, **device):
        rec, sock = live_env / "rec", short_socket_path(sock_name)
        th, _ = cue_when(sock, [(4, "make a 2 bar loop"), (14, STOP)])
        run_lead(notes([10.02 + 0.5 * i for i in range(7)], total_s=80.0),
                 rec=rec, sock=sock, device_kw=device)
        th.join(timeout=10.0)
        return [x for r in rows_of(rec)
                for x in r.get("loop", {}).get("layers", [])]

    def test_speakers_in_the_mic_are_called_out(self, live_env):
        layers = self.run(live_env, "loopg", bleed_db=-6.0)
        assert layers and all(x["band_in_mic"] for x in layers)

    def test_headphones_are_not(self, live_env):
        layers = self.run(live_env, "looph")
        assert layers and not any(x["band_in_mic"] for x in layers)


# ---------------------------------------------------- the socket and the verb

class TestCueSurface:
    def server(self, name):
        from plume.looper import Looper
        looper = Looper(SR)
        srv = realtime.CueServer(short_socket_path(name), kind="lead")
        srv.intercept = looper.handle_cue
        return srv.start(), looper

    def test_a_loop_cue_is_answered_at_once_and_never_queued(self):
        srv, looper = self.server("loopi")
        try:
            r = realtime.send_cue("make a loop of this", socket_path=srv.path)
            assert r["ok"] and "recording" in r["said"]
            assert looper.capturing
            assert srv.drain() == []
        finally:
            srv.close()

    def test_everything_else_still_queues_for_the_band(self):
        srv, looper = self.server("loopj")
        try:
            r = realtime.send_cue("drop the bass", socket_path=srv.path)
            assert r["queued"] == 1
            assert [c["text"] for c in srv.drain()] == ["drop the bass"]
            assert not looper.capturing
        finally:
            srv.close()

    def test_a_refusal_reaches_the_sender(self):
        srv, _ = self.server("loopk")
        try:
            with pytest.raises(realtime.RealtimeError, match="no loop"):
                realtime.send_cue("mute the bass loop", socket_path=srv.path)
        finally:
            srv.close()

    def test_the_sideman_still_takes_no_cues(self):
        srv = realtime.CueServer(short_socket_path("loopl"), kind="jam",
                                 accepts_cues=False).start()
        try:
            with pytest.raises(realtime.RealtimeError, match="no cues"):
                realtime.send_cue("make a loop of this", socket_path=srv.path)
        finally:
            srv.close()

    def test_the_verb_reports_what_the_looper_said(self, capsys):
        srv, _ = self.server("loopm")
        try:
            assert cli.main(["cue", "make a 4 bar loop called bass",
                             "--socket", srv.path, "--json"]) == 0
            got = json.loads(capsys.readouterr().out)
            assert got["loop"]["name"] == "bass" and got["loop"]["bars"] == 4
            assert "parsed" not in got
            assert cli.main(["cue", "cancel the loop", "--socket", srv.path]) == 0
            assert "cancelled" in capsys.readouterr().out
            assert cli.main(["cue", "cancel the loop", "--socket", srv.path]) != 0
        finally:
            srv.close()
