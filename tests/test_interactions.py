"""The modes together: direct input, MIDI, MIDI clock, loops, voice, consent, endings.

Each of these has its own tests alone.  These are the seams between them.  Everything
runs on the FAKE device and a FAKE port (tests/conftest.py), so nothing here makes a
sound or says anything about how a combination feels to play.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import FakeMidiMessage as Msg
from conftest import FakeMido, fake_audio, patched_module, short_socket_path, signal_self, wait_until
from test_looper_live import PlannerGate, cue_when, held, notes, rows_of, tone_db
from test_midi import Player, StuckMido, recorded

from plume import bandentry, cli, realtime, voice
from plume import looper as lp
from plume import midi as midi_mod
from plume import mind as mind_mod
from plume import roles as roles_mod
from plume.compose import PromptSpec, compose
from plume.listen import harmony as harmony_mod
from plume.listen.beat import BeatTracker
from plume.midi import MidiMasker
from plume.watch import Listener

SR = realtime.SR
SAFETY_MIN = 2.0     # wall clock; only a wedged take ever reaches it
C, F, G = (60, 64, 67), (65, 69, 72), (67, 71, 74)
RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]
#: A tone at another frequency than looper_live's, for a second input.
OTHER_HZ = 4409.0


class Script(Player):
    """Runs a script on the fake device's pump and presses nothing by itself."""

    def advance(self, t: float) -> None:
        self.script(t)


def clock_script(mido, bpm, *, t0=0.5, start=True):
    """A MIDI clock at this tempo, driven from stream time."""
    state = {"n": 0, "started": False}

    def run(t):
        port = mido.port
        if port is None or port.closed or t < t0:
            return
        if start and not state["started"]:
            port.send(Msg("start"))
            state["started"] = True
        while t >= t0 + state["n"] * 60.0 / bpm / 24.0:
            port.send(Msg("clock"))
            state["n"] += 1
    return run


def presses_script(mido, times, keys=C):
    """Key presses at these stream times."""
    left = list(times)

    def run(t):
        port = mido.port
        if port is None or port.closed:
            return
        while left and t >= left[0]:
            left.pop(0)
            for n in keys:
                port.send(Msg("note_on", note=n, velocity=90))
                port.send(Msg("note_off", note=n, velocity=0))
    return run


def both(*scripts):
    def run(t):
        for s in scripts:
            s(t)
    return run


def the_session():
    return compose(PromptSpec(styles=["lofi"], bpm=120.0, key="C major", movements=1,
                              mood=[], name="seams"), seed=7)


class Take:
    """One real transport over the fake device, and what it left behind."""

    def __init__(self, tmp_path, name="take"):
        self.rec = tmp_path / name
        self.sock = short_socket_path(name[:6])
        self.mido = FakeMido()
        self.sd = None
        self.out = None

    def stream_t(self) -> float:
        s = self.sd.stream if self.sd is not None else None
        return 0.0 if s is None else s.blocks * 256 / SR

    def run(self, verb, *, human=None, rate=8.0, device=None, script=None, stop_t=None,
            on_stream=None, mido_mod=True, **kw):
        device = dict(device or {})
        gate = PlannerGate() if verb == "lead" else None
        self.sd = fake_audio(human, rate_x=rate, warmup_s=device.pop("warmup_s", 1.0),
                             hold=None if gate is None else gate.busy, **device)
        if gate is not None:
            gate.stream_samples = lambda: (self.sd.stream.blocks * self.sd.stream._n
                                           if self.sd.stream is not None else 0)
        if script is not None:
            Script(self.sd, self.mido, script=lambda t: script(t))
        stopper = None
        if stop_t is not None:
            stopper = threading.Thread(target=self._stop_at, args=(stop_t, on_stream),
                                       daemon=True)
            stopper.start()
        kw.setdefault("max_minutes", SAFETY_MIN)
        with patched_module("sounddevice", self.sd), patched_module("mido", self.mido):
            if gate is None:
                self._go(verb, kw)
            else:
                with gate:
                    self._go(verb, kw)
        if stopper is not None:
            stopper.join(timeout=10.0)
        self.out = self.sd.stream.output
        return self

    def _go(self, verb, kw):
        common = {"seed": 0, "record_dir": str(self.rec), "socket_path": self.sock}
        status = kw.pop("status", False)
        if verb == "jam":
            realtime.jam(style="lofi", params=mind_mod.MindParams(), **common, **kw)
            return
        if not kw.get("tap_start"):
            kw["session"] = the_session()
        else:
            kw["prompt"] = "lofi"
        realtime.lead(role=roles_mod.RoleParams(role="lead"), status=status, **common, **kw)

    def _stop_at(self, t, on_stream):
        wait_until(lambda: self.stream_t() >= t, timeout=90.0)
        (on_stream or self.plume_stop)()

    def plume_stop(self):
        """The verb, in-process: a session in this process only gets the socket rung."""
        self.stop_rc = cli.main(["stop"])

    @property
    def ending(self):
        return json.loads((self.rec / "ending.json").read_text())

    def rows(self):
        return rows_of(self.rec)

    def manifest(self):
        return json.loads((self.rec / "session.json").read_text())

    def assert_ended_cleanly(self, reason):
        assert self.ending["reason"] == reason
        s = self.sd.stream
        assert s.blocks > 0 and s.closed
        assert all(p.closed for p in self.mido.ports)
        assert not Path(self.sock).exists()
        assert realtime.list_sessions() == []
        alive = [t.name for t in threading.enumerate()
                 if t.name in ("plume-planner", "plume-analysis")]
        assert not alive, alive


@pytest.fixture
def env(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    return tmp_path


def decoy(total_s, *, bpm=80.0, hz=370.0, gain=0.5):
    """Loud, even plucks of F#: a count and a key that are not what the keys say."""
    return notes([1.0 + i * 60.0 / bpm for i in range(int((total_s - 2.0) * bpm / 60.0))],
                 total_s=total_s, gain=gain, hz=hz)


# ----------------------------------------------------------- --direct with MIDI

@pytest.mark.usefixtures("fake_device_consent")
class TestDirectWithMidi:
    """The earlier rule: MIDI wins over --direct for what the band reads as notes."""

    SPEED = 2.0   # the band must lock and join; faster than this is CPU on a busy box

    def play(self, mido):
        beat, sequence, state = 60.0 / 110.0, (C, F, G, C), {"i": 0, "up": None}

        def run(t):
            port = mido.port
            if port is None or port.closed:
                return
            if state["up"] is not None and t >= state["up"][0]:
                for n in state["up"][1]:
                    port.send(Msg("note_off", note=n, velocity=0))
                state["up"] = None
            if t >= 1.0 + state["i"] * beat:
                keys = sequence[(state["i"] // 4) % 4]
                for n in keys:
                    port.send(Msg("note_on", note=n, velocity=90))
                state["up"], state["i"] = (t + 0.8 * beat, keys), state["i"] + 1
        return run

    @pytest.mark.parametrize("verb,extra", [("lead", {"tap_start": True}), ("jam", {})],
                             ids=["lead", "jam"])
    def test_the_band_reads_the_keys_and_not_the_direct_input(self, env, verb, extra):
        take = Take(env)
        take.run(verb, human=decoy(80.0), rate=self.SPEED, direct=True, midi="auto",
                 script=self.play(take.mido), stop_t=60.0, **extra)
        rows = [r for r in take.rows() if r.get("bpm")]
        assert rows
        assert rows[-1]["bpm"] == pytest.approx(110.0, rel=0.03)
        # I, IV, V reads as either; the input's F# drone would read as neither.
        assert rows[-1]["key"] in ("C major", "A minor")
        assert take.manifest()["midi_port"] == "Fake Keys"
        assert take.manifest()["input"]["source"] == "direct"
        assert (take.rec / "midi.ndjson").exists()

    def test_lead_does_not_treat_a_midi_take_as_a_level_from_the_input(self, env, monkeypatch):
        """A direct input's level says who is playing.  With --midi that level comes
        from an input the band is no longer reading, so the direct rules must be off."""
        seen, real = [], bandentry.BandEntry

        class Spy(real):
            def __init__(self, *a, **kw):
                seen.append(kw)
                super().__init__(*a, **kw)
        monkeypatch.setattr(bandentry, "BandEntry", Spy)
        for name, kw in (("direct", {}), ("both", {"midi": "auto"})):
            take = Take(env, name)
            take.run("lead", tap_start=True, direct=True, stop_t=2.0, rate=12.0, **kw)
        assert "after_first_heard_bars" in seen[0]
        assert "after_first_heard_bars" not in seen[1]

    def test_the_recording_keeps_the_input_it_was_told_to_read(self, env):
        """MIDI takes over listening, not recording: human.wav is still the guitar."""
        guitar = notes([0.5 + i * 0.5 for i in range(30)], total_s=20.0)
        take = Take(env)
        take.run("lead", human=guitar, rate=12.0, direct=True, midi="auto",
                 input_device="audio interface", input_channel=2,
                 device={"devices": RIG, "human_channel": 1,
                         "inputs": {0: 0.5 * np.ones(int(20.0 * SR), dtype=np.float32)}},
                 stop_t=6.0)
        y, _ = sf.read(str(take.rec / "human.wav"), dtype="float32")
        assert 0.1 < float(np.abs(y).max())
        assert float(np.abs(y[:int(0.4 * SR)]).max()) < 0.01   # not the DC decoy

    def test_watch_trusts_the_notes_even_when_the_band_is_in_the_direct_input(self, tmp_path):
        rec = recorded(tmp_path,
                       [{"t_s": 1.0 + i * 0.5, "type": "on", "note": 60, "velocity": 90,
                         "channel": 0} for i in range(20)], mic=0.3)
        n = int(2.0 * SR) * 24
        band = 0.3 * np.sin(2 * np.pi * 330.0 * np.arange(n) / SR)
        sf.write(str(rec / "machine.wav"), band.astype("float32"), SR)
        (rec / "session.json").write_text(json.dumps(
            {"input": {"source": "direct"}, "midi_port": "Fake Keys"}))
        got = Listener(rec, every=4).poll()
        assert got and all(g["input"] == "midi" and g["you_source"] == "midi" for g in got)
        assert got[0]["you_trusted"] == ["you_db", "you_busy", "you_notes"]
        assert "C" in got[0]["you_notes"]


# ---------------------------------------------------- the looper with MIDI and clock

class TestLoopOnAMidiTake:
    PHRASE = [6.02 + 0.5 * i for i in range(7)]

    def test_a_loop_is_cut_from_the_mic_while_the_band_reads_the_keys(self, env):
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a 2 bar loop"),
                                           (12, "list loops"), (14, "(stop)")])
        take.run("lead", human=notes(self.PHRASE, total_s=80.0), midi="auto", device=_slow())
        th.join(timeout=10.0)
        assert replies["make a 2 bar loop"]["ok"] is True
        assert replies["list loops"]["ok"] is True
        rows = [r for r in take.rows() if r.get("loop", {}).get("layers")]
        assert rows and rows[-1]["loop"]["layers"][0]["sounding"] is True
        # No reading of the band in the mic exists on a MIDI take, so no warning.
        assert rows[-1]["loop"]["layers"][0]["band_in_mic"] is False
        # The phrase has ended by 10 s; after that the tone can only be the loop.
        assert tone_db(take.out, rows[-1]["t_s"] - 4.0, rows[-1]["t_s"]) > -30.0

    def test_keys_alone_never_make_a_loop(self, env):
        take = Take(env)
        presses = [6.0 + 0.5 * i for i in range(10)]
        th, replies = cue_when(take.sock, [(None, "make a 1 bar loop"), (8, "that's the loop"),
                                           (11, "(stop)")])
        take.run("lead", midi="auto", script=presses_script(take.mido, presses), device=_slow())
        th.join(timeout=10.0)
        assert replies["make a 1 bar loop"]["ok"] is True
        assert not [r for r in take.rows() if r.get("loop", {}).get("layers")]

    def test_a_masker_with_no_room_says_it_has_no_reading(self):
        assert MidiMasker().coupling_measured is False


class TestLoopWithAClock:
    PHRASE = [6.02 + 0.5 * i for i in range(7)]

    def test_a_held_loop_keeps_the_band_off_a_clock_at_another_tempo(self, env, monkeypatch):
        beats = []
        real = BeatTracker.clock_beat

        def spy(self, *a, **kw):
            beats.append(a)
            return real(self, *a, **kw)
        monkeypatch.setattr(BeatTracker, "clock_beat", spy)
        take = Take(env)
        th, _ = cue_when(take.sock, [(None, "make a 2 bar loop"), (11, "faster"), (22, "(stop)")])
        take.run("lead", human=notes(self.PHRASE, total_s=80.0), midi="auto", midi_clock=True,
                 script=clock_script(take.mido, 90.0), device=_slow())
        th.join(timeout=10.0)
        assert beats, "the clock never reached the tracker, so this proves nothing"
        rows = take.rows()
        kept = [r for r in rows if r.get("loop", {}).get("layers")]
        assert kept
        assert all(r["bpm"] == pytest.approx(120.0, abs=0.01) for r in kept)
        said = [n for r in rows for n in (r["cue_effects"] or [])]
        assert any("NOT applied yet" in n for n in said), said

    def test_with_no_band_yet_the_loop_sets_the_tempo_and_the_clock_does_not(self, env,
                                                                             monkeypatch):
        beats = []
        real = BeatTracker.clock_beat

        def spy(self, *a, **kw):
            beats.append(a)
            return real(self, *a, **kw)
        monkeypatch.setattr(BeatTracker, "clock_beat", spy)
        bpm = 96.0
        human = notes([4.0 + i * 60.0 / bpm for i in range(17)], total_s=120.0)
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a loop of this"), (4, "(stop)")])
        take.run("lead", human=human, midi="auto", midi_clock=True, tap_start=True,
                 script=clock_script(take.mido, 140.0), device=_slow())
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        rows = take.rows()
        assert beats, "the clock never reached the tracker, so this proves nothing"
        assert rows, "the loop never started the band"
        assert all(r["bpm"] == pytest.approx(bpm, rel=0.015) for r in rows)

    def test_a_clock_alone_starts_nothing(self, env):
        take = Take(env)
        take.run("lead", midi="auto", midi_clock=True, tap_start=True,
                 script=clock_script(take.mido, 120.0), stop_t=14.0)
        assert take.rows() == []
        assert not take.out.any()

    def test_keys_pressed_during_a_take_are_not_a_count(self, env):
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a loop of this")])

        def go(t):
            # Pressed only once the take is open, so a count here means the guard is off.
            if replies.get("make a loop of this"):
                press(t)
        press = presses_script(take.mido, [3.0 + 0.5 * i for i in range(8)])
        take.run("lead", midi="auto", tap_start=True, script=go, stop_t=12.0, device=_slow())
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        assert take.rows() == []

    def test_the_same_keys_with_no_take_are_a_count(self, env):
        """The control for the test above: these presses do start a band."""
        take = Take(env)
        press = presses_script(take.mido, [3.0 + 0.5 * i for i in range(8)])
        take.run("lead", midi="auto", tap_start=True, script=press, stop_t=12.0, device=_slow())
        assert take.rows()


def _slow():
    """As looper_live: real time first, so the socket is up before the phrase."""
    return {"warmup_s": 4.0}


# ---------------------------------------------------- a loop cue with the band running

class TestLoopCues:
    PHRASE = [6.02 + 0.5 * i for i in range(7)]

    def test_a_typed_cue_loops_the_guitar_channel_and_not_the_voice(self, env):
        guitar = notes(self.PHRASE, total_s=80.0)
        spoken = notes(self.PHRASE, total_s=80.0, hz=OTHER_HZ)
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a 2 bar loop"), (14, "(stop)")])
        take.run("lead", human=guitar, direct=True, input_device="audio interface", input_channel=2,
                 voice_channel=3,
                 device={"devices": RIG, "human_channel": 1, "inputs": {2: spoken},
                         "warmup_s": 4.0})
        th.join(timeout=10.0)
        assert replies["make a 2 bar loop"]["ok"] is True
        rows = [r for r in take.rows() if r.get("loop", {}).get("layers")]
        assert rows
        t1 = rows[-1]["t_s"]
        looped = tone_db(take.out, t1 - 4.0, t1)
        assert looped > -30.0
        x = take.out[int((t1 - 4.0) * SR):int(t1 * SR)].mean(axis=1)
        n = x.size - x.size % 1024
        ref = np.exp(-2j * np.pi * OTHER_HZ * np.arange(1024) / SR)
        other = 2.0 * float(np.abs((x[:n].reshape(-1, 1024) * ref).mean(axis=1)).max())
        # Well under the guitar: what is left at this frequency is the band.
        assert 20.0 * np.log10(max(other, 1e-10)) < looped - 10.0
        assert (take.rec / "voice.wav").exists() and (take.rec / "human.wav").exists()

    @pytest.mark.parametrize("said", [
        "Plume, stop the loop.", "plume stop the bass loop", "plume, restart the loop",
        "plume please stop that layer", "plume, more loops", "plume, finish the loop",
        "Plume. Stop the loop.",
    ])
    def test_a_spoken_loop_cue_does_not_end_or_restart_the_take(self, said):
        assert voice.command_in(said) is None

    @pytest.mark.parametrize("said,means", [
        ("Plume, stop.", "stop"), ("plume, restart", "restart"),
        ("The loop was nice. Plume, stop.", "stop"),
        ("Plume, stop. The loop was nice.", "stop"),
        ("Plume. Stop.", "stop"), ("plume, stop the loop. Plume, stop.", "stop"),
    ])
    def test_a_real_command_still_gets_through(self, said, means):
        assert voice.command_in(said) == means

    def test_voice_keeps_the_band_going_through_a_spoken_loop_cue(self, tmp_path):
        """The real run loop: only the second sentence may stop the band."""
        said = iter(["plume, stop the loop", "plume, stop"])
        heard, lines = [], []

        class Proc:
            def __init__(self, cmd, **kw):
                self.cmd, self.env, self.ended = cmd, kw["env"], None

            def poll(self):
                return self.ended

            def send_signal(self, _sig):
                heard.append(len(lines))
                self.ended = 0

            def wait(self, timeout=None):  # noqa: ARG002
                return 0

            def kill(self):
                self.ended = -9

        class Feeder(voice.VoiceLoop):
            def tail(self):
                return np.full(16000, 0.05, dtype=np.float32)

            def step(self):
                self._last_fired.clear()
                got = super().step()
                lines.append(got)
                return got
        real, voice.VoiceLoop = voice.VoiceLoop, Feeder
        try:
            voice.run(prompt="lofi", base=tmp_path, write=lambda _s: None, poll_s=0.0,
                      transcribe=lambda _a: next(said), popen=Proc)
        finally:
            voice.VoiceLoop = real
        assert lines == [None, "stop"]
        assert heard == [1]    # the stop arrived while the second sentence was acted on

    def test_a_voice_band_gets_every_route_flag_and_its_consent(self, tmp_path, monkeypatch):
        """Its band is `plume lead --tap-start`, where typed loop cues then land."""
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        seen = {}

        class Proc:
            def __init__(self, cmd, **kw):
                seen.update(cmd=cmd, env=kw["env"])

            def poll(self):
                return 0

        extra = ["--direct", "--voice-channel", "2", "--midi", "Fake Keys", "--midi-clock"]
        band = voice.Band(prompt="lofi", base=tmp_path, extra_args=extra, popen=Proc,
                          write=lambda _s: None)
        band.start()
        assert seen["cmd"][seen["cmd"].index("lead") + 1] == "--tap-start"
        cut = seen["cmd"].index("--")
        assert seen["cmd"][cut - len(extra):cut] == extra and seen["cmd"][cut + 1:] == ["lofi"]
        assert seen["env"][realtime.AUDIO_CONSENT_ENV] == "1"

    def test_jam_refuses_a_loop_cue_out_loud(self, env, capsys):
        """Jam has no looper.  Telling the player it worked would be the worst answer."""
        take = Take(env)
        got = {}

        def ask():
            got["rc"] = cli.main(["cue", "make a loop of this", "--socket", take.sock])
            take.plume_stop()
        take.run("jam", midi="auto", stop_t=3.0, on_stream=ask, rate=12.0)
        assert got["rc"] != 0
        captured = capsys.readouterr()
        assert "loop:" not in captured.out
        assert "no cues" in captured.err

    def test_a_loop_cue_with_nothing_running_opens_nothing(self, tmp_path, monkeypatch, capsys,
                                                           plume_registry):
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        sd, mido = fake_audio(None), FakeMido()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main(["cue", "make a loop of this", "--socket",
                             short_socket_path("noone")]) != 0
        capsys.readouterr()
        assert not sd.streams and not mido.ports


# ----------------------------------------------------------- plume stop, every mode

MODES = {
    "plain": {},
    "direct": {"direct": True},
    "midi": {"midi": "auto"},
    "midi-clock": {"midi": "auto", "midi_clock": True},
    "direct-midi": {"direct": True, "midi": "auto"},
    "direct-midi-clock": {"direct": True, "midi": "auto", "midi_clock": True},
}
LEAD_ONLY = {
    "tap": {"tap_start": True},
    "tap-direct": {"tap_start": True, "direct": True},
    "tap-midi": {"tap_start": True, "midi": "auto"},
    "tap-midi-clock": {"tap_start": True, "midi": "auto", "midi_clock": True},
    "tap-direct-midi": {"tap_start": True, "direct": True, "midi": "auto"},
}


class TestPlumeStopEndsEveryMode:
    @pytest.mark.parametrize("mode", MODES)
    def test_jam(self, env, mode):
        take = Take(env)
        take.run("jam", stop_t=3.0, rate=12.0, **MODES[mode])
        assert take.stop_rc == 0
        take.assert_ended_cleanly(realtime.END_STOPPED)

    @pytest.mark.parametrize("mode", [*MODES, *LEAD_ONLY])
    def test_lead(self, env, mode):
        kw = {**MODES, **LEAD_ONLY}[mode]
        take = Take(env)
        take.run("lead", stop_t=6.0, rate=12.0, **kw)
        assert take.stop_rc == 0
        take.assert_ended_cleanly(realtime.END_STOPPED)
        if not kw.get("tap_start"):
            assert take.rows() and take.sd.stream.sounded()
        else:
            assert take.rows() == []    # nobody counted: it stopped while still waiting

    @pytest.mark.parametrize("mode", [*MODES, "tap-midi"])
    def test_lead_with_a_take_in_progress(self, env, mode):
        kw = {**MODES, **LEAD_ONLY}[mode]
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a loop of this")])
        take.run("lead", stop_t=7.0, rate=12.0, device=_slow(), **kw)
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        take.assert_ended_cleanly(realtime.END_STOPPED)

    @pytest.mark.parametrize("mode", MODES)
    def test_lead_with_a_loop_sounding_fades_it(self, env, mode):
        """Stopped in the middle of a held note, so anything but a fade is a cut."""
        take = Take(env)
        th, _ = cue_when(take.sock, [(None, "make a 1 bar loop"), (12, "(stop)")])
        take.run("lead", human=held(5.98, 9.0, total_s=80.0), device=_slow(), **MODES[mode])
        th.join(timeout=10.0)
        take.assert_ended_cleanly(realtime.END_STOPPED)
        rows = take.rows()
        assert rows[-1]["loop"]["layers"][0]["sounding"] is True
        end = int(round((rows[-1]["t_s"] + 2.0) * SR))
        from test_looper_live import envelope
        steady = envelope(take.out, end - 2000)
        assert steady > 0.1, steady
        assert envelope(take.out, end + 64) > 0.5 * steady
        assert envelope(take.out, end + lp.DECLICK_SAMPLES + 64) < 0.2 * steady


# ----------------------------------------------------------- consent, on every path

FLAGS = {
    "plain": [],
    "direct": ["--direct"],
    "midi": ["--midi", "auto"],
    "midi-clock": ["--midi", "auto", "--midi-clock"],
    "direct-midi": ["--direct", "--midi", "auto"],
    "direct-midi-clock": ["--direct", "--midi", "auto", "--midi-clock"],
}
VERBS = {"jam": ["jam"], "lead": ["lead", "lofi at 90 bpm"],
         "lead-tap": ["lead", "lofi at 90 bpm", "--tap-start"],
         "voice": ["voice", "lofi"]}


def command(verb, flags):
    """A command the parser takes: voice on a direct input also needs a voice channel."""
    extra = FLAGS[flags]
    if verb == "voice" and "--direct" in extra:
        extra = [*extra, "--input-device", "fake", "--voice-channel", "2"]
    return [*VERBS[verb], *extra]


class TestConsent:
    @pytest.fixture
    def nothing_started(self, monkeypatch):
        """Past the gate is where a session would start, so a broken gate fails here
        rather than running one."""
        started = []
        monkeypatch.setattr(realtime, "jam", lambda **kw: started.append(kw))
        monkeypatch.setattr(realtime, "lead", lambda **kw: started.append(kw))
        monkeypatch.setattr(voice, "run", lambda **kw: started.append(kw))
        return started

    @pytest.mark.parametrize("flags", FLAGS)
    @pytest.mark.parametrize("verb", VERBS)
    def test_with_no_one_at_the_terminal_nothing_opens(self, verb, flags, monkeypatch, capsys,
                                                       nothing_started):
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
        sd, mido = fake_audio(None), FakeMido()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main(command(verb, flags)) != 0
        assert "refusing to play live audio" in capsys.readouterr().err
        assert not nothing_started and not sd.streams and not mido.ports

    @pytest.mark.parametrize("flags", FLAGS)
    @pytest.mark.parametrize("verb", VERBS)
    def test_an_explicit_no_beats_a_terminal(self, verb, flags, monkeypatch, capsys,
                                             nothing_started):
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "0")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        sd, mido = fake_audio(None), FakeMido()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main(command(verb, flags)) != 0
        assert "refusing to play live audio" in capsys.readouterr().err
        assert not nothing_started and not sd.streams and not mido.ports

    def test_the_gate_does_not_stop_a_session_that_has_consent(self, monkeypatch, nothing_started,
                                                               capsys):
        """The control: with consent the same commands do reach a session."""
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
        sd, mido = fake_audio(None), FakeMido()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main(["lead", "lofi at 90 bpm", "--midi", "auto"]) == 0
        capsys.readouterr()
        assert len(nothing_started) == 1

    def test_calibrate_plays_clicks_so_it_asks_too(self, monkeypatch, capsys):
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
        sd = fake_audio(None)
        with patched_module("sounddevice", sd):
            assert cli.main(["calibrate"]) != 0
        assert "refusing to play live audio" in capsys.readouterr().err
        assert not sd.streams

    def test_listing_ports_needs_no_consent_and_opens_nothing(self, monkeypatch, capsys):
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        sd, mido = fake_audio(None), FakeMido(["Studio Keys"])
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            assert cli.main(["midi"]) == 0
        assert "Studio Keys" in capsys.readouterr().out
        assert not sd.streams and not mido.ports


# --------------------------------------------------- the other ways a take can end

COMBOS = {
    "lead-midi": ("lead", {"midi": "auto"}),
    "lead-direct-midi": ("lead", {"direct": True, "midi": "auto"}),
    "lead-tap-midi-clock": ("lead", {"tap_start": True, "midi": "auto", "midi_clock": True}),
    "lead-direct": ("lead", {"direct": True}),
    "jam-midi-clock": ("jam", {"midi": "auto", "midi_clock": True}),
    "jam-direct-midi": ("jam", {"direct": True, "midi": "auto"}),
}


@pytest.mark.usefixtures("fake_device_consent")
class TestShutdown:
    @pytest.mark.parametrize("sig,reason", [(signal.SIGINT, realtime.END_INTERRUPT),
                                            (signal.SIGTERM, realtime.END_TERMINATED)],
                             ids=["sigint", "sigterm"])
    @pytest.mark.parametrize("combo", COMBOS)
    def test_a_signal(self, env, combo, sig, reason):
        verb, kw = COMBOS[combo]
        take = Take(env)
        take.run(verb, stop_t=4.0, rate=12.0, on_stream=lambda: signal_self(sig), **kw)
        take.assert_ended_cleanly(reason)

    @pytest.mark.parametrize("combo", COMBOS)
    def test_the_session_cap(self, env, combo):
        verb, kw = COMBOS[combo]
        take = Take(env)
        take.run(verb, rate=12.0, max_minutes=0.05, **kw)
        take.assert_ended_cleanly(realtime.END_MAX_MINUTES)

    @pytest.mark.parametrize("combo", [c for c in COMBOS if "midi" in c])
    def test_a_cable_pulled_as_it_stops(self, env, combo):
        """The port is gone and raises on the way out.  The take still ends and says why."""
        verb, kw = COMBOS[combo]
        take = Take(env)
        take.mido = FakeMido(fail_close=True)

        def pull_and_stop():
            take.mido.names[:] = []
            take.plume_stop()
        take.run(verb, stop_t=4.0, rate=12.0, on_stream=pull_and_stop, **kw)
        take.assert_ended_cleanly(realtime.END_STOPPED)
        last = (take.rec / "midi.ndjson").read_text().splitlines()
        assert all(json.loads(x) for x in last)

    @pytest.mark.parametrize("combo", COMBOS)
    def test_a_device_that_will_not_stop(self, env, combo):
        """Pa_StopStream wedged: the ending has to get past it with everything closed."""
        verb, kw = COMBOS[combo]
        take = Take(env)
        take.run(verb, stop_t=4.0, rate=12.0, device={"block_on_stop": True, "block_timeout": 4.0},
                 **kw)
        take.assert_ended_cleanly(realtime.END_STOPPED)
        assert take.sd.stream.aborted      # what takes the blocking stop out of the picture

    def test_silence_on_the_keys_ends_a_take_whatever_the_mic_hears(self, env):
        """On a MIDI take the keys say whether the player is playing.  The mic keeps hearing
        loud plucks to the very end, and they must not keep the take alive."""
        beat = 60.0 / 110.0
        take = Take(env)
        presses = [1.0 + i * beat for i in range(60)]
        take.run("lead", tap_start=True, midi="auto", rate=8.0, human=decoy(200.0),
                 max_minutes=3.0, script=presses_script(take.mido, presses))
        take.assert_ended_cleanly(realtime.END_SILENCE)
        rows = [r for r in take.rows() if "playing" in r]
        assert rows
        assert take.sd.stream.sounded()
        # The short wait reaches keys too.  A playing row reads the bar two before it.
        bar_s = 240.0 / rows[-1]["bpm"]
        heard_until = [r for r in rows if r["playing"]][-1]["t_s"] - bar_s
        assert 16.0 <= rows[-1]["t_s"] + bar_s - heard_until <= 23.0


@pytest.mark.usefixtures("fake_device_consent")
class TestLoopsOnTheRealTransport:
    def test_named_length_over_held_chords_closes_and_takes_the_next_cue(self, env):
        take = Take(env)
        th, replies = cue_when(take.sock, [(1, "make a 1 bar loop"),
                                           (5, "make a 1 bar loop called next"),
                                           (6, "(stop)")])
        take.run("lead", human=held(0.0, 60.0, total_s=60.0), device=_slow())
        th.join(timeout=10.0)
        assert replies["make a 1 bar loop"]["ok"] is True
        assert replies["make a 1 bar loop called next"]["ok"] is True
        assert [r for r in take.rows() if r.get("loop", {}).get("layers")]

    def test_a_faded_take_ends_without_a_blip(self, env):
        bpm = 96.0
        human = notes([4.0 + i * 60.0 / bpm for i in range(17)], total_s=150.0, gain=0.4)
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a loop of this")])
        take.run("lead", human=human, tap_start=True, device=_slow(), max_minutes=3.0)
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        take.assert_ended_cleanly(realtime.END_SILENCE)
        mono = np.abs(take.out).max(axis=1)
        last = int(np.flatnonzero(mono > 1e-6)[-1])
        # The fade has been at the bottom for the whole of the last half second.
        tail = mono[last - SR // 2:last + 1]
        assert 20.0 * np.log10(float(tail.max()) + 1e-12) < -40.0

    def test_loops_add_no_clipping_to_a_band_that_has_its_own(self, env, monkeypatch):
        """Six layers at their full level, all held, on every bar of a loud band."""
        class Stacked(lp.Looper):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                t = np.arange(SR * 2) / SR
                wave = (0.25 * np.sin(2 * np.pi * 3137.0 * t)).astype(np.float32)
                for i in range(6):
                    self.stack.add(lp.LoopLayer(name=f"l{i}", audio=wave.copy(), bars=1,
                                                bpm=120.0, anchor_bar=0))
        monkeypatch.setattr(lp, "Looper", Stacked)
        take = Take(env)
        take.run("lead", stop_t=20.0, device=_slow())
        played = np.abs(take.out[6 * SR:]).max(axis=1)
        assert tone_db(take.out, 6.0, 18.0) > -20.0, "the loops never sounded"
        # The band alone goes over on about 2% of samples, and right around those the loops
        # don't dip because it would fix nothing.  Six loops with no dip would be most of them.
        assert float((played > 1.0).mean()) < 0.10


class TestOnAMidiTake:
    def presses(self):
        return [1.0 + i * 60.0 / 110.0 for i in range(70)]

    def test_playing_is_trusted_with_the_band_in_because_it_is_the_keys(self, env):
        take = Take(env)
        take.run("lead", tap_start=True, midi="auto", rate=8.0, stop_t=36.0, human=decoy(60.0),
                 script=presses_script(take.mido, self.presses()), max_minutes=3.0)
        rows = [r for r in take.rows() if "playing_trusted" in r]
        band_in = [r for r in rows if r.get("entry") not in (None, "listening")]
        assert band_in and all(r["playing_trusted"] is True for r in rows)

    @pytest.mark.parametrize("verb", ["lead", "jam"])
    def test_the_band_reading_the_keys_cannot_be_masked(self, env, verb, monkeypatch):
        seen = []
        real = realtime.analysis_step

        def spy(**kw):
            seen.append(kw["masker"])
            return real(**kw)
        monkeypatch.setattr(realtime, "analysis_step", spy)
        take = Take(env)
        take.run(verb, midi="auto", stop_t=4.0, rate=12.0, device={"bleed_db": -6.0})
        assert seen and all(isinstance(m, MidiMasker) for m in seen)
        assert all(m.masked_frames == 0 for m in seen)

    @pytest.mark.parametrize("verb", ["lead", "jam"])
    def test_a_note_is_stamped_on_the_mic_clock(self, env, verb, monkeypatch):
        offsets = []
        real = midi_mod.MidiInput.start

        def spy(self):
            offsets.append(self.offset_s)
            return real(self)
        monkeypatch.setattr(midi_mod.MidiInput, "start", spy)
        take = Take(env)
        take.run(verb, midi="auto", stop_t=3.0, rate=12.0, device={"latency": (0.05, 0.01)})
        assert offsets == [pytest.approx(0.05)]

    @pytest.mark.parametrize("verb", ["lead", "jam"])
    def test_a_stuck_port_does_not_stop_a_session_being_stopped(self, env, verb):
        take = Take(env)
        take.mido = StuckMido()
        t0 = time.monotonic()
        try:
            take.run(verb, midi="auto", stop_t=3.0, rate=12.0)
            took = time.monotonic() - t0
        finally:
            take.mido.let_go.set()
        assert took < 15.0     # the driver holds out for 20
        assert take.ending["reason"] == realtime.END_STOPPED
        assert take.sd.stream.closed


class TestWhatATakeTellsTheStatusLine:
    def test_the_player_hears_a_loop_was_made(self, env, capsys):
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a 2 bar loop"), (14, "(stop)")])
        take.run("lead", human=notes(TestLoopOnAMidiTake.PHRASE, total_s=80.0), device=_slow(),
                 status=True)
        th.join(timeout=10.0)
        assert replies["make a 2 bar loop"]["ok"] is True
        assert "loop 'loop1' made" in capsys.readouterr().out


class TestALoopThatSetsTheTempo:
    def test_the_key_starts_again_from_what_comes_next(self, env, monkeypatch):
        made = []
        real = harmony_mod.HarmonyTracker

        class Counted(real):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                made.append(self)
        monkeypatch.setattr(harmony_mod, "HarmonyTracker", Counted)
        bpm = 96.0
        human = notes([4.0 + i * 60.0 / bpm for i in range(17)], total_s=120.0)
        take = Take(env)
        th, replies = cue_when(take.sock, [(None, "make a loop of this"), (4, "(stop)")])
        take.run("lead", human=human, tap_start=True, device=_slow())
        th.join(timeout=10.0)
        assert replies["make a loop of this"]["ok"] is True
        assert take.rows(), "the loop never started the band"
        assert len(made) == 2    # one for the session, one from the moment the loop set it


@pytest.mark.usefixtures("fake_device_consent")
class TestAUsedDirectory:
    def mic_take(self, env):
        take = Take(env)
        take.run("lead", stop_t=8.0, rate=12.0, human=notes([1.0 + i for i in range(12)],
                                                            total_s=40.0))
        return take

    def test_a_mic_take_after_a_midi_one_does_not_read_as_midi(self, env):
        first = Take(env)
        first.run("lead", midi="auto", stop_t=4.0, rate=12.0,
                  script=presses_script(first.mido, [0.5, 1.0, 1.5]))
        assert (first.rec / "midi.ndjson").exists()
        take = self.mic_take(env)
        assert take.manifest()["input"].get("midi_port") is None
        assert not (take.rec / "midi.ndjson").exists()
        listener = Listener(take.rec)
        listener._new_midi()
        assert listener._midi is None

    def test_a_midi_take_after_a_mic_one_keeps_its_own_notes(self, env):
        self.mic_take(env)
        take = Take(env)
        take.run("lead", midi="auto", stop_t=4.0, rate=12.0,
                 script=presses_script(take.mido, [0.5, 1.0, 1.5]))
        lines = [json.loads(x) for x in (take.rec / "midi.ndjson").read_text().splitlines()]
        assert lines[0]["type"] == "port"

    def test_an_old_voice_channel_is_not_what_gets_transcribed(self, tmp_path):
        sf.write(str(tmp_path / "voice.wav"), np.zeros(4800, dtype="float32"), SR)
        rec = realtime.SessionRecorder(tmp_path)
        rec.open({})
        rec.finish({})
        assert not (tmp_path / "voice.wav").exists()

    def test_a_take_with_a_voice_channel_keeps_its_new_one(self, tmp_path):
        sf.write(str(tmp_path / "voice.wav"), np.zeros(4800, dtype="float32"), SR)
        rec = realtime.SessionRecorder(tmp_path, voice=True)
        rec.open({})
        rec.finish({})
        assert (tmp_path / "voice.wav").exists()
