"""Voice commands, with a stand-in transcriber: no model, no band, no device."""
import numpy as np
import pytest
import soundfile as sf

from plume.voice import VoiceLoop, command_in, next_take_dir


@pytest.mark.parametrize("text,want", [
    ("Plume, stop.", "stop"),
    ("plum stop please", "stop"),
    ("Bloom, restart!", "restart"),
    ("plume start over", "restart"),
    ("Plume, busier", "busier"),
    ("plume, sparser", "sparser"),
    ("I can't stop. I can't go.", None),      # a real hallucination, and a lyric
    ("stop", None),
    ("the plume of smoke", None),
    ("", None),
])
def test_commands_need_the_wake_word(text, want):
    assert command_in(text) == want


def test_next_take_dir_takes_the_next_number(tmp_path):
    (tmp_path / "takes3").mkdir()
    (tmp_path / "takes12").mkdir()
    (tmp_path / "takes6-countin").mkdir()
    assert next_take_dir(tmp_path).name == "takes13"


def _recording(path, seconds=8.0, amp=0.05):
    path.mkdir(exist_ok=True)
    t = np.arange(int(seconds * 48000)) / 48000
    sf.write(str(path / "human.wav"), (amp * np.sin(2 * np.pi * 200 * t)).astype("float32"), 48000)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_a_command_acts_once_then_cools_down(tmp_path):
    _recording(tmp_path)
    acted, clock = [], Clock()
    loop = VoiceLoop(tmp_path, act=acted.append, transcribe=lambda a: "plume stop", clock=clock)
    clock.now = 100.0
    assert loop.step() == "stop" and acted == ["stop"]
    loop._read_to = 0
    clock.now = 102.0
    assert loop.step() is None and acted == ["stop"]
    loop._read_to = 0
    clock.now = 110.0
    assert loop.step() == "stop"


def test_a_different_command_is_not_held_up_by_the_last_one(tmp_path):
    _recording(tmp_path)
    said = iter(["plume busier", "plume stop"])
    acted, clock = [], Clock()
    loop = VoiceLoop(tmp_path, act=acted.append, transcribe=lambda a: next(said), clock=clock)
    loop.step()
    loop._read_to = 0
    clock.now = 1.0
    loop.step()
    assert acted == ["busier", "stop"]


def test_it_waits_for_new_audio_and_gets_16k(tmp_path):
    _recording(tmp_path)
    seen = []
    loop = VoiceLoop(tmp_path, act=lambda n: None,
                     transcribe=lambda a: seen.append(a.size) or "")
    loop.step()
    loop.step()
    assert seen == [64000]


def test_silence_never_reaches_the_model(tmp_path):
    _recording(tmp_path, amp=0.0001)
    seen = []
    loop = VoiceLoop(tmp_path, act=lambda n: None, transcribe=lambda a: seen.append(1) or "")
    assert loop.step() is None and seen == []


def test_a_missing_recording_is_not_an_error(tmp_path):
    loop = VoiceLoop(tmp_path / "nope", act=lambda n: None, transcribe=lambda a: "")
    assert loop.step() is None


def test_a_faded_take_is_recognised_and_a_stopped_one_is_not(tmp_path):
    import json
    from plume.realtime import END_SILENCE, END_STOPPED
    from plume.voice import faded_out
    assert faded_out(tmp_path) is False
    (tmp_path / "ending.json").write_text(json.dumps({"reason": END_STOPPED}))
    assert faded_out(tmp_path) is False
    (tmp_path / "ending.json").write_text(json.dumps({"reason": END_SILENCE}))
    assert faded_out(tmp_path) is True


class FakeProc:
    """A band that lives until signalled, or until told how it ended."""
    made = []

    def __init__(self, cmd, **kw):
        self.cmd, self.env, self.code, self.signals = cmd, kw["env"], None, []
        self.dir = next(a for a in cmd if a.endswith("/"))
        FakeProc.made.append(self)

    def poll(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


def _run(tmp_path, transcripts, *, after=None):
    """Drive voice.run with a scripted transcriber; ``after(n)`` runs once per poll."""
    from plume import voice
    FakeProc.made = []
    lines, n = [], [0]

    def transcribe(_audio):
        n[0] += 1
        if after:
            after(n[0])
        try:
            return next(transcripts)
        except StopIteration:
            raise AssertionError("the loop should have ended by now") from None

    class Feeder(VoiceLoop):
        def tail(self):
            return np.full(16000, 0.05, dtype=np.float32)

    orig = voice.VoiceLoop
    voice.VoiceLoop = Feeder
    try:
        voice.run(prompt="p", base=tmp_path, write=lines.append, transcribe=transcribe,
                  popen=FakeProc, poll_s=0.0)
    finally:
        voice.VoiceLoop = orig
    return lines


def test_stop_signals_its_own_band_and_ends(tmp_path):
    import signal
    lines = _run(tmp_path, iter(["plume stop"]))
    assert FakeProc.made[0].signals == [signal.SIGINT]
    assert '{"heard": "stop"}' in lines


def test_the_band_gets_consent_it_has_no_terminal_to_give(tmp_path):
    _run(tmp_path, iter(["plume stop"]))
    assert FakeProc.made[0].env["PLUME_ALLOW_AUDIO"] == "1"


def test_restart_replaces_the_band_and_never_stacks_one(tmp_path):
    import signal
    _run(tmp_path, iter(["plume restart", "plume stop"]))
    first, second = FakeProc.made
    assert first.signals == [signal.SIGINT] and second.signals == [signal.SIGINT]
    assert first.dir != second.dir


def test_a_crash_here_leaves_no_band_playing(tmp_path):
    import signal
    from plume import voice

    def boom(_audio):
        raise RuntimeError("model fell over")

    class Feeder(VoiceLoop):
        def tail(self):
            return np.full(16000, 0.05, dtype=np.float32)

    FakeProc.made = []
    orig = voice.VoiceLoop
    voice.VoiceLoop = Feeder
    try:
        with pytest.raises(RuntimeError, match="fell over"):
            voice.run(prompt="p", base=tmp_path, write=lambda s: None, transcribe=boom,
                      popen=FakeProc, poll_s=0.0)
    finally:
        voice.VoiceLoop = orig
    assert FakeProc.made[0].signals == [signal.SIGINT]


def test_a_song_that_ended_ends_voice_and_a_faded_take_starts_another(tmp_path):
    import json
    from plume.realtime import END_SILENCE, END_SONG

    def script(n):
        proc = FakeProc.made[-1]
        take = tmp_path / proc.dir.rsplit("/", 2)[-2]
        take.mkdir(exist_ok=True)
        (take / "ending.json").write_text(
            json.dumps({"reason": END_SILENCE if n == 1 else END_SONG}))
        proc.code = 0

    lines = _run(tmp_path, iter(["", ""]), after=script)
    assert len(FakeProc.made) == 2 and lines[-1] == '{"ended": true}'


def test_a_band_that_dies_without_ending_is_reported(tmp_path):
    def script(n):
        FakeProc.made[-1].code = 1
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        _run(tmp_path, iter([""]), after=script)


def test_voice_refuses_to_start_a_band_without_whisper(tmp_path, monkeypatch):
    from plume import voice
    monkeypatch.setattr(voice.importlib.util, "find_spec", lambda name: None)
    FakeProc.made = []
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        voice.run(prompt="p", base=tmp_path, write=lambda s: None, popen=FakeProc)
    assert FakeProc.made == []


def test_the_band_gets_its_own_process_group(tmp_path):
    """A terminal's Ctrl-C must not reach it as well as our own signal."""
    seen = {}

    class Recording(FakeProc):
        def __init__(self, cmd, **kw):
            seen.update(kw)
            super().__init__(cmd, **kw)
    from plume import voice
    band = voice.Band(prompt="p", base=tmp_path, write=lambda s: None, popen=Recording)
    band.start()
    assert seen["start_new_session"] is True
    band.stop()


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_being_told_to_end_stops_the_band_on_its_way_out(tmp_path, signame):
    import os
    import signal
    from plume import voice
    FakeProc.made = []

    def terminate(_audio):
        os.kill(os.getpid(), getattr(signal, signame))
        return ""

    class Feeder(VoiceLoop):
        def tail(self):
            return np.full(16000, 0.05, dtype=np.float32)
    orig_loop = voice.VoiceLoop
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    voice.VoiceLoop = Feeder
    try:
        with pytest.raises(SystemExit):
            voice.run(prompt="p", base=tmp_path, write=lambda s: None, transcribe=terminate,
                      popen=FakeProc, poll_s=0.0)
    finally:
        voice.VoiceLoop = orig_loop
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert FakeProc.made[0].signals == [signal.SIGINT]
