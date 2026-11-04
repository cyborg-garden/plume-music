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


def test_a_prompt_that_looks_like_a_flag_stays_a_prompt(tmp_path):
    """Parsed by plume's real parser.  Nothing is started."""
    from plume import cli, voice

    argv = {}

    class Recording:
        def __init__(self, args, **_kw):
            argv["args"] = args

        def poll(self):
            return None

    band = voice.Band(prompt="--max-minutes=0", base=tmp_path, write=lambda s: None, popen=Recording)
    band.start()
    band._proc = None
    args = argv["args"][argv["args"].index("-m") + 2:]
    parsed = cli._build_parser().parse_args(args)
    assert parsed.prompt == "--max-minutes=0"
    assert parsed.max_minutes is None


# ---- why no spoken command registered, as far as code can show ----

@pytest.mark.parametrize("text,want", [
    ("Plume, that’s it.", "stop"),         # the model writes a curly apostrophe
    ("Plume, sparse.", "sparser"),
    ("Plume. Busy.", "busier"),
    ("plume stops", "stop"),
])
def test_what_the_model_writes_for_the_commands(text, want):
    assert command_in(text) == want


def test_cymbals_do_not_fold_down_into_the_speech_band(tmp_path):
    """Whisper hears 16 kHz.  Thinning 48 kHz without a low-pass folds everything above
    8 kHz, which is the band's hats and crash, onto the voice."""
    t = np.arange(int(8.0 * 48000)) / 48000
    def level(freq):
        d = tmp_path / str(freq)
        d.mkdir()
        sf.write(str(d / "human.wav"), (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 48000)
        got = []
        VoiceLoop(d, act=lambda n: None, transcribe=lambda a: got.append(a) or "").step()
        # filtered this far down, the quiet gate may not even send it to the model
        return float(np.sqrt(np.mean(got[0][800:-800] ** 2))) if got else 0.0
    assert level(1000.0) > 0.15, "speech-band audio must survive"
    assert level(12000.0) < 0.003, "12 kHz must not arrive at 4 kHz"


def test_a_voice_channel_is_what_gets_transcribed(tmp_path):
    t = np.arange(int(8.0 * 48000)) / 48000
    sf.write(str(tmp_path / "human.wav"), np.zeros_like(t, dtype="float32"), 48000)
    sf.write(str(tmp_path / "voice.wav"), (0.1 * np.sin(2 * np.pi * 300 * t)).astype("float32"), 48000)
    seen = []
    loop = VoiceLoop(tmp_path, act=lambda n: None,
                     transcribe=lambda a: seen.append(float(np.abs(a).max())) or "")
    loop.step()
    assert seen and seen[0] > 0.05


def test_every_window_leaves_a_record_of_what_was_heard(tmp_path):
    """A real take ended with no command and no way to tell why: nothing kept what the
    model wrote.  Too quiet, heard wrongly and heard rightly must look different."""
    import json
    _recording(tmp_path)
    said = iter(["the plum of smoke", "plume stop"])
    loop = VoiceLoop(tmp_path, act=lambda n: None, transcribe=lambda a: next(said))
    loop.step()
    loop._read_to = 0
    loop.step()
    rows = [json.loads(x) for x in (tmp_path / "voice.ndjson").read_text().splitlines()]
    assert [r["text"] for r in rows] == ["the plum of smoke", "plume stop"]
    assert [r["command"] for r in rows] == [None, "stop"]
    assert rows[0]["source"] == "human.wav" and rows[0]["rms_db"] > -60.0
    assert "took_s" in rows[0]


def test_a_window_too_quiet_for_the_model_says_so(tmp_path):
    import json
    _recording(tmp_path, amp=0.0001)
    VoiceLoop(tmp_path, act=lambda n: None, transcribe=lambda a: "").step()
    row = json.loads((tmp_path / "voice.ndjson").read_text().splitlines()[0])
    assert row["skipped"] == "quiet" and row["text"] is None


def test_a_slow_model_leaves_a_record_of_the_audio_it_never_heard(tmp_path):
    import json
    _recording(tmp_path, seconds=8.0)
    loop = VoiceLoop(tmp_path, act=lambda n: None, transcribe=lambda a: "")
    loop.step()
    t = np.arange(int(16.0 * 48000)) / 48000
    sf.write(str(tmp_path / "human.wav"), (0.05 * np.sin(2 * np.pi * 200 * t)).astype("float32"), 48000)
    loop.step()
    rows = [json.loads(x) for x in (tmp_path / "voice.ndjson").read_text().splitlines()]
    assert rows[0]["missed_s"] == 0.0 and rows[1]["missed_s"] == pytest.approx(4.0, abs=0.01)


def test_the_model_is_loaded_before_the_band_starts(tmp_path, monkeypatch):
    """The first transcription downloads and loads the model, and it used to happen
    inside the take: the band played on with nobody listening."""
    from plume import voice
    order = []
    monkeypatch.setattr(voice.importlib.util, "find_spec", lambda name: object())

    def whisper(audio):
        order.append("model")
        return "plume stop"
    monkeypatch.setattr(voice, "_whisper", whisper)

    class Logged(FakeProc):
        def __init__(self, cmd, **kw):
            order.append("band")
            super().__init__(cmd, **kw)

    class Feeder(VoiceLoop):
        def tail(self):
            return np.full(16000, 0.05, dtype=np.float32)
    monkeypatch.setattr(voice, "VoiceLoop", Feeder)
    voice.run(prompt="p", base=tmp_path, write=lambda s: None, popen=Logged, poll_s=0.0)
    assert order[:2] == ["model", "band"]


def test_a_recording_being_written_can_be_read_as_it_grows(tmp_path):
    """Voice shares the mic by reading a file another handle is still writing."""
    from plume.realtime import SessionRecorder
    rec = SessionRecorder(tmp_path, flush_every_s=0.0).open()
    seen = []
    loop = VoiceLoop(tmp_path, act=lambda n: None,
                     transcribe=lambda a: seen.append(a.size) or "")
    block = np.full(48000, 0.05, dtype=np.float32)
    for _ in range(5):
        rec.audio(block, block)
    assert loop.step() is None and seen == [int(4.0 * 16000)]
    rec.audio(block, block)
    assert loop.step() is None and len(seen) == 1, "one second is not a hop"
    rec.audio(block, block)
    loop.step()
    assert len(seen) == 2
    rec.close()


def test_replay_prints_what_the_model_hears_in_an_old_take(tmp_path):
    """No device, no band: the way to ask what happened in a take that is already over."""
    import json
    from plume import voice
    _recording(tmp_path, seconds=8.0)
    lines = []
    said = iter(["nothing", "plume stop", "nothing"])
    voice.replay(tmp_path / "human.wav", transcribe=lambda a: next(said), write=lines.append)
    rows = [json.loads(x) for x in lines]
    assert [r["t_s"] for r in rows] == [0.0, 2.0, 4.0]
    assert [r["command"] for r in rows] == [None, "stop", None]


def test_voice_refuses_a_direct_input_with_no_voice_channel(monkeypatch, capsys):
    """Direct means human.wav holds only the guitar, so nothing the player says would reach
    the model, and the take would look exactly like the ones that heard nothing."""
    from plume import cli, voice
    monkeypatch.setattr(voice, "run", lambda **kw: pytest.fail("must not start a band"))
    assert cli.main(["voice", "lofi", "--direct", "--input-device", "audio interface"]) != 0
    assert "voice-channel" in capsys.readouterr().err


def test_voice_passes_the_route_to_its_band(tmp_path, monkeypatch):
    from conftest import fake_audio, patched_module
    from plume import cli, voice
    seen = {}
    monkeypatch.setattr(voice, "run", lambda **kw: seen.update(kw))
    monkeypatch.setenv("PLUME_ALLOW_AUDIO", "1")
    rig = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]
    with patched_module("sounddevice", fake_audio(None, devices=rig)):
        assert cli.main(["voice", "lofi", "--base", str(tmp_path), "--direct",
                         "--input-device", "audio interface", "--input-channel", "1",
                         "--voice-channel", "2", "--device", "audio interface"]) == 0
    assert seen["extra_args"] == ["--device", "audio interface", "--input-device", "audio interface",
                                  "--input-channel", "1", "--voice-channel", "2", "--direct"]


# ---- a spoken phrase, all the way into a running session ----

def _thread_band(fake, patched):
    """A ``popen`` running the real ``plume lead`` command line in a thread, over the
    fake device, so the arguments voice builds are the ones actually parsed."""
    import threading
    from plume import cli, realtime

    class Proc:
        def __init__(self, cmd, **kw):
            self.cmd = cmd
            assert cmd[1:3] == ["-m", "plume"]
            self.argv = cmd[3:]
            self.code = None
            self.thread = threading.Thread(target=self._go, daemon=True)
            self.thread.start()

        def _go(self):
            self.code = cli.main(self.argv)

        def poll(self):
            return self.code

        def send_signal(self, sig):
            realtime.send_stop(socket_path=realtime.default_socket_path())

        def wait(self, timeout=None):
            self.thread.join(timeout)
            return self.code

        def kill(self):
            raise AssertionError("the graceful stop should have been enough")
    return Proc


def _spoken_take(seconds, said_at):
    """A voice channel: silence, with a burst of 'speech' at the given time."""
    x = np.zeros(int(seconds * 48000), dtype=np.float32)
    # Long enough to land in a window when the fake device outruns the file's flushes.
    t = np.arange(int(6.0 * 48000)) / 48000
    burst = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    at = int(said_at * 48000)
    x[at:at + burst.size] += burst
    return x


def _voice_session(tmp_path, monkeypatch, *, said, said_at, extra):
    """voice.run -> the real CLI -> the real lead over the fake device.  The transcriber
    is the only stand-in: it 'hears' ``said`` only if the burst reached it, then stops
    the take a while later."""
    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take
    from plume import realtime, voice

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    monkeypatch.setenv(realtime.SOCKET_ENV, short_socket_path("vc"))
    guitar = _take(seconds=170.0)
    fake = fake_audio(guitar, rate_x=8.0, devices=[
        {"name": "Audio Interface 2x2", "max_input_channels": 2, "max_output_channels": 2}],
        inputs={1: _spoken_take(170.0, said_at)}, bleed_db=-30.0, bleed_channels=(1,))
    heard = []

    def transcribe(audio):
        spoke = float(np.sqrt(np.mean(audio ** 2))) > 0.02
        if heard:
            heard.append(spoke)
        elif spoke:
            heard.append(spoke)
            return f"Plume, {said}."
        return "plume, stop" if len(heard) > 12 else ""

    lines = []
    with patched_module("sounddevice", fake):
        voice.run(prompt="midwest emo in C major", base=tmp_path, extra_args=extra,
                  write=lines.append, transcribe=transcribe,
                  popen=_thread_band(fake, patched_module), poll_s=0.02)
    return tmp_path / "takes1", lines, heard


ROUTE = ["--device", "audio interface", "--input-channel", "1", "--voice-channel", "2", "--direct"]


def test_a_spoken_busier_changes_the_running_session(tmp_path, monkeypatch, plume_registry):
    import json
    rec, lines, heard = _voice_session(tmp_path, monkeypatch, said="busier", said_at=55.0,
                                       extra=ROUTE)
    assert '{"heard": "busier"}' in lines and '{"cue_failed": "busier"}' not in lines
    rows = [json.loads(x) for x in (rec / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    landed = [r for r in rows if r.get("cue_effects")]
    assert landed, "the cue never reached the session's plan"
    assert "busier" in landed[0]["form"]["applied_cue"]
    bar = landed[0]["bar"]
    assert any(r["bar"] > bar and r["section"] != "intro" for r in rows)
    # it scales the song's own shape: same section, more density than before the cue
    def section_density(pred):
        return {r["section"]: r["density"] for r in rows if pred(r) and r["section"] != "intro"}
    before, after = section_density(lambda r: r["bar"] < bar), section_density(lambda r: r["bar"] > bar)
    shared = set(before) & set(after)
    assert shared and all(after[s] > before[s] for s in shared)
    logged = [json.loads(x) for x in (rec / "voice.ndjson").read_text().splitlines()]
    assert logged[0]["source"] == "voice.wav"
    assert any(r["command"] == "busier" for r in logged)


def test_a_spoken_stop_ends_the_running_session(tmp_path, monkeypatch, plume_registry):
    import json
    from plume import realtime
    rec, lines, _ = _voice_session(tmp_path, monkeypatch, said="stop", said_at=20.0, extra=ROUTE)
    assert '{"heard": "stop"}' in lines
    ending = json.loads((rec / "ending.json").read_text())
    assert ending["reason"] == realtime.END_STOPPED
    assert ending.get("bars", 0) < 100, "it should end near when the player spoke"


@pytest.mark.parametrize("args,says", [
    (["--voice-channel", "2"], "--input-device"),
    (["--device", "audio interface", "--voice-channel", "1"], "same"),
    (["--device", "audio interface", "--voice-channel", "9"], "1 to 4"),
    (["--input-device", "nope"], "Audio Interface"),
])
def test_voice_reports_a_bad_route_before_loading_the_model(monkeypatch, capsys, args, says):
    """The first load can be a download of minutes, and the child's own message only
    goes to the band log."""
    from conftest import fake_audio, patched_module
    from plume import cli, voice
    rig = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]
    monkeypatch.setattr(voice, "run", lambda **kw: pytest.fail("must not start"))
    monkeypatch.setenv("PLUME_ALLOW_AUDIO", "1")
    fake = fake_audio(None, devices=rig)
    with patched_module("sounddevice", fake):
        assert cli.main(["voice", "lofi", *args]) != 0
    assert says in capsys.readouterr().err
    assert fake.opened == [] and fake.recorded == []


# ---- a stop that fails is worse than one that ends a take by accident ----

@pytest.mark.parametrize("said", [
    "plume stop the band and the loop",
    "plume stop everything, the loop too",
    "plume stop stop the loop",
    "plume stop the loop and the band",
    "plume, stop the music, then the loop",
    "plume stop the band, the loop too",
    "plume stop the band, loop too",
    "plume stop. the loop too.",
    "plume, stop. the loop was great.",
    "plume, stop the",                     # a window that cut the sentence off
    "plume stop the loop and let's stop everything",
    "plume stop the loop let's stop it all",
    "plume stop the loop not the band stop everything",
    "plume stop the loop but let's end this",
    "plume stop the loop. stop everything.",
    "Plume, stop. Great loop.",
    "Plume, stop. Nice loop!",
    "Plume, stop. Good loop!",
    "plume stop. what a loop.",
    "plume stop what a loop",
    "plume stop. stupid loop",
    "Plume, stop, great loop",
    "Plume, stop. The loop. Stop everything.",
])
def test_a_stop_that_mentions_a_loop_still_stops(said):
    assert command_in(said) == "stop"


@pytest.mark.parametrize("said", [
    "plume, stop the loop",
    "plume stop the bass loop",
    "plume, stop the old bass loop",
    "plume stop all the loops",
    "plume stop loop 1",
    "plume, stop looping",
    "plume, finish the loop",
    "plume, stop the loop now",
    "plume stop the loop, it's too loud",
    "plume stop the loop and keep playing",
    "plume, stop this loop right now",
    "Plume, stop. The loop.",
    "Plume. Stop. The loop.",
    "plume stop the whole loop",
    "plume, stop that great loop",
    "plume stop the loop please",
    "plume stop the loop right now",
    "plume stop the loop, it's too much",
    "plume stop the loop, keep going",
    "plume stop the loop. plume stop the loop.",
    "plume stop the loop now. plume, please stop the loop.",
])
def test_a_stop_aimed_only_at_a_loop_leaves_the_take_alone(said):
    assert command_in(said) is None


@pytest.mark.parametrize("said", [
    "plume, just stop", "plume, please just stop", "plume, no, stop", "plume, wait, stop",
    "plume, um, stop", "plume, could you please stop", "plume, you can stop now",
    "plume, let's stop", "plume, everybody stop", "plume, band stop",
    "plume busier. stop.", "plume louder stop", "plume, louder, no, stop", "plume, chill, stop",
    "stop, plume", "stop plume", "stop it plume", "stop now, plume", "please stop plume",
    "plume, end", "plume, end this", "plume, end the take", "plume, finished",
    "plume, stopped", "plume, quit", "plume, halt", "plume, done", "plume, enough",
    "plume, that's enough", "plume, we're done", "plume, i'm done", "plume, kill it",
    "plume, shut up", "plume, turn it off", "plume, wrap it up",
    "plume stop the band loop", "plume stop the music loop", "plume, stop all the song loops",
    "plume, stop this stupid song loop",
    "plume, never mind, stop", "plume, not now, stop", "plume, no don't, stop",
    "plume, that's not it, stop", "plume, don't. stop.", "plume, not that. stop.",
    "not now. stop, plume", "never mind, stop plume", "don't. stop, plume",
    "plume, louder, no, actually wait, please just stop", "plume, oh my god please just stop",
    "plume, no no no no no no stop", "plume, we are done", "plume, i am done",
    "plume, we're finished", "plume, end it", "plume, enough now",
    "stop the music, plume", "stop playing, plume", "stop the band, plume",
    "stop everything, plume", "stop that, plume", "stop right now, plume",
    "stop now please, plume", "plume, why don't you stop", "plume, why don't we stop",
    "plume, can't you stop", "plume, won't you stop", "plume, if you could stop",
])
def test_a_loosely_said_stop_still_stops(said):
    assert command_in(said) == "stop"


@pytest.mark.parametrize("said", [
    "plume, don't stop", "plume, do not stop", "plume, never stop", "plume, please don't stop",
    "plume, don't you stop", "plume, keep going, don't stop", "don't stop plume",
    "plume, can you stop the loop", "plume, just stop the loop please", "plume, end the loop",
    "plume non stop", "plume, wait until i stop", "plume, when i stop", "plume, if i stop",
    "plume, before you stop", "plume, why did you stop", "plume, you wouldn't stop",
    "plume, you shouldnt stop", "don't you ever stop plume",
    "plume, keep going till i say stop",
    "when flowers bloom the rain will stop", "the roses bloom and the rain will stop",
    "fruit like a plum then stop", "hearts stop, bloom",
    "plume's done a great job", "plume's enough for me", "plume's end of the song is great",
    "plume, enough bass", "plume, end of the chorus", "plume end of phrase",
    "plume, end the verse", "plume, done well",
])
def test_a_stop_that_is_not_for_the_take_leaves_it_alone(said):
    assert command_in(said) is None


@pytest.mark.parametrize("said,want", [
    ("plume, louder again", "busier"), ("plume, in the end", None),
    ("plume, more enough", "busier"),
])
def test_only_the_word_stop_may_come_late(said, want):
    assert command_in(said) == want


@pytest.mark.parametrize("said", ["plume restart the loop", "plume, more of the loop",
                                  "plume, less of that loop please",
                                  "plume restart. the loop"])
def test_the_other_commands_still_keep_out_of_a_loop_sentence(said):
    assert command_in(said) is None


@pytest.mark.parametrize("said,want", [
    ("plume, restart. great loop.", "restart"),
    ("plume restart, that loop was bad", "restart"),
    ("plume restart the loop and end it", "restart"),
])
def test_restart_follows_the_same_rule_as_stop(said, want):
    assert command_in(said) == want


def test_a_loop_sentence_does_not_hide_a_later_command():
    assert command_in("plume restart the loop. plume stop.") == "stop"


@pytest.mark.parametrize("said", [
    "plume stop the loop and the whole band",
    "plume stop the loop then the band",
    "plume stop the loop with the band",
    "plume stop the loop and quit",
    "plume stop the loop, shut it down",
    "plume stop the loop too, please",
    "plume, stop the loop, the band is fine",
    "plume stop the loop and the bass",
    "plume stop the loop, keep the band going",
    "plume stop the loop but not the band",
    "plume stop the loop, keep playing",
    "plume stop the bass and loop",
    "plume stop great loop",
    "plume, stop, the loop",
    "plume, stop the loop and",
    "plume stop quit the loop",
    "plume stop the loop. plume stop.",
    "plume stop the loop. shut it down.",
    "plume stop. the loop is great.",
    "plume, end it. the loop.",
    "plume, that's it, the loop",
])
def test_anything_but_a_harmless_ending_after_the_loop_stops(said):
    assert command_in(said) == "stop"


@pytest.mark.parametrize("said", ["plume restart the loop and the band",
                                  "plume restart the loop, shut it down",
                                  "plume start over. the loop."])
def test_restart_has_the_same_allowlist(said):
    assert command_in(said) == "restart"


def test_a_repeated_restart_about_the_loop_is_still_about_the_loop():
    assert command_in("plume restart the loop. plume restart the loop now.") is None


def test_a_command_is_only_taken_back_by_a_bare_loop_sentence():
    assert command_in("plume restart. the loop sounded good.") == "restart"
    assert command_in("Plume. Busy. The loop was fine.") == "busier"
