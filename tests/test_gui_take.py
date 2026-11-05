"""Starting, watching and stopping a take from the page.

No real take ever starts here.  Every process is a fake, or in one test an inert
Python child that only sleeps, and the argv test drives the real transport through the
fake audio device in tests/conftest.py.  The real spawner and lister refuse to run under
pytest, and that refusal is tested too.  Nothing here says anything about real hardware.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import fake_audio, patched_module, short_socket_path, wait_until
from plume import cli, gui, guitake, realtime
from test_gui_bridge import Client

SR = 48000
SCARLETT = {"name": "Audio Interface 4x4", "inputs": 4, "outputs": 4}
LAPTOP_MIC = {"name": "MacBook Pro Microphone", "inputs": 1, "outputs": 0}
LAPTOP_OUT = {"name": "MacBook Pro Speakers", "inputs": 0, "outputs": 2}
BLACKHOLE = {"name": "BlackHole 16ch", "inputs": 16, "outputs": 16}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.code = None

    def poll(self):
        return self.code


class Rig:
    """Fakes for everything a take touches, and a way to act like the child."""

    def __init__(self, root, state_dir):
        self.root = root
        self.state_dir = state_dir
        self.devices = [LAPTOP_MIC, LAPTOP_OUT, SCARLETT]
        self.clock = FakeClock()
        self.spawned = []
        self.stopped = []
        self.cues = []
        self.others = []
        self.proc = None
        self.stop_hangs = False
        self.spawn_error = None
        self.child_writes = None
        self.list_error = None
        self.human = None

    def spawner(self, argv, log_path):
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawned.append(list(argv))
        self.log_path = log_path
        self.proc = FakeProc(90000 + len(self.spawned))
        return self.proc

    def lister(self):
        if self.list_error is not None:
            raise self.list_error
        return [dict(d) for d in self.devices]

    def sessions(self):
        recs = list(self.others)
        if self.proc is not None and self.proc.code is None:
            recs.append(realtime.LiveSession(pid=self.proc.pid, kind="lead", socket="/nowhere/sock"))
        return recs

    def stopper(self, rec):
        self.stopped.append(rec)
        if not self.stop_hangs and self.proc is not None and rec.pid == self.proc.pid:
            self.proc.code = 0
        return {"stopped": not self.stop_hangs}

    def cue(self, text, *, socket_path, source):
        self.cues.append((text, socket_path, source))

    def runner(self, **kw):
        options = {"spawner": self.spawner, "lister": self.lister, "sessions": self.sessions,
                   "stopper": self.stopper, "cue": self.cue, "clock": self.clock,
                   "now": lambda: datetime(2031, 2, 3, 4, 5, 6), "watch": False,
                   "log_path": lambda: self.state_dir / "take.log",
                   "id_of": lambda folder: f"id-of-{folder.name}"}
        options.update(kw)
        return guitake.TakeRunner(self.root, **options)

    @property
    def folder(self) -> Path:
        return Path(self.spawned[-1][7].split("=", 1)[1])

    def open_stream(self, level=0.1, seconds=0.5):
        """What the child's recorder does: open the WAV, and flush a bit of input."""
        self.human = sf.SoundFile(str(self.folder / "human.wav"), mode="w", samplerate=SR, channels=1)
        self.more_audio(level, seconds)

    def more_audio(self, level=0.1, seconds=0.5):
        self.human.write(np.full(int(SR * seconds), level, dtype=np.float32))
        self.human.flush()

    def hit_first_bar(self):
        (self.folder / "decisions.ndjson").write_text(
            json.dumps({"bar": 0, "t_s": 0.0, "bpm": 115.0}) + "\n")

    def die(self, code=0, stderr=""):
        if self.human is not None:
            self.human.close()
            self.human = None
        self.state_dir.mkdir(exist_ok=True)
        (self.state_dir / "take.log").write_text(stderr)
        self.proc.code = code


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path / "shelf", tmp_path / "state")
    yield made
    if made.human is not None:
        made.human.close()


@pytest.fixture
def runner(rig):
    return rig.runner()


# ----------------------------------------------------------------- choosing a device

@pytest.mark.parametrize("name,inputs,outputs,kind", [
    ("Audio Interface 4x4", 4, 4, "interface"),
    ("Scarlett 2i2 4th Gen", 2, 2, "interface"),
    ("ZOOM UAC-2", 2, 2, "interface"),            # a real interface, by a maker with a video app
    ("BlackHole 2ch", 2, 2, "virtual"),
    ("Soundflower (64ch)", 64, 64, "virtual"),
    ("Loopback Audio", 2, 2, "virtual"),
    ("Aggregate Device", 4, 4, "virtual"),
    ("ZoomAudioDevice", 2, 2, "virtual"),
    ("Microsoft Teams Audio", 2, 2, "virtual"),
    ("MacBook Pro Microphone", 1, 0, "builtin"),
    ("Built-in Microphone", 2, 0, "builtin"),
    ("USB Mic", 1, 0, "other"),
    ("Two In No Out", 2, 0, "other"),
])
def test_what_kind_of_device_this_is(name, inputs, outputs, kind):
    assert guitake.describe_devices(
        [{"name": name, "inputs": inputs, "outputs": outputs}])[0]["kind"] == kind


@pytest.mark.parametrize("name", [
    "HDA Intel PCH: ALC257 Analog (hw:0,0)", "sysdefault", "default", "pulse", "pipewire",
    "Laptop Mic", "Studio Display", "Some Dock Audio",
])
def test_two_in_two_out_is_not_enough_when_the_name_says_it_is_the_computers_own(name):
    """Its own mic and speakers look just like this, and it would hear the band."""
    device = guitake.describe_devices([{"name": name, "inputs": 2, "outputs": 2}])[0]
    assert (device["kind"], device["mode"], device["hear_me"]) == ("builtin", "room", False)


@pytest.mark.parametrize("name", [
    "Scarlett 2i2 USB", "Focusrite USB Audio", "MOTU M2", "Universal Audio Volt 2", "Apollo Twin",
    "Apogee Duet", "PreSonus AudioBox USB 96", "Babyface Pro", "Steinberg UR22C", "UMC204HD 192k",
    "SSL 2+", "Audient iD14", "Komplete Audio 2", "MiniFuse 2", "Some Audio Interface",
])
def test_a_named_interface_is_still_one_and_its_name_earns_no_hear_me(name):
    device = guitake.describe_devices([{"name": name, "inputs": 2, "outputs": 2}])[0]
    assert (device["kind"], device["mode"], device["hear_me"]) == ("interface", "direct", False)


def test_a_device_with_no_word_about_headphones_cannot_be_given_hear_me(rig, runner):
    rig.devices = [LAPTOP_MIC, {"name": "Mystery Box", "inputs": 2, "outputs": 2}]
    box = next(d for d in runner.home()["devices"] if d["name"] == "Mystery Box")
    with pytest.raises(guitake.Refused):
        runner.start({"device": box["id"], "hear_me": True})
    assert rig.spawned == []


@pytest.mark.parametrize("typed", ["FBm", "Fb#", "AMm"])
def test_text_that_breaks_the_key_parser_gets_the_bad_key_words(typed):
    with pytest.raises(guitake.Refused) as e:
        guitake.canonical_key(typed)
    assert (e.value.status, e.value.message) == (400, guitake.BAD_KEY)


@pytest.mark.parametrize("typed", ["D dorian", "A harmonic minor", "E phrygian"])
def test_a_mode_the_band_cannot_play_is_refused_rather_than_swapped(typed):
    with pytest.raises(guitake.Refused) as e:
        guitake.canonical_key(typed)
    assert (e.value.status, e.value.message) == (400, guitake.ONLY_MAJOR_MINOR)


@pytest.mark.parametrize("typed,want", [
    ("F#", "F# major"), ("C#", "C# major"), ("F# major", "F# major"), ("Gb", "Gb major"), ("f#m", "F# minor"),
    ("Bb", "Bb major"), ("Fm", "F minor"), ("F", "F major"), ("a minor", "A minor"), ("Eb maj", "Eb major"),
])
def test_a_typed_key_is_the_key_the_band_commits_to(typed, want):
    got = guitake.canonical_key(typed)
    assert got == want and realtime._named_key(f"in {got}") == want


def test_the_only_interface_is_chosen_on_channel_one_with_hear_me_off(rig, runner):
    home = runner.home()
    chosen = next(d for d in home["devices"] if d["id"] == home["device"])
    assert chosen["name"] == SCARLETT["name"]
    assert (home["mode"], home["channel"]) == ("direct", 1)
    assert home["hear_me"]["allowed"] is False and home["hear_me"]["default"] is False


def test_a_virtual_device_does_not_count_as_the_only_interface(rig, runner):
    rig.devices = [LAPTOP_MIC, BLACKHOLE, SCARLETT]
    home = runner.home()
    assert next(d for d in home["devices"] if d["id"] == home["device"])["name"] == SCARLETT["name"]


@pytest.mark.parametrize("devices", [
    [LAPTOP_MIC, LAPTOP_OUT, BLACKHOLE],
    [LAPTOP_MIC, SCARLETT, {"name": "Other Interface 2x2", "inputs": 2, "outputs": 2}],
])
def test_otherwise_the_built_in_mic_in_room_mode_with_no_hear_me(rig, runner, devices):
    rig.devices = devices
    home = runner.home()
    assert next(d for d in home["devices"] if d["id"] == home["device"])["name"] == LAPTOP_MIC["name"]
    assert home["mode"] == "room"
    assert home["hear_me"]["allowed"] is False and home["hear_me"]["default"] is False


def test_no_built_in_mic_falls_to_the_system_default(rig, runner):
    rig.devices = [BLACKHOLE, {"name": "USB Mic", "inputs": 1, "outputs": 0}]
    home = runner.home()
    assert home["device"] is None and home["mode"] == "room"
    runner.start({})
    assert not [a for a in rig.spawned[0] if a.startswith(("--device", "--input-device", "--direct"))]


def test_no_input_at_all_is_said_plainly(rig, runner):
    rig.devices = [LAPTOP_OUT]
    assert runner.home()["problem"] == guitake.NO_INPUT
    assert runner.start({}) == {"state": "failed", "message": guitake.NO_INPUT}
    assert rig.spawned == []


def test_a_name_that_would_land_on_an_earlier_device_is_not_offered(rig):
    # plume finds a device by substring and takes the first hit, so this would open the wrong one.
    devices = guitake.describe_devices(
        [{"name": "USB Audio Pro", "inputs": 2, "outputs": 2}, {"name": "USB Audio", "inputs": 2, "outputs": 2}])
    assert [d["name"] for d in devices] == ["USB Audio Pro"]


def test_an_interface_whose_output_would_land_elsewhere_is_not_direct():
    # --device finds the output by the same substring, so this would play out of the other one.
    devices = guitake.describe_devices([{"name": "Interface Out Box", "inputs": 0, "outputs": 2},
                                        {"name": "Interface", "inputs": 2, "outputs": 2}])
    assert [(d["name"], d["mode"], d["hear_me"]) for d in devices] == [("Interface", "room", False)]


@pytest.mark.parametrize("junk", [None, "nope", [{"name": "x"}], [{"name": "", "inputs": 1, "outputs": 1}],
                                  [{"name": "x", "inputs": "lots", "outputs": 1}], [7]])
def test_a_strange_listing_is_no_devices(junk):
    assert guitake.describe_devices(junk) == []


def test_a_missing_audio_part_is_said_plainly(rig, runner):
    rig.list_error = guitake.TakeError(guitake.AUDIO_PART_MISSING)
    assert runner.home()["problem"] == guitake.AUDIO_PART_MISSING
    assert runner.start({})["message"] == guitake.AUDIO_PART_MISSING
    assert rig.spawned == []


def test_a_device_plugged_in_later_shows_up(rig, runner):
    rig.devices = [LAPTOP_MIC]
    assert runner.home()["mode"] == "room"
    rig.devices = [LAPTOP_MIC, SCARLETT]
    assert runner.home()["mode"] == "direct"


# -------------------------------------------------------------------- what Start runs

def _argv_tail(rig):
    return rig.spawned[-1][8:]


def test_the_default_argv_exactly(rig, runner, monkeypatch):
    monkeypatch.setattr(guitake.secrets, "randbelow", lambda n: 4242)
    runner.start({})
    assert rig.spawned == [[
        sys.executable, "-P", "-m", "plume", "lead", "--tap-start", "--detect",
        f"--record={rig.root}/2031-02-03-040506/", "--seed=4242",
        f"--device={SCARLETT['name']}", "--input-channel=1", "--direct"]]


def test_hear_me_off_drops_the_monitor(rig, runner):
    runner.start({"hear_me": False})
    assert "--monitor" not in rig.spawned[0]


def test_a_room_mic_argv(rig, runner):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    runner.start({})
    assert _argv_tail(rig) == [f"--seed={rig.spawned[0][8].split('=')[1]}",
                               f"--input-device={LAPTOP_MIC['name']}"]
    assert "--direct" not in rig.spawned[0] and "--monitor" not in rig.spawned[0]


def test_hear_me_on_a_room_mic_with_no_word_about_headphones_starts_nothing(rig, runner):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    with pytest.raises(guitake.Refused) as e:
        runner.start({"hear_me": True})
    assert e.value.status == 400 and e.value.message == guitake.NO_HEAR_ME
    assert rig.spawned == [] and runner.live()["state"] == "idle"


def test_hear_me_on_a_named_room_device_with_no_word_is_refused_too(rig, runner):
    mic = next(d for d in runner.home()["devices"] if d["kind"] == "builtin")
    with pytest.raises(guitake.Refused):
        runner.start({"device": mic["id"], "hear_me": True})
    assert rig.spawned == []


def test_a_chosen_device_and_channel(rig, runner):
    rig.devices = [LAPTOP_MIC, BLACKHOLE, SCARLETT]
    box = next(d for d in runner.home()["devices"] if d["kind"] == "interface")
    runner.start({"device": box["id"], "channel": 3})
    assert f"--device={SCARLETT['name']}" in rig.spawned[0] and "--input-channel=3" in rig.spawned[0]


def test_a_room_device_on_another_channel(rig, runner):
    rig.devices = [{"name": "Stereo Mic", "inputs": 2, "outputs": 0}]
    box = runner.home()["devices"][0]
    runner.start({"device": box["id"], "channel": 2})
    assert "--input-device=Stereo Mic" in rig.spawned[0] and "--input-channel=2" in rig.spawned[0]


@pytest.mark.parametrize("channel", [0, 5, -1, 1.5, "1", True, "--direct"])
def test_a_channel_out_of_range_is_refused(rig, runner, channel):
    with pytest.raises(guitake.Refused) as e:
        runner.start({"channel": channel})
    assert e.value.status == 400 and rig.spawned == []


def test_an_unknown_device_gets_the_plain_words(rig, runner):
    with pytest.raises(guitake.Refused) as e:
        runner.start({"device": "0123456789abcdef"})
    assert e.value.message == guitake.DEVICE_MISSING and rig.spawned == []


@pytest.mark.parametrize("device", ["--direct", SCARLETT["name"], 7, ["x"]])
def test_a_device_is_an_id_never_a_name(rig, runner, device):
    with pytest.raises(guitake.Refused):
        runner.start({"device": device})
    assert rig.spawned == []


@pytest.mark.parametrize("body", [{"prompt": "--max-minutes=0"}, {"argv": ["x"]}, {"record": "/etc"}, []])
def test_an_unknown_field_is_refused(rig, runner, body):
    with pytest.raises(guitake.Refused):
        runner.start(body)
    assert rig.spawned == []


@pytest.mark.parametrize("hear_me", ["yes", 1])
def test_hear_me_must_be_a_real_boolean(rig, runner, hear_me):
    with pytest.raises(guitake.Refused):
        runner.start({"hear_me": hear_me})


@pytest.mark.parametrize("text,canon", [
    ("F major", "F major"), ("f# minor", "F# minor"), ("Eb major", "Eb major"), ("  A minor ", "A minor"),
    ("Fm", "F minor"), ("F", "F major"),
    ("in F major", "F major"), ("", None), ("  ", None), ("detect", None), ("Detect", None),
])
def test_a_key_comes_out_canonical(text, canon):
    assert guitake.canonical_key(text) == canon


@pytest.mark.parametrize("text", ["--device=evil", "-F major", "H major", "--", "blue", "-- F major", "--monitor"])
def test_a_key_that_is_not_one_is_refused(text):
    with pytest.raises(guitake.Refused) as e:
        guitake.canonical_key(text)
    assert e.value.status == 400


@pytest.mark.parametrize("text,canon", [
    ("midwest", "midwest"), ("Emo", "midwest"), ("lo-fi", "lofi"), ("Synth Wave", "synthwave"),
    ("TRAP", "trap"), ("", None), ("detect", None),
])
def test_a_style_comes_out_canonical(text, canon):
    assert guitake.canonical_style(text) == canon


@pytest.mark.parametrize("text", ["--max-minutes=0", "polka", "-x"])
def test_a_style_that_is_not_one_is_refused(text):
    with pytest.raises(guitake.Refused) as e:
        guitake.canonical_style(text)
    assert e.value.status == 400


@pytest.mark.parametrize("field", ["key", "style"])
@pytest.mark.parametrize("value", [7, ["F major"], "x" * 500])
def test_text_fields_must_be_short_text(rig, runner, field, value):
    with pytest.raises(guitake.Refused):
        runner.start({field: value})
    assert rig.spawned == []


@pytest.mark.parametrize("body,prompt", [
    ({"key": "F major"}, "in F major"), ({"style": "emo"}, "midwest"),
    ({"key": "Fm", "style": "lo-fi"}, "lofi in F minor"),
    ({"key": "F major --monitor --max-minutes=0"}, "in F major"),
    ({"style": "--max-minutes=0 trap --device=x"}, "trap"),
])
def test_only_canonical_values_reach_the_prompt(rig, runner, body, prompt):
    runner.start({**body, "hear_me": False})
    assert rig.spawned[0][-2:] == ["--", prompt]
    assert not any(a.startswith("--max-minutes") for a in rig.spawned[0])
    assert "--monitor" not in rig.spawned[0]


def _parsed(argv):
    assert argv[:4] == [sys.executable, "-P", "-m", "plume"]
    return cli._build_parser().parse_args(argv[4:])


@pytest.mark.parametrize("name", ["--record=/etc", "-x", "--monitor", "--max-minutes=0", "two words", "a=b"])
@pytest.mark.parametrize("devices,flag", [(True, "device"), (False, "input_device")])
def test_a_flag_shaped_device_name_stays_a_device_name(tmp_path, name, devices, flag):
    route = guitake.Route(name, "direct" if devices else "room", 1, False)
    args = _parsed(guitake.build_argv(route, tmp_path / "t", key=None, style=None, seed=1))
    assert getattr(args, flag) == name
    assert args.record == f"{tmp_path / 't'}/"
    assert args.monitor is False and args.max_minutes is None


def test_the_real_parser_reads_every_part_of_it(tmp_path):
    route = guitake.Route("Audio Interface 4x4", "direct", 2, True)
    args = _parsed(guitake.build_argv(route, tmp_path / "t", key="F major", style="midwest", seed=9))
    assert (args.tap_start, args.detect, args.direct, args.monitor) == (True, True, True, True)
    assert (args.device, args.input_channel, args.seed) == ("Audio Interface 4x4", 2, 9)
    assert args.prompt == "midwest in F major" and args.input_device is None


def test_the_argv_runs_the_real_transport_on_the_fake_device(tmp_path, monkeypatch, plume_registry):
    """Fake audio only.  The only thing added to the bridge's argv is a short cap and a socket."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    rig_devices = [{"name": "Laptop Mic", "max_input_channels": 1, "max_output_channels": 0},
                   {"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]
    route = guitake.Route("Audio Interface 4x4", "direct", 2, True)
    folder = tmp_path / "take"
    argv = guitake.build_argv(route, folder, key=None, style=None, seed=3)[4:]
    device = fake_audio(np.zeros(SR * 3, dtype=np.float32), devices=rig_devices, human_channel=1)
    with patched_module("sounddevice", device):
        code = cli.main([*argv, f"--socket={short_socket_path('guitake')}", "--max-minutes=0.02"])
    assert code == 0
    manifest = json.loads((folder / "session.json").read_text())
    assert manifest["detect"] is True and manifest["monitor"] is True
    assert manifest["input"]["source"] == "direct"
    assert device.opened[0]["channels"][0] == 2


# ----------------------------------------------------------------- the take's folder

def test_a_take_gets_a_fresh_timestamped_folder(rig, runner):
    runner.start({})
    assert rig.folder == rig.root / "2031-02-03-040506"
    assert rig.folder.is_dir() and rig.spawned[0][7].endswith("/")


def test_a_used_folder_is_never_used_again(rig, runner):
    used = rig.root / "2031-02-03-040506"
    used.mkdir(parents=True)
    (used / "human.wav").write_bytes(b"hers")
    runner.start({})
    assert rig.folder == rig.root / "2031-02-03-040506-2"
    assert (used / "human.wav").read_bytes() == b"hers"


def test_a_symlink_at_that_name_is_not_used_either(rig, runner, tmp_path):
    rig.root.mkdir()
    (rig.root / "2031-02-03-040506").symlink_to(tmp_path / "elsewhere")
    runner.start({})
    assert rig.folder == rig.root / "2031-02-03-040506-2"


def test_a_folder_that_cannot_be_made_is_said_plainly(rig, runner):
    rig.root.write_text("a file, not a folder")
    doc = runner.start({})
    assert doc == {"state": "failed", "message": f"Could not make a folder in {rig.root}."}
    assert rig.spawned == []


def test_home_shows_the_folder_with_a_tilde(rig):
    runner = rig.runner()
    runner.root = Path.home() / "Music" / "Plume"
    assert runner.home()["folder"] == "~/Music/Plume"


# ----------------------------------------------------------------- one take at a time

def test_a_second_start_is_a_409_with_the_live_state(rig, runner):
    runner.start({})
    with pytest.raises(guitake.Refused) as e:
        runner.start({})
    assert e.value.status == 409 and e.value.message == guitake.ALREADY_RUNNING
    assert e.value.live["state"] == "starting"
    assert len(rig.spawned) == 1


def test_a_take_from_a_terminal_blocks_start_and_offers_stop(rig, runner):
    rig.others = [realtime.LiveSession(pid=77, kind="lead", socket="/nowhere/other")]
    assert runner.live() == {"state": "elsewhere", "pid": 77, "message": guitake.TERMINAL_TAKE,
                             "stopping": False}
    with pytest.raises(guitake.Refused) as e:
        runner.start({})
    assert e.value.status == 409 and e.value.message == guitake.TERMINAL_TAKE
    assert e.value.live["state"] == "elsewhere" and rig.spawned == []
    assert runner.stop()["state"] == "elsewhere"
    assert wait_until(lambda: [r.pid for r in rig.stopped] == [77])


def test_stopping_a_terminal_take_twice_asks_once(rig, runner):
    release = threading.Event()
    calls = []

    def slow(rec):
        calls.append(rec.pid)
        release.wait(5)

    runner = rig.runner(stopper=slow)
    rig.others = [realtime.LiveSession(pid=77, kind="lead")]
    runner.stop()
    assert runner.stop()["stopping"] is True
    release.set()
    assert wait_until(lambda: not runner.live()["stopping"])
    assert calls == [77]


def test_concurrent_starts_spawn_once(rig):
    gate = threading.Event()
    entered = threading.Event()

    def slow_lister():
        entered.set()
        gate.wait(5)
        return [SCARLETT]

    runner = rig.runner(lister=slow_lister)
    outcomes = []

    def go():
        try:
            outcomes.append(runner.start({})["state"])
        except guitake.Refused as e:
            outcomes.append(e.status)

    threads = [threading.Thread(target=go) for _ in range(8)]
    threads[0].start()
    assert entered.wait(5)
    for t in threads[1:]:
        t.start()
    assert wait_until(lambda: len(outcomes) == 7)
    assert outcomes == [409] * 7
    gate.set()
    for t in threads:
        t.join(5)
    assert len(rig.spawned) == 1 and outcomes[-1] == "starting"


def test_a_refused_start_does_not_wedge_the_next_one(rig, runner):
    with pytest.raises(guitake.Refused):
        runner.start({"key": "--nope"})
    assert runner.live()["state"] == "idle"
    assert runner.start({})["state"] == "starting"


def test_a_refused_start_leaves_the_last_take_where_it_was(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    rig.hit_first_bar()
    rig.die(0)
    runner.tick()
    with pytest.raises(guitake.Refused):
        runner.start({"key": "--nope"})
    assert runner.live()["state"] == "done"


def test_a_new_start_after_a_finished_one_is_a_new_folder(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    first = rig.folder
    rig.hit_first_bar()
    rig.die(0)
    runner.tick()
    assert runner.start({})["state"] == "starting"
    assert rig.folder != first and len(rig.spawned) == 2


def test_reattach_reads_the_same_take_and_starts_nothing(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    for _ in range(3):
        assert runner.live()["state"] == "waiting"
    assert len(rig.spawned) == 1


# ------------------------------------------------------------------ the take's stages

def test_starting_until_the_recording_has_frames(rig, runner):
    assert runner.start({})["state"] == "starting"
    runner.tick()
    assert runner.live()["state"] == "starting"
    # The recorder makes the file before the stream opens, so the file alone proves nothing.
    with sf.SoundFile(str(rig.folder / "human.wav"), mode="w", samplerate=SR, channels=1) as f:
        f.flush()
    runner.tick()
    assert runner.live()["state"] == "starting"
    (rig.folder / "human.wav").unlink()
    rig.open_stream()
    runner.tick()
    assert runner.live()["state"] == "waiting"


def test_the_level_is_available_while_waiting(rig, runner):
    runner.start({})
    rig.open_stream(level=0.1)
    runner.tick()
    assert runner.live()["level_db"] == pytest.approx(-20.0, abs=0.5)


def test_playing_once_a_bar_is_decided(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    rig.hit_first_bar()
    runner.tick()
    assert runner.live()["state"] == "playing"


def test_the_live_doc_says_what_the_take_is(rig, runner):
    runner.start({"key": "F major", "channel": 2, "hear_me": True, "headphones": True})
    doc = runner.live()
    assert doc["device"] == SCARLETT["name"]
    assert (doc["mode"], doc["channel"], doc["hear_me"]) == ("direct", 2, True)
    assert (doc["key"], doc["style"]) == ("F major", None)


class StubListener:
    rows = []

    def __init__(self, folder, every):
        assert every == 1
        self.folder = folder

    def poll(self):
        out, StubListener.rows = StubListener.rows, []
        return out


def test_the_live_row_passes_only_named_fields(rig):
    runner = rig.runner(listener=StubListener)
    runner.start({})
    rig.open_stream()
    rig.hit_first_bar()
    StubListener.rows = [{"bar": 8, "bpm": float("nan"), "stage": "drums", "key": "F major",
                          "style": "midwest", "style_guess": True, "key_guess": True,
                          "you_notes": ["F"], "you_trusted": ["you_notes"], "you_why": "secret",
                          "entry_reason": "drums: heard 3 bars, no settled key", "you_db": -20.0}]
    runner.tick()
    row = runner.live()["row"]
    assert row["bar"] == 8 and row["bpm"] is None and row["style_guess"] is True
    assert row["you_trusted"] == ["you_notes"]
    assert "you_why" not in row and "entry_reason" not in row
    json.dumps(row, allow_nan=False)


def test_a_broken_reader_cannot_hurt_the_take(rig):
    class Broken:
        def __init__(self, folder, every):
            pass

        def poll(self):
            raise RuntimeError("Traceback: /secret/path")

    runner = rig.runner(listener=Broken)
    runner.start({})
    rig.open_stream()
    rig.hit_first_bar()
    runner.tick()
    assert runner.live()["state"] == "playing" and runner.live()["row"] is None


def test_the_real_listener_reads_a_real_take(rig):
    """No stub: the same reader as ``plume watch``, over files of the real shape."""
    runner = rig.runner()
    runner.start({})
    rig.open_stream(seconds=3.0)
    sf.write(str(rig.folder / "machine.wav"), np.zeros(SR * 3, dtype=np.float32), SR)
    (rig.folder / "session.json").write_text(json.dumps({"input": {"source": "direct"}}))
    (rig.folder / "decisions.ndjson").write_text(json.dumps(
        {"bar": 0, "t_s": 0.0, "bpm": 120.0, "entry": "drums", "key": "F major", "style": "midwest",
         "style_guess": True, "key_guess": False}) + "\n")
    runner.tick()
    row = runner.live()["row"]
    assert (row["bar"], row["stage"], row["style"], row["input"]) == (0, "drums", "midwest", "direct")
    assert row["style_guess"] is True and row["key_guess"] is False
    assert row["you_trusted"] == ["you_db", "you_busy", "you_notes"]


def test_stream_latency_is_read_and_home_remembers_it(rig, runner):
    runner.start({})
    (rig.folder / "stream.json").write_text(json.dumps(
        {"input_ms": 3.5, "output_ms": 4.5, "reported_ms": 8.0, "block_ms": 5.333,
         "calibrated_ms": None, "extra": "/secret"}))
    runner.tick()
    assert runner.live()["stream"] == {"input_ms": 3.5, "output_ms": 4.5, "reported_ms": 8.0,
                                       "block_ms": 5.333, "calibrated_ms": None}
    assert runner.home()["hear_me"]["reported_ms"] == 8.0


def test_a_device_only_ever_shows_the_delay_it_reported_itself(rig, runner):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, SCARLETT]
    mic = next(d for d in runner.home()["devices"] if d["name"] == LAPTOP_MIC["name"])
    runner.start({"device": mic["id"], "hear_me": False})
    (rig.folder / realtime.REC_STREAM).write_text(json.dumps({"reported_ms": 8.3}))
    runner.tick()
    home = runner.home()
    by_name = {d["name"]: d.get("reported_ms") for d in home["devices"]}
    assert by_name == {LAPTOP_MIC["name"]: 8.3, SCARLETT["name"]: None}
    # The default here is the interface, which has reported nothing yet.
    assert home["hear_me"]["reported_ms"] is None


@pytest.mark.parametrize("junk", ['{"reported_ms": "lots"}', '{"reported_ms": true}', "[1]", "not json"])
def test_a_strange_stream_file_is_no_reading(rig, runner, junk):
    runner.start({})
    (rig.folder / "stream.json").write_text(junk)
    runner.tick()
    stream = runner.live()["stream"]
    assert stream is None or stream["reported_ms"] is None


# ----------------------------------------------------------------------------- hints

def test_no_count_after_a_minute(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    assert runner.live()["hint"] is None
    for _ in range(12):
        rig.clock.t += 5.0
        rig.more_audio()
        runner.tick()
    assert runner.live()["hint"] == guitake.NO_COUNT
    assert runner.live()["state"] == "waiting"


def test_a_flat_channel_is_named(rig, runner):
    runner.start({"channel": 2})
    rig.open_stream(level=0.0)
    runner.tick()
    for _ in range(4):
        rig.clock.t += 5.0
        rig.more_audio(level=0.0)
        runner.tick()
    assert runner.live()["hint"] == "Nothing on channel 2. Press Stop, then try another channel in the menu."


@pytest.mark.parametrize("devices", [
    [LAPTOP_MIC, LAPTOP_OUT],                                     # one channel, chosen for her
    [BLACKHOLE, {"name": "USB Mic", "inputs": 1, "outputs": 0}],  # the system default
])
def test_a_flat_input_with_no_channel_to_change_says_what_could_help(rig, runner, devices):
    rig.devices = devices
    runner.start({})
    rig.open_stream(level=0.0)
    runner.tick()
    for _ in range(4):
        rig.clock.t += 5.0
        rig.more_audio(level=0.0)
        runner.tick()
    hint = runner.live()["hint"]
    assert hint == guitake.FLAT_MIC and "channel" not in hint and "microphone" in hint


def _wait_flat(rig, runner, seconds=20.0, level=0.0):
    rig.open_stream(level=level)
    runner.tick()
    for _ in range(int(seconds / 5.0)):
        rig.clock.t += 5.0
        rig.more_audio(level=level)
        runner.tick()


def _stop_and_land(rig, runner):
    runner.stop()
    assert wait_until(lambda: rig.proc.code is not None, timeout=5.0)
    rig.die(rig.proc.code)
    runner.tick()


def test_stopping_before_the_band_on_a_flat_channel_leaves_no_take_and_says_why(rig, runner):
    runner.start({"channel": 2})
    _wait_flat(rig, runner)
    folder = rig.folder
    _stop_and_land(rig, runner)
    assert runner.live() == {"state": "idle", "menu": True,
                             "message": "Nothing came in on channel 2. Try another channel."}
    assert not folder.exists()


def test_stopping_before_the_band_after_a_little_playing_leaves_no_take(rig, runner):
    runner.start({})
    _wait_flat(rig, runner, seconds=10.0, level=0.2)
    folder = rig.folder
    _stop_and_land(rig, runner)
    assert runner.live() == {"state": "idle", "message": guitake.KEPT_NOTHING}
    assert not folder.exists()


def test_stopping_at_once_says_nothing_at_all(rig, runner):
    runner.start({})
    rig.open_stream(level=0.0)
    runner.tick()
    folder = rig.folder
    _stop_and_land(rig, runner)
    assert runner.live() == {"state": "idle"}
    assert not folder.exists()


def test_a_long_wait_with_real_playing_keeps_her_recording(rig, runner):
    runner.start({})
    rig.open_stream(level=0.2, seconds=guitake.KEEP_UNCOUNTED_S)
    runner.tick()
    folder = rig.folder
    _stop_and_land(rig, runner)
    assert runner.live() == {"state": "done", "take": f"id-of-{folder.name}", "note": None}
    assert (folder / "human.wav").exists()


def test_a_long_silence_is_still_no_take(rig, runner):
    runner.start({})
    rig.open_stream(level=0.0, seconds=guitake.KEEP_UNCOUNTED_S)
    runner.tick()
    folder = rig.folder
    _stop_and_land(rig, runner)
    assert not folder.exists() and runner.live()["state"] == "idle"


def test_a_crash_before_the_band_says_so_and_leaves_no_take(rig, runner):
    runner.start({})
    rig.open_stream(level=0.2)
    runner.tick()
    folder = rig.folder
    rig.die(1, "Traceback: PortAudio error opening stream")
    runner.tick()
    assert runner.live() == {"state": "failed", "message": guitake.OPEN_FAILED}
    assert not folder.exists()


def test_a_device_that_stalls_before_the_band_says_so_on_home(rig, runner):
    runner.start({})
    rig.open_stream(level=0.2)
    runner.tick()
    folder = rig.folder
    rig.clock.t += guitake.STALL_S
    runner.tick()
    assert wait_until(lambda: rig.proc.code is not None, timeout=5.0)
    rig.die(0)
    runner.tick()
    assert runner.live() == {"state": "idle", "message": guitake.STALLED}
    assert not folder.exists()


def test_only_a_takes_own_files_are_cleared_with_a_false_start(rig, runner):
    runner.start({})
    rig.open_stream(level=0.0)
    runner.tick()
    folder = rig.folder
    (folder / "mine.txt").write_text("not plume's")
    _stop_and_land(rig, runner)
    assert (folder / "mine.txt").exists() and runner.live()["state"] == "idle"


def test_one_sound_clears_the_flat_channel_hint(rig, runner):
    runner.start({})
    rig.open_stream(level=0.0)
    runner.tick()
    rig.clock.t += 5.0
    rig.more_audio(level=0.2)
    runner.tick()
    for _ in range(3):
        rig.clock.t += 5.0
        rig.more_audio(level=0.0)
        runner.tick()
    assert runner.live()["hint"] is None


def test_a_slow_open_says_why_it_may_be_waiting(rig, runner):
    runner.start({})
    rig.clock.t += guitake.OPENING_AFTER_S
    runner.tick()
    assert runner.live()["hint"] == guitake.OPENING
    assert runner.live()["state"] == "starting"


# ----------------------------------------------------------------- how a take can end

def test_a_natural_end_lands_on_the_record(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    folder = rig.folder
    rig.hit_first_bar()
    rig.die(0)
    runner.tick()
    assert runner.live() == {"state": "done", "take": f"id-of-{folder.name}", "note": None}
    assert folder.is_dir()


def test_a_crash_with_audio_still_lands_on_the_record(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    rig.hit_first_bar()
    rig.die(1, "Traceback (most recent call last):\n  File \"/secret/path\"\nboom")
    runner.tick()
    doc = runner.live()
    assert doc["state"] == "done" and doc["note"] == guitake.ENDED_EARLY
    assert "Traceback" not in json.dumps(doc) and "/secret" not in json.dumps(doc)


@pytest.mark.parametrize("stderr,message", [
    ("error: no audio device matching 'x' (devices: y)", guitake.DEVICE_MISSING),
    ("error: 'x' has no inputs", guitake.DEVICE_MISSING),
    ("error: --input-channel 3: x has channels 1 to 2", guitake.DEVICE_MISSING),
    ("error: Error opening Stream: Invalid sample rate [PaErrorCode -9997]", guitake.OPEN_FAILED),
    ("error: PortAudio not initialized", guitake.OPEN_FAILED),
    ("error: sounddevice is not available — live audio", guitake.AUDIO_PART_MISSING),
    ("error: refusing to play live audio: PLUME_ALLOW_AUDIO is set to '0'.", guitake.SOUND_OFF),
    ("error: another plume session is already listening at /x", guitake.TERMINAL_TAKE),
    ("Traceback (most recent call last):\n  File \"/secret/path\", line 1\nValueError: boom",
     guitake.DID_NOT_START),
    ("", guitake.DID_NOT_START),
])
def test_a_failure_with_no_audio_has_only_plain_words(rig, runner, stderr, message):
    runner.start({})
    folder = rig.folder
    rig.die(1, stderr)
    runner.tick()
    doc = runner.live()
    assert doc == {"state": "failed", "message": message}
    assert not folder.exists()


def test_a_failed_start_removes_only_its_own_empty_files(rig, runner):
    runner.start({})
    folder = rig.folder
    (folder / "session.json").write_text("{}")
    (folder / "decisions.ndjson").write_text("")
    (folder / "cues.ndjson").write_text("")
    with sf.SoundFile(str(folder / "human.wav"), mode="w", samplerate=SR, channels=1):
        pass
    with sf.SoundFile(str(folder / "machine.wav"), mode="w", samplerate=SR, channels=1):
        pass
    (folder / "hers.txt").write_text("not ours")
    rig.die(1)
    runner.tick()
    assert sorted(p.name for p in folder.iterdir()) == ["hers.txt"]


def test_a_failed_start_leaves_a_symlink_alone(rig, runner, tmp_path):
    other = tmp_path / "other.json"
    other.write_text("hers")
    runner.start({})
    (rig.folder / "session.json").symlink_to(other)
    link = rig.folder / "session.json"
    rig.die(1)
    runner.tick()
    assert link.is_symlink() and other.read_text() == "hers"


@pytest.mark.parametrize("name", ["voice.wav", "human.wav", "machine.wav"])
def test_clearing_a_folder_never_takes_a_recording_with_sound_in_it(tmp_path, name):
    sf.write(str(tmp_path / name), np.zeros(100, dtype=np.float32), SR)
    guitake._clear_fresh_folder(tmp_path)
    assert (tmp_path / name).exists()


@pytest.mark.parametrize("name", ["decisions.ndjson", "cues.ndjson", "midi.ndjson"])
def test_clearing_a_folder_never_takes_a_log_with_lines_in_it(tmp_path, name):
    (tmp_path / name).write_text('{"bar": 0}\n')
    guitake._clear_fresh_folder(tmp_path)
    assert (tmp_path / name).exists()


def test_consent_off_is_said_plainly_and_leaves_no_folder(rig, runner):
    rig.spawn_error = guitake.SoundOff(guitake.SOUND_OFF)
    assert runner.start({}) == {"state": "failed", "message": guitake.SOUND_OFF}
    assert not rig.root.exists() or not list(rig.root.iterdir())


@pytest.mark.parametrize("error", [OSError("/secret/path"), guitake.TakeError("/secret"), ValueError("x")])
def test_a_spawn_failure_is_said_plainly(rig, runner, error):
    rig.spawn_error = error
    assert runner.start({}) == {"state": "failed", "message": guitake.DID_NOT_START}


def test_failing_does_not_stop_the_next_start(rig, runner):
    rig.spawn_error = OSError("nope")
    runner.start({})
    rig.spawn_error = None
    assert runner.start({})["state"] == "starting"


# ------------------------------------------------------------------------------ stop

def test_stop_uses_the_session_record_and_lands_on_the_record(rig, runner):
    runner.start({})
    rig.open_stream()
    runner.tick()
    rig.hit_first_bar()
    rig.human.close()
    rig.human = None
    assert runner.stop()["state"] in ("stopping", "done")
    assert wait_until(lambda: rig.stopped and rig.proc.code == 0)
    assert rig.stopped[0].pid == rig.proc.pid and rig.stopped[0].socket == "/nowhere/sock"
    runner.tick()
    assert runner.live()["state"] == "done"


def test_double_stop_sends_once(rig):
    release = threading.Event()
    calls = []

    def slow(rec):
        calls.append(rec.pid)
        release.wait(5)

    runner = rig.runner(stopper=slow)
    runner.start({})
    for _ in range(5):
        assert runner.stop()["state"] == "stopping"
    release.set()
    assert wait_until(lambda: calls)
    time.sleep(0.05)
    assert len(calls) == 1


def test_stop_with_nothing_running_is_fine(rig, runner):
    assert runner.stop() == {"state": "idle"}
    assert rig.stopped == []


def test_stop_before_the_child_is_registered_still_signals_it(rig, runner):
    runner = rig.runner(sessions=lambda: [])
    runner.start({})
    runner.stop()
    assert wait_until(lambda: rig.stopped)
    assert rig.stopped[0].pid == rig.proc.pid and rig.stopped[0].socket is None


def test_stop_while_the_devices_are_being_listed_starts_nothing(rig):
    gate = threading.Event()
    entered = threading.Event()

    def slow():
        entered.set()
        gate.wait(5)
        return [SCARLETT]

    runner = rig.runner(lister=slow)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(runner.start({})))
    thread.start()
    assert entered.wait(5)
    assert runner.stop()["state"] == "starting"
    gate.set()
    thread.join(5)
    assert rig.spawned == [] and outcome == [{"state": "idle"}]
    assert not list(rig.root.iterdir())


def test_stop_before_any_audio_goes_back_home_with_no_folder(rig, runner):
    runner.start({})
    folder = rig.folder
    runner.stop()
    assert wait_until(lambda: rig.proc.code == 0)
    runner.tick()
    assert runner.live() == {"state": "idle"}
    assert not folder.exists()


def test_a_wedged_stop_says_what_to_do(rig):
    rig.stop_hangs = True
    runner = rig.runner()
    runner.start({})
    runner.stop()
    assert wait_until(lambda: rig.stopped)
    assert runner.live()["message"] is None
    rig.clock.t += guitake.WEDGED_S
    doc = runner.live()
    assert doc["state"] == "stopping" and doc["message"] == guitake.WEDGED


def test_a_stopper_that_raises_is_still_only_stopping(rig):
    def bad(rec):
        raise RuntimeError("/secret/path")

    runner = rig.runner(stopper=bad)
    runner.start({})
    assert runner.stop()["state"] == "stopping"
    time.sleep(0.02)
    doc = runner.live()
    assert doc["state"] == "stopping" and "/secret" not in json.dumps(doc)


# ------------------------------------------------------------------- stalls and pages

def _to_waiting(rig, runner, **body):
    runner.start(body)
    rig.open_stream()
    runner.tick()
    assert runner.live()["state"] == "waiting"


def test_a_stalled_device_is_stopped_and_named_on_the_record(rig, runner):
    _to_waiting(rig, runner)
    rig.clock.t += guitake.STALL_S - 0.5
    runner.tick()
    assert runner.live()["state"] == "waiting"
    rig.clock.t += 0.5
    runner.tick()
    assert wait_until(lambda: rig.stopped)
    rig.hit_first_bar()
    rig.die(0)
    runner.tick()
    doc = runner.live()
    assert doc["state"] == "done" and doc["note"] == guitake.STALLED


def test_a_growing_recording_is_not_a_stall(rig, runner):
    _to_waiting(rig, runner)
    for _ in range(10):
        rig.clock.t += guitake.STALL_S - 1.0
        rig.more_audio()
        runner.live()
        runner.tick()
    assert rig.stopped == []


def test_a_slow_open_is_not_a_stall(rig, runner):
    runner.start({})
    rig.clock.t += 100.0
    runner.live()
    runner.tick()
    assert rig.stopped == []


def test_a_take_that_has_written_its_ending_is_not_a_stall(rig, runner):
    _to_waiting(rig, runner)
    (rig.folder / "ending.json").write_text("{}")
    rig.clock.t += guitake.STALL_S * 2
    runner.live()
    runner.tick()
    assert rig.stopped == []


def _keep_audio_going(rig, runner, seconds, *, poll=False):
    step = 5.0
    for _ in range(int(seconds / step)):
        rig.clock.t += step
        rig.more_audio()
        if poll:
            runner.live()
        runner.tick()


def test_no_page_while_waiting_stops_the_take(rig, runner):
    _to_waiting(rig, runner)
    _keep_audio_going(rig, runner, guitake.HEARTBEAT_S)
    assert wait_until(lambda: rig.stopped)
    rig.die(0)
    runner.tick()
    assert runner.live()["message"] == guitake.NO_PAGE


def test_a_page_polling_keeps_a_waiting_take_alive(rig, runner):
    _to_waiting(rig, runner)
    _keep_audio_going(rig, runner, guitake.HEARTBEAT_S * 3, poll=True)
    assert rig.stopped == []


def test_home_is_not_a_page_asking_after_the_take(rig, runner):
    _to_waiting(rig, runner)
    for _ in range(int(guitake.HEARTBEAT_S / 5.0)):
        rig.clock.t += 5.0
        rig.more_audio()
        runner.home()
        runner.live(poll=False)
        runner.tick()
    assert wait_until(lambda: rig.stopped)


def test_no_page_on_a_room_mic_stops_a_playing_take(rig):
    rig.devices = [LAPTOP_MIC]
    runner = rig.runner()
    _to_waiting(rig, runner)
    rig.hit_first_bar()
    runner.tick()
    assert runner.live()["state"] == "playing"
    _keep_audio_going(rig, runner, guitake.HEARTBEAT_S)
    assert wait_until(lambda: rig.stopped)


def test_no_page_on_a_direct_input_lets_the_take_fade_by_itself(rig, runner):
    _to_waiting(rig, runner)
    rig.hit_first_bar()
    runner.tick()
    _keep_audio_going(rig, runner, guitake.HEARTBEAT_S * 2)
    assert rig.stopped == []


# ----------------------------------------------------------------------------- cues

def _to_playing(rig, runner):
    _to_waiting(rig, runner)
    rig.hit_first_bar()
    runner.tick()
    assert runner.live()["state"] == "playing"


@pytest.mark.parametrize("cue", ["busier", "sparser"])
def test_a_cue_goes_to_this_takes_socket_marked_as_the_page(rig, runner, cue):
    _to_playing(rig, runner)
    assert runner.cue({"cue": cue}) == {"ok": True, "cue": cue}
    assert rig.cues == [(cue, "/nowhere/sock", "gui")]


@pytest.mark.parametrize("cue", ["darker", "go to the chorus", "", None, 7, ["busier"]])
def test_only_the_offered_cues_get_through(rig, runner, cue):
    _to_playing(rig, runner)
    with pytest.raises(guitake.Refused) as e:
        runner.cue({"cue": cue})
    assert e.value.status == 400 and rig.cues == []


def test_no_cue_without_a_playing_take(rig, runner):
    with pytest.raises(guitake.Refused) as e:
        runner.cue({"cue": "busier"})
    assert e.value.status == 409
    _to_waiting(rig, runner)
    with pytest.raises(guitake.Refused):
        runner.cue({"cue": "busier"})
    assert rig.cues == []


def test_a_cue_never_reaches_a_take_from_a_terminal(rig, runner):
    rig.others = [realtime.LiveSession(pid=77, kind="lead", socket="/nowhere/other")]
    with pytest.raises(guitake.Refused):
        runner.cue({"cue": "busier"})
    assert rig.cues == []


def test_a_cue_the_band_does_not_answer_is_said_plainly(rig):
    def absent(text, *, socket_path, source):
        raise realtime.RealtimeError(f"no running plume session at {socket_path}")

    runner = rig.runner(cue=absent)
    _to_playing(rig, runner)
    with pytest.raises(guitake.Refused) as e:
        runner.cue({"cue": "busier"})
    assert e.value.status == 409 and "/nowhere" not in e.value.message


def test_the_real_send_cue_takes_what_the_runner_passes(tmp_path):
    """Over a real socket, with no session and no audio."""
    server = realtime.CueServer(short_socket_path("guitake-cue"), kind="lead")
    server.start()
    try:
        realtime.send_cue("busier", socket_path=server.path, source="gui")
        assert wait_until(lambda: server._received == 1)
    finally:
        server.close()


# ------------------------------------------------------------- closing the bridge

def test_shutdown_stops_the_take_and_says_so(rig, runner):
    runner.start({})
    assert runner.shutdown(timeout=5.0) is True
    assert [r.pid for r in rig.stopped] == [rig.proc.pid]
    assert runner.live()["state"] == "idle"


def test_shutdown_with_nothing_running_is_instant(rig, runner):
    assert runner.shutdown() is True and rig.stopped == []


def test_shutdown_gives_up_rather_than_hang_and_says_so(rig):
    rig.stop_hangs = True
    runner = rig.runner()
    runner.start({})
    assert runner.shutdown(timeout=0.05) is False


def test_only_the_first_shutdown_waits(rig):
    rig.stop_hangs = True
    runner = rig.runner()
    runner.start({})
    assert runner.shutdown(timeout=0.01) is False
    began = time.monotonic()
    assert runner.shutdown(timeout=30.0) is False
    assert time.monotonic() - began < 1.0


def test_no_start_while_closing(rig, runner):
    runner.shutdown()
    with pytest.raises(guitake.Refused) as e:
        runner.start({})
    assert e.value.status == 409 and rig.spawned == []


@pytest.fixture
def plumed_home(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))


@pytest.mark.parametrize("how", ["ctrl-c", signal.SIGTERM, signal.SIGHUP])
def test_closing_the_bridge_stops_the_take_it_started(rig, how, plumed_home, capsys):
    def wait(server):
        server.takes.start({})
        assert rig.proc.code is None
        if how == "ctrl-c":
            raise KeyboardInterrupt
        os.kill(os.getpid(), how)
        time.sleep(5.0)
        pytest.fail("the signal did not arrive")

    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    gui.serve(takes=rig.root, open_browser=False, wait=wait, spawner=rig.spawner, lister=rig.lister,
              sessions=rig.sessions, stopper=rig.stopper)
    assert rig.proc.code == 0
    assert [r.pid for r in rig.stopped] == [rig.proc.pid]
    assert {s: signal.getsignal(s) for s in before} == before
    assert "Stopping the take" in capsys.readouterr().out


def test_a_second_ctrl_c_leaves_and_says_how_to_stop_it(rig, plumed_home, capsys):
    rig.stop_hangs = True
    original = guitake.TakeRunner.shutdown
    calls = []

    def interrupted(self, **kw):
        calls.append(1)
        if len(calls) == 1:
            self._closing = True
            raise KeyboardInterrupt
        return original(self, **kw)

    def wait(server):
        server.takes.start({})
        raise KeyboardInterrupt

    guitake.TakeRunner.shutdown = interrupted
    try:
        gui.serve(takes=rig.root, open_browser=False, wait=wait, spawner=rig.spawner, lister=rig.lister,
                  sessions=rig.sessions, stopper=rig.stopper)
    finally:
        guitake.TakeRunner.shutdown = original
    assert "plume stop" in capsys.readouterr().out


def test_stopping_a_real_child_reaps_it(rig, tmp_path, plume_registry):
    """A sleeping Python child, not plume.  stop_session waits for the pid to go, and an
    unreaped child still has one, so this hangs if nothing is reaping it."""
    code = ("import signal,sys,time\n"
            "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))\n"
            "print('up', flush=True)\n"
            "time.sleep(30)\n")
    children = []

    def inert(argv, log_path):
        child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, start_new_session=True)
        child.stdout.readline()
        children.append(child)
        return child

    runner = rig.runner(spawner=inert, sessions=lambda: [], stopper=None, watch=True, tick_s=0.02,
                        clock=time.monotonic)
    try:
        assert runner.start({})["state"] == "starting"
        runner.stop()
        assert wait_until(lambda: runner.live()["state"] == "idle", timeout=10.0)
        assert children[0].returncode == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()


class ThreadProc:
    """The child, as a thread on the fake device.  It shares this test's pid, so the real
    ``stop_session`` asks over the real socket and signals nothing."""

    def __init__(self, target):
        self.pid = os.getpid()
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def poll(self):
        return None if self.thread.is_alive() else 0


def test_the_whole_way_on_the_fake_device(tmp_path, monkeypatch, plume_registry):
    """Real argv, real transport, real recorder, real registry, socket, stop and reader.

    The one fake is the audio device, as everywhere.  This says nothing about real hardware.
    """
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")      # the FAKE device only
    from test_detect import _strummed_take  # noqa: PLC0415

    def spawner(argv, log_path):  # noqa: ARG001
        extra = [f"--socket={short_socket_path('guiwhole')}", "--max-minutes=1"]
        return ThreadProc(lambda: cli.main([*argv[4:], *extra]))

    root = tmp_path / "shelf"
    runner = guitake.TakeRunner(
        root, log_path=lambda: tmp_path / "take.log", id_of=lambda folder: folder.name, spawner=spawner,
        lister=lambda: [{"name": "Audio Interface 4x4", "inputs": 4, "outputs": 4}], tick_s=0.02)
    devices = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]
    seen = []

    def state():
        doc = runner.live()
        seen.append(doc["state"])
        return doc

    with patched_module("sounddevice", fake_audio(_strummed_take(115.0), devices=devices, rate_x=12.0)):
        assert runner.start({"hear_me": True, "headphones": True})["state"] == "starting"
        assert wait_until(lambda: state()["state"] in ("waiting", "playing"), timeout=20.0)
        assert wait_until(lambda: state()["level_db"] is not None, timeout=20.0)
        assert wait_until(lambda: state()["stream"] is not None, timeout=20.0)
        assert state()["stream"]["block_ms"] == pytest.approx(5.333, abs=0.01)
        assert wait_until(lambda: state()["state"] == "playing", timeout=20.0)
        assert wait_until(lambda: state()["row"] is not None, timeout=20.0)
        row = state()["row"]
        assert row["input"] == "direct" and row["style_guess"] is True and row["bpm"] > 0
        assert runner.cue({"cue": "busier"}) == {"ok": True, "cue": "busier"}
        folder = next(root.iterdir())
        assert runner.stop()["state"] == "stopping"
        assert wait_until(lambda: state()["state"] == "done", timeout=30.0)
    assert seen[0] != "playing" and "playing" in seen
    assert runner.live() == {"state": "done", "take": folder.name, "note": None}
    assert json.loads((folder / "ending.json").read_text())["reason"] == realtime.END_STOPPED
    assert json.loads((folder / "cues.ndjson").read_text().splitlines()[0])["source"] == "gui"


# ---------------------------------------------------------- the real spawner's guards

def _booby_trap(monkeypatch):
    def trap(*a, **k):
        raise AssertionError("a real process was started")

    monkeypatch.setattr(guitake.subprocess, "Popen", trap)
    monkeypatch.setattr(guitake.subprocess, "run", trap)


def test_the_real_spawner_will_not_run_under_pytest(monkeypatch, tmp_path):
    _booby_trap(monkeypatch)
    with pytest.raises(guitake.TakeError):
        guitake.default_spawner(["echo"], tmp_path / "take.log")
    assert not (tmp_path / "take.log").exists()


def test_the_real_lister_will_not_run_under_pytest(monkeypatch):
    _booby_trap(monkeypatch)
    with pytest.raises(guitake.TakeError):
        guitake.default_lister()


def test_the_default_runner_under_pytest_starts_nothing(rig, monkeypatch):
    _booby_trap(monkeypatch)
    runner = guitake.TakeRunner(rig.root, log_path=lambda: rig.state_dir / "take.log",
                                id_of=lambda f: None, sessions=lambda: [], lister=rig.lister, watch=False)
    assert runner.start({}) == {"state": "failed", "message": guitake.DID_NOT_START}
    with pytest.raises(guitake.TakeError):
        guitake.default_lister()


def _fake_popen(monkeypatch):
    """The real spawner's body, with the guard lifted and Popen replaced: still no process."""
    seen = {}

    def popen(argv, **kw):
        seen.update(kw, argv=argv)
        seen["log_mode"] = stat.S_IMODE(os.fstat(kw["stderr"]).st_mode)
        return FakeProc(5)

    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setattr(guitake.subprocess, "Popen", popen)
    return seen


def test_the_click_is_the_consent(monkeypatch, tmp_path):
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    seen = _fake_popen(monkeypatch)
    assert guitake.default_spawner(["x", "--y"], tmp_path / "state" / "take.log").pid == 5
    assert seen["argv"] == ["x", "--y"]
    assert seen["env"][realtime.AUDIO_CONSENT_ENV] == "1"
    assert seen["stdin"] == subprocess.DEVNULL and seen["start_new_session"] is True
    assert seen["log_mode"] == 0o600
    assert "shell" not in seen
    assert realtime.AUDIO_CONSENT_ENV not in os.environ


@pytest.mark.parametrize("off", ["0", "false", "NO", " off "])
def test_an_explicit_no_beats_the_click(monkeypatch, tmp_path, off):
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, off)
    seen = _fake_popen(monkeypatch)
    with pytest.raises(guitake.SoundOff):
        guitake.default_spawner(["x"], tmp_path / "take.log")
    assert seen == {}


def test_a_symlink_at_the_log_is_not_followed(monkeypatch, tmp_path):
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    _fake_popen(monkeypatch)
    target = tmp_path / "hers.txt"
    target.write_text("hers")
    (tmp_path / "take.log").symlink_to(target)
    with pytest.raises(OSError):
        guitake.default_spawner(["x"], tmp_path / "take.log")
    assert target.read_text() == "hers"


def test_the_lister_reads_the_json_and_never_asks_for_consent(monkeypatch):
    seen = {}

    def run(argv, **kw):
        seen.update(kw, argv=argv)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"devices": [SCARLETT]}).encode())

    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setattr(guitake.subprocess, "run", run)
    assert guitake.default_lister() == [SCARLETT]
    assert seen["argv"] == [sys.executable, "-P", "-m", "plume", "inputs", "--json"]
    assert seen["timeout"] == guitake.LIST_TIMEOUT_S
    assert "--meter" not in seen["argv"] and "env" not in seen


@pytest.mark.parametrize("stdout,message", [
    (b"not json", guitake.NO_LISTING), (b"[]", guitake.NO_LISTING), (b"{}", guitake.NO_LISTING),
    (b'{"error": "sounddevice is not available"}', guitake.AUDIO_PART_MISSING),
])
def test_a_lister_that_says_no_is_said_plainly(monkeypatch, stdout, message):
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setattr(guitake.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout=stdout))
    with pytest.raises(guitake.TakeError, match=message):
        guitake.default_lister()


def test_a_plume_py_in_the_folder_it_was_started_from_is_not_what_runs(tmp_path, monkeypatch):
    """Consent goes to the argv's child, so it has to be this plume and not the folder's."""
    (tmp_path / "plume.py").write_text("print('SHADOW')\n")
    argv = guitake.build_argv(guitake.Route(None, "room", 1, False), tmp_path, key=None, style=None, seed=1)
    src = str(Path(gui.__file__).parents[1])
    # --help opens no device: it stands in for the verb, on the argv's own prefix.
    done = subprocess.run([*argv[:argv.index("lead")], "--help"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=60, check=False, env={**os.environ, "PYTHONPATH": src})
    assert "SHADOW" not in done.stdout and done.returncode == 0


def test_a_lister_that_hangs_is_said_plainly(monkeypatch):
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setattr(guitake.subprocess, "run", run)
    with pytest.raises(guitake.TakeError, match=guitake.NO_LISTING):
        guitake.default_lister()


@pytest.fixture
def stub_sounddevice(tmp_path, monkeypatch):
    """A stand-in module the child finds first, so the real verb runs with no PortAudio."""
    def install(source):
        folder = tmp_path / "stub"
        folder.mkdir()
        (folder / "sounddevice.py").write_text(source)
        src = str(Path(gui.__file__).parents[1])
        monkeypatch.setenv("PYTHONPATH", f"{folder}{os.pathsep}{src}")
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
        monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    return install


def test_the_lister_reads_the_real_verb_with_no_real_device(stub_sounddevice):
    stub_sounddevice("def query_devices():\n"
                     "    return [{'name': 'Stub Interface', 'max_input_channels': 2,"
                     " 'max_output_channels': 2}]\n")
    listed = guitake.default_lister()
    assert listed == [{"name": "Stub Interface", "inputs": 2, "outputs": 2}]
    assert guitake.describe_devices(listed)[0]["kind"] == "interface"


def test_the_lister_names_the_missing_audio_part_from_the_real_verb(stub_sounddevice):
    stub_sounddevice("raise ImportError('no PortAudio here')\n")
    with pytest.raises(guitake.TakeError, match=guitake.AUDIO_PART_MISSING):
        guitake.default_lister()


def _functions_using(tree, predicate) -> set[str]:
    """Names of the top-level functions and methods holding a node that matches."""
    found = set()

    def visit(node, owner):
        name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
        if predicate(node):
            found.add(name)
        for child in ast.iter_child_nodes(node):
            visit(child, name)

    visit(tree, "<module>")
    return found


def test_consent_is_handled_in_one_function_only():
    """By syntax, so a comment cannot satisfy it and moving the line cannot dodge it."""
    tree = ast.parse(Path(guitake.__file__).read_text())
    names = _functions_using(tree, lambda n: isinstance(n, ast.Name) and n.id == "AUDIO_CONSENT_ENV")
    assert names == {"default_spawner", "default_meter"}
    text = _functions_using(tree, lambda n: isinstance(n, ast.Constant) and n.value == "PLUME_ALLOW_AUDIO")
    assert text == set()
    for source in (gui.__file__, guitake.__file__):
        assert "check_audio_consent" not in Path(source).read_text()


def test_only_a_listen_post_can_reach_the_meter():
    tree = ast.parse(Path(guitake.__file__).read_text())
    metering = _functions_using(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "_meter"
                                and isinstance(n.ctx, ast.Load))
    assert metering == {"one"}
    assert _functions_using(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "_read_meters") == {"listen"}
    bridge = ast.parse(Path(gui.__file__).read_text())
    callers = _functions_using(bridge, lambda n: isinstance(n, ast.Attribute) and n.attr == "listen"
                               and isinstance(n.value, ast.Name) and n.value.id == "takes")
    assert callers == {"_take_op"}


def test_only_start_can_reach_the_spawner():
    tree = ast.parse(Path(guitake.__file__).read_text())
    spawning = _functions_using(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "_spawn"
                                and isinstance(n.ctx, ast.Load))
    assert spawning == {"_start"}
    starting = _functions_using(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "_start")
    assert starting == {"start"}
    bridge = ast.parse(Path(gui.__file__).read_text())
    callers = _functions_using(bridge, lambda n: isinstance(n, ast.Attribute) and n.attr == "start"
                               and isinstance(n.value, ast.Name) and n.value.id == "takes")
    assert callers == {"_take_op"}
    assert _functions_using(bridge, lambda n: isinstance(n, ast.Attribute) and n.attr == "_take_op") == {"_post"}


def test_the_bridge_module_starts_no_process_of_its_own():
    source = Path(gui.__file__).read_text()
    for banned in ("subprocess", "Popen", "os.system", "os.exec", "os.spawn", "sounddevice"):
        assert banned not in source, banned


def test_importing_the_bridge_does_not_load_portaudio():
    code = "import sys, plume.gui, plume.guitake; assert 'sounddevice' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(Path(gui.__file__).parents[1])})


# ------------------------------------------------------------------ the routes, over HTTP

@pytest.fixture
def served(rig, tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    rig.root.mkdir()
    srv = gui.GuiServer(roots=[rig.root], log=lambda line: None, spawner=rig.spawner, lister=rig.lister,
                        sessions=rig.sessions, stopper=rig.stopper, cue=rig.cue, watch=False,
                        clock=rig.clock)
    srv.start()
    yield srv
    srv.close()


@pytest.fixture
def page(served):
    return Client(served).login()


ROUTES = [("GET", "/api/home"), ("GET", "/api/live"), ("POST", "/api/start"), ("POST", "/api/stop"),
          ("POST", "/api/cue")]


def _ask(client, method, path, **headers):
    if method == "GET":
        return client.get(path, headers=headers)[0].status
    send = {"Origin": client.origin, "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin",
            "X-Plume-Token": client.csrf, **headers}
    return client.request("POST", path, headers=send, body={})[0].status


@pytest.mark.parametrize("method,path", ROUTES)
def test_each_route_works_with_the_right_keys(rig, page, method, path):
    expected = 400 if path == "/api/cue" else 200        # {} names no cue
    assert _ask(page, method, path) == expected
    assert len(rig.spawned) == (1 if path == "/api/start" else 0)


@pytest.mark.parametrize("method,path", ROUTES)
def test_no_cookie_no_route(rig, served, method, path):
    assert _ask(Client(served), method, path) == 401
    assert rig.spawned == [] and rig.stopped == []


@pytest.mark.parametrize("method,path", ROUTES)
def test_a_cookie_without_the_page_key_is_refused(rig, served, page, method, path):
    anon = Client(served)
    anon.cookie = page.cookie
    assert _ask(anon, method, path) in (401, 403)
    assert rig.spawned == [] and rig.stopped == []


@pytest.mark.parametrize("method,path", ROUTES)
def test_a_wrong_page_key_is_refused(rig, page, method, path):
    assert _ask(page, method, path, **{"X-Plume-Token": "0" * 64}) in (401, 403)
    assert rig.spawned == []


@pytest.mark.parametrize("method,path", ROUTES)
@pytest.mark.parametrize("host", ["evil.example", "localhost", "127.0.0.1"])
def test_a_wrong_host_is_refused(rig, page, method, path, host):
    if method == "GET":
        status = page.get(path, host=host)[0].status
    else:
        status = page.request("POST", path, host=host, body={}, headers={
            "Origin": page.origin, "Content-Type": "application/json", "X-Plume-Token": page.csrf})[0].status
    assert status == 403 and rig.spawned == []


@pytest.mark.parametrize("method,path", ROUTES)
def test_a_wrong_origin_is_refused(rig, page, method, path):
    assert _ask(page, method, path, Origin="http://evil.example") == 403
    assert rig.spawned == []


@pytest.mark.parametrize("method,path", ROUTES)
@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])
def test_a_cross_site_fetch_is_refused(rig, page, method, path, site):
    assert _ask(page, method, path, **{"Sec-Fetch-Site": site}) == 403
    assert rig.spawned == []


@pytest.mark.parametrize("path", ["/api/start", "/api/stop", "/api/cue"])
def test_a_post_needs_an_origin_and_json(rig, page, path):
    assert _ask(page, "POST", path, Origin=None) == 403
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        assert _ask(page, "POST", path, **{"Content-Type": ctype}) == 415
    assert rig.spawned == [] and rig.stopped == []


@pytest.mark.parametrize("path", ["/api/start", "/api/stop", "/api/cue"])
def test_a_get_cannot_do_what_a_post_does(rig, page, path):
    assert page.get(path)[0].status == 404
    assert rig.spawned == []


def test_start_over_http_and_reattach_from_another_session(rig, served, page):
    resp, out = page.post("/api/start", {"hear_me": False})
    assert resp.status == 200 and json.loads(out)["state"] == "starting"
    resp, out = Client(served).login().get("/api/live")
    assert resp.status == 200 and json.loads(out)["device"] == SCARLETT["name"]
    assert len(rig.spawned) == 1


def test_a_second_start_over_http_is_a_409_with_state(rig, page):
    page.post("/api/start", {})
    resp, out = page.post("/api/start", {})
    doc = json.loads(out)
    assert resp.status == 409 and doc["error"] == guitake.ALREADY_RUNNING
    assert doc["live"]["state"] == "starting" and len(rig.spawned) == 1


@pytest.mark.parametrize("body,status", [
    ({"key": "--evil"}, 400), ({"style": "--evil"}, 400), ({"channel": 99}, 400),
    ({"hear_me": "yes"}, 400), ({"headphones": "yes"}, 400), ({"device": "--direct"}, 400), ({"prompt": "x"}, 400),
    ([], 400), (b"not json", 400), (b"", 400),
])
def test_a_bad_start_over_http_starts_nothing(rig, page, body, status):
    assert page.post("/api/start", body)[0].status == status
    assert rig.spawned == []


def test_an_oversized_start_is_refused(rig, page):
    assert page.post("/api/start", {"key": "x" * 20000})[0].status == 413
    assert rig.spawned == []


def test_home_over_http_has_what_the_page_needs(page, rig):
    resp, out = page.get("/api/home")
    doc = json.loads(out)
    assert resp.status == 200
    assert {"devices", "device", "mode", "channel", "hear_me", "folder", "styles", "problem"} <= set(doc)
    assert doc["styles"] == ["lofi", "midwest", "synthwave", "trap"]
    assert doc["folder"] == str(rig.root)


def test_a_new_take_lands_on_the_shelf_and_its_record(rig, served, page):
    page.post("/api/start", {})
    rig.open_stream()
    (rig.folder / "session.json").write_text(json.dumps({"kind": "lead", "started_at": 1800000000.0}))
    served.takes.tick()
    assert json.loads(page.get("/api/live")[1])["state"] == "waiting"
    rig.hit_first_bar()
    rig.die(0)
    served.takes.tick()
    doc = json.loads(page.get("/api/live")[1])
    assert doc["state"] == "done" and doc["note"] is None
    assert page.get(f"/api/take/{doc['take']}")[0].status == 200
    takes = json.loads(page.get("/api/takes")[1])["takes"]
    assert doc["take"] in [t["id"] for t in takes]


def test_stop_and_cue_over_http(rig, served, page):
    page.post("/api/start", {})
    rig.open_stream()
    rig.hit_first_bar()
    served.takes.tick()
    resp, out = page.post("/api/cue", {"cue": "busier"})
    assert resp.status == 200 and rig.cues == [("busier", "/nowhere/sock", "gui")]
    assert page.post("/api/cue", {"cue": "--evil"})[0].status == 400
    resp, out = page.post("/api/stop", {})
    assert resp.status == 200 and json.loads(out)["state"] in ("stopping", "done")
    assert wait_until(lambda: rig.stopped)


def test_no_refusal_body_holds_a_traceback_or_a_path(rig, page):
    rig.others = [realtime.LiveSession(pid=77, kind="lead", record_dir="/secret/take", socket="/secret/sock")]
    resp, out = page.post("/api/start", {})
    assert resp.status == 409
    text = out.decode()
    assert "Traceback" not in text and "/secret" not in text


def test_the_page_polling_over_http_is_what_keeps_a_take_alive(rig, served, page):
    page.post("/api/start", {})
    rig.open_stream()
    served.takes.tick()
    for _ in range(3):
        rig.clock.t += guitake.HEARTBEAT_S - 1.0
        rig.more_audio()
        page.get("/api/live")
        served.takes.tick()
    assert rig.stopped == []
    rig.clock.t += guitake.HEARTBEAT_S
    rig.more_audio()
    served.takes.tick()
    assert wait_until(lambda: rig.stopped)


def test_closing_the_server_stops_the_take(rig, served):
    served.takes.start({})
    served.close()
    assert rig.proc.code == 0 and len(rig.stopped) == 1


def test_the_existing_routes_still_need_a_take_in_the_body(page):
    assert page.post("/api/keep", {"kept": True})[0].status == 400
    assert page.post("/api/rename", {"name": "x"})[0].status == 400
