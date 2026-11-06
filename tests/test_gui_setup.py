"""The first press of Start: hearing her, asking about headphones, saving the answers.

No device opens here.  Every meter is a fake that hands back numbers, and the one test of
the real verb runs it against a stand-in ``sounddevice`` it first proves is the one loaded.
Nothing here says anything about real hardware.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import fake_audio, patched_module, short_socket_path, wait_until
from plume import cli, gui, guitake, realtime, roles as roles_mod
from test_gui_bridge import Client
from test_gui_take import AUDIO_BOX, LAPTOP_MIC, LAPTOP_OUT, VIRTUAL_DEVICE, Rig, _ask, _parsed

OTHER_BOX = {"name": "Other Box 2x2", "inputs": 2, "outputs": 2}


def levels(*peaks):
    return {"device": "x", "seconds": 4.0, "loudest_channel": 1,
            "channels": [{"channel": n + 1, "peak_db": p, "rms_db": p - 12.0}
                         for n, p in enumerate(peaks)]}


class FakeMeter:
    """Numbers by device name.  Records what would have been opened, and opens nothing."""

    def __init__(self):
        self.by_name = {}
        self.calls = []
        self.error = None
        self.gate = None
        self.saw_cancel = []
        self.barrier = None

    def __call__(self, name, seconds, cancelled):
        self.calls.append((name, seconds))
        if self.barrier is not None:
            self.barrier.wait(timeout=5.0)
        if self.gate is not None:
            self.gate.set()
            self.saw_cancel.append(cancelled.wait(timeout=5.0))
            return None
        if self.error is not None:
            raise self.error
        return self.by_name.get(name, levels(*[-90.0]))


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path / "shelf", tmp_path / "state")


@pytest.fixture
def meter():
    return FakeMeter()


@pytest.fixture
def setup_file(tmp_path):
    return tmp_path / "state" / "setup.json"


@pytest.fixture
def runner(rig, meter, setup_file):
    return rig.runner(meter=meter, setup_path=lambda: setup_file)


def good(meter, device=AUDIO_BOX, *peaks):
    meter.by_name[device["name"]] = levels(*(peaks or (-90.0, -12.0, -90.0, -90.0)))


def walk(runner, *steps):
    doc = None
    for step, answer in steps:
        doc = runner.answer({"step": step, "answer": answer})
    return doc


# -------------------------------------------------------------- step 1: play something

def test_the_first_look_asks_her_to_play_and_opens_nothing(runner, meter):
    doc = runner.setup()
    assert (doc["step"], doc["line"], doc["done"], doc["choices"]) == ("listen", "Play something.", False, [])
    assert doc["hear_me"] is False and doc["seconds"] == guitake.LISTEN_S
    runner.home()
    assert meter.calls == []


def test_it_picks_the_loudest_channel_with_no_questions(runner, meter):
    good(meter)
    doc = runner.listen({})
    assert doc["heard"]["result"] == "ok" and doc["heard"]["channel"] == 2
    assert (doc["channel"], doc["step"], doc["line"]) == (2, "headphones", "Headphones on?")
    assert meter.calls == [(AUDIO_BOX["name"], guitake.LISTEN_S)]


@pytest.mark.parametrize("peak,result,words", [
    (-0.2, "hot", "A bit hot. Turn the gain down a little and play again."),
    (-45.0, "low", "Barely hearing you. Turn the gain up a little and play again."),
])
def test_a_level_that_needs_a_hand_says_so_and_stays_on_the_step(runner, meter, peak, result, words):
    good(meter, AUDIO_BOX, peak, -90.0, -90.0, -90.0)
    doc = runner.listen({})
    assert (doc["heard"]["result"], doc["step"], doc["line"]) == (result, "listen", words)
    assert [c["answer"] for c in doc["choices"]] == ["again", "start"]


def test_a_mic_has_no_gain_to_turn(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -45.0)
    assert "the gain" not in runner.listen({})["line"]
    good(meter, LAPTOP_MIC, -0.1)
    assert "the gain" not in runner.listen({})["line"]


def test_a_wrong_gain_still_found_the_channel(rig, runner, meter):
    good(meter, AUDIO_BOX, -90.0, -0.2, -90.0, -90.0)
    runner.listen({})
    runner.start({})
    assert "--input-channel=2" in rig.spawned[-1]


def test_nothing_arriving_names_the_device_and_offers_the_others(rig, runner, meter):
    doc = runner.listen({})
    assert doc["heard"]["result"] == "nothing"
    assert AUDIO_BOX["name"] in doc["line"] and "gain" in doc["line"]
    assert [o["name"] for o in doc["others"]] == [LAPTOP_MIC["name"]]
    assert [c["answer"] for c in doc["choices"]] == ["again", "start"]


def test_a_low_level_offers_the_others_because_it_may_be_gear_she_is_not_using(rig, runner, meter):
    """Hum on an idle interface, while she plays into the laptop."""
    good(meter, AUDIO_BOX, -50.0, -90.0, -90.0, -90.0)
    doc = runner.listen({})
    assert doc["heard"]["result"] == "low"
    assert [o["name"] for o in doc["others"]] == [LAPTOP_MIC["name"]]
    good(meter, LAPTOP_MIC, -12.0)
    picked = runner.listen({"device": doc["others"][0]["id"]})
    assert picked["step"] == "headphones" and picked["device_name"] == LAPTOP_MIC["name"]


def test_a_hot_level_has_found_her_so_it_offers_nothing_else(rig, runner, meter):
    good(meter, AUDIO_BOX, -0.2, -90.0, -90.0, -90.0)
    assert runner.listen({})["others"] == []


@pytest.mark.parametrize("devices", [[], [LAPTOP_OUT]])
def test_with_nothing_plugged_in_the_one_next_step_is_to_plug_something_in(rig, runner, meter, devices):
    rig.devices = devices
    doc = runner.listen({})
    assert doc["step"] == "listen" and doc["heard"]["result"] == "failed"
    assert doc["line"] == "Plume could not find a microphone or an audio interface. Plug one in, then try again."
    assert doc["choices"] == [{"answer": "again", "label": "Try again"}] and meter.calls == []


def test_a_second_device_can_be_chosen_when_nothing_arrives(rig, runner, meter):
    doc = runner.listen({})
    good(meter, LAPTOP_MIC, -12.0)
    after = runner.listen({"device": doc["others"][0]["id"]})
    assert after["heard"]["result"] == "ok" and after["device_name"] == LAPTOP_MIC["name"]
    assert runner.home()["devices"][0]["name"] == LAPTOP_MIC["name"]
    assert next(d for d in runner.home()["devices"] if d["id"] == runner.home()["device"])["name"] == LAPTOP_MIC["name"]


def test_a_built_in_mic_names_the_permission_not_a_cable(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    line = runner.listen({})["line"]
    assert LAPTOP_MIC["name"] in line and "microphone" in line and "cable" not in line


def test_two_interfaces_listen_on_both_and_keep_the_one_that_heard_her(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, AUDIO_BOX, OTHER_BOX]
    good(meter, OTHER_BOX, -90.0, -12.0)
    meter.barrier = threading.Barrier(2)     # both at once: she plays once, not twice
    doc = runner.listen({})
    assert sorted(c[0] for c in meter.calls) == sorted([AUDIO_BOX["name"], OTHER_BOX["name"]])
    assert (doc["device_name"], doc["channel"]) == (OTHER_BOX["name"], 2)
    home = runner.home()
    assert next(d for d in home["devices"] if d["id"] == home["device"])["name"] == OTHER_BOX["name"]
    runner.start({"hear_me": False})
    assert f"--device={OTHER_BOX['name']}" in rig.spawned[-1] and "--input-channel=2" in rig.spawned[-1]


def test_no_more_than_a_few_devices_open_at_once(rig, runner, meter):
    rig.devices = [{"name": f"Box {c}", "inputs": 2, "outputs": 2} for c in "ABCDEFGH"]
    runner.listen({})
    assert len(meter.calls) == guitake.MAX_LISTEN_DEVICES


def test_never_a_virtual_device_unless_she_picks_it(rig, runner, meter):
    rig.devices = [VIRTUAL_DEVICE, LAPTOP_MIC, LAPTOP_OUT]
    runner.listen({})
    assert [c[0] for c in meter.calls] == [LAPTOP_MIC["name"]]


def test_a_lone_usb_mic_gets_heard_and_kept_rather_than_the_system_default(rig, runner, meter):
    mic = {"name": "USB Mic", "inputs": 1, "outputs": 0}
    rig.devices = [LAPTOP_OUT, mic]
    assert runner.home()["device"] is None
    good(meter, mic, -12.0)
    assert runner.listen({})["device_name"] == "USB Mic"
    runner.start({})
    assert "--input-device=USB Mic" in rig.spawned[-1]


@pytest.mark.parametrize("junk", [
    None, [], "x", {}, {"channels": "x"}, {"channels": [{"channel": 99, "peak_db": -12.0}]},
    {"channels": [{"channel": 1, "peak_db": float("nan")}]}, {"channels": [{"channel": True, "peak_db": -1.0}]},
    {"channels": [{"channel": 1, "peak_db": "loud"}]}, {"error": "boom"},
])
def test_a_strange_reading_fails_the_step_and_does_not_trap_her(rig, runner, meter, junk):
    meter.by_name[AUDIO_BOX["name"]] = junk
    doc = runner.listen({})
    assert doc["heard"]["result"] == "failed" and doc["line"] == guitake.NO_READING
    assert "start" in [c["answer"] for c in doc["choices"]]
    assert runner.start({})["state"] == "starting" and "--monitor" not in rig.spawned[-1]


def test_a_failed_child_says_only_this_files_words(rig, runner, meter):
    meter.error = guitake.TakeError(guitake.OPEN_FAILED)
    doc = runner.listen({})
    assert doc["heard"]["result"] == "failed" and doc["line"] == guitake.OPEN_FAILED


def test_a_meter_that_blows_up_is_only_a_failed_step(runner, meter):
    meter.error = RuntimeError("/nowhere/path secret")
    doc = runner.listen({})
    assert doc["heard"]["result"] == "failed" and "secret" not in json.dumps(doc)


@pytest.mark.parametrize("body", [{"device": "--direct"}, {"device": 7}, {"seconds": 60}, {"name": "x"}])
def test_a_bad_listen_opens_nothing(runner, meter, body):
    with pytest.raises(guitake.Refused) as e:
        runner.listen(body)
    assert e.value.status == 400 and meter.calls == []


def test_no_listening_while_a_take_runs(rig, runner, meter):
    runner.start({})
    with pytest.raises(guitake.Refused) as e:
        runner.listen({})
    assert e.value.status == 409 and e.value.live["state"] == "starting" and meter.calls == []


def test_no_listening_while_a_terminal_take_runs(rig, runner, meter):
    rig.others = [realtime.LiveSession(pid=4242, kind="lead")]
    with pytest.raises(guitake.Refused) as e:
        runner.listen({})
    assert e.value.status == 409 and e.value.message == guitake.TERMINAL_TAKE and meter.calls == []


def _listen_in_thread(runner, meter):
    meter.gate = threading.Event()
    answers = []
    worker = threading.Thread(target=lambda: answers.append(runner.listen({})), daemon=True)
    worker.start()
    assert meter.gate.wait(timeout=5.0)
    return worker, answers


def test_leaving_setup_closes_the_device(runner, meter):
    worker, answers = _listen_in_thread(runner, meter)
    assert runner.cancel_listen()["step"] == "listen"
    worker.join(timeout=5.0)
    assert not worker.is_alive() and meter.saw_cancel == [True]
    assert answers[0]["heard"]["result"] == "cancelled"
    assert runner._listening is None


def test_start_closes_the_listening_device_before_it_opens_one(rig, runner, meter):
    worker, answers = _listen_in_thread(runner, meter)
    assert runner.start({})["state"] == "starting"
    worker.join(timeout=5.0)
    assert meter.saw_cancel == [True] and len(rig.spawned) == 1


def test_a_second_listen_replaces_the_first_and_never_stacks(runner, meter):
    worker, answers = _listen_in_thread(runner, meter)
    meter.gate = None
    good(meter)
    doc = runner.listen({})
    worker.join(timeout=5.0)
    assert answers[0]["heard"]["result"] == "cancelled" and doc["heard"]["result"] == "ok"


def test_closing_the_bridge_closes_the_device(runner, meter):
    worker, _ = _listen_in_thread(runner, meter)
    assert runner.shutdown() is True
    worker.join(timeout=5.0)
    assert meter.saw_cancel == [True]


def test_the_real_meter_refuses_to_run_under_pytest():
    with pytest.raises(guitake.TakeError, match="does not run under pytest"):
        guitake.default_meter("Anything", 4.0, threading.Event())


def test_nothing_opens_once_it_was_called_off(monkeypatch):
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    monkeypatch.setattr(guitake.subprocess, "Popen", lambda *a, **k: pytest.fail("started a child"))
    gone = threading.Event()
    gone.set()
    assert guitake.default_meter("Anything", 4.0, gone) is None


def test_a_runner_with_no_meter_given_cannot_reach_a_device(rig):
    runner = rig.runner()
    assert runner.listen({})["heard"]["result"] == "failed"


def test_sound_off_wins_over_the_click(monkeypatch):
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "0")
    monkeypatch.setattr(guitake.subprocess, "Popen", lambda *a, **k: pytest.fail("started a child"))
    with pytest.raises(guitake.SoundOff):
        guitake.default_meter("Anything", 4.0, threading.Event())


@pytest.fixture
def stub_folder(tmp_path, monkeypatch):
    """A stand-in module, proved to be the one a child loads, before the real verb may run."""
    folder = tmp_path / "stub"
    folder.mkdir()
    marker = tmp_path / "rec-was-called"
    (folder / "sounddevice.py").write_text(
        "import numpy as np\n"
        f"MARK = {str(marker)!r}\n"
        "STUB = True\n"
        "def query_devices():\n"
        "    return [{'name': 'Stub Box', 'max_input_channels': 2, 'max_output_channels': 2}]\n"
        "def rec(frames, samplerate, channels, dtype, device):\n"
        "    open(MARK, 'w').write(str(device))\n"
        "    y = np.zeros((frames, channels), dtype='float32')\n"
        "    y[:, 1] = 0.25\n"
        "    return y\n"
        "def wait():\n"
        "    pass\n")
    src = str(Path(gui.__file__).parents[1])
    monkeypatch.setenv("PYTHONPATH", f"{folder}{os.pathsep}{src}")
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    probe = subprocess.run([sys.executable, "-P", "-c", "import sounddevice as s; print(s.STUB, s.__file__)"],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.split() == ["True", str(folder / "sounddevice.py")]
    return marker


def test_the_real_verb_is_read_properly_from_a_stand_in_device(stub_folder):
    doc = guitake.default_meter("Stub Box", 0.05, threading.Event())
    assert stub_folder.exists()
    assert guitake._reading(doc, {"inputs": 2}) == (2, pytest.approx(-12.0, abs=0.1))


def test_a_flag_shaped_device_name_stays_a_device_name_in_the_meter(stub_folder):
    with pytest.raises(guitake.TakeError, match=guitake.DEVICE_MISSING):
        guitake.default_meter("--seconds=60", 0.05, threading.Event())
    assert not stub_folder.exists()


def test_leaving_kills_a_real_child(stub_folder, tmp_path, monkeypatch):
    (tmp_path / "stub" / "sounddevice.py").write_text(
        "import time\nSTUB = True\n"
        "def query_devices():\n"
        "    return [{'name': 'Stub Box', 'max_input_channels': 2, 'max_output_channels': 2}]\n"
        "def rec(frames, samplerate, channels, dtype, device):\n"
        "    time.sleep(60)\n"
        "def wait():\n    pass\n")
    seen = {}
    real_popen = subprocess.Popen

    def popen(*a, **k):
        seen["proc"] = real_popen(*a, **k)
        return seen["proc"]

    monkeypatch.setattr(guitake.subprocess, "Popen", popen)
    cancelled = threading.Event()
    answers = []
    worker = threading.Thread(target=lambda: answers.append(guitake.default_meter("Stub Box", 30.0, cancelled)))
    worker.start()
    assert wait_until(lambda: "proc" in seen)
    cancelled.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive() and answers == [None] and seen["proc"].poll() is not None


# --------------------------------------------------------------- steps 2 to 5: the talk

@pytest.fixture
def heard(runner, meter):
    good(meter)
    runner.listen({})
    return runner


def test_no_headphones_means_hear_me_is_off_and_not_offered(heard):
    doc = walk(heard, ("headphones", "no"))
    assert (doc["step"], doc["hear_me"], doc["done"]) == ("count", False, True)


def test_can_hear_yourself_leaves_hear_me_off(heard):
    doc = walk(heard, ("headphones", "yes"))
    assert (doc["step"], doc["line"]) == ("hears_self", "Can you hear yourself?")
    doc = walk(heard, ("hears_self", "yes"))
    assert (doc["step"], doc["hear_me"]) == ("count", False)


def test_cannot_hear_yourself_asks_about_the_volume_first(heard):
    doc = walk(heard, ("headphones", "yes"), ("hears_self", "no"))
    assert (doc["step"], doc["line"], doc["hear_me"]) == ("volume", "Check your headphone volume.", False)
    assert [c["label"] for c in doc["choices"]] == ["That was it", "Still nothing"]


def test_the_volume_was_the_answer_so_hear_me_stays_off(heard):
    doc = walk(heard, ("headphones", "yes"), ("hears_self", "no"), ("volume", "fixed"))
    assert (doc["step"], doc["hear_me"]) == ("count", False)


def test_still_nothing_turns_hear_me_on(heard):
    doc = walk(heard, ("headphones", "yes"), ("hears_self", "no"), ("volume", "still"))
    assert (doc["step"], doc["hear_me"], doc["done"]) == ("count", True, True)
    assert heard.home()["hear_me"] == {"allowed": True, "default": True, "reported_ms": None}


def test_the_count_says_how_a_pickup_counts(rig, runner, meter):
    good(meter)
    runner.listen({})
    doc = walk(runner, ("headphones", "no"))
    assert doc["line"] == "Knock or clap 8 even beats." and "muted strings" in doc["note"]
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    assert walk(runner, ("headphones", "no"))["note"] is None


@pytest.mark.parametrize("step,answer", [("headphones", "maybe"), ("headphones", True), ("nope", "yes"), (None, "yes")])
def test_a_bad_answer_changes_nothing(heard, step, answer):
    before = heard.setup()
    with pytest.raises(guitake.Refused) as e:
        heard.answer({"step": step, "answer": answer})
    assert e.value.status == 400 and heard.setup() == before


@pytest.mark.parametrize("steps", [[("hears_self", "no")], [("volume", "still")],
                                   [("headphones", "no"), ("hears_self", "no")],
                                   [("headphones", "yes"), ("volume", "still")]])
def test_a_stale_page_cannot_answer_its_way_to_hear_me(heard, steps):
    with pytest.raises(guitake.Refused) as e:
        walk(heard, *steps)
    assert e.value.status == 409 and heard.setup()["hear_me"] is False


def test_answers_about_another_device_stay_with_that_device(rig, heard):
    mic = next(d for d in heard.home()["devices"] if d["name"] == LAPTOP_MIC["name"])
    doc = heard.answer({"device": mic["id"], "step": "headphones", "answer": "yes"})
    assert doc["device"] == mic["id"] and doc["step"] == "listen"
    assert heard.setup()["step"] == "headphones"
    assert heard.setup(mic["id"])["device_name"] == LAPTOP_MIC["name"]


def test_an_unknown_device_in_an_answer_is_refused(heard):
    with pytest.raises(guitake.Refused) as e:
        heard.answer({"device": "f" * 16, "step": "headphones", "answer": "yes"})
    assert e.value.status == 400


# ------------------------------------------------------------- step 4: wireless can lag

def test_a_wireless_name_gets_one_line_that_calls_itself_a_guess(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, {"name": "Her Wireless Earbuds", "inputs": 0, "outputs": 2, "default_output": True}]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    doc = walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    assert doc["step"] == "latency" and "a guess" in doc["line"] and "wired headphones" in doc["line"].lower()
    assert walk(runner, ("latency", "ok"))["step"] == "count"
    assert runner.setup()["step"] == "count"


def test_a_large_reported_delay_gets_the_same_line_with_its_number(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, {"name": "Odd Output", "inputs": 0, "outputs": 2, "default_output": True, "output_ms": 180.4}]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    line = walk(runner, ("headphones", "yes"), ("hears_self", "yes"))["line"]
    assert "180 ms" in line and "guess" not in line


def test_a_late_line_leads_with_what_to_do(rig, runner, meter):
    """The page makes its first sentence the big one."""
    rig.devices = [LAPTOP_MIC, {"name": "Odd Output", "inputs": 0, "outputs": 2, "default_output": True, "output_ms": 180.4}]
    good(meter, LAPTOP_MIC, -14.0)
    runner.listen({})
    doc = walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    assert doc["step"] == "latency" and doc["line"].startswith("Try wired headphones. ")


@pytest.mark.parametrize("output", [
    {"name": "Bluetooth Speaker", "inputs": 0, "outputs": 2, "default_output": True},
    {"name": "Odd Output", "inputs": 0, "outputs": 2, "default_output": True, "output_ms": 180.4},
])
def test_someone_on_speakers_is_never_told_to_try_wired_headphones(rig, runner, meter, output):
    rig.devices = [LAPTOP_MIC, output]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    doc = walk(runner, ("headphones", "no"))
    assert doc["step"] == "count" and doc["done"] is True


def test_a_question_that_does_not_finish_setup_waits_for_no_look_at_the_devices(rig, runner, meter):
    """Each look is a child that can take a second, and she has only just been heard."""
    good(meter)
    runner.listen({})
    walk(runner, ("headphones", "yes"))
    looks = []
    listed = runner._list
    runner._list = lambda: (looks.append(1), listed())[1]
    walk(runner, ("hears_self", "no"))
    assert looks == []
    rig.clock.t += guitake.LOOK_RECENT_S + 1.0
    walk(runner, ("hears_self", "no"))
    assert looks == [1], "a look from a while ago is looked at again"


def test_headphones_on_is_answered_about_the_devices_there_are_now(rig, runner, meter):
    good(meter)
    runner.listen({})
    looks = []
    listed = runner._list
    runner._list = lambda: (looks.append(1), listed())[1]
    walk(runner, ("headphones", "yes"))
    assert looks == [1]


def test_a_quick_wired_output_is_not_nagged(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, {"name": "Plain Output", "inputs": 0, "outputs": 2, "default_output": True, "output_ms": 12.0}]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    assert walk(runner, ("headphones", "yes"), ("hears_self", "yes"))["step"] == "count"


def test_a_wireless_output_elsewhere_says_nothing_about_a_direct_device(rig, runner, meter):
    rig.devices = [{**AUDIO_BOX, "output_ms": 6.0}, {"name": "Wireless Buds", "inputs": 0, "outputs": 2, "default_output": True}]
    good(meter)
    runner.listen({})
    assert walk(runner, ("headphones", "yes"), ("hears_self", "yes"))["step"] == "count"


def _take_reports(rig, runner, **stream):
    """A take whose stream says this about itself, and which then ends."""
    runner.start({})
    rig.open_stream()
    (rig.folder / guitake.REC_STREAM).write_text(json.dumps(stream))
    runner.tick()
    rig.die(0)
    runner.tick()


def test_the_output_delay_of_the_last_take_counts_too(rig, runner, meter):
    _take_reports(rig, runner, input_ms=3.0, output_ms=200.0, reported_ms=203.0)
    good(meter)
    runner.listen({})
    doc = walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    assert doc["step"] == "latency" and "200 ms" in doc["line"]


def test_an_input_delay_is_no_reason_to_try_wired_headphones(rig, runner, meter):
    """The threshold is about the output, and a stream reports both added together."""
    _take_reports(rig, runner, input_ms=45.0, output_ms=40.0, reported_ms=85.0)
    good(meter)
    runner.listen({})
    assert walk(runner, ("headphones", "yes"), ("hears_self", "yes"))["step"] == "count"


def test_a_take_reporting_a_delay_does_not_undo_a_finished_setup(rig, runner, meter):
    """After setup a press of Start goes straight to the count."""
    good(meter)
    runner.listen({})
    assert walk(runner, ("headphones", "yes"), ("hears_self", "yes"))["done"] is True
    _take_reports(rig, runner, input_ms=3.0, output_ms=200.0, reported_ms=203.0)
    assert runner.setup()["step"] == "count"
    assert next(d for d in runner.home()["devices"] if d["name"] == AUDIO_BOX["name"])["setup_done"] is True


def test_a_second_listen_on_a_finished_device_does_not_stop_at_the_delay_either(rig, runner, meter):
    good(meter)
    runner.listen({})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    _take_reports(rig, runner, input_ms=3.0, output_ms=200.0, reported_ms=203.0)
    rig.devices = [*rig.devices, OTHER_BOX]
    assert runner.listen({"device": next(d["id"] for d in runner.home()["devices"] if d["name"] == AUDIO_BOX["name"])})["step"] == "count"


def test_the_round_trip_shown_in_settings_is_still_the_whole_of_it(rig, runner):
    _take_reports(rig, runner, input_ms=45.0, output_ms=40.0, reported_ms=85.0)
    assert next(d for d in runner.home()["devices"] if d["name"] == AUDIO_BOX["name"])["reported_ms"] == 85.0


# ------------------------------------------------- something plugged in during the flow

WIRED = {"name": "Wired Headphones", "inputs": 0, "outputs": 2, "default_output": True}


def _heard_on_a_laptop(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -14.0)
    return runner.listen({})


def test_plugging_headphones_in_when_asked_keeps_every_answer(rig, runner, meter):
    """It asked, so she did.  The commonest path for someone who only saw the clip."""
    doc = _heard_on_a_laptop(rig, runner, meter)
    assert doc["step"] == "headphones"
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, WIRED]
    mic = {"device": doc["device"]}
    assert runner.answer({**mic, "step": "headphones", "answer": "yes"})["step"] == "hears_self"
    assert runner.answer({**mic, "step": "hears_self", "answer": "no"})["step"] == "volume"
    last = runner.answer({**mic, "step": "volume", "answer": "still"})
    assert last["done"] is True and last["hear_me"] is True
    assert len(meter.calls) == 1, "and she is not asked to play again"
    runner.start({"device": doc["device"]})
    assert "--monitor" in rig.spawned[-1]
    assert runner.setup()["done"] is True


def test_headphones_coming_out_after_the_yes_are_asked_about_again(rig, runner, meter):
    plugged = [LAPTOP_MIC, LAPTOP_OUT, WIRED]
    rig.devices = plugged
    good(meter, LAPTOP_MIC, -14.0)
    doc = runner.listen({})
    mic = {"device": doc["device"]}
    runner.answer({**mic, "step": "headphones", "answer": "yes"})
    runner.answer({**mic, "step": "hears_self", "answer": "no"})
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    last = runner.answer({**mic, "step": "volume", "answer": "still"})
    assert (last["step"], last["done"], last["hear_me"]) == ("headphones", False, False)
    assert len(meter.calls) == 1
    runner.start({"device": doc["device"]})
    assert "--monitor" not in rig.spawned[-1]


def test_the_device_she_was_heard_on_going_away_never_says_done(rig, runner, meter):
    good(meter)
    doc = runner.listen({})
    picked = {"device": doc["device"]}
    runner.answer({**picked, "step": "headphones", "answer": "yes"})
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    with pytest.raises(guitake.Refused) as e:
        runner.answer({**picked, "step": "hears_self", "answer": "yes"})
    assert (e.value.status, e.value.message) == (400, guitake.DEVICE_MISSING)
    assert rig.spawned == []


def test_a_second_interface_plugged_in_during_the_flow_does_not_move_the_take(rig, runner, meter):
    good(meter)
    doc = runner.listen({})
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX, OTHER_BOX]
    picked = {"device": doc["device"]}
    assert runner.answer({**picked, "step": "headphones", "answer": "no"})["done"] is True
    runner.start({})
    assert f"--device={AUDIO_BOX['name']}" in rig.spawned[-1]


def test_a_finished_flow_that_saw_the_same_devices_takes_one_more_look_and_no_more(rig, runner, meter):
    good(meter)
    doc = runner.listen({})
    picked = {"device": doc["device"]}
    runner.answer({**picked, "step": "headphones", "answer": "yes"})
    looks = []
    listed = runner._list
    runner._list = lambda: (looks.append(1), listed())[1]
    assert runner.answer({**picked, "step": "hears_self", "answer": "yes"})["done"] is True
    assert looks == [1]


# --------------------------------------------------------------- saved between runs

def _finished(runner, meter):
    good(meter)
    runner.listen({})
    return walk(runner, ("headphones", "yes"), ("hears_self", "no"), ("volume", "still"))


def test_after_setup_start_goes_straight_to_the_count_with_what_it_learned(rig, runner, meter):
    _finished(runner, meter)
    calls = len(meter.calls)
    assert runner.setup()["done"] is True and runner.home()["setup"]["done"] is True
    runner.start({})
    argv = rig.spawned[-1]
    assert f"--device={AUDIO_BOX['name']}" in argv and "--input-channel=2" in argv and "--monitor" in argv
    assert len(meter.calls) == calls


def test_the_answers_survive_a_new_bridge_and_are_private(rig, runner, meter, setup_file):
    _finished(runner, meter)
    assert stat.S_IMODE(os.stat(setup_file).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(setup_file.parent).st_mode) == 0o700
    again = rig.runner(meter=meter, setup_path=lambda: setup_file)
    assert again.setup()["done"] is True and again.setup()["hear_me"] is True


def test_no_device_name_is_written_down(runner, meter, setup_file):
    _finished(runner, meter)
    assert AUDIO_BOX["name"] not in setup_file.read_text()


def test_a_planted_symlink_is_not_followed_either_way(rig, tmp_path, meter, setup_file):
    target = tmp_path / "elsewhere.json"
    target.write_text("keep me")
    setup_file.parent.mkdir(parents=True)
    setup_file.symlink_to(target)
    runner = rig.runner(meter=meter, setup_path=lambda: setup_file)
    assert runner.setup()["step"] == "listen"
    _finished(runner, meter)
    assert target.read_text() == "keep me"
    assert not setup_file.is_symlink() and json.loads(setup_file.read_text())["version"] == 1


@pytest.mark.parametrize("text", [
    "not json", "[]", '{"version": 99, "devices": {}}', '{"version": 1, "devices": 7}',
    '{"version": 1, "last": 7, "devices": {"../x": {"channel": 1}}}',
    '{"version": 1, "devices": {"abc": {"level": "ok", "hear_me": true}}}',
])
def test_a_bad_file_is_only_a_first_run(rig, meter, setup_file, text):
    setup_file.parent.mkdir(parents=True)
    setup_file.write_text(text)
    runner = rig.runner(meter=meter, setup_path=lambda: setup_file)
    doc = runner.setup()
    assert (doc["step"], doc["hear_me"]) == ("listen", False)
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]


def test_a_file_cannot_turn_hear_me_on_without_a_yes(rig, runner, meter, setup_file):
    good(meter)
    runner.listen({})
    walk(runner, ("headphones", "no"))
    doc = json.loads(setup_file.read_text())
    entry = next(iter(doc["devices"].values()))
    entry["hear_me"] = True
    setup_file.write_text(json.dumps(doc))
    again = rig.runner(meter=meter, setup_path=lambda: setup_file)
    again.start({})
    assert "--monitor" not in rig.spawned[-1]


def test_a_folder_that_cannot_be_written_still_gets_through_setup(rig, tmp_path, meter):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    runner = rig.runner(meter=meter, setup_path=lambda: blocker / "setup.json")
    assert _finished(runner, meter)["done"] is True
    runner.start({})
    assert "--monitor" in rig.spawned[-1]


def test_set_up_again_forgets_and_opens_nothing(rig, runner, meter):
    _finished(runner, meter)
    calls = len(meter.calls)
    doc = runner.reset_setup()
    assert (doc["step"], doc["hear_me"], doc["done"]) == ("listen", False, False)
    assert len(meter.calls) == calls
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]


def test_a_changed_device_list_reruns_only_the_first_step_on_an_interface(rig, runner, meter):
    _finished(runner, meter)
    rig.devices = [*rig.devices, {"name": "New Thing", "inputs": 0, "outputs": 2}]
    assert runner.setup()["step"] == "listen"
    good(meter)
    doc = runner.listen({})
    assert (doc["step"], doc["hear_me"]) == ("count", True)


def test_a_new_virtual_device_is_not_a_changed_list(rig, runner, meter):
    _finished(runner, meter)
    rig.devices = [*rig.devices, VIRTUAL_DEVICE]
    assert runner.setup()["done"] is True


def _room_finished(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    return walk(runner, ("headphones", "yes"), ("hears_self", "no"), ("volume", "still"))


def test_a_changed_list_on_a_room_mic_asks_about_headphones_again(rig, runner, meter):
    """Its sound goes wherever the system sends it, and that may have just become speakers."""
    assert _room_finished(rig, runner, meter)["hear_me"] is True
    rig.devices = [*rig.devices, {"name": "New Speakers", "inputs": 0, "outputs": 2}]
    doc = runner.setup()
    assert (doc["step"], doc["hear_me"]) == ("listen", False)
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]


def test_switching_the_system_output_counts_as_a_changed_list(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, {**LAPTOP_OUT, "default_output": True}, {"name": "Phones", "inputs": 0, "outputs": 2}]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    walk(runner, ("headphones", "yes"), ("hears_self", "no"), ("volume", "still"))
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, {"name": "Phones", "inputs": 0, "outputs": 2, "default_output": True}]
    assert runner.setup()["hear_me"] is False


def test_plugging_in_an_interface_beats_the_mic_setup_knew(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -30.0)
    runner.listen({})
    assert runner.setup()["device_name"] == LAPTOP_MIC["name"]
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX]
    assert runner.setup()["device_name"] == AUDIO_BOX["name"]
    meter.calls.clear()
    good(meter, LAPTOP_MIC, -6.0)                    # an acoustic guitar is louder in the room
    good(meter)
    doc = runner.listen({})
    assert sorted(c[0] for c in meter.calls) == sorted([LAPTOP_MIC["name"], AUDIO_BOX["name"]])
    assert doc["device_name"] == AUDIO_BOX["name"]


def test_a_mic_that_still_hears_her_is_kept_when_the_new_interface_is_silent(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX]
    assert runner.listen({})["device_name"] == LAPTOP_MIC["name"]


def test_a_different_device_starts_its_own_setup(rig, runner, meter):
    _finished(runner, meter)
    mic = next(d for d in runner.home()["devices"] if d["name"] == LAPTOP_MIC["name"])
    assert runner.setup(mic["id"])["step"] == "listen"
    runner.start({"device": mic["id"]})
    assert "--monitor" not in rig.spawned[-1]


def test_a_device_gone_falls_back_safely(rig, runner, meter):
    _finished(runner, meter)
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    home = runner.home()
    assert next(d for d in home["devices"] if d["id"] == home["device"])["name"] == LAPTOP_MIC["name"]
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]


# ------------------------------------------------------------------ the new Hear me rule

def test_hear_me_starts_off_on_every_device_from_any_name(rig, runner):
    for device in ([AUDIO_BOX], [{"name": "Monitor Controller", "inputs": 2, "outputs": 2}], [LAPTOP_MIC, LAPTOP_OUT]):
        rig.devices = device
        home = runner.home()
        assert home["hear_me"]["default"] is False and home["hear_me"]["allowed"] is False
        assert all(d["hear_me"] is False for d in home["devices"])


def test_hear_me_with_no_answer_about_headphones_is_refused_and_starts_nothing(rig, runner):
    with pytest.raises(guitake.Refused) as e:
        runner.start({"hear_me": True})
    assert (e.value.status, e.value.message) == (400, guitake.NO_HEAR_ME)
    assert rig.spawned == [] and runner.live()["state"] == "idle"


def test_hear_me_with_a_saved_no_is_refused(rig, runner, meter):
    good(meter)
    runner.listen({})
    walk(runner, ("headphones", "no"))
    with pytest.raises(guitake.Refused):
        runner.start({"hear_me": True})
    assert rig.spawned == []


def test_a_saved_yes_on_one_device_says_nothing_about_another(rig, runner, meter):
    _finished(runner, meter)
    mic = next(d for d in runner.home()["devices"] if d["name"] == LAPTOP_MIC["name"])
    with pytest.raises(guitake.Refused):
        runner.start({"device": mic["id"], "hear_me": True})


def tick(runner, on=True, device=None):
    body = {"step": "hear_me", "answer": "on" if on else "off"}
    if device is not None:
        body["device"] = device
    return runner.answer(body)


def test_a_tick_is_a_yes_kept_for_this_device_and_start_needs_nothing_more(rig, runner):
    assert tick(runner)["hear_me"] is True
    runner.start({})
    assert "--monitor" in rig.spawned[-1]
    assert runner.setup()["hear_me"] is True


def test_turning_it_off_in_settings_is_kept(rig, runner, meter):
    _finished(runner, meter)
    assert tick(runner, on=False)["hear_me"] is False
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]
    assert runner.setup()["hear_me"] is False


@pytest.mark.parametrize("answer", ["yes", "maybe", ""])
def test_a_tick_is_only_on_or_off(runner, answer):
    with pytest.raises(guitake.Refused):
        runner.answer({"step": "hear_me", "answer": answer})
    assert runner.setup()["hear_me"] is False


def test_a_tick_for_a_device_that_has_gone_is_refused(rig, runner):
    with pytest.raises(guitake.Refused) as e:
        tick(runner, device="f" * 16)
    assert e.value.status == 400 and runner.setup()["hear_me"] is False


@pytest.mark.parametrize("value", [True, False, "yes", 1, 0])
def test_start_is_never_a_word_about_headphones(rig, runner, value):
    """Words sent with Start come from a page that may be hours old, on other gear."""
    with pytest.raises(guitake.Refused):
        runner.start({"hear_me": True, "headphones": value})
    with pytest.raises(guitake.Refused):
        runner.start({"headphones": value})
    assert rig.spawned == [] and runner.setup()["hear_me"] is False


def test_hear_me_sent_with_start_does_not_change_what_is_saved(rig, runner, meter):
    _finished(runner, meter)
    runner.start({"hear_me": False})
    assert "--monitor" not in rig.spawned[-1]
    assert runner.setup()["hear_me"] is True


def test_a_tick_from_before_the_headphones_came_out_starts_no_monitor(rig, runner, meter):
    """The feedback case: headphones out, so the mic's sound now goes to the speakers."""
    plugged = {"name": "Wired Headphones", "inputs": 0, "outputs": 2, "default_output": True}
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, plugged]
    good(meter, LAPTOP_MIC, -14.0)
    runner.listen({})
    tick(runner)
    runner.start({})
    assert "--monitor" in rig.spawned[-1], "the check needs the tick to have worked"
    rig.die(0)
    runner.tick()
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -45.0)
    assert runner.listen({})["heard"]["result"] == "low"
    runner.start({})
    assert "--monitor" not in rig.spawned[-1]
    entry = json.loads(Path(runner._setup._path_of()).read_text())["devices"]
    assert not any(e.get("hear_me") or e.get("headphones") for e in entry.values())


@pytest.mark.parametrize("on", [True, False])
def test_a_tick_leaves_a_finished_setup_finished(rig, runner, meter, on):
    good(meter)
    runner.listen({})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    assert runner.setup()["done"] is True
    tick(runner, on=on)
    doc = runner.setup()
    assert doc["done"] is True and doc["hear_me"] is on
    runner.start({})
    assert ("--monitor" in rig.spawned[-1]) is on


@pytest.mark.parametrize("devices,name", [
    ([LAPTOP_MIC, LAPTOP_OUT], LAPTOP_MIC["name"]),
    ([{"name": "USB Mic", "inputs": 1, "outputs": 0}, LAPTOP_MIC, LAPTOP_OUT], "USB Mic"),
])
def test_a_room_mic_may_hear_itself_with_headphones(rig, runner, devices, name):
    rig.devices = devices
    mic = next(d for d in runner.home()["devices"] if d["name"] == name)
    tick(runner, device=mic["id"])
    runner.start({"device": mic["id"]})
    argv = rig.spawned[-1]
    assert f"--input-device={name}" in argv and "--monitor" in argv and "--direct" not in argv


def test_the_system_default_may_hear_itself_with_headphones(rig, runner):
    rig.devices = [LAPTOP_OUT, VIRTUAL_DEVICE]
    tick(runner)
    runner.start({})
    assert "--monitor" in rig.spawned[-1]
    assert not [a for a in rig.spawned[-1] if a.startswith(("--device", "--input-device"))]


def test_the_real_parser_reads_a_room_monitor(tmp_path):
    route = guitake.Route("Laptop Mic", "room", 1, True)
    args = _parsed(guitake.build_argv(route, tmp_path / "t", key=None, style=None, seed=1))
    assert (args.input_device, args.monitor, args.direct, args.device) == ("Laptop Mic", True, False, None)


ROOM_RIG = [{"name": "Laptop Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Laptop Speakers", "max_input_channels": 0, "max_output_channels": 2}]


@pytest.mark.parametrize("named", [True, False])
def test_the_monitor_works_on_a_room_mic_and_leaks_into_neither_recording(tmp_path, monkeypatch, plume_registry, named):
    """Fake audio only.  With no count the band never starts, so anything out is her own."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    t = np.arange(48000 * 2) / 48000
    tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    fake = fake_audio(tone, devices=ROOM_RIG, capture=True)
    extra = {"input_device": "Laptop Mic"} if named else {}
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.03, monitor=True, tap_start=True,
                      record_dir=str(tmp_path / "rec"), socket_path=short_socket_path("rmon"), **extra)
    out = fake.stream.output
    assert np.abs(out).max() == pytest.approx(0.2, abs=0.01)
    assert np.array_equal(out[:, 0], out[:, 1])
    machine, _ = sf.read(str(tmp_path / "rec" / "machine.wav"), dtype="float32")
    assert machine.size and np.abs(machine).max() == 0.0
    human, _ = sf.read(str(tmp_path / "rec" / "human.wav"), dtype="float32")
    assert np.abs(human).max() == pytest.approx(0.2, abs=0.01)
    doc = json.loads((tmp_path / "rec" / "session.json").read_text())
    assert doc["monitor"] is True and doc["input"]["source"] == "room"


def test_the_room_argv_runs_the_real_transport_on_the_fake_device(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    route = guitake.Route("Laptop Mic", "room", 1, True)
    folder = tmp_path / "take"
    argv = guitake.build_argv(route, folder, key=None, style=None, seed=3)[4:]
    fake = fake_audio(np.zeros(48000 * 3, dtype=np.float32), devices=ROOM_RIG)
    with patched_module("sounddevice", fake):
        code = cli.main([*argv, f"--socket={short_socket_path('rargv')}", "--max-minutes=0.02"])
    assert code == 0
    manifest = json.loads((folder / "session.json").read_text())
    assert manifest["monitor"] is True and manifest["input"]["source"] == "room"
    assert fake.opened[0]["channels"][0] == 1


# ------------------------------------------------------------------ gear, with no brands

@pytest.mark.parametrize("name,inputs,outputs,kind", [
    ("Totally Unknown Maker 8i6", 8, 6, "interface"),
    ("Mystery Box", 2, 2, "interface"),
    ("USB Audio Device", 2, 2, "interface"),
    ("Guitar Link", 1, 2, "interface"),
    ("Laptop Mic", 2, 2, "builtin"),
    ("Internal Microphone", 2, 0, "builtin"),
    ("Microphone Array (Realtek)", 2, 0, "builtin"),
    ("Built-in Audio Analog Stereo", 2, 2, "builtin"),
    ("HDA Intel PCH: ALC257 Analog (hw:0,0)", 2, 2, "builtin"),
    ("Studio Display", 1, 2, "builtin"),
    ("Some Dock Audio", 2, 2, "builtin"),
    ("HD Webcam", 1, 2, "builtin"),
    ("HDMI Audio", 2, 2, "builtin"),
    ("default", 32, 32, "builtin"),
    ("sysdefault", 128, 128, "builtin"),
    ("pulse", 32, 32, "builtin"),
    ("pipewire", 64, 64, "builtin"),
    ("VoiceMeeter Output", 2, 2, "virtual"),
    ("samplerate", 128, 128, "virtual"),
    ("speexrate", 128, 128, "virtual"),
    ("upmix", 8, 8, "virtual"),
    ("vdownmix", 6, 6, "virtual"),
    ("dmix", 2, 2, "virtual"),
    ("surround51", 6, 6, "virtual"),
    ("HD-Audio Generic: ALC257 Analog (hw:1,0)", 2, 2, "builtin"),
    ("HDA Intel PCH: ALC3246 Analog (hw:0,0)", 2, 2, "builtin"),
    ("sof-hda-dsp: - (hw:0,0)", 2, 2, "builtin"),
    ("bcm2835 Headphones: - (hw:0,0)", 2, 2, "builtin"),
    ("Some Box 4x4: USB Audio (hw:3,0)", 4, 4, "interface"),
    ("USB Audio Device: - (hw:2,0)", 2, 2, "interface"),
    ("Monitor of Something", 2, 2, "virtual"),
    ("USB Mic", 1, 0, "other"),
    ("Big Studio Microphone", 2, 2, "other"),
    ("Two In No Out", 2, 0, "other"),
    ("Bluetooth Headset", 1, 2, "other"),
])
def test_no_list_of_brands_decides_what_a_device_is(name, inputs, outputs, kind):
    assert guitake.describe_devices([{"name": name, "inputs": inputs, "outputs": outputs}])[0]["kind"] == kind


def test_a_laptop_with_software_devices_listed_is_listened_to_on_its_own_mic(rig, runner, meter):
    """What ALSA usually lists, typed from memory and not captured from a machine."""
    rig.devices = [{"name": "HD-Audio Generic: ALC257 Analog (hw:1,0)", "inputs": 2, "outputs": 2},
                   {"name": "sysdefault", "inputs": 128, "outputs": 128},
                   {"name": "samplerate", "inputs": 128, "outputs": 128},
                   {"name": "speexrate", "inputs": 128, "outputs": 128},
                   {"name": "upmix", "inputs": 8, "outputs": 8},
                   {"name": "vdownmix", "inputs": 6, "outputs": 6},
                   {"name": "default", "inputs": 32, "outputs": 32}]
    runner.listen({})
    assert [name for name, _ in meter.calls] == ["HD-Audio Generic: ALC257 Analog (hw:1,0)"]
    doc = runner.home()
    assert doc["mode"] == "room" and doc["devices"][0]["kind"] == "builtin"


def test_software_coming_and_going_is_not_a_changed_list_on_linux_either():
    rows = [{"name": "HDA Intel PCH: ALC3246 Analog (hw:0,0)", "inputs": 2, "outputs": 2}]
    with_software = [*rows, {"name": "upmix", "inputs": 8, "outputs": 8}]
    assert guitake.device_list_mark(rows) == guitake.device_list_mark(with_software)


def test_there_is_no_allow_list_of_makers_any_more():
    source = Path(guitake.__file__).read_text().lower()
    for maker in ("scarlett", "focusrite", "motu", "apogee", "presonus", "behringer", "tascam"):
        assert maker not in source


def test_hear_me_comes_from_no_device_name():
    for name in ("Mystery Box", "Laptop Mic", "Audio Interface 4x4"):
        assert guitake.describe_devices([{"name": name, "inputs": 2, "outputs": 2}])[0]["hear_me"] is False


def test_extra_fields_from_the_lister_are_read_only_when_they_make_sense(rig, runner):
    rig.devices = [{**AUDIO_BOX, "output_ms": "soon", "default_output": "yes"}]
    assert runner.home()["devices"][0]["name"] == AUDIO_BOX["name"]


def test_the_lister_adds_what_a_real_library_reports(monkeypatch):
    class Sd:
        def query_devices(self, kind=None):
            rows = [{"name": "A", "max_input_channels": 2, "max_output_channels": 2,
                     "default_low_output_latency": 0.0125},
                    {"name": "B", "max_input_channels": 1, "max_output_channels": 0,
                     "default_low_output_latency": 0.09}]
            return rows[0] if kind == "output" else rows

    assert realtime.input_devices(Sd()) == [
        {"name": "A", "inputs": 2, "outputs": 2, "output_ms": 12.5, "default_output": True},
        {"name": "B", "inputs": 1, "outputs": 0}]


# ------------------------------------------------------------------ over HTTP

@pytest.fixture
def served(rig, meter, tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    srv = gui.GuiServer(roots=[rig.root], log=lambda line: None, spawner=rig.spawner, lister=rig.lister,
                        sessions=rig.sessions, stopper=rig.stopper, cue=rig.cue, watch=False,
                        meter=meter).start()
    yield srv
    srv.close()


@pytest.fixture
def page(served):
    return Client(served).login()


SETUP_POSTS = ["/api/setup/listen", "/api/setup/answer", "/api/setup/cancel", "/api/setup/reset"]


def test_the_whole_flow_over_http_saves_in_the_state_folder(rig, meter, page, tmp_path):
    good(meter)
    assert json.loads(page.get("/api/home")[1])["setup"]["step"] == "listen"
    resp, out = page.get("/api/setup")
    assert resp.status == 200 and json.loads(out)["line"] == "Play something."
    assert meter.calls == []
    assert json.loads(page.post("/api/setup/listen", {})[1])["step"] == "headphones"
    assert json.loads(page.post("/api/setup/answer", {"step": "headphones", "answer": "no"})[1])["done"] is True
    saved = tmp_path / "state" / "setup.json"
    assert stat.S_IMODE(os.stat(saved).st_mode) == 0o600
    assert page.post("/api/start", {})[0].status == 200
    assert "--input-channel=2" in rig.spawned[-1] and "--monitor" not in rig.spawned[-1]


def test_a_setup_for_one_device_by_query(rig, page):
    mic = next(d for d in json.loads(page.get("/api/home")[1])["devices"] if d["name"] == LAPTOP_MIC["name"])
    assert json.loads(page.get(f"/api/setup?device={mic['id']}")[1])["device_name"] == LAPTOP_MIC["name"]
    assert page.get("/api/setup?device=--direct")[0].status == 400


def test_hear_me_with_no_answer_over_http_is_a_400(rig, page):
    resp, out = page.post("/api/start", {"hear_me": True})
    assert resp.status == 400 and json.loads(out)["error"] == guitake.NO_HEAR_ME
    assert rig.spawned == []


def test_reset_and_cancel_over_http(page):
    assert json.loads(page.post("/api/setup/cancel", {})[1])["step"] == "listen"
    assert json.loads(page.post("/api/setup/reset", {})[1])["step"] == "listen"


def test_bad_bodies_over_http(meter, page):
    assert page.post("/api/setup/listen", {"device": "--direct"})[0].status == 400
    assert page.post("/api/setup/answer", {"step": "x", "answer": "y"})[0].status == 400
    assert page.post("/api/setup/cancel", {"x": 1})[0].status == 400
    assert page.post("/api/setup/reset", {"x": 1})[0].status == 400
    assert meter.calls == []


@pytest.mark.parametrize("method,path", [("GET", "/api/setup"), *[("POST", p) for p in SETUP_POSTS]])
def test_every_setup_route_has_the_same_checks(meter, page, method, path):
    token = page.csrf
    assert _ask(page, method, path, Host="evil.example") == 403
    assert _ask(page, method, path, Origin="http://evil.example") == 403
    assert _ask(page, method, path, **{"Sec-Fetch-Site": "cross-site"}) == 403
    page.csrf = "0" * 64
    assert _ask(page, method, path) in (401, 403)
    page.csrf = token
    jar, page.cookie = page.cookie, None
    assert _ask(page, method, path) == 401
    page.cookie = jar
    assert meter.calls == []


@pytest.mark.parametrize("path", SETUP_POSTS)
def test_a_setup_post_needs_an_origin_and_json_and_a_get_does_nothing(meter, page, path):
    assert _ask(page, "POST", path, Origin=None) == 403
    assert _ask(page, "POST", path, **{"Content-Type": "text/plain"}) == 415
    assert page.get(path)[0].status == 404
    assert meter.calls == []


# ------------------------------------------------------------------ other computers

def test_a_machine_with_no_unix_sockets_gets_a_plain_refusal(monkeypatch):
    monkeypatch.delattr(gui.socket, "AF_UNIX", raising=False)
    with pytest.raises(gui.GuiError, match="macOS or Linux"):
        gui.serve(open_browser=False, wait=lambda s: None)


def test_this_machine_is_not_refused():
    assert gui.platform_problem() is None
