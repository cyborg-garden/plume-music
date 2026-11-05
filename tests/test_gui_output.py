"""Sending the band somewhere other than the input's own outs.

No device opens here.  The bridge runs on fakes, the transport on the fake audio device in
tests/conftest.py, and the setup file is a temporary one.  Nothing here says anything about real
hardware, and above all nothing about whether two real devices hold together.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import fake_audio, patched_module, short_socket_path
from plume import cli, guitake, realtime
from test_gui_setup import FakeMeter, good
from test_gui_take import AUDIO_BOX, LAPTOP_MIC, LAPTOP_OUT, VIRTUAL_DEVICE, Rig, _parsed

SR = realtime.SR
SPEAKERS = {"name": "Room Speakers", "inputs": 0, "outputs": 2}
OTHER_SPEAKERS = {"name": "Other Speakers", "inputs": 0, "outputs": 2}
SPEAKERS_ID = guitake.describe_outputs([SPEAKERS])[0]["id"]
OTHER_ID = guitake.describe_outputs([OTHER_SPEAKERS])[0]["id"]
BOX_ID = guitake.describe_devices([AUDIO_BOX])[0]["id"]
MIC_ID = guitake.describe_devices([LAPTOP_MIC])[0]["id"]


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path / "shelf", tmp_path / "state")
    made.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX, SPEAKERS, OTHER_SPEAKERS]
    yield made
    if made.human is not None:
        made.human.close()


@pytest.fixture
def meter():
    return FakeMeter()


@pytest.fixture
def setup_file(tmp_path):
    return tmp_path / "state" / "setup.json"


@pytest.fixture
def runner(rig, meter, setup_file):
    return rig.runner(meter=meter, setup_path=lambda: setup_file)


def choose(runner, answer, device=BOX_ID):
    return runner.answer({"step": "output", "answer": answer, "device": device})


def opt_in(runner, on=True, channel=None, device=BOX_ID):
    body = {"step": "send", "answer": "on" if on else "off", "device": device}
    if channel is not None:
        body["channel"] = channel
    return runner.answer(body)


def tail(rig):
    return [a for a in rig.spawned[-1] if a.startswith(("--device", "--input-device", "--output-device", "--monitor", "--direct"))]


def headphones_yes(runner, device=BOX_ID):
    return runner.answer({"step": "hear_me", "answer": "on", "device": device})


# -------------------------------------------------------------------------- what is on offer

def test_only_real_outputs_are_offered():
    names = [o["name"] for o in guitake.describe_outputs(
        [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX, VIRTUAL_DEVICE, {"name": "default", "inputs": 2, "outputs": 2},
         {"name": "dmix", "inputs": 0, "outputs": 2}])]
    assert names == [LAPTOP_OUT["name"], AUDIO_BOX["name"]]


def test_a_name_inside_another_is_offered_because_the_child_takes_only_the_whole_name():
    rows = [{"name": "Speakers Two", "inputs": 0, "outputs": 2}, {"name": "Speakers", "inputs": 0, "outputs": 2},
            {"name": "speakers", "inputs": 0, "outputs": 2}]
    assert [o["name"] for o in guitake.describe_outputs(rows)] == ["Speakers Two", "Speakers"]
    table = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
             *({"name": r["name"], "max_input_channels": 0, "max_output_channels": 2} for r in rows[:2])]
    assert realtime.resolve_input(fake_audio(None, devices=table), input_device="Mic", output_device="Speakers").device == (0, 2)


def test_home_lists_them_and_says_same_as_input_by_default(runner):
    doc = runner.home()
    assert SPEAKERS["name"] in [o["name"] for o in doc["outputs"]]
    box = next(d for d in doc["devices"] if d["id"] == BOX_ID)
    assert box["output"] is None and box["send_channel"] is None and box["output_problem"] is None


# ------------------------------------------------------------------------------- build_argv

def test_argv_passes_the_output_only_when_a_different_one_was_chosen(tmp_path):
    same = guitake.build_argv(guitake.Route("Box", "direct", 1, False), tmp_path, key=None, style=None, seed=1)
    other = guitake.build_argv(guitake.Route("Box", "direct", 1, False, 4, "--monitor"), tmp_path, key=None, style=None, seed=1)
    assert not any(a.startswith("--output-device") for a in same)
    assert other == [*same, "--output-device=--monitor"], "one token, so a name cannot read as a flag"
    args = _parsed(other)
    assert args.output_device == "--monitor" and args.monitor is False and args.device == "Box"


def test_same_as_input_is_exactly_todays_argv(rig, runner):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX]
    runner.start({})
    assert tail(rig) == [f"--device={AUDIO_BOX['name']}", "--direct"]


def test_a_saved_output_reaches_the_lead_command_for_that_input_only(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    assert tail(rig) == [f"--device={AUDIO_BOX['name']}", "--direct", f"--output-device={SPEAKERS['name']}"]
    rig.die()
    runner.tick()
    runner.start({"device": MIC_ID})
    assert not any(a.startswith("--output-device") for a in rig.spawned[-1])


def test_a_room_mic_can_be_sent_elsewhere_with_no_monitor(rig, runner):
    choose(runner, SPEAKERS_ID, device=MIC_ID)
    runner.start({"device": MIC_ID})
    assert tail(rig) == [f"--input-device={LAPTOP_MIC['name']}", f"--output-device={SPEAKERS['name']}"]


def test_going_back_to_the_same_drops_the_flag(rig, runner):
    choose(runner, SPEAKERS_ID)
    choose(runner, "same")
    runner.start({"device": BOX_ID})
    assert not any(a.startswith("--output-device") for a in rig.spawned[-1])


def test_choosing_its_own_outs_is_the_same_as_not_choosing(rig, runner):
    choose(runner, BOX_ID)
    assert runner.home()["devices"][0]["output"] is None
    runner.start({"device": BOX_ID})
    assert not any(a.startswith("--output-device") for a in rig.spawned[-1])


def test_choosing_opens_nothing_and_starts_nothing(rig, runner, meter):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    assert rig.spawned == [] and meter.calls == []


# ---------------------------------------------------------------------------- it is refused

def test_an_output_that_has_gone_is_refused_and_never_swapped(rig, runner):
    choose(runner, SPEAKERS_ID)
    rig.devices = [d for d in rig.devices if d is not SPEAKERS]
    with pytest.raises(guitake.Refused) as e:
        runner.start({"device": BOX_ID})
    assert e.value.status == 400 and e.value.message == guitake.OUTPUT_MISSING
    assert rig.spawned == [] and runner.live()["state"] == "idle"


def test_home_says_it_has_gone_and_keeps_it_for_when_it_comes_back(rig, runner):
    choose(runner, SPEAKERS_ID)
    rig.devices = [d for d in rig.devices if d is not SPEAKERS]
    box = next(d for d in runner.home()["devices"] if d["id"] == BOX_ID)
    assert box["output"] == SPEAKERS_ID and box["output_problem"] == guitake.OUTPUT_MISSING
    rig.devices = [*rig.devices, SPEAKERS]
    runner.start({"device": BOX_ID})
    assert f"--output-device={SPEAKERS['name']}" in rig.spawned[-1]


def test_a_press_of_start_hears_of_it_before_any_setup(rig, runner):
    choose(runner, SPEAKERS_ID)
    rig.devices = [d for d in rig.devices if d is not SPEAKERS]
    with pytest.raises(guitake.Refused) as e:
        runner.setup(BOX_ID, check_output=True)
    assert e.value.message == guitake.OUTPUT_MISSING
    assert runner.setup(BOX_ID)["device"] == BOX_ID, "only a press of Start asks that"


@pytest.mark.parametrize("answer", ["0" * 16, "../../etc", "--monitor", "Room Speakers", ""])
def test_only_an_output_the_server_listed_can_be_chosen(runner, answer):
    with pytest.raises(guitake.Refused):
        choose(runner, answer)


def test_a_virtual_device_cannot_be_chosen_by_its_id(rig, runner):
    rig.devices = [*rig.devices, VIRTUAL_DEVICE]
    with pytest.raises(guitake.Refused):
        choose(runner, guitake.describe_devices([VIRTUAL_DEVICE])[0]["id"])


def test_a_refusal_to_open_the_pair_lands_as_plain_words_and_in_settings(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    rig.die(1, f"error: {realtime.OPEN_TOGETHER_FAILED}: Illegal combination of I/O devices")
    runner.tick()
    assert runner.live() == {"state": "failed", "message": guitake.OUTPUT_REFUSED, "settings": True}


def test_an_output_that_goes_as_the_child_opens_it_is_said_too(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    rig.die(1, "error: no audio output matching 'Room Speakers' (devices: x)")
    runner.tick()
    assert runner.live()["message"] == guitake.OUTPUT_MISSING


def test_any_other_failure_on_a_different_output_still_goes_to_settings(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    rig.die(1, "Traceback: PortAudioError: Invalid sample rate")
    runner.tick()
    assert runner.live() == {"state": "failed", "message": guitake.OPEN_FAILED, "settings": True}


def test_a_failure_on_its_own_output_is_as_it_was(rig, runner):
    runner.start({"device": BOX_ID})
    rig.die(1, "Traceback: PortAudioError: Invalid sample rate")
    runner.tick()
    assert runner.live() == {"state": "failed", "message": guitake.OPEN_FAILED}


# ------------------------------------------------------------- the instrument, and the rules

def test_the_opt_in_is_off_by_default(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_the_opt_in_sends_the_instrument(rig, runner):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    runner.start({"device": BOX_ID})
    assert "--monitor" in rig.spawned[-1] and f"--output-device={SPEAKERS['name']}" in rig.spawned[-1]
    assert runner.live()["output"] == SPEAKERS["name"] and runner.live()["hear_me"] is True


def test_a_yes_about_headphones_never_reaches_other_speakers(rig, runner):
    """The trap: Hear me saved for her headphones, then the output moved to the room."""
    headphones_yes(runner)
    runner.start({"device": BOX_ID})
    assert "--monitor" in rig.spawned[-1], "the check needs Hear me on to begin with"
    rig.die()
    runner.tick()
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_coming_back_to_her_own_output_finds_hear_me_as_she_left_it(rig, runner):
    headphones_yes(runner)
    choose(runner, SPEAKERS_ID)
    choose(runner, "same")
    runner.start({"device": BOX_ID})
    assert "--monitor" in rig.spawned[-1]


def test_the_opt_in_is_for_that_output_and_not_the_next(rig, runner):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    choose(runner, OTHER_ID)
    runner.start({"device": BOX_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_the_opt_in_does_not_outlive_going_back_and_returning(rig, runner):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    choose(runner, "same")
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_the_opt_in_is_for_that_channel_because_a_mic_may_be_on_the_next(rig, runner):
    choose(runner, SPEAKERS_ID)
    opt_in(runner, channel=1)
    runner.start({"device": BOX_ID, "channel": 2})
    assert "--monitor" not in rig.spawned[-1]
    rig.die()
    runner.tick()
    runner.start({"device": BOX_ID, "channel": 1})
    assert "--monitor" in rig.spawned[-1]


def test_setup_moving_to_another_channel_takes_the_opt_in_back(rig, runner, meter):
    choose(runner, SPEAKERS_ID)
    opt_in(runner, channel=1)
    good(meter)                                     # the loudest is channel 2
    runner.listen({"device": BOX_ID})
    runner.start({"device": BOX_ID})
    assert "--input-channel=2" in rig.spawned[-1] and "--monitor" not in rig.spawned[-1]


def test_start_cannot_ask_for_it_without_the_opt_in(rig, runner):
    headphones_yes(runner)
    choose(runner, SPEAKERS_ID)
    with pytest.raises(guitake.Refused) as e:
        runner.start({"device": BOX_ID, "hear_me": True})
    assert e.value.message == guitake.NO_SEND and rig.spawned == []


def test_a_room_mic_never_goes_through_another_output(rig, runner):
    runner.answer({"step": "hear_me", "answer": "on", "device": MIC_ID})
    choose(runner, SPEAKERS_ID, device=MIC_ID)
    with pytest.raises(guitake.Refused):
        opt_in(runner, device=MIC_ID)
    with pytest.raises(guitake.Refused):
        runner.answer({"step": "hear_me", "answer": "on", "device": MIC_ID})
    with pytest.raises(guitake.Refused):
        runner.start({"device": MIC_ID, "hear_me": True})
    runner.start({"device": MIC_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_an_opt_in_in_a_hand_edited_file_earns_a_room_mic_nothing(rig, runner, setup_file):
    choose(runner, SPEAKERS_ID, device=MIC_ID)
    doc = json.loads(setup_file.read_text())
    doc["devices"][MIC_ID].update(send_output=SPEAKERS_ID, send_channel=1)
    setup_file.write_text(json.dumps(doc))
    fresh = rig.runner(setup_path=lambda: setup_file)
    fresh.start({"device": MIC_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_a_room_mic_on_its_own_output_keeps_the_old_rule(rig, runner):
    choose(runner, "same", device=MIC_ID)
    with pytest.raises(guitake.Refused) as e:
        runner.start({"device": MIC_ID, "hear_me": True})
    assert e.value.message == guitake.NO_HEAR_ME


def test_there_is_no_opt_in_with_no_other_output(runner):
    with pytest.raises(guitake.Refused) as e:
        opt_in(runner)
    assert e.value.message == guitake.NO_SEND


def test_turning_the_opt_in_off_is_kept(rig, runner):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    opt_in(runner, on=False)
    runner.start({"device": BOX_ID})
    assert "--monitor" not in rig.spawned[-1]


@pytest.mark.parametrize("channel", [0, 5, True, "1", 1.5])
def test_a_channel_out_of_range_is_refused(runner, channel):
    choose(runner, SPEAKERS_ID)
    with pytest.raises(guitake.Refused):
        opt_in(runner, channel=channel)


def test_channel_means_nothing_on_any_other_step(runner):
    with pytest.raises(guitake.Refused):
        runner.answer({"step": "output", "answer": "same", "channel": 1})


# ---------------------------------------------------------------------- saved, per input

def test_it_is_saved_per_input_in_setup_json_with_no_names(rig, runner, setup_file):
    choose(runner, SPEAKERS_ID)
    opt_in(runner)
    text = setup_file.read_text()
    assert SPEAKERS["name"] not in text and AUDIO_BOX["name"] not in text
    saved = json.loads(text)["devices"]
    assert saved[BOX_ID]["output"] == SPEAKERS_ID and saved[BOX_ID]["send_output"] == SPEAKERS_ID
    assert MIC_ID not in saved
    again = rig.runner(setup_path=lambda: setup_file).home()
    assert next(d for d in again["devices"] if d["id"] == BOX_ID)["output"] == SPEAKERS_ID


def test_something_else_plugged_in_keeps_the_choice(rig, runner):
    choose(runner, SPEAKERS_ID)
    rig.devices = [*rig.devices, {"name": "New Thing", "inputs": 0, "outputs": 2}]
    runner.start({"device": BOX_ID})
    assert f"--output-device={SPEAKERS['name']}" in rig.spawned[-1]


def test_set_up_again_forgets_it(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.reset_setup()
    runner.start({"device": BOX_ID})
    assert not any(a.startswith("--output-device") for a in rig.spawned[-1])


def test_nothing_can_be_changed_under_a_running_take(rig, runner):
    runner.start({"device": BOX_ID})
    argv = list(rig.spawned[-1])
    for body in ({"step": "output", "answer": SPEAKERS_ID, "device": BOX_ID},
                 {"step": "send", "answer": "on", "device": BOX_ID},
                 {"step": "hear_me", "answer": "on", "device": BOX_ID}):
        with pytest.raises(guitake.Refused) as e:
            runner.answer(body)
        assert e.value.status == 409 and e.value.live["state"] == "starting"
    assert rig.spawned[-1] == argv and len(rig.spawned) == 1


# ------------------------------------------------------------------------ setup's questions

def test_setup_asks_about_headphones_only_of_her_own_output(runner, meter):
    good(meter)
    assert runner.listen({"device": BOX_ID})["step"] == "headphones"
    doc = choose(runner, SPEAKERS_ID)
    assert doc["done"] is True and doc["step"] == "count" and doc["hear_me"] is False


def test_coming_back_still_asks_what_was_never_asked(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, {**AUDIO_BOX, "output_ms": 120.0}, SPEAKERS]
    good(meter)
    runner.listen({"device": BOX_ID})
    choose(runner, SPEAKERS_ID)
    doc = choose(runner, "same")
    assert doc["step"] == "headphones"
    doc = runner.answer({"step": "headphones", "answer": "yes", "device": BOX_ID})
    doc = runner.answer({"step": "hears_self", "answer": "yes", "device": BOX_ID})
    assert doc["step"] == "latency", "the late line was never said, so it still is"


def test_other_speakers_numbers_are_not_kept_as_this_inputs_own(rig, runner):
    choose(runner, SPEAKERS_ID)
    runner.start({"device": BOX_ID})
    (rig.folder / "stream.json").write_text(json.dumps(
        {"input_ms": 5.0, "output_ms": 200.0, "reported_ms": 205.0, "block_ms": 5.3, "calibrated_ms": None}))
    runner.tick()
    assert runner.live()["stream"]["output_ms"] == 200.0
    box = next(d for d in runner.home()["devices"] if d["id"] == BOX_ID)
    assert box["reported_ms"] is None


def test_what_the_page_needs_when_setup_moves_to_another_input(rig, runner, meter):
    """The page compares this with what home said.  A yes then reaches the lead command."""
    other = {"name": "Other Box", "inputs": 2, "outputs": 2}
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, AUDIO_BOX, other, SPEAKERS]
    home = runner.home()
    assert home["device"] == MIC_ID
    choose(runner, SPEAKERS_ID, device=MIC_ID)
    good(meter)
    doc = runner.listen({})
    assert doc["device"] == BOX_ID and doc["output"] is None and doc["heard"]["result"] == "ok"
    assert choose(runner, SPEAKERS_ID)["done"] is True
    runner.start({"device": BOX_ID})
    assert f"--output-device={SPEAKERS['name']}" in rig.spawned[-1]


# -------------------------------------------------------------------------- the transport

@pytest.mark.parametrize("words,page", [("no audio output matching 'x' (devices: y)", guitake.OUTPUT_MISSING),
                                        ("'Input Only Box' has no outputs", guitake.OUTPUT_MISSING),
                                        (f"{realtime.OPEN_TOGETHER_FAILED}: nope", guitake.OUTPUT_REFUSED)])
def test_the_transports_words_are_what_the_page_knows(words, page):
    assert guitake._failure_words(f"error: {words}", guitake.DID_NOT_START) == page


def test_the_argv_runs_the_real_transport_on_the_fake_device(tmp_path, monkeypatch, plume_registry):
    """Fake audio only.  The only thing added to the bridge's argv is a short cap and a socket."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    table = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4},
             {"name": "Room Speakers", "max_input_channels": 0, "max_output_channels": 2}]
    route = guitake.Route("Audio Interface 4x4", "direct", 2, True, 4, "Room Speakers")
    folder = tmp_path / "take"
    argv = guitake.build_argv(route, folder, key=None, style=None, seed=3)[4:]
    device = fake_audio(np.zeros(SR * 3, dtype=np.float32), devices=table, human_channel=1)
    with patched_module("sounddevice", device):
        code = cli.main([*argv, f"--socket={short_socket_path('guiout')}", "--max-minutes=0.02"])
    assert code == 0 and device.opened[0]["device"] == (0, 1)
    manifest = json.loads((folder / "session.json").read_text())
    assert manifest["monitor"] is True and manifest["input"]["output"] == "Room Speakers"
