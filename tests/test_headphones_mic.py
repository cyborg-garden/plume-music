"""A mic, with the band only in headphones: the band cannot reach it, so it is believed.

No device opens here and nothing sounds.  The bridge runs on fakes, the transport on the fake
audio device in tests/conftest.py, and the setup file is a temporary one.  Nothing here says
anything about real hardware: how much of a band leaks out of real headphones into a real mic
is not something these can know.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import fake_audio, needs_laptop_speed, patched_module, playhead_rate, short_socket_path
from plume import cli, guitake, realtime, record
from plume import mind as mind_mod
from plume import roles as roles_mod
from plume.realtime import resolve_input
from plume.watch import Listener, you_trust
from test_direct_input import ALL_YOU, RIG, _direct_take, _play, _take_dir, consent, maskers, rows_of  # noqa: F401
from test_gui_output import (BOX_ID, MIC_ID, OTHER_ID, SPEAKERS, SPEAKERS_ID, choose, meter, rig,  # noqa: F401
                             runner, setup_file)
from record_fixture import ending_doc, song
from test_gui_page import (COUNT, HEADPHONES, HEARD, HEARING, NODE, PLAYING, ROW, SEEN, docs, first_run,  # noqa: F401
                           live_page, run_page, setup_doc)
from test_gui_setup import good, levels, walk
from test_gui_take import AUDIO_BOX, LAPTOP_MIC, LAPTOP_OUT, _parsed

FLAG = "--band-in-headphones"
QUESTION = "Is the band only in your headphones? If you can hear it in the room at all, say No."


def band(runner, answer, device=MIC_ID, output=SPEAKERS_ID):
    """Her answer, with the output it was asked about, as the page sends it."""
    return runner.answer({"step": "band", "answer": answer, "device": device, "output": output})


def heard_on_the_mic(runner, meter, output=SPEAKERS_ID):
    """A mic she has been heard on, with its sound sent to an output she chose."""
    choose(runner, output, device=MIC_ID)
    good(meter, LAPTOP_MIC, -12.0)
    return runner.listen({"device": MIC_ID})


# ------------------------------------------------------------------------ the one question

def test_a_mic_sent_to_a_chosen_output_is_asked_one_plain_question(runner, meter):
    doc = heard_on_the_mic(runner, meter)
    assert (doc["step"], doc["line"], doc["done"]) == ("band", QUESTION, False)
    # The words name no output, so the one she chose is said beside them and sent back with the answer.
    assert doc["note"] == f"The band goes out to {SPEAKERS['name']}." and doc["band_output"] == SPEAKERS_ID
    assert doc["choices"] == [{"answer": "yes", "label": "Yes"}, {"answer": "no", "label": "No"}]


def test_a_yes_makes_the_take_believe_the_mic(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    doc = band(runner, "yes")
    assert doc["done"] is True and doc["step"] == "count"
    runner.start({"device": MIC_ID})
    argv = rig.spawned[-1]
    assert FLAG in argv and f"--input-device={LAPTOP_MIC['name']}" in argv
    assert "--direct" not in argv, "a mic is still a mic"


def test_a_no_is_todays_room_mic_and_is_not_asked_twice(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    assert band(runner, "no")["done"] is True
    assert runner.setup(MIC_ID)["step"] == "count"
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


def test_no_answer_is_todays_room_mic_and_start_still_works(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


def test_a_yes_is_about_that_output_and_not_the_next(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    band(runner, "yes")
    doc = choose(runner, OTHER_ID, device=MIC_ID)
    assert doc["step"] == "band" and doc["done"] is False and doc["band_output"] == OTHER_ID
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


def test_a_yes_does_not_outlive_going_back_and_returning(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    band(runner, "yes")
    choose(runner, "same", device=MIC_ID)
    assert choose(runner, SPEAKERS_ID, device=MIC_ID)["step"] == "band"


def test_something_plugged_in_asks_again(rig, runner, meter):
    """What is plugged in has changed, so where the band comes out may have too."""
    heard_on_the_mic(runner, meter)
    band(runner, "yes")
    rig.devices = [*rig.devices, {"name": "New Thing", "inputs": 0, "outputs": 2}]
    assert runner.setup(MIC_ID)["step"] == "listen"
    assert runner.listen({"device": MIC_ID})["step"] == "band"
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


def test_an_instrument_plugged_in_is_never_asked(rig, runner, meter):
    choose(runner, SPEAKERS_ID)
    good(meter)
    doc = runner.listen({"device": BOX_ID})
    assert doc["done"] is True and doc["step"] == "count"
    with pytest.raises(guitake.Refused):
        band(runner, "yes", device=BOX_ID)
    runner.start({"device": BOX_ID})
    assert FLAG not in rig.spawned[-1]


def test_headphones_on_is_the_same_question_on_the_systems_output_so_it_is_asked_once(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    assert runner.listen({})["step"] == "headphones"
    assert walk(runner, ("headphones", "yes"))["step"] == "hears_self"
    assert walk(runner, ("hears_self", "yes"))["done"] is True


@pytest.mark.parametrize("answer,believed", [("yes", True), ("no", False)])
def test_headphones_on_says_where_the_band_is_for_a_mic_on_the_systems_output(rig, runner, meter, answer, believed):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    walk(runner, ("headphones", answer))
    runner.start({})
    assert (FLAG in rig.spawned[-1]) is believed


def test_headphones_coming_out_take_the_belief_with_them(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    rig.devices = [*rig.devices, {"name": "New Speakers", "inputs": 0, "outputs": 2}]
    runner.start({})
    assert FLAG not in rig.spawned[-1]


def test_headphones_on_an_interface_say_nothing_about_a_mic(rig, runner, meter):
    good(meter)
    runner.listen({"device": BOX_ID})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    runner.start({"device": BOX_ID})
    assert FLAG not in rig.spawned[-1] and "--direct" in rig.spawned[-1]


@pytest.mark.parametrize("answer", ["maybe", "on", "", None, True])
def test_only_a_yes_or_a_no_is_an_answer(runner, meter, answer):
    heard_on_the_mic(runner, meter)
    with pytest.raises(guitake.Refused):
        band(runner, answer)


# ------------------------------------------------- a fact of its own, about this mic, channel and output

def tick(runner, on, device=None):
    body = {"step": "hear_me", "answer": "on" if on else "off"}
    return runner.answer(body if device is None else {**body, "device": device})


def test_the_hear_me_tick_is_never_a_word_about_the_band(rig, runner, meter):
    """No to headphones, then Hear me ticked on and off again in Settings.  The tick is a yes
    about hearing herself, and through speakers the band is still in the mic."""
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    walk(runner, ("headphones", "no"))
    tick(runner, True)
    runner.start({})
    assert FLAG not in rig.spawned[-1]
    rig.die(0)
    runner.tick()
    tick(runner, False)
    runner.start({})
    assert FLAG not in rig.spawned[-1]
    assert runner.home()["band_in_headphones"] is False


def test_a_tick_alone_leaves_the_question_to_be_asked(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    tick(runner, True)
    runner.start({})
    assert FLAG not in rig.spawned[-1] and "--monitor" in rig.spawned[-1]


def test_an_answer_from_before_this_was_asked_opts_nobody_in(rig, meter, setup_file):
    """A saved yes about headphones only ever meant Hear me was safe.  She is asked, once."""
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    first = rig.runner(meter=meter, setup_path=lambda: setup_file)
    good(meter, LAPTOP_MIC, -12.0)
    first.listen({})
    walk(first, ("headphones", "yes"), ("hears_self", "yes"))
    doc = json.loads(setup_file.read_text())
    for name in ("band", "band_channel", "band_output"):
        assert name in doc["devices"][MIC_ID]
        del doc["devices"][MIC_ID][name]
    setup_file.write_text(json.dumps(doc))
    old = rig.runner(meter=meter, setup_path=lambda: setup_file)
    old.start({})
    assert FLAG not in rig.spawned[-1]
    rig.die(0)
    old.tick()
    asked = old.setup()
    assert (asked["step"], asked["line"], asked["note"]) == ("band", QUESTION, None)
    assert asked["band_output"] == guitake.SYSTEM_OUTPUT
    assert old.answer({"step": "band", "answer": "yes", "output": asked["band_output"]})["done"] is True
    old.start({})
    assert FLAG in rig.spawned[-1]
    rig.die(0)
    old.tick()
    assert old.setup()["done"] is True, "and not asked again"


def test_an_answer_says_which_output_it_was_asked_about(rig, runner, meter):
    """Two pages open: one is asked about the first output, the other then chooses a second."""
    heard_on_the_mic(runner, meter)
    choose(runner, OTHER_ID, device=MIC_ID)
    with pytest.raises(guitake.Refused) as e:
        band(runner, "yes", output=SPEAKERS_ID)
    assert e.value.status == 409
    doc = runner.setup(MIC_ID)
    assert doc["step"] == "band" and doc["band_output"] == OTHER_ID, "asked again, of the output it is now"
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


@pytest.mark.parametrize("body", [{"step": "band", "answer": "yes"},
                                  {"step": "band", "answer": "yes", "output": "Room Speakers"},
                                  {"step": "band", "answer": "yes", "output": None},
                                  {"step": "headphones", "answer": "yes", "output": SPEAKERS_ID}])
def test_an_answer_about_the_band_needs_its_output_and_no_other_answer_takes_one(runner, meter, body):
    heard_on_the_mic(runner, meter)
    with pytest.raises(guitake.Refused) as e:
        runner.answer({**body, "device": MIC_ID})
    assert e.value.status == 400


def test_the_answer_is_for_the_channel_it_was_given_on(rig, runner, meter):
    """Another channel of the same device may be another mic, in another room."""
    two = {**LAPTOP_MIC, "inputs": 2}
    rig.devices = [two, LAPTOP_OUT, SPEAKERS]
    choose(runner, SPEAKERS_ID, device=MIC_ID)
    meter.by_name[two["name"]] = levels(-12.0, -90.0)
    runner.listen({"device": MIC_ID})
    band(runner, "yes")
    runner.start({"device": MIC_ID, "channel": 2})
    assert FLAG not in rig.spawned[-1]
    rig.die(0)
    runner.tick()
    runner.start({"device": MIC_ID})
    assert FLAG in rig.spawned[-1]
    rig.die(0)
    runner.tick()
    meter.by_name[two["name"]] = levels(-90.0, -12.0)        # heard on the other channel now
    assert runner.listen({"device": MIC_ID})["step"] == "band"


def test_an_answer_about_headphones_is_not_taken_while_another_output_is_chosen(rig, runner, meter):
    """A page that has not caught up.  It would wait there and count once she went back."""
    heard_on_the_mic(runner, meter)
    for step in ("headphones", "hears_self"):
        with pytest.raises(guitake.Refused) as e:
            runner.answer({"step": step, "answer": "yes", "device": MIC_ID})
        assert e.value.status == 409
    assert choose(runner, "same", device=MIC_ID)["step"] == "headphones"
    runner.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


def test_a_device_that_turns_into_a_mic_brings_no_yes_with_it(rig, runner, meter, setup_file):
    """Software coming and going is not a changed list, and it can change what a device is taken
    to be.  A yes about an interface's own headphones is not a yes about the system's output."""
    good(meter)
    runner.listen({"device": BOX_ID})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    before = guitake.device_list_mark(guitake._rows(rig.devices))
    # A software device whose name holds this one's takes over its output, so it is a mic now.
    name = rig.devices[-3]["name"]
    rig.devices = [{"name": f"Loopback of {name}", "inputs": 0, "outputs": 2}, *rig.devices]
    assert guitake.device_list_mark(guitake._rows(rig.devices)) == before
    now = next(d for d in runner.home()["devices"] if d["id"] == BOX_ID)
    assert now["mode"] == "room", "the stand-in list did not turn it into a mic, so this proves nothing"
    assert now["band_in_headphones"] is False
    runner.start({"device": BOX_ID})
    assert FLAG not in rig.spawned[-1]
    rig.die(0)
    runner.tick()
    assert runner.setup(BOX_ID)["step"] == "band"


def test_the_systems_output_moving_to_another_listed_device_asks_again(rig, runner, meter):
    phones = {"name": "Phones", "inputs": 0, "outputs": 2}
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, {**phones, "default_output": True}]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    walk(runner, ("headphones", "yes"), ("hears_self", "yes"))
    runner.start({})
    assert FLAG in rig.spawned[-1]
    rig.die(0)
    runner.tick()
    rig.devices = [LAPTOP_MIC, {**LAPTOP_OUT, "default_output": True}, phones]
    runner.start({})
    assert FLAG not in rig.spawned[-1]
    assert runner.home()["band_in_headphones"] is False


def test_a_yes_stamped_for_another_system_output_earns_nothing(rig, meter, setup_file):
    """Even with the list as it was: the stamp is what counts, not only the list."""
    phones = {"name": "Phones", "inputs": 0, "outputs": 2}
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT, {**phones, "default_output": True}]
    first = rig.runner(meter=meter, setup_path=lambda: setup_file)
    good(meter, LAPTOP_MIC, -12.0)
    first.listen({})
    walk(first, ("headphones", "yes"), ("hears_self", "yes"))
    doc = json.loads(setup_file.read_text())
    assert doc["devices"][MIC_ID]["band_output"] == guitake.describe_outputs([phones])[0]["id"]
    doc["devices"][MIC_ID]["band_output"] = SPEAKERS_ID
    setup_file.write_text(json.dumps(doc))
    fresh = rig.runner(meter=meter, setup_path=lambda: setup_file)
    fresh.start({})
    assert FLAG not in rig.spawned[-1]


def test_the_answer_about_the_band_is_given_of_the_devices_there_are_now(rig, runner, meter):
    """She may have just plugged them in because it asked."""
    heard_on_the_mic(runner, meter)
    looks = []
    listed = runner._list
    runner._list = lambda: (looks.append(1), listed())[1]
    band(runner, "yes")
    assert looks == [1]


def test_an_answer_about_the_band_is_not_saved_on_a_look_from_before(rig, runner, meter, setup_file):
    """The devices cannot be listed just now.  A yes is not kept against what they were a moment ago."""
    heard_on_the_mic(runner, meter)
    rig.list_error = guitake.TakeError(guitake.NO_LISTING)
    with pytest.raises(guitake.Refused) as e:
        band(runner, "yes")
    assert e.value.status == 409
    assert "band" not in json.loads(setup_file.read_text())["devices"][MIC_ID]


def test_a_yes_in_a_hand_edited_file_makes_no_instrument_a_mic(rig, runner, meter, setup_file):
    choose(runner, SPEAKERS_ID)
    good(meter)
    runner.listen({"device": BOX_ID})
    doc = json.loads(setup_file.read_text())
    doc["devices"][BOX_ID].update(band=True, band_channel=doc["devices"][BOX_ID]["channel"], band_output=SPEAKERS_ID)
    setup_file.write_text(json.dumps(doc))
    fresh = rig.runner(meter=meter, setup_path=lambda: setup_file)
    assert next(d for d in fresh.home()["devices"] if d["id"] == BOX_ID)["band_in_headphones"] is False
    fresh.start({"device": BOX_ID})
    assert FLAG not in rig.spawned[-1] and "--direct" in rig.spawned[-1]


def test_the_command_never_carries_both_flags(tmp_path):
    route = guitake.Route(AUDIO_BOX["name"], "direct", 1, False, 4, None, isolated=True)
    argv = guitake.build_argv(route, tmp_path / "t", key=None, style=None, seed=1)
    assert "--direct" in argv and FLAG not in argv


def test_home_says_which_mic_is_believed(rig, runner, meter):
    heard_on_the_mic(runner, meter)
    assert next(d for d in runner.home()["devices"] if d["id"] == MIC_ID)["band_in_headphones"] is False
    band(runner, "yes")
    home = runner.home()
    assert next(d for d in home["devices"] if d["id"] == MIC_ID)["band_in_headphones"] is True
    assert next(d for d in home["devices"] if d["id"] == BOX_ID)["band_in_headphones"] is False


# ------------------------------------------------------------------------ hear me is as it was

def test_a_yes_about_the_band_never_turns_hear_me_on(rig, runner, meter):
    """A mic through another output would howl, whatever the band is doing."""
    heard_on_the_mic(runner, meter)
    doc = band(runner, "yes")
    assert doc["hear_me"] is False
    with pytest.raises(guitake.Refused) as e:
        runner.start({"device": MIC_ID, "hear_me": True})
    assert e.value.message == guitake.NO_SEND
    with pytest.raises(guitake.Refused):
        runner.answer({"step": "hear_me", "answer": "on", "device": MIC_ID})
    runner.start({"device": MIC_ID})
    assert "--monitor" not in rig.spawned[-1]


def test_hear_me_on_a_mic_still_takes_a_yes_about_headphones(rig, runner, meter):
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    runner.listen({})
    with pytest.raises(guitake.Refused) as e:
        runner.start({"hear_me": True})
    assert e.value.message == guitake.NO_HEAR_ME


# ------------------------------------------------------------------------------- saved

def test_the_answer_is_saved_with_no_names_and_survives_a_new_bridge(rig, runner, meter, setup_file):
    heard_on_the_mic(runner, meter)
    band(runner, "yes")
    text = setup_file.read_text()
    assert LAPTOP_MIC["name"] not in text and SPEAKERS["name"] not in text
    fresh = rig.runner(setup_path=lambda: setup_file)
    assert fresh.setup(MIC_ID)["done"] is True
    fresh.start({"device": MIC_ID})
    assert FLAG in rig.spawned[-1]


def test_a_saved_yes_about_another_output_earns_nothing(rig, runner, meter, setup_file):
    heard_on_the_mic(runner, meter)
    band(runner, "yes")
    doc = json.loads(setup_file.read_text())
    assert doc["devices"][MIC_ID]["band_output"] == SPEAKERS_ID
    doc["devices"][MIC_ID]["output"] = OTHER_ID
    setup_file.write_text(json.dumps(doc))
    fresh = rig.runner(setup_path=lambda: setup_file)
    assert fresh.setup(MIC_ID)["step"] == "band"
    fresh.start({"device": MIC_ID})
    assert FLAG not in rig.spawned[-1]


# ------------------------------------------------------------------------------ the transport

def test_the_route_says_a_mic_the_band_cannot_reach_and_not_a_pickup():
    route = resolve_input(fake_audio(None, devices=RIG), input_device="laptop mic", band_in_headphones=True)
    assert route.source == "room" and route.isolated is True
    doc = route.describe()
    assert doc["source"] == "room" and doc["isolated"] is True
    # Every threshold was tuned on a mic, so this is not the unmeasured case a pickup is.
    assert "thresholds_calibrated" not in doc


def test_a_room_mic_and_a_pickup_say_what_they_always_said():
    sd = fake_audio(None, devices=RIG)
    assert resolve_input(sd).describe() == {"source": "room", "device": None, "channel": 1, "voice_channel": None}
    direct = resolve_input(sd, "audio interface", direct=True)
    assert direct.isolated is True and "isolated" not in direct.describe()


def test_the_real_parser_reads_the_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    route = guitake.Route(LAPTOP_MIC["name"], "room", 1, False, 1, SPEAKERS["name"], isolated=True)
    argv = guitake.build_argv(route, "takes", key=None, style=None, seed=3)
    assert _parsed(argv).band_in_headphones is True
    assert cli.main(argv[4:]) == 0
    assert seen["band_in_headphones"] is True and "direct" not in seen


def test_an_instrument_plugged_in_keeps_its_own_word_when_both_are_said():
    route = resolve_input(fake_audio(None, devices=RIG), "audio interface", direct=True, band_in_headphones=True)
    assert route.source == "direct" and route.band_in_headphones is False
    assert route.describe()["thresholds_calibrated"] is False and "isolated" not in route.describe()


def test_jam_takes_the_flag_and_hands_it_on_to_lead(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["jam", "--band-in-headphones"]) == 0
    assert seen["band_in_headphones"] is True
    monkeypatch.undo()
    handed = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: handed.update(kw))
    realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                 role=roles_mod.RoleParams(role="lead"), band_in_headphones=True)
    assert handed["band_in_headphones"] is True


def test_jam_turns_masking_off_for_a_mic_in_headphones(tmp_path, consent, maskers):
    for name, kw in (("j-phones", {"band_in_headphones": True}), ("j-room", {})):
        with patched_module("sounddevice", fake_audio(np.zeros(realtime.SR * 4, dtype=np.float32), devices=RIG)):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                         record_dir=str(tmp_path / name), max_minutes=0.02, **kw)
    assert [m._enabled for m in maskers] == [False, True]
    assert json.loads((tmp_path / "j-phones" / "session.json").read_text())["input"]["isolated"] is True


def test_voice_passes_the_flag_to_its_band(tmp_path, monkeypatch):
    from plume import voice
    seen = {}
    monkeypatch.setattr(voice, "run", lambda **kw: seen.update(kw))
    monkeypatch.setenv("PLUME_ALLOW_AUDIO", "1")
    with patched_module("sounddevice", fake_audio(None, devices=RIG)):
        assert cli.main(["voice", "lofi", "--base", str(tmp_path), "--band-in-headphones"]) == 0
    assert seen["extra_args"] == ["--band-in-headphones"]


def test_bare_lead_passes_no_word_about_headphones(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["lead", "--tap-start", "lofi"]) == 0
    assert "band_in_headphones" not in seen


def _mic_take(tmp_path, name, human, *, rate_x=12.0, minutes=0.7, fake=None, **kw):
    """A take on the fake device's mic.  No faster than a shared runner can follow."""
    device = fake_audio(human, devices=RIG, rate_x=playhead_rate(rate_x), **(fake or {}))
    with patched_module("sounddevice", device):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / name),
                      status=False, socket_path=short_socket_path(name), max_minutes=minutes, **kw)
    return tmp_path / name


def test_a_mic_with_the_band_in_headphones_is_believed_once_the_band_is_in(tmp_path, consent, maskers, monkeypatch):
    """Everything a mic the band cannot reach turns on, in one take, with nothing of the band in
    the mic.  It checks what the take did, not how fast, so it runs on a shared runner too."""
    from plume import bandentry, duet as duet_mod
    made, parts, heard = {}, [], []

    def spy(cls, *names):
        class Spy(cls):
            def __init__(self, *a, **kw):
                made.setdefault(cls.__name__, []).append({n: kw.get(n) for n in names})
                super().__init__(*a, **kw)
        monkeypatch.setattr(bandentry, cls.__name__, Spy)
    spy(bandentry.BandFollow, "trusted")
    spy(bandentry.SilenceEnding, "end_after_s")

    class Band(duet_mod.DuetEngine):
        def render_bar(self, plan, *a, **kw):
            heard.append(tuple(self.heard_notes))
            return super().render_bar(plan, *a, **kw)
    monkeypatch.setattr(duet_mod, "DuetEngine", Band)
    real_parts = realtime.follow_parts
    monkeypatch.setattr(realtime, "follow_parts", lambda *a, **kw: (parts.append(1), real_parts(*a, **kw))[1])
    # C major until the band is well in, then she moves somewhere it was not
    human = _direct_take().copy()
    t = np.arange(15 * realtime.SR) / realtime.SR
    human[30 * realtime.SR:45 * realtime.SR] = (
        sum(np.sin(2 * np.pi * f * t) for f in (138.59, 174.61, 207.65, 277.18)) * 0.03).astype(np.float32)
    rec = _mic_take(tmp_path, "phones", human, band_in_headphones=True)
    rows = rows_of(rec)
    first_in = next(i for i, r in enumerate(rows) if r["entry"] != "drums")
    in_band = rows[first_in:]
    assert all(r["playing_trusted"] for r in rows)
    assert any(r["playing"] for r in in_band)
    assert not any("room mic" in json.dumps(r.get("chords")) for r in in_band)
    assert not any("band_in_mic" in r for r in rows)
    # Nothing of the band's is there to mask, and each part that listens is told it hears only her.
    assert [m._enabled for m in maskers] == [False]
    assert made["BandFollow"] == [{"trusted": True}]
    assert made["SilenceEnding"] == [{"end_after_s": bandentry.END_AFTER_S}]
    assert parts, "her level by the half bar is only read off an input that hears only her"
    assert heard[first_in] and len(set(heard[first_in:])) > 1, "the ear goes on learning her notes under the band"
    manifest = json.loads((rec / "session.json").read_text())
    assert manifest["input"]["source"] == "room" and manifest["input"]["isolated"] is True
    assert record.input_route(manifest) == "headphones"


def test_a_room_mic_turns_none_of_that_on(tmp_path, consent, maskers, monkeypatch):
    """The same spies on a mic with no word about headphones, or the test above proves nothing."""
    from plume import bandentry
    made = []

    class Follow(bandentry.BandFollow):
        def __init__(self, *a, **kw):
            made.append(kw.get("trusted"))
            super().__init__(*a, **kw)
    monkeypatch.setattr(bandentry, "BandFollow", Follow)
    with patched_module("sounddevice", fake_audio(np.zeros(realtime.SR), devices=RIG)):
        realtime.lead(prompt="lofi", seed=0, tap_start=True, role=roles_mod.RoleParams(role="lead"),
                      status=False, socket_path=short_socket_path("r-spy"), max_minutes=0.01)
    assert made == [False] and [m._enabled for m in maskers] == [True]


def test_a_guess_at_the_style_trusts_how_busy_she_is_but_not_how_she_strums(consent, monkeypatch):
    from plume import bandentry
    made = []

    class Pick(bandentry.StylePick):
        def __init__(self, *a, **kw):
            made.append((kw.get("trust_busy"), kw.get("hear_technique")))
            super().__init__(*a, **kw)
    monkeypatch.setattr(bandentry, "StylePick", Pick)
    for name, kw in (("p-phones", {"band_in_headphones": True}), ("p-room", {})):
        with patched_module("sounddevice", fake_audio(np.zeros(realtime.SR), devices=RIG)):
            realtime.lead(prompt=None, seed=0, tap_start=True, detect=True,
                          role=roles_mod.RoleParams(role="lead"), status=False,
                          socket_path=short_socket_path(name), max_minutes=0.01, **kw)
    assert made == [(True, False), (False, False)]


@needs_laptop_speed
def test_she_stops_and_the_take_ends_on_its_own(tmp_path, consent):
    rec = _mic_take(tmp_path, "ends", _direct_take(), band_in_headphones=True)
    assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE


def _noisy(human, level=0.0126, start_s=8.0):
    """The room: noise with no notes in it, above the level plume calls silence."""
    rng = np.random.default_rng(3)
    out = human.copy()
    at = int(start_s * realtime.SR)
    out[at:] += (rng.standard_normal(out.size - at) * level).astype(np.float32)
    return out


def test_she_can_stop_in_a_noisy_room(tmp_path, consent):
    """With the band in, a level alone would read the room as her and never let the take end."""
    rec = _mic_take(tmp_path, "noisy", _noisy(_direct_take(stops_at=45.0, total=400.0)),
                    band_in_headphones=True)
    rows = rows_of(rec)
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert any(r["playing"] for r in in_band), "she is still heard over the room while she plays"
    late = [r for r in in_band if r["t_s"] > 60.0]
    assert late and not any(r["playing"] for r in late)
    assert all(r["playing_trusted"] for r in rows)
    if os.environ.get("PLUME_SKIP_MACHINE_SPEED") != "1":
        assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE


# -------------------------------------------------------- the take checks her word for itself

def _bleeding(share=0.5, seconds=2.0, seed=0):
    """(mic, band): the band arriving 20 ms late in the mic, under something of her own."""
    rng = np.random.default_rng(seed)
    n = int(seconds * realtime.SR)
    band = (rng.standard_normal(n) * 0.05).astype(np.float32)
    hers = (rng.standard_normal(n) * 0.05).astype(np.float32)
    late = np.concatenate([np.zeros(960, dtype=np.float32), band[:-960]])
    return (hers * (1.0 - share) ** 0.5 + late * share ** 0.5).astype(np.float32), band


def test_the_measure_reads_how_much_of_the_mic_is_the_band():
    from plume import bandentry
    for share in (0.0, 0.5, 1.0):
        mic, band = _bleeding(share)
        assert bandentry.band_share(mic, band, realtime.SR) == pytest.approx(share, abs=0.08)
    assert bandentry.band_share(np.zeros(100), np.zeros(100), realtime.SR) == 0.0


def _a_band(seconds=2.0, seed=0):
    """Something like a band: chords on the notes of C major, and drums as bursts of noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * realtime.SR)
    t = np.arange(n) / realtime.SR
    chord = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / (1 + 0.2 * j)
                for j, f in enumerate((65.41, 130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 523.25, 783.99, 1046.5)))
    drums = rng.standard_normal(n) * (np.exp(-((t * 4.0) % 1.0) * 12.0))
    return (0.02 * chord + 0.05 * drums).astype(np.float32)


def _her_on_the_bands_notes(seconds=2.0, level=0.03, seed=5):
    """Held notes, dead in tune with the band's own, and a little room noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * realtime.SR)
    t = np.arange(n) / realtime.SR
    held = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in (261.63, 329.63, 392.0, 523.25, 783.99, 1046.5))
    return (level * held + rng.standard_normal(n) * 3e-4).astype(np.float32)


def _through(shape, band):
    from conftest import _Shaper
    return _Shaper(shape, band.size).block(band)


def _one_lag_share(mic, band):
    """What a single delayed copy explains: the measure this replaced, kept here to show the case."""
    n = mic.size
    x = np.fft.irfft(np.fft.rfft(mic, 2 * n) * np.conj(np.fft.rfft(band, 2 * n)), 2 * n)[:int(0.7 * realtime.SR)]
    return float(np.max(np.abs(x)) ** 2 / (np.sum(mic.astype(np.float64) ** 2) * np.sum(band.astype(np.float64) ** 2)))


@pytest.mark.parametrize("shape", [
    {"low_cut_hz": 200.0, "rt60_s": 0.05, "wet": 0.0},      # a small speaker, with no bass
    {"low_cut_hz": 300.0, "rt60_s": 0.3, "wet": 1.0},       # and a room after it
    {"low_cut_hz": 300.0, "rt60_s": 0.6, "wet": 1.5},       # a livelier room
])
def test_the_band_through_a_small_speaker_and_a_room_is_still_the_band(shape):
    """The mic holds nothing but the band, and not a clean copy of it: its bass is gone and a
    room's tail is on it.  One delayed copy explains little of that.  A short filter does."""
    from conftest import small_speaker_in_a_room
    from plume import bandentry
    band = _a_band()
    mic = 0.5 * _through(small_speaker_in_a_room(**shape), band)
    mic = mic + np.random.default_rng(9).standard_normal(mic.size).astype(np.float32) * 3e-4
    assert bandentry.band_share(mic, band, realtime.SR) >= 1.4 * bandentry.BAND_IN_MIC_SHARE
    if shape["wet"]:
        assert _one_lag_share(mic, band) < 0.25, "the case is not the one this is for"
    check = bandentry.BandInMic(realtime.SR)
    for n in range(bandentry.BAND_IN_MIC_WINDOWS):
        assert check.latched is False
        band = _a_band(seed=n)
        check.feed(0.5 * _through(small_speaker_in_a_room(**shape), band), band)
    assert check.latched is True


@pytest.mark.parametrize("louder_db", [6.0, 10.0])
def test_the_band_is_found_under_a_player_louder_than_it(louder_db):
    """She plays notes, and notes leave most of the spectrum to whatever else is in the mic."""
    from plume import bandentry
    band = _a_band()
    leak = 0.5 * np.concatenate([np.zeros(960, dtype=np.float32), band[:-960]])
    her = _her_on_the_bands_notes(level=1.0)
    her *= float(np.sqrt(np.mean(leak ** 2) / np.mean(her ** 2))) * 10.0 ** (louder_db / 20.0)
    assert bandentry.band_share(leak + her, band, realtime.SR) >= 1.4 * bandentry.BAND_IN_MIC_SHARE
    assert _one_lag_share(leak + her, band) < 0.25, "the case is not the one this is for"


def test_her_own_held_notes_on_the_bands_partials_are_not_the_band():
    """A clean mic.  She holds the band's own notes, in tune, for as long as it listens."""
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    worst = 0.0
    for n in range(40):
        band, her = _a_band(seed=n), _her_on_the_bands_notes(seed=100 + n)
        worst = max(worst, bandentry.band_share(her, band, realtime.SR))
        check.feed(her, band)
    assert check.latched is False
    assert worst < 0.6 * bandentry.BAND_IN_MIC_SHARE, worst


def test_the_share_is_there_by_the_time_it_says_it_latched():
    """Another thread reads the share as soon as it sees the latch."""
    from plume import bandentry
    seen = []

    class Watched(bandentry.BandInMic):
        def __setattr__(self, name, value):
            if name == "latched" and value:
                seen.append(self.share)
            super().__setattr__(name, value)
    check = Watched(realtime.SR)
    for n in range(bandentry.BAND_IN_MIC_WINDOWS):
        check.feed(*_bleeding(0.6, seed=n))
    assert check.latched and seen and seen[0] is not None


def test_a_fade_already_running_is_kept_when_the_long_wait_comes_back():
    """Trust is withdrawn while the band is fading out.  It goes on fading, to the end: the band
    is in the mic from here, so "not silent" is no longer her coming back."""
    from plume.bandentry import END_AFTER_S, SilenceEnding
    ending, bar_s = SilenceEnding(end_after_s=END_AFTER_S), 2.0
    t, gains = 0.0, []
    ending.update(t, True, bar_s)
    while not gains or gains[-1] == 0.0:
        t += bar_s
        gains.append(ending.update(t, False, bar_s)[0])
    ending.long_wait()
    done = False
    for _ in range(12):
        t += bar_s
        gain, done = ending.update(t, True, bar_s)       # the mic is never silent again
        assert gain <= gains[-1] < 0.0, "the fade carried on down"
        gains.append(gain)
        if done:
            break
    assert done, "and it ended"


def test_with_no_fade_running_the_long_wait_is_a_room_mics():
    from plume.bandentry import END_AFTER_S, SilenceEnding
    ending, bar_s = SilenceEnding(end_after_s=END_AFTER_S), 2.0
    ending.update(0.0, True, bar_s)
    ending.long_wait()
    assert ending.update(bar_s, True, bar_s) == (0.0, False), "playing is still playing"
    quiet = 1
    while ending.update((quiet + 1) * bar_s, False, bar_s)[0] == 0.0:
        quiet += 1
    assert quiet * bar_s >= 28.0


def test_it_takes_several_stretches_of_evidence_and_then_never_goes_back():
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(bandentry.BAND_IN_MIC_WINDOWS):
        assert check.latched is False, "one loud bar is not enough"
        mic, band = _bleeding(0.6, seed=n)
        for at in range(0, mic.size, 4800):                 # in blocks, as the take feeds it
            check.feed(mic[at:at + 4800], band[at:at + 4800])
    assert check.latched is True and check.share >= bandentry.BAND_IN_MIC_SHARE
    for n in range(20):
        check.feed(*_bleeding(0.0, seed=100 + n))
    assert check.latched is True


def test_a_clean_mic_and_a_stray_loud_stretch_do_not_trip_it():
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(40):
        check.feed(*_bleeding(0.6 if n % 4 == 0 else 0.0, seed=n))
    assert check.latched is False


def test_silence_on_either_side_is_no_evidence():
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(10):
        mic, band = _bleeding(0.9, seed=n)
        check.feed(mic, band * 1e-4)        # the band is not sounding
        check.feed(mic * 1e-5, band)        # nothing in the mic
    assert check.latched is False


def test_the_band_in_the_mic_is_found_out_and_the_take_falls_back_for_good(tmp_path, consent, maskers, monkeypatch):
    """At the speed of a real take.  Her word was wrong, or the headphones came out: the band is
    in the mic at -6 dB.  The take notices within a few bars, says so in its log, and from there
    is a room mic until it ends."""
    from plume import bandentry
    endings = []

    class Ending(bandentry.SilenceEnding):
        def __init__(self, *a, **kw):
            endings.append(kw.get("end_after_s"))
            super().__init__(*a, **kw)
    monkeypatch.setattr(bandentry, "SilenceEnding", Ending)
    rec = _mic_take(tmp_path, "found", _direct_take(stops_at=45.0, total=400.0), rate_x=1.0, minutes=0.9,
                    fake={"bleed_db": -6.0}, band_in_headphones=True)
    rows = rows_of(rec)
    first_in = next(i for i, r in enumerate(rows) if r["entry"] != "drums")
    found = next((i for i, r in enumerate(rows) if "band_in_mic" in r), None)
    assert found is not None, "the band at -6 dB in the mic was never noticed"
    assert found <= first_in + 4, "by a few bars after the band came in, at the latest"
    said = rows[found]["band_in_mic"]
    assert said["since_bar"] == rows[found]["bar"] and said["share"] >= 0.25 and "room mic" in said["reason"]
    after = rows[found:]
    assert all(r.get("band_in_mic") == said for r in after), "decided once, and kept"
    # A room mic is believed over drums alone, as it always was, and never once the band is in.
    assert not any(r["playing_trusted"] for r in rows[max(found, first_in):]), "it never goes back to believing the mic"
    assert all("room mic" in json.dumps(r["chords"]) for r in after if r.get("chords"))
    assert [m._enabled for m in maskers] == [True], "and it masks its own sound again"
    assert endings == [bandentry.END_AFTER_S], "the ending it had is kept, not started over"
    # The pages are told a room mic's story about this take, not that it heard her throughout.
    assert record.read_take(rec)["route"] == "room"


def test_her_playing_over_a_clean_mic_never_trips_it(tmp_path, consent):
    """At the speed of a real take, playing the band's own chords the whole way through."""
    rec = _mic_take(tmp_path, "kept", _direct_take(stops_at=390.0, total=400.0), rate_x=1.0, minutes=1.15,
                    band_in_headphones=True)
    rows = rows_of(rec)
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert len(in_band) >= 8 and all(r["playing_trusted"] for r in rows)
    assert not any("band_in_mic" in r for r in rows)


def test_a_mic_in_headphones_still_waits_for_notes_before_the_band(tmp_path, consent, monkeypatch):
    """A mic hears the room as well as her, so a level alone is not her playing: the ear decides
    before the band, the way it does on any mic, and the band keeps the wait that goes with it."""
    from plume import bandentry
    made = []

    class Spy(bandentry.BandEntry):
        def __init__(self, *a, **kw):
            made.append(kw.get("after_first_heard_bars"))
            super().__init__(*a, **kw)
    monkeypatch.setattr(bandentry, "BandEntry", Spy)
    with patched_module("sounddevice", fake_audio(np.zeros(realtime.SR), devices=RIG)):
        realtime.lead(prompt="lofi", seed=0, tap_start=True, role=roles_mod.RoleParams(role="lead"),
                      status=False, socket_path=short_socket_path("e-phones"), max_minutes=0.01,
                      band_in_headphones=True)
    assert made == [None]


@needs_laptop_speed
def test_talk_in_the_room_before_the_band_is_not_her_playing(tmp_path, consent):
    """Noise with no notes in it, as loud as playing.  A pickup would never hear it; a mic does."""
    human = _direct_take(stops_at=6.0, total=60.0)       # the count, and then she never plays
    rng = np.random.default_rng(3)
    human[8 * realtime.SR:] += (rng.standard_normal(human.size - 8 * realtime.SR) * 0.03).astype(np.float32)
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=12.0)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "talk"),
                      status=False, socket_path=short_socket_path("talk"), max_minutes=0.2,
                      band_in_headphones=True)
    rows = rows_of(tmp_path / "talk")
    assert rows and all(r["entry"] == "drums" and not r["playing"] for r in rows)


# --------------------------------------------------------------------- what the pages are told

def _manifest(**audio):
    return {"input": {"source": "room", **audio}}


def test_the_record_knows_three_kinds_of_listening():
    assert record.input_route(_manifest()) == "room"
    assert record.input_route(_manifest(isolated=True)) == "headphones"
    assert record.input_route(_manifest(isolated="yes")) == "room", "only a real yes"
    assert record.input_route({"input": {"source": "direct"}}) == "direct"
    assert record.input_route({"midi_port": "keys", **_manifest(isolated=True)}) == "midi"
    # The file is still a mic's, whatever the band could reach.
    assert record.recorded_from(_manifest(isolated=True)) == "mic"


def test_the_record_speaks_of_her_and_not_of_the_mic():
    rows = song(30, band_bar=8)
    rows[20]["follow"] = {"moved": ["softer"], "bars": 2}
    said = [e["sentence"] for e in record.translate(rows, ending_doc(), None, route="headphones")]
    room = [e["sentence"] for e in record.translate(rows, ending_doc(), None, route="room")]
    assert any("I heard you get softer" in t for t in said)
    assert any("The mic got quieter" in t for t in room)
    # Before the band it is a mic like any other, and waits for notes.
    assert any("That takes me about three bars" in t for t in said)
    lines = " ".join(line["text"] for line in record.roast(rows, ending_doc(), None, route="headphones"))
    assert "I could still hear you" in lines and "no idea what you did" not in lines


def test_the_live_feed_believes_a_mic_in_headphones_and_checks_it(tmp_path):
    path = _take_dir(tmp_path / "ok", playing=0.2)
    (path / "session.json").write_text(json.dumps(_manifest(isolated=True)))
    doc = Listener(path, every=4).poll()[0]
    assert doc["input"] == "headphones" and doc["you_trusted"] == ALL_YOU
    assert doc["you_db"] == doc["mic_db"], "nothing of the band's is there to take away"
    assert doc["band_in_mic"] is False


def test_the_live_feed_goes_by_the_takes_own_decision_and_makes_none_of_its_own(tmp_path):
    """The band is in this mic either way.  Only the take's logged decision changes what is said."""
    for name, decided in (("believed", False), ("withdrawn", True)):
        path = _take_dir(tmp_path / name, playing=0.0, bleed=0.5)
        (path / "session.json").write_text(json.dumps(_manifest(isolated=True)))
        if decided:
            rows = [json.loads(line) for line in (path / "decisions.ndjson").read_text().splitlines()]
            for row in rows[2:]:
                row["band_in_mic"] = {"since_bar": 2, "share": 0.9, "reason": "x"}
            (path / "decisions.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
        docs = Listener(path, every=4).poll()
        assert [d["band_in_mic"] for d in docs] == [decided] * len(docs)
        assert all(d["input"] == ("room" if decided else "headphones") for d in docs)
        assert all(d["you_trusted"] == ([] if decided else ALL_YOU) for d in docs), "no flapping bar to bar"


def test_trust_is_by_what_the_band_can_reach():
    assert you_trust("headphones", "full", -20.0, 0.0)[0] == ALL_YOU
    assert you_trust("room", "full", -20.0, 0.0)[0] == []


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_take_page_does_not_say_it_lost_her_when_the_band_was_in_headphones(tmp_path, docs):
    script = f"await page.open(ids.tap); return {HEARING};"
    room = run_page(tmp_path, docs, script)
    assert room["noHidden"] is False
    docs["docs"][docs["ids"]["tap"]]["route"] = "headphones"
    out = run_page(tmp_path, docs, script)
    assert out["noHidden"] is True and out["lineHidden"] is False
    # The line runs to the end of the take, as it does for any input the band cannot reach.
    assert out["width"] > room["width"]


# ------------------------------------------------------------------------- what the page does

needs_node = pytest.mark.skipif(NODE is None, reason="needs node")
BAND_STEP = setup_doc("band", QUESTION, "yes", "no", device=MIC_ID, device_name="Test Mic", mode="room",
                      note="The band goes out to Test Phones.", output=SPEAKERS_ID, band_output=SPEAKERS_ID)


@needs_node
def test_the_page_sends_the_output_it_asked_about_with_the_answer(tmp_path, docs):
    home = json.loads(json.dumps(docs["home"]))
    mic = next(d for d in home["devices"] if d["mode"] == "room")
    asked = {**BAND_STEP, "device": mic["id"]}
    home.update(device=mic["id"], mode="room")          # home already shows this mic, so nothing has moved
    out = first_run(tmp_path, {**docs, "home": home}, f"""
        page.replies['setup/listen'] = [200, {json.dumps({**asked, "output": None, "heard": HEARD, "others": []})}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "device": mic["id"], "mode": "room"})}];
        $('setup-yes').click(); await page.settle();
        return {{ seen, said: page.posts('setup/answer') }};""")
    assert out["seen"]["say"] == "Is the band only in your headphones?" and out["seen"]["choices"] == ["Yes", "No"]
    assert out["seen"]["more"] == "If you can hear it in the room at all, say No."
    assert out["seen"]["note"] == "The band goes out to Test Phones."
    assert out["said"] == [{"step": "band", "answer": "yes", "device": mic["id"], "output": SPEAKERS_ID}]


@needs_node
def test_no_other_answer_carries_an_output(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps({**HEADPHONES, "heard": HEARD, "others": []})}];
        $('start').click(); await page.settle();
        page.replies['setup/answer'] = [200, {json.dumps(COUNT)}];
        $('setup-no').click(); await page.settle();
        return page.posts('setup/answer');""")
    assert out == [{"step": "headphones", "answer": "no", "device": HEADPHONES["device"]}]


@needs_node
def test_the_live_page_says_what_the_take_decided_and_what_to_do(tmp_path, docs):
    """From the take's own row.  The page works nothing out for itself."""
    row = {**ROW, "input": "room", "you_trusted": [], "playing": False, "playing_trusted": False, "fade_db": 0.0}
    seen = "{ hint: $('live-hint').hidden ? '' : $('live-hint').textContent, how: $('live-how').hidden ? '' : $('live-how').textContent }"
    found = live_page(tmp_path, docs, {**PLAYING, "mode": "room", "hear_me": False, "row": {**row, "band_in_mic": True}},
                      f"return {seen};")
    assert found["hint"] == ("Plume hears the band in this mic, so it has stopped going by what the mic hears. "
                             "Press Stop when you are done.")
    assert found["how"] == "A room mic may not hear you stop, so press Stop when you are done."
    for quiet in ({**row, "band_in_mic": False}, {**row, "input": "headphones", "you_trusted": ["you_db"], "playing_trusted": True}):
        out = live_page(tmp_path, docs, {**PLAYING, "mode": "room", "hear_me": False, "row": quiet}, f"return {seen};")
        assert out["hint"] == ""
    believed = live_page(tmp_path, docs, {**PLAYING, "mode": "room", "hear_me": False,
                                          "row": {**row, "input": "headphones", "playing_trusted": True}}, f"return {seen};")
    assert believed["how"] == "", "a mic the band cannot reach hears her stop"


def test_the_bridge_passes_the_takes_decision_to_the_page():
    assert "band_in_mic" in guitake._ROW_FIELDS


@needs_node
def test_home_does_not_call_a_mic_in_headphones_a_room_mic(tmp_path):
    from test_gui_static_logic import live_js
    mic = {"name": "Mic", "mode": "room"}
    assert live_js(tmp_path, f"live.routeLine({json.dumps(mic)}, 1, '', '', 'Phones', true)") == "Mic, mic, band in headphones, out to Phones"
    assert live_js(tmp_path, f"live.routeLine({json.dumps(mic)}, 1, '', '', 'Phones', false)") == "Mic, room mic, out to Phones"
    assert live_js(tmp_path, "live.routeLine(null, 1, '', '', '', true)") == "System default, mic, band in headphones"
    box = {"name": "Box", "mode": "direct"}
    assert live_js(tmp_path, f"live.routeLine({json.dumps(box)}, 2, '', '', '', true)") == "Box, channel 2"


@needs_node
def test_settings_does_not_say_speakers_would_howl_of_an_output_she_said_is_headphones(tmp_path, docs):
    home = json.loads(json.dumps(docs["home"]))
    mic = next(d for d in home["devices"] if d["mode"] == "room")
    out_id = home["outputs"][0]["id"]
    seen = "$('room-output').hidden ? '' : $('room-output').textContent"
    said = {}
    for believed in (True, False):
        mic.update(output=out_id, band_in_headphones=believed)
        home.update(device=mic["id"], mode="room", output=out_id, band_in_headphones=believed)
        said[believed] = run_page(tmp_path, {**docs, "home": home}, f"""
            $('nav-settings').click(); await page.settle();
            return {{ line: {seen}, gear: $('home-gear').textContent }};""")
    assert said[False]["line"] == "Hear me stays off here: a room mic through speakers would howl."
    assert said[True]["line"] == "Hear me stays off on an output you chose. A mic can only hear itself on the system output."
    assert "band in headphones" in said[True]["gear"] and "room mic" in said[False]["gear"]


# ----------------------------------------------- a real speaker, a real room, and every way in

def _rows_and_found(rec):
    rows = rows_of(rec)
    return rows, next((i for i, r in enumerate(rows) if "band_in_mic" in r), None)


@pytest.mark.parametrize("name,shape", [("nobass", {"low_cut_hz": 300.0, "rt60_s": 0.05, "wet": 0.0}),
                                        ("room", {"low_cut_hz": 300.0, "rt60_s": 0.5, "wet": 1.5})])
def test_a_shaped_leak_is_found_out_too(tmp_path, consent, name, shape):
    """The band reaches the mic through a small speaker, and then through a room as well."""
    from conftest import small_speaker_in_a_room
    rec = _mic_take(tmp_path, name, _direct_take(stops_at=45.0, total=400.0), rate_x=4.0, minutes=0.4,
                    fake={"bleed_db": -6.0, "bleed_shape": small_speaker_in_a_room(**shape)},
                    band_in_headphones=True)
    rows, found = _rows_and_found(rec)
    assert found is not None, "the band through a speaker and a room was never noticed"
    first_in = next((i for i, r in enumerate(rows) if r["entry"] != "drums"), len(rows))
    assert found <= first_in + 4
    assert not any(r["playing_trusted"] for r in rows[found:] if r["entry"] != "drums")


def test_she_stops_with_the_band_leaking_and_is_not_heard_playing_on(tmp_path, consent):
    """The band at -6 dB in the mic, and she stops at 45 seconds.  Bars that say she is playing
    and can be believed, after she has stopped, are the harm: there must be next to none."""
    rec = _mic_take(tmp_path, "stops", _direct_take(stops_at=45.0, total=400.0), rate_x=4.0, minutes=0.45,
                    fake={"bleed_db": -6.0}, band_in_headphones=True)
    rows, found = _rows_and_found(rec)
    assert found is not None
    wrong = [r["bar"] for r in rows if r["t_s"] > 52.0 and r["playing"] and r["playing_trusted"]]
    assert wrong == [], wrong
    assert rows[found]["t_s"] < 45.0, "found while she was still playing"


def test_plain_lead_checks_her_word_too(tmp_path, consent, maskers):
    """No count, so the band plays from the first bar.  The flag is honoured here, and so checked."""
    for name, fake in (("lead-leak", {"bleed_db": -6.0}), ("lead-clean", {})):
        device = fake_audio(_direct_take(stops_at=390.0, total=400.0), devices=RIG, rate_x=playhead_rate(4.0), **fake)
        with patched_module("sounddevice", device):
            realtime.lead(prompt="lofi in C major", seed=0, role=roles_mod.RoleParams(role="lead"),
                          record_dir=str(tmp_path / name), status=False,
                          socket_path=short_socket_path(name), max_minutes=0.15, band_in_headphones=True)
    rows, found = _rows_and_found(tmp_path / "lead-leak")
    assert found is not None and rows[found]["band_in_mic"]["since_bar"] == rows[found]["bar"]
    assert all(r.get("band_in_mic") == rows[found]["band_in_mic"] for r in rows[found:])
    assert _rows_and_found(tmp_path / "lead-clean")[1] is None
    assert [m._enabled for m in maskers] == [True, False], "masking comes back only where the band was found"


def test_jam_checks_her_word_too(tmp_path, consent, maskers):
    from test_live_correctness import strum
    for name, fake in (("jam-leak", {"bleed_db": -6.0}), ("jam-clean", {})):
        # a chord on every beat, which is what gets a jam's band to come in and stay
        device = fake_audio(strum(100.0, 200.0), devices=RIG, rate_x=playhead_rate(4.0), **fake)
        with patched_module("sounddevice", device):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), record_dir=str(tmp_path / name),
                         max_minutes=0.3, band_in_headphones=True)
    rows, found = _rows_and_found(tmp_path / "jam-leak")
    assert found is not None, "the band in the mic was never noticed in a jam"
    assert all("band_in_mic" in r for r in rows[found:]) and "room mic" in rows[found]["band_in_mic"]["reason"]
    assert _rows_and_found(tmp_path / "jam-clean")[1] is None
    assert [m._enabled for m in maskers] == [True, False]


# ------------------------------------------------------------ after the take stops believing

def _hops(x):
    return [float(v) for v in realtime.HopRms().process(x)]


def _mic_and_band(seconds=40.0, leak=0.06):
    """Her, strumming, with the band under her in the mic about 24 dB down."""
    from test_live_correctness import strum
    from plume.listen.features import FeatureExtractor
    band = np.concatenate([_a_band(seed=n) for n in range(int(seconds // 2))])
    band[:6 * realtime.SR] = 0.0          # she starts alone, as in any take
    t = np.arange(band.size) / realtime.SR
    held = sum(np.sin(2 * np.pi * f * t) for f in (130.81, 196.0, 329.63)) * 0.05
    her = strum(100.0, band.size / realtime.SR)[:band.size] + held.astype(np.float32)
    late = np.concatenate([np.zeros(960, dtype=np.float32), band[:-960]])
    frames = FeatureExtractor(sr=realtime.SR).process((her + leak * late).astype(np.float32))
    levels = _hops(band)[:len(frames)]
    return frames[:len(levels)], levels


def test_a_mask_switched_on_partway_masks_about_what_one_on_from_the_start_does():
    """It has been listening all along, so it knows her level and how much of the band the mic
    holds.  One that learned nothing would find every band frame louder than a player it had
    never heard, and blank the lot."""
    frames, levels = _mic_and_band()
    half = len(frames) // 2
    always = realtime.SelfMasker()
    always.mask(frames[:half], levels[:half])
    out_a, flags_a = always.mask(frames[half:], levels[half:])
    late = realtime.SelfMasker(enabled=False)
    out, flags = late.mask(frames[:half], levels[:half])
    assert not any(flags) and all(a is b for a, b in zip(out, frames[:half])), "off, it touches nothing"
    assert late.human_rms_db > -60.0, "and it has still learned how loud she is"
    late.enable()
    out_b, flags_b = late.mask(frames[half:], levels[half:])
    share_a, share_b = sum(flags_a) / len(flags_a), sum(flags_b) / len(flags_b)
    assert 0.0 < share_a < 0.9, "the stand-in take is not one a room mic can work with"
    assert abs(share_b - share_a) <= 0.1, (share_a, share_b)
    onsets_a, onsets_b = sum(f.onset for f in out_a), sum(f.onset for f in out_b)
    assert onsets_a > 0 and onsets_b >= 0.8 * onsets_a, (onsets_a, onsets_b)


def test_a_take_that_stops_believing_the_mic_still_hears_her_as_a_room_mic_would(tmp_path, consent, monkeypatch):
    """Her headphones come out a minute in, with the whole band playing, and a moderate amount
    of it reaches the mic while she strums on.  After the take falls back it masks about what a
    room mic does over the same stretch, and still finds where her notes land."""
    marks = []

    class Spy(realtime.SelfMasker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            marks.append({"at": None, "self": self, "seen": [], "masked": []})

        def enable(self):
            marks[-1]["at"] = self.frames_seen
            super().enable()

        def mask(self, frames, levels):
            out = super().mask(frames, levels)
            marks[-1]["seen"].append(self.frames_seen)
            marks[-1]["masked"].append(self.masked_frames)
            return out
    monkeypatch.setattr(realtime, "SelfMasker", Spy)
    human = _direct_take(stops_at=390.0, total=400.0)
    fake = {"bleed_db": -18.0, "bleed_from_s": 60.0}
    fell_back = _mic_take(tmp_path, "fell", human, rate_x=4.0, minutes=0.8, fake=fake, band_in_headphones=True)
    room = _mic_take(tmp_path, "room18", human, rate_x=4.0, minutes=0.8, fake=fake)
    late, always = marks[0], marks[1]
    assert late["at"] is not None, "the take never fell back, so this proves nothing"

    def masked_from(mark, start):
        """The share of frames masked from frame ``start`` to the end of the take."""
        at = next(i for i, seen in enumerate(mark["seen"]) if seen >= start)
        return (mark["masked"][-1] - mark["masked"][at]) / (mark["seen"][-1] - mark["seen"][at])
    after, as_room = masked_from(late, late["at"]), masked_from(always, late["at"])
    assert 0.0 < as_room < 0.9
    # No more than a room mic blanks, or her notes are lost with the band's; and it does mask.
    assert 0.6 * as_room <= after <= as_room + 0.08, (after, as_room)
    rows, found = _rows_and_found(fell_back)

    def placed(some):
        """The share of bars in which her notes were found and placed against the beat."""
        return sum(r.get("human_offset_ms") is not None for r in some) / len(some)
    room_placed = placed(rows_of(room)[found + 2:])
    print(f"masked after falling back {after:.2f}, a room mic over the same stretch {as_room:.2f}, "
          f"bars with her notes placed {placed(rows[found + 2:]):.2f} against {room_placed:.2f} over {len(rows[found + 2:])} bars")
    assert room_placed > 0.0, "a room mic finds none either, so this proves nothing"
    assert placed(rows[found + 2:]) >= 0.8 * room_placed, (placed(rows[found + 2:]), room_placed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("where", ["mic", "band"])
def test_a_stretch_with_a_sample_that_is_not_a_number_is_not_judged(bad, where):
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(12):
        mic, band = _bleeding(0.0, seed=n)
        (mic if where == "mic" else band)[1000 + n] = bad
        assert bandentry.band_share(mic, band, realtime.SR) == 0.0
        check.feed(mic, band)
    assert check.latched is False and check.share is None


def test_listening_for_the_band_costs_little(monkeypatch):
    """Counted, not timed: how many samples go through a transform for one stretch.  It listens
    between 300 Hz and 4 kHz, so it has no need of 48,000 samples a second."""
    from plume import bandentry
    done = []
    for name in ("rfft", "irfft"):
        real = getattr(np.fft, name)
        monkeypatch.setattr(np.fft, name, lambda a, n=None, axis=-1, _real=real: (
            done.append(int(np.asarray(a).size if n is None else n * (np.asarray(a).size // np.asarray(a).shape[axis]))),
            _real(a, n=n, axis=axis))[1])
    mic, band = _bleeding(0.6)
    assert bandentry.band_share(mic, band, realtime.SR) >= bandentry.BAND_IN_MIC_SHARE
    assert sum(done) <= 200_000, sum(done)


def test_a_few_strong_frequencies_do_not_stand_for_the_whole_band():
    """A third of the frequencies agree perfectly and the rest not at all.  That is her on the
    band's notes, not the band: the middle value says so where an average would not."""
    from plume import bandentry
    rng = np.random.default_rng(4)
    n = 2 * realtime.SR
    t = np.arange(n) / realtime.SR
    notes = np.arange(310.0, 3990.0, 150.0)
    shared = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in notes)
    band = (0.02 * shared + rng.standard_normal(n) * 0.02).astype(np.float32)
    her = (0.02 * shared + rng.standard_normal(n) * 3e-4).astype(np.float32)
    share = bandentry.band_share(her, band, realtime.SR)
    assert share < 0.5 * bandentry.BAND_IN_MIC_SHARE, share


def test_evidence_against_counts_as_much_as_evidence_for():
    """Every other stretch says yes and the ones between say no, for a long time.  That never
    adds up: a no takes back a yes."""
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(40):
        check.feed(*_bleeding(0.6 if n % 2 == 0 else 0.0, seed=n))
    assert check.latched is False
    for n in range(bandentry.BAND_IN_MIC_WINDOWS):
        check.feed(*_bleeding(0.6, seed=200 + n))
    assert check.latched is True, "and it has not gone deaf: a run of yes still latches"


def test_a_guess_at_the_style_stops_counting_how_busy_she_is():
    from plume.bandentry import StylePick
    pick = StylePick(key=None, trust_busy=True)
    pick.heard_bar(8)
    assert pick.busy == 2.0
    pick.distrust()
    pick.heard_bar(8)
    assert pick.busy is None


def test_the_style_guess_is_told_when_the_take_stops_believing_the_mic(tmp_path, consent, monkeypatch):
    """Found over the drums, before the style is picked: how busy the mic sounds is then the
    drums as much as her."""
    from plume import bandentry
    made = []

    class Pick(bandentry.StylePick):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)
    monkeypatch.setattr(bandentry, "StylePick", Pick)
    device = fake_audio(_direct_take(stops_at=390.0, total=400.0), devices=RIG, rate_x=playhead_rate(4.0), bleed_db=-6.0)
    with patched_module("sounddevice", device):
        realtime.lead(prompt=None, seed=0, tap_start=True, detect=True, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "pick"), status=False, socket_path=short_socket_path("pick"),
                      max_minutes=0.3, band_in_headphones=True)
    rows, found = _rows_and_found(tmp_path / "pick")
    assert found is not None and len(made) == 1
    assert made[0].busy is None


def test_a_fade_under_way_when_the_take_stops_believing_the_mic_finishes(tmp_path, consent, monkeypatch):
    """She has stopped in a noisy room and the band is fading out when the take stops believing
    the mic.  From there "playing" only means the mic is not silent, and a noisy room never is.
    The fade still runs to its end and the take ends.

    When the check trips is set by hand here, on the first bar of the fade: a leak that arrives
    then would be heard as her before it was found out, and bring the band back first."""
    from plume import bandentry
    checks = []

    class Check(bandentry.BandInMic):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            checks.append(self)

        def feed(self, mic, band):
            pass

    class Ending(bandentry.SilenceEnding):
        def update(self, *a, **kw):
            gain, done = super().update(*a, **kw)
            if gain < 0.0 and checks and not checks[0].latched:
                checks[0].share = 0.9
                checks[0].latched = True
            return gain, done
    monkeypatch.setattr(bandentry, "BandInMic", Check)
    monkeypatch.setattr(bandentry, "SilenceEnding", Ending)
    rec = _mic_take(tmp_path, "fade", _noisy(_direct_take(stops_at=45.0, total=400.0)), rate_x=4.0, minutes=0.6,
                    band_in_headphones=True)
    rows, found = _rows_and_found(rec)
    assert found is not None and rows[found - 1]["fade_db"] < 0.0, "it was not during a fade"
    after = rows[found:]
    assert any(r["playing"] for r in after), "the room is not noisy enough for this to prove anything"
    assert not any(r["playing_trusted"] for r in after)
    fades = [r["fade_db"] for r in rows[found - 1:]]
    assert all(b <= a < 0.0 for a, b in zip(fades, fades[1:])), fades
    assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE


def test_the_help_says_the_flag_is_about_a_mic_and_not_the_keys():
    text = " ".join(cli._build_parser().parse_args.__self__._subparsers._group_actions[0].choices["lead"].format_help().split())
    assert "--band-in-headphones" in text
    assert "With --midi the band listens to the keys, so this changes nothing" in text


def test_a_stretch_that_is_not_judged_counts_neither_way():
    """Three stretches say yes, one holds a sample that is no number, and one more says yes.
    The bad one took nothing back, so that is the fourth."""
    from plume import bandentry
    check = bandentry.BandInMic(realtime.SR)
    for n in range(bandentry.BAND_IN_MIC_WINDOWS - 1):
        check.feed(*_bleeding(0.6, seed=n))
    mic, band = _bleeding(0.0, seed=50)
    mic[5] = float("nan")
    check.feed(mic, band)
    assert check.latched is False
    check.feed(*_bleeding(0.6, seed=51))
    assert check.latched is True
