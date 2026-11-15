"""A mic, with the band only in headphones: the band cannot reach it, so it is believed.

No device opens here and nothing sounds.  The bridge runs on fakes, the transport on the fake
audio device in tests/conftest.py, and the setup file is a temporary one.  Nothing here says
anything about real hardware: how much of a band leaks out of real headphones into a real mic
is not something these can know.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import fake_audio, needs_laptop_speed, patched_module, short_socket_path
from plume import cli, guitake, realtime, record
from plume import roles as roles_mod
from plume.realtime import resolve_input
from plume.watch import Listener, you_trust
from test_direct_input import ALL_YOU, RIG, _direct_take, _play, _take_dir, consent, maskers, rows_of  # noqa: F401
from test_gui_output import (BOX_ID, MIC_ID, OTHER_ID, SPEAKERS, SPEAKERS_ID, choose, meter, rig,  # noqa: F401
                             runner, setup_file)
from record_fixture import ending_doc, song
from test_gui_page import HEARING, NODE, docs, run_page  # noqa: F401
from test_gui_setup import good, walk
from test_gui_take import LAPTOP_MIC, LAPTOP_OUT, _parsed

FLAG = "--band-in-headphones"
QUESTION = "Is the band only in your headphones?"


def band(runner, answer, device=MIC_ID):
    return runner.answer({"step": "band", "answer": answer, "device": device})


def heard_on_the_mic(runner, meter, output=SPEAKERS_ID):
    """A mic she has been heard on, with its sound sent to an output she chose."""
    choose(runner, output, device=MIC_ID)
    good(meter, LAPTOP_MIC, -12.0)
    return runner.listen({"device": MIC_ID})


# ------------------------------------------------------------------------ the one question

def test_a_mic_sent_to_a_chosen_output_is_asked_one_plain_question(runner, meter):
    doc = heard_on_the_mic(runner, meter)
    assert (doc["step"], doc["line"], doc["done"]) == ("band", QUESTION, False)
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
    assert doc["step"] == "band" and doc["done"] is False
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


def test_an_instrument_plugged_in_is_never_asked(runner, meter):
    choose(runner, SPEAKERS_ID)
    good(meter)
    doc = runner.listen({"device": BOX_ID})
    assert doc["done"] is True and doc["step"] == "count"
    with pytest.raises(guitake.Refused):
        band(runner, "yes", device=BOX_ID)


def test_the_question_is_not_asked_of_a_mic_on_the_systems_output(rig, runner, meter):
    """That one is already asked whether headphones are on, which is the same thing there."""
    rig.devices = [LAPTOP_MIC, LAPTOP_OUT]
    good(meter, LAPTOP_MIC, -12.0)
    assert runner.listen({})["step"] == "headphones"
    with pytest.raises(guitake.Refused):
        runner.answer({"step": "band", "answer": "yes"})


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


def test_bare_lead_passes_no_word_about_headphones(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["lead", "--tap-start", "lofi"]) == 0
    assert "band_in_headphones" not in seen


@needs_laptop_speed
def test_a_mic_with_the_band_in_headphones_is_believed_once_the_band_is_in(tmp_path, consent, maskers):
    """The same take as the room mic's, with nothing of the band in the mic."""
    rec = _play(tmp_path, human=_direct_take(), name="phones", band_in_headphones=True)
    rows = rows_of(rec)
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert in_band and all(r["playing_trusted"] for r in rows)
    assert any(r["playing"] for r in in_band) and any(not r["playing"] for r in in_band)
    assert not any("room mic" in json.dumps(r.get("chords")) for r in in_band)
    # Nothing of the band's is there to mask, and it hears her stop.
    assert [m._enabled for m in maskers] == [False]
    assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE
    manifest = json.loads((rec / "session.json").read_text())
    assert manifest["input"]["source"] == "room" and manifest["input"]["isolated"] is True
    assert record.input_route(manifest) == "headphones"


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
    # A yes that was wrong: the band is in the mic, so it is measured and not believed.
    path = _take_dir(tmp_path / "leak", playing=0.0, bleed=0.5)
    (path / "session.json").write_text(json.dumps(_manifest(isolated=True)))
    doc = Listener(path, every=4).poll()[0]
    assert doc["input"] == "headphones" and doc["you_trusted"] == []
    assert "headphones" in doc["you_why"]


def test_trust_is_by_what_the_band_can_reach():
    assert you_trust("headphones", "full", -20.0, 0.0)[0] == ALL_YOU
    assert you_trust("headphones", "full", -20.0, 0.5)[0] == []
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
