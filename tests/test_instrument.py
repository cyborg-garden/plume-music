"""What she plays, when she says: an optional choice, and each place it is used.

Not set is the default, and with it nothing differs from a take that was never told.
Each use is small: the strum is not listened for on an instrument that plays one note
at a time, and the band's parts in her register step back.
"""
from __future__ import annotations

import json

import pytest
from conftest import fake_audio, patched_module, short_socket_path

from plume import bandentry, cli, duet, instrument, realtime
from plume import roles as roles_mod
from plume.styles_lib import list_styles, load_style


# ------------------------------------------------------------------ the choice

@pytest.mark.parametrize("text,canon", [
    (None, None), ("", None), ("  ", None), ("not set", None),
    ("guitar", "guitar"), (" Guitar ", "guitar"), ("bass", "bass"), ("keys", "keys"),
    ("piano", "keys"), ("voice", "lead"), ("lead", "lead"), ("other", "other"),
])
def test_a_choice_comes_out_canonical(text, canon):
    assert instrument.canonical(text) == canon


@pytest.mark.parametrize("text", ["kazoo", "guitar; rm", 3, ["bass"]])
def test_anything_else_is_refused(text):
    with pytest.raises(ValueError):
        instrument.canonical(text)


def test_every_choice_has_a_label_and_not_set_comes_first():
    assert instrument.INSTRUMENTS == ("guitar", "bass", "keys", "lead", "other")
    assert set(instrument.LABELS) == set(instrument.INSTRUMENTS)
    assert instrument.NOT_SET_LABEL


# ------------------------------------------------------------ not set: as before

@pytest.mark.parametrize("name", [None, "other", "guitar"])
def test_without_a_choice_that_changes_anything_the_band_is_the_style_itself(name):
    for style in list_styles():
        assert instrument.make_room(style, name) == style
        plain = bandentry.plain_intro_style(style)
        assert instrument.make_room(plain, name) is plain


@pytest.mark.parametrize("name", [None, "other", "guitar", "keys"])
def test_the_strum_is_listened_for_as_before(name):
    assert instrument.hears_strum(name) is True


# ------------------------------------------------------------- the style guess

@pytest.mark.parametrize("name", ["bass", "lead"])
def test_one_note_at_a_time_is_never_heard_as_a_strum(name):
    """A strummed chord needs strings sounding together: on a bass line or a single
    lead line a long attack is a slide or a swell, so it is not counted as one, and a
    style guess on it never lands on folk for a strum she cannot play."""
    assert instrument.hears_strum(name) is False
    pick = bandentry.StylePick(trust_busy=True,
                               hear_technique=instrument.hears_strum(name))
    assert pick.folk_ear is None and pick.carry_on().folk_ear is None


# ------------------------------------------------------- her register in the band

def _db(style: dict, role: str) -> float:
    return float(style.get("role_gain_db", {}).get(role, duet._ROLE_GAIN_DB.get(role, -6.0)))


@pytest.mark.parametrize("name,roles", [
    ("bass", ("bass", "sub")),
    ("keys", ("keys", "pad")),
    ("lead", ("lead", "arp")),
])
def test_the_band_parts_in_her_register_step_back_and_nothing_else_moves(name, roles):
    for style in list_styles():
        base = load_style(style)
        room = instrument.make_room(style, name)
        for role in set(base["voices"]) | set(base.get("role_gain_db", {})) | set(roles):
            want = _db(base, role) + (instrument.ROOM_DB[name][role] if role in roles else 0.0)
            assert _db(room, role) == pytest.approx(want), (style, role)
        assert {k: v for k, v in room.items() if k != "role_gain_db"} == \
               {k: v for k, v in base.items() if k != "role_gain_db"}


def test_the_steps_back_are_small():
    for trims in instrument.ROOM_DB.values():
        assert all(-6.0 <= db < 0.0 for db in trims.values())


def test_room_is_made_on_a_changed_style_too_and_leaves_it_alone():
    plain = bandentry.plain_intro_style("lofi")
    before = json.dumps(plain, sort_keys=True)
    room = instrument.make_room(plain, "bass")
    assert json.dumps(plain, sort_keys=True) == before
    assert _db(room, "bass") == pytest.approx(_db(plain, "bass") - 6.0)
    assert room["drum_patterns"] == plain["drum_patterns"]


# ---------------------------------------------------------------- the CLI

@pytest.fixture
def stub_lead(monkeypatch):
    seen = {}
    monkeypatch.setattr(realtime, "check_device", lambda *a: None)
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    return seen


def test_the_cli_hands_the_choice_on(stub_lead):
    assert cli.main(["lead", "--tap-start", "--detect", "--instrument", "bass"]) == 0
    assert stub_lead["instrument"] == "bass"


def test_without_the_flag_the_transport_gets_exactly_what_it_did(stub_lead):
    assert cli.main(["lead", "--tap-start", "--detect"]) == 0
    assert "instrument" not in stub_lead


def test_an_unknown_instrument_is_a_clean_error(stub_lead, capsys):
    assert cli.main(["lead", "--tap-start", "--detect", "--instrument", "kazoo"]) != 0
    assert "kazoo" in capsys.readouterr().err
    assert stub_lead == {}


# ------------------------------------------------------------ the take itself

def test_the_record_says_what_she_plays_and_not_set_says_so():
    doc = realtime.session_manifest(kind="lead", style="detect", role="lead", seed=0)
    assert doc["instrument"] is None
    doc = realtime.session_manifest(kind="lead", style="detect", role="lead", seed=0,
                                    instrument="keys")
    assert doc["instrument"] == "keys"


def test_the_transport_refuses_an_unknown_instrument_before_a_device(monkeypatch):
    monkeypatch.setattr(realtime, "_import_sounddevice", lambda: pytest.fail("reached audio"))
    with pytest.raises(realtime.RealtimeError, match="kazoo"):
        realtime.lead(prompt=None, detect=True, tap_start=True, instrument="kazoo")


RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.mark.parametrize("name", [None, "bass"])
def test_a_detect_take_uses_the_choice_where_it_should(tmp_path, monkeypatch, plume_registry,
                                                       name):
    """A very short take: whether the strum is listened for, the room each engine is
    given, and the record.  Not set hands the pick and the engine what main does."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")      # the FAKE device only
    picks, specs = [], []
    real_pick, real_engine = bandentry.StylePick, duet.DuetEngine

    class Pick(real_pick):
        def __init__(self, **kw):
            picks.append(kw)
            super().__init__(**kw)

    class Engine(real_engine):
        def __init__(self, *, style, **kw):
            specs.append(style)
            super().__init__(style=style, **kw)
    monkeypatch.setattr(bandentry, "StylePick", Pick)
    monkeypatch.setattr(duet, "DuetEngine", Engine)
    kw = {} if name is None else {"instrument": name}
    with patched_module("sounddevice", fake_audio(None, devices=RIG, rate_x=6.0)):
        realtime.lead(prompt=None, detect=True, tap_start=True, seed=0, status=False,
                      role=roles_mod.RoleParams(role="lead"), max_minutes=0.04,
                      socket_path=short_socket_path("inst"), record_dir=str(tmp_path / "rec"),
                      input_device="audio interface", input_channel=1, direct=True, **kw)
    assert picks and specs
    assert picks[0]["hear_technique"] is (name is None)
    doc = json.loads((tmp_path / "rec" / realtime.REC_MANIFEST).read_text())
    assert doc["instrument"] == name
    for s in specs:
        base = load_style(s if isinstance(s, str) else s["name"])
        if name is None:
            assert isinstance(s, str) or _db(s, "bass") == _db(base, "bass")
        else:
            assert isinstance(s, dict), "every engine gets the room made"
            assert _db(s, "bass") == pytest.approx(_db(base, "bass") - 6.0)
