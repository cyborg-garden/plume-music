"""What she plays, when she says: an optional choice, and each place it is used.

Not set is the default, and with it nothing differs from a take that was never told.
Each use is small: the style guess leans a little away from folk on an instrument
that plays one note at a time, and the band's parts in her register step back.
"""
from __future__ import annotations

import json

import pytest
from conftest import fake_audio, patched_module, short_socket_path

from plume import bandentry, cli, duet, instrument, playstyle, realtime
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


@pytest.mark.parametrize("text", ["kazoo", "guitar; rm", 3, ["bass"],
                                  "synth"])     # a synth may be keys or a lead line
def test_anything_else_is_refused(text):
    with pytest.raises(ValueError):
        instrument.canonical(text)


def test_every_choice_has_a_label_and_not_set_comes_first():
    assert instrument.INSTRUMENTS == ("guitar", "bass", "keys", "lead", "other")
    assert set(instrument.LABELS) == set(instrument.INSTRUMENTS)


# ------------------------------------------------------------ not set: as before

@pytest.mark.parametrize("name", [None, "other", "guitar"])
def test_without_a_choice_that_changes_anything_the_band_is_the_style_itself(name):
    for style in list_styles():
        assert instrument.make_room(style, name) == style
        plain = bandentry.plain_intro_style(style)
        assert instrument.make_room(plain, name) is plain


@pytest.mark.parametrize("name", [None, "other", "guitar", "keys"])
def test_the_style_guess_is_as_before(name):
    assert instrument.style_prior(name) is None


# ------------------------------------------------------------- the style guess

def _strum_pick(audio, *, bpm=96.0, bars=7, **kw):
    """The first ``bars`` bars of ``audio`` into a direct input's StylePick."""
    from plume.listen.features import HOP, FeatureExtractor
    fe = FeatureExtractor(sr=48000)
    frames = []
    for i in range(0, len(audio), 4800):
        frames.extend(fe.process(audio[i:i + 4800]))
    bar_s = 240.0 / bpm
    per = bar_s * 48000 / HOP
    pick = bandentry.StylePick(trust_busy=True, hear_technique=True, **kw)
    for b in range(bars):
        bar = frames[int(round(b * per)):int(round((b + 1) * per))]
        pick.heard_bar(sum(bool(f.onset) for f in bar), frames=bar, bar_start=b * bar_s,
                       bar_s=bar_s)
    return pick


@pytest.mark.parametrize("name", ["bass", "lead"])
def test_one_note_at_a_time_leans_away_from_folk_only_a_little(name):
    """A strummed chord needs strings sounding together, so on a bass line or a single
    lead line a long attack is more likely a slide or a swell: folk costs a little.  A
    strum share just inside folk's band goes to the next style; one a few points in is
    still folk.  Every other style scores as before."""
    prior = instrument.style_prior(name)
    assert set(prior) == {"folk"} and playstyle.MARGIN < prior["folk"] < playstyle.NO_FIT
    edge = playstyle.PROFILES["folk"]["strum"][0]
    for strum, folk in ((edge + 0.01, False), (edge + 0.08, True), (0.8, True)):
        heard = playstyle.Playing(strokes=2.0, swing=0.5, offbeat=0.1, steady=0.8,
                                  strum=strum)
        assert playstyle.score_styles(96.0, "major", heard)[0].style == "folk"
        told = playstyle.score_styles(96.0, "major", heard, prior)
        assert (told[0].style == "folk") is folk, (strum, told[:2])
        assert {s.style: s.score for s in told if s.style != "folk"} == \
               {s.style: s.score for s in playstyle.score_styles(96.0, "major", heard)
                if s.style != "folk"}
        assert playstyle.fit("folk", heard, prior) == \
               pytest.approx(playstyle.fit("folk", heard) - prior["folk"])


def test_a_strum_heard_on_a_bass_is_still_folk_and_says_why():
    """Not a rule: double stops strummed on a bass are a strum, and a strum gets folk."""
    import folk_fixtures as ff
    pick = _strum_pick(ff.make_take("strum"), style_prior=instrument.style_prior("bass"))
    assert pick.at_entry(96.0, "G major") == "folk", pick.why
    assert "instrument 0.50" in pick.why


def test_after_the_band_is_in_the_guess_keeps_leaning(monkeypatch):
    """A change of mind is scored with the same lean as the entry."""
    seen = []
    real = bandentry.score_styles
    monkeypatch.setattr(bandentry, "score_styles",
                        lambda *a, **kw: seen.append(a[3:] or kw) or real(*a, **kw))
    monkeypatch.setattr(bandentry, "fit",
                        lambda *a, **kw: seen.append(a[2:] or kw) or playstyle.fit(*a, **kw))
    prior = instrument.style_prior("lead")
    pick = bandentry.StylePick(trust_busy=True, style_prior=prior)
    for b in range(64):
        pick.heard_bar(8, onset_times=[b * 2.5 + i * 0.3125 for i in range(8)],
                       bar_start=b * 2.5, bar_s=2.5)
        if b == 7:
            pick.at_entry(96.0, "A minor")
        elif b > 7:
            pick.reconsider(b - 7, 96.0, "A major" if b > 20 else "A minor")
    assert len(seen) > 2 and all(s == (prior,) for s in seen), seen


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


@pytest.mark.parametrize("how", [{}, {"plain": True}, {"bridge": True}])
@pytest.mark.parametrize("name", [None, "bass", "keys"])
def test_every_engine_a_take_builds_has_her_room(name, how):
    """The first engine, and each one a change of style builds as the band comes in
    (the plain beat's harmony, the bridge from it, the style itself): one builder makes
    them all, with her register stepped back, and not set the style as main builds it."""
    eng = realtime._band_engine("lofi", name, key_fallback="A minor", seed=0, **how)
    spec = (bandentry.plain_intro_style("lofi") if how.get("plain")
            else bandentry.bridged_style("lofi") if how.get("bridge") else load_style("lofi"))
    got = eng._style
    for role in ("bass", "sub", "keys", "pad", "lead", "arp", "drums"):
        trim = instrument.ROOM_DB.get(name, {}).get(role, 0.0)
        assert _db(got, role) == pytest.approx(_db(spec, role) + trim), (role, how)
    assert {k: v for k, v in got.items() if k != "role_gain_db"} == \
           {k: v for k, v in spec.items() if k != "role_gain_db"}


def test_a_take_builds_its_engines_only_through_the_one_builder():
    """A change of style builds a new engine mid-take; built anywhere but
    :func:`realtime._band_engine` it would lose her room after the band came in."""
    import inspect
    assert "DuetEngine(" not in inspect.getsource(realtime.lead)
    assert inspect.getsource(realtime._band_engine).count("make_room(") == 1


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
    """A short take, just past the count: the lean the style pick is given, the room
    each engine is given (the first, and the one the count's change of style builds),
    and the record.  Not set hands the pick and the engines what main does."""
    from test_bandentry import _take
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
    with patched_module("sounddevice", fake_audio(_take(seconds=9.0), devices=RIG,
                                                   rate_x=6.0)):
        realtime.lead(prompt=None, detect=True, tap_start=True, seed=0, status=False,
                      role=roles_mod.RoleParams(role="lead"), max_minutes=0.06,
                      socket_path=short_socket_path("inst"), record_dir=str(tmp_path / "rec"),
                      input_device="audio interface", input_channel=1, direct=True, **kw)
    assert picks and len(specs) >= 2, "the count changes the style"
    assert isinstance(specs[1], dict) and specs[1]["drum_patterns"] == \
        bandentry.plain_intro_style(specs[1]["name"])["drum_patterns"]
    assert picks[0]["hear_technique"] is True
    if name is None:
        assert "style_prior" not in picks[0]
    else:
        assert picks[0]["style_prior"] == instrument.style_prior(name)
    doc = json.loads((tmp_path / "rec" / realtime.REC_MANIFEST).read_text())
    assert doc["instrument"] == name
    for s in specs:
        base = load_style(s if isinstance(s, str) else s["name"])
        if name is None:
            assert isinstance(s, str) or _db(s, "bass") == _db(base, "bass")
        else:
            assert isinstance(s, dict), "every engine gets the room made"
            assert _db(s, "bass") == pytest.approx(_db(base, "bass") - 6.0)
