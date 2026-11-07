"""The band's entrance after the plain click (Juniper, 2026-11-07).

On a detected lofi take the click held one hit on every beat for bars, then
stopped dead on the entry bar while the style's drums came in under it, their loudest
hits between the beats and nothing on 3.  She heard it come "in awkwardly on the
beat".  Now the click carries on through the drums' swell and fades out as they reach
full, and for the first phrase the drums play the style's most on-beat figures, its
swing eased in.  After that phrase the style's own pattern choice takes over.

Nothing here plays sound: live takes run on the fake device.
"""
import numpy as np
import pytest
from record_fixture import ending_doc, song
from test_detect import _detect_take, _rows, _strummed_take, consent  # noqa: F401

from plume import notation, realtime, record
from plume.bandentry import (CLICK_ROLE, PLAIN_INTRO_GAIN_DB, PLAIN_INTRO_VOICES,
                             bridged_style, click_bridge_level, entry_ramp_db,
                             plain_intro_style, plain_phrase_feel)
from plume.duet import DuetEngine, plain_entry_grids
from plume.mind import BarPlan, MindParams
from plume.styles_lib import list_styles, load_style

BPM = 80.0
BAR_S = 240.0 / BPM


def _plan(bar, roles, density=0.5):
    return BarPlan(bar=bar, state="playing", roles=roles, gain_db=0.0, density=density,
                   register_ceiling_midi=None, reason="test")


def _render(duet, bar, roles, *, entry_feel=None, density=0.5):
    duet.render_bar(_plan(bar, roles, density), realtime.machine_beat_state(bar, bar * BAR_S, BPM),
                    realtime.machine_harmony_state(bar * BAR_S, "D major", "D"),
                    entry_feel=entry_feel)
    return duet.last_events


def _beats(events, role):
    """Where ``role``'s hits fell, in beats from the bar line."""
    return [e["onset_s"] / (BAR_S / 4) for e in events if e["role"] == role]


# ------------------------------------------------------------------ the click's bridge

def test_the_click_fades_out_as_the_drums_swell_in():
    p = MindParams()
    levels = [click_bridge_level(k, p.ramp_bars) for k in range(p.ramp_bars + 3)]
    drums = [entry_ramp_db(k, under_db=p.enter_under_db, ramp_bars=p.ramp_bars)
             for k in range(p.ramp_bars + 3)]
    for k, (click, drum_db) in enumerate(zip(levels, drums)):
        if drum_db < 0.0:
            assert 0.0 < click < 1.0, f"bar {k}: drums under full, so the click is there"
        else:
            assert click == 0.0, f"bar {k}: drums at full, so the click is gone"
    assert levels[:p.ramp_bars] == sorted(levels[:p.ramp_bars], reverse=True)
    assert click_bridge_level(0, 0) == 0.0          # no swell, nothing to bridge
    assert click_bridge_level(-1, 3) == 0.0


@pytest.mark.parametrize("name", list_styles())
def test_the_bridged_style_is_the_style_with_the_click_kept(name):
    s, own = bridged_style(name), load_style(name)
    assert s["voices"][CLICK_ROLE] == PLAIN_INTRO_VOICES[CLICK_ROLE]
    assert {r: v for r, v in s["voices"].items() if r != CLICK_ROLE} == own["voices"]
    assert s["swing"] == own["swing"] and s["timing_bias_ms"] == own["timing_bias_ms"]
    assert CLICK_ROLE not in own["voices"]          # loaded fresh, not changed in place


@pytest.mark.parametrize("name", list_styles())
def test_a_click_at_full_sounds_as_it_did_over_the_intro(name):
    """No step down on the entry bar beyond the bridge's own."""
    def click_peak(style, gain_db):
        duet = DuetEngine(style=style, seed=0)
        out = np.concatenate([duet.render_bar(
            BarPlan(bar=b, state="playing", roles={CLICK_ROLE: 1.0}, gain_db=gain_db,
                    density=0.5, register_ceiling_midi=None, reason="test"),
            realtime.machine_beat_state(b, b * BAR_S, BPM),
            realtime.machine_harmony_state(b * BAR_S, "D major", "D")) for b in range(2)])
        return float(np.abs(out).max())
    intro = click_peak(plain_intro_style(name), PLAIN_INTRO_GAIN_DB)
    bridge = click_peak(bridged_style(name), 0.0)
    assert bridge == pytest.approx(intro, rel=0.02)


# ------------------------------------------------------------------ the plain first phrase

def _plain(name):
    return DuetEngine(style=name, seed=0)._entry_grids


@pytest.mark.parametrize("name", list_styles())
def test_the_plain_phrase_kick_is_on_the_beat_in_every_style(name):
    grids = _plain(name)
    kick = notation.steps_for_cycle(grids["kick"], 0)
    assert kick and kick[0].start == 0.0, "the one"
    assert all(abs(s.start * 4 - round(s.start * 4)) < 1e-9 for s in kick)
    # and beat 3 is never left empty by the kick and the backbeat both
    on3 = [s for r in ("kick", "sub", "snare", "clap") if r in grids
           for s in notation.steps_for_cycle(grids[r], 0) if abs(s.start - 0.5) < 1e-9]
    assert on3


@pytest.mark.parametrize("name", list_styles())
def test_the_plain_phrase_is_chosen_from_the_style_not_rewritten(name):
    """Each role plays one of the style's own figures; the kick only drops hits, and adds
    a beat-3 kick only when the style's kick and backbeat both leave it empty."""
    duet = DuetEngine(style=name, seed=0)
    for role, steps in duet._entry_grids.items():
        own = [{(s.start, s.token) for s in v} for v in duet._grids[role]]
        mine = {(s.start, s.token) for s in steps}
        if role in ("kick", "sub"):
            extra = mine - set().union(*own)
            assert extra <= {(0.5, "bd:v0.6")}, role
        else:
            assert mine in own, role


def test_four_on_the_floor_and_the_jazz_kick_stay_as_they_are():
    for name in ("house", "synthwave"):
        kick = [s.start for s in _plain(name)["kick"]]
        assert kick == [0.0, 0.25, 0.5, 0.75], name
    assert [s.start for s in _plain("jazz")["kick"]] == [0.0, 0.25, 0.5, 0.75]
    assert [s.token for s in _plain("jazz")["hat"]][:2] == ["rd:v0.72", "rd:v0.68"]


def test_lofi_s_plain_phrase_is_a_plain_backbeat():
    grids = _plain("lofi")
    assert [(s.start, s.token) for s in grids["kick"]] == [(0.0, "bd"), (0.5, "bd:v0.6")]
    assert [s.start for s in grids["snare"]] == [0.25, 0.75]


def test_a_style_can_name_its_own_entry_patterns():
    s = load_style("lofi")
    s["entry_patterns"] = {"kick": "bd ~ ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ ~ ~", "hat": "hh ~ ~ ~"}
    grids = DuetEngine(style=s, seed=0)._entry_grids
    assert [s.start for s in grids["kick"]] == [0.0, 0.5]
    assert [s.start for s in grids["hat"]] == [0.0]


def test_plain_entry_grids_adds_nothing_where_the_backbeat_has_beat_3():
    parse = notation.parse
    grids = plain_entry_grids({"kick": [parse("bd ~ ~ bd ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~")],
                               "snare": [parse("~ ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~")]})
    assert [s.start for s in grids["kick"]] == [0.0]


@pytest.mark.parametrize("name", list_styles())
def test_the_rendered_plain_phrase_hits_the_beat_and_the_style_comes_back(name):
    duet = DuetEngine(style=name, seed=0)
    drums = {r: 1.0 for r in duet._grids}
    plain = _beats(_render(duet, 0, drums, entry_feel=0.0), "kick")
    # straight on its first bar: only the humanize jitter, a few ms either way
    assert plain and all(abs(b - round(b)) < 0.03 for b in plain), plain
    # the phrase over, the style's own figure at this density
    after = DuetEngine(style=name, seed=0)
    own = _beats(_render(after, 4, drums), "kick")
    again = _beats(_render(duet, 4, drums), "kick")
    assert again == pytest.approx(own)


def test_lofi_s_own_kick_is_back_after_the_phrase():
    duet = DuetEngine(style="lofi", seed=0)
    drums = {r: 1.0 for r in duet._grids}
    first = _beats(_render(duet, 0, drums, entry_feel=0.0), "kick")
    later = _beats(_render(duet, 4, drums), "kick")
    assert len(first) == 2
    assert any(abs(b - round(b)) > 0.2 for b in later), "the style's pushed kick again"


def test_the_swing_eases_in_over_the_phrase():
    duet = DuetEngine(style="lofi", seed=0)
    feel = [plain_phrase_feel(k, 4) for k in range(6)]
    assert feel[:4] == [0.0, 0.25, 0.5, 0.75] and feel[4:] == [None, None]
    swings = []
    for f in feel[:4] + [None]:
        duet._entry_feel = f
        swings.append(duet._groove_at(BPM).swing)
    assert swings[0] == 0.5 and swings[-1] == load_style("lofi")["swing"]
    assert swings == sorted(swings)
    duet._entry_feel = 0.0
    assert all(v == 0.0 for v in duet._groove_at(BPM).timing_bias_ms.values())


# ------------------------------------------------------------------ live, on the fake device

def test_the_click_bridges_the_entry_and_the_first_phrase_is_plain(tmp_path, consent,  # noqa: F811
                                                                   monkeypatch):
    """No bar where the click has stopped while the drums are still under full."""
    import plume.duet as duet_mod
    heard: list[tuple[int, set]] = []

    class Listening(duet_mod.DuetEngine):
        def render_bar(self, plan, beat, harmony, **kw):
            out = super().render_bar(plan, beat, harmony, **kw)
            heard.append((int(plan.bar), {e["role"] for e in self.last_events},
                          kw.get("entry_feel")))
            return out

    monkeypatch.setattr(duet_mod, "DuetEngine", Listening)
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.5)
    rows = _rows(rec)
    band = [r for r in rows if r["entry"] != "drums"]
    assert band and len(band) > 6, "the band never came in"
    p = MindParams()
    sounded = {b: roles for b, roles, _ in heard}
    feel = {b: f for b, _, f in heard}
    for i, row in enumerate(band):
        under = any(v < 1.0 for r, v in row["roles"].items()
                    if r in ("kick", "snare", "hat"))
        if i < p.ramp_bars:
            assert under
            assert 0.0 < row["click_bridge"] < 1.0
            assert row["roles"][CLICK_ROLE] == row["click_bridge"]
            assert CLICK_ROLE in sounded[row["bar"]], f"bar {row['bar']}: no click heard"
            assert "click fading out" in row["entry_reason"]
        else:
            assert not under, "the drums are full once the click is gone"
            assert "click_bridge" not in row and CLICK_ROLE not in row["roles"]
            assert CLICK_ROLE not in sounded[row["bar"]]
        if i < p.entry_boundary_bars:
            assert row["plain_phrase"] == pytest.approx(i / p.entry_boundary_bars)
            assert feel[row["bar"]] == pytest.approx(i / p.entry_boundary_bars)
            assert "plain first phrase" in row["entry_reason"]
        else:
            assert "plain_phrase" not in row and feel[row["bar"]] is None
    # the click over the intro never went away before the bridge took it
    drums = [r for r in rows if r["entry"] == "drums"]
    assert all(CLICK_ROLE in r["roles"] for r in drums)
    # and the page says what happened, on the bar it did
    events = record.translate(rows, None)
    entry = next(e for e in events if e["label"] == "bass and chords")
    assert "The click fades out as the style's drums swell in." in entry["sentence"]
    assert "Their first phrase stays plain, on the beat." in entry["sentence"]


def test_a_style_she_named_has_no_click_to_bridge(tmp_path, consent):  # noqa: F811
    rec = _detect_take(tmp_path, _strummed_take(85.0), prompt="trap", max_minutes=0.3)
    rows = _rows(rec)
    assert rows
    assert not any("click_bridge" in r or "plain_phrase" in r for r in rows)
    assert not any(CLICK_ROLE in r["roles"] for r in rows)


def test_the_recap_claims_no_bridge_that_did_not_happen():
    events = record.translate(song(40, band_bar=9), ending_doc())
    entry = next(e for e in events if e["label"] == "bass and chords")
    assert entry["sentence"] == "Bar 9. Bass and chords in, in F major."
