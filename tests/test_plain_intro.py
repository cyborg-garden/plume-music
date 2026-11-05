"""A plain beat until plume knows her style and key (Juniper, 2026-11-06).

"the lofi intro drum beat felt weird and clunky ... it should give a simple beat until
it both key and genre deduces".  A detected take used to start on the count's pick,
a guess from the tempo alone, with all that style's swing and character.  Now the
drums stage plays a plain kick, snare and hat, and the style's own drums come in
with the band, on that bar line.  A style she named is hers from the first bar.
"""
import numpy as np
import pytest
from test_detect import _detect_take, _rows, _strummed_take, consent  # noqa: F401

from plume import realtime
from plume.bandentry import (HARMONY_ROLES, MELODY_ROLES, PLAIN_INTRO_GAIN_DB,
                             PLAIN_INTRO_VOICES, plain_intro_style)
from plume.duet import DuetEngine
from plume.mind import BarPlan
from plume.styles_lib import list_styles, load_style


@pytest.mark.parametrize("name", list_styles())
def test_the_plain_beat_keeps_nothing_of_the_style_s_drums(name):
    s = plain_intro_style(name)
    assert s["swing"] == 0.5 and not s["timing_bias_ms"]
    drums = {r for r in s["voices"] if r not in HARMONY_ROLES + MELODY_ROLES}
    assert drums == set(PLAIN_INTRO_VOICES)
    assert {r: s["voices"][r] for r in drums} == PLAIN_INTRO_VOICES
    assert "slow_drum_patterns" not in s and not s["fills"]
    # Its harmony is still the style's, for the bar the band comes in on.
    assert s["progressions"] == load_style(name)["progressions"]
    assert load_style(name)["drum_patterns"] != s["drum_patterns"]   # not changed in place


def test_the_plain_beat_is_kick_on_one_and_three_snare_on_two_and_four():
    duet = DuetEngine(style=plain_intro_style("lofi"), seed=0)
    plan = BarPlan(bar=0, state="playing", roles={r: 1.0 for r in PLAIN_INTRO_VOICES},
                   gain_db=0.0, density=0.3, register_ceiling_midi=None, reason="test")
    beat = realtime.machine_beat_state(0, 0.0, 80.0)
    duet.render_bar(plan, beat, realtime.machine_harmony_state(0.0, "D major", None))
    bar_s = 3.0
    hits = {}
    for e in duet.last_events:
        hits.setdefault(e["role"], []).append(round(e["onset_s"] / (bar_s / 4), 1))
    assert sorted(hits) == ["hat", "kick", "snare"]
    assert hits["kick"] == [0.0, 2.0] and hits["snare"] == [1.0, 3.0]
    assert len(hits["hat"]) == 8


# ------------------------------------------------------------------ live, on the fake device

def test_a_detected_take_starts_plain_and_the_style_comes_in_with_the_band(tmp_path, consent):  # noqa: F811
    rec = _detect_take(tmp_path, _strummed_take(85.0), max_minutes=0.5)
    rows = _rows(rec)
    drums = [r for r in rows if r["entry"] == "drums"]
    band = [r for r in rows if r["entry"] != "drums"]
    assert drums and band, "the band never came in"
    assert {r["intro_beat"] for r in drums} == {"plain"}
    assert {r["intro_beat"] for r in band} == {"style"}
    assert {r["style"] for r in band} == {"midwest"}
    # quieter than the style's drums will be, by the plain beat's own amount
    assert drums[-1]["gain_db"] - band[0]["gain_db"] == pytest.approx(PLAIN_INTRO_GAIN_DB,
                                                                      abs=0.01)
    import soundfile as sf
    machine, _ = sf.read(str(rec / "machine.wav"), dtype="float32")
    bar_s = 240.0 / rows[0]["bpm"]
    first = int(drums[0]["t_s"] * realtime.SR)
    assert float(np.abs(machine[first:first + int(bar_s * realtime.SR)]).max()) > 1e-3


def test_a_style_she_named_plays_its_own_drums_from_the_first_bar(tmp_path, consent):  # noqa: F811
    rec = _detect_take(tmp_path, _strummed_take(85.0), prompt="trap", max_minutes=0.3)
    rows = _rows(rec)
    assert rows and {r["intro_beat"] for r in rows} == {"style"}
    assert {r["style"] for r in rows} == {"trap"}
