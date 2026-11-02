"""Harmonic commitment (v2.6) — the opinionated accompanist, headless."""
import pytest

from plume.governor import HarmonyGovernor, progression_symbols
from plume.listen.harmony import HarmonyState
from plume.mind import MindParams


def _h(key, chord, *, key_conf=0.6, chord_conf=0.5, t_s=0.0):
    return HarmonyState(t_s=t_s, key=key, chord=chord, key_conf=key_conf,
                        chord_conf=chord_conf,
                        register_low_midi=None, register_high_midi=None)


def test_progression_symbols_deterministic_and_in_key():
    a = progression_symbols("lofi", "C major")
    b = progression_symbols("lofi", "C major")
    assert a and a == b and all(isinstance(s, str) for s in a)
    assert progression_symbols("lofi", "A minor") != []


def test_disabled_is_pass_through():
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=0))
    h = _h("C major", "Fm")
    assert g.govern(5, h) is h


def test_tracking_passes_through_until_commit():
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=4))
    for bar in range(3):
        h = _h("C major", "Fm")
        assert g.govern(bar, h) is h        # untouched while undecided
    assert g.committed_key is None


def test_commits_after_stable_key_and_cycles_progression():
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=4))
    out = [g.govern(bar, _h("C major", ["G", "E", "Em", "Gm"][bar % 4]))
           for bar in range(12)]
    assert g.committed_key == "C major"
    prog = progression_symbols("lofi", "C major")
    committed = out[3:]                     # commit lands on the 4th stable bar
    got = [o.chord for o in committed]
    want = [prog[i % len(prog)] for i in range(len(committed))]
    # the flapping tracker chords (G/E/Em/Gm) are ignored: the emitted chords
    # are the progression, in order, one per bar
    assert got == want
    assert all(o.chord_conf == 1.0 and o.key == "C major" for o in committed)


def test_unlocked_tick_holds_and_never_advances():
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=2))
    g.govern(0, _h("C major", None))
    g.govern(1, _h("C major", None))        # commits here
    held = g.govern(None, _h("D major", "D"))
    again = g.govern(None, _h("E major", "E"))
    assert held.chord == again.chord        # held, not advanced
    assert held.key == "C major"            # excursions on ticks don't count
    nxt = g.govern(2, _h("C major", None))
    assert nxt.chord is not None


def test_short_excursion_is_color_sustained_move_modulates():
    g = HarmonyGovernor(
        "lofi", MindParams(harmony_commit_bars=2, harmony_release_bars=3))
    g.govern(0, _h("C major", None))
    g.govern(1, _h("C major", None))        # committed to C major
    # two bars away, then back: color, not modulation
    g.govern(2, _h("G major", None))
    g.govern(3, _h("G major", None))
    assert g.govern(4, _h("C major", None)).key == "C major"
    assert g.committed_key == "C major"
    # three CONSECUTIVE bars away: modulation, progression restarts there
    g.govern(5, _h("G major", None))
    g.govern(6, _h("G major", None))
    out = g.govern(7, _h("G major", None))
    assert g.committed_key == "G major" and out.key == "G major"
    assert out.chord == progression_symbols("lofi", "G major")[0]


def test_style_without_progressions_holds_key_keeps_tracker_chords(monkeypatch):
    import plume.governor as gov
    monkeypatch.setattr(gov, "progression_symbols", lambda *a: [])
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=2))
    g.govern(0, _h("C major", "Am"))
    out = g.govern(1, _h("C major", "Am", chord_conf=0.4))
    assert g.committed_key == "C major"
    assert out.key == "C major" and out.key_conf >= 0.9
    assert out.chord == "Am" and out.chord_conf == 0.4   # tracker's, untouched


def test_none_state_passes_through():
    g = HarmonyGovernor("lofi", MindParams(harmony_commit_bars=2))
    assert g.govern(3, None) is None


def test_transports_actually_route_through_the_governor():
    # Non-vacuity floor (same rationale as the placement test): every other
    # test here passes with the transport wiring deleted.
    import inspect

    from plume import realtime, sim
    assert inspect.getsource(realtime).count("governor.govern(") == 2
    assert inspect.getsource(sim).count("governor.govern(") == 2
