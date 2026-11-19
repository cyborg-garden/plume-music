"""The key the band comes in on: mode, maj7 home chords, relative keys, the entry check.

Synthetic frames only (duck-typed, as listen/harmony.py reads them): no audio, no
device.  Each scenario is one the key tracker used to get wrong, built from the
pitch content that made it wrong.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from plume.bandentry import BandEntry
from plume.listen.harmony import HarmonyTracker
from plume.theory import parse_key

HOP_S = 512 / 48000
BAR_S = 2.0

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11
MAJ7, MIN7, DOM7 = (0, 4, 7, 11), (0, 3, 7, 10), (0, 4, 7, 10)
MAJ9, MIN9 = (0, 4, 7, 11, 14), (0, 3, 7, 10, 14)


def frames(t0, dur_s, pcs, low=None):
    """``dur_s`` of one held sound: ``pcs`` {pitch class: weight}, ``low`` the bass
    band's share by pitch class (a direct pickup reads only a few hundredths)."""
    c = np.zeros(12)
    for pc, w in pcs.items():
        c[pc % 12] += w
    c /= c.sum()
    lo = np.zeros(12)
    for pc, w in (low or {}).items():
        lo[pc % 12] = w
    n = int(round(dur_s / HOP_S))
    return [SimpleNamespace(t_s=t0 + (k + 1) * HOP_S, rms_db=-20.0, chroma=c,
                            chroma_low=lo, centroid_hz=600.0) for k in range(n)]


def chord(root, intervals, w=1.0):
    return {(root + i) % 12: w for i in intervals}


def play(events, secs):
    """Loop ``events`` [(seconds, pcs, low)] for ``secs``; the tracker's last state."""
    tr, t, i, state = HarmonyTracker(), 0.0, 0, None
    while t < secs:
        dur, pcs, low = events[i % len(events)]
        state = tr.process(frames(t, dur, pcs, low))
        t += dur
        i += 1
    return state


def low_string(root):
    """A low open string left ringing: its root, and the partials a string adds,
    the fifth (3rd harmonic) and a MAJOR third (5th harmonic)."""
    return {root: 1.0, root + 7: 0.35, root + 4: 0.2}


# ---- mode: a drone under a minor melody

def test_a_ringing_low_tonic_under_a_minor_melody_is_minor():
    """The tonic string rings under every note, so the tonic is about half of all
    the pitch heard and the fit to E major and E minor tie; the string's own
    overtone even adds a G sharp.  Her melody's G, C and D say minor."""
    melody = [G, F + 1, E, D, B, A, G, B, C, B, G, E]
    events = []
    for note in melody:
        pcs = low_string(E)
        pcs[note] = pcs.get(note, 0.0) + 0.8
        events.append((0.25, pcs, {E: 0.02}))
    state = play(events, 20.0)
    assert parse_key(state.key) == parse_key("E minor"), state.key


def test_a_ringing_low_tonic_under_a_major_melody_stays_major():
    """The same drone under the major melody: the mode rule must not make every
    drone minor."""
    melody = [G + 1, F + 1, E, D + 1, B, A, G + 1, B, C + 1, B, G + 1, E]
    events = []
    for note in melody:
        pcs = low_string(E)
        pcs[note] = pcs.get(note, 0.0) + 0.8
        events.append((0.25, pcs, {E: 0.02}))
    state = play(events, 20.0)
    assert parse_key(state.key) == parse_key("E major"), state.key


# ---- a maj7 home chord holds the minor triad on its third

def test_a_held_maj7_home_chord_is_its_root_not_the_minor_on_its_third():
    """C E G B holds E minor's triad.  With the root only faintly in the bass, as
    a pickup hears a strummed guitar, it was read as E minor."""
    state = play([(BAR_S, chord(C, MAJ7), {C: 0.03})], 20.0)
    assert parse_key(state.key) == parse_key("C major"), state.key


def test_a_maj7_home_chord_progression_is_its_root_not_the_minor_on_its_third():
    events = [(BAR_S, chord(r, iv), {r: 0.03})
              for r, iv in ((C, MAJ7), (C, MAJ7), (F, MAJ7), (G, DOM7))]
    state = play(events, 24.0)
    assert parse_key(state.key) == parse_key("C major"), state.key


def test_a_maj9_turnaround_is_its_home_key_not_a_fifth_away():
    events = [(BAR_S, chord(r, iv), {r: 0.03})
              for r, iv in ((C, MAJ9), (A, MIN9), (D, MIN9), (G, DOM7))]
    state = play(events, 24.0)
    assert parse_key(state.key) == parse_key("C major"), state.key


# ---- relative keys: the band comes in on what the evidence says now

def _triad(root, minor=False):
    return {root: 1.0, (root + (3 if minor else 4)) % 12: 1.0, (root + 7) % 12: 1.0}


def _take(bar_chord, bars, *, adopt):
    """A detect-mode take, a bar at a time, as the live loop runs it: the tracker
    hears each bar, the entry reads it, and (``adopt``) the tracker is handed the
    key the band came in on."""
    tr, entry, keys = HarmonyTracker(), BandEntry("midwest", key_fallback_bars=16), []
    for bar in range(bars):
        root, minor = bar_chord(bar)
        h = tr.process(frames(bar * BAR_S, BAR_S, _triad(root, minor), {root: 0.03}))
        e = entry.update(bar, h, True)
        if adopt and entry.band_is_in and entry.bars_in == 0 and e.key != h.key:
            tr.adopt(e.key)
        keys.append(e.key)
    return [k for k in keys if k is not None]


A_MINOR_LOOP = [(A, True), (D, True), (A, True), (E, False)]


def _opens_on_the_relative_major(bar):
    return (C, False) if bar < 2 else A_MINOR_LOOP[(bar - 2) % 4]


def test_a_minor_loop_opened_on_its_relative_major_brings_the_band_in_minor():
    """Two bars of C, then i iv i V in A minor.  The tracker's belief formed on
    the opening C, and C major is A minor's relative, which its hysteresis keeps
    from replacing a belief: the band came in in C major and stayed there."""
    keys = _take(_opens_on_the_relative_major, 20, adopt=True)
    assert keys, "the band never came in"
    assert parse_key(keys[0]) == parse_key("A minor"), keys[0]


def test_the_key_the_band_came_in_on_is_not_released_back_to_the_stale_belief():
    """Without the tracker adopting the band's key, its stale belief disagreed
    with the band for the governor's release bars and pulled the band back."""
    keys = _take(_opens_on_the_relative_major, 24, adopt=True)
    assert {parse_key(k) for k in keys} == {parse_key("A minor")}, sorted(set(keys))


def test_a_major_loop_opened_on_its_relative_minor_brings_the_band_in_major():
    loop = [(C, False), (F, False), (G, False), (C, False)]
    keys = _take(lambda bar: (A, True) if bar < 2 else loop[(bar - 2) % 4], 20,
                 adopt=True)
    assert keys and parse_key(keys[0]) == parse_key("C major"), keys[:1]


def test_a_steady_key_is_unchanged_by_the_entry_check():
    keys = _take(lambda bar: A_MINOR_LOOP[bar % 4], 20, adopt=True)
    assert keys and {parse_key(k) for k in keys} == {parse_key("A minor")}


# ---- the entry check, alone

def _heard(key, best, lead):
    state = HarmonyTracker().process([])
    return state.__class__(**{**state.__dict__, "key": key, "key_conf": 0.8,
                              "key_best": best, "key_lead": lead})


def _entry_key(best, lead):
    entry = BandEntry("midwest", min_drum_bars=4, after_first_heard_bars=4)
    key = None
    for bar in range(12):
        key = entry.update(bar, _heard("C major", best if bar >= 3 else "C major",
                                       lead), True).key or key
    return key


def test_the_band_enters_on_a_clearly_leading_present_key():
    assert _entry_key("A minor", 0.2) == "A minor"


def test_a_tie_in_the_present_evidence_leaves_the_band_on_the_belief():
    assert _entry_key("A minor", 0.0) == "C major"


def test_adopt_sets_the_belief_and_keeps_its_hysteresis():
    tr = HarmonyTracker()
    c_major = [(BAR_S, _triad(r, m), {r: 0.03})
               for r, m in ((C, False), (F, False), (G, False), (C, False))]
    t = 0.0
    for _ in range(3):
        for dur, pcs, low in c_major:
            tr.process(frames(t, dur, pcs, low))
            t += dur
    tr.adopt("A minor")
    state = tr.process(frames(t, 1.0, _triad(C), {C: 0.03}))
    # One more second of C is no reason to leave the relative the band is on.
    assert parse_key(state.key) == parse_key("A minor"), state.key
