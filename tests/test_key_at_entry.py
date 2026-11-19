"""The key the band comes in on: mode, maj7 home chords, relative keys, the entry check.

Synthetic frames only (duck-typed, as listen/harmony.py reads them): no audio, no
device, except the one live-loop test at the end.  Each scenario is one the band
used to come in wrong on, built from the pitch content that made it wrong.  The
takes run as the live loop runs them: the tracker hears each bar, the band entry
reads it, and the tracker is handed the key the band came in on.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from plume.bandentry import ENTRY_BEST_LEAD, BandEntry
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


def _triad(root, minor=False):
    return {root: 1.0, (root + (3 if minor else 4)) % 12: 1.0, (root + 7) % 12: 1.0}


def take(events, bars, bar_s=BAR_S):
    """A detect-mode take of ``events`` [(seconds, pcs, low)] looped for ``bars``
    bars: every key the band played, from the bar it came in."""
    seq, t, i = [], 0.0, 0
    while t < bars * bar_s:
        dur, pcs, low = events[i % len(events)]
        seq += frames(t, dur, pcs, low)
        t += dur
        i += 1
    tr, entry, keys, k = HarmonyTracker(), BandEntry("midwest", key_fallback_bars=16), [], 0
    for bar in range(bars):
        batch = []
        while k < len(seq) and seq[k].t_s <= (bar + 1) * bar_s:
            batch.append(seq[k])
            k += 1
        h = tr.process(batch)
        e = entry.update(bar, h, True)
        if entry.band_is_in and entry.bars_in == 0 and e.key != h.key:
            tr.adopt(e.key)
        keys.append(e.key)
    return [key for key in keys if key is not None]


def came_in_on(events, bars=20, bar_s=BAR_S):
    keys = take(events, bars, bar_s)
    assert keys, "the band never came in"
    return parse_key(keys[0])


def bar_chords(chords):
    """One bar per (root, minor) triad, the root faintly in the bass."""
    return [(BAR_S, _triad(r, m), {r: 0.03}) for r, m in chords]


# ---- mode: a drone under a minor melody

def low_string(root):
    """A low open string left ringing: its root, and the partials a string adds,
    the fifth (3rd harmonic) and a MAJOR third (5th harmonic)."""
    return {root: 1.0, root + 7: 0.35, root + 4: 0.2}


def _drone_melody(melody):
    events = []
    for note in melody:
        pcs = low_string(E)
        pcs[note] = pcs.get(note, 0.0) + 0.8
        events.append((0.25, pcs, {E: 0.02}))
    return events


def test_a_ringing_low_tonic_under_a_minor_melody_brings_the_band_in_minor():
    """The tonic string rings under every note, so the tonic is about half of all
    the pitch heard, and the string's own overtone adds a G sharp.  Her melody's
    G, C and D say minor."""
    melody = [G, F + 1, E, D, B, A, G, B, C, B, G, E]
    assert came_in_on(_drone_melody(melody)) == parse_key("E minor")


def test_a_ringing_low_tonic_under_a_major_melody_brings_the_band_in_major():
    melody = [G + 1, F + 1, E, D + 1, B, A, G + 1, B, C + 1, B, G + 1, E]
    assert came_in_on(_drone_melody(melody)) == parse_key("E major")


# ---- a maj7 home chord holds the minor triad on its third

def test_a_held_maj7_home_chord_is_its_root_not_the_minor_on_its_third():
    """C E G B holds E minor's triad.  With the root only faintly in the bass, as
    a pickup hears a strummed guitar, the band came in on E minor."""
    assert came_in_on([(BAR_S, chord(C, MAJ7), {C: 0.03})]) == parse_key("C major")


def test_a_maj7_home_chord_progression_is_its_root_not_the_minor_on_its_third():
    events = [(BAR_S, chord(r, iv), {r: 0.03})
              for r, iv in ((C, MAJ7), (C, MAJ7), (F, MAJ7), (G, DOM7))]
    assert came_in_on(events) == parse_key("C major")


def _sounded(midi_notes, decay=0.4):
    """The chroma of notes as a string sounds them: partials 1-8 at decay ** (n - 1)."""
    c = {}
    for m in midi_notes:
        for n in range(1, 9):
            pc = (m + int(round(12 * np.log2(n)))) % 12
            c[pc] = c.get(pc, 0.0) + decay ** (n - 1)
    return c


def test_a_maj7_vamp_on_the_home_chord_and_its_IV_comes_in_on_the_home_chord():
    """Imaj7 IVmaj7, as strings sound them, with the IV's root the lowest.  The I's
    maj7 holds the minor triad on its third, and the loop holds the IV's V I: the
    band came in on the iii, and matched against corpus profiles alone, on the IV."""
    for tonic in (C, A - 1, B - 1):
        r = 48 + (tonic - 48) % 12
        loop = [(BAR_S, _sounded([r + i for i in MAJ7], 0.6), {tonic: 0.03}),
                (BAR_S, _sounded([r - 7 + i for i in MAJ7], 0.6), {(tonic + 5) % 12: 0.06})]
        want = parse_key(f"{('C', 'Ab', 'Bb')[(C, A - 1, B - 1).index(tonic)]} major")
        assert came_in_on(loop, bars=24) == want, tonic


def test_a_I_IV_vamp_with_the_IV_lowest_comes_in_on_the_I():
    """C F C F with F the stronger bass note holds F major's V I as well: corpus
    profiles weigh a major key's tonic and fifth nearly alike."""
    for tonic, name in ((C, "C"), (E, "E"), (A - 1, "Ab")):
        loop = [(BAR_S, _triad(tonic), {tonic: 0.03}),
                (BAR_S, _triad((tonic + 5) % 12), {(tonic + 5) % 12: 0.15})]
        assert came_in_on(loop, bars=24) == parse_key(f"{name} major"), name


# ---- relative keys: the band comes in on everything she has played

A_MINOR_LOOP = [(A, True), (D, True), (A, True), (E, False)]


def test_a_minor_loop_opened_on_its_relative_major_brings_the_band_in_minor():
    """Two bars of C, then i iv i V in A minor.  The tracker's belief formed on
    the opening C, and C major is A minor's relative, which its hysteresis keeps
    from replacing a belief: the band came in in C major and stayed there."""
    loop = bar_chords([(C, False)] * 2 + A_MINOR_LOOP * 6)
    assert came_in_on(loop[:2] + loop[2:], bars=20) == parse_key("A minor")


def test_the_key_the_band_came_in_on_is_not_released_back_to_the_stale_belief():
    """Without the tracker adopting the band's key, its stale belief disagreed
    with the band for the governor's release bars and pulled the band back."""
    keys = take(bar_chords([(C, False)] * 2 + A_MINOR_LOOP * 6), 24)
    assert {parse_key(k) for k in keys} == {parse_key("A minor")}, sorted(set(keys))


def test_a_major_loop_opened_on_its_relative_minor_brings_the_band_in_major():
    loop = bar_chords([(A, True)] * 2 + [(C, False), (F, False), (G, False), (C, False)] * 6)
    assert came_in_on(loop) == parse_key("C major")


def test_a_steady_key_is_unchanged_by_the_entry_check():
    keys = take(bar_chords(A_MINOR_LOOP), 20)
    assert keys and {parse_key(k) for k in keys} == {parse_key("A minor")}


def test_a_minor_loop_on_its_relative_majors_chords_keeps_its_minor_key():
    """i bVI bIII bVII (Am F C G) holds C major's chords more often than A minor's."""
    keys = take(bar_chords([(A, True), (F, False), (C, False), (G, False)]), 20)
    assert keys and {parse_key(k) for k in keys} == {parse_key("A minor")}, sorted(set(keys))


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


def test_the_band_enters_on_an_entry_reading_that_leads_the_belief():
    assert _entry_key("A minor", 1.0) == "A minor"
    assert _entry_key("A minor", ENTRY_BEST_LEAD) == "A minor"


def test_a_near_tie_leaves_the_band_on_the_belief():
    assert _entry_key("A minor", 0.0) == "C major"
    assert _entry_key("A minor", 0.9 * ENTRY_BEST_LEAD) == "C major"


def test_the_entry_reading_reports_no_lead_over_the_belief_it_agrees_with():
    tr, t = HarmonyTracker(), 0.0
    for _ in range(6):
        for dur, pcs, low in bar_chords([(C, False), (F, False), (G, False), (C, False)]):
            state = tr.process(frames(t, dur, pcs, low))
            t += dur
    assert state.key_best == state.key and state.key_lead == 0.0


def test_the_entry_reading_leaves_the_belief_alone():
    """The belief (``key``) is the tracker's hysteretic one, as it was before the
    entry reading existed: the reading only reports."""
    tr, t = HarmonyTracker(), 0.0
    for dur, pcs, low in bar_chords([(C, False)] * 2 + A_MINOR_LOOP * 3):
        state = tr.process(frames(t, dur, pcs, low))
        t += dur
    assert parse_key(state.key) == parse_key("C major")
    assert parse_key(state.key_best) == parse_key("A minor") and state.key_lead > 0.0


def test_adopt_sets_the_belief_and_keeps_its_hysteresis():
    tr, t = HarmonyTracker(), 0.0
    for _ in range(3):
        for dur, pcs, low in bar_chords([(C, False), (F, False), (G, False), (C, False)]):
            tr.process(frames(t, dur, pcs, low))
            t += dur
    tr.adopt("A minor")
    state = tr.process(frames(t, 1.0, _triad(C), {C: 0.03}))
    # One more second of C is no reason to leave the relative the band is on.
    assert parse_key(state.key) == parse_key("A minor"), state.key


# ---- the live loop hands the band's key to the tracker

def test_the_live_loop_hands_the_key_the_band_came_in_on_to_the_tracker(
        tmp_path, monkeypatch, plume_registry):
    """realtime.lead: the planner sets the key the band came in on, the analysis
    thread hands it to its tracker (HarmonyTracker.adopt).  The entry here is made to
    come in on a key the tracker never believed, so the hand-off must happen."""
    import dataclasses
    import threading

    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take as strummed_take

    from plume import bandentry, realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    forced = "F# major"
    real_update = bandentry.BandEntry.update

    def update(self, bar, harmony, playing, **kw):
        out = real_update(self, bar, harmony, playing, **kw)
        if self.band_is_in and self.bars_in == 0:
            self._gov.commit(forced, bar)
            out = dataclasses.replace(out, key=forced)
        return out

    adopted, handed = [], threading.Event()
    real_adopt = HarmonyTracker.adopt

    def adopt(self, key):
        adopted.append(key)
        handed.set()
        return real_adopt(self, key)

    monkeypatch.setattr(bandentry.BandEntry, "update", update)
    monkeypatch.setattr(HarmonyTracker, "adopt", adopt)
    sock = short_socket_path("adopt")

    def stop_once_handed():
        if handed.wait(60.0):
            for _ in range(40):
                try:
                    realtime.send_stop(socket_path=sock)
                    return
                except (OSError, realtime.RealtimeError):
                    threading.Event().wait(0.05)

    threading.Thread(target=stop_once_handed, daemon=True).start()
    human = strummed_take(seconds=60.0, room_s=30.0)
    with patched_module("sounddevice", fake_audio(human, rate_x=12.0)):
        realtime.lead(prompt=None, detect=True, seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"), socket_path=sock,
                      status=False, max_minutes=0.6)
    assert adopted == [forced]
