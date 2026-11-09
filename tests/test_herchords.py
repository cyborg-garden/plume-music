"""Her chords (roadmap step 8): the band follows the chord she means, only when sure.

Every fixture here is synthetic: chroma vectors built from pitch classes, and a few
bars of plucked tones rendered in the test.  Nothing reads a recorded take.
"""
from __future__ import annotations

import numpy as np
import pytest

from plume import herchords as hc
from plume.listen.features import FeatureExtractor

SR = 48000
KEY, STYLE = "C major", "lofi"
TEMPLATE = ("Dm7", "G7", "Cmaj7", "Cmaj7")      # lofi's top progression in C


def pcs_chroma(pcs, *, weights=None, noise=0.0, seed=0):
    c = np.zeros(12)
    for i, p in enumerate(pcs):
        c[p % 12] += 1.0 if weights is None else weights[i]
    if noise:
        c += np.random.default_rng(seed).uniform(0, noise, 12)
    return c / c.sum()


C_, DM, EM, F_, G_, AM = (0, 4, 7), (2, 5, 9), (4, 7, 11), (5, 9, 0), (7, 11, 2), (9, 0, 4)
SCALE = (0, 2, 4, 5, 7, 9, 11)


def follow(bars, *, lag=2, start=0, key=KEY, style=STYLE):
    """Feed her bars (chroma, or None for a rest) through the live timing: bar N is
    planned after hearing bar N-lag.  Returns the decision per planned bar."""
    f = hc.ChordFollower()
    out = {}
    for i, chroma in enumerate(bars):
        heard = start + i
        f.hear(heard, chroma, playing=chroma is not None)
        bar = heard + lag
        out[bar] = f.decide(bar, key, style, TEMPLATE[bar % 4])
    return out


# ------------------------------------------------------------- reading one bar

def test_candidates_use_the_styles_own_spelling():
    syms = [c.sym for c in hc.candidates(KEY, STYLE)]
    assert syms[0] == "Cmaj7" and syms[1] == "Dm7" and syms[4] == "G7" and syms[5] == "Am7"
    assert [c.sym for c in hc.candidates("A minor", "synthwave")][0] == "Am"


@pytest.mark.parametrize("pcs, sym", [(C_, "Cmaj7"), (AM, "Am7"), (F_, "Fmaj7"),
                                      (G_, "G7"), (DM, "Dm7"), (EM, "Em7")])
def test_a_clear_triad_names_its_chord(pcs, sym):
    r = hc.read_bar(pcs_chroma(pcs, noise=0.05), hc.candidates(KEY, STYLE))
    assert r.sym == sym and r.fit >= hc.FIT_MIN


def test_a_scale_melody_names_no_chord():
    r = hc.read_bar(pcs_chroma(SCALE), hc.candidates(KEY, STYLE))
    assert r.sym is None and r.fit < hc.FIT_MIN


def test_a_bar_between_two_chords_names_neither():
    """C and E with A and G equally: C major and A minor both fit; not sure which."""
    r = hc.read_bar(pcs_chroma((0, 4, 7, 9), weights=(1, 1, 0.6, 0.6)),
                    hc.candidates(KEY, STYLE))
    assert r.sym is None


def test_a_chord_outside_the_key_names_none():
    r = hc.read_bar(pcs_chroma((1, 5, 8)), hc.candidates(KEY, STYLE))   # Db major
    assert r.sym is None


def test_silence_reads_nothing():
    assert hc.read_bar(np.zeros(12), hc.candidates(KEY, STYLE)) is None


# ------------------------------------------------------------- hold and loops

def test_she_holds_one_chord_and_the_band_stays_on_it():
    d = follow([pcs_chroma(AM, noise=0.05, seed=i) for i in range(10)])
    held = [d[b] for b in sorted(d) if b >= 4]
    assert all(x["source"] == "her_hold" for x in held)
    assert {x["chord"] for x in held} == {"Am7", "Am9"}        # gentle colour only
    assert "following her Am7" in held[0]["reason"]


def test_the_held_chord_moves_gently_every_other_two_bars():
    d = follow([pcs_chroma(AM, seed=i) for i in range(12)])
    seq = [d[b]["chord"] for b in range(4, 12)]
    assert seq == ["Am7", "Am7", "Am9", "Am9", "Am7", "Am7", "Am9", "Am9"]


def test_one_bar_of_hers_is_not_a_hold():
    d = follow([pcs_chroma(AM)])
    assert d[2]["source"] == "template" and d[2]["chord"] == TEMPLATE[2]


@pytest.mark.parametrize("loop", [(C_, AM), (C_, AM, F_, G_), (F_, F_, C_, C_)])
def test_a_repeating_loop_is_predicted_on_time(loop):
    """Each band bar plays the chord she plays in that same bar, although the band
    only heard her two bars earlier."""
    names = {C_: "Cmaj7", AM: "Am7", F_: "Fmaj7", G_: "G7"}
    bars = [pcs_chroma(loop[i % len(loop)], noise=0.04, seed=i) for i in range(20)]
    d = follow(bars)
    p = len(loop)
    settled = [b for b in sorted(d) if b >= 2 * p + 2 and b < 20]
    assert settled
    for b in settled:
        assert d[b]["source"] == "her_loop", d[b]
        assert d[b]["chord"] == names[loop[b % p]], (b, d[b])
        assert d[b]["period"] == p


def test_a_loop_of_two_bars_per_chord_is_one_loop_not_a_hold():
    bars = [pcs_chroma((C_, AM, F_, G_)[(i // 2) % 4]) for i in range(24)]
    d = follow(bars)
    names = ("Cmaj7", "Am7", "Fmaj7", "G7")
    for b in range(18, 24):
        assert d[b]["source"] == "her_loop" and d[b]["period"] == 8
        assert d[b]["chord"] == names[(b // 2) % 4]


def test_a_rest_keeps_her_loop_going():
    bars = [pcs_chroma((C_, AM)[i % 2]) for i in range(8)] + [None, None]
    d = follow(bars)
    assert d[10]["source"] == "her_loop" and d[10]["chord"] == "Cmaj7"
    assert d[11]["source"] == "her_loop" and d[11]["chord"] == "Am7"


def test_single_note_melody_keeps_the_template():
    rng = np.random.default_rng(4)
    bars = [pcs_chroma(rng.choice(SCALE, 5, replace=False)) for _ in range(12)]
    d = follow(bars)
    assert all(x["source"] == "template" and x["chord"] == TEMPLATE[b % 4]
               for b, x in d.items())


def test_when_she_changes_chord_the_band_does_not_keep_the_old_one():
    bars = [pcs_chroma(AM)] * 6 + [pcs_chroma(F_)] * 4
    d = follow(bars)
    # Heard bar 6 is the first F; the band's bar 8 can no longer hold Am.
    assert d[8]["chord"] not in ("Am7", "Am9")
    assert d[8]["source"] == "template" and "reads Fmaj7" in d[8]["reason"]
    assert d[9]["source"] == "her_hold" and d[9]["chord"].startswith("Fmaj")


def test_two_unsure_bars_hand_back_to_the_template():
    bars = [pcs_chroma(AM)] * 6 + [pcs_chroma(SCALE)] * 2
    d = follow(bars)
    assert d[8]["source"] == "her_hold"           # one fill bar is not a change
    assert d[9]["source"] == "template" and "name no chord" in d[9]["reason"]


def test_nothing_heard_for_long_hands_back_to_the_template():
    d = follow([pcs_chroma(AM)] * 4 + [None] * hc.STALE_BARS)
    last = d[max(d)]
    assert last["source"] == "template" and "nothing of hers since" in last["reason"]


def test_a_loop_heard_before_the_key_is_known_is_followed_at_once():
    f = hc.ChordFollower()
    for b in range(8):
        f.hear(b, pcs_chroma((C_, AM)[b % 2]), playing=True)
    assert f.decide(10, None, STYLE, None)["source"] == "template"
    d = f.decide(10, KEY, STYLE, "Dm7")
    assert d["source"] == "her_loop" and d["chord"] == "Cmaj7"


def test_every_decision_is_a_plain_logged_dict():
    import json
    d = follow([pcs_chroma(AM)] * 4)
    for x in d.values():
        json.dumps(x)
        assert set(x) >= {"chord", "source", "reason", "period", "template", "heard_bar",
                          "her_chord", "her_fit"}
        assert x["source"] in ("template", "her_hold", "her_loop")


# ------------------------------------------------------------- room mic bleed

def test_clean_bar_is_her_chroma_and_bass():
    frames = [(pcs_chroma(AM), pcs_chroma([9]))] * 5 + [(pcs_chroma(AM), None)] * 5
    h = hc.clean_bar(frames)
    assert np.allclose(h.chroma, pcs_chroma(AM)) and np.allclose(h.bass, pcs_chroma([9]))
    assert h.notes == pytest.approx(3.0)
    assert hc.clean_bar([(pcs_chroma(AM),)] * 3).bass is None


def test_a_bar_with_no_pitch_is_no_evidence_and_says_so():
    """Heard by level but nothing pitched in it (a knock): no chord, and the log
    does not blame the band, which a direct input cannot hear."""
    assert hc.clean_bar([(np.zeros(12), None)] * 4).chroma is None
    f = hc.ChordFollower()
    f.hear(0, hc.clean_bar([(np.zeros(12), None)] * 4), playing=True)
    d = f.decide(2, KEY, STYLE, TEMPLATE[2])
    assert d["source"] == "template" and "pitch" in d["reason"]


# ------------------------------------------------------------- through the ear

def _pluck(midis, seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    for m in midis:
        f0 = 440.0 * 2 ** ((m - 69) / 12)
        for h in range(1, 6):
            y += np.sin(2 * np.pi * f0 * h * t) / h ** 1.5
    return y * np.exp(-t * 1.2)


VOICINGS = {"C": (48, 55, 60, 64, 67), "Am": (45, 52, 57, 60, 64),
            "F": (41, 48, 53, 57, 60), "G": (43, 50, 55, 59, 62)}


def _bars_of(audio, bar_s, sr=SR):
    fe = FeatureExtractor(sr=sr)
    frames = []
    for i in range(0, audio.size, 2048):
        frames.extend(fe.process(audio[i:i + 2048].astype(np.float32)))
    n = int(audio.size / sr / bar_s)
    bars = [[] for _ in range(n)]
    for fr in frames:
        b = int(fr.t_s // bar_s)
        if b < n:
            bars[b].append((fr.chroma, fr.chroma_low))
    return bars


@pytest.mark.parametrize("loop", [("C", "Am", "F", "G"), ("Am",), ("F", "C")])
def test_a_rendered_chord_loop_is_followed_bar_for_bar(loop):
    bar_s = 2.0
    audio = np.concatenate([0.05 * _pluck(VOICINGS[loop[b % len(loop)]], bar_s)
                            for b in range(16)])
    bars = _bars_of(audio, bar_s)
    names = {"C": "Cmaj7", "Am": "Am7", "F": "Fmaj7", "G": "G7"}
    f = hc.ChordFollower()
    hits = 0
    for heard, frames in enumerate(bars):
        f.hear(heard, hc.clean_bar(frames), playing=True)
        bar = heard + 2
        if bar >= 16 or heard < 2 * len(loop):
            continue
        d = f.decide(bar, KEY, STYLE, TEMPLATE[bar % 4])
        assert d["source"] != "template", (bar, d)
        hits += d["chord"].startswith(names[loop[bar % len(loop)]][:-1])
    assert hits == len([b for b in range(16) if 2 * len(loop) <= b - 2 < 14])


def test_on_a_room_mic_the_band_never_takes_a_chord_for_hers():
    """The masker's band share is a bound, not a measurement, so a room mic can't
    tell her held Am from the band's own: the template plays, and says why."""
    f = hc.ChordFollower(band_in_mic=True)
    for heard in range(12):
        f.hear(heard, hc.clean_bar([(pcs_chroma(AM),)] * 10), playing=True)
        d = f.decide(heard + 2, KEY, STYLE, TEMPLATE[(heard + 2) % 4])
        assert d["source"] == "template" and d["chord"] == TEMPLATE[(heard + 2) % 4]
        assert "room mic" in d["reason"]


@pytest.mark.parametrize("midi,harmonics,ring", [
    (64, 12, 1.0), (64, 12, 4.0), (45, 12, 1.0), (45, 20, 2.0), (48, 6, 1.0), (55, 6, 1.0),
    (64, 5, 1.0)])
def test_one_bright_note_held_is_not_a_chord(midi, harmonics, ring):
    """A lone note carries its own fifth and third in its overtones: a bright E
    read as E minor at fit 0.9 with no G in it.  Held or repeated, it is still
    one note, and the band keeps the template."""
    bar_s = 2.0
    t = np.arange(int(bar_s / 4 * ring * SR)) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    note = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, harmonics + 1)
               if f0 * h < SR / 2) * np.exp(-t * 1.2)
    audio = np.zeros(int(12 * bar_s * SR) + note.size)
    for i in range(48):
        s = int(i * bar_s / 4 * SR)
        audio[s:s + note.size] += 0.05 * note
    bars = _bars_of(audio[:int(12 * bar_s * SR)], bar_s)
    f = hc.ChordFollower()
    for heard, frames in enumerate(bars):
        f.hear(heard, hc.clean_bar(frames), playing=True)
        d = f.decide(heard + 2, KEY, STYLE, TEMPLATE[(heard + 2) % 4])
        assert d["source"] == "template", (heard, d)


def test_a_chord_missing_its_third_names_no_chord():
    """Root and fifth loud, the third a trace: a drone or a power chord, not a chord
    whose quality was heard."""
    c = pcs_chroma((9, 4, 0), weights=(1.0, 0.6, 0.1))
    r = hc.read_bar(c, hc.candidates(KEY, STYLE))
    assert r.sym is None and r.best == "Am7" and "barely sounds" in r.why


def test_a_quiet_minor_third_still_names_the_chord_and_a_major_one_does_not():
    """Six strings double the root and fifth and leave one third, a fifth of the
    loudest.  No overtone of the other chord tones lands on a minor third, so that is
    a real string; a major third at the same level could be the root's own fifth
    harmonic, and stays unsure."""
    cands = hc.candidates(KEY, STYLE)
    minor = hc.read_bar(pcs_chroma((4, 11, 7), weights=(1.0, 0.55, 0.2)), cands)
    assert minor.sym == "Em7", minor
    major = hc.read_bar(pcs_chroma((0, 7, 4), weights=(1.0, 0.55, 0.2)), cands)
    assert major.sym is None and major.best == "Cmaj7" and "barely sounds" in major.why


OPEN_STRINGS = (40, 45, 50, 55, 59, 64)


def _frets(shape):
    """A guitar chord shape, low string first ("022000"; x: string not played)."""
    return tuple(o + int(f) for o, f in zip(OPEN_STRINGS, shape) if f != "x")


def _held(midis, key, *, bars=8, harmonics=5, rolloff=1.5):
    """She strikes one voicing once a bar and lets it ring; the decisions for the
    bars planned once two of hers are heard."""
    bar_s = 2.0
    t = np.arange(int(bar_s * SR)) / SR
    one = sum(np.sin(2 * np.pi * 440.0 * 2 ** ((m - 69) / 12) * h * t) / h ** rolloff
              for m in midis for h in range(1, harmonics + 1)) * np.exp(-t * 1.2)
    f = hc.ChordFollower()
    out = []
    for heard, frames in enumerate(_bars_of(np.tile(0.05 * one, bars), bar_s)):
        f.hear(heard, hc.clean_bar(frames), playing=True)
        if heard >= 1:
            out.append(f.decide(heard + 2, key, STYLE, TEMPLATE[(heard + 2) % 4]))
    return out


@pytest.mark.parametrize("shape, key, sym", [
    ("022000", KEY, "Em"), ("022000", "E minor", "Em"), ("244222", "F# minor", "F#m"),
    ("799777", "D major", "Bm"), ("x02210", KEY, "Am")])
@pytest.mark.parametrize("harmonics, rolloff", [(5, 1.5), (3, 2.0), (12, 1.0)])
def test_a_full_minor_chord_with_one_third_among_six_strings_is_followed(
        shape, key, sym, harmonics, rolloff):
    """Open Em and the barre minors have one third among five or six strings, and
    measured 0.20-0.28 of the loudest tone: under the quarter a major third needs."""
    got = _held(_frets(shape), key, harmonics=harmonics, rolloff=rolloff)
    assert got and all(d["source"] == "her_hold" for d in got), got[0]
    assert all(d["chord"].startswith(sym) for d in got), got[0]


@pytest.mark.parametrize("shape, key", [("022100", "E major"), ("320033", "G major")])
def test_a_full_major_chord_with_one_quiet_third_keeps_the_template_and_says_why(
        shape, key):
    """A known limit, kept on the safe side: with one major third among six strings
    (0.16-0.24 of the loudest on a plain tone) the bar looks like a bright single
    note, whose fifth harmonic is that same third.  The band keeps the template
    rather than claim a chord it can't tell from one note."""
    got = _held(_frets(shape), key)
    assert got and all(d["source"] == "template" for d in got), got[0]
    assert all("barely sounds" in d["reason"] for d in got[1:]), got[1]


@pytest.mark.parametrize("root", [40, 45, 47, 52])
@pytest.mark.parametrize("key", [KEY, "E minor", "A minor", "D major"])
def test_a_bright_power_chord_is_not_a_minor_chord(root, key):
    """Root, fifth and octave with no third: the minor chord on that root has no
    third in it at all, however bright the tone."""
    got = _held((root, root + 7, root + 12), key, harmonics=20, rolloff=0.7)
    assert got
    for d in got:
        assert d["source"] == "template" or "m" not in d["chord"][1:3], d


def test_a_minor_third_that_another_note_overtones_is_not_her_string():
    """F# with the D above it, in C# minor: D's third harmonic is an A, and F# A C#
    then looks like F# minor with a quiet third.  The A is D's, not a string."""
    cands = hc.candidates("C# minor", STYLE)
    bar = pcs_chroma((6, 2, 1, 9), weights=(1.0, 0.65, 0.25, 0.2))
    r = hc.read_bar(bar, cands)
    assert r.sym is None and r.best == "F#m7" and "barely sounds" in r.why, r
    # the same third with no D under it is a string of its own
    assert hc.read_bar(pcs_chroma((6, 1, 9), weights=(1.0, 0.55, 0.2)), cands).sym == "F#m7"
    got = _held((54, 62), "C# minor", harmonics=20, rolloff=1.0)
    assert got and all(d["source"] == "template" for d in got), got[-1]


def test_a_drone_with_the_minor_third_on_one_beat_in_four_is_still_a_drone():
    """Root and fifth ringing, a melody over them that touches the third once a bar:
    the third is a trace (0.06-0.12 of the root), and the band keeps the template."""
    bar_s, n = 2.0, 8
    audio = np.zeros(int((n + 1) * bar_s * SR))
    for b in range(n):
        for half in range(2):
            s = int((b + half / 2) * bar_s * SR)
            y = 0.05 * _pluck((40, 47, 52), 1.6)
            audio[s:s + y.size] += y
        for q, note in enumerate((55, 57, 59, 62)):          # G once, then A B D
            s = int((b + q / 4) * bar_s * SR)
            y = 0.05 * _pluck((note,), 0.5)
            audio[s:s + y.size] += y
    f = hc.ChordFollower()
    for heard, frames in enumerate(_bars_of(audio[:int(n * bar_s * SR)], bar_s)):
        f.hear(heard, hc.clean_bar(frames), playing=True)
        d = f.decide(heard + 2, "E minor", STYLE, "Am7")
        assert d["source"] == "template", (heard, d)


# ------------------------------------------------------------- the live loop

def _strums(midis, *, bpm=75.0, every_beats=2, seconds=135.0, level=0.08, notes=None):
    """After the count (tests/test_bandentry._take's four knocks at 2 s), a guitar:
    one chord strummed every ``every_beats`` beats, or, with ``notes``, one note of
    the list at a time."""
    from test_bandentry import _take
    x = _take(seconds=8.0, room_s=seconds - 8.0)
    period = 60.0 / bpm
    at, i = 2.0 + 12 * period, 0
    while at + 2.0 < seconds:
        tones = (notes[i % len(notes)],) if notes else midis
        y = (level * _pluck(tones, 1.6)).astype(np.float32)
        s = int(at * SR)
        x[s:s + y.size] += y[:max(0, x.size - s)]
        at += every_beats * period
        i += 1
    return x


def _live(tmp_path, human, name, **kw):
    from conftest import fake_audio, patched_module, short_socket_path

    from plume import realtime
    from plume import roles as roles_mod
    fake = fake_audio(human, devices=[{"name": "Audio Interface 4x4",
                                       "max_input_channels": 4,
                                       "max_output_channels": 4}],
                      rate_x=12.0, **kw.pop("fake", {}))
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi in C major", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / name), status=False,
                      socket_path=short_socket_path(name), max_minutes=0.5,
                      input_device="audio interface", input_channel=1, **kw)
    import json
    rows = [json.loads(line) for line in (tmp_path / name / "decisions.ndjson").open()]
    return [r for r in rows if isinstance(r.get("bar"), int)]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    """The FAKE device only (tests/conftest.py): nothing here opens a real one."""
    from plume import realtime
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")


def test_live_direct_input_she_holds_am_and_the_band_sits_on_it(tmp_path, consent):
    rows = _live(tmp_path, _strums(VOICINGS["Am"]), "hold-direct", direct=True)
    band = [r for r in rows if r.get("entry") not in (None, "drums")]
    assert band, "the band never came in"
    assert all("chords" in r for r in band)
    _sits_on_am(band)


def _sits_on_am(band):
    held = [r for r in band if r["chords"]["source"] == "her_hold"]
    # The fake device loops the take, count and all, so a few bars hear nothing.
    assert len(held) >= 0.85 * len(band), [r["chords"] for r in band]
    assert band[0]["chords"]["source"] == "her_hold"   # in on her chord at once
    assert {r["chord"] for r in held} <= {"Am7", "Am9"}
    assert {r["chords"]["source"] for r in band} <= {"her_hold", "template"}
    # the template it set aside is logged beside it, and it did differ
    assert any(r["chords"]["template"] != "Am7" for r in held)


def test_live_room_mic_she_holds_am_and_the_band_says_it_cannot_tell(tmp_path, consent):
    """On a room mic the band hears its own chord; it keeps its template, and the
    log says so rather than claiming her chord."""
    rows = _live(tmp_path, _strums(VOICINGS["Am"]), "hold-room", fake={"bleed_db": -6.0})
    band = [r for r in rows if r.get("entry") not in (None, "drums")]
    assert band, "the band never came in"
    assert all(r["chords"]["source"] == "template" for r in band)
    assert all(r["chord"] == r["chords"]["template"] for r in band)
    assert all("room mic" in r["chords"]["reason"] for r in band)


def test_live_room_mic_melody_over_bleed_keeps_the_template(tmp_path, consent):
    """Single notes on a room mic with the band loud in it: the band hears its own
    chord in the mic, and must not take that for hers."""
    melody = (60, 64, 62, 67, 69, 65, 71, 72, 64, 62)
    rows = _live(tmp_path, _strums((), notes=melody, every_beats=1), "melody-room",
                 fake={"bleed_db": -6.0})
    band = [r for r in rows if r.get("entry") not in (None, "drums")]
    assert band, "the band never came in"
    followed = [r for r in band if r["chords"]["source"] != "template"]
    assert len(followed) <= 1, [r["chords"] for r in followed]
    assert len({r["chord"] for r in band}) >= 3          # the progression keeps moving
