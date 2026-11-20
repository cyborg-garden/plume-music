"""The key read from chroma and transcribed notes together (listen/notes.py).

Each audio case is a short rendered part with a known key, fed block by block through
the live path's own listening: FeatureExtractor frames and the note listener into one
HarmonyTracker.  The model needs the optional ``notes`` extra; without it those cases
skip, and the rest check the level gate, the wait for the notes, the fallback to
chroma, the listener's bounded work and its thread, and the live wiring.
"""
import gc
import json
import threading
import time

import numpy as np
import pytest
from plume import realtime
from plume.listen import harmony as H
from plume.listen import notes as N
from plume.listen.features import FeatureExtractor
from plume.listen.harmony import HarmonyTracker

SR = 48000
BLOCK = 4096
PC = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9,
      "Bb": 10, "B": 11}

needs_model = pytest.mark.skipif(not N.notes_available(),
                                 reason="optional notes extra (onnxruntime) not installed")


# ------------------------------------------------------------------ rendered players

def tone(midi, dur, *, amp=0.2, decay=1.5, held=False, sr=SR):
    """A harmonic tone, ten partials at 1/k: plucked (exponential decay) or held."""
    t = np.arange(int(dur * sr)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 11) if f0 * k < sr / 2)
    env = (np.ones_like(t) if held else np.exp(-t * decay)) * np.minimum(1.0, t / 0.005)
    n = min(len(t), int(0.01 * sr))
    env[-n:] *= np.linspace(1, 0, n)
    return (amp * x * env).astype(np.float32)


def render(events, dur, *, held=False):
    """events: (start_s, midi, length_s, amp), over the tests' usual noise floor."""
    out = np.zeros(int(dur * SR), np.float32)
    for at, m, ln, a in events:
        x = tone(m, ln, amp=a, decay=1.0 / max(ln, 0.25), held=held)
        i = int(at * SR)
        x = x[:max(0, len(out) - i)]
        out[i:i + len(x)] += x
    return out + (3e-4 * np.random.default_rng(3).standard_normal(len(out))).astype(np.float32)


#: The open-position shapes, low to high.
OPEN = {"C": [48, 52, 55, 60, 64], "G": [43, 47, 50, 55, 59, 67],
        "Am": [45, 52, 57, 60, 64], "F": [41, 48, 53, 57, 60, 65]}


def voicing(name):
    """A six-string voicing, low to high: the open shape where there is one, else
    root, fifth, root (or seventh), third, fifth."""
    if name in OPEN:
        return OPEN[name]
    seventh = name.endswith("maj7")
    name = name[:-4] if seventh else name
    minor = name.endswith("m")
    root = 40 + (PC[name[:-1] if minor else name] - 40) % 12
    v = [root, root + 7, root + (11 if seventh else 12), root + (15 if minor else 16), root + 19]
    return v + [root + 24] if root <= 45 else v


def strums(names, *, bar_s=2.0, bars=8, per_bar=2, t0=0.0):
    """Each chord one bar, strummed low string first ``per_bar`` times."""
    ev = []
    for b in range(bars):
        for s in range(per_bar):
            at = t0 + b * bar_s + s * bar_s / per_bar
            for i, m in enumerate(voicing(names[b % len(names)])):
                ev.append((at + 0.012 * i, m, bar_s / per_bar, 0.12))
    return ev


def picked(names, *, bar_s=2.0, bars=8):
    """Each chord one bar, one string at a time."""
    ev = []
    for b in range(bars):
        v = voicing(names[b % len(names)])
        for s, k in enumerate([0, 2, 3, 4, 3, 2, 4, 3]):
            ev.append((b * bar_s + s * bar_s / 8, v[k], bar_s / 3, 0.16))
    return ev


def line(midis, *, step_s=0.5, reps=2, amp=0.3, hold=0.9):
    """One note at a time."""
    return [((r * len(midis) + i) * step_s, m, step_s * hold, amp)
            for r in range(reps) for i, m in enumerate(midis)]


def hum(n, hz, db=-50.0):
    """Mains hum with two harmonics at ``db`` dBFS."""
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * hz * t) + 0.5 * np.sin(2 * np.pi * 2 * hz * t + 1)
         + 0.35 * np.sin(2 * np.pi * 3 * hz * t + 2)) / 1.2
    return (10 ** (db / 20) * np.sqrt(2) * x).astype(np.float32)


def listen_to(x, *, notes=True):
    """(the key after each second of ``x``, the tracker)."""
    fe = FeatureExtractor(sr=SR)
    tr = HarmonyTracker(notes=N.NoteListener(SR) if notes else None)
    keys = []
    for i in range(0, len(x), BLOCK):
        blk = x[i:i + BLOCK]
        tr.listen(blk)
        h = tr.process(fe.process(blk))
        if (i + BLOCK) // SR > len(keys):
            keys.append(h.key)
    return keys, tr


def read_key(x, *, notes=True):
    return listen_to(x, notes=notes)[0][-1]


# ------------------------------------------------------------------ what the two readings fix

@needs_model
def test_a_drone_under_a_minor_melody_is_minor():
    # A low A and E ring the whole time, louder than the tune: the tonic is most of
    # what sounds, so only the melody's third can say major or minor.  In chroma the
    # drone's fifth harmonic is a C sharp, the major third: chroma alone says A major.
    dur = 16.0
    ev = [(k * 4.0, 45, dur - k * 4.0, 0.3) for k in range(4)] + [(0.0, 52, dur, 0.2)]
    tune = [69, 72, 76, 74, 72, 71, 69, 67, 69, 72, 74, 76, 72, 71, 69, 69]
    ev += [(i * 0.5, m, 0.6, 0.18) for i, m in enumerate(tune * 2)]
    x = render(ev, dur + 0.5)
    assert read_key(x) == "A minor"
    assert read_key(x, notes=False) == "A major"


@needs_model
def test_a_major_loop_is_not_the_minor_key_on_its_third():
    # C G Am F: in chroma the overtones feed E and B, and chroma alone says E minor.
    x = render(strums(["C", "G", "Am", "F"]), 16.5)
    assert read_key(x) == "C major"
    assert read_key(x, notes=False) != "C major"


@needs_model
def test_a_maj7_home_chord_over_its_root_is_major_within_four_bars():
    # Cmaj7 holds E minor (E G B) and G major's third and fifth; chroma alone names a
    # key a fifth up for the first five bars.
    x = render(strums(["Cmaj7", "Fmaj7"], bars=4), 8.5)
    assert read_key(x) == "C major"
    assert read_key(x, notes=False) != "C major"


@needs_model
def test_the_same_loop_started_on_its_minor_chord_stays_minor():
    # Am F C G holds the notes of C major for the same lengths; what was heard first
    # is kept against the relative key, as it is on chroma alone.
    x = render(strums(["Am", "F", "C", "G"]), 16.5)
    assert read_key(x) == "A minor"
    assert read_key(render(picked(["Am", "F", "C", "G"]), 16.5)) == "A minor"


@needs_model
@pytest.mark.parametrize("names,key,by_s", [(["F", "G", "C", "C"], "C major", 10),
                                            (["F", "G", "Am", "Am"], "A minor", 18)])
def test_a_part_that_opens_away_from_home_finds_home(names, key, by_s):
    # No weight is given to the first chord: both open on F, are first heard as F
    # major, and neither stays there.
    keys, _ = listen_to(render(strums(names, bars=12), 24.5))
    assert keys[5] == "F major"
    assert set(keys[by_s:]) == {key}


@needs_model
def test_bass_lines_and_a_single_line_are_read():
    # Roots and fifths with no third anywhere: the notes alone cannot say major or
    # minor (they say G minor) and chroma alone hears the fifth as home (D major).
    root_fifth = [31, 38, 31, 38, 36, 43, 36, 43, 38, 45, 38, 45, 31, 38, 31, 38]
    assert read_key(render(line(root_fifth), 16.5)) == "G major"
    # A minor riff that leans on its seventh: chroma alone drifts to Eb major.
    riff = [29, 29, 32, 29, 36, 34, 32, 29, 27, 27, 31, 27, 34, 32, 31, 27]
    assert read_key(render(line(riff), 16.5)) == "F minor"
    # A held single-line tune in C: chroma alone says E minor.
    tune = [60, 64, 67, 64, 65, 62, 59, 62, 60, 64, 67, 72, 71, 67, 62, 60]
    assert read_key(render(line(tune, amp=0.15, hold=0.95), 16.5, held=True)) == "C major"


@needs_model
def test_a_change_to_a_near_key_is_followed():
    # C to G, one sharp apart, the change at 12 s.
    ev = strums(["C", "F", "G", "C"], bar_s=1.5) + strums(["G", "C", "D", "G"], bar_s=1.5,
                                                          bars=20, t0=12.0)
    keys, _ = listen_to(render(ev, 42.5))
    assert keys[11] == "C major"
    assert keys[-1] == "G major"


# ------------------------------------------------------------------ hum and the level gate

@needs_model
@pytest.mark.parametrize("hz,part,key", [
    (60, picked(["Am", "F", "C", "G"]), "A minor"),
    (50, strums(["Dm", "Bb", "F", "C"]), "D minor"),
])
def test_mains_hum_before_the_first_note_is_not_part_of_the_key(hz, part, key):
    # Three seconds of hum at -50 dBFS, then the part over it.  The model scales each
    # window to its loudest part and reports the hum as one clean low note until the
    # playing starts.
    x = np.concatenate([np.zeros(3 * SR, np.float32), render(part, 16.5)])
    x = x + hum(len(x), hz)
    keys, tr = listen_to(x)
    assert keys[3 + 8] == key and keys[-1] == key
    # the hum's own notes were heard at the hum's level, 40 dB and more under full
    # scale, and are counted for nothing beside the playing
    split = int((-40.0 - H._NOTE_LEVEL_LOW_DB) // H._NOTE_LEVEL_STEP_DB)
    assert tr._note_hist[:split].sum() > H._NOTE_MASS_FLOOR
    assert np.allclose(tr._note_counts(), tr._note_hist[split:].sum(axis=0))


def _frames(seconds, level_db, pcs, *, t0=0.0, octave=48):
    """Note frames as the listener gives them: ``pcs`` sounding at ``level_db``."""
    p = np.zeros(N.N_KEYS, np.float32)
    for pc in pcs:
        p[octave + pc - N.MIDI_LOW] = 0.9
    n = int(seconds / N.FRAME_S)
    return [(t0 + (i + 0.5) * N.FRAME_S, p, level_db) for i in range(n)]


def test_notes_far_under_the_loud_playing_count_for_nothing():
    tr = HarmonyTracker(notes=object())
    tr.hear_notes(_frames(3.0, -50.0, [10]))                 # hum: one low B flat
    # nothing louder has been heard: it is all there is, and it counts
    assert tr._note_counts()[10] > H._NOTE_MASS_FLOOR
    tr.hear_notes(_frames(1.0, -20.0, [0, 4, 7], t0=3.0))    # playing, 30 dB over it
    counts = tr._note_counts()
    assert counts[10] == 0.0
    assert counts[0] > 0.0 and counts[0] == counts[4] == counts[7]


@pytest.mark.parametrize("under_db,least,most", [(15.0, 1.0, 1.2), (25.0, 0.4, 0.7),
                                                  (35.0, 0.0, 0.0)])
def test_how_much_a_softer_passage_counts(under_db, least, most):
    # Two seconds of loud playing, then two of one note ``under_db`` under it: a soft
    # passage is playing too and counts in full, a far softer one for part, and what
    # is 30 dB and more under the loud playing for nothing.
    tr = HarmonyTracker(notes=object())
    tr.hear_notes(_frames(2.0, -12.0, [0, 4, 7]))
    tr.hear_notes(_frames(2.0, -12.0 - under_db, [9], t0=2.0))
    counts = tr._note_counts()
    assert least * counts[0] <= counts[9] <= most * counts[0]


def test_one_knock_does_not_set_the_level_of_the_playing():
    # Quiet playing, then a knock at full scale that the model hears a note in: the
    # loud level is where a share of the notes is, not the loudest frame.
    tr = HarmonyTracker(notes=object())
    tr.hear_notes(_frames(4.0, -45.0, [0, 4, 7]))
    tr.hear_notes(_frames(0.05, -1.0, [6], t0=4.0))
    tr.hear_notes(_frames(2.0, -45.0, [0, 4, 7], t0=4.05))
    counts = tr._note_counts()
    assert counts[0] > 100.0 and counts[0] > 50 * counts[6]


def test_scattered_notes_in_room_noise_are_not_counted():
    # The model reports a note here and there in noise; less than half a note's worth
    # at a time does not count, at any level.
    tr = HarmonyTracker(notes=object())
    p = np.zeros(N.N_KEYS, np.float32)
    frames = []
    for i in range(400):
        q = p.copy()
        if i % 10 == 0:
            q[30 + i % 7] = 0.55
        frames.append(((i + 0.5) * N.FRAME_S, q, -30.0))
    tr.hear_notes(frames)
    assert tr._note_hist.sum() == 0.0


# ------------------------------------------------------------------ the two readings together

class _Deaf:
    """A listener whose model never hears a note."""

    def process(self, block):
        return []


def _chroma_frames(x):
    fe = FeatureExtractor(sr=SR)
    return [fe.process(x[i:i + BLOCK]) for i in range(0, len(x), BLOCK)]


def test_the_key_waits_for_the_notes_and_then_reads_chroma_alone():
    blocks = _chroma_frames(render(strums(["C", "F", "G", "C"]), 16.5))
    plain, deaf = HarmonyTracker(), HarmonyTracker(notes=_Deaf())
    seen = []
    for frames in blocks:
        a, b = plain.process(frames), deaf.process(frames)
        seen.append((a.t_s, a.key, b.key))
    # chroma alone has named the key by 7 s; with a listener that has said nothing
    # yet, no key is named on chroma alone that early
    assert [k for t, k, _ in seen if t < 7.0][-1] == "C major"
    assert all(k is None for t, _, k in seen if t < 7.0)
    # a part the model cannot hear still gets its key, from chroma
    assert seen[-1][1] == seen[-1][2] == "C major"


def test_the_key_is_the_two_readings_averaged(monkeypatch):
    # Chroma that says C major, notes that say F sharp major: with all of the weight
    # on one reading the key is that reading's, and neither wins the even average.
    blocks = _chroma_frames(render(strums(["C", "F", "G", "C"], bars=4), 8.5))

    def key(share):
        monkeypatch.setattr(H, "_NOTE_SHARE", share)
        tr = HarmonyTracker(notes=object())
        tr.hear_notes(_frames(8.0, -20.0, [6, 10, 1]))
        for frames in blocks:
            h = tr.process(frames)
        return h.key

    assert key(0.0) == "C major"
    assert key(1.0) == "F# major"
    assert H._NOTE_SHARE == 1.0                               # (still patched)
    monkeypatch.undo()
    assert H._NOTE_SHARE == 0.5
    assert key(0.5) not in ("C major", "F# major")


def test_without_the_model_the_key_comes_from_chroma(monkeypatch):
    monkeypatch.setattr(N, "notes_available", lambda: False)
    monkeypatch.delenv(realtime.NOTES_ENV, raising=False)
    assert N.note_listener(SR) is None
    tr = realtime.harmony_tracker()
    assert not tr.hears_notes
    tr.listen(np.zeros(BLOCK, np.float32))               # a no-op without a listener
    assert read_key(render(strums(["C", "F", "G", "C"]), 16.5), notes=False) == "C major"


# ------------------------------------------------------------------ the listener

@needs_model
def test_the_listener_hears_each_frame_once_in_order_with_its_level():
    nl = N.NoteListener(SR)
    x = np.concatenate([tone(69, 3.0, held=True), np.zeros(SR, np.float32)])
    frames = []
    for i in range(0, len(x), BLOCK):
        got = nl.process(x[i:i + BLOCK])
        heard_s = (i + BLOCK) / SR
        # no frame is given out before the model has heard what follows it
        assert all(f[0] <= heard_s - (N._RIGHT_MARGIN - 0.5) * N.FRAME_S for f in got)
        frames.extend(got)
    t = np.array([f[0] for f in frames])
    assert np.all(np.diff(t) > 0)                        # each frame once, in order
    assert np.allclose(np.diff(t), N.FRAME_S)            # and none skipped
    # everything up to about a second and a quarter before the end is out
    assert t[-1] > len(x) / SR - 1.25
    sounding = np.array([f[1] for f in frames if 0.2 < f[0] < 2.5])
    assert np.median(np.argmax(sounding, axis=1)) == 69 - N.MIDI_LOW
    # the level under each frame is the input's: the tone's, then silence
    rms_db = 10 * np.log10(np.mean(x[SR:2 * SR] ** 2))
    assert all(abs(f[2] - rms_db) < 1.0 for f in frames if 1.0 < f[0] < 2.0)
    assert all(f[2] < -100.0 for f in frames if f[0] > 3.2)


class _Session:
    """Stands in for the model: counts its runs, and can be made to take its time."""

    def __init__(self, gate=None):
        self.runs, self.sizes, self.gate = 0, set(), gate

    def get_inputs(self):
        return [type("I", (), {"name": "in"})()]

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in ("out:0", "out:1", "out:2")]

    def run(self, _names, feeds):
        if self.gate is not None:
            self.gate.wait(30.0)
        self.runs += 1
        self.sizes.add(feeds["in"].shape)
        return [np.zeros((1, N.MODEL_FRAMES, N.N_KEYS), np.float32)]


def test_a_steady_low_note_has_a_steady_level():
    # The low B of a five-string bass is 31 Hz: one model frame is a third of a cycle
    # of it, so the level is taken over several frames, or it would swing 3 dB with
    # the waveform and spread one held note over the level bins.
    nl = N.NoteListener(SR, session=_Session())
    x = (0.2 * np.sin(2 * np.pi * 30.87 * np.arange(4 * SR) / SR)).astype(np.float32)
    frames = []
    for i in range(0, len(x), BLOCK):
        frames.extend(nl.process(x[i:i + BLOCK]))
    level = np.array([f[2] for f in frames if 1.0 < f[0] < 2.5])
    assert len(level) > 100 and level.max() - level.min() < 1.5
    assert abs(level.mean() - 20 * np.log10(0.2 / np.sqrt(2))) < 0.5


def test_the_listener_runs_the_model_once_a_second_on_a_fixed_window():
    sess = _Session()
    nl = N.NoteListener(SR, session=sess)
    seconds = 30
    for _ in range(seconds * SR // 1024):                # small blocks, as live
        nl.process(np.zeros(1024, np.float32))
    assert seconds - 1 <= sess.runs <= seconds           # never more than one a second
    assert sess.sizes == {(1, N.MODEL_N, 1)}             # always one 2 s window
    assert nl._buf.size <= nl._keep                      # and no more audio is kept


def _note_threads():
    return [t for t in threading.enumerate() if t.name == "plume-notes" and t.is_alive()]


def _gone(before, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if len(_note_threads()) <= before:
            return True
        time.sleep(0.01)
    return False


def test_on_its_own_thread_the_listener_gives_the_same_frames():
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(5 * SR) / SR)).astype(np.float32)
    inline = N.NoteListener(SR, session=_Session())
    before = len(_note_threads())
    apart = N.NoteListener(SR, session=_Session(), background=True)
    assert len(_note_threads()) == before + 1
    a, b = [], []
    for i in range(0, len(x), 1024):
        a.extend(inline.process(x[i:i + 1024]))
        b.extend(apart.process(x[i:i + 1024]))
        apart.wait()
    b.extend(apart.process(np.zeros(0, np.float32)))     # what the last window gave
    assert len(a) == len(b) > 300
    assert all(fa[0] == fb[0] and fa[2] == fb[2] for fa, fb in zip(a, b))
    assert apart.replaced == 0
    apart.close()
    assert _gone(before)


def test_a_slow_model_never_holds_up_the_caller_and_loses_no_frames_it_can_still_read():
    gate = threading.Event()                             # the model is stuck until set
    sess = _Session(gate)
    before = len(_note_threads())
    nl = N.NoteListener(SR, session=sess, background=True)
    block = np.zeros(1024, np.float32)
    frames = []
    started = time.monotonic()
    for _ in range(4 * SR // 1024 + 1):                  # four seconds: four windows due
        frames.extend(nl.process(block))
    assert time.monotonic() - started < 2.0              # never waited for the model
    assert frames == [] and sess.runs == 0
    assert nl.replaced == 2                              # one running, one waiting, two replaced
    gate.set()
    nl.wait()
    frames.extend(nl.process(np.zeros(0, np.float32)))
    assert sess.runs == 2                                # the stuck window and the newest
    t = np.array([f[0] for f in frames])
    assert np.all(np.diff(t) > 0)                        # in order, none twice
    # the newest window still covers the two seconds before it: only what fell
    # between the first window and that is lost
    assert t[-1] > 4.0 - 0.3 and np.allclose(np.diff(t[t > 2.0]), N.FRAME_S)
    nl.close()
    assert _gone(before)


def test_dropping_the_listener_ends_its_thread():
    # A take makes a new tracker, and so a new listener, when its count lands.
    before = len(_note_threads())
    tr = HarmonyTracker(notes=N.NoteListener(SR, session=_Session(), background=True))
    tr.listen(np.zeros(SR, np.float32))
    assert len(_note_threads()) == before + 1
    del tr
    gc.collect()
    assert _gone(before)


# ------------------------------------------------------------------ the live wiring

def test_the_live_tracker_takes_notes_only_from_audio_and_can_be_switched(monkeypatch):
    made = []
    monkeypatch.setattr(N, "note_listener", lambda sr, **kw: made.append((sr, kw)) or object())
    monkeypatch.delenv(realtime.NOTES_ENV, raising=False)
    assert realtime.harmony_tracker().hears_notes
    assert made == [(realtime.SR, {"background": True})]     # on its own thread, live
    assert not realtime.harmony_tracker("a midi port").hears_notes   # MIDI is notes already
    monkeypatch.setenv(realtime.NOTES_ENV, "0")
    assert not realtime.harmony_tracker().hears_notes
    assert len(made) == 1
    monkeypatch.setenv(realtime.NOTES_ENV, "inline")
    assert realtime.harmony_tracker().hears_notes
    assert made[-1] == (realtime.SR, {"background": False})


def test_a_tracker_that_takes_no_arguments_still_stands_in_when_notes_are_off(monkeypatch):
    # Tests and tools swap in a tracker class of their own, made with no arguments.
    class Plain:
        def __init__(self):
            self.made = True

    monkeypatch.setattr(H, "HarmonyTracker", Plain)
    monkeypatch.setenv(realtime.NOTES_ENV, "0")
    assert realtime.harmony_tracker().made
    assert realtime.harmony_tracker("a midi port").made


def test_the_analysis_step_hands_the_listener_the_unmasked_input():
    class Listener:
        def __init__(self):
            self.blocks = []

        def process(self, block):
            self.blocks.append(np.array(block))
            return []

    from plume.listen.beat import BeatTracker
    ear = Listener()
    human = (0.1 * np.sin(2 * np.pi * 220 * np.arange(2048) / SR)).astype(np.float32)
    realtime.analysis_step(
        extractor=FeatureExtractor(sr=SR), hop_rms=realtime.HopRms(),
        masker=realtime.SelfMasker(), tracker=BeatTracker(),
        harmony=HarmonyTracker(notes=ear),
        human_block=human, machine_block=np.zeros(2048, np.float32))
    assert len(ear.blocks) == 1 and np.array_equal(ear.blocks[0], human)


@needs_model
@pytest.mark.ends_by_what_was_heard
def test_a_live_take_with_the_note_reading_on_counts_in_and_enters_in_key(
        tmp_path, monkeypatch, plume_registry):
    # The one live take that runs as an install with the extra does: the listener on
    # its own thread, beside the count detection and the tempo following.  Four
    # knocks at 100, then G major strummed twice a beat; it ends three bars after the
    # band comes in.
    from test_detect import _detect_take, _rows

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    monkeypatch.delenv(realtime.NOTES_ENV, raising=False)
    made, heard = [], []
    real_listener, real_hear = N.note_listener, HarmonyTracker.hear_notes

    def listener(sr, **kw):
        made.append(kw)
        return real_listener(sr, **kw)

    def hear(self, frames):
        heard.append(len(frames))
        return real_hear(self, frames)

    monkeypatch.setattr(N, "note_listener", listener)
    monkeypatch.setattr(HarmonyTracker, "hear_notes", hear)
    bpm, seconds = 100.0, 70.0
    period = 60.0 / bpm
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int((seconds + 20.0) * SR)).astype(np.float32) * 3e-4
    for i in range(4):
        at, n = int((2.0 + i * period) * SR), int(0.004 * SR)
        x[at:at + n] += rng.standard_normal(n).astype(np.float32) * 0.8
    start = 2.0 + 8 * period
    bars = int((seconds - start) / (4 * period))
    part = render(strums(["G", "C", "D", "G"], bar_s=4 * period, bars=bars, per_bar=8),
                  bars * 4 * period)
    i = int(start * SR)
    x[i:i + len(part)] += part
    before = len(_note_threads())
    rows = _rows(_detect_take(tmp_path, x, name="notes", until_band=True))
    # the count was heard and the tempo taken from it, with the model running beside
    assert rows and abs(rows[-1]["bpm"] - bpm) < 2.0
    band = [r for r in rows if r["entry"] != "drums"]
    assert band and band[0]["key"] == "G major"
    # each of the take's trackers had a listener on its own thread, and notes came
    assert made and all(kw == {"background": True} for kw in made)
    assert sum(heard) > 500
    gc.collect()
    assert _gone(before)
