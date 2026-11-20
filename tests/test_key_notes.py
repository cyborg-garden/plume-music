"""The key read from transcribed notes (listen/notes.py) on synthetic signals.

Each case is a short rendered part with a known key, fed block by block through the
live path's own listening: FeatureExtractor frames and the note listener into one
HarmonyTracker.  The model needs the optional ``notes`` extra; without it those
cases skip, and the rest check that the key still comes from chroma, that the live
path hands the listener its audio, and that the listener's work is bounded.
"""
import numpy as np
import pytest

from plume import realtime
from plume.listen import notes as N
from plume.listen.features import FeatureExtractor
from plume.listen.harmony import HarmonyTracker

SR = 48000
BLOCK = 4096

needs_model = pytest.mark.skipif(not N.notes_available(),
                                 reason="optional notes extra (onnxruntime) not installed")


def tone(midi, dur, *, amp=0.2, decay=1.5, sr=SR):
    """A plucked harmonic tone: ten partials at 1/k, exponential decay."""
    t = np.arange(int(dur * sr)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 11) if f0 * k < sr / 2)
    env = np.exp(-t * decay) * np.minimum(1.0, t / 0.005)
    env[-int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))
    return (amp * x * env).astype(np.float32)


def render(events, dur):
    """events: (start_s, midi, length_s, amp)"""
    out = np.zeros(int(dur * SR), np.float32)
    for at, m, ln, a in events:
        x = tone(m, ln, amp=a, decay=1.0 / max(ln, 0.25))
        i = int(at * SR)
        out[i:i + len(x)] += x[:max(0, len(out) - i)]
    return out


# Chords as a six-string instrument in standard tuning voices them, low to high.
CHORDS = {
    "C": [48, 52, 55, 60, 64], "G": [43, 47, 50, 55, 59, 67],
    "Am": [45, 52, 57, 60, 64], "F": [41, 48, 53, 57, 60, 65],
    "E": [40, 47, 52, 56, 59, 64], "A": [45, 52, 57, 61, 64],
    "B": [47, 54, 59, 63, 66],
    "Cmaj7": [48, 52, 55, 59, 64], "Fmaj7": [41, 48, 53, 57, 60, 64],
}


def strums(names, *, bar_s=2.0, bars=8, per_bar=2):
    """Each chord one bar, strummed low string first ``per_bar`` times."""
    ev = []
    for b in range(bars):
        for s in range(per_bar):
            t0 = b * bar_s + s * bar_s / per_bar
            for i, m in enumerate(CHORDS[names[b % len(names)]]):
                ev.append((t0 + 0.012 * i, m, bar_s / per_bar, 0.12))
    return render(ev, bars * bar_s + 1.0)


def read_keys(x, *, notes=True):
    """The key after each second of ``x``."""
    fe = FeatureExtractor(sr=SR)
    tr = HarmonyTracker(notes=N.NoteListener(SR) if notes else None)
    keys = []
    for i in range(0, len(x), BLOCK):
        blk = x[i:i + BLOCK]
        tr.listen(blk)
        h = tr.process(fe.process(blk))
        if (i + BLOCK) // SR > len(keys):
            keys.append(h.key)
    return keys


def read_key(x, *, notes=True):
    return read_keys(x, notes=notes)[-1]


@needs_model
def test_a_drone_under_a_minor_melody_is_minor():
    # A low A and E ring the whole time, louder than the tune: the tonic is most of
    # what sounds, so only the melody's third can say major or minor.  In chroma the
    # drone's fifth harmonic is a C sharp, the major third.
    dur = 16.0
    drone = render([(0.0, 45, dur, 0.35), (0.0, 52, dur, 0.2)], dur + 0.5)
    for k in range(1, 4):                          # struck again every two bars
        drone += render([(k * 4.0, 45, dur - k * 4.0, 0.3)], dur + 0.5)
    tune = [69, 72, 76, 74, 72, 71, 69, 67, 69, 72, 74, 76, 72, 71, 69, 69]
    mel = render([(i * 0.5, m, 0.6, 0.18) for i, m in enumerate(tune * 2)], dur + 0.5)
    assert read_key(drone + mel) == "A minor"


@needs_model
@pytest.mark.parametrize("names", [["Cmaj7"], ["Cmaj7", "Cmaj7", "Fmaj7", "Fmaj7"]])
def test_a_maj7_home_chord_is_not_the_minor_chord_on_its_third(names):
    # Cmaj7 holds E minor (E G B), and in chroma its overtones feed B and E more.
    assert read_key(strums(names)) == "C major"


@needs_model
def test_one_loop_of_chords_is_major_or_its_relative_minor_by_where_it_starts():
    # The same four chords for the same lengths: only the order differs.
    assert read_key(strums(["C", "G", "Am", "F"])) == "C major"
    assert read_key(strums(["Am", "F", "C", "G"])) == "A minor"


@needs_model
def test_a_change_of_key_is_still_followed_after_the_opening():
    # The opening note leans the reading toward where the part began; it must not
    # pin it there when the part moves to a far key and stays.
    x = np.concatenate([strums(["C", "F", "G", "C"]), strums(["E", "A", "B", "E"], bars=16)])
    keys = read_keys(x)
    assert keys[15] == "C major"
    assert keys[-1] == "E major"


@needs_model
def test_room_noise_before_the_first_note_is_not_the_opening():
    # A fresh listener hears only faint noise between a count and the first note.  The
    # model scales each window to its loudest part and reports scattered notes there;
    # they must not count, least of all as where the part began.
    rng = np.random.default_rng(3)
    part = strums(["Am", "F", "C", "G"])
    x = np.concatenate([np.zeros(3 * SR, np.float32), part])
    x += (3e-4 * rng.standard_normal(len(x))).astype(np.float32)
    fe = FeatureExtractor(sr=SR)
    tr = HarmonyTracker(notes=N.NoteListener(SR))
    for i in range(0, len(x), BLOCK):
        blk = x[i:i + BLOCK]
        tr.listen(blk)
        h = tr.process(fe.process(blk))
        if (i + BLOCK) / SR < 3.0:
            assert tr._open_t is None and tr._note_hist.sum() < 1.0
    assert tr._open_t is not None and tr._open_t >= 3.0
    assert h.key == "A minor"


@needs_model
def test_the_listener_hears_each_frame_once_and_in_order():
    nl = N.NoteListener(SR)
    x = np.concatenate([tone(69, 3.0, decay=0.3), np.zeros(SR, np.float32)])
    frames = []
    for i in range(0, len(x), BLOCK):
        frames.extend(nl.process(x[i:i + BLOCK]))
    t = np.array([f[0] for f in frames])
    assert np.all(np.diff(t) > 0)                        # each frame once, in order
    assert np.allclose(np.diff(t), N.FRAME_S)            # and none skipped
    # everything up to about a second and a quarter before the end is out
    assert t[-1] > len(x) / SR - 1.25
    sounding = np.array([f[1] for f in frames if 0.2 < f[0] < 2.5])
    assert np.median(np.argmax(sounding, axis=1)) == 69 - N.MIDI_LOW


class _CountingSession:
    """Stands in for the model: counts its runs and the size of what it is given."""

    def __init__(self):
        self.runs, self.sizes = 0, set()

    def get_inputs(self):
        return [type("I", (), {"name": "in"})()]

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in ("out:0", "out:1", "out:2")]

    def run(self, _names, feeds):
        self.runs += 1
        self.sizes.add(feeds["in"].shape)
        return [np.zeros((1, N.MODEL_FRAMES, N.N_KEYS), np.float32)]


def test_the_listener_runs_the_model_once_a_second_on_a_fixed_window():
    sess = _CountingSession()
    nl = N.NoteListener(SR, session=sess)
    seconds = 30
    for _ in range(seconds * SR // 1024):                # small blocks, as live
        nl.process(np.zeros(1024, np.float32))
    assert seconds - 1 <= sess.runs <= seconds           # never more than one a second
    assert sess.sizes == {(1, N.MODEL_N, 1)}             # always one 2 s window
    assert nl._buf.size <= nl._keep                      # and no more audio is kept


def test_without_the_model_the_key_still_comes_from_chroma(monkeypatch):
    monkeypatch.setattr(N, "notes_available", lambda: False)
    monkeypatch.delenv(realtime.NOTES_ENV, raising=False)
    assert N.note_listener(SR) is None
    tr = realtime.harmony_tracker()
    assert not tr.hears_notes
    tr.listen(np.zeros(BLOCK, np.float32))               # a no-op without a listener
    assert read_key(strums(["C", "F", "G", "C"]), notes=False) == "C major"


def test_the_live_tracker_takes_notes_only_from_audio_and_can_be_switched_off(monkeypatch):
    made = []
    monkeypatch.setattr(N, "note_listener", lambda sr: made.append(sr) or object())
    monkeypatch.delenv(realtime.NOTES_ENV, raising=False)
    assert realtime.harmony_tracker().hears_notes
    assert made == [realtime.SR]
    assert not realtime.harmony_tracker("a midi port").hears_notes   # MIDI is notes already
    monkeypatch.setenv(realtime.NOTES_ENV, "0")
    assert not realtime.harmony_tracker().hears_notes
    assert made == [realtime.SR]


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
