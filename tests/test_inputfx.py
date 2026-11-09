"""Her sound on Hear me (plume.inputfx): chosen by name, heard only by her, never late.

Everything here is offline: synthetic plucks and chords pushed through the effect a
buffer at a time, the way the audio callback does.  Nothing is played.
"""
from __future__ import annotations

import sys
import tracemalloc

import numpy as np
import pytest
from scipy.signal import lfilter

from plume import inputfx, inputguard, realtime
from plume.inputfx import SOUNDS, InputFx

SR, N = realtime.SR, realtime.BLOCK
PLAYED = SOUNDS[1:]


def pluck(f: float, seconds: float, seed: int = 0) -> np.ndarray:
    """Karplus-Strong, vectorised a period at a time (each period depends on the last)."""
    rng = np.random.default_rng(seed)
    p = int(round(SR / f))
    buf = rng.uniform(-1.0, 1.0, p)
    out = np.empty(int(seconds * SR))
    for i in range(0, out.size, p):
        k = min(p, out.size - i)
        out[i:i + k] = buf[:k]
        buf = 0.996 * 0.5 * (buf + np.roll(buf, -1))
    return out


def playing(seconds: float = 6.0, peak_db: float = -9.0) -> np.ndarray:
    """Single notes, then strummed chords, at a direct-input level."""
    x = np.zeros(int(seconds * SR))
    t = 0.05
    for i, f in enumerate((196.0, 246.9, 293.7, 392.0)):
        s = int(t * SR)
        y = pluck(f, 1.0, seed=i)[:max(0, x.size - s)]
        x[s:s + y.size] += y
        t += 0.4
    for chord in ((98.0, 146.8, 196.0, 246.9, 293.7, 392.0), (110.0, 164.8, 220.0, 261.6, 329.6)):
        for k, f in enumerate(chord):
            s = int((t + 0.012 * k) * SR)
            y = 0.7 * pluck(f, 2.0, seed=10 + k)[:max(0, x.size - s)]
            x[s:s + y.size] += y
        t += 1.6
    x *= 10.0 ** (peak_db / 20.0) / np.max(np.abs(x))
    return x.astype(np.float32)


def through(fx: InputFx, x: np.ndarray, duck=lambda i: False) -> np.ndarray:
    out = np.zeros((x.size // N * N, 2), dtype=np.float32)
    for i in range(0, out.shape[0], N):
        y = fx.process(x[i:i + N], duck(i))
        out[i:i + N] = x[i:i + N, None] if y is None else y
    return out


def lufs(st: np.ndarray) -> float:
    """BS.1770 integrated loudness (K-weighted, gated), for stereo at 48 kHz."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1, -2, 1], [1, -1.99004745483398, 0.99007225036621]
    y = lfilter(b2, a2, lfilter(b1, a1, st, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = np.array([np.mean(np.sum(y[i:i + blk] ** 2, axis=1)) for i in range(0, len(y) - blk, hop)])
    ms = ms[-0.691 + 10 * np.log10(ms + 1e-20) > -70]
    ms = ms[-0.691 + 10 * np.log10(ms) > -0.691 + 10 * np.log10(np.mean(ms)) - 10]
    return -0.691 + 10 * np.log10(np.mean(ms))


# ------------------------------------------------------------------ choosing a sound

def test_names_and_aliases():
    assert inputfx.canonical_sound("Dreamy") == "dreamy"
    assert inputfx.canonical_sound("clean room") == "clean"
    assert inputfx.canonical_sound("lo-fi") == "lofi"
    assert inputfx.canonical_sound("match") == "match"
    assert inputfx.canonical_sound("wobbly") is None
    assert inputfx.canonical_sound(None) is None


def test_match_takes_the_style_and_the_default_is_off():
    assert InputFx(SR, N).playing == "off"
    assert inputfx.resolve("match", "lofi") == "lofi"
    assert inputfx.resolve("match", "synthwave") == "dreamy"
    assert inputfx.resolve("match", "no-such-style") == "clean"
    assert inputfx.resolve("match", None) == "clean"
    fx = InputFx(SR, N, "match")
    fx.set_style("lofi")
    assert fx.playing == "lofi"
    fx = InputFx(SR, N, "crunch")
    fx.set_style("lofi")                        # a sound she named is hers to keep
    assert fx.playing == "crunch"


def test_every_bundled_style_has_a_sound():
    from plume.styles_lib import list_styles
    for style in list_styles():
        assert inputfx.resolve("match", style) in PLAYED


def test_a_cue_needs_the_word_sound():
    """"dreamy" alone is a mood for the band, and "dry"/"wet" its tone cues."""
    assert inputfx.parse_sound_cue("sound dreamy") == "dreamy"
    assert inputfx.parse_sound_cue("dreamy sound") == "dreamy"
    assert inputfx.parse_sound_cue("my sound off") == "off"
    assert inputfx.parse_sound_cue("dreamy") is None
    assert inputfx.parse_sound_cue("dry") is None
    assert inputfx.parse_sound_cue("wet") is None
    assert inputfx.parse_sound_cue("make it sound darker") is None   # the band's, not hers
    assert InputFx(SR, N).handle_cue("make it sound darker") is None
    fx = InputFx(SR, N, "clean")
    assert fx.handle_cue("busier") is None
    reply = fx.handle_cue("sound lofi")
    assert reply["ok"] and reply["sound"] == "lofi" and fx.playing == "lofi"
    assert fx.handle_cue("sound wobbly")["ok"] is False
    assert fx.playing == "lofi"


@pytest.mark.parametrize("text", [
    "darker sound", "sound brighter", "a warmer sound", "bigger sound", "sound louder",
    "softer sound", "busier sound", "sound sparser", "sound like jazz", "sound like house",
    "my sound darker", "dry sound", "sound dry", "wetter sound", "sound wet",
])
def test_a_band_cue_that_says_sound_still_reaches_the_band(text):
    """The band understands these.  Her sound takes a cue only when it names one of
    her sounds outright, or when the band would not have understood it anyway."""
    from plume.mind import parse_cue
    assert parse_cue(text)["kind"] != "unknown", "pick a cue the band knows"
    fx = InputFx(SR, N, "clean")
    assert fx.handle_cue(text) is None
    assert fx.playing == "clean"


@pytest.mark.parametrize("text", [
    "sound lo-fi", "my sound lo-fi", "lo-fi sound", "sound: lo-fi", "sound lo fi", "lo fi sound",
    "sound lofi tape", "lofi tape sound", "sound Lofi Tape", "sound lo-fi tape",
])
def test_another_spelling_of_her_sound_is_still_hers(text):
    """"lo-fi" is a spelling of her lofi sound and "Lofi tape" is its name on the page.
    The band reads both as a style, and one hyphen must not change who acts."""
    from plume.mind import parse_cue
    assert parse_cue(text)["kind"] == "style", "pick a spelling the band would take"
    fx = InputFx(SR, N, "crunch")
    reply = fx.handle_cue(text)
    assert reply is not None and reply["ok"] is True and reply["sound"] == "lofi"
    assert fx.playing == "lofi"


def test_every_label_on_the_page_names_its_sound():
    from plume import guitake
    for name, label in guitake.SOUND_CHOICES:
        assert inputfx.canonical_sound(label) == name
        fx = InputFx(SR, N, "crunch" if name != "crunch" else "clean")
        reply = fx.handle_cue(f"sound {label}")
        assert reply is not None and reply["ok"] is True, label
        assert fx.asked == name


@pytest.mark.parametrize("text,op", [
    ("loop this sound", "record"), ("loop that sound", "record"), ("sound: loop this", "record"),
    ("mute loop sound", "mute"), ("sound mute loop1", "mute"), ("list loops sound", "list"),
    ("sound loop off", "stop"),
])
def test_a_loop_cue_that_says_sound_reaches_the_looper(text, op):
    """lead asks her sound first and the looper second, so her sound has to leave a
    loop cue alone: its list of sounds would otherwise be the answer, and no loop."""
    from plume.looper import Looper, parse_loop_cue
    assert parse_loop_cue(text)["op"] == op, "pick a cue the looper knows"
    fx = InputFx(SR, N, "clean")
    assert fx.handle_cue(text) is None
    assert fx.playing == "clean"
    looper = Looper(SR)
    alone = Looper(SR).handle_cue(text)
    reply = fx.handle_cue(text) or looper.handle_cue(text)        # as realtime.lead wires it
    assert reply["op"] == op and reply["ok"] == alone["ok"]


def test_the_echo_is_a_dotted_eighth_on_the_tempo():
    assert inputfx.echo_samples(120, SR) == round(0.75 * 0.5 * SR)
    assert inputfx.echo_samples(40, SR) <= 0.8 * SR       # slow: the dotted sixteenth
    assert inputfx.echo_samples(float("nan"), SR) == inputfx.echo_samples(100, SR)


def test_the_echo_lands_where_the_tempo_says():
    fx = InputFx(SR, N, "dreamy")
    fx.set_bpm(120)
    x = np.zeros(3 * SR, dtype=np.float32)
    at = SR // 2
    x[at:at + 200] = pluck(330.0, 200 / SR)[:200].astype(np.float32) * 0.3
    y = through(fx, x).sum(axis=1)
    d = inputfx.echo_samples(120, SR)
    env = np.abs(y)
    before = env[at + d - 2000:at + d - 300].max()
    on = env[at + d - 50:at + d + 400].max()
    assert on > 2.0 * before


@pytest.mark.parametrize("before,after", [(120, 80), (80, 140)])
def test_the_echo_follows_a_tempo_change(before, after):
    """The band moves tempo mid-take: the next echo lands on the new tempo, not the old."""
    fx = InputFx(SR, N, "dreamy")
    fx.set_bpm(before)
    x = np.zeros(5 * SR, dtype=np.float32)
    at = 2 * SR + 77
    x[at:at + 200] = 0.3 * np.hanning(200).astype(np.float32)
    out = np.zeros((x.size // N * N, 2), dtype=np.float32)
    for i in range(0, out.shape[0], N):
        if i == 188 * N:                        # about a second in, before she plays
            fx.set_bpm(after)
        y = fx.process(x[i:i + N], False)
        out[i:i + N] = x[i:i + N, None] if y is None else y
    env = np.abs(out.sum(axis=1))
    new, old = inputfx.echo_samples(after, SR), inputfx.echo_samples(before, SR)
    on_new = env[at + new - 50:at + new + 400].max()
    on_old = env[at + old - 50:at + old + 400].max()
    assert on_new > 4.0 * on_old


# ------------------------------------------------------------------ the promises

def test_off_is_her_input_untouched():
    fx = InputFx(SR, N, "off")
    x = playing(1.0)
    for i in range(0, x.size - N, N):
        assert fx.process(x[i:i + N], False) is None


@pytest.mark.parametrize("peak_db", [-24.0, -9.0, -3.0], ids=["soft", "normal", "hot"])
@pytest.mark.parametrize("sound", PLAYED)
def test_nothing_jumps_in_loudness(sound, peak_db):
    """At any playing level: a compressor with a fixed makeup gain is as loud as her dry
    input at one level only, so switching to it at another would jump."""
    x = playing(peak_db=peak_db)
    dry = np.repeat(x[:x.size // N * N, None], 2, axis=1)
    fx = InputFx(SR, N, sound)
    fx.set_bpm(92)
    assert abs(lufs(through(fx, x)) - lufs(dry)) <= 1.5


@pytest.mark.parametrize("sound", PLAYED)
def test_no_sound_is_late(sound):
    """The response starts on the sample that came in, bar the tape wobble's floor."""
    fx = InputFx(SR, N, sound)
    x = np.zeros(20 * N, dtype=np.float32)
    at = 12 * N + 100                           # well after the fade in
    x[at] = 0.5
    y = np.abs(through(fx, x).sum(axis=1))
    first = int(np.argmax(y > 0.02 * y.max()))
    allowed = int(np.ceil(inputfx.added_latency_ms(sound, SR) * SR / 1000.0)) + 1
    assert at <= first <= at + allowed
    assert (inputfx.added_latency_ms(sound, SR) == 0.0) == (sound != "lofi")


_ALLOC_PROBE = """
import sys, tracemalloc
import numpy as np
from plume.inputfx import InputFx
n = 1024
fx = InputFx(48000, n, sys.argv[1])
x = (np.random.default_rng(0).standard_normal(n * 40) * 0.05).astype(np.float32)
for i in range(12):
    fx.process(x[i * n:(i + 1) * n], False)
worst = 0
for i in range(12, 40):
    tracemalloc.start()
    fx.process(x[i * n:(i + 1) * n], False)
    worst = max(worst, tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()
print(worst)
"""


@pytest.mark.parametrize("sound", PLAYED)
def test_it_allocates_nothing_in_the_callback(sound):
    """Buffers are made up front: what a buffer of it costs is a few Python objects,
    never an array (at 1024 frames, one float32 buffer is 4 KiB).  Measured in a fresh process:
    tracemalloc counts every thread, and threads other tests left behind would count."""
    import os
    import subprocess
    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(inputfx.__file__))}
    done = subprocess.run([sys.executable, "-c", _ALLOC_PROBE, sound], env=env,
                          capture_output=True, text=True, timeout=120, check=True)
    assert int(done.stdout.strip()) < 4096


@pytest.mark.parametrize("sound", PLAYED)
def test_its_work_does_not_grow_with_the_buffer(sound):
    """No per-sample Python: the same lines run for a buffer of 128 as for 256."""
    def lines(frames):
        fx = InputFx(SR, frames, sound)
        x = (np.random.default_rng(0).standard_normal(frames) * 0.05).astype(np.float32)
        for _ in range(8):
            fx.process(x, False)
        count = [0]

        def trace(frame, event, arg):
            if event == "line":
                count[0] += 1
            return trace
        sys.settrace(trace)
        try:
            fx.process(x, False)
        finally:
            sys.settrace(None)
        return count[0]
    assert lines(128) == lines(256)


@pytest.mark.parametrize("sound", PLAYED)
def test_the_tail_dies_away(sound):
    fx = InputFx(SR, N, sound)
    x = np.zeros(12 * SR, dtype=np.float32)
    x[N * 10:N * 10 + SR] = playing(1.0)
    y = through(fx, x)
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y[-SR:] ** 2)) < 1e-4


@pytest.mark.parametrize("sound", PLAYED)
def test_a_dying_tail_never_turns_subnormal(sound):
    """On x86 (most Windows and Linux machines) arithmetic on subnormal floats is many
    times slower, and a feedback tail decaying in exact digital silence reaches them
    within seconds.  The tails are kept out of that range, so a quiet input never makes
    the callback slow."""
    fx = InputFx(SR, N, sound)
    noise = (np.random.default_rng(0).standard_normal(SR) * 0.2).astype(np.float32)
    for i in range(0, SR, N):
        fx.process(noise[i:i + N], False)
    quiet = np.zeros(N, dtype=np.float32)
    tiny = np.finfo(np.float32).tiny
    for b in range(int(60 * SR / N)):
        fx.process(quiet, False)
        if b % 25 == 0:
            m = fx.mem
            assert not np.any((m != 0) & (np.abs(m) < tiny)), f"subnormal at {b * N / SR:.1f} s"


def test_switching_sounds_has_no_jump():
    fx = InputFx(SR, N, "clean")
    t = np.arange(4 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 196.0 * t)).astype(np.float32)
    out = np.zeros((x.size // N * N, 2), dtype=np.float32)
    for k, i in enumerate(range(0, out.shape[0], N)):
        if k == 200:
            fx.choose("crunch")
        if k == 400:
            fx.choose("off")
        y = fx.process(x[i:i + N], False)
        out[i:i + N] = x[i:i + N, None] if y is None else y
    step = np.abs(np.diff(out[:, 0]))
    # A switch passes through her dry sound for a moment, so around each one no step
    # is bigger than the old sound, the new one, or her dry input make on their own.
    dry = np.abs(np.diff(x)).max()
    for k in (200, 400):
        at = k * N
        around = step[at - N:at + 6 * N].max()
        steady = max(step[at - 60 * N:at - 2 * N].max(), step[at + 20 * N:at + 60 * N].max(), dry)
        assert around <= 1.1 * steady
    assert fx.cur == 0 and fx.process(x[:N], False) is None


# ------------------------------------------------------------------ the guard still wins

@pytest.mark.parametrize("sound", PLAYED)
def test_a_buzz_is_still_ducked_and_rings_on_in_no_tail(sound):
    """A loose cable: the duck mutes her sound just as it mutes her dry input, and the
    tails are cleared, so nothing of the buzz rings after it."""
    t = np.arange(2 * SR) / SR
    hum = np.clip(1.4 * np.sin(2 * np.pi * 50.0 * t), -1, 1).astype(np.float32)
    x = np.concatenate([np.zeros(N * 8, np.float32), hum, np.zeros(SR, np.float32)])
    guard = inputguard.InputGuard(SR)
    fx = InputFx(SR, N, sound)
    n = x.size // N * N
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(0, n, N):
        realtime.mix_monitor(out[i:i + N], x[i:i + N], guard.slot, fx)
        guard.feed(x[i:i + N])
    start = N * 8
    assert np.abs(out[start + int(0.3 * SR):start + 2 * SR]).max() < 1e-3
    after = start + 2 * SR
    assert np.abs(out[after:after + int(0.4 * SR)]).max() < 1e-3   # still held, and no tail


@pytest.mark.parametrize("sound", PLAYED)
def test_the_ceiling_holds_after_any_sound(sound):
    t = np.arange(SR) / SR
    band = (0.95 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = np.sign(np.sin(2 * np.pi * 82.0 * t)).astype(np.float32)
    fx = InputFx(SR, N, sound)
    slot = inputguard.monitor_slot()
    out = np.zeros((SR // N * N, 2), dtype=np.float32)
    for i in range(0, out.shape[0], N):
        out[i:i + N, 0] = band[i:i + N]
        out[i:i + N, 1] = band[i:i + N]
        realtime.mix_monitor(out[i:i + N], x[i:i + N], slot, fx)
    assert np.max(np.abs(out)) <= realtime.MONITOR_CEILING + 1e-6
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("sound", PLAYED)
def test_one_bad_sample_does_not_ring_on(sound, bad):
    """A driver glitch: one sample that is not a number.  Every feedback loop would keep
    it for the rest of the take, and the band's output with it.  It must cost no more
    than the buffer it came in."""
    t = np.arange(400 * N) / SR
    band = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = playing(400 * N / SR + 0.1)[:400 * N].copy()
    x[20 * N + 7] = bad
    fx = InputFx(SR, N, sound)
    slot = inputguard.monitor_slot()
    out = np.zeros((400 * N, 2), dtype=np.float32)
    for i in range(0, out.shape[0], N):
        out[i:i + N, 0] = band[i:i + N]
        out[i:i + N, 1] = band[i:i + N]
        realtime.mix_monitor(out[i:i + N], x[i:i + N], slot, fx)
    assert np.all(np.isfinite(out[21 * N:])), "the bad sample rang on"
    assert np.all(np.isfinite(out)), "her sound passed the bad sample on"
    assert np.all(np.isfinite(fx.mem))
    assert all(np.isfinite(v) for v in (fx.env, fx.cg, fx.mg))
    assert np.abs(out[30 * N:]).max() > 0.05      # and her sound is still there after it


def test_mix_monitor_without_a_sound_is_as_it_was():
    t = np.arange(N) / SR
    band = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    o = np.stack([band, band], axis=1)
    realtime.mix_monitor(o, x, inputguard.monitor_slot(), InputFx(SR, N, "off"))
    np.testing.assert_allclose(o[:, 0], band + x, atol=1e-6)
    np.testing.assert_allclose(o[:, 1], band + x, atol=1e-6)


def test_no_monitor_means_no_sound(capsys):
    assert realtime._input_sound("dreamy", False, "lofi") is None
    assert "Hear me" in capsys.readouterr().err
    assert realtime._input_sound(None, False, "lofi") is None
    assert realtime._input_sound("off", False, "lofi") is None
    assert capsys.readouterr().err == ""
    with pytest.raises(realtime.RealtimeError):
        realtime._input_sound("wobbly", True, "lofi")
    fx = realtime._input_sound("match", True, "lofi")
    assert fx is not None and fx.playing == "lofi"
    assert "recording stays dry" in capsys.readouterr().err


@pytest.mark.parametrize("sound", [None, "off"])
def test_hear_me_with_sound_off_can_still_be_given_a_sound(sound, capsys):
    """Off is where Sound starts, and the guide says a take can be told `sound dreamy`.
    So Hear me always has her sound within reach, playing nothing until it is asked."""
    fx = realtime._input_sound(sound, True, "lofi")
    assert fx is not None and fx.playing == "off"
    assert capsys.readouterr().err == ""                  # nothing is on: nothing to say
    x = playing(0.5)
    for i in range(0, x.size - N, N):
        assert fx.process(x[i:i + N], False) is None      # her input as it comes in
    reply = fx.handle_cue("sound dreamy")
    assert reply["ok"] is True and fx.playing == "dreamy"


# ------------------------------------------------------------------ a take, on the fake device

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")      # the FAKE device only


def test_human_wav_stays_dry_while_she_hears_her_sound(tmp_path, consent):
    """plume's ears read human.wav, so it is her input exactly; only the monitor changes."""
    import json

    import soundfile as sf
    from conftest import fake_audio, patched_module, short_socket_path

    from plume import roles as roles_mod
    guitar = playing(2.0, peak_db=-12.0)
    fake = fake_audio(guitar, devices=RIG, human_channel=1, capture=True)
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.05, monitor=True, tap_start=True, direct=True,
                      record_dir=str(tmp_path / "rec"), socket_path=short_socket_path("sndl"),
                      input_device="audio interface", input_channel=2, sound="dreamy")
    human, _ = sf.read(str(tmp_path / "rec" / "human.wav"), dtype="float32")
    assert human.size > SR
    expect = np.resize(guitar, human.size)
    np.testing.assert_allclose(human, expect, atol=1e-4)
    out = fake.stream.output
    # what she heard is her sound: in stereo, and not her dry input
    assert not np.array_equal(out[:, 0], out[:, 1])
    assert json.loads((tmp_path / "rec" / "session.json").read_text())["sound"] == "dreamy"


def test_a_sound_cue_reaches_a_running_take(tmp_path, consent):
    import threading

    from conftest import fake_audio, patched_module, short_socket_path, wait_until

    from plume import roles as roles_mod
    sock = short_socket_path("sndc")
    fake = fake_audio(playing(2.0), devices=RIG, human_channel=1)
    replies = []

    def take():
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.08, monitor=True, tap_start=True, direct=True,
                      socket_path=sock, input_device="audio interface", input_channel=2,
                      sound="clean")
    with patched_module("sounddevice", fake):
        th = threading.Thread(target=take)
        th.start()

        def asked():
            try:
                replies.append(realtime.send_cue("sound lofi", socket_path=sock))
                return True
            except realtime.RealtimeError:
                return False
        assert wait_until(asked, timeout=10.0)
        loop = realtime.send_cue("make a 4 bar loop", socket_path=sock)
        th.join(30.0)
    assert replies[-1]["ok"] is True and replies[-1]["sound"] == "lofi"
    assert loop.get("op") != "sound"                  # other cues still reach the looper


@pytest.mark.parametrize("sound", [None, "off"])
def test_a_take_started_with_sound_off_takes_a_sound_cue(tmp_path, consent, sound):
    """The default take: Hear me on, Sound off.  `sound dreamy` has to change her sound,
    not be queued for the band as a cue it will not understand.  A loop cue that says
    "sound" is still the looper's."""
    import json
    import threading

    from conftest import fake_audio, patched_module, short_socket_path, wait_until

    from plume import roles as roles_mod
    sock = short_socket_path("sndo")
    fake = fake_audio(playing(2.0), devices=RIG, human_channel=1)
    replies = []
    extra = {} if sound is None else {"sound": sound}

    def take():
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.08, monitor=True, tap_start=True, direct=True,
                      record_dir=str(tmp_path / "rec"), socket_path=sock,
                      input_device="audio interface", input_channel=2, **extra)
    with patched_module("sounddevice", fake):
        th = threading.Thread(target=take)
        th.start()

        def asked():
            try:
                replies.append(realtime.send_cue("sound dreamy", socket_path=sock))
                return True
            except realtime.RealtimeError:
                return False
        assert wait_until(asked, timeout=10.0)
        loop = realtime.send_cue("loop this sound", socket_path=sock)
        th.join(30.0)
    assert replies[-1]["ok"] is True and replies[-1].get("op") == "sound"
    assert replies[-1]["sound"] == "dreamy" and replies[-1]["queued"] == 0
    assert loop["op"] == "record"
    # the take's record says what it started with: no sound
    assert json.loads((tmp_path / "rec" / "session.json").read_text())["sound"] is None


def test_the_cli_flag_and_the_environment(monkeypatch):
    from plume import cli
    parse = cli._build_parser().parse_args
    monkeypatch.delenv("PLUME_SOUND", raising=False)
    assert "sound" not in cli._route_kwargs(parse(["lead", "lofi", "--monitor"]))
    assert cli._route_kwargs(parse(["lead", "lofi", "--monitor", "--sound", "crunch"]))["sound"] == "crunch"
    assert cli._route_kwargs(parse(["jam", "--style", "lofi", "--monitor", "--sound", "lofi"]))["sound"] == "lofi"
    monkeypatch.setenv("PLUME_SOUND", "dreamy")
    assert cli._route_kwargs(parse(["lead", "lofi", "--monitor"]))["sound"] == "dreamy"
    assert "sound" not in cli._route_kwargs(parse(["lead", "lofi"]))      # no Hear me, no sound
