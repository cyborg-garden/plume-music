"""Lining a take up with a video of it, and the tool that reads and writes videos.

No device, no sound, no network.  The lining-up tests are pure numbers at 8 kHz.
The ones that touch a video make a tiny one with ffmpeg and are skipped, with the
reason shown, only where there is no ffmpeg at all.  On CI there must be one.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from plume import takevideo
from video_fixture import (CAMERA_SECRETS, FFMPEG, PLACE, described, float_sound, heard_by_camera,
                           make_camera_video, make_video, needs_ffmpeg, placed, playing, talking)

R = takevideo.ALIGN_RATE
#: How close the start must be found.  Half a video frame at 100 frames a second,
#: and far under what an ear hears as apart.
TOLERANCE_S = 0.005


def camera(take, start, length, *, seed=3, noise=0.01, talk=0.15):
    """What the camera heard: the playing across a room, talking all through, and hiss."""
    rng = np.random.default_rng(seed)
    return (placed(heard_by_camera(take, R), start, length) + talking(length / R, R, seed, talk)
            + rng.normal(0.0, noise, length)).astype(np.float32)


# ------------------------------------------------------------------- lining up

def test_a_take_is_found_inside_a_longer_video():
    take = playing(20.0, R)
    start = 58569                                   # 7.321 s in
    found = takevideo.align(camera(take, start, 40 * R), take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.confident and found.why is None


def test_a_video_much_longer_than_the_take():
    take = playing(12.0, R, seed=5)
    start = 301 * R + 1234
    found = takevideo.align(camera(take, start, 600 * R), take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.confident


def test_a_video_that_starts_in_the_middle_of_the_take():
    take = playing(30.0, R, seed=7)
    start = -9 * R - 321                            # the camera started nine seconds late
    found = takevideo.align(camera(take, start, 40 * R), take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.start_s < 0 and found.confident


def test_a_take_longer_than_the_video():
    take = playing(40.0, R, seed=9)
    start = -5 * R - 77                             # the video is the middle fifteen seconds
    found = takevideo.align(camera(take, start, 15 * R), take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.confident


def test_a_video_that_ends_before_the_take_does():
    take = playing(30.0, R, seed=11)
    start = 6 * R + 5
    found = takevideo.align(camera(take, start, 20 * R), take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.confident


def test_loud_talking_and_hiss_over_the_playing_do_not_move_it():
    take = playing(20.0, R, seed=13)
    start = 4 * R + 4001
    heard = camera(take, start, 30 * R, noise=0.05, talk=0.5)       # louder than the playing
    found = takevideo.align(heard, take.astype(np.float32))
    assert found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)
    assert found.confident


def test_other_playing_is_not_trusted():
    take = playing(20.0, R, seed=15)
    heard = camera(playing(20.0, R, seed=16), 5 * R, 40 * R)        # someone else's song
    found = takevideo.align(heard, take.astype(np.float32))
    assert not found.confident and found.why is None
    assert found.confidence < takevideo.CONFIDENT


def test_talking_alone_is_not_trusted():
    take = playing(20.0, R, seed=17)
    found = takevideo.align(talking(40.0, R, 18).astype(np.float32), take.astype(np.float32))
    assert not found.confident


def test_one_knock_in_each_is_not_a_song():
    # A tap on the camera and one loud note: as sharp a match as there is, and worth nothing.
    take = np.zeros(20 * R, dtype=np.float32)
    take[10 * R] = 0.9
    rng = np.random.default_rng(19)
    heard = rng.normal(0.0, 0.001, 40 * R).astype(np.float32)
    heard[25 * R] = 0.9
    assert not takevideo.align(heard, take).confident


@pytest.mark.parametrize("heard", [np.zeros(0, dtype=np.float32), np.zeros(20 * R, dtype=np.float32),
                                   np.full(20 * R, 1e-5, dtype=np.float32)])
def test_a_video_with_no_usable_sound_says_so(heard):
    found = takevideo.align(heard, playing(10.0, R).astype(np.float32))
    assert found.why == "silent" and not found.confident and found.start_s == 0.0


def test_a_silent_take_is_not_trusted():
    heard = camera(playing(10.0, R), 2 * R, 20 * R)
    assert not takevideo.align(heard, np.zeros(10 * R, dtype=np.float32)).confident


def test_a_slow_camera_clock_is_measured_and_does_not_move_the_start():
    take = playing(120.0, R, seed=21)
    heard = heard_by_camera(take, R)
    # The camera's clock runs 100 parts in a million slow, more than any phone: its copy is that much longer.
    stretched = np.interp(np.arange(int(len(heard) * 1.0001)) / 1.0001, np.arange(len(heard)), heard)
    start = 3 * R
    found = takevideo.align(placed(stretched, start, 130 * R).astype(np.float32), take.astype(np.float32))
    assert found.confident
    assert found.drift_ppm == pytest.approx(100.0, abs=25.0)
    # 100 ppm over two minutes is 12 ms end to end; the start found sits inside that.
    assert found.start_s == pytest.approx(start / R, abs=0.015)


def test_a_short_take_has_no_drift_to_report():
    take = playing(20.0, R)
    assert takevideo.align(camera(take, R, 30 * R), take.astype(np.float32)).drift_ppm is None


def test_where_the_player_first_plays():
    take = np.zeros(10 * R, dtype=np.float32)
    take[3 * R:4 * R] = 0.5
    assert takevideo.first_sound_s(take) == pytest.approx(3.0, abs=0.001)
    assert takevideo.first_sound_s(np.zeros(100, dtype=np.float32)) == 0.0
    assert takevideo.first_sound_s(np.zeros(0, dtype=np.float32)) == 0.0


# ---------------------------------------------------------------- laying it in

def ramp(frame, count):
    """A take whose mix is its own frame number, so where each sample came from shows."""
    return np.arange(frame + 1, frame + 1 + count).astype("<i2").tobytes()


def test_the_take_is_laid_in_at_its_place_with_silence_before():
    got = np.frombuffer(takevideo.laid_in(ramp, 5, 0, 8), dtype="<i2").tolist()
    assert got == [0, 0, 0, 0, 0, 1, 2, 3]
    assert np.frombuffer(takevideo.laid_in(ramp, 5, 8, 4), dtype="<i2").tolist() == [4, 5, 6, 7]
    assert np.frombuffer(takevideo.laid_in(ramp, 100, 0, 4), dtype="<i2").tolist() == [0, 0, 0, 0]


def test_a_take_that_began_before_the_video_starts_part_way_in():
    assert np.frombuffer(takevideo.laid_in(ramp, -10, 0, 3), dtype="<i2").tolist() == [11, 12, 13]


def test_the_band_falls_later_than_the_player_by_the_round_trip():
    assert takevideo.video_frame_of(2.0, 12000, 48000) == 96000 + 12000
    assert takevideo.video_frame_of(-1.0, 0, 48000) == -48000


# ------------------------------------------------------------------ whose file

def test_a_video_plume_made_is_known_by_how_it_ends(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"\x00\x00\x00\x08ftyp" + takevideo.stamp({"start_s": 1.5}))
    assert takevideo.is_own_video(path)


@pytest.mark.parametrize("body", [b"", b"just a film", b"plVd", b"\x00\x00\x00\x10freeplVd{}xx",
                                  b"\x00\x00\x00\x0efreeplVd[1]", b"\x00\x00\x00\x12freeplVd{\"r\":1}"])
def test_anything_else_is_the_players(tmp_path, body):
    path = tmp_path / "video.mp4"
    path.write_bytes(body)
    assert not takevideo.is_own_video(path)


def test_a_video_of_plumes_that_was_added_to_is_the_players(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"film" + takevideo.stamp({}) + b"more")
    assert not takevideo.is_own_video(path)
    assert not takevideo.is_own_video(tmp_path / "missing.mp4")
    assert not takevideo.is_own_video(tmp_path)


# -------------------------------------------------------------------- the tool

class _Packaged(types.ModuleType):
    def __init__(self, exe):
        super().__init__("imageio_ffmpeg")
        self.exe = exe

    def get_ffmpeg_exe(self):
        if self.exe is None:
            raise RuntimeError("no binary for this machine")
        return self.exe


def test_the_packaged_ffmpeg_is_used_first(tmp_path, monkeypatch):
    exe = tmp_path / "packaged"
    exe.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Packaged(str(exe)))
    monkeypatch.setattr(takevideo.shutil, "which", lambda name: "/somewhere/on/the/path")
    assert takevideo.find_ffmpeg() == str(exe)


def test_one_on_the_path_is_used_when_the_package_has_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Packaged(None))
    monkeypatch.setattr(takevideo.shutil, "which", lambda name: "/somewhere/on/the/path")
    assert takevideo.find_ffmpeg() == "/somewhere/on/the/path"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)          # not installed at all
    assert takevideo.find_ffmpeg() == "/somewhere/on/the/path"


def test_no_ffmpeg_anywhere_is_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(takevideo.shutil, "which", lambda name: None)
    assert takevideo.find_ffmpeg() is None


def test_ci_has_an_ffmpeg_so_the_video_tests_really_run():
    """A skip is honest on a laptop with no ffmpeg.  On CI it would hide every test below."""
    if not os.environ.get("CI"):
        pytest.skip("only CI must have one")
    assert FFMPEG is not None, "CI installs imageio-ffmpeg with plume; the video tests must not skip"


@needs_ffmpeg
def test_a_video_is_read_for_what_it_holds(tmp_path):
    with_sound = make_video(tmp_path / "a.mp4", seconds=2.0, sound=playing(2.0, 48000))
    without = make_video(tmp_path / "b.mp4", seconds=3.0)
    assert takevideo.probe(FFMPEG, with_sound) == takevideo.Probe(pytest.approx(2.0, abs=0.3), True, "mpeg4")
    assert takevideo.probe(FFMPEG, without) == takevideo.Probe(pytest.approx(3.0, abs=0.3), False, "mpeg4")


@needs_ffmpeg
def test_what_is_not_a_video_is_refused(tmp_path):
    import soundfile as sf

    words = tmp_path / "words.mp4"
    words.write_text("not a film at all\n" * 50)
    song = tmp_path / "song.wav"
    sf.write(song, playing(1.0, 8000), 8000)
    for path in (words, song, tmp_path / "missing.mp4"):
        with pytest.raises(takevideo.VideoError) as refused:
            takevideo.probe(FFMPEG, path)
        assert refused.value.why == "not_video"


@needs_ffmpeg
def test_a_videos_sound_comes_back_on_the_videos_clock(tmp_path):
    sound = np.zeros(3 * 48000)
    sound[48000] = 0.9                                   # a click one second in
    path = make_video(tmp_path / "a.mp4", seconds=3.0, sound=sound)
    heard = takevideo.decode(FFMPEG, path)
    assert len(heard) == pytest.approx(3 * R, abs=R // 4)
    assert int(np.argmax(np.abs(heard))) / R == pytest.approx(1.0, abs=TOLERANCE_S)
    with open(path, "rb") as f:                          # and piped in, it is the same sound
        assert np.array_equal(takevideo.decode(FFMPEG, f), heard)


def test_the_copy_keeps_a_phones_picture_playable_everywhere():
    """A phone's newer picture format is only played by some players under one of its two names."""
    plain = takevideo.write_command("ffmpeg", "in.mov", "out.tmp", rate=48000, duration_s=2.0, keep_sound=True,
                                    codec="h264")
    newer = takevideo.write_command("ffmpeg", "in.mov", "out.tmp", rate=48000, duration_s=2.0, keep_sound=True,
                                    codec="hevc")
    assert "-tag:v" not in plain
    assert newer[newer.index("-tag:v") + 1] == "hvc1"
    for cmd in (plain, newer):
        assert cmd[cmd.index("-c:v") + 1] == "copy" and cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[cmd.index("-b:a") + 1] == "192k" and cmd[cmd.index("-ac", cmd.index("-c:a")) + 1] == "2"
        assert cmd[cmd.index("-map_metadata") + 1] == "-1" and "+faststart" in cmd
        assert cmd[-3:] == ["-f", "mp4", "out.tmp"]


@needs_ffmpeg
def test_the_kind_of_picture_is_read(tmp_path):
    path = make_video(tmp_path / "a.mp4", seconds=1.0)
    assert takevideo.probe(FFMPEG, path).codec == "mpeg4"


@needs_ffmpeg
def test_stopping_part_way_stops_the_tool(tmp_path):
    path = make_video(tmp_path / "a.mp4", seconds=2.0, sound=playing(2.0, 48000))
    with pytest.raises(takevideo.Cancelled):
        takevideo.decode(FFMPEG, path, cancelled=lambda: True)
    out = tmp_path / "out.mp4"
    with pytest.raises(takevideo.Cancelled):
        takevideo.write_video(FFMPEG, path, out, mixed=lambda frame, count: bytes(2 * count), rate=48000,
                              band_at=0, duration_s=2.0, keep_sound=True, cancelled=lambda: True)


# ------------------------------------------------------------------ a loop

def phrase(seed, jitter=None):
    """Four seconds of the same eight notes.  ``jitter`` plays them as a hand would: each
    note a little early or late and a little louder or softer, never twice the same."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.1, 3.4, 8))
    pitches = rng.choice((110.0, 146.8, 164.8, 196.0, 220.0, 293.7, 329.6), 8)
    out = np.zeros(4 * R)
    for at, pitch in zip(times, pitches):
        level = 0.2
        if jitter is not None:
            at, level = at + jitter.uniform(-0.02, 0.02), level * jitter.uniform(0.7, 1.3)
        t = np.arange(int(0.5 * R)) / R
        note = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in (1, 2, 3)) * np.exp(-t / 0.2)
        begin = int(at * R)
        out[begin:begin + len(note)] += (note * np.minimum(1.0, t / 0.003) * level)[:len(out) - begin]
    return out


def test_an_exact_loop_that_also_sounds_outside_the_take_is_not_trusted():
    """A looper plays the same samples every time round.  Where the take sits among
    ten identical rounds cannot be heard, so plume must ask and not guess."""
    one = phrase(51)
    take = np.tile(one, 5)                           # rounds three to seven of ten
    start = 2 * R + 2 * len(one)
    heard = (placed(heard_by_camera(np.tile(one, 10), R), 2 * R, 50 * R)
             + talking(50.0, R, 52, 0.05)).astype(np.float32)
    found = takevideo.align(heard, take.astype(np.float32))
    assert not found.confident, f"placed {found.start_s - start / R:+.2f} s off at confidence {found.confidence:.2f}"


def test_a_loop_played_by_hand_is_still_found():
    hand = np.random.default_rng(53)
    take = np.concatenate([phrase(54, hand) for _ in range(5)])
    start = 3 * R + 777
    found = takevideo.align(camera(take, start, 30 * R), take.astype(np.float32))
    assert found.confident and found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)


def test_a_loop_played_by_hand_is_found_among_more_rounds_of_it():
    hand = np.random.default_rng(55)
    rounds = [phrase(56, hand) for _ in range(10)]
    take = np.concatenate(rounds[3:8])
    start = 2 * R + 3 * 4 * R
    heard = (placed(heard_by_camera(np.concatenate(rounds), R), 2 * R, 50 * R)
             + talking(50.0, R, 57, 0.05)).astype(np.float32)
    found = takevideo.align(heard, take.astype(np.float32))
    assert found.confident and found.start_s == pytest.approx(start / R, abs=TOLERANCE_S)


# ------------------------------------------------------------------- headroom

@needs_ffmpeg
@pytest.mark.parametrize("keep_sound", [True, False])
def test_a_loud_take_does_not_clip_once_it_is_a_video(tmp_path, keep_sound):
    """The encoder's copy of a sound overshoots the sound it was given.  A mix handed
    over at full scale comes back over it, and a player clips that."""
    rate, seconds = 48000, 4.0
    rng = np.random.default_rng(61)
    # As loud as a mix gets: playing driven hard into full scale, and bursts of hiss at it.
    loud = np.clip(playing(seconds, rate, seed=62) * 20.0, -1.0, 1.0)
    loud[rate:2 * rate] = np.clip(rng.normal(0.0, 0.6, rate), -1.0, 1.0)
    pcm = (loud * 32767).astype("<i2")
    source = make_video(tmp_path / "in.mp4", seconds=seconds, sound=np.clip(talking(seconds, rate, 63, 0.6), -1, 1))
    out = tmp_path / "out.mp4"

    def mixed(frame, count):
        got = pcm[frame:frame + count]
        return got.tobytes() + bytes(2 * (count - len(got)))

    takevideo.write_video(FFMPEG, source, out, mixed=mixed, rate=rate, band_at=0, duration_s=seconds,
                          keep_sound=keep_sound)
    sound = float_sound(out)
    assert float(np.max(np.abs(sound))) <= 1.0
    assert float(np.sqrt(np.mean(sound ** 2))) > 0.2            # and it is still loud, not turned down to nothing


# -------------------------------------------------- what the camera wrote about itself

#: What each kind of file must be carrying going in, or the test proves nothing.
MUST_CARRY = {"mp4": ("secret-title", "secret-artist", "secret-comment", "secret-picture-handler",
                      "secret-sound-handler", "secret-chapter", "secret-subtitle"),
              "mov": CAMERA_SECRETS[:8] + ("secret-chapter", "secret-subtitle")}
#: What ffmpeg's account of a file says when it holds each kind of thing.
TELLS = ("Chapter", "Subtitle", "Data:", "timecode", "creation_time", "location", "(fra)", "secret")


@needs_ffmpeg
@pytest.mark.parametrize("keep_sound", [True, False])
@pytest.mark.parametrize("container", ["mp4", "mov"])
def test_nothing_the_camera_wrote_about_itself_is_in_the_saved_video(tmp_path, container, keep_sound):
    """Where it was filmed, on what, when, and every name on every track.  A phone writes
    all of it, and a video shared with a band in it must carry none of it."""
    source = make_camera_video(tmp_path / "camera", container)
    before, told = source.read_bytes(), described(source)
    for secret in MUST_CARRY[container]:
        assert secret.encode() in before, f"the made-up video does not carry {secret}"
    # An mp4 keeps the place in a box of numbers, so the text of it is not there to find.
    assert b"loci" in before and PLACE in told and "rotation of 90" in told
    assert (PLACE.encode() in before) is (container == "mov")
    for tell in TELLS:
        assert tell in told, f"the made-up video has no {tell}"
    found = takevideo.probe(FFMPEG, source)
    out = tmp_path / "video.mp4"
    takevideo.write_video(FFMPEG, source, out, mixed=lambda frame, count: bytes(2 * count), rate=48000,
                          band_at=0, duration_s=found.duration_s, keep_sound=keep_sound, codec=found.codec)
    after, told = out.read_bytes(), described(out)
    # (The encoder's own name is not looked for: the made-up picture carries it inside its
    # frames, and the frames are copied as they are.)
    for secret in CAMERA_SECRETS + ("loci", PLACE, PLACE.strip("+/"), "2001", "01:02:03", "FFMP", "Lavf"):
        assert secret.encode() not in after, f"{secret} is in the saved video"
    for tell in TELLS + ("(deu)", "(eng)"):
        assert tell not in told, f"the saved video has a {tell}"
    tracks = [line.split(": ", 2)[1] for line in told.splitlines() if line.lstrip().startswith("Stream #")]
    assert tracks == ["Video", "Audio"]
    # The one thing kept: which way up the picture is.
    assert "rotation of 90" in told
