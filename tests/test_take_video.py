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
                           make_camera_video, make_video, needs_ffmpeg, placed, playing, stalled_tools, talking)

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


# ------------------------------------------------------- which picture, and which tool

def make_with_cover(tmp_path):
    """A video whose first picture track is a still: cover art, ahead of the film."""
    import subprocess

    film = make_video(tmp_path / "film.mp4", seconds=2.0, sound=playing(2.0, 48000))
    cover = tmp_path / "cover.jpg"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=32x32",
                    "-frames:v", "1", str(cover)], check=True, timeout=120, stdin=subprocess.DEVNULL)
    out = tmp_path / "covered.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(cover), "-i", str(film),
                    "-map", "0:v", "-map", "1:v", "-map", "1:a", "-c", "copy", "-disposition:v:0", "attached_pic",
                    "-f", "mp4", str(out)], check=True, timeout=120, stdin=subprocess.DEVNULL)
    return film, out


@needs_ffmpeg
def test_the_film_is_copied_and_never_its_cover_art(tmp_path):
    from video_fixture import picture_md5

    film, covered = make_with_cover(tmp_path)
    found = takevideo.probe(FFMPEG, covered)
    told = [line for line in described(covered).splitlines() if "Stream #0:" in line and "Video" in line]
    assert len(told) == 2, "the made-up video has no cover art"
    film_track = next(i for i, line in enumerate(described(covered).splitlines()) if "mpeg4" in line)
    assert found.codec == "mpeg4"
    assert f"Stream #0:{found.picture}" in described(covered).splitlines()[film_track]
    out = tmp_path / "video.mp4"
    takevideo.write_video(FFMPEG, covered, out, mixed=lambda frame, count: bytes(2 * count), rate=48000,
                          band_at=0, duration_s=found.duration_s, keep_sound=True, codec=found.codec,
                          picture=found.picture)
    assert picture_md5(out) == picture_md5(film)
    assert takevideo.probe(FFMPEG, out).duration_s == pytest.approx(2.0, abs=0.3)


def test_a_still_ahead_of_the_film_is_passed_over():
    """Where the cover art sits depends on the kind of file.  When it comes first, the
    film is still the track that is copied."""
    told = """Input #0, matroska,webm, from 'in.mkv':
  Duration: 00:00:07.50, start: 0.000000, bitrate: 900 kb/s
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 600x600, 90k tbr, 90k tbn (attached pic)
  Stream #0:1(und): Audio: aac (LC), 48000 Hz, stereo, fltp (default)
  Stream #0:2(und): Video: h264 (High), yuv420p, 1280x720, 30 fps, 30 tbr, 1k tbn (default)
"""
    found = takevideo.read_probe(told)
    assert found == takevideo.Probe(7.5, True, "h264", 2)
    cmd = takevideo.write_command("ffmpeg", "in.mkv", "out.tmp", rate=48000, duration_s=7.5, keep_sound=True,
                                  codec=found.codec, picture=found.picture)
    assert "0:2" in cmd and "0:v:0" not in cmd
    with pytest.raises(takevideo.VideoError):
        takevideo.read_probe(told.replace("h264 (High)", "vp8"))
    with pytest.raises(takevideo.VideoError):                 # a song with its cover is not a film
        takevideo.read_probe("\n".join(told.splitlines()[:4]))


@needs_ffmpeg
def test_a_picture_that_cannot_be_copied_into_the_saved_file_is_refused_at_once(tmp_path):
    import subprocess

    odd = tmp_path / "odd.avi"                        # a film, but in a form an mp4 cannot hold as it is
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=black:s=32x32:r=5:d=1", "-c:v", "huffyuv", str(odd)], check=True, timeout=120,
                   stdin=subprocess.DEVNULL)
    with pytest.raises(takevideo.VideoError) as refused:
        takevideo.probe(FFMPEG, odd)
    assert refused.value.why == "not_video"


@pytest.mark.parametrize("codec,fits", [("h264", True), ("hevc", True), ("mpeg4", True), ("av1", True),
                                        ("vp9", True), ("vp8", False), ("theora", False), ("huffyuv", False),
                                        ("prores", False), ("", False)])
def test_which_pictures_an_mp4_can_hold_as_they_are(codec, fits):
    assert takevideo.can_copy(codec) is fits


def fake_tool(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_an_ffmpeg_in_the_folder_plume_runs_in_is_never_the_one_used(tmp_path, monkeypatch):
    """Windows looks in the current folder before the PATH, and a PATH can name it too.
    A take folder, or a download, must not be able to supply the tool plume runs."""
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    here, empty = tmp_path / "here", tmp_path / "empty"
    fake_tool(here)
    empty.mkdir()
    monkeypatch.chdir(here)
    for path in (str(empty), "." + os.pathsep + str(empty), os.pathsep + str(empty), "", "sub" + os.pathsep + "."):
        monkeypatch.setenv("PATH", path)
        assert takevideo.find_ffmpeg() is None, path


def test_an_ffmpeg_on_the_path_proper_is_found_by_its_whole_address(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    real = fake_tool(tmp_path / "tools")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "." + os.pathsep + str(tmp_path / "tools"))
    found = takevideo.find_ffmpeg()
    assert found is not None and os.path.isabs(found) and os.path.samefile(found, real)
    # The folder plume runs in is fine when the PATH names it in full: that is a choice someone made.
    monkeypatch.chdir(tmp_path / "tools")
    assert os.path.samefile(takevideo.find_ffmpeg(), real)


# ------------------------------------------------------------ a tool that hangs

@pytest.fixture
def stalled(monkeypatch):
    """Every tool plume starts hangs for ever: real children, all ended by the end of the
    test, and the test fails rather than hangs if plume stops ending them itself."""
    yield from stalled_tools(monkeypatch)


@pytest.fixture
def no_patience(monkeypatch):
    """The time a tool is given, cut to a moment, so a test of running out of it need not wait."""
    monkeypatch.setattr(takevideo, "_TIME_BASE_S", 0.05)
    monkeypatch.setattr(takevideo, "_TIME_PER_S", 0.0)


def after(n):
    """A cancel that comes once the work has been asked ``n`` times whether to go on."""
    asked = iter([False] * n)
    return lambda: next(asked, True)


def silence(frame, count):
    return bytes(2 * count)


def all_stopped(started):
    return bool(started) and all(proc.poll() is not None for proc in started)


def test_a_tool_is_given_time_for_the_media_it_has(monkeypatch):
    assert takevideo.time_limit(0.0) == takevideo._TIME_BASE_S
    assert takevideo.time_limit(600.0) == takevideo._TIME_BASE_S + 600.0 * takevideo._TIME_PER_S
    assert takevideo.time_limit(600.0) > takevideo.time_limit(60.0) >= 60.0
    assert takevideo.time_limit(None) == takevideo.time_limit(takevideo.MAX_VIDEO_S)


def test_a_tool_that_hangs_reading_a_file_is_stopped(tmp_path, stalled, no_patience):
    with pytest.raises(takevideo.VideoError) as failed:
        takevideo.probe("ffmpeg", tmp_path / "a.mp4")
    assert failed.value.why == "failed" and all_stopped(stalled)


def test_a_tool_that_hangs_decoding_is_stopped(tmp_path, stalled, no_patience):
    with pytest.raises(takevideo.VideoError) as failed:
        takevideo.decode("ffmpeg", tmp_path / "a.mp4", duration_s=5.0)
    assert failed.value.why == "failed" and all_stopped(stalled)
    with open(__file__, "rb") as f, pytest.raises(takevideo.VideoError):          # and when the file is piped in
        takevideo.decode("ffmpeg", f, duration_s=5.0)
    assert all_stopped(stalled)


def test_a_tool_that_hangs_writing_is_stopped(tmp_path, stalled, no_patience):
    with pytest.raises(takevideo.VideoError) as failed:
        takevideo.write_video("ffmpeg", tmp_path / "a.mp4", tmp_path / "out.mp4", mixed=silence, rate=48000,
                              band_at=0, duration_s=30.0, keep_sound=True)
    assert failed.value.why == "failed" and all_stopped(stalled)


def test_a_tool_that_hangs_measuring_is_stopped(tmp_path, stalled, no_patience):
    with pytest.raises(takevideo.VideoError):
        takevideo.loudest("ffmpeg", tmp_path / "a.mp4", duration_s=5.0)
    assert all_stopped(stalled)


def test_cancel_stops_a_tool_that_hangs_wherever_it_hangs(tmp_path, stalled):
    with pytest.raises(takevideo.Cancelled):
        takevideo.probe("ffmpeg", tmp_path / "a.mp4", cancelled=after(2))
    with pytest.raises(takevideo.Cancelled):
        takevideo.decode("ffmpeg", tmp_path / "a.mp4", cancelled=after(2))
    with pytest.raises(takevideo.Cancelled):
        takevideo.write_video("ffmpeg", tmp_path / "a.mp4", tmp_path / "out.mp4", mixed=silence, rate=48000,
                              band_at=0, duration_s=30.0, keep_sound=False, cancelled=after(2))
    with pytest.raises(takevideo.Cancelled):
        takevideo.loudest("ffmpeg", tmp_path / "a.mp4", cancelled=after(2))
    assert len(stalled) == 4 and all_stopped(stalled)


def test_the_sound_is_handed_over_a_decibel_under_full_scale_and_kept_there(monkeypatch):
    """About 1 dB of room, and a peak that comes back over half a decibel under full scale
    is written again.  Pinned, so neither can drift away quietly."""
    written, peaks = [], []

    def write_once(*a, gain, **kw):
        written.append(gain)

    monkeypatch.setattr(takevideo, "_write_once", write_once)
    monkeypatch.setattr(takevideo, "loudest", lambda *a, **kw: peaks.pop(0))

    def gains(*heard):
        written.clear()
        peaks[:] = list(heard)
        takevideo.write_video("ffmpeg", "in.mp4", "out.mp4", mixed=silence, rate=48000, band_at=0,
                              duration_s=1.0, keep_sound=True)
        return list(written)

    one_db = 10 ** (-1 / 20)
    assert gains(0.80) == [pytest.approx(one_db, abs=0.002)]
    assert gains(0.94) == [pytest.approx(one_db, abs=0.002)]          # -0.54 dB: kept
    again = gains(0.97, 0.90)                                         # -0.26 dB: too close, made again
    assert len(again) == 2 and again[1] == pytest.approx(one_db * one_db / 0.97, abs=0.002)
    assert len(gains(1.4, 1.2, 1.1)) == 3                             # at most twice more, then kept
    for keep_sound in (True, False):
        cmd = " ".join(takevideo.write_command("ffmpeg", "in.mp4", "out.mp4", rate=48000, duration_s=1.0,
                                               keep_sound=keep_sound))
        assert f"{one_db:.3f}" in cmd


def test_once_the_stall_guard_has_tripped_no_new_tool_can_hang():
    """A guard that ends only the tools alive when it trips leaves the next one to hang the
    job.  After it trips, any tool asked for is refused at once."""
    import subprocess

    from video_fixture import Stalled

    tools = Stalled(0.01)
    try:
        assert tools.has_tripped.wait(60.0)
        with pytest.raises(OSError):
            tools.spawn(["ffmpeg"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        assert all(proc.poll() is not None for proc in tools)
    finally:
        tools.close()
