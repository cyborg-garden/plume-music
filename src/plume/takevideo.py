"""A video of a take, with the band laid in.

A player films herself playing along.  Her camera hears her and the room, and not the
band, which is in her headphones.  This finds where in the video the take begins, by
matching her recording against the video's own sound, and writes the video again with
the band and her clean recording in it.  The picture is copied, never encoded again.

Everything that touches a video runs ``ffmpeg`` as a child: the one a package brought
(``imageio-ffmpeg``), else one on the PATH.  No video is ever read into memory: its
sound is decoded at a low rate to line it up, and the mix is fed to the child a block
at a time.

Three clocks meet here, all in seconds:

- the video's, from its first frame;
- the take's own, which is ``machine.wav``'s: the band's recording from its first sample;
- the player's recording, which runs behind the band's by the take's round trip
  (:func:`plume.record.playing_delay_s`).

``start_s`` is where the first sample of the player's recording falls in the video.  It
is what is matched and what a nudge moves.  The band's first sample falls later than
that by the round trip.  A video that began after the take did has a negative start.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import oscompat

#: The rate both sounds are compared at.  Enough for every pitch a hand can pick out,
#: and 8 kHz mono keeps an hour of video in tens of megabytes.
ALIGN_RATE = 8000
#: The rate of the outline the first pass matches: how loud, 500 times a second.
ENV_RATE = 500
_HOP = ALIGN_RATE // ENV_RATE
#: A video longer than this is not taken in.
MAX_VIDEO_S = 4 * 3600.0
#: The two must share at least this much, or half of the shorter one if that is less.
_MIN_OVERLAP_S = 8.0
#: How far the second pass may move the first's answer.
_FINE_REACH_S = 0.15
#: The second pass listens to this much, where the player is loudest.
_FINE_WINDOW_S = 20.0
#: Chance alone reaches about four of these; below this the second pass has found nothing.
_FINE_FLOOR = 6.0
#: Two halves of the playing must put the start within this of each other.
_FINE_AGREE_S = 0.005
#: ... and from here up it is as sure as it gets.
_FINE_SURE = 20.0
#: At or above this, the place is trusted and nobody is asked.
CONFIDENT = 0.5
#: Quieter than this at its loudest, a sound is nothing to line up with.
_SILENT_PEAK = 10 ** (-60 / 20)
_DRIFT_MIN_S = 40.0
_PCM_BLOCK = 1 << 16
_LIMITER_AHEAD_MS = 5
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Stream #0:\d+.*?: (Video|Audio): (.*)")


#: The file a finished video is saved as, in the take's folder.
REC_VIDEO = "video.mp4"
#: How plume knows a ``video.mp4`` for its own: the file ends with a box players skip
#: ("free"), which begins with this.  A player may keep a video of theirs under that name.
VIDEO_MARK = b"plVd"
#: Raised when a video would come out different for the same take and the same film.
VIDEO_RULE = 1
_VIDEO_TAIL_BYTES = 1024


def stamp(doc: dict) -> bytes:
    """The box that ends a video plume made, saying how it was made."""
    body = VIDEO_MARK + json.dumps({"rule": f"video:{VIDEO_RULE}", **doc}, sort_keys=True).encode()
    return (8 + len(body)).to_bytes(4, "big") + b"free" + body


def is_own_video(path) -> bool:
    """True when the file at ``path`` is a video plume saved, by the box it ends with.

    As for a saved mix: looked for at the end and nowhere else, a length that runs
    exactly to the end of the file, and JSON that names a rule.  Anything less is the
    player's and is left alone: a symlink, a folder, a missing or unreadable file, or
    a video of plume's that something has since added to.
    """
    try:
        fd = oscompat.open_private(path, oscompat.read_flags())
    except OSError:
        return False
    try:
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                return False
            f.seek(max(0, st.st_size - _VIDEO_TAIL_BYTES))
            tail = f.read(_VIDEO_TAIL_BYTES)
    except OSError:
        return False
    at = tail.rfind(b"free" + VIDEO_MARK)
    if at < 4 or int.from_bytes(tail[at - 4:at], "big") != len(tail) - (at - 4):
        return False
    try:
        doc = json.loads(tail[at + 8:])
    except (ValueError, RecursionError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("rule"), str)


class VideoError(RuntimeError):
    """A clean failure.  ``why`` is a short code the page has words for."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


class Cancelled(Exception):
    """The player asked for the work to stop."""


@dataclass(frozen=True)
class Probe:
    duration_s: float
    has_audio: bool
    #: The picture's format, as ffmpeg names it: "h264", "hevc".
    codec: str = ""


@dataclass(frozen=True)
class Alignment:
    """Where the player's recording begins in the video, and how sure that is."""
    start_s: float
    confidence: float
    #: How much the two clocks part over the take, in parts per million.  Measured and
    #: reported only: a phone and an interface differ by far less than an ear can tell.
    drift_ppm: float | None
    #: Why there is no trusted place: "silent" when the video has no usable sound.
    why: str | None = None

    @property
    def confident(self) -> bool:
        return self.why is None and self.confidence >= CONFIDENT


# ---------------------------------------------------------------------- the tool

def find_ffmpeg() -> str | None:
    """The ffmpeg to run: the packaged one, else one on the PATH, else None."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:  # noqa: BLE001 - not installed, or installed with no binary for this machine
        pass
    return shutil.which("ffmpeg")


def _quiet() -> dict:
    """No console window flashes up on Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _stop(proc) -> None:
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


def probe(ffmpeg: str, path) -> Probe:
    """What a file holds.  Raises ``VideoError("not_video")`` for anything but a video."""
    try:
        done = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", os.fspath(path)],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=120, check=False, **_quiet())
    except (OSError, subprocess.TimeoutExpired):
        raise VideoError("not_video") from None
    text = done.stderr.decode("utf-8", errors="replace")
    kinds = [(kind, rest) for kind, rest in _STREAM_RE.findall(text)]
    # A song's cover art is a picture, not a film.
    video = [rest for kind, rest in kinds if kind == "Video" and "attached pic" not in rest]
    found = _DURATION_RE.search(text)
    if not video or not found:
        raise VideoError("not_video")
    hours, minutes, seconds = found.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if not 0 < duration <= MAX_VIDEO_S:
        raise VideoError("not_video" if duration <= 0 else "too_long")
    codec = re.match(r"[a-z0-9_]+", video[0])
    return Probe(duration, any(kind == "Audio" for kind, _ in kinds), codec.group(0) if codec else "")


def decode(ffmpeg: str, source, *, rate: int = ALIGN_RATE, cancelled=None,
           start_s: float | None = None, seconds: float | None = None) -> np.ndarray:
    """A file's first sound as mono floats at ``rate``, on the file's own clock.

    ``source`` is a path, or an open binary file, which is piped in so that a file
    opened safely stays the file that is read.  Sound that begins late in a video is
    padded so that sample 0 is the video's time 0.
    """
    piped = not isinstance(source, (str, os.PathLike))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if not piped:
        cmd += ["-nostdin"]
    if start_s is not None:
        cmd += ["-ss", f"{max(0.0, start_s):.6f}"]
    cmd += ["-i", "pipe:0" if piped else os.fspath(source)]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.6f}"]
    cmd += ["-map", "0:a:0", "-vn", "-sn", "-dn", "-af", "aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", str(rate), "-f", "s16le", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdin=source if piped else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_quiet())
    except OSError:
        raise VideoError("failed") from None
    blocks, limit, got = [], int(MAX_VIDEO_S * rate) * 2, 0
    try:
        while True:
            if cancelled is not None and cancelled():
                raise Cancelled
            block = proc.stdout.read(1 << 20)
            if not block:
                break
            blocks.append(block)
            got += len(block)
            if got > limit:
                raise VideoError("too_long")
    finally:
        _stop(proc)
    data = b"".join(blocks)
    return np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


# ------------------------------------------------------------------- lining up

def _outline(x: np.ndarray) -> np.ndarray:
    """Where a sound gets louder, ``ENV_RATE`` times a second, around zero.

    A camera across the room and a cable hear the same notes with different tone and
    level, and one of them hears talking too.  What they share is when each note
    begins, so this keeps the rises in loudness, in decibels, and nothing else.
    """
    if len(x) < _HOP * 8:
        return np.zeros(0)
    sos = signal.butter(2, [150.0, 3200.0], btype="bandpass", fs=ALIGN_RATE, output="sos")
    power = signal.sosfilt(sos, x.astype(np.float64)) ** 2
    frames = len(power) // _HOP
    power = power[:frames * _HOP].reshape(frames, _HOP).mean(axis=1)
    # Two frames together: a note's start is not split by where a frame happens to fall.
    power = np.convolve(power, np.ones(2) / 2.0, mode="same")
    floor = max(float(power.max()) * 1e-5, 1e-12)        # 50 dB under the loudest moment
    level = np.log(power + floor)
    rise = np.maximum(np.diff(level, prepend=level[0]), 0.0)
    return rise - rise.mean()


def _energy_between(cumulative: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return cumulative[hi] - cumulative[lo]


def _scores(video: np.ndarray, take: np.ndarray):
    """How well the take's outline fits the video's at every place they could meet.

    Returns ``(lags, score)``: the take's first frame sits at video frame ``lag``.
    Each score is a correlation over only the part the two share, so a take that
    starts before the video, or runs past its end, is judged on what there is.
    """
    n, m = len(video), len(take)
    size = 1 << int(np.ceil(np.log2(n + m)))
    raw = np.fft.irfft(np.fft.rfft(video, size) * np.conj(np.fft.rfft(take, size)), size)
    lags = np.arange(-(m - 1), n)
    raw = raw[lags % size]
    lo_v, hi_v = np.maximum(lags, 0), np.minimum(lags + m, n)
    overlap = hi_v - lo_v
    video_sq = np.concatenate(([0.0], np.cumsum(video ** 2)))
    take_sq = np.concatenate(([0.0], np.cumsum(take ** 2)))
    energy = (_energy_between(video_sq, lo_v, hi_v)
              * _energy_between(take_sq, lo_v - lags, hi_v - lags))
    score = raw / np.sqrt(np.maximum(energy, 1e-18))
    need = min(_MIN_OVERLAP_S * ENV_RATE, 0.5 * min(n, m))
    # A little in favour of places that share more: a short stretch fits by luck more easily.
    score *= (overlap / max(1, min(n, m))) ** 0.25
    score[overlap < need] = -np.inf
    return lags, score


def _peak(values: np.ndarray, at: int) -> float:
    """Where a peak really is, between samples, from the three around it."""
    if at <= 0 or at >= len(values) - 1:
        return float(at)
    left, mid, right = values[at - 1], values[at], values[at + 1]
    if not (np.isfinite(left) and np.isfinite(right)):
        return float(at)
    bend = left - 2.0 * mid + right
    return float(at) if bend >= 0 else at + 0.5 * (left - right) / bend


def _fine(video: np.ndarray, take: np.ndarray, start_s: float) -> tuple[float, float]:
    """The start to within a sample, from the sounds themselves near the first answer.

    Returns ``(start_s, strength)``.  Compared in tone-blind form (each pitch counts the
    same however loud each recording made it), so a boomy room and a bright pickup still
    meet.  ``strength`` is how far the best fit stands above all the others, in their own
    spread: two recordings of the same playing give tens, anything else about four.  It
    is the proof that this is the place, and what the confidence is made from.  A weak
    one moves nothing: the first answer stays.
    """
    lag = round(start_s * ALIGN_RATE)
    lo, hi = max(0, -lag), min(len(take), len(video) - lag)
    width = int(_FINE_WINDOW_S * ALIGN_RATE)
    if hi - lo < ALIGN_RATE:
        return start_s, 0.0
    if hi - lo > width:
        loud = np.concatenate(([0.0], np.cumsum(take[lo:hi].astype(np.float64) ** 2)))
        lo += int(np.argmax(loud[width:] - loud[:-width]))
        hi = lo + width
    a, b = video[lo + lag:hi + lag].astype(np.float64), take[lo:hi].astype(np.float64)
    if min(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) < _SILENT_PEAK:
        return start_s, 0.0
    reach = int(_FINE_REACH_S * ALIGN_RATE)
    near, strength = _fit(a, b, reach)
    best = int(np.argmax(near))
    if strength < _FINE_FLOOR:
        return start_s, strength
    # One loud knock in each recording fits as sharply as a whole song does.  So the
    # player's odd seconds and her even seconds are each asked alone, and must both
    # say the same place: playing does, a knock cannot.
    seconds = np.arange(len(b)) // ALIGN_RATE % 2
    for half in (0, 1):
        some, some_strength = _fit(a, np.where(seconds == half, b, 0.0), reach)
        if abs(int(np.argmax(some)) - best) > _FINE_AGREE_S * ALIGN_RATE:
            some_strength = 0.0
        strength = min(strength, some_strength * math.sqrt(2.0))      # half the sound, so 1/sqrt(2) the height
    if strength < _FINE_FLOOR:
        return start_s, strength
    return (lag + _peak(near, best) - reach) / ALIGN_RATE, strength


def _fit(a: np.ndarray, b: np.ndarray, reach: int) -> tuple[np.ndarray, float]:
    """How well ``b`` fits ``a`` moved by up to ``reach`` samples either way, and how far
    the best of those stands above every other move, in their spread."""
    size = 1 << int(np.ceil(np.log2(2 * len(a))))
    cross = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    freqs = np.fft.rfftfreq(size, 1.0 / ALIGN_RATE)
    band = (freqs >= 150.0) & (freqs <= 3200.0)
    size_of = np.abs(cross)
    # The floor keeps pitches neither recording holds from counting as much as the rest.
    cross = np.where(band, cross / (size_of + 0.01 * float(size_of[band].mean()) + 1e-30), 0.0)
    fit = np.fft.irfft(cross, size)
    near = np.concatenate((fit[-reach:], fit[:reach + 1]))
    spread = float(np.std(fit))
    return near, (float(near.max() / spread) if spread > 0 else 0.0)


def _drift(video: np.ndarray, take: np.ndarray, start_s: float) -> float | None:
    """How far the two clocks part, from where the first and the last of the shared
    stretch each fit best.  None when there is too little of it to say."""
    lag = round(start_s * ALIGN_RATE)
    lo, hi = max(0, -lag), min(len(take), len(video) - lag)
    piece = min((hi - lo) // 3, int(30.0 * ALIGN_RATE))
    if (hi - lo) < _DRIFT_MIN_S * ALIGN_RATE:
        return None
    reach = int(0.02 * ALIGN_RATE)
    places = []
    for begin in (lo, hi - piece):
        near, strength = _fit(video[begin + lag:begin + lag + piece].astype(np.float64),
                              take[begin:begin + piece].astype(np.float64), reach)
        if strength < _FINE_FLOOR:
            return None
        places.append(_peak(near, int(np.argmax(near))) - reach)
    return float((places[1] - places[0]) / (hi - piece - lo) * 1e6)


def align(video: np.ndarray, take: np.ndarray) -> Alignment:
    """Find the player's recording in the video's sound.  Both mono at ``ALIGN_RATE``."""
    if len(video) == 0 or float(np.max(np.abs(video))) < _SILENT_PEAK:
        return Alignment(0.0, 0.0, None, "silent")
    video_outline, take_outline = _outline(video), _outline(take)
    if (len(video_outline) < ENV_RATE or len(take_outline) < ENV_RATE
            or not np.any(take_outline) or not np.any(video_outline)):
        return Alignment(0.0, 0.0, None, "silent" if not np.any(video_outline) else None)
    lags, score = _scores(video_outline, take_outline)
    if not np.any(np.isfinite(score)):
        return Alignment(0.0, 0.0, None)
    best = int(np.argmax(score))
    if not float(score[best]) > 0:
        return Alignment(0.0, 0.0, None)
    start_s = (int(lags[best]) + _peak(score, best) - best) / ENV_RATE
    start_s, strength = _fine(video, take, start_s)
    confidence = float(np.clip((strength - _FINE_FLOOR) / (_FINE_SURE - _FINE_FLOOR), 0.0, 1.0))
    drift = _drift(video, take, start_s) if confidence > 0 else None
    return Alignment(float(start_s), confidence, drift)


def first_sound_s(take: np.ndarray) -> float:
    """When the player first plays, on her recording's clock: where a preview should listen."""
    if len(take) == 0:
        return 0.0
    level = np.abs(take)
    loud = np.flatnonzero(level > 0.25 * float(level.max())) if level.max() > 0 else []
    return float(loud[0] / ALIGN_RATE) if len(loud) else 0.0


# ---------------------------------------------------------------------- the mix

def video_frame_of(start_s: float, delay_frames: int, rate: int) -> int:
    """The video's frame, at ``rate``, where the band's first sample falls."""
    return round(start_s * rate) + delay_frames


def _silence(count: int) -> bytes:
    return bytes(2 * count)


def laid_in(mixed, band_at: int, begin: int, count: int) -> bytes:
    """``count`` frames of the take's mix from video frame ``begin``, silence around it.

    ``mixed(frame, count)`` is the take's own mix, the one Together plays, from the
    band's frame ``frame``.  ``band_at`` is the video frame of the band's first sample.
    """
    first = begin - band_at                       # the take frame wanted first; may be negative
    lead = min(count, max(0, -first))
    return _silence(lead) + (mixed(first + lead, count - lead) if count > lead else b"")


def _graph(keep_sound: bool, rate: int) -> list[str]:
    """What the child does with the two sounds.  Input 0 is the video, input 1 the take."""
    if not keep_sound:
        return ["-map", "0:v:0", "-map", "1:a:0", "-af", "pan=stereo|c0=c0|c1=c0"]
    # Summed at full level, then held under full scale only where the sum would pass it.
    # amerge and pan rather than amix: every ffmpeg adds these the same way.
    graph = (f"[0:a:0]aresample={rate}:async=1:first_pts=0,aformat=channel_layouts=stereo,apad[cam];"
             "[1:a:0]pan=stereo|c0=c0|c1=c0[take];"
             "[cam][take]amerge=inputs=2,pan=stereo|c0=c0+c2|c1=c1+c3,"
             f"alimiter=limit=0.891:attack={_LIMITER_AHEAD_MS}:release=80:level=disabled,"
             # The limiter looks ahead, and hands its sound over that much late.  Taken back
             # off here, or the band would trail the picture by it.
             f"atrim=start={_LIMITER_AHEAD_MS / 1000.0},asetpts=PTS-STARTPTS[out]")
    return ["-filter_complex", graph, "-map", "0:v:0", "-map", "[out]"]


def write_command(ffmpeg: str, source, out, *, rate: int, duration_s: float, keep_sound: bool,
                  codec: str = "") -> list[str]:
    """The child's command line: the video in, the take on its standard input, an mp4 out."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", os.fspath(source),
           "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
           *_graph(keep_sound, rate), "-c:v", "copy"]
    if codec == "hevc":
        # This format has two names in an mp4, and some players only know this one.
        cmd += ["-tag:v", "hvc1"]
    return cmd + ["-c:a", "aac", "-b:a", "192k", "-ar", str(rate), "-ac", "2",
                  "-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact",
                  "-movflags", "+faststart", "-t", f"{duration_s:.6f}", "-f", "mp4", os.fspath(out)]


def write_video(ffmpeg: str, source, out, *, mixed, rate: int, band_at: int, duration_s: float,
                keep_sound: bool, codec: str = "", cancelled=None, progress=None) -> None:
    """Write ``source`` again at ``out`` with the take in its sound.

    The picture is copied as it is.  The sound is the video's own, if kept, with the
    take's mix on top, as AAC.  Nothing the camera wrote about itself comes along:
    no tags, no place, no chapters, no data tracks.
    """
    # A little past the end, so the sound never stops short of the picture.
    total = max(1, math.ceil((duration_s + 0.1) * rate))
    cmd = write_command(ffmpeg, source, out, rate=rate, duration_s=duration_s, keep_sound=keep_sound,
                        codec=codec)
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, **_quiet())
    except OSError:
        raise VideoError("failed") from None
    try:
        try:
            for begin in range(0, total, _PCM_BLOCK):
                if cancelled is not None and cancelled():
                    raise Cancelled
                count = min(_PCM_BLOCK, total - begin)
                proc.stdin.write(laid_in(mixed, band_at, begin, count))
                if progress is not None:
                    progress((begin + count) / total)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            raise VideoError("failed") from None
        while True:
            if cancelled is not None and cancelled():
                raise Cancelled
            try:
                code = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if code != 0:
            raise VideoError("failed")
    finally:
        _stop(proc)


def preview(ffmpeg: str, source, *, mixed, rate: int, band_at: int, from_s: float, seconds: float,
            has_audio: bool) -> np.ndarray:
    """A few seconds of how it would sound, as mono floats: the video's own sound and the take."""
    begin, count = max(0, round(from_s * rate)), round(seconds * rate)
    take = np.frombuffer(laid_in(mixed, band_at, begin, count), dtype="<i2").astype(np.float32) / 32768.0
    if not has_audio:
        return take
    cam = decode(ffmpeg, source, rate=rate, start_s=begin / rate, seconds=seconds)[:count]
    out = take.copy()
    out[:len(cam)] += cam
    return out
