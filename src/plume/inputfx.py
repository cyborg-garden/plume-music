"""Effects on what she hears of herself: a few sounds for Hear me.

A handful of finished sounds, not a pedalboard.  Each one is chosen whole, by name:

* ``clean``  -- clean room: light compression and a small room.
* ``dreamy`` -- chorus, a dotted-eighth echo on plume's own tempo, and a long reverb
  with an octave shimmer in its tail.
* ``lofi``   -- tape: a slow wobble, warm saturation, and the top and bottom rolled off.
* ``crunch`` -- a small amp breaking up, through a speaker-cabinet curve.
* ``off``    -- her input as it comes in.

``match`` is not a sound of its own: it picks one from the style (``STYLE_SOUNDS``).
Nothing is on unless she asks for it.

Where it sits.  Only on Hear me: the input is copied to the analysis ring (and so to
``human.wav``, which plume's ears read) before any of this, so the recording stays dry.
After it comes ``realtime.mix_monitor``, which ducks on a buzzing input and then
soft-limits, so a buzz is still muted and nothing passes -1 dBFS.  A duck also clears
every tail, so no buzz rings on in a reverb after it.

Latency.  Nothing here looks ahead.  Every sound reads the sample that just came in,
except the tape wobble, whose pitch bend needs a little delay to bend (see
``added_latency_ms``).  Echoes are echoes: they are meant to come later.

Real time.  ``InputFx.process`` runs in the audio callback.  It allocates nothing:
every buffer is made here, in one arena, before the stream opens.  It is numpy ufuncs
writing into those buffers, slices, and plain float arithmetic, so it calls no C
function that test_callback_purity would flag.  Recursive filters cannot be written that
way, so there are none: every feedback loop (the reverb's lines, the echo) is at least
one buffer long, so a whole buffer is computed from what is already in the loop, and
the damping inside the loops is a two-tap FIR.  The long tone curves (cabinet, tape) are
minimum-phase FIRs, run as one matrix product over a fixed sliding view.  The callback
always gets buffers of one size (``frames``); any other size passes her input dry.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

SOUNDS = ("off", "clean", "dreamy", "lofi", "crunch")
LABELS = {"off": "Off", "clean": "Clean room", "dreamy": "Dreamy",
          "lofi": "Lofi tape", "crunch": "Crunch"}
#: A sensible sound for each style, used only when she chose ``match``.  A style not
#: here gets ``clean``.
STYLE_SOUNDS = {"lofi": "lofi", "hiphop": "lofi", "trap": "lofi",
                "jazz": "clean", "rnb": "clean", "folk": "clean",
                "midwest": "dreamy", "synthwave": "dreamy", "house": "dreamy"}
MATCH = "match"
MATCH_LABEL = "Match the style"
#: A sound's own name as she may type it: its label on the page, and "lo-fi" for lofi.
#: Hyphens are read as spaces.  A cue with one of these is hers as surely as the name.
_SPELLINGS = {**{label.lower(): name for name, label in LABELS.items()},
              MATCH_LABEL.lower(): MATCH, "lo fi": "lofi", "lo fi tape": "lofi"}
#: Looser words for a sound.  In a cue these give way to the band when the band knows
#: the word ("sound dry" is the band's tone).
_ALIASES = {"room": "clean", "dream": "dreamy", "tape": "lofi", "drive": "crunch",
            "amp": "crunch", "none": "off", "dry": "off", "plain": "off", "style": MATCH,
            "auto": MATCH}

_F32 = np.float32
#: A switch fades the old sound to dry over one buffer, then the new one in over this many.
_FADE_IN_BUFFERS = 4
#: The echo never runs longer than this; slower tempos echo on the dotted sixteenth.
_MAX_ECHO_S = 0.8
#: Added to every feedback loop each time round: far below hearing (-360 dB), far above
#: the subnormal floats a tail decaying in digital silence would otherwise reach, which
#: x86 processors handle many times slower.
_NO_SUBNORMAL = 1e-18
#: The level match on the compressing sounds: how fast it follows her playing (per buffer,
#: about a second), the quietest input it learns from (-70 dBFS), and how far it may go.
_MATCH_RATE = 0.005
_MATCH_FLOOR = 1e-7
_MATCH_MOST = 4.0
#: A buffer whose mean power is not below this is not her playing (full scale is 1): a
#: NaN or an inf from the driver, which every feedback loop would keep for the whole take.
_NOT_AUDIO = 1e4


def _typed(text) -> str | None:
    if not isinstance(text, str):
        return None
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _own_name(text) -> str | None:
    """The sound this names outright (its name, or a spelling of it), or None."""
    t = _typed(text)
    if t in SOUNDS or t == MATCH:
        return t
    return _SPELLINGS.get(t)


def canonical_sound(text) -> str | None:
    """A sound's name from what someone typed, or None.  ``match`` is allowed."""
    return _own_name(text) or _ALIASES.get(_typed(text))


def sound_for_style(style) -> str:
    return STYLE_SOUNDS.get(str(style or "").lower(), "clean")


def resolve(sound, style=None) -> str:
    """The sound to play: ``match`` becomes the style's own."""
    name = canonical_sound(sound) or "off"
    return sound_for_style(style) if name == MATCH else name


def _sound_words(text) -> list[str] | None:
    """The words after (or before) "sound", when the cue is about her sound at all:
    "sound dreamy", "my sound off", "dreamy sound".  "make it sound darker" is not."""
    if not isinstance(text, str):
        return None
    words = text.lower().replace(":", " ").split()
    if words[:2] == ["my", "sound"]:
        return words[2:]
    if words[:1] == ["sound"]:
        return words[1:]
    if 2 <= len(words) <= 3 and words[-1] == "sound":
        return words[:-1]
    return None


def parse_sound_cue(text) -> str | None:
    """A cue that changes her sound: "sound dreamy", "dreamy sound", "my sound off".

    The word "sound" has to be there.  "dreamy" alone is a mood for the band, and "dry"
    and "wet" are the band's own tone cues.
    """
    rest = _sound_words(text)
    return canonical_sound(" ".join(rest)) if rest else None


def echo_samples(bpm, sr: int) -> int:
    """A dotted eighth at this tempo, halved until it is no longer than _MAX_ECHO_S."""
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        bpm = 0.0
    if not 30.0 <= bpm <= 400.0:
        bpm = 100.0
    s = 0.75 * 60.0 / bpm
    while s > _MAX_ECHO_S:
        s *= 0.5
    return int(round(s * sr))


def added_latency_ms(sound: str, sr: int = 48000) -> float:
    """What a sound adds to the path from her strings to her ears, in milliseconds.

    Zero for all but lofi: the tape wobble is a moving delay, so its pitch can bend,
    and this is the most it ever adds (1.2 ms on average).  The minimum-phase curves
    have no bulk delay (their energy is at the first taps), and the compressor sets
    each buffer's gain from that same buffer.
    """
    if resolve(sound) == "lofi":
        return round(1000.0 * (_TAPE_BASE_S + _TAPE_WOW_S + _TAPE_FLUTTER_S), 2)
    return 0.0


# ------------------------------------------------------------------ design, at init

def _min_phase(mag: np.ndarray, taps: int) -> np.ndarray:
    """A minimum-phase FIR with this magnitude (on an rfft grid), by the cepstrum."""
    nfft = 2 * (mag.size - 1)
    cep = np.fft.irfft(np.log(np.maximum(mag, 1e-6)), nfft)
    fold = np.zeros(nfft)
    fold[0] = cep[0]
    fold[1:nfft // 2] = 2.0 * cep[1:nfft // 2]
    fold[nfft // 2] = cep[nfft // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold)), nfft)[:taps]
    h *= np.hanning(2 * taps)[taps:]           # fade the tail, not the head
    return h


def _curve(sr: int, points, nfft: int = 4096) -> np.ndarray:
    """Magnitude on an rfft grid from (hz, db) points, straight lines in log frequency."""
    f = np.linspace(0.0, sr / 2.0, nfft // 2 + 1)
    hz = np.array([p[0] for p in points], dtype=float)
    db = np.array([p[1] for p in points], dtype=float)
    return 10.0 ** (np.interp(np.log10(np.maximum(f, 1.0)), np.log10(hz), db) / 20.0)


#: Cabinet: a closed 1x12 more or less.  Lows tightened, a low-mid scoop, a presence
#: lift, and nothing of the fizz above 5 kHz.
_CAB = ((20, -30), (70, -8), (110, 0), (250, -1), (500, -5), (900, -3), (2200, 2),
        (3200, 1), (4500, -6), (6500, -24), (10000, -45), (24000, -60))
#: Tape and a small speaker: warm, no deep lows, no air.
_TAPE = ((20, -24), (70, -6), (140, 0), (1500, 0), (2800, -2), (4200, -9),
         (7000, -24), (12000, -40), (24000, -50))

#: Tape wobble: a slow wow and a little flutter around a short delay.
_TAPE_BASE_S, _TAPE_WOW_S, _TAPE_FLUTTER_S = 0.0012, 0.0008, 0.0001
_TAPE_WOW_HZ, _TAPE_FLUTTER_HZ = 0.55, 5.3


class _Arena:
    """Every buffer the callback touches, in one block of memory, so one line clears it."""

    def __init__(self):
        self._shapes: list = []

    def plan(self, shape) -> int:
        self._shapes.append(shape)
        return len(self._shapes) - 1

    def build(self):
        sizes = [int(np.prod(s)) for s in self._shapes]
        self.mem = np.zeros(sum(sizes), dtype=_F32)
        out, at = [], 0
        for s, n in zip(self._shapes, sizes):
            out.append(self.mem[at:at + n].reshape(s))
            at += n
        return out


class _Ring:
    """A delay line of ``rows`` lines, mirrored: every sample is written twice, at i and
    i + c, so any read of a buffer's length is one slice.  ``c`` is a multiple of the
    buffer, so a write never wraps."""

    def __init__(self, arena: _Arena, span: int, n: int, rows: int = 0):
        self.c = -(-(span + 2 * n) // n) * n
        self.rows, self.n = rows, n
        self._i = arena.plan((rows, 2 * self.c + n) if rows else (2 * self.c + n,))
        self.w = 0

    def bind(self, views):
        self.buf = views[self._i]


class _Fir:
    """A FIR as one matrix product: a fixed sliding view over [history | this buffer]."""

    def __init__(self, arena: _Arena, taps: np.ndarray, n: int):
        self.taps = np.ascontiguousarray(taps[::-1], dtype=_F32)   # newest sample last
        self.L, self.n = self.taps.size, n
        self._i = arena.plan((self.L - 1 + n,))
        self._h = arena.plan((self.L - 1,))

    def bind(self, views):
        self.work, self.hist = views[self._i], views[self._h]
        s = self.work.strides[0]
        self.view = as_strided(self.work, shape=(self.n, self.L), strides=(s, s), writeable=False)


class _Fdn:
    """Eight delay lines, a Householder mix, damped by a two-tap FIR in the loop."""

    def __init__(self, arena: _Arena, sr: int, n: int, delays, t60_s: float, damp: float):
        # A line has to be at least a buffer long to be done a buffer at a time: with
        # longer buffers than these were tuned for, the room grows to fit.
        k = -(-n // min(delays))
        self.d = [int(x) * k for x in delays]
        self.ring = _Ring(arena, max(self.d), n, rows=8)
        g = 10.0 ** (-3.0 * np.array(self.d) / (t60_s * sr))
        house = np.eye(8) - 2.0 / 8.0
        self.mix = np.ascontiguousarray(house * g[None, :], dtype=_F32)
        self.inject = np.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=_F32)[:, None] * 0.35
        self.taps_out = np.array([[1, 0, 1, 0, -1, 0, 1, 0],
                                  [0, 1, 0, -1, 0, 1, 0, 1]], dtype=_F32) * 0.5
        self.a, self.b = float(damp), float(1.0 - damp)
        self._t = arena.plan((8, n))
        self._prev = arena.plan((8, 1))          # each line's last sample of the buffer before
        self._tmp = arena.plan((8, n))
        self._f = arena.plan((8, n))

    def bind(self, views):
        self.ring.bind(views)
        self.t, self.tmp, self.f = views[self._t], views[self._tmp], views[self._f]
        self.prev = views[self._prev]


# ------------------------------------------------------------------- the effect

class InputFx:
    """Her sound on Hear me.  Built before the stream; ``process`` runs in the callback.

    ``slot[0]`` is the sound wanted (an index into SOUNDS), ``slot[1]`` the echo in
    samples.  Other threads write them (``choose``, ``set_bpm``); the callback only reads.
    """

    def __init__(self, sr: int, frames: int, sound: str = "off", style=None):
        self.sr, self.n = int(sr), int(frames)
        n = self.n
        self.asked = canonical_sound(sound) or "off"
        self.slot = np.array([0.0, float(echo_samples(100.0, sr))], dtype=np.float64)
        a = _Arena()
        self._x = a.plan((2, n))                 # her input, on both rows
        self._m = a.plan((n,))
        self._m2 = a.plan((n,))
        self._r = a.plan((2, n))
        self._out = a.plan((2, n))
        self._wet = a.plan((2, n))
        self._echo = a.plan((n + 1,))
        self._prev = a.plan((1,))
        # The lines.  Room: short, quickly gone.  Hall: long, for dreamy.
        self.room = _Fdn(a, sr, n, (443, 541, 617, 709, 797, 877, 967, 1049), 0.55, 0.55)
        self.hall = _Fdn(a, sr, n, (1601, 1877, 2141, 2459, 2803, 3203, 3637, 4001), 4.2, 0.7)
        self.cab = _Fir(a, _min_phase(_curve(sr, _CAB), 192), n)
        self.tape = _Fir(a, _min_phase(_curve(sr, _TAPE), 128), n)
        self.dry_ring = _Ring(a, int(0.03 * sr), n)              # chorus and tape read this
        self.echo_ring = _Ring(a, int(_MAX_ECHO_S * sr) + n, n)
        self.grain = 8 * n                                        # shimmer grain, buffers long
        self.shim_ring = _Ring(a, self.grain + 2 * n, n)
        self._views = a.build()
        self.mem = a.mem
        v = self._views
        # Same-shape operands only: numpy buffers a broadcast, which allocates.
        self.x2, self.r2 = v[self._x], v[self._r]
        self.x, self.r = self.x2[0], self.r2[0]
        self.m, self.m2 = v[self._m], v[self._m2]
        self.out, self.wet, self.echo = v[self._out], v[self._wet], v[self._echo]
        self.prev = v[self._prev]
        for part in (self.room, self.hall, self.cab, self.tape, self.dry_ring, self.echo_ring,
                     self.shim_ring):
            part.bind(v)
        self.out_t = self.out.T                                    # (n, 2), what mix_monitor adds
        self.steps = (np.arange(n, dtype=_F32) + 1.0) / _F32(n)   # 1/n .. 1
        half = self.grain // 2
        tri = 1.0 - np.abs(np.arange(self.grain) - half) / half
        self.win = tri.astype(_F32)                                # a and a half-grain later sum to 1
        # Callback state: plain floats and ints.
        self.cur = 0                  # the sound playing
        self.fade = 0.0               # how much of it (vs dry) at the end of the last buffer
        self.ducked = False
        self.env = 0.0                # compressor: smoothed power
        self.cg = 1.0                 # compressor: gain at the end of the last buffer
        self.p = 0.0                  # her input's power in this buffer (summed, not mean)
        self.pin = self.pout = 0.0    # level match: her input's power, and the sound's
        self.mg = 1.0                 # level match: gain at the end of the last buffer
        self.lfo_c, self.lfo_s = 1.0, 0.0        # chorus / wow oscillator (cos, sin)
        self.fl_c, self.fl_s = 1.0, 0.0          # flutter
        self.echo_d = int(self.slot[1])
        self.gp = 0                   # shimmer grain position
        self._rot = {}
        self._rot["chorus"] = self._rotation(0.32)
        self._rot["wow"] = self._rotation(_TAPE_WOW_HZ)
        self._rot["flutter"] = self._rotation(_TAPE_FLUTTER_HZ)
        self._run = (None, self._clean, self._dreamy, self._lofi, self._crunch)
        self.style = style
        self.choose(self.asked, style)

    # -- other threads

    def _rotation(self, hz: float):
        w = 2.0 * np.pi * hz * self.n / self.sr
        return float(np.cos(w)), float(np.sin(w))

    @property
    def playing(self) -> str:
        """The sound asked for now (the callback follows within a few buffers)."""
        return SOUNDS[int(self.slot[0])]

    def choose(self, sound, style=None) -> str:
        """Ask for a sound by name (``match`` takes the style's).  Returns what plays."""
        self.asked = canonical_sound(sound) or "off"
        if style is not None:
            self.style = style
        name = resolve(self.asked, self.style)
        self.slot[0] = float(SOUNDS.index(name))
        return name

    def set_style(self, style) -> None:
        """The style is known (or changed).  Only ``match`` listens."""
        self.style = style
        if self.asked == MATCH:
            self.choose(MATCH, style)

    def set_bpm(self, bpm) -> None:
        self.slot[1] = float(echo_samples(bpm, self.sr))

    def handle_cue(self, text) -> dict | None:
        """The reply to a sound cue ("sound dreamy"), or None if it is not one.

        A loop cue is the looper's, whatever else it says ("loop this sound").  A cue
        that names one of her sounds outright ("sound lofi", "sound lo-fi", "lofi tape
        sound") is hers.  Anything else that says "sound" ("darker sound", "sound like
        jazz", "sound dry") goes on to the band when the band understands it; only a cue
        nobody understands gets the list of sounds.
        """
        rest = _sound_words(text)
        if rest is None:
            return None
        # Imported here: a cue thread, never the callback, and looper and mind are not
        # this module's to load.
        from .looper import parse_loop_cue  # noqa: PLC0415
        if parse_loop_cue(text) is not None:
            return None
        name = parse_sound_cue(text)
        if _own_name(" ".join(rest)) is None:
            from .mind import parse_cue  # noqa: PLC0415
            if parse_cue(text)["kind"] != "unknown":
                return None
        if name is None:
            return {"ok": False, "op": "sound",
                    "error": "plume knows these sounds: " + ", ".join(SOUNDS)
                             + ", and match (picked from the style)"}
        now = self.choose(name)
        return {"ok": True, "op": "sound", "sound": now, "said": f"your sound: {LABELS[now]}"}

    # -- the audio callback

    def process(self, x, duck):
        """Her sound for this buffer as (n, 2), or None for her input as it is."""
        n = x.shape[0]
        want = int(self.slot[0])
        if duck or n != self.n:
            if duck and not self.ducked:
                self.mem[:] = 0.0
                self.ducked = True
            return None
        self.ducked = False
        if self.fade == 0.0 and self.cur != want:
            self.mem[:] = 0.0              # the new sound starts from silence
            self.cur = want
            self.pin = self.pout = 0.0
            self.mg = 1.0
        if self.cur == 0:
            return None
        self.x2[0] = x
        p = float(np.matmul(self.x, self.x))
        if not p < _NOT_AUDIO * n:
            # Not a number, or nothing an input can be.  One such sample would go round
            # every loop here for the rest of the take, and into the band's output with
            # it: this buffer of her is silence instead, and the tails carry on.
            self.x2[0] = 0.0
            p = 0.0
        self.x2[1] = self.x
        self.p = p
        f0 = self.fade
        f1 = 0.0 if want != self.cur else min(1.0, f0 + 1.0 / _FADE_IN_BUFFERS)
        self._run[self.cur]()
        self.fade = f1
        if f0 == f1 == 1.0:
            return self.out_t
        # out = dry + ramp * (out - dry), the ramp from f0 to f1 across the buffer.
        np.multiply(self.steps, f1 - f0, out=self.r)
        self.r += f0
        self.r2[1] = self.r
        self.out -= self.x2
        self.out *= self.r2
        self.out += self.x2
        return self.out_t

    def _compress(self, thresh_db: float, ratio: float, att: float, fall: float):
        """Light compression of self.m, in place: one gain per buffer, ramped across it."""
        m = self.m
        p = float(np.matmul(m, m)) / self.n
        k = att if p > self.env else fall
        self.env += k * (p - self.env)
        t = 10.0 ** (thresh_db / 10.0)
        g = (self.env / t) ** (0.5 * (1.0 / ratio - 1.0)) if self.env > t else 1.0
        np.multiply(self.steps, g - self.cg, out=self.r)
        self.r += self.cg
        m *= self.r
        self.cg = g

    def _match(self, y):
        """Bring ``y`` (in place) to her input's loudness, over about a second.

        A compressor or an amp with a fixed makeup gain is as loud as her dry input at
        one playing level only: quieter playing comes out louder, harder playing quieter.
        This follows her own level slowly, so a strum still gets evened out but a switch
        between sounds does not jump at any playing level.  It learns only while she plays.
        """
        p = self.p
        if p > _MATCH_FLOOR * self.n:
            self.pin += _MATCH_RATE * (p - self.pin)
            self.pout += _MATCH_RATE * (float(np.matmul(y, y)) - self.pout)
        g = (self.pin / self.pout) ** 0.5 if self.pout > 0.0 else 1.0
        g = _MATCH_MOST if g > _MATCH_MOST else (1.0 / _MATCH_MOST if g < 1.0 / _MATCH_MOST else g)
        np.multiply(self.steps, g - self.mg, out=self.r)
        self.r += self.mg
        y *= self.r
        self.mg = g

    def _fir(self, fir: _Fir, src, dst):
        L1 = fir.L - 1
        fir.work[:L1] = fir.hist
        fir.work[L1:] = src
        fir.hist[:] = fir.work[self.n:]
        np.matmul(fir.view, fir.taps, out=dst)

    def _reverb(self, fdn: _Fdn, src):
        """One buffer of the lines into self.wet (2, n), fed with ``src``."""
        ring, t, n = fdn.ring, fdn.t, self.n
        w, c = ring.w, ring.c
        for i in range(8):
            s = w - fdn.d[i] + (c if w < fdn.d[i] else 0)
            t[i] = ring.buf[i, s:s + n]
        np.matmul(fdn.taps_out, t, out=self.wet)
        np.multiply(t, fdn.a, out=fdn.f)                 # damping: a*now + b*before
        fdn.tmp[:, :1] = fdn.prev
        fdn.tmp[:, 1:] = t[:, :-1]
        fdn.tmp *= fdn.b
        fdn.f += fdn.tmp
        fdn.prev[:] = t[:, n - 1:]
        np.matmul(fdn.mix, fdn.f, out=fdn.tmp)
        np.matmul(fdn.inject, src[None, :], out=fdn.f)
        fdn.f += fdn.tmp
        fdn.f += _NO_SUBNORMAL
        ring.buf[:, w:w + n] = fdn.f
        ring.buf[:, w + c:w + c + n] = fdn.f
        ring.w = (w + n) % c

    def _write(self, ring: _Ring, src):
        w = ring.w
        ring.buf[w:w + self.n] = src
        ring.buf[w + ring.c:w + ring.c + self.n] = src

    def _tap(self, ring: _Ring, d0: float, d1: float, dst):
        """dst = the line read d0 samples back at the buffer's start, gliding to d1 at its
        end (this buffer already written).  Two linear reads, crossfaded across it."""
        w, c, n, r = ring.w, ring.c, self.n, self.r
        i0, i1 = int(d0), int(d1)
        s0, s1 = w - i0 + (c if w < i0 + 1 else 0), w - i1 + (c if w < i1 + 1 else 0)
        a0, a1 = d0 - i0, d1 - i1
        np.multiply(ring.buf[s0:s0 + n], 1.0 - a0, out=dst)
        np.multiply(ring.buf[s0 - 1:s0 - 1 + n], a0, out=r)
        dst += r
        np.multiply(ring.buf[s1:s1 + n], 1.0 - a1, out=self.m2)
        np.multiply(ring.buf[s1 - 1:s1 - 1 + n], a1, out=r)
        self.m2 += r
        self.m2 -= dst
        self.m2 *= self.steps
        dst += self.m2

    # -- the sounds; each leaves its whole output in self.out (2, n)

    def _clean(self):
        self.m[:] = self.x
        self._compress(-26.0, 2.5, 0.45, 0.04)
        self._match(self.m)
        self.m *= 1.02
        self._reverb(self.room, self.m)
        np.multiply(self.wet, 0.55, out=self.out)
        self.out[0] += self.m
        self.out[1] += self.m

    def _dreamy(self):
        n, out, m = self.n, self.out, self.m
        ring = self.dry_ring
        self._write(ring, self.x)
        c, s = self.lfo_c, self.lfo_s
        rc, rs = self._rot["chorus"]
        c1, s1 = c * rc - s * rs, s * rc + c * rs
        self.lfo_c, self.lfo_s = c1, s1
        base, depth = 0.009 * self.sr, 0.0022 * self.sr
        self._tap(ring, base + depth * s, base + depth * s1, out[0])
        self._tap(ring, base + depth * c, base + depth * c1, out[1])
        ring.w = (ring.w + n) % ring.c
        out *= 0.5
        out += self.x2                                     # dry + chorus, per side
        # The echo: a dotted eighth on plume's tempo, darker each time round.
        e, er = self.echo, self.echo_ring
        d_new = int(self.slot[1])
        w, cc = er.w, er.c
        s0 = w - self.echo_d + (cc if w < self.echo_d else 0)
        e[1:] = er.buf[s0:s0 + n]
        if d_new != self.echo_d:
            s1 = w - d_new + (cc if w < d_new else 0)
            np.subtract(er.buf[s1:s1 + n], e[1:], out=m)
            m *= self.steps
            e[1:] += m
            self.echo_d = d_new
        np.multiply(e[1:], 0.24, out=m)                    # feedback, two-tap damped
        np.multiply(e[:-1], 0.16, out=self.r)
        m += self.r
        m += self.x
        m += _NO_SUBNORMAL
        e[0] = e[n]
        self._write(er, m)
        er.w = (w + n) % cc
        # Shimmer: an octave up from the hall's own past, back into the hall.
        sh, g, half = self.shim_ring, self.gp, self.grain // 2
        D0 = self.grain + n
        sa = sh.w - D0 + g + (sh.c if sh.w < D0 - g else 0)
        gb = (g + half) % self.grain
        sb = sh.w - D0 + gb + (sh.c if sh.w < D0 - gb else 0)
        np.multiply(sh.buf[sa:sa + 2 * n:2], self.win[g:g + n], out=self.m2)
        np.multiply(sh.buf[sb:sb + 2 * n:2], self.win[gb:gb + n], out=self.r)
        self.m2 += self.r
        self.m2 *= 0.22
        self.gp = (g + n) % self.grain
        np.add(out[0], out[1], out=m)                      # chorus, echo and shimmer into the hall
        m *= 0.5
        m += e[1:]
        m += self.m2
        self._reverb(self.hall, m)
        np.add(self.wet[0], self.wet[1], out=self.r)
        self._write(sh, self.r)
        sh.w = (sh.w + n) % sh.c
        np.multiply(e[1:], 0.32, out=m)
        out[0] += m
        out[1] += m
        self.wet *= 0.62
        out += self.wet
        out *= 0.79

    def _lofi(self):
        n, m, ring = self.n, self.m, self.dry_ring
        self._write(ring, self.x)
        c, s = self.lfo_c, self.lfo_s
        rc, rs = self._rot["wow"]
        c1, s1 = c * rc - s * rs, s * rc + c * rs
        fc, fs = self.fl_c, self.fl_s
        qc, qs = self._rot["flutter"]
        fc1, fs1 = fc * qc - fs * qs, fs * qc + fc * qs
        self.lfo_c, self.lfo_s, self.fl_c, self.fl_s = c1, s1, fc1, fs1
        sr = self.sr
        d0 = (_TAPE_BASE_S + _TAPE_WOW_S * s + _TAPE_FLUTTER_S * fs) * sr
        d1 = (_TAPE_BASE_S + _TAPE_WOW_S * s1 + _TAPE_FLUTTER_S * fs1) * sr
        self._tap(ring, d0, d1, m)
        ring.w = (ring.w + n) % ring.c
        # Saturation, a little lopsided like tape, then its tone.
        m *= 2.2
        m += 0.15
        np.tanh(m, out=m)
        m -= 0.14888                                       # tanh(0.15): no offset at rest
        self._fir(self.tape, m, self.m2)
        np.multiply(self.m2, 0.68, out=self.out[0])
        self.out[1] = self.out[0]

    def _crunch(self):
        m = self.m
        m[:] = self.x
        self._compress(-30.0, 3.0, 0.5, 0.05)            # same bite at any input gain
        # Brighten into the drive (the lows stay tight), break up, then the cabinet.
        self.r[0] = self.prev[0]
        self.r[1:] = m[:-1]
        self.prev[0] = m[self.n - 1]
        self.r *= -0.55
        m += self.r
        m *= 9.0
        m += 0.12
        np.tanh(m, out=m)
        m -= 0.11943                                       # tanh(0.12)
        self._fir(self.cab, m, self.m2)
        self._match(self.m2)
        self.m2 *= 0.94
        self._reverb(self.room, self.m2)
        self.out[0] = self.m2
        self.out[1] = self.m2
        self.wet *= 0.25
        self.out += self.wet
