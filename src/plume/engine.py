"""plume.engine — compile + offline render (SPEC section 9).

``compile_events`` turns a Session into a flat, time-sorted list of
``NoteEvent``; ``render`` realizes those events as a PCM_24 WAV through the
per-track fx chain and master chain. Live mode (v1) reuses exactly this
split: it compiles one bar at a time against a moving clock.

Determinism: one ``np.random.default_rng(session.seed)`` is created per
render and passed explicitly through compile (groove jitter) and then the
voice renderers — same session JSON + same seed => bit-identical WAV.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from plume import fx as _fx
from plume import notation, theory
from plume.score import Session, Track
from plume.voices import PRESETS, render_note

SR = 48000
TAIL_S = 1.5                 # release/reverb tail past the last bar
DEFAULT_VELOCITY = 0.8

__all__ = ["NoteEvent", "compile_events", "render", "play", "SR"]


# ---------------------------------------------------------------------------
# groove import — builder C is concurrent; fall back to a SPEC-section-4-exact
# implementation so the engine works standalone. Once plume/groove.py exists
# it is used instead.
# ---------------------------------------------------------------------------
try:                                          # pragma: no cover - env dependent
    from plume.groove import GrooveParams, apply_groove
except ImportError:                           # pragma: no cover - env dependent
    @dataclass(frozen=True)
    class GrooveParams:                        # type: ignore[no-redef]
        swing: float
        timing_bias_ms: dict[str, float]
        humanize_ms: float
        humanize_vel: float

    def apply_groove(onset_beats: float, *, role: str, bpm: float,
                     g: GrooveParams, rng: np.random.Generator,
                     velocity: float) -> tuple[float, float]:
        """SPEC section 4 fallback: swing on even 16ths, role bias, jitter."""
        spb = 60.0 / bpm
        onset_s = onset_beats * spb
        frac = onset_beats % 1.0
        if abs(frac - 0.25) < 1e-3 or abs(frac - 0.75) < 1e-3:
            onset_s += (g.swing - 0.5) * 2.0 * (spb / 4.0)
        onset_s += g.timing_bias_ms.get(role, 0.0) / 1000.0
        onset_s += float(rng.normal(0.0, max(g.humanize_ms, 0.0) / 1000.0))
        v = velocity + (0.08 if abs(onset_beats) < 1e-6 else 0.0)
        v += float(rng.normal(0.0, max(g.humanize_vel, 0.0)))
        return onset_s, float(np.clip(v, 0.05, 1.0))


@dataclass(frozen=True)
class NoteEvent:
    t_s: float
    track: str
    role: str
    freq_hz: float | None
    dur_s: float
    velocity: float
    glide_from_hz: float | None
    params: dict = field(default_factory=dict)
    gain: float = 1.0     # linear pre-master gain (section-energy loudness arc)


# voice preset -> role (SPEC section 4: roles = track voice family)
_ROLES = {
    "808kick": "kick", "808sub": "bass", "hat": "hat", "openhat": "hat",
    "clap": "clap", "snare": "snare", "pad": "pad", "pluck": "pluck",
    "epiano": "epiano", "lead": "lead", "crackle": "crackle",
    "housekick": "kick", "boomkick": "kick", "offhat": "hat", "brush": "snare",
    "upright": "bass", "roundbass": "bass",
}

#: What section energy scales to brighten a voice: a filter cutoff, or for the
#: FM voices and the organ (plume.synths) the modulation index and the
#: percussive upper drawbars, which are their brightness.
_BRIGHTNESS_KEYS = ("cutoff_hz", "cutoff_start_hz", "index", "percussion")


def _low_octave_hz(pc: int) -> float:
    """Pitch class -> frequency in (or nearest to) the 35-55 Hz octave."""
    cands = [theory.midi_to_hz(12 * (o + 1) + pc) for o in (0, 1, 2)]
    inside = [h for h in cands if 35.0 <= h <= 55.0]
    if inside:
        return inside[0]
    return min(cands, key=lambda h: min(abs(h - 35.0), abs(h - 55.0)))


def _strip_vel(token: str) -> tuple[str, float | None]:
    """Split the notation ':v<float>' velocity suffix off a token."""
    if ":v" in token:
        base, _, v = token.partition(":v")
        return base, float(v)
    return token, None


def _energy_params(tr: Track, energy: float) -> dict:
    """Track params with fx brightness gated by section energy."""
    params = dict(tr.params or {})
    if tr.kind != "drum":
        defaults = PRESETS.get(tr.voice, {})
        for k in _BRIGHTNESS_KEYS:
            if k in params or k in defaults:
                base = float(params.get(k, defaults[k]))
                params[k] = base * (0.6 + 0.4 * float(energy))
    return params


def _tonal_freqs(token: str, role: str, track_name: str,
                 prev_voicing: dict[str, list[int]]) -> list[float]:
    """Note token -> [hz]; chord token -> voiced hz list (mono bass: root)."""
    try:
        return [theory.midi_to_hz(theory.note_to_midi(token))]
    except ValueError:
        pass
    try:
        chord = theory.parse_chord(token)
    except ValueError as exc:
        raise ValueError(
            f"track {track_name!r}: token {token!r} is neither a note nor a chord"
        ) from exc
    if role == "bass":
        return [_low_octave_hz(chord.root)]
    voiced = theory.voice_chord(chord, prev=prev_voicing.get(track_name))
    prev_voicing[track_name] = voiced
    return [theory.midi_to_hz(m) for m in voiced]


def _compile(session: Session, rng: np.random.Generator) -> list[NoteEvent]:
    key = theory.parse_key(session.key)
    tonic_hz = _low_octave_hz(key.tonic)
    g = GrooveParams(
        swing=float(session.swing),
        timing_bias_ms=dict(session.timing_bias_ms or {}),
        humanize_ms=float(session.humanize_ms),
        humanize_vel=float(session.humanize_vel),
    )
    spb = 60.0 / float(session.bpm)
    bar_s = 4.0 * spb
    pattern_cache: dict[tuple[str, str], list[notation.Step]] = {}
    prev_voicing: dict[str, list[int]] = {}
    last_bass: dict[str, tuple[float, float]] = {}   # track -> (end pos in bars, hz)
    events: list[NoteEvent] = []
    bar_global = 0
    for mv in session.movements:
        for sec in mv.sections:
            sec_params = {t.name: _energy_params(t, sec.energy) for t in session.tracks}
            # loudness arc: section energy maps to a small pre-master gain
            # offset (-3 dB at energy 0 .. 0 dB at energy 1) so movements
            # breathe in level, not just in density (SPEC section 9 amendment)
            e = float(np.clip(sec.energy, 0.0, 1.0))
            sec_gain = float(10.0 ** ((-3.0 + 3.0 * e) / 20.0))
            for b in range(sec.bars):
                bar_start = bar_global * bar_s
                is_last = b == sec.bars - 1
                for tr in session.tracks:
                    pname = (sec.plays or {}).get(tr.name, "")
                    if not pname:
                        continue        # tacet
                    use = "fill" if (sec.fill and is_last and "fill" in tr.patterns) else pname
                    if use not in tr.patterns:
                        raise ValueError(
                            f"track {tr.name!r}: pattern {use!r} referenced by "
                            f"section {sec.name!r} is not defined")
                    ck = (tr.name, use)
                    if ck not in pattern_cache:
                        pattern_cache[ck] = notation.parse(tr.patterns[use])
                    steps = notation.steps_for_cycle(pattern_cache[ck], b)
                    role = _ROLES.get(tr.voice, tr.voice)
                    params = sec_params[tr.name]
                    for st in steps:
                        token, tok_vel = _strip_vel(st.token)
                        vel = ((tok_vel if tok_vel is not None else DEFAULT_VELOCITY)
                               * (0.6 + 0.4 * float(sec.energy)))
                        onset_beats = st.start * 4.0
                        dur_s = st.dur * bar_s
                        glide_from: float | None = None
                        if tr.kind == "drum":
                            freqs: list[float | None] = [tonic_hz if role == "kick" else None]
                        else:
                            freqs = list(_tonal_freqs(token, role, tr.name, prev_voicing))
                            if role == "bass" and len(freqs) == 1:
                                pos = bar_global + st.start
                                prev = last_bass.get(tr.name)
                                if prev is not None and abs(prev[0] - pos) < 1e-4:
                                    glide_from = prev[1]          # 808 legato glide
                                last_bass[tr.name] = (pos + st.dur, float(freqs[0]))
                        for f in freqs:
                            t_off, v_adj = apply_groove(
                                onset_beats, role=role, bpm=float(session.bpm),
                                g=g, rng=rng, velocity=vel)
                            events.append(NoteEvent(
                                t_s=bar_start + t_off, track=tr.name, role=role,
                                freq_hz=f, dur_s=dur_s, velocity=v_adj,
                                glide_from_hz=glide_from, params=params,
                                gain=sec_gain))
                bar_global += 1
    events.sort(key=lambda e: (e.t_s, e.track, e.freq_hz or 0.0))
    return events


def compile_events(session: Session) -> list[NoteEvent]:
    """Compile the whole session to time-sorted NoteEvents (deterministic)."""
    return _compile(session, np.random.default_rng(int(session.seed)))


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _apply_track_fx(buf: np.ndarray, tr: Track, bpm: float) -> np.ndarray:
    """Mono buffer -> track fx chain -> constant-power-panned stereo (N, 2).

    Chain: hpf/lpf -> drive -> delay -> chorus -> pan. (SPEC lists chorus
    before delay; delay is mono so it runs first to preserve chorus stereo.)
    """
    fxd = tr.fx or {}
    x = buf
    if float(fxd.get("hpf_hz", 0.0)) > 0.0:
        x = _fx.svf_highpass(x, SR, float(fxd["hpf_hz"])).astype(np.float64)
    if float(fxd.get("lpf_hz", 0.0)) > 0.0:
        x = _fx.svf_lowpass(x, SR, float(fxd["lpf_hz"])).astype(np.float64)
    if float(fxd.get("drive", 0.0)) > 0.0:
        x = _fx.drive(x, float(fxd["drive"])).astype(np.float64)
    if float(fxd.get("delay", 0.0)) > 0.0:
        x = _fx.delay(x.astype(np.float32), SR, bpm,
                      wet=float(fxd["delay"])).astype(np.float64)
    if float(fxd.get("chorus", 0.0)) > 0.0:
        st = _fx.chorus(x.astype(np.float32), SR, float(fxd["chorus"])).astype(np.float64)
    else:
        st = np.stack([x, x], axis=1)
    pan = float(np.clip(tr.pan, -1.0, 1.0))
    theta = (pan + 1.0) * (np.pi / 4.0)
    return st * np.array([np.cos(theta), np.sin(theta)])[None, :]


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-10)))


def render(session: Session, *, out_path: str, movement: int | None = None,
           bars: tuple[int, int] | None = None,
           stems_dir: str | None = None) -> dict:
    """Offline render to a PCM_24 WAV. Returns the manifest dict.

    ``movement``: 0-based movement index; ``bars``: (start, stop) 0-based
    half-open bar window *within* the selected scope. Stems (post-fx,
    pre-master) are written per track when ``stems_dir`` is given.
    """
    rng = np.random.default_rng(int(session.seed))
    events = _compile(session, rng)

    bar_s = 240.0 / float(session.bpm)
    counts = [sum(s.bars for s in mv.sections) for mv in session.movements]
    total = sum(counts)
    if total <= 0:
        raise ValueError("session has no bars to render")
    if movement is not None:
        if not 0 <= movement < len(session.movements):
            raise ValueError(f"movement {movement} out of range "
                             f"(0..{len(session.movements) - 1})")
        start_bar, n_bars = sum(counts[:movement]), counts[movement]
    else:
        start_bar, n_bars = 0, total
    if bars is not None:
        a, b = int(bars[0]), int(bars[1])
        if a < 0 or b > n_bars or b <= a:
            raise ValueError(f"bars window {bars!r} out of range (0..{n_bars})")
        start_bar += a
        n_bars = b - a

    t0 = start_bar * bar_s
    t1 = t0 + n_bars * bar_s
    tol = 0.025                       # groove jitter can pull an onset early
    win = [ev for ev in events if (t0 - tol) <= ev.t_s < (t1 - 1e-9)]
    n = int(round((n_bars * bar_s + TAIL_S) * SR))
    kick_times = sorted(ev.t_s - t0 for ev in win
                        if ev.role == "kick" and ev.t_s - t0 >= 0.0)
    master = session.master or {}

    mix = np.zeros((n, 2))
    send = np.zeros(n)                # reverb send bus (mono)
    track_st: dict[str, np.ndarray] = {}
    for tr in session.tracks:
        evs = [ev for ev in win if ev.track == tr.name]
        buf = np.zeros(n)
        for ev in evs:
            y = render_note(tr.voice, sr=SR, dur_s=ev.dur_s, freq_hz=ev.freq_hz,
                            velocity=ev.velocity, rng=rng, params=ev.params,
                            glide_from_hz=ev.glide_from_hz).astype(np.float64)
            y *= float(getattr(ev, "gain", 1.0))
            i0 = int(round((ev.t_s - t0) * SR))
            if i0 >= n:
                continue
            if i0 < 0:
                y = y[-i0:]
                i0 = 0
            j = min(n, i0 + y.size)
            if j > i0:
                buf[i0:j] += y[: j - i0]
        buf *= 10.0 ** (float(tr.gain_db) / 20.0)
        if evs:
            st = _apply_track_fx(buf, tr, float(session.bpm))
            if float((tr.fx or {}).get("duck", 0.0)) > 0.0 and kick_times:
                st = _fx.ducker(st.astype(np.float32), SR, kick_times,
                                float(tr.fx["duck"])).astype(np.float64)
        else:
            st = np.zeros((n, 2))
        track_st[tr.name] = st
        mix += st
        send_amt = float((tr.fx or {}).get("reverb", 0.0))
        if send_amt > 0.0:
            send += send_amt * st.mean(axis=1)

    # master chain: gain -> reverb send -> drive -> lofi -> limiter
    # (v0.1 amendment: master.gain_db is a real gain stage, default 0 dB;
    # there is NO implicit normalization anywhere in the path)
    mg = float(master.get("gain_db", 0.0))
    if mg != 0.0:
        g = 10.0 ** (mg / 20.0)
        mix *= g
        send *= g
    mr = float(master.get("reverb", 0.0))
    if mr > 0.0:
        send += mr * mix.mean(axis=1)
    if np.any(send != 0.0):
        fv = _fx.freeverb(send.astype(np.float32), SR, 1.0).astype(np.float64)
        mix = mix + (fv - send[:, None])          # wet-only from the send bus
    md = float(master.get("drive", 0.0))
    if md > 0.0:
        mix = _fx.drive(mix, md).astype(np.float64)
    ml = float(master.get("lofi", 0.0))
    if ml > 0.0:
        mix = _fx.lofi(mix.astype(np.float32), SR, ml).astype(np.float64)
    pre_limiter_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    out = _fx.limiter(mix, float(master.get("limit_db", -1.0)))
    post_peak = float(np.max(np.abs(out))) if out.size else 0.0
    limiter_gr_db = max(0.0, _db(pre_limiter_peak) - _db(post_peak))

    out_path = str(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    sf.write(out_path, out, SR, subtype="PCM_24")

    if stems_dir is not None:
        os.makedirs(stems_dir, exist_ok=True)
        headroom = 10.0 ** (-3.0 / 20.0)          # stems keep -3 dBFS headroom
        for tname, st in track_st.items():
            speak = float(np.max(np.abs(st))) if st.size else 0.0
            if speak > headroom:
                st = st * (headroom / speak)
            sf.write(os.path.join(stems_dir, f"{tname}.wav"),
                     st.astype(np.float32), SR, subtype="PCM_24")

    return {
        "out": out_path,
        "duration_s": n / SR,
        "bars": n_bars,
        "events": len(win),
        "peak_db": _db(post_peak),
        "rms_db": _db(np.sqrt(np.mean(out.astype(np.float64) ** 2)) if out.size else 0.0),
        "limiter_gain_reduction_db": round(limiter_gr_db, 4),
        "per_track_rms_db": {
            tname: _db(np.sqrt(np.mean(st ** 2)))
            for tname, st in track_st.items()
        },
    }


def play(session: Session, **render_kwargs) -> None:
    """Render to a temp WAV then play through the default output device."""
    import tempfile

    render_kwargs.pop("out_path", None)
    # Sound out of a machine is a physical act in someone's room: playback is
    # opt-in when nothing is attached to the terminal.  See realtime.py's
    # AUDIO_CONSENT_ENV note for the incident that put this here.
    from .realtime import check_audio_consent
    check_audio_consent()
    from .realtime import register_session, sigint_stops_this, unregister_session
    # A render can take a while.  `plume stop` has to reach it, or the sound starts after.
    with sigint_stops_this():
        register_session(kind="play")            # so `plume stop` can find it
        try:
            try:
                import sounddevice as sd
            except Exception as exc:             # pragma: no cover - optional dep
                raise RuntimeError(
                    "playback needs the optional 'sounddevice' package "
                    "(pip install sounddevice)") from exc
            with tempfile.TemporaryDirectory(prefix="plume-play-") as td:
                path = os.path.join(td, "take.wav")
                render(session, out_path=path, **render_kwargs)
                data, sr = sf.read(path, dtype="float32")
                try:
                    sd.play(data, sr)
                    sd.wait()
                except Exception as exc:         # pragma: no cover - env dependent
                    raise RuntimeError(
                        f"audio playback failed (no output device?): {exc}") from exc
        finally:
            unregister_session()
