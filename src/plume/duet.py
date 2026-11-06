"""plume.duet — bar-by-bar accompanist (SPEC-V1 section 5).

``DuetEngine.render_bar(plan, beat, harmony)`` returns stereo float32 of
exactly one bar at ``beat.bpm``:

- **Bar-length exactness**: the ideal bar length ``4 * 60/bpm * sr`` is
  rounded to samples with the fractional remainder carried into the next
  bar, so N consecutive bars accumulate zero drift at any bpm.
- **Overlap/tail buffer**: voice tails (release, decay) may exceed the bar;
  an internal float64 overlap buffer carries them into the next returned
  bar (or ``flush_tail()``), so consecutive bars join without clicks —
  concatenating rendered bars is bitwise the same as a continuous render.
- **Graceful harmonic degradation** (:func:`choose_voicing`): chord belief
  with high confidence -> triad voicing (voice-led); low confidence ->
  root+fifth; no chord -> tonic pedal of the key belief; no key ->
  ``key_fallback`` pedal tone.
- **Register ceiling**: tonal voicings are octave-dropped below
  ``plan.register_ceiling_midi`` when set.
- **Density -> sparser patterns**: drum-grid variants from the style are
  ranked by onset count and selected by ``plan.density``; very low density
  additionally thins to on-beat hits; bass/pad handlers pick sparser
  figures at low density.
- **Style roles** come from the style file's ``voices`` map (styles_lib);
  role levels in ``plan.roles`` scale each role's bus linearly (0 = tacet).
- **Gentle entrances and exits** (v1.4): the machine bus carries a musical
  envelope on top of the plan's per-bar gain — see below.

Gentle entrances and exits (v1.4)
---------------------------------
The first session with a human and an acoustic guitar produced the verdict
"it should have a softer fade in, music is gentle and cross faded".  Entry
used to be a per-bar gain staircase that started and stopped on a bar edge,
so an entrance read as a hard cut on sustained material.  Four mechanisms
replace that, all of them driven by *musical* durations that come from the
style file (``style["fades"]``), not from code:

1. **Equal-power fade-in / fade-out** across ``fade_in_bars`` /
   ``fade_out_bars``.  The envelope is a phase in [0, 1] advanced one bar at
   a time, so a two-bar fade is one smooth curve across two ``render_bar``
   calls, not a per-bar step.  It is applied to the *returned* bar only —
   the overlap/tail buffer keeps raw samples — which makes the bus exactly
   ``continuous_render * env(t)`` with ``env`` continuous across every bar
   join.  An interrupted fade reverses from the gain it had reached
   (``fx.phase_for_gain``), never from a jump.
2. **Texture crossfade.**  When the role set, density or register ceiling
   changes materially between two sounding bars, the bar is rendered twice —
   once with the *previous* texture, once with the new — and the two are
   equal-power crossfaded over ``texture_xfade_beats``.  A density or role
   change is a blend, not a splice.  (This costs one extra bar render, and
   only on the bars where the texture actually moves.)
3. **Bus-gain glide.**  A bar whose ``gain_db`` differs from the previous
   one starts at the previous level and slides to its own over the same
   window, so the entry ramp and the dynamic arc stop stepping at bar edges.
4. **Announce gestures.**  When ``plan.announce`` is set, the announcing bar
   carries a quiet gesture on the believed harmony, shaped to start and end
   at exactly zero inside the bar, sitting in the last ``announce_beats`` of
   it.  That is the reciprocal cue RESEARCH.md asks for: the human hears the
   change coming and can make room for it.  Two kinds are voiced, and they
   are deliberately different gestures because they answer different
   questions:

   * ``"entry"`` -> a **pickup** on the bottom of the voicing, rendered only
     when the bar is otherwise tacet.  "I am about to come in."
   * ``"texture"`` -> a **swell** an octave over the top of the voicing
     (register-ceiling capped), rendered *over* whatever the bar is already
     playing.  "What I am playing is about to change."  Without it a texture
     announcement was a promise the renderer did not keep: the mind's reason
     string said ``announcing texture next bar (swell)`` and the change
     arrived as a crossfade, which is a blend — something you notice only
     after it has happened, not a cue you can play towards.  v1.4.1.

     Two things make it a swell rather than a gesture named one.  It is a
     crescendo **roll**, not one note under a rising window: a rising window
     swells a sustaining voice and deletes the attack of a plucked one, and
     role names do not carry timbre (lofi's ``lead`` is a pluck, synthwave's
     a supersaw), so the shaped single note measured as a *diminuendo* on
     lofi — peak at 10% of the gesture, monotone decay after.  And its level
     is **measured against the band it shares** — ``announce_gain_db`` under
     the mono-sum RMS of everything else in its window — because a fixed
     offset on the announcing role's mix gain is not that quantity: the same
     -18 dB landed -28 dB under an e-piano pad and -48.5 dB under a pluck,
     and at -48 dB the cue moved the bar's level by -0.00 dB.

   ``"exit"`` is NOT voiced as a separate gesture: the exit already *is* an
   audible gesture (the fade-out of mechanism 1 begins on the announced bar's
   successor and the gain has begun dropping), and stacking a cue on top of a
   fade reads as a fault, not a courtesy.

   ``announce`` / ``announce_kind`` are read with ``getattr`` defaults, so a
   ``BarPlan`` that does not carry them yet simply never announces.

With no fade running, no texture change and nothing announced, the envelope
is identically 1.0 and is not applied at all — the bar is bit-identical to
the pre-v1.4 render, which is what keeps the bar-join/continuity and
determinism contracts above exactly as they were.

Reuses v0 synthesis exclusively (voices.render_note, groove.apply_groove,
theory voicing, per-role svf filters from fx).  Time-based per-track fx
(delay/chorus/reverb/ducker) are deliberately not run inside the bar loop:
they are stateful across bars and would break the exact overlap-add
identity at bar joins; the offline mix bus (sim) owns any global fx.

Determinism: pure function of (seed, call sequence).  Each bar uses its own
``np.random.default_rng((seed, f(plan.bar)))`` stream, so a bar's audio
depends only on (seed, plan, beat, harmony) plus the voice-leading memory
accumulated over prior calls.

This module intentionally does NOT import ``plume.mind`` at runtime —
``BarPlan`` is duck-typed (attribute access only) so the duet engine and
its tests stand alone while the mind is built concurrently.
"""
from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import dataclasses

import numpy as np

from plume import figures
from plume import fx as _fx
from plume.governor import progression_symbols
from plume.groove import GrooveParams, apply_groove
from plume.styles_lib import load_style
from plume.theory import (
    Key,
    midi_to_hz,
    parse_chord,
    parse_key,
    voice_chord,
)
from plume.voices import render_note

if TYPE_CHECKING:  # contracts only; never imported at runtime
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarPlan

__all__ = ["DuetEngine", "choose_voicing", "CHORD_CONF_FULL", "FADE_DEFAULTS"]

# chord_conf at or above this -> full triad voicing; below -> root+fifth
CHORD_CONF_FULL = 0.6

#: Fallback for ``style["fades"]``.  Every bundled style ships its own block
#: (lofi/synthwave fade long, trap stays punchy); these are only used for a
#: style file written before v1.4.  Durations are MUSICAL, not seconds.
FADE_DEFAULTS: dict[str, float] = {
    "fade_in_bars": 1.0,        # bars to rise from silence to full level
    "fade_out_bars": 1.0,       # bars to fall from full level to silence
    "texture_xfade_beats": 1.0, # crossfade window for a texture/gain change
    # Announce level.  For the texture swell this is dB under the BAND the
    # cue shares (measured); for the entry pickup, whose bar is tacet, it is
    # dB under the announcing role's bus (absolute).
    "announce_gain_db": -18.0,
    "announce_beats": 2.0,      # length of either announce gesture
}

# A texture difference smaller than these is not worth a crossfade: the
# crossfade costs a second render of the bar, and inaudible churn would pay
# that on every bar of a steady passage.
_XFADE_LEVEL_EPS = 0.15       # per-role level change
_XFADE_DENSITY_EPS = 0.05     # density change
_XFADE_CEILING_EPS = 0.5      # register-ceiling change, in semitones

# Bus-gain steps: below the first is inaudible, above the second is a state
# change (beat loss) rather than a fader move and should not be smeared.
_GLIDE_MIN_DB = 0.25
_GLIDE_MAX_DB = 12.0

# Fraction of the announce gesture spent swelling; the rest is its release.
# The texture swell rises for longer than the entry pickup: a pickup is a
# breath before a downbeat, a swell is a lift towards a change, and the two
# must not sound like the same cue.
_ANNOUNCE_SWELL_FRAC = 0.7
_TEXTURE_SWELL_FRAC = 0.88

# Role preference per announce kind.  Entry speaks from the bottom of the
# voicing on whatever sustains; texture speaks from above it, so a plucked /
# lead voice is preferred — it has to be heard THROUGH a bar that is already
# playing, and a second sustained pad note would only thicken the pad.
_ANNOUNCE_ROLES: dict[str, tuple[str, ...]] = {
    "entry": ("pad", "keys", "lead", "bass"),
    "texture": ("lead", "arp", "pad", "keys"),
}
# Semitones above the top of the voicing for the texture swell.  An octave —
# the same interval `_lead_events` already uses — so it is clear of the pad
# without being a new harmonic claim.  Still passed through `_apply_ceiling`:
# the cue is a courtesy, and stepping over the human's register to deliver it
# would defeat the point.
_TEXTURE_SWELL_ST = 12

# The texture swell is a crescendo ROLL, not one long note under a rising
# window.  A rising window swells a sustaining voice and does the opposite to
# a plucked one — it deletes the attack and leaves the decay, which measured
# on lofi (whose `lead` is a pluck) peaks at 10% of the gesture and falls
# monotonically from there: the promised "swell" was audibly a diminuendo.
# Role names cannot tell the two apart — lofi's `lead` is a pluck, synthwave's
# is a supersaw, trap's `pad` is a pluck — so the gesture is built to rise on
# either, by re-articulating at a musical rate with the velocity climbing.
_TEXTURE_ROLL_PER_BEAT = 2.0        # re-articulations per beat
_TEXTURE_ROLL_VEL = (0.35, 0.62)    # velocity of the first and last hit
_TEXTURE_ROLL_OVERLAP = 1.6         # each hit's length, in roll steps

# Engine state that a re-render of the SAME bar must roll back (the live
# planner re-renders a bar when the predicted downbeat moves).
_REVISABLE = (
    "_env_mode", "_env_phase", "_env_gain", "_env_fade_bars",
    "_prev_sounding", "_prev_texture", "_prev_gain_db", "_prev_fields",
    "_last_bar_len",
)

# density thresholds for figure selection
_DENSITY_BEATS_ONLY = 0.25    # drums: thin the sparsest grid to on-beat hits
_BASS_WHOLE = 0.3             # bass: whole-note root below this
_BASS_HALVES = 0.65           # bass: beats 1+3 below this, style figure above
_PAD_ACTIVE = 0.5             # pad: stabs/arp figures at/above this

_DEFAULT_VELOCITY = 0.8
#: From this density a motif plays through its phrases, with a longer echo.
_MELODY_LIFT = 0.75

# per-role mix defaults (dB) and pans — mirrors the v0 compose tables
_ROLE_GAIN_DB = {
    "kick": 0.0, "snare": -2.0, "clap": -4.0, "hat": -8.0, "openhat": -9.0,
    "bass": -1.0, "pad": -7.0, "keys": -7.0, "lead": -6.0, "arp": -9.0,
    "crackle": -18.0, "tom": -5.0, "crash": -14.0,
}
_ROLE_PAN = {"hat": 0.25, "openhat": 0.2, "clap": -0.1, "lead": -0.15,
             "arp": 0.2, "tom": -0.2, "crash": -0.3}


_CHORUS_PEAK_ROOM = 2.0 ** 0.5


def _hold_level(wet: np.ndarray, dry: np.ndarray, peak_room: float = 1.0) -> np.ndarray:
    """``wet`` at the RMS of ``dry``, its peak no more than ``peak_room`` times that of ``dry``
    and never past full scale because of this."""
    wet_rms = float(np.sqrt(np.mean(wet ** 2))) if wet.size else 0.0
    if wet_rms <= 0.0:
        return wet
    dry_peak = float(np.abs(dry).max())
    ceiling = max(dry_peak, min(1.0, peak_room * dry_peak))
    return wet * min(float(np.sqrt(np.mean(dry ** 2))) / wet_rms,
                     ceiling / float(np.abs(wet).max()))


def _saturate(y: np.ndarray, amount: float) -> np.ndarray:
    """fx.drive at the note's own level: the peaks round off, so it is as loud as before
    with a lower peak.  This drives one note at a time, where an offline render drives the
    whole mix."""
    return _hold_level(_fx.drive(y, amount).astype(np.float64), y)


def _chorus(y: np.ndarray, sr: int, depth: float, at_s: float) -> np.ndarray:
    """fx.chorus on one note, level held.  ``at_s`` is when the note plays: the wobble carries
    on from there, as it does across a whole part offline, so a pitch that falls in a dip of
    the delay on one note does not fall in it on every note.  Offline the part goes through
    one chorus and overlapping notes share it; here each note has its own.  Half the note goes
    through the delay, so the level comes back at about 3 dB and the peak gets that much room."""
    wet = _fx.chorus(y.astype(np.float32), sr, depth, phase_s=at_s).astype(np.float64)
    return _hold_level(wet, y, peak_room=_CHORUS_PEAK_ROOM)


# roles Mind may name that a style spells differently
_ROLE_ALIASES = {"keys": "pad"}

_MIDI_FLOOR = 24


def _strip_vel(token: str) -> tuple[str, float | None]:
    """Split the notation ':v<float>' velocity suffix off a token."""
    if ":v" in token:
        base, _, v = token.partition(":v")
        return base, float(v)
    return token, None


def _low_octave_hz(pc: int) -> float:
    """Pitch class -> frequency in (or nearest to) the 35-55 Hz octave."""
    cands = [midi_to_hz(12 * (o + 1) + pc) for o in (0, 1, 2)]
    inside = [h for h in cands if 35.0 <= h <= 55.0]
    if inside:
        return inside[0]
    return min(cands, key=lambda h: min(abs(h - 35.0), abs(h - 55.0)))


def _near_low(pc: int, low_midi: int = 48) -> int:
    return low_midi + ((pc - low_midi) % 12)


def _apply_ceiling(notes: list[int], ceiling: float | None) -> list[int]:
    """Octave-drop every note to or below the ceiling (floor at midi 24)."""
    if ceiling is None:
        return sorted(notes)
    out = []
    for m in notes:
        while m > ceiling and m - 12 >= _MIDI_FLOOR:
            m -= 12
        out.append(m)
    return sorted(set(out))


def choose_voicing(
    chord: str | None,
    chord_conf: float,
    key: str | None,
    *,
    key_fallback: Key,
    prev: list[int] | None = None,
    ceiling_midi: float | None = None,
) -> tuple[list[int], str]:
    """Graceful harmonic degradation (SPEC-V1 section 5) -> (midi notes, tier).

    Tiers:
      ``triad``          chord belief present and chord_conf >= CHORD_CONF_FULL
      ``root_fifth``     chord belief present but confidence below that
      ``key_pedal``      no chord belief; tonic pedal of the believed key
      ``fallback_pedal`` no key belief either; tonic pedal of key_fallback
    """
    if chord is not None:
        try:
            ch = parse_chord(chord)
        except ValueError:
            ch = None
        if ch is not None:
            if chord_conf >= CHORD_CONF_FULL:
                notes = voice_chord(ch, low_midi=48, prev=prev)
                return _apply_ceiling(notes, ceiling_midi), "triad"
            r = _near_low(ch.root)
            return _apply_ceiling([r, r + 7], ceiling_midi), "root_fifth"
    if key is not None:
        try:
            k = parse_key(key)
        except ValueError:
            k = None
        if k is not None:
            return _apply_ceiling([_near_low(k.tonic)], ceiling_midi), "key_pedal"
    return (_apply_ceiling([_near_low(key_fallback.tonic)], ceiling_midi),
            "fallback_pedal")


def _tonic_pc(harmony, key_fallback: Key) -> int:
    """Pitch class of the believed key's tonic (fallback key if none)."""
    key = getattr(harmony, "key", None)
    if key is not None:
        try:
            return parse_key(key).tonic
        except ValueError:
            pass
    return key_fallback.tonic


def _root_pc(harmony, key_fallback: Key) -> int:
    """Pitch class of the current chord root (key tonic if no chord)."""
    chord = getattr(harmony, "chord", None)
    if chord is not None:
        try:
            return parse_chord(chord).root
        except ValueError:
            pass
    return _tonic_pc(harmony, key_fallback)


class _PlanEcho:
    """Duck-typed stand-in for the PREVIOUS bar's texture at the CURRENT bar.

    ``render_bar`` renders this alongside the real plan to crossfade the old
    texture out; it carries the previous bar's roles/density/ceiling/gain but
    the current bar number, so the drum cycle and the per-bar rng stay in
    step with where the music actually is.
    """

    __slots__ = ("bar", "state", "roles", "gain_db", "density",
                 "register_ceiling_midi", "reason", "announce",
                 "announce_kind")

    def __init__(self, bar: int, fields: dict):
        self.bar = int(bar)
        self.state = "comping"
        self.roles = dict(fields["roles"])
        self.gain_db = float(fields["gain_db"])
        self.density = float(fields["density"])
        self.register_ceiling_midi = fields["ceiling"]
        self.reason = "texture crossfade: previous texture, fading out"
        self.announce = False           # never re-announce on the echo
        self.announce_kind = ""


class DuetEngine:
    """Bar-by-bar accompanist over one style (SPEC-V1 section 5)."""

    def __init__(self, *, style: "str | dict", key_fallback: str = "A minor",
                 seed: int, sr: int = 48000,
                 fade_in_bars: float | None = None,
                 fade_out_bars: float | None = None,
                 texture_xfade_beats: float | None = None,
                 announce_gain_db: float | None = None,
                 announce_beats: float | None = None):
        # A style is a bundled one's name, or a style already loaded and changed (the
        # plain beat a detected take starts on: bandentry.plain_intro_style).
        self._style = (load_style(style) if isinstance(style, str)
                       else json.loads(json.dumps(style)))
        self._key_fallback = parse_key(key_fallback)
        self._seed = int(seed)
        self._sr = int(sr)
        self._frac = 0.0                          # fractional-sample carry
        self._tail = np.zeros((0, 2), dtype=np.float64)
        self._prev_voicing: list[int] | None = None
        self._groove = GrooveParams(
            swing=float(self._style["swing"]),
            timing_bias_ms={str(k): float(v)
                            for k, v in self._style["timing_bias_ms"].items()},
            humanize_ms=3.0,
            humanize_vel=0.05,
        )
        # drum grid variants per role, sorted sparse -> busy (by onset count)
        from plume import notation
        self._notation = notation
        self._grids: dict[str, list[list]] = {}
        #: Pitch classes the player has been heard on, for the melody to walk.
        self.heard_notes: tuple[int, ...] = ()
        for role, variants in self._style["drum_patterns"].items():
            parsed = [notation.parse(str(p)) for p in variants]
            parsed.sort(key=lambda steps: (len(steps), tuple(
                (s.start, s.token) for s in steps)))
            self._grids[str(role)] = parsed
        self._slow_grids: dict[str, list[list]] = {
            str(role): [notation.parse(str(p)) for p in variants]
            for role, variants in self._style.get("slow_drum_patterns", {}).items()}
        # introspection: events chosen for the most recent render_bar
        self.last_events: list[dict] = []
        self._progressions: dict[str, list[str]] = {}

        # ---- gentle entrances/exits (v1.4) --------------------------------
        # Style data first, constructor override second (tests and the live
        # tuner need to be able to say "no fade" without editing a JSON).
        f = dict(FADE_DEFAULTS)
        f.update({str(k): float(v)
                  for k, v in (self._style.get("fades") or {}).items()})
        self.fade_in_bars = float(f["fade_in_bars"] if fade_in_bars is None
                                  else fade_in_bars)
        self.fade_out_bars = float(f["fade_out_bars"] if fade_out_bars is None
                                   else fade_out_bars)
        self.texture_xfade_beats = float(
            f["texture_xfade_beats"] if texture_xfade_beats is None
            else texture_xfade_beats)
        self.announce_gain_db = float(
            f["announce_gain_db"] if announce_gain_db is None
            else announce_gain_db)
        self.announce_beats = float(
            f["announce_beats"] if announce_beats is None else announce_beats)

        # envelope: a phase in [0,1] through the current fade, plus the gain
        # it will hold once the fade ends (1.0 = sounding, 0.0 = silent).
        self._env_mode: str | None = None       # "in" | "out" | None
        self._env_phase = 0.0
        self._env_gain = 1.0
        self._env_fade_bars = 0.0
        # transition tracking (None = "no bar rendered yet", not "silent")
        self._prev_sounding: bool | None = None
        self._prev_texture: tuple | None = None
        self._prev_gain_db: float | None = None
        self._prev_fields: dict | None = None
        self._last_bar_len = 0
        # re-render support: the live planner may render one bar twice
        self._last_bar_rendered: int | None = None
        self._resume: dict | None = None
        # introspection for the most recent render_bar
        self.last_xfade_samples = 0
        self.last_env_gain_rms = 1.0
        #: Mean square of the bar this engine last returned, measured BEFORE
        #: the bus envelope (fade / glide) was applied — i.e. the level the
        #: bar will hold once any fade completes.  This is what a caller
        #: calibrating the machine bus against the human must measure: the
        #: delivered bar's own RMS is the steady level TIMES the fade, so
        #: calibrating on it inside a fade-in solves for the wrong number and
        #: over-boosts the whole take (v1.4.1; measured at +3.2 dB overall,
        #: +1.35 dB steady-state, on the `steady` fixture).  It is a pure
        #: function of (seed, plan, beat, harmony) and the voice-leading
        #: memory — never of the fade state — so a calibration taken from it
        #: is identical whether or not a fade happens to be running.
        self.last_bar_ms_pre_env = 0.0
        self.last_announce: dict | None = None

    # ------------------------------------------------------------------ api

    @property
    def roles(self) -> tuple[str, ...]:
        """Role names this style can voice (from the style's voices map)."""
        return tuple(str(r) for r in self._style["voices"])

    def render_bar(self, plan: "BarPlan", beat: "BeatState",
                   harmony: "HarmonyState", *,
                   continuation: bool = False) -> np.ndarray:
        """Render exactly one bar at ``beat.bpm`` -> stereo float32 (n, 2).

        The bar is the raw overlap-added render multiplied by the machine
        bus envelope (fade in/out, gain glide) and, when the texture moved,
        equal-power crossfaded with the previous texture.  With no fade
        running and no texture change both are the identity.

        ``continuation`` (v1.4.1) says "this bar follows the last one on the
        bus, even though it carries the same bar number".  It disambiguates
        the two callers that can repeat a bar index, which want opposite
        things:

        * the live planner RE-RENDERS a bar when the predicted downbeat
          moves — the earlier render is thrown away, so the envelope must be
          rolled back (``continuation=False``, the default);
        * the mind's held wobble bars (``_plan_unlock_hold``) arrive at the
          unlocked tick cadence with the beat counter FROZEN, so up to four
          consecutive sounding bars share one bar number — and every one of
          them is kept and placed end to end.  Rolling the envelope back for
          those replays the same slice of the fade over and over, which is
          heard as an entrance that keeps restarting.  They pass
          ``continuation=True``.
        """
        bpm = float(beat.bpm)
        if not np.isfinite(bpm) or bpm <= 0.0:
            raise ValueError(f"render_bar requires beat.bpm > 0, got {bpm}")
        bar_idx = int(getattr(plan, "bar", 0))

        # A revision of the bar we just rendered must not advance the
        # envelope a second time: roll the fade state back and redo it.
        if (not continuation and self._last_bar_rendered == bar_idx
                and self._resume is not None):
            self._restore_fade(self._resume)
        else:
            self._resume = self._snapshot_fade()

        n = self._next_len(bpm)
        self._last_bar_len = n
        sounding = self._plan_sounds(plan)
        self._update_fade(plan, sounding)

        # --- render, crossfading the previous texture out if it moved -----
        texture = self._texture_of(plan)
        xfade_n = self._xfade_samples(plan, sounding, texture, bpm, n)
        prev_voicing = (list(self._prev_voicing)
                        if self._prev_voicing is not None else None)
        raw = self._render_raw(plan, beat, harmony, n)
        if xfade_n > 0 and self._prev_fields is not None:
            # The echo render is a measurement, not the plan: its voice-leading
            # memory and its introspection are discarded, the real plan's kept.
            keep = (self._prev_voicing, self.last_events, self.last_announce)
            self._prev_voicing = prev_voicing
            old = self._render_raw(_PlanEcho(bar_idx, self._prev_fields),
                                   beat, harmony, n)
            self._prev_voicing, self.last_events, self.last_announce = keep
            # Two textures that share most of their voices are strongly
            # correlated; equal-power would swell by up to +3 dB across the
            # window.  Measure it rather than guess (see fx.crossfade).
            raw = _fx.crossfade(old, raw, xfade_n,
                                correlation=_fx.correlation(old, raw, xfade_n))
        self.last_xfade_samples = xfade_n

        # --- overlap-add; the retained tail stays RAW ----------------------
        # The envelope is applied only to what we hand back, so a tail that
        # crosses into the next bar is scaled by the envelope value at its
        # real playback time.  That is what makes the bus exactly
        # ``continuous_render * env(t)`` with no step at any bar join.
        k = self._tail.shape[0]
        buf = np.zeros((max(raw.shape[0], k, n), 2), dtype=np.float64)
        buf[: raw.shape[0]] += raw
        if k:
            buf[:k] += self._tail
        self._tail = buf[n:].copy()
        out = buf[:n]

        # The bar's own level, BEFORE the bus envelope — see the attribute's
        # note in __init__.  Taken here because `out` is still the plain
        # overlap-added render at this point, and measured on the float32 the
        # caller will actually be handed: with no envelope running the
        # returned bar IS `out.astype(np.float32)`, so quantising first makes
        # the two the same number rather than the same number to within
        # float32 rounding.  A calibration solved against it then lands on
        # the delivered samples exactly.
        self.last_bar_ms_pre_env = (
            float(np.mean(out.astype(np.float32).astype(np.float64) ** 2))
            if out.size else 0.0)

        env = self._env_segment(n, n)
        glide = self._glide(plan, sounding, xfade_n, bpm, n)
        if env is not None:
            out = out * env[:, None]
        if glide is not None:
            out = out * glide[:, None]
        gains = [g for g in (env, glide) if g is not None]
        self.last_env_gain_rms = (
            1.0 if not gains
            else float(np.sqrt(np.mean(np.prod(gains, axis=0) ** 2))))

        self._remember(plan, sounding, texture, bar_idx)
        return out.astype(np.float32)

    def flush_tail(self) -> np.ndarray:
        """Remaining voice tail after the last bar -> stereo float32.

        The bus envelope keeps running over the tail, so an exit rings out
        under its own fade instead of being truncated at the last bar edge.
        """
        tail = self._tail
        self._tail = np.zeros((0, 2), dtype=np.float64)
        if tail.shape[0]:
            env = self._env_segment(tail.shape[0], max(1, self._last_bar_len))
            if env is not None:
                tail = tail * env[:, None]
        return tail.astype(np.float32)

    # -------------------------------------------------- gentle entries/exits

    def _snapshot_fade(self) -> dict:
        return {k: getattr(self, k) for k in _REVISABLE}

    def _restore_fade(self, snap: dict) -> None:
        for k, v in snap.items():
            setattr(self, k, v)

    def _plan_sounds(self, plan) -> bool:
        """True if this plan asks for at least one role this style can voice."""
        voices = self._style["voices"]
        for role, level in (getattr(plan, "roles", {}) or {}).items():
            if float(level) <= 0.0:
                continue
            if role in voices or _ROLE_ALIASES.get(str(role), "") in voices:
                return True
        return False

    @staticmethod
    def _texture_of(plan) -> tuple:
        roles = {str(r): float(v)
                 for r, v in (getattr(plan, "roles", {}) or {}).items()
                 if float(v) > 0.0}
        return (roles, float(getattr(plan, "density", 0.5)),
                getattr(plan, "register_ceiling_midi", None))

    @staticmethod
    def _texture_moved(a: tuple, b: tuple) -> bool:
        """Is the difference between two textures worth a crossfade?"""
        ra, da, ca = a
        rb, db, cb = b
        if set(ra) != set(rb):
            return True
        if any(abs(ra[r] - rb[r]) > _XFADE_LEVEL_EPS for r in ra):
            return True
        if abs(da - db) > _XFADE_DENSITY_EPS:
            return True
        if (ca is None) != (cb is None):
            return True
        return (ca is not None
                and abs(float(ca) - float(cb)) > _XFADE_CEILING_EPS)

    def _start_fade(self, direction: str, bars: float,
                    *, from_gain: float | None = None) -> None:
        """Begin (or reverse into) a fade, continuous from the current gain."""
        g = self._env_gain if from_gain is None else float(from_gain)
        if bars <= 0.0:
            # A fade length of zero means NO fade — the v1.3 behavior, where
            # the plan's own gain_db is the whole story.  It must not mean
            # "instantly silent": that would mute a ``fading`` bar the mind
            # still wants heard, and it would stop the fade-disabled render
            # being a usable reference for the faded one.
            self._env_mode = None
            self._env_gain = 1.0
            return
        self._env_mode = direction
        self._env_fade_bars = float(bars)
        self._env_phase = _fx.phase_for_gain(g, direction=direction)
        self._env_gain = g

    def _update_fade(self, plan, sounding: bool) -> None:
        """Decide what the envelope should be doing on this bar.

        Precedence matters.  A sounding bar that the mind has marked as
        ``fading`` (or announced as an exit) is going OUT even though it has
        roles; a sounding bar after silence is coming IN; and a sounding bar
        that arrives while an exit is under way reverses that exit from
        wherever it had got to rather than snapping back to full.
        """
        state = str(getattr(plan, "state", "") or "")
        announce = bool(getattr(plan, "announce", False))
        kind = str(getattr(plan, "announce_kind", "") or "")

        if not sounding:
            if self._prev_sounding:
                # sound -> silence: the ring-out fades instead of stopping
                self._start_fade("out", self.fade_out_bars)
            return

        if state == "fading" or (announce and kind == "exit"):
            if self._env_mode != "out":
                self._start_fade("out", self.fade_out_bars)
            return

        from_silence = (self._prev_sounding is False
                        or (self._prev_sounding is None and state == "entering"))
        if from_silence:
            # Nothing was on the bus, so starting the curve at 0 is exact.
            self._start_fade("in", self.fade_in_bars, from_gain=0.0)
        elif self._env_mode == "out" or (self._env_mode is None
                                         and self._env_gain < 1.0):
            self._start_fade("in", self.fade_in_bars)

    def _env_segment(self, n: int, bar_len: int) -> np.ndarray | None:
        """``n`` samples of bus envelope, advancing the fade clock.

        ``None`` means unity — the caller skips the multiply entirely, which
        is what keeps a no-fade render bit-identical to the raw one.  Fade
        length is measured in BARS, so ``bar_len`` re-derives the per-sample
        phase step every bar and a fade rides a tempo change correctly.
        """
        if n <= 0:
            return None
        if self._env_mode is None:
            return None if self._env_gain == 1.0 else np.full(n, self._env_gain)
        total = max(1.0, self._env_fade_bars * float(bar_len))
        p0 = self._env_phase
        p1 = p0 + n / total
        g = _fx.fade_curve(n, direction=self._env_mode, start=p0, end=p1)
        self._env_phase = p1
        if p1 >= 1.0:
            self._env_gain = 1.0 if self._env_mode == "in" else 0.0
            self._env_mode = None
        else:
            self._env_gain = float(
                _fx.equal_power_gain(p1, direction=self._env_mode))
        return g

    def _xfade_samples(self, plan, sounding: bool, texture: tuple,
                       bpm: float, n: int) -> int:
        """Crossfade window for this bar, 0 when the texture did not move."""
        if not (sounding and self._prev_sounding and self._prev_texture):
            return 0
        if not self._texture_moved(self._prev_texture, texture):
            return 0
        w = int(round(self.texture_xfade_beats * (60.0 / bpm) * self._sr))
        return max(0, min(w, n))

    def _glide(self, plan, sounding: bool, xfade_n: int, bpm: float,
               n: int) -> np.ndarray | None:
        """Bus-gain glide so a per-bar ``gain_db`` step is heard as a move.

        Skipped when the bar is already crossfading — the echo render there
        carries the OLD gain, so the level is continuous without help.
        """
        if xfade_n > 0 or not sounding or not self._prev_sounding:
            return None
        if self._prev_gain_db is None:
            return None
        delta = self._prev_gain_db - float(getattr(plan, "gain_db", 0.0))
        if abs(delta) < _GLIDE_MIN_DB:
            return None
        delta = float(np.clip(delta, -_GLIDE_MAX_DB, _GLIDE_MAX_DB))
        w = int(round(self.texture_xfade_beats * (60.0 / bpm) * self._sr))
        if w <= 0:                      # a zero window means no glide at all
            return None
        return _fx.db_glide(n, delta, min(w, n))

    def _remember(self, plan, sounding: bool, texture: tuple,
                  bar_idx: int) -> None:
        self._prev_sounding = sounding
        self._prev_texture = texture
        self._prev_gain_db = float(getattr(plan, "gain_db", 0.0))
        self._prev_fields = {
            "roles": dict(getattr(plan, "roles", {}) or {}),
            "gain_db": float(getattr(plan, "gain_db", 0.0)),
            "density": float(getattr(plan, "density", 0.5)),
            "ceiling": getattr(plan, "register_ceiling_midi", None),
        }
        self._last_bar_rendered = bar_idx

    def _swell_roll(self, preset: str, freq_hz: float, m: int, rng
                    ) -> np.ndarray:
        """``m`` samples of a crescendo roll on one pitch — see the constants.

        Re-articulating is what lets the gesture rise on a plucked voice as
        well as a sustaining one, and it is what a player actually does to
        swell an instrument that cannot sustain.
        """
        hits = max(2, int(round(self.announce_beats * _TEXTURE_ROLL_PER_BEAT)))
        step = m / hits
        dur_s = max(0.02, (step * _TEXTURE_ROLL_OVERLAP) / self._sr)
        lo, hi = _TEXTURE_ROLL_VEL
        out = np.zeros(m, dtype=np.float64)
        for k in range(hits):
            y = render_note(preset, sr=self._sr, dur_s=dur_s, freq_hz=freq_hz,
                            velocity=lo + (hi - lo) * (k / (hits - 1)),
                            rng=rng, params={}).astype(np.float64)
            a = int(round(k * step))
            b = min(m, a + y.shape[0])
            if b > a:
                out[a:b] += y[: b - a]
        return out

    @staticmethod
    def _band_rms(placed: list[tuple[int, np.ndarray]], a: int, b: int
                  ) -> float:
        """RMS of the mono-sum of everything already placed in ``[a, b)``.

        The bar the cue has to be heard through, measured over the cue's own
        window.  Zero on a tacet bar, which is the entry pickup's case.
        """
        n = b - a
        if n <= 0 or not placed:
            return 0.0
        buf = np.zeros((n, 2), dtype=np.float64)
        for i0, st in placed:
            lo = max(a, i0)
            hi = min(b, i0 + st.shape[0])
            if hi > lo:
                buf[lo - a:hi - a] += st[lo - i0:hi - i0]
        return float(np.sqrt(np.mean(buf.mean(axis=1) ** 2)))

    def _announce_gesture(self, kind: str, voicing: list[int], spb: float,
                          n_bar: int, gain_db: float, ceiling: float | None,
                          rng, placed: list[tuple[int, np.ndarray]]
                          ) -> tuple[np.ndarray, dict] | None:
        """The reciprocal cue for ``kind``, as a quiet gesture in this bar.

        Shaped to start AND end at exactly zero inside the bar, so it leaves
        no ringing tail for the next bar's own fade or crossfade to chop off
        — the cue is a breath before the change, not a note left hanging.

        ``"entry"`` is a pickup from the BOTTOM of the voicing (it plays into
        silence, so it only has to be heard at all).  ``"texture"`` is a
        crescendo roll an octave OVER the top of it (it plays through a bar
        that is already sounding, so it has to be heard past the band), with
        a longer rise so the two gestures are not the same shape.  Both are
        capped by the register ceiling.

        Level.  ``announce_gain_db`` has always been documented as the cue's
        level *under the bar's own bus*, but it used to be applied as a fixed
        offset on the announcing role's mix gain, which is not that number at
        all: the same -18 dB lands about -28 dB under an e-piano pad and
        **-48.5 dB under a pluck** (both measured on the lofi bus), and at
        -48 dB the promised cue is inaudible and moves the bar's level by
        -0.00 dB.  So the texture cue's level is now *measured* against the
        band it shares — ``announce_gain_db`` under the mono-sum RMS of
        everything else in the cue's window — which is portable across
        styles, presets and densities.  The entry pickup keeps the absolute
        level: its bar is tacet, so there is nothing to be under.
        """
        voices = self._style["voices"]
        role = next((r for r in _ANNOUNCE_ROLES.get(kind, ())
                     if r in voices), None)
        if role is None or not voicing:
            return None
        preset = str(voices[role])
        if kind == "texture":
            midi = _apply_ceiling([max(voicing) + _TEXTURE_SWELL_ST],
                                  ceiling)[-1]
            swell_frac = _TEXTURE_SWELL_FRAC
            m = min(int(round(self.announce_beats * spb * self._sr)), n_bar)
            if m < 4:
                return None
            y = self._swell_roll(preset, midi_to_hz(midi), m, rng)
        else:
            midi = int(voicing[0])
            swell_frac = _ANNOUNCE_SWELL_FRAC
            y = render_note(preset, sr=self._sr,
                            dur_s=max(0.05, self.announce_beats * spb),
                            freq_hz=midi_to_hz(midi), velocity=0.55, rng=rng,
                            params={}).astype(np.float64)
            m = min(y.shape[0], n_bar)
            if m < 4:
                return None
            y = y[:m]
        r = max(1, min(m - 1, int(round(m * swell_frac))))
        w = np.empty(m, dtype=np.float64)
        w[:r] = _fx.fade_curve(r, direction="in")
        # closed curve on the release so the last sample is exactly 0.0
        w[r:] = _fx.equal_power_gain(np.linspace(0.0, 1.0, m - r),
                                     direction="out")
        sig = y * w
        own = float(np.sqrt(np.mean(sig ** 2)))
        if not (own > 0.0):
            return None
        i0 = max(0, n_bar - m)
        band = self._band_rms(placed, i0, i0 + m)
        if band > 0.0:
            # `st` below splits the signal equally, so its mono-sum is
            # `sig * gain / sqrt(2)`; solve that against the band.
            gain = (band * 10.0 ** (self.announce_gain_db / 20.0)
                    * math.sqrt(2.0) / own)
        else:
            gain = 10.0 ** ((self.announce_gain_db
                             + _ROLE_GAIN_DB.get(role, -6.0) + gain_db) / 20.0)
        mono = sig * gain
        st = np.repeat(mono[:, None], 2, axis=1) * (1.0 / math.sqrt(2.0))
        info = {"role": "announce", "preset": preset, "voice_role": role,
                "onset_s": i0 / self._sr, "dur_s": m / self._sr,
                "midi": midi, "freq_hz": midi_to_hz(midi),
                "gain_db": self.announce_gain_db, "kind": kind}
        return st, info

    # ------------------------------------------------------------ internals

    def _next_len(self, bpm: float) -> int:
        """Bar length in samples: round to samples, carry the remainder."""
        ideal = 4.0 * (60.0 / bpm) * self._sr + self._frac
        n = int(round(ideal))
        self._frac = ideal - n
        return n

    def _bar_rng(self, bar: int) -> np.random.Generator:
        # non-negative, injective over positive/negative bar numbers
        b = int(bar)
        return np.random.default_rng(
            (self._seed & 0xFFFFFFFF, abs(b) * 2 + (1 if b < 0 else 0)))

    def _render_raw(self, plan: "BarPlan", beat: "BeatState",
                    harmony: "HarmonyState", n: int) -> np.ndarray:
        """One bar's events summed into a float64 (>= n, 2) buffer.

        Pure per bar given (seed, plan, beat, harmony, n) and the current
        voice-leading memory.  The returned buffer starts at the bar start;
        samples past ``n`` are the bar's own tail (handled by render_bar).
        """
        sr = self._sr
        bpm = float(beat.bpm)
        spb = 60.0 / bpm
        bar_s = 4.0 * spb
        rng = self._bar_rng(getattr(plan, "bar", 0))
        density = float(np.clip(getattr(plan, "density", 0.5), 0.0, 1.0))
        ceiling = getattr(plan, "register_ceiling_midi", None)
        gain_db = float(getattr(plan, "gain_db", 0.0))
        roles: dict[str, float] = dict(getattr(plan, "roles", {}) or {})
        voices: dict = self._style["voices"]
        fx_defaults: dict = self._style.get("fx_defaults", {})
        live_fx: dict = self._style.get("live_fx", {})

        chord = getattr(harmony, "chord", None)
        chord_conf = float(getattr(harmony, "chord_conf", 0.0) or 0.0)
        key = getattr(harmony, "key", None)
        voicing, tier = choose_voicing(
            chord, chord_conf, key, key_fallback=self._key_fallback,
            prev=self._prev_voicing, ceiling_midi=ceiling)
        if tier == "triad":
            self._prev_voicing = list(voicing)
        root_pc = _root_pc(harmony, self._key_fallback)
        tonic_pc = _tonic_pc(harmony, self._key_fallback)

        # ---- gather events: (role, preset, onset_beats, dur_s, freq, vel)
        events: list[tuple[str, str, float, float, float | None, float]] = []
        for role in sorted(roles):
            level = float(roles[role])
            if level <= 0.0:
                continue
            preset = voices.get(role) or voices.get(_ROLE_ALIASES.get(role, ""))
            if preset is None:
                continue
            preset = str(preset)
            if role in self._grids:
                events += self._drum_events(role, preset, density, bar_s,
                                            tonic_pc, plan, root_pc=root_pc,
                                            key_known=key is not None)
            elif role == "bass":
                events += self._bass_events(
                    preset, density, spb, root_pc,
                    chord=chord if tier == "triad" else None, key=key,
                    bar=int(getattr(plan, "bar", 0)))
            elif role == "lead":
                if self._style.get("melody", {}).get("motifs"):
                    events += self._motif_events(
                        preset, density, spb, voicing, ceiling,
                        int(getattr(plan, "bar", 0)), tonic_pc, key)
                else:
                    events += self._lead_events(preset, density, spb, voicing,
                                                ceiling)
            elif preset == "crackle" or role == "crackle":
                events.append((role, preset, 0.0, bar_s, None, 0.7))
            else:  # pad / keys / unknown tonal role: chordal figure
                events += self._pad_events(
                    role, preset, density, spb, voicing,
                    chord=chord if tier == "triad" else None, ceiling=ceiling,
                    bar=int(getattr(plan, "bar", 0)))

        # ---- render + place
        self.last_events = []
        placed: list[tuple[int, np.ndarray]] = []
        ducked: list[tuple[int, float]] = []
        end_max = n
        for role, preset, onset_beats, dur_s, freq, base_vel in events:
            level = float(roles.get(role, 0.0))
            onset_s, vel = apply_groove(
                onset_beats, role=role, bpm=bpm, g=self._groove_at(bpm), rng=rng,
                velocity=base_vel)
            onset_s = max(0.0, onset_s)
            y = render_note(preset, sr=sr, dur_s=dur_s, freq_hz=freq,
                            velocity=vel, rng=rng, params={}).astype(np.float64)
            fxd = fx_defaults.get(role, {})
            if float(fxd.get("hpf_hz", 0.0)) > 0.0:
                y = _fx.svf_highpass(y, sr, float(fxd["hpf_hz"])).astype(np.float64)
            if float(fxd.get("lpf_hz", 0.0)) > 0.0:
                y = _fx.svf_lowpass(y, sr, float(fxd["lpf_hz"])).astype(np.float64)
            lfx = live_fx.get(role, {})
            if float(lfx.get("drive", 0.0)) > 0.0:
                y = _saturate(y, float(lfx["drive"]))
            role_db = self._style.get("role_gain_db", {}).get(
                role, _ROLE_GAIN_DB.get(role, -6.0))
            gain = 10.0 ** ((role_db + gain_db) / 20.0)
            y *= gain * level
            pan = float(np.clip(_ROLE_PAN.get(role, 0.0), -1.0, 1.0))
            theta = (pan + 1.0) * (np.pi / 4.0)
            spread = (_chorus(y, sr, float(lfx["chorus"]),
                              int(getattr(plan, "bar", 0)) * bar_s + onset_s)
                      if float(lfx.get("chorus", 0.0)) > 0.0 else y[:, None])
            st = spread * np.array([np.cos(theta), np.sin(theta)])[None, :]
            i0 = int(round(onset_s * sr))
            placed.append((i0, st))
            if float(lfx.get("duck", 0.0)) > 0.0:
                ducked.append((len(placed) - 1, float(lfx["duck"])))
            end_max = max(end_max, i0 + st.shape[0])
            midi = (None if freq is None
                    else int(round(69.0 + 12.0 * np.log2(freq / 440.0))))
            self.last_events.append({
                "role": role, "preset": preset, "onset_s": onset_s,
                "dur_s": dur_s, "freq_hz": freq, "midi": midi,
                "velocity": vel, "level": level, "tier": tier,
            })

        # ---- pump: the same dip the offline ducker gives, from this bar's kicks
        kicks = [e["onset_s"] for e in self.last_events if e["role"] == "kick"]
        if ducked and kicks:
            for k, depth in ducked:
                i0, st = placed[k]
                dip = _fx.ducker(np.ones(i0 + st.shape[0]), sr, kicks, depth)[i0:]
                placed[k] = (i0, st * dip[:, None])

        # ---- reciprocal cue: announce the change in the bar BEFORE it
        # An entry cue is only meaningful on an otherwise tacet bar (if the
        # machine is already playing there is nothing to come in from); a
        # texture cue exists precisely to be heard THROUGH the bar it shares.
        self.last_announce = None
        kind = str(getattr(plan, "announce_kind", "") or "")
        if bool(getattr(plan, "announce", False)) and (
                (kind == "entry" and not events) or kind == "texture"):
            cue = self._announce_gesture(kind, voicing, spb, n, gain_db,
                                         ceiling, rng, placed)
            if cue is not None:
                st, info = cue
                i0 = max(0, n - st.shape[0])
                placed.append((i0, st))
                end_max = max(end_max, i0 + st.shape[0])
                info["onset_s"] = i0 / sr
                self.last_announce = info
                self.last_events.append(info)

        buf = np.zeros((end_max, 2), dtype=np.float64)
        for i0, st in placed:
            buf[i0:i0 + st.shape[0]] += st
        return buf

    # ---- per-role figure builders ----------------------------------------

    def _groove_at(self, bpm: float) -> GrooveParams:
        """The style's feel, or its slow-tempo feel below ``slow_below_bpm``."""
        if bpm >= float(self._style.get("slow_below_bpm", 0.0)):
            return self._groove
        return dataclasses.replace(
            self._groove,
            swing=float(self._style.get("slow_swing", self._groove.swing)),
            timing_bias_ms=dict(self._style.get("slow_timing_bias_ms",
                                                self._groove.timing_bias_ms)))

    def _drum_events(self, role: str, preset: str, density: float,
                     bar_s: float, tonic_pc: int, plan, *,
                     root_pc: int | None = None,
                     key_known: bool = True) -> list[tuple]:
        variants = self._grids[role]
        # A slow count is her real tempo, and half-time on top of it would crawl,
        # so below this the style can state a plain backbeat instead.
        if 240.0 / bar_s < float(self._style.get("slow_below_bpm", 0.0)):
            variants = self._slow_grids.get(role, variants)
        idx = min(len(variants) - 1, int(density * len(variants)))
        steps = self._notation.steps_for_cycle(
            variants[idx], int(getattr(plan, "bar", 0)))
        if density < _DENSITY_BEATS_ONLY:
            steps = [s for s in steps
                     if abs((s.start * 4.0) - round(s.start * 4.0)) < 1e-6]
        out = []
        # An 808 is a pitched drum, so it plays a note of the key whatever role it
        # is in.  As the kick it is the low end and holds the tonic; as a sub under
        # a bass player it doubles the bass's root.  Only the "kick" role used to be
        # tuned, so midwest's sub 808 boomed at 45 Hz in every key.  With no key
        # known it is not tuned at all: a note of the fallback key is a guess at a
        # key nobody has heard, and a room mic hears it.
        if preset.startswith("808") and not key_known:
            kick_hz = None
        elif preset.startswith("808"):
            pc = root_pc if role == "sub" and root_pc is not None else tonic_pc
            kick_hz = _low_octave_hz(pc)
        else:
            kick_hz = _low_octave_hz(tonic_pc) if role == "kick" else None
        for s in steps:
            _, tok_vel = _strip_vel(s.token)
            vel = tok_vel if tok_vel is not None else _DEFAULT_VELOCITY
            out.append((role, preset, s.start * 4.0, s.dur * bar_s,
                        kick_hz, vel))
        return out

    def _next_root(self, chord: str | None, key: str | None, root_pc: int) -> int:
        """The root of the chord after this one, where the style's progression says.

        A chord the progression does not hold, as when following a tracker, is taken to
        stay where it is: the line then leans back into its own root.
        """
        if chord is None or key is None:
            return root_pc
        symbols = self._progressions.get(key)
        if symbols is None:
            try:
                symbols = progression_symbols(str(self._style["name"]), key)
            except (ValueError, KeyError, OSError):
                symbols = []
            self._progressions[key] = symbols
        if chord not in symbols:
            return root_pc
        return parse_chord(symbols[(symbols.index(chord) + 1) % len(symbols)]).root

    def _bass_events(self, preset: str, density: float, spb: float,
                     root_pc: int, *, chord: str | None = None,
                     key: str | None = None, bar: int = 0) -> list[tuple]:
        hz = _low_octave_hz(root_pc) * 2.0 ** int(self._style.get("bass_octave", 0))
        line_style = str(self._style.get("bass_style", ""))
        if line_style in figures.FIGURE_BASS_STYLES and density >= _BASS_WHOLE:
            # No chord belief means no third to lean on: root and fifth only.
            intervals = parse_chord(chord).intervals if chord is not None else (0, 7)
            if line_style == figures.WALKING:
                rng = np.random.default_rng((self._seed, 73, bar))
                line = figures.walking_line(root_pc, intervals,
                                            self._next_root(chord, key, root_pc), rng)
                return [("bass", preset, float(k), 0.95 * spb, midi_to_hz(m),
                         0.85 if k % 2 == 0 else 0.76)
                        for k, m in enumerate(line)]
            return [("bass", preset, ob, d * spb,
                     hz * 2.0 ** (figures.degree_semitones(intervals, deg) / 12.0), v)
                    for ob, d, v, deg in figures.BASS_FIGURES[line_style]]
        if density < _BASS_WHOLE:
            figure = [(0.0, 3.6, 0.85)]
        elif density < _BASS_HALVES:
            figure = [(0.0, 1.9, 0.85), (2.0, 1.6, 0.7)]
        else:
            style = str(self._style.get("bass_style", "roots_eighths"))
            slow = 60.0 / spb < float(self._style.get("slow_below_bpm", 0.0))
            if style == "trap_hits" and slow:        # locks to the slow kick
                figure = [(0.0, 1.6, 0.85), (1.75, 0.65, 0.72), (2.5, 1.3, 0.8)]
            elif style == "trap_hits":               # locks to the midwest kick
                figure = [(0.0, 1.3, 0.85), (1.5, 0.9, 0.72), (2.5, 1.3, 0.8)]
            elif style == "sub_hits":
                figure = [(0.0, 1.5, 0.85), (1.75, 0.7, 0.7),
                          (2.5, 1.2, 0.8)]
            else:  # roots_eighths / pump_eighths
                figure = [(k * 0.5, 0.45, 0.85 if k % 2 == 0 else 0.7)
                          for k in range(8)]
        return [("bass", preset, ob, d * spb, hz, v) for ob, d, v in figure]

    def _comp_events(self, role: str, preset: str, pad_style: str, density: float,
                     spb: float, voicing: list[int], chord: str | None,
                     ceiling: float | None, bar: int) -> list[tuple]:
        """A chord figure in the genre's own voicing, its rhythm changing by the bar."""
        notes = list(voicing)
        if pad_style == "house_stabs":
            notes = [m + 12 for m in voicing]
        elif chord is not None:
            notes = figures.rootless_voicing(parse_chord(chord),
                                             shell=pad_style == "shell_comp")
        notes = _apply_ceiling(notes, ceiling)
        comps = figures.COMP_FIGURES[pad_style]
        pick = int(np.random.default_rng((self._seed, 97, bar)).integers(len(comps)))
        figure = comps[pick][:1] if density < _BASS_WHOLE else comps[pick]
        return [(role, preset, ob, d * spb, midi_to_hz(m), v)
                for ob, d, v in figure for m in notes]

    def _pad_events(self, role: str, preset: str, density: float, spb: float,
                    voicing: list[int], *, chord: str | None = None,
                    ceiling: float | None = None, bar: int = 0) -> list[tuple]:
        pad_style = str(self._style.get(f"{role}_style")
                        or self._style.get("pad_style", "held"))
        out: list[tuple] = []
        if pad_style == "none":
            return out
        if pad_style in figures.COMP_FIGURES:
            return self._comp_events(role, preset, pad_style, density, spb, voicing,
                                     chord, ceiling, bar)
        if pad_style == "high":
            # The top of the chord an octave up: air over a lift.
            return [(role, preset, 0.0, 3.92 * spb, midi_to_hz(m + 12), 0.42)
                    for m in sorted(voicing)[-2:]]
        if pad_style == "twinkle" and density >= _PAD_ACTIVE and len(voicing) > 1:
            # Up-and-back eighths over the chord, soft, accenting the beat.
            notes = sorted(voicing)
            walk = notes + notes[-2:0:-1]
            for k in range(8):
                out.append((role, preset, k * 0.5, 0.9 * spb,
                            midi_to_hz(walk[k % len(walk)]),
                            0.5 if k % 2 == 0 else 0.34))
            return out
        if pad_style == "arp16" and density >= _PAD_ACTIVE and len(voicing) > 1:
            for k in range(8):                      # eighth-note arp
                m = voicing[k % len(voicing)]
                out.append((role, preset, k * 0.5, 0.45 * spb,
                            midi_to_hz(m), 0.55))
            return out
        if pad_style == "stabs" and density >= _PAD_ACTIVE:
            slots = (0, 7, 10)                       # 16th-grid stabs
            for sl in slots:
                for m in voicing:
                    out.append((role, preset, sl / 4.0, 0.5 * spb,
                                midi_to_hz(m), 0.6))
            return out
        for m in voicing:                            # held: whole-bar chord
            out.append((role, preset, 0.0, 3.92 * spb, midi_to_hz(m), 0.5))
        return out

    def _motif_events(self, preset: str, density: float, spb: float,
                      voicing: list[int], ceiling: float | None, bar: int,
                      tonic_pc: int, key: str | None) -> list[tuple]:
        """A short tune that repeats, with room around it for the player.

        Eight-bar sections share one motif, so it comes back and can be learned.
        Each four bars states it, varies it, then rests a bar; every other four
        bars sits out entirely.  Beat-one notes land on chord tones and the rest
        walk a pentatonic scale, which stays out of a guitar's way harmonically.
        """
        phrase_bar = bar % 4
        lifted = density >= _MELODY_LIFT
        if phrase_bar == 3 or (bar // 4 % 2 == 1 and not lifted):
            return []
        cfg = self._style["melody"]
        rng = np.random.default_rng((self._seed, 91, bar // 8))
        motifs = cfg["motifs"]
        motif = motifs[int(rng.integers(len(motifs)))]
        contour = rng.integers(-2, 3, size=16)
        low, high = cfg.get("register", [72, 86])
        minor = "min" in str(key or "").lower()
        steps = (0, 3, 5, 7, 10) if minor else (0, 2, 4, 7, 9)
        pcs = {(tonic_pc + s) % 12 for s in steps}
        # Walk the notes she has been playing, where they are in key.
        major = (0, 2, 4, 5, 7, 9, 11)
        natural = (0, 2, 3, 5, 7, 8, 10)
        diatonic = {(tonic_pc + s) % 12 for s in (natural if minor else major)}
        hers = {n for n in self.heard_notes if n in diatonic}
        if len(hers) >= 3:
            pcs = hers
        scale = sorted(m for m in range(low - 12, high + 13) if m % 12 in pcs)
        chord_pcs = {m % 12 for m in voicing}
        tones = [m for m in scale if m % 12 in chord_pcs] or scale
        centre = (low + high) // 2
        current = min(tones, key=lambda m: abs(m - centre))
        # The walk may step past the register; what sounds is the nearest note of
        # the same pool inside it.  Clamping the number itself lands on whatever the
        # register edge is (72 is a C), which in most keys is not in the key at all.
        in_scale = [m for m in scale if low <= m <= high] or scale
        in_tones = [m for m in tones if low <= m <= high] or in_scale
        out = []
        for i, beat in enumerate(motif[phrase_bar % len(motif)]):
            idx = scale.index(min(scale, key=lambda m: abs(m - current)))
            idx = int(np.clip(idx + contour[(i + phrase_bar * 4) % 16], 0,
                              len(scale) - 1))
            current = scale[idx]
            if float(beat) == 0.0:
                current = min(tones, key=lambda m: abs(m - current))
            pool = in_tones if float(beat) == 0.0 else in_scale
            pitch = current if low <= current <= high else min(
                pool, key=lambda m: abs(m - current))
            pitch = _apply_ceiling([pitch], ceiling)[-1]
            vel = 0.62 if float(beat) % 2.0 == 0.0 else 0.48
            hz = midi_to_hz(pitch)
            out.append(("lead", preset, float(beat), 1.8 * spb, hz, vel))
            if not cfg.get("echo", True):
                continue
            # A dotted-eighth echo, built from notes: the bar loop runs no delay.
            out.append(("lead", preset, float(beat) + 0.75, 1.2 * spb, hz, vel * 0.42))
            if lifted:
                out.append(("lead", preset, float(beat) + 1.5, 1.0 * spb, hz, vel * 0.2))
        return out

    def _lead_events(self, preset: str, density: float, spb: float,
                     voicing: list[int], ceiling: float | None) -> list[tuple]:
        top = max(voicing) + 12
        top = _apply_ceiling([top], ceiling)[-1]
        onsets = [0.0] if density < _PAD_ACTIVE else [0.0, 2.5]
        return [("lead", preset, ob, 1.4 * spb, midi_to_hz(top), 0.6)
                for ob in onsets]
