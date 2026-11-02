# SPEC — plume v0

v0 = **prompted mode end-to-end**: prompt → session (JSON score) → deterministic
WAV render → analysis → iterate, via an LLM-native CLI. Live mode (v1) builds on
the same score model; its interaction spec lives in RESEARCH.md and is *not*
implemented in v0, but v0 interfaces must not preclude it.

**Contract status: FROZEN for the v0 build.** Builders implement exactly these
interfaces. If an interface is genuinely broken, fix it and note the deviation
in your report — do not silently drift.

## Ground rules

- Python ≥3.11. Deps: `numpy`, `scipy`, `soundfile`. Optional: `sounddevice`
  (only imported inside `play`). Tests: `pytest`. Nothing else in v0.
- Sample rate **48000**, float32 everywhere internally, stereo out. WAV written
  as PCM_24 via soundfile.
- **Determinism**: same session JSON + same seed ⇒ bit-identical WAV. One
  `np.random.default_rng(seed)` created per render, passed explicitly — no
  global RNG, no time-based anything.
- **No per-sample Python loops** in DSP. Vectorized numpy; recursive filters
  via `scipy.signal.lfilter` or vectorized TPT SVF with per-block coefficients.
- Package layout: `src/plume/{__init__,theory,notation,score,styles_lib,groove,compose,voices,fx,engine,analyze,cli}.py`,
  style data in `src/plume/styles/*.json`, tests in `tests/`.
  `pyproject.toml` exposes console script `plume = plume.cli:main`.
- Every CLI command supports `--json` (machine output to stdout, nothing else
  on stdout; human-pretty output is the default). Errors: nonzero exit +
  `{"error": ...}` on stdout in `--json` mode.
- Units: gain in dB (field names end `_db`), pan −1..+1, times in seconds end
  `_s`, milliseconds end `_ms`, frequencies end `_hz`, musical positions in
  beats (float, quarter notes). `velocity` 0..1.

## 1. theory.py

```python
NOTE_NAMES: list[str]  # ["c","cs","d","ds","e","f","fs","g","gs","a","as","b"] (flats accepted as input: "eb"=="ds")
def note_to_midi(name: str) -> int          # "c4"->60, "eb2"->39; raises ValueError
def midi_to_hz(midi: float) -> float        # 440 * 2**((m-69)/12); float in for glides
def parse_key(s: str) -> Key                # "C min", "f# minor", "Eb maj", "a dorian"
    # v0.1 amendment: also accepts glued forms "Fm", "f#m", "Ebmin", "Gmaj";
    # bare tonic ("F") = major; 'm'/'min'/'minor' suffix = minor. Errors state
    # the accepted formats and the mode list.
@dataclass(frozen=True) class Key: tonic: int  # pitch class 0-11
                                   mode: str   # "major","minor","harmonic_minor","dorian","phrygian"
    def scale(self) -> list[int]             # pitch classes, ascending from tonic
def parse_chord(sym: str) -> Chord           # "Cm9","Abmaj7","G7","Fsus2","Eb6","Dm7b5","C" etc.
@dataclass(frozen=True) class Chord: root: int; quality: str; intervals: tuple[int,...]; sym: str
def roman_to_chord(numeral: str, key: Key) -> Chord
    # "i","VI","III","VII","iv","v","V","ii7","V7","Imaj7","vi7","IVmaj7","bVII" — case = quality,
    # supports 7/9/maj7 suffixes; borrowed bVII/iv handled; in minor, "V" = major dominant (harmonic minor).
def voice_chord(chord: Chord, *, low_midi: int = 48, spread: bool = False,
                prev: list[int] | None = None) -> list[int]
    # returns midi notes. spread=True: root in octave 2-3, upper structure above with a gap
    # (lofi voicing rule). prev given -> minimal-motion voice leading (nearest inversion).
```

## 2. notation.py — mini-notation (Strudel subset)

Grammar (one pattern string = **one bar** = one cycle):

```
pattern  := layer ("," layer)*          # layers play simultaneously
layer    := seq
seq      := elem+                       # whitespace separated; bar divided equally among elems
elem     := atom | group | elem "*" INT | elem "@" INT | alt
group    := "[" seq "]"                 # subdivides its slot
alt      := "<" atom+ ">"               # per-cycle alternation (bar 1 -> first, bar 2 -> second, ...)
atom     := TOKEN | "~"                 # "~" = rest
TOKEN    := /[A-Za-z#][A-Za-z0-9#b.]*/  # e.g. bd, hh, c1, eb2, Cm9, Abmaj7
```

`elem*n` repeats elem n times within its slot; `elem@n` stretches elem to n
slots. No nesting limits. Errors raise `NotationError` with position info.

```python
@dataclass(frozen=True) class Step:
    start: float      # 0..1 within the bar
    dur: float        # fraction of bar
    token: str
    cycle_alts: tuple[str, ...] | None = None   # set when token came from <...>; token = alts[0]
def parse(pattern: str) -> list[Step]                    # steps sorted by start; rests omitted
def steps_for_cycle(steps: list[Step], cycle: int) -> list[Step]  # resolves <...> for bar number `cycle`
```

Token semantics are decided by the *track kind* (engine): drum tracks interpret
tokens as the voice's hit (any token = trigger; token `x`/voice-name both fine);
tonal tracks interpret note names (`c1`, `eb3`) or chord symbols (`Cm9`) —
chords render as `voice_chord(...)` unless the voice is mono (bass: takes root).
Suffix `:v0.6` on any token sets per-step velocity: `bd:v0.5`.

## 3. score.py — the session document

```python
@dataclass class Track:
    name: str
    voice: str                        # preset name, see voices.PRESETS
    kind: str                         # "drum" | "tonal"
    gain_db: float = 0.0
    pan: float = 0.0
    params: dict[str, float | str] = field(default_factory=dict)   # voice param overrides
    fx: dict[str, float] = field(default_factory=dict)
        # optional keys: lpf_hz, hpf_hz, drive, chorus (0-1), delay (0-1 wet, 3/16 dotted default),
        # reverb (0-1 send), duck (0-1 depth, sidechained from the kick track)
    patterns: dict[str, str] = field(default_factory=dict)         # pattern name -> mini-notation

@dataclass class SectionSpec:
    name: str
    bars: int
    energy: float                     # 0..1, gates layers & brightness
    plays: dict[str, str]             # track name -> pattern name ("" / absent = tacet)
    chords: list[str] = field(default_factory=list)  # chord symbols, cycled across bars (one per bar)
    fill: bool = True                 # fill in last bar of the section

@dataclass class Movement:
    name: str
    sections: list[SectionSpec]

@dataclass class Session:
    name: str
    bpm: float
    key: str                          # e.g. "F min" — parse with theory.parse_key
    swing: float = 0.5                # 0.5 straight .. 0.66 triplet (MPC-style, even 16ths delayed)
    seed: int = 0
    style: str = ""                   # primary style name (informs master chain, e.g. lofi treatments)
    timing_bias_ms: dict[str, float] = field(default_factory=dict)  # role -> ms (e.g. {"snare": 20, "kick": -12})
    humanize_ms: float = 4.0          # timing jitter SD, ≤10
    humanize_vel: float = 0.08        # velocity jitter SD (fraction)
    tracks: list[Track] = ...
    movements: list[Movement] = ...
    master: dict[str, float] = ...    # gain_db (v0.1, default 0), reverb (0-1), drive (0-1),
                                      # lofi (0-1), limit_db (-1.0 ceiling)

def save(session: Session, path: str | Path) -> None      # json, sorted keys, indent=2, version field "plume_version": 0
def load(path: str | Path) -> Session                     # tolerant of missing optional fields
def total_bars(session: Session) -> int
```

JSON layout mirrors the dataclasses 1:1 (`dataclasses.asdict`). This file IS
the API's core noun — everything the CLI does is a read-modify-write of it.

## 4. groove.py

```python
@dataclass(frozen=True) class GrooveParams:
    swing: float; timing_bias_ms: dict[str, float]; humanize_ms: float; humanize_vel: float
def apply_groove(onset_beats: float, *, role: str, bpm: float, g: GrooveParams,
                 rng: np.random.Generator, velocity: float) -> tuple[float, float]
    # returns (onset_seconds_offset_adjusted, velocity_adjusted)
    # swing: if onset falls on an even 16th (positions .25/.75 within a beat), delay by
    #   (swing-0.5)*2 * (sixteenth_duration)   [50%=0, 58%≈32% of a 16th, 66%=triplet]
    # then add timing_bias_ms.get(role, 0), then N(0, humanize_ms) jitter, then velocity jitter
    # (clip velocity to 0.05..1). Downbeat (integer beat 0 of bar) gets +0.08 velocity accent pre-clip.
```

Roles = track voice family: "kick","snare","hat","clap","bass","pad","pluck","epiano","lead".

## 5. styles — data + loader (styles_lib.py + styles/*.json)

Three files: `trap.json`, `lofi.json`, `synthwave.json`. Schema:

```jsonc
{
  "name": "lofi",
  "bpm_range": [70, 90],
  "default_mode": "minor",
  "swing": 0.57,
  "timing_bias_ms": {"kick": -12, "snare": 18},
  "progressions": {                       // v0.1 amendment: mode-keyed pools —
    "major": [                            // compose picks the pool matching the
      {"numerals": ["ii7","V7","Imaj7","Imaj7"], "weight": 3}   // session key's
    ],                                    // mode (minor-family modes -> "minor")
    "minor": [
      {"numerals": ["i7","iv7","VII7","IIImaj7"], "weight": 3}
    ]
  },
  "drum_patterns": {                      // mini-notation per role, several variants each
    "kick":  ["bd ~ ~ ~ ~ ~ [~ bd] ~ ~ ~ bd ~ ~ ~ ~ ~"],
    "snare": ["~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~"],
    "hat":   ["hh ~ hh hh ~ hh hh ~ hh ~ hh hh ~ hh ~ hh"]
  },
  "fills": {"hat": ["[hh*4] [hh*6]", "hh*8"]},   // last-bar substitutions
  "bass_style": "roots_eighths" | "sub_hits" | "pump_eighths",
  "pad_style":  "held" | "stabs" | "arp16",
  "melody": {"density": 0.4, "octave": 5, "rhythm_grid": "8"},
  "voices": {"kick": "808kick", "bass": "808sub", "pad": "pad", ...},   // role -> preset
  "fx_defaults": {"pad": {"lpf_hz": 1800, "reverb": 0.4, "chorus": 0.5}, ...},
  "gain_db": {"hat": -5.0},               // v0.1 optional: per-role gain override
                                          // of compose's built-in defaults

  "master": {"reverb": 0.15, "lofi": 0.6, "drive": 0.2},
  "arrangement_templates": {
    "3": [ {"name": "movement-1", "energy_curve": [0.25, 0.4], "bars": 32}, ... ]
  }
}
```

```python
def load_style(name: str) -> dict         # from package data; raises for unknown
def list_styles() -> list[str]
```

Content requirements: trap grids use half-time (snare beat 3), hat-roll fill
lexicon with `*3` triplet bursts and `[hh*8]` 1/32 runs, 808-slide bass
patterns; lofi per the drunk-feel numbers; synthwave four-on-floor + gated
snare + pump-eighth bass + 1/16 arp. Progressions from RESEARCH.md's verified
pools (trap: i-VI-III-VII, i-VI-V, i-iv-v, i-VII-VI…; lofi: ii-V-I family;
synthwave: i-VI-VII, i-VII-VI-V Andalusian…).

v0.1 amendment (#5): every style carries BOTH a major and a minor pool, and every
minor-pool progression must be diatonic to the natural minor key (the raised 7th
allowed only in the harmonic-minor dominant) with a minor-quality tonic chord.
Compose selects the pool by the session key's mode — a minor-key session can never
receive major-key harmony again.

## 6. compose.py — prompt → Session

```python
@dataclass class PromptSpec:
    styles: list[str]            # ordered, first = primary
    bpm: float | None; key: str | None; movements: int; mood: list[str]; name: str
def parse_prompt(text: str) -> PromptSpec
    # rule-based: style keywords (trap/lofi/lo-fi/synthwave/ambient…), "NNN bpm", key names
    # ("in F minor"), "N movements"/"N major movements", mood adjectives (chill, dark, eerie,
    # hopeful, aggressive). Unknowns -> None/defaults. Never errors on free text.
def compose(spec: PromptSpec, *, seed: int | None = None) -> Session
    # seed default: stable hash of prompt text. Steps:
    # 1. style blend: primary style wins structure; secondary styles pull bpm midpoint, add
    #    their signature layers (e.g. +synthwave arp over lofi base), merge fx/master hints.
    # 2. pick bpm (spec or blended range midpoint), key (spec or rng: minor tonics weighted to
    #    "sad keys" F/A/C/D), progression (weighted rng), voicings via voice_chord(spread per style).
    # 3. build tracks from style voices + fx_defaults; patterns from drum grids (rng variant),
    #    bass_style/pad_style expanded to mini-notation against the progression, melody via motif
    #    engine (below); name patterns "main","alt","sparse","fill".
    # 4. arrangement: movements from template for spec.movements (default 3): each movement gets
    #    sections (8/16-bar multiples) with energy from the template curve; energy gates layers:
    #    <0.3 = kick+bass+pad sparse; 0.3-0.6 += hats+keys; >0.6 = full + melody; last section of
    #    last movement = outro (energy fade, thin to pad+crackle).
    # 5. mood adjustments: "dark/eerie" -> harmonic minor V, lpf down, more reverb; "chill" ->
    #    swing +0.02, density down; "aggressive" -> energy floor 0.5, drive up.
def make_motif(key: Key, chords: list[Chord], style_melody: dict, rng) -> list[str]
    # 1-2 bar motif on scale tones, chord tones on strong beats, range ≤ 1.5 octaves, leaps >6st
    # resolve stepwise opposite; then per-chord diatonic transposition across the progression;
    # returns one mini-notation string per bar of the progression.
```

## 7. voices.py — the instruments

Common interface (stateless pure functions; the engine sums the returns):

```python
def render_note(preset: str, *, sr: int, dur_s: float, freq_hz: float | None,
                velocity: float, rng: np.random.Generator,
                params: dict, glide_from_hz: float | None = None) -> np.ndarray  # mono float32
PRESETS: dict[str, dict]   # preset name -> default params (documented per voice below)
def preset_info() -> dict  # JSON-able: every preset, its params, ranges, description (CLI discoverability)
```

Recipes (from RESEARCH.md; params overridable via Track.params):

- **808kick**: sine, pitch env `f(t) = f0 + (sweep_mult*f0 - f0) * exp(-t/tau)`,
  `f0` = tuned to session key's tonic in 35–55Hz octave (engine passes freq_hz),
  defaults `sweep_mult=3.5`, `tau_ms=30`; amp: 1ms attack, exp decay
  `decay_ms=350`; click: 3ms noise burst hp@2kHz mixed at 0.3·velocity;
  `drive=2.5` tanh.
- **808sub**: sine, no sweep, attack 8ms, decay 900ms (gated by step dur),
  drive 1.5; supports `glide_from_hz` (exp pitch ramp over `glide_ms=60`).
  Engine triggers glide when consecutive bass steps are adjacent (legato).
- **hat / openhat**: 6 squares at [205.3, 304.4, 369.6, 522.7, 540, 800]Hz
  (×freq scale param), bp@10kHz→hp@7kHz, exp decay 40ms closed / 400ms open,
  slight rng phase; velocity → level + decay scale.
- **clap**: noise bp 1–2kHz Q1.5; envelope = 3 bursts spaced 11ms + 200ms tail.
- **snare**: sines 185+330Hz decay 100ms + noise bp 1–8kHz decay 180ms, 50/50, soft-clip.
- **pad**: 5-saw supersaw (polyBLEP), detune ±25 cents, SVF lp (cutoff from
  fx/energy), attack 120ms release 300ms; stereo via two detuned renders.
- **pluck**: saw → SVF lp, filter env 4kHz→400Hz over 120ms, amp decay 250ms.
- **epiano**: 2-op FM (mod 1:1, index 2→0.3 over 200ms) + 14:1 tine ping fast-decay; velocity→index.
- **lead**: 3-saw supersaw + portamento (glide support), square sub −1 oct at 0.3.
- **crackle**: sparse impulse noise (rate 90/s, rng) lp@7kHz — lofi bed, loops per bar.

All voices: `np.float32`, no NaN/inf, peak ≤ 1.0 pre-mix, no per-sample loops
(polyBLEP via vectorized formula; FM via phase-array sine composition).

## 8. fx.py

```python
def svf_lowpass(x, sr, cutoff_hz, q=0.707) -> np.ndarray      # TPT/Zavalishin, block-vectorized via lfilter equivalence
def svf_highpass(x, sr, cutoff_hz, q=0.707) -> np.ndarray
def drive(x, amount) -> np.ndarray                            # tanh(x*a)/tanh(a), a=1+3*amount
def chorus(x, sr, depth=0.5) -> np.ndarray                    # 2 mod delay lines 1-10ms, LFOs 0.5/0.8Hz -> stereo (N,2)
def delay(x, sr, bpm, wet, feedback=0.45, note="3/16") -> np.ndarray  # feedback loop with lp@4kHz inside
def freeverb(x, sr, wet) -> np.ndarray                        # 8 combs + 4 allpass per channel, stereo spread; scipy.lfilter combs
def ducker(x, sr, trigger_times_s, depth, dip_db=-9, attack_ms=8, release_ms=220) -> np.ndarray
def lofi(x, sr, amount) -> np.ndarray    # pitch wobble (0.6Hz, ±8 cents via resample-free delay mod), lp 5-10kHz, bit 12 @ amount>0.5
def limiter(x, ceiling_db=-1.0) -> np.ndarray                 # soft-clip into ceiling; never hard-clips
    # v0.1 amendment: knee limiter — bit-transparent below ceiling-3dB, tanh
    # above (the old full-range tanh squashed every mix toward the ceiling,
    # hiding gain changes)
```

Mono in → mono out unless noted; engine handles stereo placement (constant-power pan).

## 9. engine.py — compile + offline render

```python
@dataclass(frozen=True) class NoteEvent:
    t_s: float; track: str; role: str; freq_hz: float | None; dur_s: float
    velocity: float; glide_from_hz: float | None; params: dict
def compile_events(session: Session) -> list[NoteEvent]
    # walk movements/sections/bars; resolve pattern per track (SectionSpec.plays; fill=True ->
    # style fill pattern in last bar when available as pattern "fill"); steps_for_cycle for <...>;
    # tonal tokens -> theory (chords voiced, prev-voicing continuity per track; bass takes roots);
    # apply_groove per event; energy scales velocity (0.6+0.4*energy) and gates fx brightness.
def render(session: Session, *, out_path: str, movement: int | None = None,
           bars: tuple[int, int] | None = None, stems_dir: str | None = None) -> dict
    # returns manifest: {out, duration_s, bars, events, peak_db, rms_db,
    #                    limiter_gain_reduction_db, per_track_rms_db}   (v0.1)
    # pipeline: events -> per-track mono buffers (sum render_note at sample offsets) -> track fx
    # (svf, drive, chorus, delay) -> pan to stereo -> duck (tracks with fx.duck>0, sidechain =
    # kick-role events) -> sum -> master.gain_db (v0.1: real gain stage, default 0 dB;
    # NO implicit normalization anywhere) -> master reverb send -> master drive -> lofi
    # (if style/master) -> limiter at master.limit_db -> PCM_24 WAV.
    # limiter_gain_reduction_db = pre-limiter peak dB minus post-limiter peak dB (0 =
    # limiter untouched) so an LLM can see when the ceiling is doing the work.
    # Also writes stems when stems_dir given (post-fx, pre-master, scaled to keep
    # -3 dBFS headroom — v0.1 #12).
    # v0.1 amendment (#14 loudness arc): at compile, each section's energy maps to a
    # pre-master gain offset lerp(-3 dB..0 dB, energy 0..1) carried on NoteEvent.gain,
    # so movements breathe in level as well as density. Limiter ceiling unchanged.
def play(session: Session, **render_kwargs) -> None   # render to temp then sounddevice.play; friendly error if no device
```

Tail: pad section end with 1.5s release/reverb tail past the last bar.

## 10. analyze.py

```python
def analyze(wav_path: str, bpm_hint: float | None = None) -> dict
```
Returns (all computed with numpy/scipy only): `duration_s, sample_rate, channels,
peak_db, rms_db, lufs_approx (K-weighted RMS), clipped_sample_count, silence_ratio
(frames < -60dB), spectral_centroid_hz_mean, band_energy_db {sub<60, low 60-250,
mid 250-2k, high 2k-8k, air >8k}, onset_count, onset_density_per_s,
tempo_bpm_estimate (autocorr of spectral-flux envelope, 60-200 range),
tempo_confidence (0-1), tempo_candidates, stereo_width (1 - |corr(L,R)|),
bar_rms_db: list` (per-bar loudness when the bpm hint is present).

v0.1 amendments (#8, #13):
- `tempo_candidates`: ranked list of `{"bpm", "confidence"}` — the top comb-scored
  autocorrelation peaks plus the half/double octave relatives of the leader, so
  metrical-level ambiguity (half-time trap, double-time lofi) is visible.
- `bpm_hint` now ALSO biases the tempo estimate: a log-Gaussian prior centered on
  the hint (sigma 0.25 octaves) replaces the default 115-bpm prior, so the hint
  picks the metrical level for both `tempo_bpm_estimate` and `bar_rms_db`.
- CLI hint precedence: explicit `--bpm-hint` > the resolved session's bpm > none.

## 11. cli.py

`plume <command> [args] [--session PATH] [--json]`. Session resolution: explicit
`--session`, else `PLUME_SESSION` env, else exactly one `*.plume.json` in cwd,
else error listing candidates.

```
plume make "chill trap over lofi ambient synthwave at 120 bpm, 3 movements" [--name x] [--seed N]
plume new NAME --style lofi [--bpm 80] [--key "F min"] [--movements 3] [--seed N]
plume show [--events] [--movement N]          # session summary / compiled event stats
                                                # v0.1: --json includes the FULL session
                                                # document under "document"; unused tracks
                                                # (never played by any section) are flagged
plume set KEY VALUE                            # dotted paths: bpm, swing, master.gain_db, tracks.pad.fx.lpf_hz
                                                # v0.1: paths are VALIDATED against the session
                                                # schema — unknown keys error with the valid set
                                                # + did-you-mean; type-destroying writes refused
plume get [PATH] [--list]                      # v0.1: read a value (JSON out); no PATH = whole
                                                # document; --list enumerates settable paths
plume unset PATH                               # v0.1: remove an optional leaf key (master.*,
                                                # timing_bias_ms.*, tracks.NAME.{fx,params,patterns}.*);
                                                # structural keys refused
plume notation                                 # v0.1: mini-notation reference; --json returns
                                                # {grammar, tokens, examples}
plume track add NAME --voice PRESET [--kind drum|tonal] [--gain-db X] [--pan X]
plume track rm NAME / plume track list
plume pattern set TRACK NAME "mini-notation"   # validates by parsing; prints step count
plume pattern get TRACK [NAME]
plume arrange show / plume arrange set --spec-json JSON|@FILE.json   # movements/sections
                                                # v0.1: @FILE.json reads the spec from disk; the
                                                # exact schema is documented in --help; arrange
                                                # show --json round-trips into arrange set
plume render [--out take.wav] [--movement N] [--bars A B] [--stems DIR]
plume analyze PATH [--bpm-hint N]              # v0.1: hint defaults to the session bpm
plume play [--movement N]
plume voices / plume styles                   # discoverability (JSON-able catalogs)
plume doctor                                   # env check: deps, audio device, versions
```

`make` prints: session path + a compact musical summary (key, bpm, progression,
movements table). Every mutating command prints the new state hash (sha256 of
canonical JSON, first 12 chars) so an LLM can verify effects.

## 12. Tests (minimum bar)

- notation: grammar cases incl. nesting, `*`, `@`, `<>`, `,`, velocity suffix, errors.
- theory: note/chord/roman parsing, voice leading continuity.
- groove: swing math at 50/58/66%, bias application, determinism with seed.
- compose: prompt parsing table; composed session validates (tracks reference
  real presets; sections reference real patterns; bars are 8-multiples).
- voices: every preset renders 0.5s @440Hz without NaN, peak ≤1, non-silent.
- engine: 4-bar render of a tiny fixture session → WAV exists, duration correct
  (±50ms), non-silent, no clipping; **same seed twice → identical bytes**;
  different seed → different bytes.
- analyze: on the fixture render: tempo estimate within ±3 bpm of session bpm.
- cli: `make` → `render` → `analyze` round-trip via subprocess in tmpdir.

## 13. v1 pointers (do not build, do not preclude)

Live mode adds: `listen` daemon (JSON-RPC over localhost HTTP + NDJSON event
log), `BeatTracker`/`PitchTracker` interfaces (BeatNet / PESTO / chroma-HMM
implementations), predicted-beat clock with phase correction (α=tightness),
the negotiation state machine (RESEARCH.md behaviors, all thresholds as
session params), sounddevice duplex engine with render-ahead ring buffer,
schedule-aware self-masking. The v0 `Session`/`NoteEvent`/voices/fx are reused
verbatim — that's why the engine renders from compiled events, not patterns
directly: live mode compiles one bar at a time against a moving clock.
