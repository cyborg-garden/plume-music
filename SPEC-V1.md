# SPEC — plume v1 "the sideman" (live mode)

v1 adds listening. One pipeline, two transports: **sim** (offline: stream a WAV
through the identical code path, deterministic, faster-than-real-time — the
test/verification surface) and **jam** (real-time duplex mic/speaker). v0 code
(voices, fx, styles, groove, theory, notation, engine) is reused, not modified —
the only permitted v0 change is *additive* (new CLI subcommands registered in
cli.py; a shared helper moved only if imports force it, noted as deviation).

**Contract status: FROZEN for the v1 build.** Same rules as SPEC.md: exact
signatures; deviations reported, never silent. Ground rules from SPEC.md apply
(48kHz float32, determinism from passed rng + input, no per-sample loops,
numpy/scipy/soundfile only; sounddevice optional at import time for jam).

Research base: RESEARCH.md "interaction spec" + the original research sweep
(jazz-negotiation recommendations = the 20 behaviors; realtime-beat-tracking;
pitch-key-following; macos-audio-io). Package: `src/plume/listen/{features,beat,harmony}.py`,
`src/plume/mind.py`, `src/plume/duet.py`, `src/plume/sim.py`, `src/plume/realtime.py`.

## 1. listen/features.py — causal frame features

> **Amended in v1.2 — see §11.1, §11.2, §11.3.** Chroma now analyses its own
> longer window (`CHROMA_WIN = 4096`) and requires harmonic corroboration; the
> "rectangular pitch-class mapping" below is superseded.
>
> **Amended in v1.4 — see §13.40, §13.45, §13.46.** `Frame` carries a new
> defaulted `chroma_low` field: a register-tilted bass-band chroma whose sum is
> the low band's share of the frame's pitched evidence. Two limits are recorded
> there rather than fixed: close triads in the C2 octave still merge, and
> `ONSET_DELTA` is still absolute.

```python
HOP = 512          # 10.67ms at 48k
WIN = 2048
@dataclass(frozen=True) class Frame:
    t_s: float             # time of frame END relative to stream start
    rms_db: float          # frame RMS in dBFS (floor -90)
    flux: float            # half-wave-rectified log-magnitude spectral flux, >= 0
    onset: bool            # causal peak-pick on flux (adaptive threshold: median+delta over trailing ~0.35s, min separation 60ms)
    onset_strength: float  # flux value when onset else 0
    chroma: np.ndarray     # shape (12,) float32, L1-normalized (zeros if silent)
    centroid_hz: float     # spectral centroid (0 if silent)
class FeatureExtractor:
    def __init__(self, sr: int = 48000): ...
    def process(self, block: np.ndarray) -> list[Frame]
        # accepts arbitrary-length mono float32 blocks; internal buffer; emits completed frames.
        # Chroma: rectangular pitch-class mapping of rFFT bins 60-5000Hz (A440), vectorized.
    def flush(self) -> list[Frame]
```

Determinism: pure function of the input stream. No rng.

## 2. listen/beat.py — causal tempo/phase/downbeat with prediction

> **Amended in v1.2 — see §11.4–§11.10.** Tempo is chosen by a full lag sweep
> with a metric prior (not octave relatives of an ACF peak), the ACF window is
> 12 s, `tap()` is latched, phase can be re-acquired, and `bpm_hint` is new.
>
> **Amended in v1.3 — see §12.26.** Period hysteresis is no longer one rule:
> once the tracker has been locked on an octave for `OCTAVE_SETTLE_S`, changing
> octave costs `OCTAVE_SWITCH_EVALS` consecutive evaluations instead of 2, any
> locked octave change emits a `tempo_octave_change` event and increments
> `BeatState.tempo_octave_changes`, and a settled one cuts confidence.
> Salience is measured on the grid in force, after the adoption decision.

```python
@dataclass(frozen=True) class BeatState:
    t_s: float
    bpm: float                 # current period hypothesis (0.0 until first estimate)
    confidence: float          # 0..1
    locked: bool               # confidence above lock threshold for >= 2 consecutive bars
    bar: int                   # 0-based bar counter since lock (-1 before lock)
    beat_in_bar: int           # 0..3 (meter fixed 4/4 in v1)
    phase_beats: float         # continuous position in bar [0, 4)
    next_beat_t_s: float       # PREDICTED next beat time (math.inf before estimate)
    next_downbeat_t_s: float   # PREDICTED next bar start (math.inf before lock)
    beat_interval_s: float     # 60/bpm (inf before estimate)

class BeatTracker:
    def __init__(self, *, tightness: float = 0.35, bpm_range: tuple[float, float] = (60.0, 200.0),
                 lock_conf: float = 0.55, drift_pct: float = 3.0): ...
    def process(self, frames: list[Frame]) -> BeatState   # feed every frame batch; returns current state
    def tap(self, t_s: float) -> None
        # count-in: >=4 taps at roughly even spacing hard-set bpm (median IOI) + downbeat phase
        # (last tap = beat 0 of bar 0) + locked=True, confidence=1.0; tracker then only corrects drift.
```

Algorithm contract (the ADAM findings, encoded):
- **Tempo (period)**: autocorrelation of the flux envelope over a trailing ~6s
  window, evaluated at lags for bpm_range (include half/double checks);
  re-estimate every ~0.5s. **Period hysteresis**: switch bpm only if the new
  hypothesis differs >drift_pct% AND persists ≥2 evaluations (or is an exact
  half/double resolved toward the range 70–160 preferring onset-density fit).
- **Phase**: maintain predicted next-beat time; on each detected onset within
  ±35% of a predicted beat, correct partially: `next_beat += tightness * error`
  (α-correction; never snap). Between onsets, free-run on the current period.
- **Downbeat**: score the 4 beat-phase hypotheses over a rolling 2-bar window
  by (a) onset_strength accent profile and (b) chroma-change alignment
  (harmonic change lands on downbeats); commit when the best hypothesis leads
  by a margin for a full bar; changing a committed downbeat requires 2 bars of
  contrary evidence.
- **Confidence**: product of autocorr peak salience and fraction of recent
  predicted beats matched by onsets (window 8 beats); smoothed (EMA ~0.8). No
  events / flat flux ⇒ decays toward 0 (rubato ⇒ unlocks: `locked` False after
  confidence < 0.3 for 1+ bar; bar counter freezes, phase free-runs).
- Deterministic: pure function of frames + taps.

## 3. listen/harmony.py — key/chord belief

> **Amended in v1.2 — see §11.11, §11.12.** Key belief requires 2 s of voiced
> evidence (not 53 ms), and a relative-key challenger is held off by a much
> larger margin because the two share a pitch collection.
>
> **Amended in v1.4 — see §13.41–§13.44.** The key histogram is weighted by
> register (`chroma + _KEY_BASS_WEIGHT * chroma_low`), a key may only be
> acquired while tonal content is sustained, `key_conf` is discounted until
> enough evidence stands behind it, and the relative-key margin generalises to
> any key one accidental away. Chord matching is unchanged by all of it.

```python
@dataclass(frozen=True) class HarmonyState:
    t_s: float
    key: str | None            # e.g. "F minor" (parse_key-compatible); None below confidence floor
    key_conf: float            # 0..1
    chord: str | None          # triad symbol spelled per key ("Bbm", "Ab") or None
    chord_conf: float
    register_low_midi: float | None   # rolling p10/p90 of active pitch content, None if silent
    register_high_midi: float | None
class HarmonyTracker:
    def __init__(self, *, key_halflife_s: float = 15.0, chord_selftrans: float = 0.9): ...
    def process(self, frames: list[Frame]) -> HarmonyState
```

Key: pitch-class histogram of chroma with exponential decay (half-life
key_halflife_s), correlated against all 24 Krumhansl-Kessler profiles;
switch keys only when challenger beats incumbent by margin 0.05 sustained 3s.
Chord: 24 maj/min triad templates + N (silence); per-frame cosine scores
smoothed by an online HMM (self-transition chord_selftrans, fixed-lag 3
frames); emit only when posterior > 0.5 twice consecutively. Register: from
chroma-weighted spectral distribution (approximate; documented). Deterministic.

## 4. mind.py — the negotiation state machine

> **Amended in v1.2 — see §11.13–§11.16.** Bar silence is onset-guarded and
> relative to the human's own level; the mind trusts `BarFeatures.silent`;
> `entry_conf` is 0.45; `min_listen_bars` counts bars actually heard, via the
> new `Mind.credit_listened_bars`.

```python
STATES = ("waiting", "listening", "entering", "comping", "thinned", "fading", "tacet")
@dataclass class MindParams:      # EVERY threshold; defaults = research values; all settable via CLI --set k=v
    min_listen_bars: int = 4
    entry_conf: float = 0.55          # beat confidence required to arm entry
    entry_boundary_bars: int = 4      # enter only at multiples of this
    enter_under_db: float = -8.0      # first-entry gain relative to human's bar RMS
    ramp_bars: int = 3                # bars to ramp from enter_under_db to 0 (machine-internal)
    density_high: float = 2.5         # human onsets/beat that triggers thinning
    density_hysteresis: float = 0.8   # come back when density < high - hysteresis (one-bar delay)
    silence_thin_bars: int = 2        # human silent this long -> thin to skeleton
    silence_fade_bars: int = 6        # -> begin fading
    silence_tacet_bars: int = 10      # -> fully out, re-arm entry
    silence_rms_db: float = -45.0     # "silent" bar threshold
    conf_floor: float = 0.3           # beat conf below this -> fading/tacet (beat-loss repair)
    register_margin_st: float = 5.0   # keep voicings this many semitones away from human register
    energy_alpha: float = 0.5         # EMA for following the human's dynamic arc
@dataclass(frozen=True) class BarFeatures:      # summarized by sim/realtime per completed human bar
    bar: int; rms_db: float; onset_density: float   # onsets per beat
    silent: bool; register_low_midi: float | None; register_high_midi: float | None
@dataclass(frozen=True) class BarPlan:
    bar: int                      # the UPCOMING bar this plan is for
    state: str                    # element of STATES
    roles: dict[str, float]      # role -> level 0..1 (0/absent = tacet). roles from the style: kick/bass/pad/hat/keys...
    gain_db: float                # machine bus gain for the bar
    density: float                # 0..1 pattern-density hint (duet picks sparser patterns when low)
    register_ceiling_midi: float | None   # voice below this (register avoidance), None = free
    reason: str                   # human-readable ("entering: conf 0.71 at bar 8 boundary")
class Mind:
    def __init__(self, params: MindParams): ...
    def plan_bar(self, beat: BeatState, harmony: HarmonyState, human: BarFeatures | None) -> BarPlan
        # called once per bar boundary (or per second while unlocked, with human=None);
        # encodes: enter-late (min_listen_bars AND locked AND conf>=entry_conf AND boundary),
        # enter-under + ramp, inverse density coupling w/ hysteresis + one-bar delay,
        # silence 3-stage FSM (thin/fade/tacet + re-arm), beat-loss repair (conf < conf_floor ->
        # fade to tacet WITHOUT advancing silence counters), register avoidance, dynamic-arc
        # following (gain tracks EMA of human rms delta), texture changes ONLY at
        # entry_boundary_bars boundaries except emergency exits (beat-loss, tacet).
```

Pure and deterministic: no rng, no I/O; all memory inside Mind. `reason` strings
make the decision log (NDJSON) self-explanatory — they are part of the contract.

## 5. duet.py — bar-by-bar accompanist

```python
class DuetEngine:
    def __init__(self, *, style: str, key_fallback: str = "A minor", seed: int, sr: int = 48000): ...
    def render_bar(self, plan: BarPlan, beat: BeatState, harmony: HarmonyState) -> np.ndarray
        # returns stereo float32 of exactly one bar at beat.bpm (round to samples; carry remainder).
        # Chooses per-role patterns from the style's drum grids / bass_style / pad_style honoring
        # plan.density (sparse variants at low density) and plan.roles levels; harmony: current
        # chord/key belief -> root/voicing via theory (graceful degradation: chord_conf high ->
        # triad voicing, low -> root+fifth, None -> tonic pedal of key belief, no key -> key_fallback
        # pedal tone). Grooved via v0 groove with style swing. Voice tails may exceed the bar:
        # maintain an internal overlap buffer so consecutive bars crossfade correctly.
        # Registers: honor plan.register_ceiling_midi for tonal voicings.
    def flush_tail(self) -> np.ndarray    # remaining reverb/voice tail after the last bar
```
Deterministic per (seed, inputs). Reuses v0 voices/fx; no new synthesis code.

## 6. sim.py — the offline twin (THE verification surface)

> **Amended in v1.2 — see §11.13, §11.16–§11.18.** `simulate` takes an
> additive `bpm_hint`; a sounding plan issued off a bar boundary is now
> rendered; `--json` decimates `bpm_trace`. The fixture corpus alone was never
> a sufficient verification surface — see `tests/test_real_material.py`.
>
> **Amended in v1.3 — see §12.22–§12.24.** The decisions line's `density` is
> now `machine_density` (it collided with the human's density in `reason`);
> the QA gates in `tests/duet_qa.py` gained a minimum decisive sample and now
> gate on `enter_under.converges`.

```python
@dataclass class SimResult:
    out_path: str; decisions_path: str; duration_s: float
    entered_bar: int | None; states_by_bar: list[str]
    bpm_trace: list[tuple[float, float]]        # (t_s, bpm)
    final_key: str | None; manifest: dict       # peaks, rms, per-bus rms, params used
def simulate(input_wav: str, *, style: str = "lofi", out_path: str, seed: int = 0,
             params: MindParams | None = None, tightness: float | None = None,
             mix_human_db: float = 0.0, taps_s: list[float] | None = None) -> SimResult
```
Pipeline: read WAV (mono-sum) → chunk in 4096-sample blocks → FeatureExtractor
→ BeatTracker (apply taps_s as count-in if given) / HarmonyTracker → at each
predicted bar boundary: summarize BarFeatures for the completed human bar, call
Mind.plan_bar, DuetEngine.render_bar for the upcoming bar, place the rendered
bar at its predicted start time on the machine bus (sample-accurate) → final
mix human (at mix_human_db) + machine, soft-limit at −1dB → PCM_24 WAV.
Decisions NDJSON: one line per bar: `{bar, t_s, state, reason, bpm, conf,
key, chord, roles, gain_db, register_ceiling_midi, machine_density}` + one
`{"event": "…"}` line for lock/unlock/key changes. Deterministic
byte-identical per (input, args). `machine_density` is the machine's chosen
0..1 density; the human's measured density is onsets/beat and appears only
inside `reason` (v1.3, §12.24).

## 7. realtime.py — jam (real-time transport)

```python
def jam(*, style: str, seed: int, params: MindParams, device: str | None = None,
        monitor: bool = False, record_dir: str | None = None) -> None
```
sounddevice duplex stream 48k/256, callback copies to/from lock-free ring
buffers ONLY (no allocation). Threads: analysis (drains input ring → features
→ beat/harmony), planner (fires at predicted bar boundaries → Mind → Duet
renders next bar into the output schedule; render-ahead ≥ 1 bar, revisable
until 1 beat before the downbeat), main (status line: state/bpm/conf/key,
Ctrl-C = graceful ending: current bar finishes + tail). record_dir: writes
human.wav + machine.wav + decisions.ndjson. Import sounddevice lazily; clean
error without a device. Schedule-aware self-masking v1: while machine bus RMS
in a frame exceeds (human estimate − 6dB), suppress onset triggers from
feeding phase correction (flag frames `self_masked`) — full spectral masking
deferred. This module gets smoke tests only (no device in CI); all musical
logic lives in the tested sim path.

## 8. CLI additions (cli.py — additive only)

```
plume sim INPUT.wav [--style S] [--out duet.wav] [--decisions d.ndjson] [--seed N]
           [--tightness F] [--mind k=v ...] [--taps "0.5,1.0,1.5,2.0"] [--mix-human-db F] [--json]
plume jam [--style S] [--seed N] [--tightness F] [--mind k=v ...] [--record DIR] [--device NAME]
plume listen INPUT.wav [--json]     # NDJSON stream of BeatState/HarmonyState per ~0.5s + events; no synthesis
plume mind                          # print MindParams: names, defaults, meanings (--json structured)
```
`--mind k=v` sets any MindParams field (validated; unknown key errors listing
valid names). sim/listen/mind require no session file. `--json` on sim returns
the SimResult manifest.

## 9. Fixtures + behavior test matrix (tests/fixtures_gen.py + tests/test_behaviors.py)

`fixtures_gen.py`: `make_human(kind, *, bpm=96, bars=24, key="A minor", seed=1) -> np.ndarray`
— synthesizes "human guitarist" audio using v0 voices (pluck strums on chord
tones + occasional bass notes, humanized): kinds = `steady` | `drift` (bpm
+6% linearly over the middle 8 bars) | `stops` (2 silent bars mid, then
resumes) | `long_stop` (8 silent bars) | `busy_middle` (onset density triples
for 4 bars) | `rubato_intro` (first 8s free/no pulse, then steady) | `high_register`
(strums voiced above C5). Deterministic per seed.

Behavior matrix (each = one test, sim-based, asserting the decisions log):
1. steady: machine enters at a bar ≥ min_listen_bars, ON an entry_boundary
   multiple, in state "entering", with first-bar machine-bus RMS ≥6dB below the
   human bar RMS; bpm estimate within ±3 of truth after lock.
2. drift: bpm trace follows to within ±4 bpm by 2 bars after drift ends; no
   spurious half/double jumps (bpm never leaves [truth·0.9, truth·1.15]).
3. stops: within silence_thin_bars+1 of the stop → state "thinned"; human
   resumes → comping again within 2 bars; never advances to tacet.
4. long_stop: reaches "fading" then "tacet"; re-enters after resume only at a
   boundary, in "entering" with the under-gain again.
5. busy_middle: plan.density during the busy bars < density during calm bars
   (inverse coupling); recovery delayed ≥1 bar after density falls (hysteresis).
6. rubato_intro: no machine audio before lock (machine bus silent while
   unlocked); no "entering" until locked.
7. high_register: plans carry register_ceiling_midi below the human's
   register_low_midi − margin; rendered tonal voicings respect it (check via
   decisions + spectral centroid of machine bus vs steady fixture).
8. determinism: simulate twice → byte-identical duet WAV + identical NDJSON.
9. taps count-in: with taps at the true beat, lock is immediate (entered_bar ≤
   min_listen_bars + entry_boundary_bars) and phase error at first machine
   downbeat < 30ms vs truth grid.
10. listen-only: `plume listen` on steady fixture emits monotonically
    increasing confidence to lock, final key = fixture key (or relative — assert
    tonic pitch-class ∈ {key, relative}).
Plus unit tests per module (features onset timing ±1 frame on clicks; beat
α-correction math; harmony template/decay behavior; mind FSM transitions
table-driven; duet bar length exactness across bpm; sim mix levels).

## 10. Definition of done (v1)

- Full suite green (v0's 400 + all new tests).
- `plume sim` on all 7 fixture kinds: renders, decision logs coherent, all
  behavior tests pass, deterministic.
- A flagship duet: `plume sim` over a 24-bar steady strummer in F minor, lofi
  style — enters late, under, breathes with the human — saved to
  renders/duet_demo.wav + .ndjson (committed manifest, wav on disk).
- `plume jam` starts and errors cleanly with no device; with a device it runs
  (manual test documented, not CI).
- README updated (sim/jam quickstart); SPEC-V1 amendments noted inline if any.

---

## 11. v1.2 amendments — real-material corrections (2026-08-14)

v1 passed all ten synthetic-fixture behavior tests and its musical QA, then
failed on real music. A cold-start verifier using only the CLI found the
headline feature unreachable: the sideman never entered on any real input.
This section records what changed and why. **Every amendment below is a
deviation from the frozen v1 contract, adopted because the contract was wrong
about reality.** Each is guarded by `tests/test_real_material.py`, which runs
the full sim pipeline over plume's own v0 renders plus a bare click track.

**Root cause of the whole class.** `tests/fixtures_gen.py` peak-normalizes
every fixture to 0.85 and writes silence as exact digital zeros, so the fixture
family has **no input-gain axis, no crest-factor axis, no metric-alias axis and
no low-register axis**. Bar RMS lands at -15..-24 dBFS and silence at -63 dB;
nothing in the 40 dB band where real recordings live is ever generated. Four
separate absolute thresholds were therefore unreachable by construction, and
all four fired constantly on real input.

### 1 (section 1) — chroma gets its own longer window

`CHROMA_WIN = 4096` (11.7 Hz bins) for the chroma path only; `WIN = 2048`
still drives RMS / flux / onset / centroid. Both windows end at the same
sample, so frame timing, onset timing and block-invariance are unchanged.
*Why:* a 2048-point Hann main lobe is 93.8 Hz wide, while A3 (220.0) and C4
(261.6) are 41.6 Hz apart — they merged into one lobe whose interpolated
centroid landed near 243 Hz, pitch class **B**, a note in neither, flickering
with the partials' relative phase. That is precisely the register real music
puts its bass and chord roots in.

### 2 (section 1) — chroma requires harmonic corroboration

`CHROMA_REQUIRE_F0`: a candidate fundamental scores only if some peak matched
it at h == 1. *Why:* `CHROMA_TOL_ST = 0.35` was meant to reject implausible
harmonic hypotheses, but `12*log2(h)` is within 0.35 semitones of an integer
for **every** h in 1..9, so it only ever tested the peak's own
frequency-measurement error. A single spectral peak deposited
`{p: 0.661, p-7: 0.243, p-4: 0.096}` — the minor triad a perfect fifth below
the note played. A pure 440 Hz A read as `Dm` at 0.999 confidence. This is the
surviving mechanism behind out-of-key machine bass under the human's chord.

### 3 (section 1) — known limit: the onset threshold is still absolute

`ONSET_DELTA = 4.0` is an absolute offset above the trailing flux median.
Measured flux medians: 0.00 on the sparse pluck fixtures but 10.2 (lo-fi) and
43.4 (trap) on real mixes, and at -20 dB input gain onsets thin from ~10/s to
~2.8/s, which delays lock. **Not changed in v1.2** — a scale-aware threshold
needs its own fixture re-tune. Documented, and asserted only as "the tracker
must still lock with runway left to play".

### 4 (section 2) — tempo by FULL LAG SWEEP, not octave relatives

Deleted: the two-peak-family candidate generator and its
`for mult in (1.0, 0.5, 2.0)` loop; `density_fit` / `_DENSITY_SUBDIVS`; the
binary `_BAND_PREF_OUT` flag. Added: every integer lag in the bpm range is
scored (~47 per evaluation), by rectified ACF comb x log-normal tempo prior x
metric nesting x continuity.
*Why:* on real programmed music the strongest ACF lag is a **dotted (3-unit)**
grouping — hats group sixteenths 2+1 — so the truth was a 4/3, 3/2, 2/3 or 3/4
relative of the peak and the octave-only pool **could not propose it at all**.
Measured: a fresh 140 bpm trap render made 140 reachable at **0 of 4**
evaluations (pool = {90.7, 181.5, 362.9}); the tracker reported 185.6 = 140x4/3.
`density_fit` was provably an exact tie across the octave for onsets/beat in
{1.5, 2, 2.2, 4, 4.4, 5, 6, 6.7} — including the 6.7 the lo-fi mix measures —
because `_DENSITY_SUBDIVS` was closed under x2.
*Correction to the earlier diagnosis:* the lo-fi alias is exactly 4/3 and 2x,
**not** 80*sqrt(2), and the ACF-peak ratio is identical at swing 0.50 / 0.57 /
0.62. Swing is not involved.
Cost: 18 ms -> 46 ms per 50 s of audio (0.04% -> 0.09% of realtime).

### 5 (section 2) — ACF window 6 s -> 12 s, comb re-weighted to the bar

`_COMB_WEIGHTS = (1.0, 0.6, 0.35, 0.8)`: the m=4 (bar) term is near-primary.
*Why:* the bar is measurably the cleanest periodicity in real material (ACF
+0.807 at the bar vs +0.191 at the beat, lo-fi mix) and it is what separates
the true beat from its 3/2 and 3/4 aliases. At 6 s the bar lag of anything
under 120 bpm was outside the search window entirely.

### 6 (section 2) — the tempo prior is a wide log-normal, and nesting is gated

`_PRIOR_CENTER_BPM = 110`, `_PRIOR_SIGMA_OCT = 0.9`. The nesting bonus (bar lag
~= 4x beat lag) applies only when the supra-beat peak **outranks its own half** —
otherwise a plain click train, whose ACF simply decays across its comb of
peaks, hands the 4-grouping bonus to the double-tempo candidate and a true
60 bpm reads as 120. A narrow prior (sigma = 0.55) is not a tie-breaker but an
override of evidence, and broke exactly that case.

### 7 (section 2) — confidence reads the grid, not one lag

Salience is now the strongest periodicity the winning grid explains, over the
beat and its metric multiples. *Why:* reading only the candidate's own lag
under-reports real material badly (a grid that predicts the bar correctly is
well evidenced even when the beat-level ACF is weak); v1 accidentally got a
high number here by reading the **alias's** peak.

### 8 (section 2) — the TAP LATCH

Three latches hold while a count-in is anchored: salience is not overwritten by
the ACF, confidence is floored at 0.75, and **unlock no longer discards the
anchor**. Drift correction is bounded relative to the tapped tempo, not to the
last accepted step.
*Why:* the spec promised "hard-sets tempo and phase with confidence 1;
afterwards only corrects drift", but only the PERIOD was protected. Traced:
`TAP t=3.00 conf=1.000 sal=1.000` then `eval t=3.01 SAL 1.000->0.430`, with
confidence EMA-decaying to 0.59 over 11 s — and confidence is what gates lock
and what the sideman's entry gate reads. The one-bar unlock then cleared
`_tap_anchored` permanently. Drift was an unbounded random walk: 80.00 tapped
-> 78.33 after 30 s.

### 9 (section 2) — phase re-acquisition

After `REACQUIRE_MISSES` consecutive audible predicted beats with no onset near
them, an **unlocked, non-tap-anchored** tracker re-seeds its phase from the most
recent onset. *Why:* `_phase_correct` ignores onsets beyond +/-35% of a beat, so
a grid a half beat off is mathematically unreachable by correction — perfect
tempo, zero hit rate, confidence -> 0, forever. Reproduced deterministically,
and hit unprompted on the repo's own `rubato_intro` fixture.

### 10 (section 2) — new: `BeatTracker(bpm_hint=...)`

States a KNOWN tempo and narrows the prior around it (about +/-10%) **without
claiming a phase** — which is why it works on material the user did not cut on
a downbeat, the case where a count-in deadlocks. Exposed as
`plume sim --bpm-hint BPM`.

### 11 (section 3) — key belief needs seconds, not milliseconds

`_KEY_MIN_EVIDENCE_S = 2.0` of decayed voiced time replaces the effective
5-frame gate. *Why:* chroma is L1-normalised, so every voiced frame adds
exactly 1.0 mass and `_KEY_MASS_FLOOR = 5.0` was a floor of **53 ms**. A
`key_change` event fired at t=0.064 s on every input, and the key named there
was WRONG on 3 of 4 renders (lo-fi C major -> "C minor", trap C minor -> "F#
minor", flagship C minor -> "C major") and held for 5-11 s. `_KEY_CONF_FLOOR =
0.3` was no second guard: a histogram containing one pitch class already
correlates r = 0.684.

### 12 (section 3) — relative keys are sticky

A challenger that is the incumbent's relative major/minor must clear
`_KEY_RELATIVE_STICKY` (0.60) instead of the ordinary 0.05 margin.
*Why:* a key and its relative are the **same pitch collection**, so a
pitch-class histogram cannot distinguish them in principle — only metric
position and bass register can, and this tracker sees neither. On loop-based
modal material the raw distribution actively favours the relative (trap is
Cm/Eb/Ab/Bb: Eb appears in 3 of 4 chords, C in 2), and C minor lost to Eb major
by +0.32 to +0.38 — far above the ordinary margin, so the existing hysteresis
never engaged. Swapping in Temperley Kostka-Payne or Albrecht-Shanahan profiles
answers Eb major too and additionally breaks C-major lo-fi into F major: it is
not a profile-set bug. **Known limit:** a genuine modulation that arrives via a
non-relative key can still land on the relative; `renders/flagship.wav` does
this (C minor -> Bb major -> Eb major). Impact is bounded — same pitch
collection, so scale choice and accidental spelling are unaffected; only the
tonal centre is. Tests accept the true key or its relative.

### 13 (sections 4 and 6) — bar silence is onset-guarded and RELATIVE

The bar-silence rule is now, in order: **(a)** detected onsets in the
edge-trimmed core => NOT silent; **(b)** otherwise silent below the effective
threshold, which is the absolute `silence_rms_db` floor raised to
`ref_db - SILENCE_REL_DB` (15 dB) once the human's running level is known.
*Why:* an absolute floor cannot decide whether a human is playing. A sparse
percussive bar's MEAN RMS runs 13-25 dB below its peak frames: a 120 bpm click
at -21 dBFS peak, with **four detected onsets per bar** and the tracker locked
at conf 0.93, measured -52.5 dB and was flagged silent on every bar. Onsets are
the level-independent evidence of playing that the code already computed and
then ignored on the RMS path. 15 dB and not 25: the `stops` fixture's
resume-bleed bar drops 15.4 dB below its running reference, and at 25 dB v1
behavior 3's single `thinned` bar disappears.

### 14 (section 4) — the mind trusts `BarFeatures.silent`

`Mind.plan_bar` no longer ORs in its own absolute-RMS test. *Why:* it was a
second, independent, unguarded copy of the gate — proven by monkeypatching
`sim._BarAccum.is_silent` to always return False, which changed **nothing**.
Only the layer holding the frame data (per-frame RMS, onsets, the edge trim,
the human's running level) can decide silence. `realtime.bar_features_dict`
held a third copy; it now shares `SILENCE_REL_DB` with sim so the two cannot
drift apart again.

### 15 (section 4) — `entry_conf` 0.55 -> 0.45

*Why:* 0.55 was EXACTLY `BeatTracker.lock_conf`, and the entry gate samples
confidence instantaneously at the bar boundary with no hysteresis of its own,
while the tracker's lock has plenty. Measured on a real render at an ordinary
live-mic level (RMS -27 dBFS): boundary confidences 0.56, 0.55, 0.50, 0.51, ...,
0.55 — the two samples that cleared 0.55 landed on bars 0 and 9, neither a
multiple of 4, so `entered_bar` was None over a 49.5 s take with the reason
"locked but conf 0.55 < entry threshold 0.55". A test now asserts
`MindParams().entry_conf < BeatTracker()._lock_conf`.

### 16 (sections 4 and 6) — `min_listen_bars` counts bars HEARD

`Mind.credit_listened_bars(n)` (new, additive) lets sim/realtime credit bars
the human played before the tracker locked; `sim._summarize` caps the measured
span at one bar. *Why:* the only call site that ever passed a `BarFeatures` was
the locked-boundary path, and the accumulator was never reset before the first
boundary — so 11.9 s of real music (about 5 bars) was credited as exactly ONE
bar with a diluted RMS, and `min_listen_bars` meant "4 bars **after** an
internal lock event the user cannot see". Entry landed at t=25.1 s of a 49.5 s
take. Credited bars satisfy `min_listen_bars` but cannot arm entry on their
own: `_MIN_BARS_ON_GRID = 2` non-silent bars must also have been measured on
the locked grid, because the first such bar is the one during which the lock
happened, and entering on a freshly-crossed lock threshold is exactly the
"entered on a wrong tempo hypothesis" failure the enter-late gate exists to
prevent.

### 17 (section 6) — the unlocked tick can sound

`sim` now renders and places a plan issued off a bar boundary when its state is
still sounding. *Why:* the beat-loss repair's `fading` bar is issued on the
unlocked tick path, which never rendered anything — so a mid-take rubato made
the machine stop dead instead of fading out, which is the one thing the repair
exists to avoid.

### 18 (sections 6 and 8) — reporting papercuts

`plume sim --json` decimates `bpm_trace` to 32 evenly spaced points (first and
last always kept) and reports `bpm_trace_points_total`; `--full-trace` restores
every point, and the NDJSON always has all of them. Reason strings no longer
emit `-0.0dB`. `SimResult.manifest` carries `bpm_hint`, and `no_entry_reason`
already named the gate that stayed closed.

### 19 — accepted, documented behaviours (not bugs)

- **Half-time on half-time material.** `renders/trap.wav` tracks 70 rather than
  its notated 140: its beat-level flux ACF is **negative** (-0.122,
  anti-correlated backbeat) while the bar level is +0.666, so a half-tempo
  reading is a legitimate perceptual answer that simply is not the notated one.
  Forcing 140 structurally requires a bonus that breaks a true 60 bpm click.
  `--bpm-hint 140` recovers it (measured 138.9). Same class: steady 180 bpm
  locks half-time at 90. *(v1.3: the regression guard no longer reads that
  gitignored file — see amendment 21.)*
- **The chord belief is one fixed-lag report behind** on 1-chord-per-bar loops
  (`_HMM_LAG` smoothing is reactive by construction). Unchanged in v1.2.

### 20 — test-suite changes made alongside these fixes

Two existing tests asserted values that only held because of a bug, and were
corrected to assert the true behaviour:

- `tests/test_mind.py::test_mindparams_fields_and_defaults_match_spec` —
  `entry_conf` 0.55 -> 0.45 (amendment 15).
- `tests/test_beat.py::test_silence_decays_confidence_and_unlocks` — the unlock
  deadline moved 28.0 s -> 29.0 s. Unlock time is a function of the confidence
  held when the music stopped, and v1.2 measures that confidence correctly
  (0.967 at t=16 s) instead of under-reporting it, so the same decay crosses
  the threshold 0.2 s later. The behaviour under test is unchanged.

No other existing test was modified.

---

## 12. v1.3 amendments — verifiability (2026-08-14)

v1.2 fixed the sideman. v1.3 fixes the *evidence*: a suite that skipped its
only coverage of a feature on any clean clone, a QA report nobody could
regenerate, two QA gates that could not fail, and two places where a name or a
number said something the code did not.

### 21 (sections 9 and 10) — no test may read a gitignored artifact

`*.wav` is gitignored and **no WAV is tracked**, so any test reading `renders/`
passed only on the laptop that made the file and skipped everywhere else.

- `tests/test_real_material.py::test_bpm_hint_recovers_the_notated_tempo_on_half_time_material`
  read `renders/trap.wav` and called `pytest.skip` when it was absent. It was
  the **only** test of `bpm_hint`, so on a clean clone that feature had **zero**
  coverage. It now synthesizes its own half-time trap feel from plume's voices
  (kick on 1, snare on 3, hats on the offbeat eighths, 808 sub under the kick —
  nothing on 2 and 4, so the beat-lag flux ACF is anti-correlated). Measured:
  **70.0 bpm unhinted, 138.9 with `bpm_hint=140`**. Both halves are asserted —
  if the material ever stops being half-time-ambiguous the hint proves nothing,
  so that must fail loudly rather than pass quietly.
- `tests/duet_qa.py` read `renders/duet_demo_input.wav`, which was gitignored
  with **no generator in the repo** — the flagship block of the committed
  `renders/duet_qa_report.json` could be read but never checked. It is now
  built by `make_flagship_input()` from `FLAGSHIP = {steady, 96 bpm, 24 bars,
  F minor, seed 1}`, which *is* the section 10 sentence in code. The WAV is
  still not committed (8 MB of derivable data); the report now carries `input`
  and `input_from` so a reader knows exactly what produced it.
  **The regenerated input is not byte-identical to the old one.** Its length
  (2 880 000 samples), rate, channel count and 0.85 peak match this generator
  exactly and its chroma matches F minor to three decimals, so the old file
  was certainly made this way — but fixtures_gen has moved since and the seed
  was never recorded. That is precisely the failure mode: an artifact with no
  generator can only be guessed at. The committed flagship numbers moved when
  this landed, and are now checkable.

Rule of thumb going forward: **generate, don't commit.** A test that can
silently not run is not a test.

### 22 (section 6) — `bass_roots` needs a minimum decisive sample

The QA gate scored only bars whose low-register chroma had a clear winner
(`chroma_conf >= 0.30`) with **no floor on how many bars that was** — so a take
with one decisive bar scored 1/1 and passed. Two floors now, and they do
different jobs:

- `bars_decisive >= 4` — one full cycle of fixtures_gen's 4-chord progression,
  the smallest sample in which a wrong root could appear at all. This is the
  floor that binds, and the number comes from the fixture's harmony rather than
  from the observed pass/fail line.
- `decisive_frac >= 0.20` — a collapse detector for long takes, deliberately
  set low and **not** a quality bar. Being explicit about why: the synthwave
  takes legitimately measure only ~0.35–0.41 decisive (sustained, gliding sub
  bass often leaves a bar's low chroma with no clear winner), **and the
  fraction moves with the tracker** — while the listen stack was being changed
  under this script, `duet_steady_synthwave` measured 5/17 = 0.294 and then
  6/17 = 0.353 from the same fixture. A fraction set anywhere near passing
  material would fail on beat/harmony work that is not a bass regression at
  all. That the bass measurement is this weak on sustained sub voices is itself
  the finding, which is why `decisive_frac` is now reported on every take.

Non-decisive measurements are **surfaced rather than dropped**:
`non_decisive_out_of_key` lists every smeared bar whose weak winner is out of
key (`duet_drift_synthwave` had three measuring Bb in A minor), and the
console summary prints them. They are not gated — below the salience floor the
winner is a coin flip between two overlapping bass tails — but a rise there is
the early warning that the decisive bars are about to follow.

### 23 (section 6) — `enter_under.converges` is gated, not decorative

`converges` was computed, was **False** on both `duet_stops_*` takes, and was
then left out of `pass` — so a metric nobody could fail sat in the report
looking like a check. It was false for a measurable reason: convergence was
measured over *every* post-entry bar including the human's stop bars, where
the human is digital silence and machine-minus-human is a division-by-silence
artefact (`later_comping_delta_db` reported **+62.0 and +224.2 dB**).
Convergence now skips bars whose human level is under −60 dBFS, reports which
ones it skipped (`bars_skipped_human_silent`) and the surviving
`later_mean_delta_db`, and **is required to pass**. All 7 takes converge:
means −0.78 … +0.98 dB.

### 24 (section 8) — `machine_density`, and help that reads the dataclass

Two ways the log and the docs said things the code did not:

- **NDJSON field collision.** A decisions line carried `"density": 0.3`
  alongside the reason `comping sparse: human busy (7.9 onsets/beat > 2.5)` —
  the machine's chosen density (0..1) and the human's measured density
  (onsets/beat), two quantities in two units under one name on one line. The
  machine's is now **`machine_density`**; the human's remains reason text only.
  `BarPlan.density` is unchanged — the ambiguity was only ever in the log.
  Readers updated: `tests/test_behaviors.py`, `tests/test_fixes_v1_1.py`.
- **CLI doc drift.** `plume sim --help` carried a hand-written MindParams table
  claiming `entry_conf float 0.55` — the pre-amendment-15 value — while
  `plume mind` read the dataclass and printed the true 0.45. The table is now
  generated by `cli._mind_table()` from `dataclasses.fields(MindParams)`, the
  same source `plume mind` uses, so there is no second copy of a number left to
  go stale. Only the prose lives in `cli._MIND_PARAM_DOCS`.

### 25 — artifacts regenerated

`renders/duet_*.{ndjson,qa.json,sim.json}`, `renders/qa/*.machine.ndjson`,
`renders/duet_demo.json` and `renders/duet_qa_report.json` were regenerated by
`.venv/bin/python tests/duet_qa.py` — required by the `machine_density` rename,
the new QA fields, and the now-synthesized flagship input. `all_pass: true`.

These artifacts are a snapshot of the whole listen → mind → duet stack, so they
also capture whatever `listen/{beat,harmony,features}.py` looked like at the
moment of the run. **Regenerate them once more after any further change to that
stack**, and treat a diff in them as a signal to read rather than as noise —
that is what they are for.

### 26 (section 2) — the tempo octave is sticky once settled, and never moves silently

**The defect.** Period hysteresis was one rule for two different decisions:
"switch bpm only if the new hypothesis differs >`drift_pct`% AND persists ≥2
evaluations". A 4% drift correction and a **halving of the metre** were
therefore governed identically — one second of contrary evidence. Measured on
24-bar and 32-bar 140 bpm trap renders: the tracker locked at ~140, held
139.8–140.7 for **fifty seconds across thirty bar boundaries**, then at t=54.1
dropped to 69.88 and stayed there for the rest of the take.

Three things made it invisible and harmful:

- **No event.** `locked` never changed, so a consumer watching lock state saw a
  perfectly healthy tracker on a grid at half the human's tempo.
- **Confidence ROSE across the collapse** (0.694 → 0.736 → 0.769). A half-tempo
  grid predicts half as many beats and every one of them is still a beat the
  human played, so the predicted-beat hit rate went 0.875 → 1.000. The
  certainty metric *rewards* a coarser grid.
- **Every onset-density reading downstream halved**, which silently disables
  the mind's inverse-density coupling. Musically: a drummer does not halve the
  tempo at bar 20 because the texture thinned.

**What separates a correction from a collapse — measured, not assumed.** Three
cases, challenger exactly one octave from the grid in force:

| case | when | challenger score ÷ grid in force | consecutive evals |
|---|---|---|---|
| `groove80` (truth 80, tracked 160) | 1.5 s post-lock | 1.01–1.08 | **82, unbroken** |
| `busy80` (truth 80, tracked 160) | ~1 s post-lock | 1.10–1.17 | **82, unbroken** |
| trap collapse (truth 140) | 50 s post-lock | 1.04–1.07 | **6, then gone** |

The **score margin does not separate them at all** — the collapse's 1.053 sits
inside the legitimate correction's band. A margin gate was built, measured, and
deleted: any margin that blocks the collapse also blocks a real correction. Two
things do separate them, and both are now used.

**The rule.**

- `_octave_is_settled` — locked, and `OCTAVE_SETTLE_S` (8 s) elapsed since the
  later of the lock and the last grid change outside the drift band. A tracker
  one bar past its lock is still reading the metre; one forty seconds into a
  stable take is being played along with.
- A **settled** octave yields only to `OCTAVE_SWITCH_EVALS` (16, ~8 s)
  consecutive winning evaluations; one evaluation where the grid in force wins
  back resets the clock. An **unsettled** one keeps the ordinary 2-evaluation
  hysteresis, which is what lets the tracker correct plume's own 80 bpm
  backbeat groove off the 160 alias 1.5 s after locking.
- Within-octave adjustment is untouched: `drift_pct` hysteresis exactly as
  before. `_is_octave_jump` measures |log2(new/old)| against a whole octave
  with a 6% tolerance, so the 4/3, 3/2 and 2/3 dotted aliases the v1.2 lag
  sweep exists to reach stay on the fast path.
- **Not applied before lock at all.** The deliberately-accepted half-time
  reading of trap material (beat-level flux ACF negative — an anti-correlated
  backbeat — while the bar level is strongly positive) is a pre-lock decision
  and is unchanged.

**Observability.** Any octave change *while locked* — settled or not — appends
a `tempo_octave_change` to `BeatTracker.events` (`t_s`, `from_bpm`, `to_bpm`,
`settled`, `conf_before`, `conf_after`) and increments
`BeatState.tempo_octave_changes`, a **monotone counter** rather than an edge
flag so a consumer polling at any cadence cannot miss it. `sim` writes the
matching NDJSON line. A **settled** change also multiplies confidence by
`_OCTAVE_CONF_DROP` (0.5): re-reading a metre you have played along with for a
phrase is a loss of certainty, not the gain the mechanical hit rate reports.
An unsettled correction gets the event but no cut — halving confidence for
being right would only delay the sideman's entry.

**Salience now describes the grid in force.** `_evaluate_tempo` wrote the
winning candidate's salience into confidence and *then* asked hysteresis
whether to adopt it, so while hysteresis was refusing a switch confidence was
already reporting the challenger's evidence. Traced on the same take: at
t=53.14 the tracker held 138.89 with salience 0.443; at t=53.64 the half-tempo
rival won the sweep, was **not** adopted, and salience jumped to 0.872 anyway.
The assignment now happens after `_adopt_bpm` and reads the lag of `self._bpm`.

*Known residual (not fixed, documented):* the evidence term is `max` over a
candidate's metric multiples, so a grid and its half share most of their
support by construction — after the fix the same evaluation reads 0.813
instead of 0.872. Salience cannot discriminate f from f/2 and is not asked to;
the discrimination lives in the comb/prior/nesting ranking and in this
stickiness rule.

**Tests.** `tests/test_beat.py`: the octave-jump predicate (including that
dotted aliases are *not* octaves), the settled-vs-unsettled evaluation cost, a
challenger that never persists, a genuine sustained double-time that still
changes octave (late by a phrase, by construction) and announces it with
confidence falling, the count-in's authority over both, and the
salience-ordering invariant. `tests/test_real_material.py`: a 24-bar trap take
whose settled tempo must not move, must not move without an event, whose
logging path is exercised end-to-end, **and a non-vacuity test that reproduces
the collapse on the same input with the pre-v1.3 rule restored** — if that
ever stops collapsing, the render needs replacing, not the guard relaxing.

---

## 13. v1.4 amendments — the key belief hears the bass (2026-08-14)

Amendments 11 and 12 stopped the key estimator naming a key from 53 ms and
stopped it sliding into the relative. Three things they did not fix: the
estimator had **no register information at all**, it would name a key from
material that is not pitched in any musical sense, and its first commitment
was made on ~2 s of audio and then asserted at full confidence.

*Numbering:* this block starts at 40 rather than continuing §12, because §12
was still being appended to by concurrent `listen/beat.py` work when these
landed. 27-39 are deliberately free for it.

### 40 (section 1) — `Frame.chroma_low`, a register-tilted bass chroma

New Frame field, shape (12,) float32. Built from the *same* corroborated
harmonic-salience votes as `chroma`, keeping only candidate fundamentals in
`[CHROMA_BASS_MIN_MIDI, CHROMA_BASS_MAX_MIDI]` = midi 28..60 = 41.2-261.6 Hz,
each weighted by `0.5 ** ((midi - 28) / CHROMA_BASS_HALFLIFE_ST)` with
`CHROMA_BASS_HALFLIFE_ST = 5.0` semitones, so within the band the **lowest**
fundamental dominates (a C2 root outweighs an E3 by 9x).

It is **not** L1-normalised. It is divided by the frame's full vote mass, so
`chroma_low.sum()` is the low band's tilted **share** of the frame's pitched
evidence, in [0, 1] — a magnitude as well as a direction, which makes it
self-gating. Measured medians: pluck fixtures with an 808 bass 0.127, plume's
renders 0.040-0.056, the `high_register` fixture 0.006, a click track 0.000.
An L1-normalised version would hand a click the same unit of "root evidence"
as a bass note. `chroma` is untouched, so chord matching is unchanged.

Frames are duck-typed throughout `listen/`, and `chroma_low` is defaulted, so
pre-v1.4 frame stubs still work — `harmony` falls back to the flat chroma.

### 41 (section 3) — the key histogram is weighted by register

`_KEY_BASS_WEIGHT = 2.0`: the key histogram accumulates
`chroma + 2.0 * chroma_low` instead of `chroma`.

*Why:* `chroma` is flat across registers, so a bass root counts exactly as
much as an upper extension — right for CHORD templates (which have no
register) and wrong for KEY, where the bass note is the strongest single piece
of evidence a listener has. **Minimal repro (pure sines):** C2 + E3 G3 B3 D4,
a Cmaj9, gave chord `C` (correct) and key **`G major`**, because the
collection {C,D,E,G,B} scores 23.67 against the G-major KK profile and 22.28
against the C-major one once the root's register is discarded. Doubling the C2
loudly recovered `C major` — i.e. the estimator was responding to pitch-class
*mass*, not to root evidence. After: `C major`, and the loud-root variant
still answers `C major`.

The weight is > 1 because the term is a share, not a unit vector. Measured
safe band on the whole probe corpus (the repro, pure tones, all four renders,
the pluck fixtures): **1.0 to 3.0 all give the same correct answer
everywhere**; 2.0 is the middle. Above ~4 the renders' relative-major pull
drags the incumbent's own correlation toward `_KEY_CONF_FLOOR` and synthwave
starts dropping its key entirely.

### 42 (section 3) — unpitched material may not acquire a key

`_KEY_DUTY_FLOOR = 0.5`: a key may only be **acquired** while the decayed
voiced-time fraction (`_evidence_s / _elapsed_s`, both decayed at the same
half-life) clears the floor.

*Why:* a bare click track named a key at conf ~0.65 and the sideman comped on
it. What separates a click from music is not the spectrum of any one frame — a
1 kHz click really is a tone — but **sustain**. Measured voiced-frame
fraction: click track 0.14, every render and every fixture 1.00. A kick-only
pattern, which does put energy in the bass band, is rejected the same way.
Unpitched input now reports `key=None`, `key_conf=0.0`, and the duet falls back
to its `key_fallback`.

It is an acquisition condition only, never a retention one: a long rest must
not erase an established key (the `long_stop` fixture is 8 silent bars), so
retention still uses the absolute evidence floor alone.

### 43 (section 3) — early belief is provisional, and says so

`key_conf` is discounted by `min(1, evidence_s / _KEY_COMMIT_EVIDENCE_S)` with
`_KEY_COMMIT_EVIDENCE_S = 8.0`.

*Why:* the first `key_change` fired at **exactly t=2.112 s in every run** — the
instant the evidence clock crossed `_KEY_MIN_EVIDENCE_S`, on every input alike
— and the belief was then asserted at 0.64-0.89 confidence on ~2 s of audio.
Because a key is only reported once its confidence clears `_KEY_CONF_FLOOR`,
the discount also makes the first declaration a property of the material:
measured spread across four synthesized inputs went from 0.085 s (2.112-2.197)
to ~1 s (3.1-5.2).

**Not** changed with it: the switch hysteresis. An earlier version of this
amendment also let a belief below the commitment threshold be re-pointed at
the running argmax with no margin and no sustain, reasoning that an
uncommitted belief is not worth defending. Measured, that is strictly worse: a
16-bar C-minor synthwave render states C minor for its first ~0.7 s of
evidence and then spends the take on its bVII and bIII, so free re-evaluation
walked the belief to Bb major and left it there — and trap did the same.
Early correlation ordering on loop-based material is not a better estimate
than the incumbent, only a noisier one.

### 44 (section 3) — closely-related keys are sticky too

`_KEY_NEIGHBOR_STICKY = 0.15` applies to any challenger whose diatonic
collection differs from the incumbent's by one accidental (symmetric
difference of 2 pitch classes). Amendment 12's relative-key rule is now the
`distance == 0` case of the same quantity; `_COLL_DIST` is the pairwise table.

*Why:* amendment 12 handled the extreme case of a general fact. Two keys one
accidental apart share 6 of 7 pitch classes, so a pitch-class histogram
separates them by very little. `renders/flagship.wav` (C minor throughout)
walked C minor -> **Bb major** -> Eb major; Bb major is not the relative, so
the 0.60 rule never applied and the ordinary 0.05 margin was cleared by a
passing bVII-heavy phrase. The A-minor pluck fixtures drifted A minor ->
F major -> C major within 8 s for the same reason.

The value is bounded from above by a measurement, not by taste: on material
unambiguously and entirely in one key the correlation gap to its one-accidental
neighbour **tops out around 0.19** (measured 0.187, E major over G# minor on a
saturated E-A-B loop). A margin at or above that would make the rule
*absolute* — a real modulation to a neighbouring key could never be followed
at all. 0.15 sits just under the ceiling.

**Measured cost:** following a genuine modulation to a neighbouring key takes
roughly twice as long — on the stub scenario in
`test_a_neighbour_key_is_sticky_but_a_real_modulation_still_lands`, ~28 s of
new material under the old margin versus ~49 s now. That is the trade this
amendment makes deliberately: plume's material is overwhelmingly single-key
loops over 1-3 minute takes, where a stable belief is worth more than fast
neighbour tracking.

### 45 (section 1) — known limit: chroma still merges CLOSE triads at C2

Pre-existing, found while verifying amendment 41, and **not** introduced by it.
Amendment 1 widened the chroma window to 4096 to stop A3+C4 (41.6 Hz apart)
merging into one lobe. The same merge returns an octave lower and 4096 does not
stop it there: the main lobe is 46.9 Hz wide while C2-E2 are 17.0 Hz apart and
E2-G2 are 15.6 Hz apart. A close triad voiced in the C2 octave reads as noise —
measured on pure C2+E2+G2 sines, chroma is F#:0.22 G#:0.14 D:0.13 B:0.12 with
**no C at all**, and the key comes out "B minor". Resolving 15.6 Hz needs
`CHROMA_WIN = 16384` (341 ms), which would smear chord changes badly.
Documented rather than fixed: bass parts are voiced in octaves and fifths, not
close triads, and a bass note against a chord an octave up (C2 + C3 E3 G3)
resolves correctly today (chroma C:0.62 E:0.20 G:0.19).

### 46 (section 1) — known limit, still open: the onset threshold is absolute

Amendment 3 recorded that `ONSET_DELTA = 4.0` is an absolute flux offset while
measured flux medians span 0.00 (sparse fixtures) to 85.9 (trap). Re-measured
for v1.4 with a scale-aware rule `median + max(ONSET_DELTA, k * MAD)`:

| material | 0 dB | -20 dB | -40 dB | ratio -20/0 |
|---|---|---|---|---|
| lo-fi render, absolute delta | 7.69/s | 5.50/s | 1.08/s | 0.72 |
| lo-fi render, `k = 4` | 4.40/s | 3.78/s | 1.08/s | **0.86** |
| trap render, absolute delta | 11.58/s | 10.20/s | 3.28/s | 0.88 |
| trap render, `k = 4` | 5.32/s | 4.60/s | 3.01/s | 0.87 |
| pluck fixtures (all kinds) | unchanged for k <= 8 | | | |

The rule does what it claims — gain invariance improves and sparse fixtures are
untouched, because their MAD (<= 0.61) never reaches the absolute floor — but
it also **cuts the onset rate on every dense mix by ~40% at full level**, which
is a large change to the beat tracker's only input, and it does not help at
-40 dB (there the flux itself falls under the absolute floor and `k * MAD` no
longer binds). Not shipped in v1.4: the change belongs with `listen/beat.py`
work, where its effect on tempo lock can be measured, not alongside a harmony
change.

## 14. v2.6 amendment — the relock bias (2026-08-27)

### 47 (section 2) — a lock LOSS no longer erases the metre

Amendment 26 made the tempo octave sticky *once settled*, and gated the whole
mechanism on `self._locked`. That gate is the hole: the moment confidence
falls through `UNLOCK_CONF` for a bar, the tracker has no memory of the metre
it just spent a minute playing, and the next evaluation is a cold reading. A
busier texture at the SAME tempo then relocks on the double.

**Measured on the live take** (`takes5/`, 944 s of open-tuned acoustic guitar,
`--input acoustic`, lo-fi follow): lock lost while comping at 82 bpm, re-entry
at 118; lost at 90, re-entry at 184. Downstream this is amendment 26's damage
reached through the unlocked door — the mind comps on a grid at the wrong
multiple of the human's tempo, and every onset-density reading it takes is
wrong by that multiple.

**The rule.** While unlocked, the tempo sweep multiplies each candidate by a
log-normal in octaves (`_RELOCK_SIGMA_OCT` 0.35) centred on the remembered
tempo, at a strength that is

- **1.0** while the human is still playing after the loss, decaying to 0 over
  `RELOCK_HOLD_BARS` (32) bars of playing, and
- **0 outright** once they have been silent longer than `RELOCK_BREAK_BARS`
  (8) bars — *silence ends a take; playing does not*.

A log-normal and not an octave test, because the aliases a relock actually
picks are not all octaves: the live take went 82 -> 118, a 3/2. Not applied
while locked (amendment 26 is in charge there) or while tap-anchored (the
count-in is), and a fresh `tap()` clears the memory outright.

**The remembered tempo is a MEDIAN, and that is the load-bearing detail.** The
first cut remembered `self._bpm` at the instant of the unlock, which is
routinely a tempo that has already collapsed — by definition confidence has
been under `UNLOCK_CONF` for a full bar by then. On the quiet 32-bar lo-fi
render the tracker holds 80.11 locked at conf 0.74 for forty seconds, slides
127.84 (conf 0.55) -> 181.45 (conf 0.20) over four seconds, and only then drops
the lock. Anchoring the bias on 181.45 pinned the outro at **167.91 against a
truth of 80** — it converted a wobble the tracker used to recover from into a
permanent collapse, and `test_acoustic`'s P2 guard caught it. The shipped
reference is the median over the last `RELOCK_REF_BEATS` (32) beats that were
both locked and at or above `lock_conf`; the collapse never earned confidence,
so the median reads 80.

**Sizing.** Instrumenting the sweep across a synthetic relock (quarter-note
texture at 80, a stop, an eighth-note texture at the same 80) the double-tempo
alias beats the truth by **1.99x** at the first evaluation after the resume and
by 1.06x once the ACF window has filled; 1.99x is what the bias must overcome.
At sigma 0.35 an octave-away rival is handicapped 59x at full strength and
55x -> 40x across that fixture's ten seconds between resume and relock. The
width is set by the other end of the trade, not by that margin: sigma 0.5 was
tried first and cost the P2 guard above, while 0.35 restores it and improves
its off-grid fraction 0.11 -> 0.06. A genuinely new tempo is handicapped too
(a real count-off at 120 against a lost 80: 4.04x) and still wins, because the
evidence for it has no 80-periodicity in it at all — a test, not an argument.

`RELOCK_HOLD_BARS` was swept with `tests/relock_replay.py` over the takes5
replay, against how far the reported tempo sits from the tempo held at the last
lock while unlocked — the number the mind actually enters on, since
`MindParams.entry_conf` gates on confidence, not on lock:

| hold | median wander | within ±11% |
|---|---|---|
| bias off | 0.279 oct | 35.8% |
| 8 bars | 0.255 oct | 34.9% |
| 16 bars | 0.229 oct | 36.1% |
| **32 bars** | **0.217 oct** | **38.9%** |
| 64 bars | 0.206 oct | 39.9% |

**Two clocks, split on silence — the other first cut that was wrong.** A single
clock running from the unlock burned its window on the rest and expired before
the tracker got anywhere: on takes5 the tracker loses lock at t=495 holding
104 bpm and does not lock again until t=594, and across those 99 seconds *the
human never stops playing* while the reported tempo wanders 59.8 -> 118.4 ->
187.7 -> 91.5. The silence test reads a new `_last_onset_t` and not the
`_onsets` deque, which is trimmed to `ACF_WINDOW_S` and therefore answers any
question about a rest longer than twelve seconds wrongly. The weight is read at
the TOP of `_evaluate_tempo`, above every early return, because the rest of
that function gives up on silence and silence is exactly when the bias must
expire.

**Honest scope.** The same replay shows the two lock losses in that take
relocking at 0.88x and 1.22x of the lost tempo either way, and the locked
fraction unmoved at **13.7%**. This amendment steadies the tempo the mind reads
*between* locks. It does not make sustained acoustic guitar lockable — the
tracker spending 86% of a take unlocked is a separate, larger defect, and
amendment 46's scale-aware onset threshold is the untried lead on it.

**Tests** (`tests/test_beat.py`, "relock bias"): a short stop relocks at the
tempo that was lost; the fixture is pinned as genuinely ambiguous cold (the
non-vacuity guard, without which the bias could be deleted with the suite still
green); a genuinely new tempo still wins; and a long stop is read cold, which
pins the break window as a real boundary rather than one that can silently grow
to forever. `tests/relock_replay.py` is the takes5 measurement, kept runnable
and deliberately NOT a test — it reads a gitignored recording, which amendment
21 forbids a test to do.

## 15. v2.7 amendment — the metre stops moving under a settled take (2026-08-28)

### 48 (section 2) — the commitment gate is about commitment, not about the ratio

Amendment 26 made a LOCKED grid resist re-interpretation once settled, and
reached that gate only for whole-octave jumps. Every other alias — 4/3, 3/2,
9/8, 5/4 — kept the ordinary two-evaluation period hysteresis, on the explicit
reasoning (recorded in `test_is_octave_jump_only_fires_on_whole_octaves`) that
a dotted alias is what the v1.2 full lag sweep exists to *reach*, and that
making it sticky would freeze the tracker onto its own dotted aliases.

**Three live acoustic takes falsified that.** On `takes6/` — 370 s of
fingerstyle guitar, no count-in, the take Juniper played on 08-27 — the locked
grid was re-read 54 times, 8.8 a minute. The octave, the only family amendment
26 guarded, was **two of them**:

| family | 4/3 | 3/2 | 9/8 | 5/4 | octave |
|---|---|---|---|---|---|
| count | 22 | 13 | 8 | 6 | 2 |

A guard that covers 4% of its own failure mode is not a guard. So every
out-of-drift re-read of a locked grid now goes through `_grid_hysteresis`
(renamed from `_octave_hysteresis`, because it was never about octaves) and
faces `OCTAVE_SWITCH_EVALS` consecutive winning evaluations once the grid is
settled.

**The amendment-26 reasoning survives, relocated to settledness.** What makes
an early correction legitimate is that the tracker has not committed yet, not
that the correction happens to be an octave. An UNSETTLED grid still yields in
two evaluations at any ratio, so the sweep can still climb off a dotted alias
in its first bars — which is the only place that correction was ever wanted.
Pinned both ways by `test_a_settled_grid_resists_a_dotted_alias_exactly_like_an_octave`
and `test_an_unsettled_grid_still_corrects_a_dotted_alias_in_two_evals`, the
second of which fails if the two paths ever report the same number.

**Measured, `tests/grid_guard_replay.py`, three recorded takes.** The A/B arm
restores the amendment-26 routing by subclassing the tracker and sending
non-octave re-reads back to `_period_hysteresis` — the real method, which is
why extracting it from `_adopt_bpm` was part of this change. Re-reads are
counted off the tempo trajectory rather than off `BeatTracker.events`, because
the old arm emits no events for non-octave changes and an A/B whose two arms
are read by different instruments is not an A/B. "On-truth" is the fraction of
the take whose reported tempo is within 5% of the take's octave-folded median
tempo — the number the mind comps on, since `MindParams.entry_conf` gates on
confidence and not on lock.

| take | re-reads | on-truth | locked | median conf |
|---|---|---|---|---|
| takes6 (706 s) | 51 → **16** | 42.3% → **47.0%** | 46.5% → 38.0% | 0.45 → 0.45 |
| takes7 (159 s) | 21 → **0** | 43.4% → **49.4%** | 59.1% → 10.0% | 0.36 → 0.36 |
| takes5 (945 s) | 29 → **6** | 25.0% → 23.8% | 13.7% → 9.5% | 0.34 → 0.33 |

**Honest reading of that table, including the part that is a cost.** Re-reads
fall 69-100%. Tempo accuracy improves on the two 08-27 takes and is 1.2 points
*worse* on takes5. Median confidence — what the mind actually gates on — is
unchanged everywhere to two decimals. The locked fraction falls, on takes7 from
59.1% to 10.0%, and that is a real cost and not an artifact: holding one grid
whose predicted beats miss costs hit rate, where chasing the best-scoring alias
every half second inflates it. It is the same inflation amendment 26 documented
in the trap collapse ("a coarser grid predicts fewer beats, all of which land
on beats the human played, so confidence ROSE across the collapse"). A lock
whose grid moved 21 times in 159 seconds was not a lock; the honest number is
the lower one. It is recorded here as a cost anyway, because a future reader
comparing locked fractions across amendments deserves to know this one moved.

**Reporting.** `tempo_octave_changes` keeps its amendment-26 meaning exactly —
whole octaves only — because `mind.py`'s ritard guard is written against it and
a 4/3 re-read is already disqualified there by `ritard_step_max_pct` (10%). The
wider count is the new, additive `BeatState.tempo_grid_changes`, and it is what
`sim` and `plume listen` now key their forensic lines on: a log keyed on the
octave counter showed 2 of the 54 re-reads `takes6/` actually contained.
Non-octave commits are announced as `tempo_grid_change`; octaves keep
`tempo_octave_change`. Both event dicts gained `octave` and `ratio`, so a
decisions log can be counted by alias family without re-deriving it.

**Scope this does not reach.** It stops the metre changing under a settled
take. It does not make the first reading right, and cannot: measured
independently of the tracker, `takes6/human.wav` has an ACF salience of
0.06-0.19 per window against 0.4+ for a clear beat, and single 30 s windows
read 55, 58, 122, 134, 63, 160, 165 against a true 111.7. A 12-second window
genuinely cannot separate 111 from its 4/3 and 3/2 relatives. Longer-horizon
integration — commit and hold across a phrase rather than re-deciding every
0.5 s — is the separate defect, and the one worth attacking next.

**A consequence nobody was looking for: this amendment removed the corpus
failure that justified the DEFAULT front end.** `test_the_mode_flag_is_load_bearing_in_both_directions`
pinned acoustic mode ending `lofi80_32` on 92.98 bpm against a true 80 at
-20 dBFS — 1.162, a 4/3 alias taken over a thinning outro on the fast path.
Amendment 48 fixes it; both modes now end at 80.36, and at full gain the old
routing was worse still and unmeasured (attack ended that render on **181.5**).
Re-running the A/B as that test instructs leaves acoustic mode with exactly two
measured deficits — `trap140` locking at 6.7 s against 3.8 s, and ~0.05 of
median confidence on `lofi80` — while it now wins the quiet cases outright
(100% locked at ~15 s against 48-59% at ~35 s). The test has been re-pointed at
the late lock, which is the honest remaining pin. **v2.7 does not change the
default**: that is a decision about every test in the suite and both flagship
demos, and this amendment records the measurement rather than making the call.
The 2026-08-19 A/B table in `beat.py` now carries a dated correction so the
92.98 is not inherited as still true.

**A second knock-on, same shape.** `test_realtime.py`'s
`TestTheMaskDoesNotStampTheMachinesTempoOnTheHuman` carried a sentinel proving
the v1.7 transient gate is load-bearing: with the gate off, the self-mask's
periodic blanking taught the tracker plume's own 96.2 bpm on acoustic material
that never contained it. On the `fingerpick` fixture that sentinel now *passes
where it must fail* — the reading is 84.22 against a true 84, because 96.2/84
is 1.145 and amendment 48 will not let a settled grid take a non-octave alias
in one second. The blanking is undiminished: 16% of frames ungated against 1.7%
gated. So the sentinel was re-derived at the layer the gate actually acts on —
`test_the_gate_is_what_stops_the_bus_being_blanked` pins the frame-level
fraction on both fixtures, where no downstream guard can rescue it — and the
end-to-end half keeps `strum`, which still reads 96.15 against a true 72
(96.2/72 = 4/3) because that fixture does not settle before the holes
accumulate. If amendment 48 is reverted, `fingerpick` goes back in that list
and fails again.
