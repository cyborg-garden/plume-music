# SPEC — plume v2 "who's holding time" (lead mode & role negotiation)

v0 = plume composes. v1 = plume follows. **v2 = plume can lead, and the two of
you can trade who's leading, mid-song, without stopping.**

The v1 sideman assumes the human owns the clock and the machine earns its way
in. That's one chair. The other chair is just as real: the machine starts —
from a prompt, or from a session you already wrote — and *you* join *it*. And
the moment you join, it must stop being a backing track and start being a
player: following your energy, waiting for you at the turnaround, conceding
tempo when you push, changing what it plays because of what you did.

**Contract status: FROZEN for the v2 build.** Ground rules from SPEC.md and
SPEC-V1.md apply (48kHz float32, determinism, no per-sample loops, spec
deviations documented not silent). Everything in v0/v1 is reused; v2 adds
`src/plume/roles.py`, extends `mind.py`, `sim.py`, `realtime.py`, `cli.py`.

## 0. The core idea: authority is a dial, not a mode

Two separable questions, each a float in [0, 1]:

- **`clock_authority`** — who owns *time*. `0.0` = the human's beat is truth,
  machine phase-corrects hard toward them (v1 sideman). `1.0` = the machine's
  clock free-runs and never phase-corrects (Band-in-a-Box's virtue: an
  unshakeable groove is a feature; a player can trust it and lean on it).
  In between = **mutual entrainment**, which is what human bands actually do:
  both parties bend a little. This is the α from v1, generalized and made
  bidirectional.
- **`form_authority`** — who owns *the form*: what section we're in, when it
  changes, what the harmony is. `1.0` = the machine plays its composed
  arrangement and the human fits in. `0.0` = the machine takes its harmony and
  sections from what it hears (v1).

v1 sideman = `(clock 0.0, form 0.0)`. Backing track = `(1.0, 1.0)` with no
listening. Everything interesting is in the middle, and **the dials move during
the session** — that movement is the negotiation.

## 1. roles.py — the role state machine

```python
ROLES = ("lead", "follow", "trade", "solo_support")
@dataclass class RoleParams:
    role: str = "follow"                  # starting role
    clock_authority: float | None = None  # None -> role default (lead 0.9, follow 0.15, trade 0.5, solo_support 0.6)
    form_authority: float | None = None   # None -> role default (lead 1.0, follow 0.0, trade 0.5, solo_support 0.85)

    # --- mutual entrainment (the heart of lead mode) ---
    concede_ms: float = 25.0        # sustained human offset (ms, |mean| over concede_bars) that starts a concession
    concede_bars: int = 2           # bars of consistent push/drag before conceding
    concede_rate: float = 0.25      # fraction of the human's deviation absorbed per concession event
    concede_max_pct: float = 8.0    # never drift more than this % from the composed tempo
    # --- role handoff ---
    handoff_silence_bars: int = 8   # human lays out this long while machine follows -> machine takes lead
    handoff_density: float = 2.0    # human onsets/beat above this while machine leads -> machine yields toward follow
    handoff_hysteresis_bars: int = 4  # min bars between role changes (no flapping)
    trade_bars: int = 4             # trading unit ("fours")
    # --- harmonic meeting ---
    conform_semitones: int = 2      # if the human's key sits within this distance of ours, transpose to meet them
    conform_bars: int = 4           # bars of consistent out-of-key evidence before transposing
    conform_enabled: bool = True

@dataclass(frozen=True) class RoleState:
    t_s: float; bar: int
    role: str
    clock_authority: float; form_authority: float
    tempo_offset_pct: float          # accumulated concession from the composed tempo
    transpose_semitones: int         # accumulated harmonic conformance
    reason: str                      # human-readable, contract (same discipline as BarPlan.reason)

class RoleNegotiator:
    def __init__(self, params: RoleParams): ...
    def update(self, *, bar: int, beat: BeatState, harmony: HarmonyState,
               human: BarFeatures | None, machine_grid_t_s: float) -> RoleState
```

Behaviors (all evaluated at bar boundaries, all thresholds above):

1. **Concede tempo** (lead mode's soul). Measure the mean signed offset between
   the human's detected beats and the machine's own grid over `concede_bars`.
   If |mean| > `concede_ms` consistently, absorb `concede_rate` of it into the
   machine's tempo — a *period* change, gradual, capped at `concede_max_pct`.
   The human pushes, the machine picks up; the human lays back, it settles.
   Never jumps: a concession is a ramp applied over the following bar.
   Log it (`"conceding: human +31ms for 2 bars -> 120.0 -> 121.9 bpm"`).
2. **Yield the lead.** While leading, if the human's density exceeds
   `handoff_density` for `handoff_hysteresis_bars`, slide `clock_authority` and
   `form_authority` down (toward follow) — they've taken over; get out of the way.
3. **Take the lead.** While following, if the human is silent for
   `handoff_silence_bars` and the machine still has a groove, slide authority up
   and keep the form moving — someone has to hold the floor.
4. **Trade.** `role="trade"`: alternate authority every `trade_bars` at phrase
   boundaries, with the v1 human-wins collision rule (if the human plays over
   the machine's bars, the machine backs off within a beat).
5. **Meet them harmonically.** If the human's key belief is consistently a
   different key within `conform_semitones` of the machine's for `conform_bars`
   (they capo'd, they're in F# and we're in F), transpose the machine's
   arrangement to their key rather than fighting. Beyond that distance, hold and
   log (they're playing something else; conforming would be guessing).

Pure/deterministic. No rng, no I/O.

### v2.0 amendments — `roles.py` (section 1)

Written while building `src/plume/roles.py` and `tests/test_roles.py`. Every
one was forced by a **measurement on real material** — plume's own v0 renders
(trap 140 / D minor, lo-fi 80 / C major, at 0 dB and −20 dB) driven through the
real `FeatureExtractor` / `BeatTracker` / `HarmonyTracker` in a closed loop
where the machine's grid advances at the tempo the negotiator just chose. Each
has a regression test and a non-vacuity test beside it.

**A1 — sign convention.** `+` offset = the human is **ahead** (rushing), which
is the only reading consistent with the spec's own worked example ("human
+31ms … 120.0 → 121.9 bpm"). The arithmetic is fixed by that example too: the
deviation is a fraction of the machine's **beat** period, and `concede_rate` of
that fraction is applied as a tempo scale (0.031/0.5 = 6.2%, ×0.25 = 1.55% →
121.86 → "121.9"). `tests/test_roles.py` asserts that log line verbatim.

**A2 — additive fields.** `RoleParams` gains five fields (appended, so the
spec's field order is intact); `RoleState` gains `tempo_target_pct` (a renderer
cannot ramp toward a number it cannot see), `bpm`, and `events` (machine-readable
tags, so a behavior test never has to grep prose). `RoleNegotiator.__init__`
gains keyword-only `composed_bpm` / `composed_key` — session context, without
which the spec's own reason string and conform rule cannot be evaluated.
`update()` gains keyword-only `human_offset_ms`. All defaulted; every
spec-shaped construction still works.

**A3 — the derived offset is not a 25 ms instrument, and this is the big one.**
Differencing `BeatState.next_beat_t_s` against the machine's grid produces
samples out to **±210 ms** on real audio — half a beat — because a swung groove
biases the prediction and a sub-percent tempo mismatch walks the phase across
the beat. Fed to the concede rule unfiltered it drove a single **−8%**
concession straight into the cap **in the wrong direction**. Three guards, all
derived from the spec's own words rather than added to them:

- `concede_plausible_frac` (0.2) — an offset past a fifth of a beat is a phase
  estimate that missed, not a human pushing. Expressive timing is tens of ms.
- `concede_spread_ms` (25) — "consistently" must mean the bars *agree*, not
  merely that they fell on the same side. Two bars 100 ms apart average to a
  number neither bar measured.
- a single concession is clamped to one bar's ramp (2%). The spec says a
  concession is "a ramp applied over the **following bar**" and §7 bounds a bar
  at 2%; a target needing three bars of ramping contradicts both.

With all three, the worst drift on a matched-tempo human across the four takes
falls to 3.4% (most do not move at all); without them the same takes saturate
the 8% cap. Inertness is the safe failure and it is the one these gates buy —
**a caller that wants the machine to actually negotiate tempo should pass
`human_offset_ms`.** `roles.onset_phase_offset_ms()` is provided for it — a
circular mean of onset phase against the machine's sixteenth grid, which
recovers a known 30 ms shift to within a few ms at both gains where the derived
path barely separates the conditions at all.

**A4 — known limit: a pure tempo mismatch is the tracker's job.** A human at a
different *tempo* produces a phase error that ramps (25 ms/bar at 1.25%), and a
ramp fails the spread test. Admitting 2-sample ramps was measured to walk a
matched take to 6.0% drift while buying nothing stable, because a phase→period
law is an integrator and against a partner who does not also bend it oscillates
rather than locks. A tempo difference belongs to `BeatState.bpm`. What this rule
is for is the thing the tracker *cannot* report: two players at the same tempo,
one of them pushing.

**A5 — yielding needs a relative gate.** Real programmed material measures a
steady **5.0 onsets/beat**, 2.5× `handoff_density`, so the bare absolute rule
handed the lead away at the first hysteresis window on anything with hats.
`handoff_density_ratio` (1.5) adds "…and busier than *they* have been", with the
baseline frozen during a burst. The `takes_over` fixture (density ramps ~1 → 4)
passes both gates, so behavior test 6 is unaffected. Hats are not a takeover.

**A6 — conform on pitch *collections*, not tonics.** A chroma histogram cannot
distinguish a key from its relative (the C-major lo-fi render is reported as A
minor on every frame, at both gains). A tonic-distance rule therefore logs a
spurious hold against a human in our own key, and — worse — can transpose to the
wrong key (F♯ minor vs a human in F minor reported as A♭ major reads as 2 st and
would conform). `roles.collection_root()` is stable under exactly that
ambiguity. `conform_key_conf` gates on key confidence.

**A7 — small ones.** Concessions require `clock_authority ≥ 0.5` (below it v1's
phase correction is already doing this job, and two corrections fight).
Unusable evidence *clears* the window rather than skipping it. In `trade`,
authority is set rather than slid (a 3-bar slew would smear a 4-bar trade) and
a collision is a one-bar backoff, not a turn change; the "within a beat"
reaction is the renderer's, since `RoleState` is a per-bar contract.

**Cost on the live path:** 135,000× real time (34 bars over 56.4 s of audio in
0.42 ms). The beat tracker's ~180× headroom is untouched.

### v2.1 amendments — `roles.py` (section 1), the fixer pass

Four defects in the v2.0 concede rule, each found by measurement and each with
a regression test beside it. The headline: **v2.0's concede rule was a
phase→period law with no phase-return term, and such a law diverges by
construction.** Measured in the rendered audio (machine-bus low-band bar
tracker, agreeing with the log to ≤50 ms): on the 32-bar `rushing` joiner the
machine ended +3.26% with its downbeat **2.5 beats** ahead of the human's; on
`dragging`, 3.7 beats behind. Nothing in the decision log said so.

**A8 — the plausibility bound was in the wrong unit, and every wrapped sample
walked through it.** `onset_phase_offset_ms` is a circular mean over ONE
SUBDIVISION, so its unambiguous range is ±half a sixteenth (±62.5 ms at 120
bpm) — while `concede_plausible_frac` is a fifth of a BEAT (±100 ms). Measured
at 120 bpm: a human 70 ms ahead reads −55 ms, 90 ms ahead reads −35 ms, 150 ms
ahead reads +25 ms — sign reversed, gate passed. End to end that made the
machine's response *direction* a coin flip once the human's tempo differed by
more than ~5% (4 of 12 `simulate_lead` runs went the wrong way; the same 132
bpm human gave 116.5 bpm at one seed and 127.0 at another). `update()` gains
keyword-only **`human_offset_unit_s`**; when the caller says what subdivision
it measured against, the bound tightens to `concede_plausible_unit_frac` (0.4)
of it. A caller that omits it still works, gated by the coarser bound alone.

**A9 — unwrap against a PREDICTION, not against the last reading.** Inside the
bound a reading is unambiguous, but the phase it is unambiguous *about* runs to
several subdivisions once the clocks differ; and unwrapping to the nearest
branch of the previous reading assumes the phase barely moved, which is exactly
false after a concession (a 2% concession at 140 bpm opens 34 ms per bar
against a 53 ms half-range). Measured: bar 8 of the trap140 rush take, true
phase ≈ −2 ms, read −22.9 ms, unwrapped to **+82 ms** on a one-millisecond coin
flip, and the machine read its own divergence as a fresh push and ran to
+5.98%. The prediction is `previous + (machine_bar − human_bar)`, using
`BeatState.bpm` only to say *where to look* — guarded on `locked` and on the
ratio, so an octave relative cannot mislead it.

**A10 — the phase-return term (`_negotiate_drift`).** When the gap is OPENING
at a steady rate, the two clocks are at different tempi and the tempo
difference IS that rate; `_DRIFT_RATE` (0.8) of it is absorbed, same ramp bound
and same cap as a push. Narrow on purpose (amendment A4 measured what admitting
bare ramps costs): `_DRIFT_DIFFS` (4) consecutive differences sharing a sign,
an end-to-end rate past `concede_drift_ms` (12 ms/bar), and the gap opening
rather than closing. The rate is measured END TO END rather than as a mean of
differences and is NOT spread-tested — a push must be steady to be a push, a
drift only has to be there. (Measured on `dragging`: the window read
+20/+110/+191/+211/+301, unmistakably a ramp, but the per-bar differences were
+90/+81/+20/+90 — a 70 ms spread against a 25 ms tolerance — so the spread test
threw away every window and the machine sat at −3.47% for eighteen bars.)

**A11 — where the two laws divide.** The push law is suppressed only when the
phase is BOTH still moving (past `concede_drift_ms`) AND further out than
`concede_plausible_frac` of a beat. Both halves are load-bearing and each was
measured: suppressing on movement alone is wrong because a human who starts
pushing IS an opening gap (it rang the mutual-entrainment loop at 87 ms where
it settles at 30); suppressing on magnitude alone is wrong because a large
offset that is NOT moving is a standing offset the drift law cannot see, and
leaving it to nobody deadlocked that loop at a permanent −85 ms.

**A12 — swing.** `onset_phase_offset_ms` gains `swing`. A swung machine
measuring a swung human against a STRAIGHT grid reads the groove they share as
a lay-back: a v0 lo-fi render (swing 0.57) against a machine playing the same
lo-fi style at the same tempo read a standing −30 ms and conceded to it. A
swung sixteenth is displaced 35 ms at 120 bpm, larger than `concede_ms`, so the
whole rule fired on the groove. At `swing = 0.5` the code path is arithmetically
the identity.

**A13 — the lock gate applies to the DERIVED estimator only.** `grid_offset_ms`
reads `BeatState`, so there the gate is load-bearing; a measured
`human_offset_ms` never touches `BeatState` at all. This is not a convenience:
re-run with the lock applied to both paths and all four shifted real-material
takes end at exactly +0.00% with zero concessions and zero separation between a
human pushed 30 ms early and one dragged 35 ms late. `concede_min_conf` still
gates both paths.

**MEASURED, before → after** (same corpus, same seeds):

| take | v2.0 | v2.1 |
|---|---|---|
| `rushing` joiner, final gap | 2.5 beats | 0.85 beats |
| `dragging` joiner, final gap | 3.7 beats | 0.50 beats |
| real trap140 rush/drag, worst gap | 3.3 / 5.1 beats | 1.6 beats |
| synthetic tape (48 bars, cannot bend) | 1330 ms | 130 ms |
| mutual entrainment, peak residual | 40 ms | 30 ms |
| real-material takes at 0 dB vs −20 dB | different | **bit-identical** |

**KNOWN LIMIT, stated rather than hidden.** Closing the phase a concession
itself opened takes a counter-excursion, so against a partner who cannot bend
(a fixed recording) the tempo curve is a bounded hunt rather than a settle:
`rushing` goes +1.5% then −3.9%, inside the cap, ramped, ending 0.85 beats from
the human. The direction of the FIRST move is the part that is a claim about
the human, and that is what section 7 now asserts. A phase-DEBT formulation —
absorb exactly `concede_rate × offset` of phase and then return — would settle
instead of hunt, and is the obvious next step; it is not in this pass because
it also drives the standing tempo offset to zero, which section 7 behaviors 2/3
and the real-material sign guards both currently read as the evidence that the
machine met the human at all.

### v2.2 amendments — `roles.py` (section 1), the tempo-authority pass

Governed by `docs/decisions/2026-08-15-tempo-authority.md`, which **supersedes
this section's framing of conceding as "lead mode's soul"**. Behavior 1 above
is unchanged as code and is no longer the default.

**A14 — tempo authority is explicit, and lead mode HOLDS.**

```python
tempo_policy: str | None = None   # None -> the role's default; "hold" | "meet" | "chase"
```

| policy | meaning | roles that default to it |
|---|---|---|
| `hold` | the machine is the reference; never concede to clock-level DRIFT | `lead` |
| `meet` | behavior 1 above, with every v2.0/v2.1 guard, unchanged | `trade`, `solo_support` |
| `chase` | the human owns the clock (v1 follow); the negotiator writes no tempo | `follow` |

Settable per session: `--roles tempo_policy=meet` (and `=none` for the role's
default). The field is `str | None` rather than the literal `str = "hold"` the
decision doc sketches, for the reason `clock_authority`/`form_authority` have
been `float | None` since v2.0: `RoleParams`' own `role` default is `follow`,
and a follower that holds its own clock is not a follower. No CLI change was
needed — `_role_rows`/`_coerce_role_value` derive the type from the annotation,
so `str?` already parses.

**Why.** SPEC-V2 made mutual entrainment the default and then measured the
consequence: **conceding to a partner who cannot bend is divergent by
construction**, because the machine's new tempo becomes the thing the human is
now ahead of. v2.0 met a human sitting 30 ms in front and ended the take **2.5
beats ahead of them**, with a decision log showing a tidy concession and a
settled tempo. v2.1 bounded that (a hunt inside the cap) but could not remove
it, and said so in the KNOWN LIMIT above. RESEARCH.md had already recorded the
answer: *"an unshakeable clock is a feature, not a limitation; musicians adapt
to a trustworthy machine."*

**A15 — holding is not deafness. Three things `hold` does NOT mean.**

1. **The measurement still happens, and is now PUBLISHED.** `RoleState` gains
   `human_offset_ms: float | None` (the signed offset, positive = the human is
   ahead, `None` = no usable evidence — never conflated with 0.0) and
   `tempo_policy: str`. Every bar with a usable reading emits the event
   `tempo_hold` and the reason `holding: human +28ms (policy=hold)`. It costs
   nothing (the concede rule computed it anyway), a silent hold is
   indistinguishable from a broken measurement, and it is exactly the signal
   the roadmap's v2.5 "mirror" wants. A machine that follows your rushing
   destroys its own practice value.
2. **Form-level tempo INTENT still lands.** Drift and a deliberate ritardando
   are different subsystems and the split is architectural, not conditional:
   `roles.py` only ever writes `tempo_offset_pct`, while cue tempo is a
   separate multiplier applied downstream (`sim._LeadCues.bar_ratio()`, folded
   in as `m_bpm = rstate.bpm * ratio`) and the ending-ritardando detector is
   `mind.Mind._form_track_tempo`, which drives the FORM. Verified end to end
   rather than assumed — `ritard`/`faster`/`slower`/`half time`/`double time`
   produce **bar-for-bar identical** tempo curves under `hold` and under
   `meet`, and a detected ritardando still ends the tune under `hold`. A
   structural test also asserts `roles.py` never references a cue, so a future
   refactor cannot re-couple them.
3. **The other four behaviors are untouched.** Yield, take-lead, trade and
   conform all still fire under `hold`. Only DRIFT is gated.

**MEASURED, v2.1 → v2.2** (lead mode's default, same corpus, same seeds):

| take | v2.1 (`meet`) | v2.2 (`hold`) |
|---|---|---|
| synthetic tape, 48 bars (cannot bend), worst `tempo_offset_pct` | 1.70% | **0.00%** |
| ...and its peak residual gap after bar 8 | 129.7 ms, hunting | 30.0 ms, constant — the human's own standing offset, unmoved |
| real corpus (trap140 / lofi80, 0 and −20 dB, and 30 ms-rushed copies — 8 takes), worst | 3.85% (`trap140_rush`, both gains) | **0.00% on all 8** |
| `simulate_lead`, real 30 ms-rushing joiner, 24 bars | tempo moves | every bar at 90.0 bpm; bar *k* lands at exactly *k*·240/bpm s |
| explicit tempo cue (`ritard`/`faster`/`slower`/`half`/`double`) under `hold` vs `meet` | — | identical, bar for bar |

The v2.1 KNOWN LIMIT does not arise: under `hold` a fixed recording produces
**exactly** 0.0% — asserted as an equality, not a tolerance, because nothing
writes the field at all.

**A16 — an unset policy follows the chair; an explicit one is a pin.** The
policy is resolved per bar against the CURRENT role, so a lead that yields
stops holding a clock it no longer owns and takes the hold back with the lead.
MEASURED on plume's own lo-fi 80 render, which really does ramp in density and
really does make the machine yield: the take reads `hold` while leading and
`chase` after. An explicit `tempo_policy=` never moves — same distinction the
authority dials draw with `_overrides_live`.

**Callers must pass nothing new to keep working, but three things follow.**

1. `RoleParams(role="lead")` is now hold. Any caller or test that wants the
   old behavior passes `tempo_policy="meet"` — `tests/test_roles.py` does it
   through one `LeadMeet()` helper, which is the pattern to copy.
2. `sim.simulate_lead`'s `role_states` rows and `realtime`'s role line should
   surface `human_offset_ms` and `tempo_policy` alongside `tempo_offset_pct`.
   The NDJSON `_lead_line` already logs the measured offset; the practice
   ("mirror") feedback is only reachable from the log once the structured row
   carries it too.
3. **Section 7's behaviors 2 and 3 ("the human is met", both directions) are
   now statements about `meet`, not about lead mode's default.** They stay
   exactly as specified and their tests must name the policy. Section 7 wants
   a fourth behavior beside them — *the human is HELD against* — asserting on
   the same joiners that the machine's rendered grid is a perfect arithmetic
   sequence and the decision log says `holding: human ±Nms` every bar.

## 2. Adaptive arrangement navigation (mind.py extension)

In lead mode the machine owns a composed arrangement (a v0 `Session`), but it
must not march through it like a tape. `Mind` gains an arrangement cursor:

```python
@dataclass(frozen=True) class FormPlan:
    bar: int
    section_idx: int                # index into the flattened movement/section list
    action: str                     # "advance" | "hold" | "extend" | "jump" | "end"
    bars_left_in_section: int
    reason: str
class Mind:
    def plan_form(self, role: RoleState, beat: BeatState, harmony: HarmonyState,
                  human: BarFeatures | None, session: Session, cue: "Cue | None") -> FormPlan
```

Rules (each testable from the decision log):
- **Wait at the turnaround.** At a section boundary, if the human is mid-phrase
  (energy rising / no resolution detected), `extend` the current section by a
  phrase unit rather than cutting them off — the "one more chorus" gesture.
  Bounded by `max_extend_bars` (default 16).
- **Energy chases the human.** Sustained human RMS above the section's energy
  for 2+ bars → `jump` to the next higher-energy section at the boundary;
  sustained below → drop to a lower-energy one. The composed arrangement becomes
  a *map*, not a script.
- **Announce it.** Any machine-initiated section change is preceded by a fill or
  riser in the last bar (reciprocal cueing — the human gets to anticipate too).
- **End together.** Ritardando detection (v1's ending suite) overrides the form:
  ritard → ritard along and prepare the final hit; human plays the head again →
  final-chorus mode.

### v2 amendments — section 2 (form navigation), builder C

Implemented in `mind.py` as an additive block (`FormParams`, `FormPlan`,
`Mind.plan_form` + `_form_*` state). `plan_bar`, `MindParams` and the v1
behavior matrix are untouched. Tests: `tests/test_form.py` (101 synthetic +
28 real-material cases).

1. **`FormPlan` gains three defaulted fields**: `announce: bool`,
   `applied_cue: str | None`, `dropped_cue: str | None`. The frozen five keep
   their names, types and order, so every spec-shaped construction still
   works. Reason: the spec's own rules are otherwise unobservable — "announce
   it" is a claim about the bar BEFORE a change, and a cue that applied (or
   expired unapplied) has to reach the decision log as data, not as prose
   inside `reason`.
2. **Thresholds live in a new `FormParams`, not `MindParams`.** `MindParams`'
   field set and defaults are asserted by `tests/test_mind.py` and printed by
   `plume mind`; `max_extend_bars=16` is there with the rest.
   `Mind.__init__` gains an optional second argument (`Mind(params)` still
   works positionally).
3. **`role` is duck-typed and may be `None`.** `roles.py` is a sibling module
   built in parallel, so `mind.py` does not import it — `plan_form` reads
   `role.form_authority` defensively. An unreadable/absent role means FULL
   form authority: being asked to navigate a form implies owning one.
   Below `form_authority_min` (0.5) the machine initiates nothing —
   extensions still happen (deferring to the human is not an initiation) and
   cues still land (the third chair is explicit), but energy jumps do not.
4. **"Energy chases the human" is measured RELATIVELY, not against
   `SectionSpec.energy`.** The spec says "sustained human RMS above the
   section's energy", but those are not comparable quantities: section energy
   is 0..1 and bar RMS is dBFS, which moves 1:1 with input gain. Any fixed
   mapping is v1's bug again — a rule that fires constantly on a hot input and
   never on a quiet one. Implemented as a fast EMA of the human's own level
   against a slow one standing in for their session baseline; the delta is
   gain-invariant by construction, and `tests/test_form.py` asserts the
   property exactly (shift every real bar's level by ±30 dB, get a
   byte-identical plan sequence).
5. **Ritardando detection is new here; "v1's ending suite" does not exist.**
   There is no ritard detector anywhere in v1 (`listen/`, `mind.py`). The
   minimal version: consecutive bar-boundary tempo estimates, each strictly
   lower than the last, totalling `ritard_pct` (4%), only while the tracker is
   locked and the human is playing. **Two guards were added after real
   material broke the naive rule**: a quiet synthwave render locks at 199.5
   bpm, commits an octave re-interpretation to 99.9, and 199.5 → 99.95 →
   99.87 read as a "-50% ritardando" — the machine ended the tune on bar 2 and
   sat there for the remaining 24 bars. So (a) any increment of
   `BeatState.tempo_octave_changes` clears the history, and (b) a single-bar
   drop over `ritard_step_max_pct` (10%) is disqualifying. The spec's other
   ending trigger — "the human plays the head again" — is **NOT implemented**:
   it needs melodic matching against the composed theme, which nothing in
   v0/v1 provides.
6. **Action vocabulary, pinned.** `hold` = stay inside the current section
   (including the first bar of a take); `advance` = the composed next section;
   `jump` = anywhere else in the map; `extend` = the turnaround wait;
   `end` = the ending suite owns the form (sticky). `bars_left_in_section`
   counts bars AFTER the upcoming one, so `0` marks the last bar of the span
   and is exactly when `announce` is set.
7. **`announce` marks the last bar of every section span**, which is also
   where v0's `SectionSpec.fill` already puts a fill. The testable invariant
   is therefore "every machine-initiated section change is immediately
   preceded by a plan with `announce=True`". Cue-driven changes are exempt:
   rule 3 says *machine*-initiated, and a cue arrives with no notice by
   definition.
8. **`max_extend_bars` bounds ONE section span, not the song.** "One more
   chorus" is a gesture you can also make in the next section; a song-long
   budget would mean a long take's third turnaround can never be waited at.
9. **Past the last section the cursor wraps** (`loop_form`, default on) —
   the machine holds the floor rather than stopping. With `loop_form=False`
   the end of the map is an `end`.
10. **Cost, measured**: ~5 µs/bar, i.e. ~390,000x real time for the cursor
    alone. v1's beat tracker (~100-180x real time on these renders) is
    unaffected; `tests/test_form.py` asserts >2000x so the headroom that makes
    live jam possible cannot silently regress.

## 3. Cues — the third chair, live

```python
@dataclass(frozen=True) class Cue:
    text: str                       # "darker", "drop to half time", "go to the chorus", "take it out"
    at: str = "next-phrase"         # "now" | "next-bar" | "next-phrase" | "next-section"
    ttl_bars: int = 16
def parse_cue(text: str) -> dict    # -> {"kind": "energy"|"section"|"texture"|"tempo"|"end"|"style", "value": ...}
```
Cues are queued and applied at their boundary (never mid-phrase), and land in
the decision log with their effect. Rule-based parsing in v2 (an LLM can also
just write the resulting `set`/`arrange` calls directly — cues are the *terse*
path, not the only one). This is how the agent chair changes the music while
the human is still playing.

### v2 amendments — section 3 (cues), builder C

Implemented in `mind.py` (`Cue`, `parse_cue`, `CUE_KINDS`, `CUE_BOUNDARIES`,
queueing inside `Mind.plan_form`, introspection via `Mind.pending_cues()`).

1. **`parse_cue` returns a seventh kind, `"unknown"`, and NEVER raises** — not
   on `None`, a non-string, empty text, emoji, or a 5000-character string. A
   cue arriving over a socket mid-song must not be able to take the band down,
   and guessing a kind for text we did not understand is the worse failure: it
   would change the music for no reason the player could trace. Junk is
   applied as an explicit no-op and logged as "not understood".
2. **Return shape**: `{"kind", "value", "text"}` plus kind-specific extras
   (`delta` for energy, `ratio` for tempo). `value` is always a short
   canonical token (`"up"`/`"down"`, `"half"`/`"double"`/`"faster"`/
   `"slower"`/`"ritard"`, a section name or `"next"`/`"prev"`, `"darker"`/
   `"brighter"`/`"sparser"`/…, a style name, `"out"`).
3. **Match order is fixed**: end → tempo → section → style → texture → energy.
   "drop to half time" is a tempo cue, not a jump to the drop nor an energy
   drop; "last chorus" is an ending, not a jump. Section names that are also
   ordinary words (`drop`, `break`, `build`, `head`, `hook`) require a
   directional phrase ("go to the…", "back to the…") to read as a section cue.
4. **`at` is honoured as written; "never mid-phrase" describes the default.**
   `next-phrase` (the default) waits for the next bar whose position inside
   the section is a multiple of `phrase_bars`; `now` is an explicit override by
   the third chair and applies immediately. Boundaries are evaluated against
   the *resolved* position of the upcoming bar, so a cue that would land in a
   section the cursor is leaving lands in the one it arrives at instead. An
   unrecognized `at` is treated as `next-phrase` rather than raising.
5. **TTL**: a cue that has not applied `ttl_bars` bars after it was issued is
   dropped and reported in `dropped_cue` + `reason`. Bars are counted by
   `plan_form` calls, not by `BeatState.bar`, so an unlocked tracker cannot
   freeze the clock a cue is expiring against.
6. **Several cues can come due at one boundary.** Their texts are joined with
   `"; "` in `applied_cue` and each lands in `reason` with its own effect,
   rather than silently dropping all but one. Once the form is ending the
   queue is cleared — an ending outranks queued instructions too.
7. **Only `section`, `energy` and `end` cues move the cursor.** `tempo`,
   `texture` and `style` cues are applied and reported (they are the
   renderer's and the role negotiator's business) with the form unchanged; a
   section cue naming a section that is not in the map is ignored, with the
   reason saying so.

### v2.1 amendments — section 3 (cues), the fixer pass

**C1 — the parser's vocabulary and the composer's had an EMPTY intersection,
so the whole `section` kind was dead on real material.** `parse_cue` knew
nineteen generic song-section names; v0 `compose` only ever emits
`settle-a` / `drift-b-outro` / `establish-a` / `peak-b` / `neon-rise-a` /
`apex-b`. Measured: `plume cue`'s own help example, `go to the chorus`, logged
"no such section in the map — ignored"; naming the session's real section,
`go to drift-a`, logged "not understood — no change". Two fixes, both needed:
a **directional phrase now CAPTURES whatever follows it** as a section name
("go to the" is an unambiguous statement that a section is being named, and
what the user calls their sections is their business), and the handful of words
that name a *position in a song* rather than a section of this one resolve by
**role in the arc** — intro→first, outro→last, chorus/hook/drop→hottest,
breakdown→coldest, verse→coldest-inner. The arc mapping is a guess, so it is a
last resort and the log names the section it actually chose. A name that is
genuinely not in the map is still refused, out loud.

**C2 — "applied" was a lie for every kind except `energy`.** Measured in a
machine solo with full authority (role=lead, clock 0.90, form 1.00 — nothing to
yield to): `--cue "8.0:faster@next-bar"` logged `(tempo:faster): applied`, the
bpm stayed at 96.0 on every later bar, and the WAV was **byte-identical** to
the no-cue run. Same for `slower`, `double time`, `half time`, `darker`,
`brighter`, `more space`. The worst was the help's own example: `take it out`
logged "ending the tune here" at bar 5 and then played all sixteen bars to the
same 40.39 s. The decision log is the only instrument a cold user has, and it
was asserting actions that did not occur — the exact failure mode that hid v1's
four bugs. So:

- `Mind.plan_form` no longer says "applied" for kinds it does not own. It says
  **"understood, form unchanged — for the renderer"**, which is a claim it can
  actually make.
- The renderer (`sim._LeadCues`) now implements them: `faster`/`slower` are
  ±6% on the machine's bpm, `half`/`double` are ×0.5/×2, `ritard` compounds 4%
  a bar to a 0.55× floor, `sparser`/`busier` scale the planner's density,
  `darker`/`brighter` are a first-order tone tilt on the machine bus from the
  cue's bar (one continuous filter over the remaining span, crossfaded over
  50 ms, so it is a move rather than a click).
- Everything else — `style`, `warmer`, `dry`, `wet` — reports
  **"understood but NOT applied"** with the reason, in the same bar, and is
  byte-identical by construction.
- Every effect lands in the NDJSON as `cue_effects` (a list, per bar) and the
  manifest gains `cues_applied` / `cues_dropped` / `cues_never_applied` /
  `cues_undelivered`, so `--json` answers "did my cue land?" without anyone
  reading the log.

`tests/test_simlead.py::test_every_cue_kind_either_changes_the_audio_or_says_it_did_not`
is the matrix: every listed kind must change the bytes, and the one that does
not must say so.

**C3 — the ending is real.** `FormPlan.action == "end"` now ends the take:
`_CUE_END_BARS` (2) more bars and out, with the tail. It had never been read by
any renderer.

**C4 — a cue that never reached its boundary vanished.** A `next-section` cue
in a run that sits entirely inside one section never applied, never appeared as
`dropped_cue`, and had no field in the manifest. Anything still queued when the
take ends is now reported as `{"event": "cue_never_applied"}` with the boundary
it was waiting for.

## 4. simlead — the offline twin, inverted

```python
def simulate_lead(input_wav: str | None, *, session: Session | None = None,
                  prompt: str | None = None, out_path: str, seed: int = 0,
                  role: RoleParams | None = None, mind: MindParams | None = None,
                  human_offset_s: float = 0.0, cues: list[tuple[float, Cue]] | None = None) -> SimResult
```
Machine's clock is authoritative and starts at t=0 from `session` (or from
`prompt` via v0 compose). The human WAV is mixed in starting at
`human_offset_s` — *they* are joining *us*. Everything else (features, beat,
harmony on the human's signal, role negotiation, form planning, duet rendering)
runs identically to v1 sim. Deterministic; NDJSON gains `role`, `clock_authority`,
`tempo_offset_pct`, `transpose_semitones`, and `form` records.

The human's beat tracking here must run on the *human's* signal only — in sim
that's exact (we have the isolated WAV); in `jam` it relies on v1's
schedule-aware self-masking, which matters far more in lead mode because the
machine is playing continuously. Sim is therefore the ground truth for the
behavior tests; jam inherits with a documented caveat.

### v2 amendments — section 4 (simlead), builder D

Implemented in `src/plume/sim.py` as an ADDITIVE block: `simulate` and its
byte-identical outputs are untouched (v1's suite is the contract this module
exists to protect), so lead mode gets its own entry point and its own helpers,
all under `_lead_` / `_Lead`. Tests: `tests/test_simlead.py` (39 cases, of
which 8 assert against the rendered machine bus and 7 run on real v0 material
at two gains).

1. **`Mind.plan_bar` is NOT used in lead mode; `_LeadPlanner` replaces it.**
   Everything else in the sentence above — features, beat, harmony,
   `RoleNegotiator.update`, `Mind.plan_form`, `DuetEngine.render_bar` — is
   the same code on the same per-bar cadence. `plan_bar` cannot be, because
   both of its central rules are *inverted* here. Its entry gate
   (`min_listen_bars` / `entry_conf` / a locked grid) exists so a machine that
   must earn its way in never enters on a wrong tempo hypothesis; in lead mode
   the machine **is** the tempo hypothesis, and behavior 1 wants audio in bar 0
   — including with `input_wav=None`, where the gate can never open at all.
   And its silence ladder (thin → fade → tacet) exists so the sideman stops
   when the human stops; behavior 7 wants the exact opposite. **Measured on the
   same `drops_out` WAV**: v1's `simulate` plans `entering → thinned → fading`
   and winds down; lead mode is still comping at the last bar. That contrast is
   a test (`test_the_inversion_is_real_...`), so "behavior 7 passes" is not
   just a planner that cannot emit `tacet` failing to emit it. The bar plan is
   built from the section's composed `energy` plus the role state, and the
   reactive rules that genuinely still apply — inverse density coupling, the
   dynamic arc, register avoidance — are re-expressed against the **same
   `MindParams` fields**, so there is no second set of numbers to keep in sync.
   Output is an ordinary `BarPlan` in the v1 state vocabulary.
2. **A caller must MEASURE the human's offset, so sim does.** Amendment A3
   says the machine will barely negotiate tempo without `human_offset_ms`, and
   section 6 measured that the raw detector is not the instrument either:
   `Frame.t_s` is the END of the analysis window, so it reads ~11 ms late and
   hop-quantized. A **constant** bias on a **signed** threshold is not a wash,
   it is a direction preference — it turns a +30 ms rush into +19 ms (under
   `concede_ms`, invisible) while turning a −35 ms lay-back into −46 (fires
   early). Lead mode would only ever concede *downward*. So `sim` refines every
   detection the way `tests/test_fixtures_gen.py` measures the joiners (walk
   back down a 1 ms RMS envelope to the foot of the attack) and feeds
   `roles.onset_phase_offset_ms` against the machine's sixteenth grid. Measured
   end to end: `rushing` → +1.7% final drift, `dragging` → −3.6%, `on_grid` →
   exactly zero concessions.
3. **The grid is accumulated, never recomputed.** Bar *k*'s boundary is where
   the previous *k* rendered bars ended, so `DuetEngine`'s fractional-sample
   carry is the only thing deciding bar length and the take's duration IS the
   integral of the tempo curve. Measured over a 32-bar take whose tempo moves
   three times: `Σ 240/bpm_k` lands on the rendered end of the last bar to
   **0.01 ms**, and every intermediate boundary to under 2 ms. Concessions
   therefore change *rendered bar lengths* — measured in the machine bus by
   sub-hop attack times: 2.000 s early, 1.964 s late, matching the logged
   +1.75% to within 0.6 points. With conceding disabled the same measurement
   is flat at 2.0025 s (the non-vacuity control).
4. **No machine-bus calibration.** v1 calibrates the machine bus so its RMS
   lands `gain_db` under the human's, because the sideman is joining *them*.
   Here the human is joining *us*: the machine plays at its natural level and
   `mix_human_db` is the human's fader. A mid-take recalibration would also
   step the machine's level at the join bar, which is precisely the disruption
   behavior 8 forbids. Measured: dropping a real human 20 dB moves the machine
   bus 0.2 dB.
5. **Additive keyword arguments**, all defaulted, spec field order intact:
   `style` (defaults to the session's own), `bars` (defaults to the composed
   length, extended to cover the human's take), `form: FormParams`,
   `mix_human_db`, `decisions_path` (as in v1). **`role` defaults to
   `RoleParams(role="lead")`, not `RoleParams()`** — the latter starts in
   `follow` at `clock_authority` 0.15, which contradicts this section's own
   first sentence.
6. **`SimResult` gains four defaulted lists** — `role_states`, `form_plans`,
   `bar_times_s`, `machine_bpm_by_bar` — one entry per machine bar, mutually
   index-aligned. `simulate` leaves all four empty. `entered_bar` is `0`
   whenever anything sounded: the machine is in from bar 0, which is what lead
   mode *means*. `final_key` is the MACHINE's key (transposed if it conformed)
   — the key the take is actually in; what was heard is
   `manifest["human_final_key"]`.
7. **NDJSON: `bpm`, `key` and `chord` describe the MACHINE now**, because in
   lead mode they are what is being *played* rather than what is being *heard*.
   Nothing is lost — the human's readings are on the same line under `human_*`
   (`human_bpm`, `human_key`, `human_chord`, `human_silent`, `human_density`,
   `human_rms_db`, `human_offset_ms`) — and nothing is silently reinterpreted.
   `form` is nested on the bar line rather than emitted as a separate record:
   it is a decision about the same bar at the same boundary, and splitting it
   would make every reader join two streams to answer "what did the machine do
   in bar 12".
8. **One cue is delivered per bar boundary**, earliest-issued first.
   `Mind.plan_form` takes a single cue per call and queues internally, so
   several cues issued at the same instant all still land (one boundary apart
   at worst) with their own `at`/`ttl_bars` still governing when they *apply*
   — measured: two cues issued at t=5.0 s both appear in one `applied_cue`.
   A cue whose time falls past the last boundary is reported as
   `{"event": "cue_undelivered"}` rather than silently dropped.
9. **The chord cursor is sim's, not the form cursor's.** `FormPlan` exposes
   `bars_left_in_section` but not the span, so the position inside a section is
   tracked here: reset on a section change, +1 otherwise, indexing
   `SectionSpec.chords` cyclically. An `extend` keeps the cycle running, which
   is what "cycled across bars" means for a section that got longer.
10. **Cost, measured.** Everything lead mode ADDS to the per-bar analysis — the
    attack refinement, the phase measurement, the role negotiator, the form
    cursor and the lead planner — runs at **~49,000x real time** (1.6 ms for
    80 s of audio, 32 bars). The beat tracker is on the same code path it was
    in v1 and is unaffected. `tests/test_simlead.py` asserts a 100x floor, so
    the headroom that makes live jam possible cannot silently regress.
12. **v2.1 — THE FADER TRAP, and it is v1's bug wearing a different hat.**
    Everything the negotiator knows about the human comes out of
    `FeatureExtractor`, whose onset picker is an ABSOLUTE threshold
    (`median(flux) + ONSET_DELTA`, and `ONSET_DELTA` is 4.0 flux units, not a
    ratio). Flux scales with the signal, so pulling a real human's fader down
    deletes their onsets. MEASURED on ONE v0 lo-fi render at five gains with
    **nothing about the playing changed**: onset density 5.08 → 4.21 → 3.04 →
    1.94 → 0.72 per beat from 0 to −30 dB; four concessions at 0 dB and NONE
    below −20 dB; and the machine **yielded the lead at bar 12** at −6, −12 and
    −20 dB. Turning a player down handed them the band. The joiner fixtures
    showed none of it, which is why the suite was green.

    Fixed by taking the fader out before anybody listens: the analysis chain
    (features → beat / harmony / attacks) runs on the human's signal scaled by
    `sim.lead_analysis_gain(x)` — one constant over the whole take, so the
    fader is removed exactly while every dynamic INSIDE the take survives. The
    MIX is untouched; `mix_human_db` is still the human's fader and still only
    changes what you hear. A take below −60 dBFS is left alone: silence must
    stay silence.

    The reference (−20 dBFS on a 90th-percentile 10 ms envelope) is MEASURED,
    not chosen. Onset density against gain on four takes from +12 to −30 dB:
    the joiners PLATEAU at exactly their truth (1.00 and 2.50 onsets/beat)
    below about −6 dB and gain spurious onsets above it (`on_grid` reads 1.74
    at +12 dB), while v0 material keeps accumulating low-level flux crossings
    all the way up. Normalizing UP invents onsets; normalizing DOWN converges
    on the true attacks. At −20 dBFS the joiners sit on their plateau and v0
    trap still measures ~5.1 onsets/beat, which is the number amendment A5 and
    `handoff_density_ratio` were written against.

    Result: **the same take at 0 dB and at −20 dB now produces bit-identical
    decisions**, which `tests/test_roles.py` asserts across its whole
    real-material half.

13. **v2.1 — the per-bar `reason` no longer contradicts `role`.** 29 of 44
    logged bars had `"role": "follow"` while `"reason"` began
    `"leading: section ..."`, and `reason` is the primary human-readable
    field — read cold, the machine appeared to still be leading a bar it had
    explicitly yielded. The opening word is now derived from the same
    `clock_authority` the yield is (`leading:` / `yielded:`).

11. **Known limit: the machine solo case has no listener.** With
    `input_wav=None` there is no beat state, so `concede`, `conform` and
    `yield` are all unreachable by construction and the take is the composed
    arrangement played at the composed tempo with the form cursor navigating
    it. That is correct, but it means a machine-solo run exercises the form
    cursor and the renderer only — every negotiation behavior is tested against
    a human.

## 5. CLI

```
plume lead "chill trap at 120, 3 movements" [--role lead|trade] [--record DIR]   # live: machine starts, you join
plume lead --session my.plume.json [--from-bar N]
plume simlead HUMAN.wav (--prompt "..." | --session s.plume.json) [--out duet.wav]
        [--human-offset 8.0] [--role R] [--roles k=v ...] [--mind k=v ...] [--cue "8.0:darker" ...] [--json]
plume cue "go to the chorus" [--at next-phrase]     # to a running jam/lead session (local socket)
plume roles                                          # print RoleParams: names, defaults, meanings (--json)
```
`plume jam` gains `--role` (default `follow` = v1 behavior, unchanged) and
`--lead-in "prompt"`. Backwards compatible: every v1 invocation behaves exactly
as before.

### v2 amendments — section 5 (CLI + live transport), builder E

Implemented in `cli.py` (additive block below the v1 commands) and
`realtime.py` (additive block below `jam`). Tests: `tests/test_cli_v2.py`
(139 cases). `tests/test_cli.py` and `tests/test_cli_v1.py` are untouched and
still green.

1. **`HUMAN.wav` is optional on `simlead`.** Section 8 requires a machine solo
   from a bare prompt, which the section 5 usage line forbids. `nargs="?"`.
2. **`--prompt`/`--session` are resolved through the v0 session rules.**
   `--session` is already a global option with a defined resolution order
   (explicit path, bare name + `.plume.json`, `PLUME_SESSION`, the single
   `*.plume.json` in cwd); `simlead`/`lead` reuse it rather than adding a
   second, differently-behaved session flag. Giving both `--prompt` and an
   explicit `--session` is an error, not a silent precedence rule.
3. **`--role` defaults to `lead` on `simlead`/`lead`** (`RoleParams`' own
   default is `follow`). The verb is named for the chair the machine is
   sitting in; a `plume lead` that defaults to following is a bug with a
   docstring. On `jam` the default stays absent = v1.
4. **`--roles K=V` types are read from `dataclasses.fields(RoleParams)`'
   annotations**, not from the runtime type of the default — `clock_authority`
   and `form_authority` default to `None`, so a value-derived coercion has
   nothing to coerce against. `float | None` fields also accept
   `none`/`default`/`auto` to mean "the role's default". Unknown keys and
   unknown role names both error with the valid list plus `difflib`'s
   did-you-mean, and `RoleParams.__post_init__`'s `ValueError`s surface as
   clean one-line CLI errors.
5. **`--roles role=X` beats `--role X`.** Two flags name that field; the
   explicit table is the more specific statement. Silently ignoring one of
   them is exactly the drift this file avoids elsewhere.
6. **`--cue "T:TEXT"`, with an optional `@BOUNDARY` suffix**
   (`"16:go to the chorus@next-section"`). The suffix is only read as a
   boundary when it actually names one, so a cue whose text contains an `@`
   stays text. The time is split on the FIRST colon, so a colon inside the cue
   survives.
7. **Optional `simulate_lead` keywords are offered only when it takes them.**
   `sim.simulate_lead` grew `style`/`bars`/`mix_human_db`/`decisions_path`
   beyond the frozen section 4 signature; the CLI inspects the callee and
   passes what it accepts, so it works against both. `--decisions` uses
   `decisions_path` when available and otherwise relocates the file sim wrote
   (v1's approach). Both paths are tested.
8. **`plume cue` speaks line-delimited JSON over AF_UNIX**
   (`$PLUME_SOCKET`, else `<tmpdir>/plume-live.sock`; `--socket` overrides).
   One socket per machine: a second live session **refuses to bind** rather
   than silently stealing the first one's cues, but a leftover file from a
   crashed session is reclaimed after proving nobody is listening on it.
   `CueServer`'s accept loop is a daemon thread that only appends to a list
   under a lock; the planner drains it at bar boundaries. Junk (bad JSON, a
   non-object, an empty text, a 200 kB body) comes back as
   `{"ok": false, "error": ...}` to its own sender and **cannot end the
   session** — a cue arriving mid-song must not be able to take the band down.
   Nothing listening is a clean error naming the socket, not a traceback.
9. **`jam --role`/`--lead-in` DISPATCH to `lead()`; they are not a second
   implementation.** "The machine starts" is what those flags mean, and one
   live transport is the only way `plume jam`'s v1 default can be *guaranteed*
   byte-identical. `_wants_lead()` is the guard and has a truth table test;
   `tests/test_cli_v2.py` also asserts that a bare `plume jam` passes exactly
   the seven v1 keyword arguments and not one more. **Consequence:** the cue
   socket belongs to lead mode. A plain `plume jam` opens none — the v1
   sideman has no form cursor for a cue to move, and binding a socket that
   could only no-op would be a worse answer than the error `plume cue`
   already gives ("no running plume session at …").
10. **`MachineClock` is the v2 addition to the transport.** Pure, no wall
    clock, no rng: bars are committed one at a time at whatever tempo the
    negotiator just chose, in output-stream seconds — which is exactly the
    coordinate `TimelineBuffer.write_at` addresses. A concession is therefore
    a real change in rendered bar length, never a jump inside a bar.
11. **Live lead renders through `DuetEngine`, not `engine.render`.** This is
    the one place the live path is weaker than `simlead`, and it is a
    structural limit rather than a choice: `engine.render` renders a whole
    session to a file and cannot be driven a bar at a time by a live clock,
    and `DuetEngine` is the only bar-granular renderer in the codebase. Live
    lead takes the Session's tempo, key, chord cycle, style and section map
    (each section's `energy` becomes the density hint, its `plays` keys select
    the roles) and voices them through `DuetEngine`. That is the arrangement
    as a MAP, which is what section 2 asks for — but it is **not** a
    sample-exact live replay of `plume render`. `simlead` is the ground truth
    for the full arrangement.
12. **`BarPlan.state` in lead mode is `"comping"`.** There is no `"leading"`
    in `mind.STATES` and adding one would change a frozen v1 vocabulary that
    `sim` and every decision-log consumer switches on. The lead-ness of a bar
    is carried by the `role`/`form` records beside it, which is where a test
    can assert it without grepping prose.
13. **The graceful ending happens INSIDE the stream.** v1's `jam` drains the
    tail in an `except KeyboardInterrupt:` clause that runs *after* `with
    stream:` has closed the device — at which point the playhead has stopped
    and the drain is a 30-second nap that plays nothing. `lead()` waits for
    `timeline.playhead >= timeline.end` while the stream is still open, so the
    last bar and the voice tail are actually heard. Not back-ported into v1's
    `jam`: that body is frozen for this build.
14. **Yielding the clock lowers the machine bus** (`YIELD_GAIN_DB`, −5 dB at
    `clock_authority` 0). Authority is not primarily a volume dial, but a
    machine that has handed over the lead and is still at full level is not
    out of the way in any sense a player hears. The register ceiling is
    applied only once the machine is *sharing* the floor (`clock_authority <
    0.75`); while it owns the floor, the composed arrangement's own register
    is the point.

**Cost on the live path, measured** (`tests/test_cli_v2.py`, trap 140 v0
render, 17 bars / 29.1 s of audio): the v2 decision chain — role negotiation →
form navigation → bar plan — runs at **~15,000× real time (≈110 µs/bar)**.
`DuetEngine.render_bar` beside it is v1 code `jam` already pays per bar and
measures ~31× real time, so the planner as a whole is ~31× and the beat
tracker's ~180× headroom is untouched. The test asserts a floor on the v2
addition (>1000×) and on the combined planner (>10×), so a regression fails
loudly rather than showing up as a dropout in a live session.

### v2.1 amendments — section 5 (CLI), the fixer pass

15. **The tty consent guard runs AFTER the dependency/device check.** On a
    headless box the guard fired first and told the user to set
    `PLUME_ALLOW_AUDIO=1` — advice that cannot work, because setting it only
    reveals the real, unfixable device error one line later. A dead end dressed
    as a next step is worse than no guidance. Nothing is loosened: on a box
    that can play, the guard still fires before any stream is opened, and
    `tests/test_realtime.py` asserts both halves.
16. **An unrecognised `--cue "...@BOUNDARY"` is an error.** `--cue
    "4:darker@whenever"` used to exit 0 and absorb `@whenever` into the cue
    TEXT, running at the default boundary. `plume cue --at` has always
    validated this; the two surfaces now agree. A tail with a space in it (or
    with a space before the `@`) is still prose: `"email me @home"` is text.
17. **`--human-offset` is bounded at one hour.** `plume simlead ... --bars 8
    --human-offset 9999` exited 0 and wrote a **2.89 GB, 2h47m** WAV from a
    request for twenty seconds of music, because the offset pads the human
    with silence. The error names the size the number implies.
18. **A tempo in a prompt no longer needs the literal token `bpm`.** Both
    lead-mode help texts documented examples that hit this: `plume roles`
    printed `--prompt 'chill trap at 120'` and `plume lead --help` documented
    `'chill trap at 120, 3 movements'`, and BOTH composed at 140. "at 120",
    "@120", "around 96", "tempo 140" now read as tempi, guarded by a plausible
    range (40–300) and by a unit check so "at 3 movements" and "at 4 bars"
    still mean what they say.
19. **The prompt path has a feedback channel.** `--prompt "asdfjkl qqq zzz"`
    exited 0 and rendered a full session at the defaults with no warning and no
    `understood`/`ignored` field anywhere; combined with 18 there was no way to
    tell a perfect prompt from an ignored one short of diffing the composed
    manifest against what you asked for. `compose.describe_prompt()` reports
    what each rule claimed and which content words nothing claimed; the CLI
    prints one line on stderr (commentary on the request, not part of the
    result) and puts the dict in `--json` under `prompt`. An EMPTY `--prompt`
    is now its own error instead of falling through to session auto-discovery
    and reporting "multiple session files in cwd".

**Live coverage without a device.** `realtime.lead()` is exercised end to end
against a fake duplex `sounddevice` whose "mic" is one of plume's own v0
renders and whose stream pumps faster than real time — threads, timeline, cue
socket, records and all. There is no audio device in CI, and without this the
planner thread's wiring would be asserted precisely nowhere. It terminates on
its own via `FormParams(loop_form=False)`, and a companion test sends a real
cue over the real socket to the running session and finds it in the decision
log. **Still not covered by any automated test:** actual PortAudio device
handling, latency/xrun behavior, and self-masking against a real acoustic
path — `plume jam --role lead` on real hardware remains a manual test.

### v2.2 amendments — section 5 (the live session's life), the lifecycle pass

**Why.** The consent guard (v2.1, item 15) stops a live session from
*starting* unattended. It does nothing about one that has already started.
Both times an agent left music playing on a human's machine, the session was
consented-to at the start and abandoned after; the only way to end it was to
find the PID. A guard on the door and none on the room is half a fix.

20. **`plume sessions` — a running session can be named.** Lists every live
    session with pid, kind, role, style, uptime, time left on its cap, and the
    exact command that ends it. `--json` carries the same fields plus
    `stop_command`. Nothing running is not an error: it says so and exits 0.
    The `--help` field table is read from `realtime.LiveSession` at runtime,
    like `_mind_table`/`_roles_table` — hand-written tables have drifted in
    this repo before.
21. **A session registry, so this works across terminals.** One
    `<pid>.json` under `$PLUME_SESSION_DIR` (default
    `<tmpdir>/plume-sessions/`), written after the device is resolved and
    removed in the transport's `finally`. Records whose process is gone are
    pruned silently on every read; so are unreadable ones — this is a cache
    of running processes, not a document anyone edits. Registration failure
    never raises: bookkeeping must not be able to take the band down, and a
    session with no record is still stoppable by signal.
22. **`plume stop` — graceful first, over the socket the cue system already
    uses.** `CueServer` gained two control verbs, `{"op": "stop"}` and
    `{"op": "status"}`, checked before the cue path and never touching the cue
    queue. `stop` sets an event the transport's status loop polls, which turns
    into its ordinary graceful ending: finish the bar, flush the tail, write
    the records. The ladder is socket → SIGINT → SIGTERM, stopping as soon as
    the process is gone. **SIGKILL is deliberately absent**: it would cut the
    tail and discard the records, so a wedged session is reported, with the
    command a human can run, rather than killed silently. The socket handshake
    doubles as identity proof, and a record whose pid is this very process
    gets the socket rung only — plume is a library, and `stop` must not
    SIGTERM its own caller. `--timeout` (default `DEFAULT_STOP_TIMEOUT`, 45s)
    is the budget for the two *graceful* rungs **together**, not per rung, and
    it is an upper bound rather than a delay: every wait ends the instant the
    process does. It is deliberately longer than the transports' own graceful
    ending, which is internally bounded at ~40s (a 10s planner join plus a 30s
    tail drain) — see item 28.
23. **`plume jam` gets a control socket; `plume cue`'s contract is
    unchanged.** The v1 sideman has no form cursor for a cue to move, so it
    binds a *private* control-only path (`<session_dir>/<pid>.sock`) instead
    of the shared cue socket: `plume cue` at the default path still answers
    "no running plume session", and a cue sent to a sideman is refused with
    the reason. If the socket cannot bind, the session plays anyway and `stop`
    falls back to signals.
24. **`--max-minutes` wires up `DEFAULT_MAX_MINUTES` (60).** A live session
    ends itself after the cap; `--max-minutes N` overrides; `--max-minutes 0`
    disables it — an interactive human explicitly opting out. The cap is
    opt-OUT rather than opt-in because the failure it exists for is precisely
    the case where nobody is present to opt into anything. `--max-minutes 0`
    is **not** a way around the consent guard: a non-interactive
    `plume jam --max-minutes 0` is refused exactly as a bare one is, and
    `tests/test_session_lifecycle.py` asserts that specifically. The deadline
    is `session_deadline(started, max_minutes)`, a pure function of its two
    arguments, unit-tested with no clock and no device.
25. **A session says why it ended** — in the status line and in
    `ending.json` beside the recording (`reason`, `capped`, `max_minutes`,
    `elapsed_s`, `bars`). A take that stopped because a one-hour cap fired
    used to be indistinguishable from one somebody ended on purpose.
26. **`jam`'s graceful ending actually drains now.** Its tail flush sat in the
    `except KeyboardInterrupt` handler, i.e. *after* `with stream` had exited
    — the playhead had stopped moving, so the "graceful" wait was a
    thirty-second nap that played nothing. `lead` had already been fixed; jam
    had not, and a graceful `plume stop` that drops the tail is not graceful.
27. **`plume jam` with no new flags still calls exactly the v1 signature.**
    `max_minutes` is passed only when the flag is given (the cap is applied
    inside `realtime.jam`, so "not passed" and "passed the default" behave
    identically), and `test_cli_v2.py`'s exact-kwargs test stays green
    unmodified.

28. **A backgrounded session ignored SIGINT, so the graceful rung was dead
    exactly where it was needed most.** Measured, not assumed: a live session
    started in the background by a *non-interactive* shell — `plume jam &` in
    a script, an agent's subprocess, a CI runner — inherits SIGINT as
    `SIG_IGN`, and CPython does not override an inherited `SIG_IGN`. A child
    launched that way reported `signal.getsignal(SIGINT) is SIG_IGN` and
    played straight through a SIGINT. That is precisely the population the unattended-session
    incident came from: agent-started, backgrounded, unattended. The only rung
    that landed was SIGTERM, which killed the process mid-take and discarded
    the recording. `GracefulSignals` now installs handlers for SIGINT and
    SIGTERM around both transports, turning each into the same
    finish-the-bar/flush/write ending; a **second** signal restores the
    default handler so a user hammering Ctrl-C is never trapped; and the
    handlers are restored on exit. It degrades silently when it cannot install
    (off the main thread, or an unsupported platform). SIGHUP is deliberately
    left alone — its default already ends a session when the terminal closes,
    which is the safe direction.

**Lifecycle coverage on real material.** The whole lifecycle — register,
discover, stop over the socket, cap, deregister — is run end to end through
the real `realtime.lead` transport (real threads, real socket, real planner,
real `DuetEngine`) over plume's own v0 `engine.render` output at two gains,
with only the device faked; and then again across a **genuine process
boundary**, with `plume sessions` and `plume stop` run as separate processes
against a backgrounded session. Four defects surfaced this way and none of
them would have surfaced from unit tests over fixtures:

- a malformed registry file raised `AttributeError` out of "what is playing?";
- `--max-minutes -5` reported the *device* error instead of the argument error
  on a machine with no audio hardware (argument validation now precedes the
  hardware probe);
- `plume stop` gave SIGINT a *fraction* of `--timeout`, so SIGTERM landed in
  the middle of a working flush and destroyed the take — the thing whose whole
  job is not to destroy takes (item 22);
- and the `SIG_IGN` inheritance above (item 28), which no in-process test could
  have seen, because a test never runs as a backgrounded job of a
  non-interactive shell.

**Still not covered:** pid reuse between the registry write and the signal — a
real if small window, documented in `stop_session` rather than hidden — and
the ladder's SIGTERM rung against a session that genuinely ignores SIGINT,
which is now unreachable for plume's own sessions and is exercised only with a
faked `os.kill`.

## 6. Fixture humans who *join* (tests/fixtures_gen.py extension)

`make_joiner(kind, *, machine_bpm, machine_key, bars, seed)` → audio + truth,
where the human is reacting to a machine already playing:
- `on_grid` — joins in time and in key (the easy case)
- `rushing` — consistently +30ms ahead (must trigger concession, machine speeds up)
- `dragging` — consistently −35ms behind
- `late_join` — silent 8 bars, then joins on a boundary
- `wrong_key` — plays a semitone up (must trigger conform/transpose)
- `far_key` — plays a tritone away (must NOT transpose; machine holds)
- `takes_over` — density ramps to 4 onsets/beat (must trigger yield-the-lead)
- `drops_out` — plays 8 bars then stops for 12 (machine must keep leading, not collapse)

### v2 amendments — section 6 (measured, not chosen)

**Signature.** `make_joiner(kind, *, machine_bpm=120.0, machine_key="A minor",
bars=24, seed=1) -> (audio, truth)`. Defaults added so the eight kinds are
callable positionally-free from any test; every v2 default (120bpm, `bars=24`)
satisfies all eight. Minimum bars: `late_join` 12, `drops_out` 20,
`takes_over` 8. Bad kind / too-few-bars / non-positive bpm all raise
`ValueError`.

**Sign convention.** `mean_offset_ms` is **positive when the human is AHEAD of
(earlier than) the machine grid**. So `rushing` = `+30`, `dragging` = `-35`,
matching this section's prose under one rule. `truth["mean_offset_ms"]` is the
*realized* mean including jitter (a `nominal_offset_ms` extra carries the round
number), so a test can compare an audio measurement against it directly.

**Truth extras** beyond the seven contract keys, all non-contractual but relied
on by the fixture tests: `machine_key`, `machine_beat_times_s`,
`machine_bar_times_s`, `grid_times_s` (the nominal grid position each onset came
from, index-aligned with `onset_times_s`), `nominal_offset_ms`, `kind`, `bars`,
`sr`.

**A joiner is straight and tight, not grooved.** `make_human`'s
`GrooveParams(swing=0.55, humanize_ms=3.0)` and 9ms strum stagger are wrong
here: a swung 16th moves an onset ~30ms at 120bpm, which *is* the rushing
signal, and a 27ms stagger across a triad is comparable to the deviation being
measured. Joiners use `swing=0.5`, `humanize_ms=4.0` (zero-mean, so the mean
offset survives), and a 4ms stagger. Measured accuracy of a
sub-hop attack-time estimate against the fixtures' own truth: **within 3-5ms on
every kind** (`on_grid` -3.0 vs +0.3, `rushing` +28.0 vs +30.3, `dragging`
-37.0 vs -34.7). The raw `listen.features` reading is ~11ms late and quantized
to the 512-sample hop, which is not enough to separate the two.

**The joiner's 808sub uses `release_ms=250`.** The preset's 30ms gate release
on a still-audible low sine is a step, and a step is broadband: it lands in the
spectral flux as a full-sized **onset at the note's end**. Measured in
isolation — eight 0.875s notes, one per second — the preset yields sixteen
detected onsets (one per attack, one per release at +885ms); `release_ms=250`
yields eight. A phantom onset 875ms after the beat is a phantom beat to
anything measuring the human's timing. Not fixed in `voices.py`: that is a
fixture-side choice about what the simulated human plays.

**MEASURED NEGATIVE RESULT — a summed mix cannot be used to time the human.**
This is the evidence behind section 4's "must run on the human's signal only".
Summing a joiner with plume's own v0 trap render at 120bpm puts the machine's
on-grid attacks inside `ONSET_MIN_SEP_S` (60ms) of the human's, so the detector
emits **one** onset per beat, somewhere between the two players — and where
between them is a function of the **mix balance**, not of the playing.
Measured on `dragging` (truth -34.7ms): whole-mix reading **-17.5ms at matched
RMS, -32.0ms with the human 6dB hotter**, while the isolated human reads
**-38.0ms at both**, and at a naive equal-peak (human ~10dB down) mix the
reading collapses to -1ms. An isolated reading is additionally gain-invariant
from 0 dB down to -40 dB (it returns nothing below about -50 dB). Consequence
for section 7: behaviors 2, 3 and 8 must be driven from sim's isolated human
signal; a `jam` implementation reading a summed bus will negotiate against its
own clock. `tests/test_fixtures_gen.py` asserts this as a fader-dependence
experiment rather than a bare "it fails", so that a detector which genuinely
separates the two players fails the test loudly instead of silently.

## 7. Behavior matrix (tests/test_lead_behaviors.py)

Each = one `simulate_lead` run asserted against the decision log + audio:
1. **Machine starts alone.** With `input_wav=None` or a human joining at bar 8,
   machine audio exists from bar 0 (unlike v1, where silence-until-confident is
   correct).
2. **Rushing human is met.** `rushing` → ≥1 concede event within 4 bars; final
   machine tempo higher than composed by 1–8%; never exceeds `concede_max_pct`;
   ramped (no single-bar jump > 2%).
3. **Dragging human is met**, symmetric.
4. **On-grid human changes nothing.** `on_grid` → zero concessions (guards
   against a jittery concede rule).
5. **Wrong key → transpose.** `wrong_key` → exactly one transpose event to the
   human's key within `conform_bars`+2; machine bass pitch classes move
   accordingly. `far_key` → zero transposes, and a log line saying why.
6. **Takes over → machine yields.** `takes_over` → `clock_authority` falls,
   role → follow, machine density drops (inverse coupling still applies).
7. **Drops out → machine holds.** `drops_out` → machine keeps playing through
   the human's silence; no tacet; form keeps advancing.
8. **Late join is absorbed.** `late_join` → no disruption at the join bar
   (no tempo jump, no section skip), human is simply incorporated.
9. **Turnaround waiting.** Human mid-phrase at a section boundary → `extend`
   appears in the log and the section is longer than composed, bounded.
10. **Cue lands on a boundary.** A cue issued mid-phrase applies at the next
    phrase boundary, is logged with its effect, and changes the audio
    (measurable: energy/centroid shift after, not before).
11. **Trading.** `role="trade"` → authority alternates every `trade_bars` at
    boundaries; human playing over machine bars makes the machine back off
    within a beat.
12. **Determinism.** Every one of the above, run twice, byte-identical.

### v2.1 amendments — section 7 (the behavior matrix)

**The matrix is now a matrix.** `tests/test_lead_behaviors.py` exists and holds
one row per behavior, each a `simulate_lead` run asserted against the decision
log AND (wherever the claim is about sound) against the rendered machine bus.
`test_the_matrix_is_complete` inspects the module's own functions, so a row
that is deleted, skipped, or silently broken by a fixture change fails there
rather than quietly shrinking the matrix. Before this the twelve rows were
spread across three module suites and "behavior 9 is covered" was a claim
nobody could check.

**Behaviors 2 and 3, restated — and this is a case of the spec being wrong
about reality.** As written they ask for "final machine tempo higher than
composed by 1–8%". The `rushing` fixture is the same recording at the same
tempo displaced 30 ms; it cannot bend. A machine that permanently keeps the
1–8% it conceded to it is therefore a machine *walking away from its partner*,
and measured, v2.0 did exactly that: it ended +3.26% with its downbeat 2.5
beats ahead. What behavior 2 is really claiming is (a) the machine met them,
promptly, in the right direction, by an audible-but-musical amount, and (b) it
was still with them at the end. Both are asserted; (b) — the machine's last bar
line within one beat of the human's — is the one v2.0 could not pass, and it is
strictly stronger than the sentence it replaces.

## 8. Definition of done

- Full suite green (v0 + v1 + v2), including all 12 lead behaviors.
- `plume simlead` works from a bare prompt with no human WAV (machine solo) and
  with each joiner fixture.
- A flagship pair in `renders/`: `lead_demo.wav` (machine leads a 120bpm trap
  session, human joins at bar 8 rushing, machine concedes and meets them) and
  `trade_demo.wav`, each with NDJSON + manifest.
- `plume jam --role lead` runs live (manual test documented).
- README + DREAM roadmap updated: v2 is **the band**, not just the studio.
