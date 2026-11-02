# Decision: hold time by default. Follow intent, not error.

**Date:** 2026-08-15 · **Raised by:** Juniper · **Status:** **IMPLEMENTED**
(2026-08-15, `roles.py` v2.2 — see SPEC-V2 §1's v2.2 amendment block A14–A16)
· **Supersedes:** SPEC-V2 §1's framing of conceding as "lead mode's soul"

> **Implemented, and what it measured.** `RoleParams.tempo_policy` is
> `"hold" | "meet" | "chase"`, unset meaning the role's default
> (`lead`→hold, `follow`→chase, `trade`/`solo_support`→meet), overridable with
> `--roles tempo_policy=meet`. Tests: `tests/test_tempo_policy.py` (76 cases,
> half of them on plume's own v0 renders at two gains).
>
> | | `meet` (the old default) | `hold` (the new one) |
> |---|---|---|
> | synthetic tape, 48 bars, a partner who cannot bend | worst tempo offset **1.70%**, peak residual gap **129.7 ms**, still hunting at bar 48 | **exactly 0.00%**, the human's 30.0 ms offset standing still and legible |
> | real corpus — trap140 / lofi80 at 0 and −20 dB plus 30 ms-rushed copies, 8 takes | worst **3.85%** (`trap140_rush`, identically at both gains) | **0.00% on all eight** |
> | `simulate_lead`, real rushing joiner, 24 bars | tempo moves | 90.0 bpm on every bar; bar *k* lands at exactly *k*·240/bpm seconds |
>
> The "shape" section below asked for one thing the implementation did
> differently: the field is `str | None` (unset → the role's default) rather
> than a literal `str = "hold"`, because `RoleParams`' own `role` default is
> `follow` and a follower that holds its own clock is not a follower. That is
> the same shape `clock_authority`/`form_authority` have had since v2.0, and
> the CLI already parses it with no change.
>
> **Holding is not deafness — verified, not assumed.** The offset is measured
> and published every bar (`RoleState.human_offset_ms`, event `tempo_hold`,
> reason `holding: human +28ms (policy=hold)`), which is the v2.5 mirror
> signal. Tempo *intent* is untouched: cue tempo is a separate downstream
> multiplier (`sim._LeadCues.bar_ratio()`) and the ending ritardando detector
> is `mind.Mind._form_track_tempo`, so `ritard`/`faster`/`slower`/`half time`/
> `double time` produce **bar-for-bar identical** curves under `hold` and
> `meet`, and a detected ritardando still ends the tune under `hold`. A
> structural test asserts `roles.py` never references a cue, so the two
> subsystems cannot be quietly re-coupled.
>
> Also measured, and worth knowing: an unset policy follows the *chair*. On
> plume's own lo-fi 80 render the machine legitimately yields the lead
> mid-take, so that take reads `hold` while leading and `chase` after.

## The correction

SPEC-V2 made **mutual entrainment the default** in lead mode: the human pushes,
the machine speeds up to meet them. Juniper's objection: a machine holding the
groove should **not** chase the human's timing *errors*. That is correct, and it
is what the research base already said before I wrote the spec — RESEARCH.md
records Band-in-a-Box's virtue verbatim: *"an unshakeable clock is a feature,
not a limitation; musicians adapt to a trustworthy machine."* I quoted that and
then built the opposite.

The rhythm section is the **reference**. If it moves every time a player pushes,
there is no reference, and the tempo random-walks: the human rushes, the machine
follows, which makes the machine's new tempo the thing the human is now ahead
of, and it walks away. Conceding is a positive feedback loop unless it is
gated hard.

## The distinction that matters: error vs intent

Both look like "the human is not on our grid." They are completely different
musical events and must be handled by different subsystems.

| | **Error** (drift, rushing, sloppiness) | **Intent** (a real tempo change) |
|---|---|---|
| Shape | oscillates, inconsistent sign | consistent direction, monotone |
| Where | anywhere, mid-phrase | at structural boundaries — final phrase, section end |
| Company | nothing else changes | usually paired with a dynamic change (a ritard fades, a push lifts) |
| Right response | **hold.** Let them find you. | **follow.** Ritard with them, end together. |

So: **clock-level drift conceding defaults OFF; form-level tempo intent stays
ON.** The ending ritardando detector built during the v2 form work is already a
separate subsystem from the concede rule, which makes this a clean split rather
than a rewrite.

## The shape to implement

Replace the single `concede_*` cluster's implicit default with an explicit
policy on `RoleParams`:

```
tempo_policy: str = "hold"     # "hold" | "meet" | "chase"
```

- **`hold`** (new default for `role="lead"`) — never concede to drift. Still
  follows *intent*: a detected ritardando at a phrase ending, and any explicit
  cue. The human's measured offset is still computed and still logged (it is
  free, and it is the practice-feedback signal — see below).
- **`meet`** — the current mutual-entrainment behavior, with all the guards the
  builders added (plausibility, cross-bar agreement, one-bar ramp clamp).
- **`chase`** — v1 follow-mode behavior: the human owns the clock.

`role="follow"` keeps `chase`; `role="trade"` uses `meet`. All three stay
settable per-session via `--roles tempo_policy=meet`.

## Why holding is also the better product

A machine that follows your rushing **destroys its own practice value.** You
cannot hear an error you are being accommodated for. Holding steady is what
makes the drift audible — and the offset measurement, which the concede rule
needs anyway, is exactly the number the v2.5 "mirror" practice-feedback idea
wants (*you're pushing the & of 2 by 24 ms*). Same measurement, two products:
hold the time, and show the human what they did.

This also means the roadmap's v2.5 gets most of its machinery for free from a
subsystem we are now mostly turning off by default.

## Follow-on (roadmap, not this change)

Voice cues: `parse_cue()` already exists and takes text, so speech recognition
is a front-end, not a new subsystem — shout "fade out" or "one more" and it
enters the existing boundary-aligned cue queue. Two real problems to solve
first: the machine is playing loudly into the same room (the mic-bleed /
self-masking problem v1 already has, worse), and a false positive that ends the
tune is expensive — so voice cues want confirmation for destructive actions
("take it out") and instant application for additive ones. Filed alongside
v4 "Eyes"; it is the same multi-modal move, one modality over.
