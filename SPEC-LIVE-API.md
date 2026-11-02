# SPEC — the third chair, live (puppeting plume while a human plays)

> "it should be like a cli/api or something that you can just puppet master
> live haha ai layered programs and on demand responsive UI is the future"
> — Juniper, 2026-08-15

DREAM.md promised three chairs: the human, the engine, and **an agent sitting
slightly outside time** — arranging while the engine keeps the groove. Today
that agent can compose a session, render it, and read a log after the fact. It
cannot reach into a running session and *play*. This spec is that reach.

## 0. Why this is architecturally possible at all

An LLM is far too slow to be in the audio loop, or even the beat loop. It
thinks in seconds; a beat is 600ms and a sample is 20 microseconds. The naive
reading is that a model can't be a band member.

The naive reading is wrong, and the research base said why before v1 was built:
**ReaLJam's anticipation protocol** — commit the near future early, stream the
plan, let a local scheduler render it exactly on the beat, and allow revision
until a deadline. Latency stops mattering when you are deciding the *next*
phrase rather than the current sample. plume already works this way: bars are
rendered ahead and revisable until one beat before their downbeat, and cues
apply at phrase boundaries, never mid-phrase.

So the three chairs are three **clock rates**, and each is filled by whatever
can actually keep that time:

| chair | rate | who | decides |
|---|---|---|---|
| reflex | audio / beat | the DSP + tracker | phase, onsets, what sounds now |
| judgment | bar | `mind.py` / `roles.py` | come in, thin, hold, concede, yield |
| imagination | phrase / section | **an agent** | what this piece is becoming |

A model that tries to occupy the first two chairs will always be late. Given
the third, it is early — which is the only place a partner is ever useful.

## 1. Two channels

**State out — `plume watch`.** A live NDJSON stream of everything the session
believes: per-bar state, tempo, confidence, key/chord, role, authority,
measured human offset, form position, plus events (lock, unlock, entry, exit,
concession, cue applied, section change). This is the v2.3 streaming-records
work generalized from "a file written for forensics" to "a feed meant to be
read while it matters".

```
plume watch [--session-pid N] [--since t] [--follow] [--json]
```
Multiple readers, no back-pressure on the audio path, one line per event,
flushed. A reader that falls behind misses lines rather than stalling the band.

**Commands in — `plume tell`.** Everything the agent can say, delivered over
the existing live socket, applied at a musical boundary.

```
plume tell "come in"                 # arm entry at the next phrase
plume tell "lay out for 8"           # bounded exit, then return
plume tell "take it"                 # hand the lead over (role -> lead)
plume tell "hold the tempo"          # tempo_policy=hold, live
plume tell "darker, sparser"         # texture cues (already parsed by parse_cue)
plume tell "go to the chorus"        # form navigation
plume tell "set commit_bars 16"      # any Mind/Role param, live
plume tell --at now "stop"           # the escape hatch, unquantized
```
Every command returns *when it will land* (`{"applied_at_bar": 42, "boundary":
"next-phrase"}`), not just an ack — an agent must be able to plan against the
future it just created. `parse_cue()` already exists and takes text, so this is
mostly plumbing plus a verb table, not a new subsystem.

## 2. The loop an agent actually runs

```
tail plume watch  →  decide  →  plume tell  →  hear the consequence in the feed
```
Because commands are boundary-quantized, an agent that takes three seconds to
think is still early for a change landing eight bars out. Because the feed
carries *reasons*, not just numbers, the agent can tell the difference between
"it chose to lay out" and "it lost the beat" — which is the difference between
leaving it alone and intervening.

This is the same render→analyze→adjust loop that made v0 work for an LLM,
running at phrase rate instead of take rate. That loop is the reason plume
exists; this makes it live.

## 3. The UI falls out of the feed

Juniper's second point — *on-demand responsive UI* — follows from the above
rather than needing its own machinery. The feed **is** the state; a view is a
projection of it, and which projection is useful depends on what you are doing:

- **learning** — your timing offset per beat position, drift trend, where you
  rush (the v2.5 "mirror"; the data is already in every bar record)
- **lead** — the form map, where we are in it, what's coming, cues pending
- **follow** — almost nothing. A state word and a confidence. Get out of the way.

The interesting version of "responsive UI" is not a mode switch: it is that a
model reading the feed can *decide what is worth showing right now* and render
it, rather than a designer having frozen that decision months earlier. plume's
job is to make that cheap — emit honest, complete, well-named state and never
force a layout. The surface can be regenerated; the feed is the contract.

## 4. Constraints that do not bend

- **Consent and stoppability outrank the puppet.** No command may open the
  audio device without the existing consent gate, disable the duration cap, or
  make a session unstoppable. `plume stop` must work *while* an agent is
  driving, and must not require the agent's cooperation. (An agent left a live
  session running unattended; that must remain impossible.)
- **The human wins collisions.** v1's rule holds at this layer too: an agent
  telling plume to play over a human who has started playing loses.
- **Never mid-phrase.** Only the explicit escape hatch (`--at now`, and stop)
  bypasses boundary quantization.
- **The feed must be honest.** If a command was understood but not applied, the
  feed says so — v2 already learned this the hard way, when cues reported
  "applied" and did nothing.

## 5. Build order

1. `plume watch` over the streaming records from v2.3 (read side first — it is
   independently useful, and it is how anything else gets debugged).
2. `plume tell` with the verb table above over the existing cue socket.
3. Agent-facing wrapper (MCP server or equivalent) exposing watch/tell, so a
   model can be handed the chair without shelling out.
4. A worked example: an agent that jams — reads the feed, waits, and decides
   once per phrase whether to change anything. Its transcript is the artifact.
