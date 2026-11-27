# The story feed

One list of true events for a take. The page reads it. An agent reads it. Both
read the same list, so they never disagree about what happened.

Status: draft for the owner's read. The reader (`plume.story`) is built and
tested. The socket parts (the `events` op, the ring, named senders) are specified
here and not built.

## The rule

A sentence appears only if a fact in the take's own records supports it. Every
event names that fact. No sentence is a guess, a plan stated as a result, or a
summary of what the player did.

A wrong line does more harm than no line. When the records cannot say, the feed
says less.

## What an event is

```json
{
  "id": "c3-applied",
  "seq": 12,
  "kind": "cue",
  "state": "applied",
  "outcome": "landed",
  "bar": 33,
  "sentence": "Busier, from bar 33.",
  "intent": null,
  "settled": null,
  "detail": "Bar 33. Someone asked for busier at bar 30. It landed 6.1 seconds later.",
  "who": "Claude",
  "sender": "Claude",
  "name": "busier",
  "cue": 3,
  "facts": [
    {"in": "rows", "index": 33, "field": "form.applied_cue", "part": 0},
    {"in": "cues", "index": 3, "field": "text"}
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | Steady across reads while its facts stay in the input. A reader keys on it. |
| `seq` | Its place in this list. |
| `kind` | `cue`, `band` or `end`. |
| `state` | For a cue, what may be claimed. See the next section. |
| `outcome` | For an applied cue, what it did. |
| `bar` | The bar the event belongs to. `null` if no bar had sounded yet. |
| `sentence` | One short line. This is the story line. |
| `intent` | Where a received cue waits. Never a result. |
| `settled` | On a cue's own event: `true` once its result is known. `null` on other events. |
| `detail` | The take page's longer sentence for the same fact, for Details. |
| `who` | The word used for the sender: a claimed name, "You", "Another terminal" or "Someone". |
| `sender` | The claimed name, cleaned, or `null`. |
| `name` | The cue's words, only if they are exactly a cue plume knows. |
| `cue` | Which cue row this is about. |
| `facts` | Where in the input each claim stands. |

An event's sentence can change between reads. A tempo cue reads "reached the
band, what it did is not written yet" until the next bar is written, then
"Faster, from bar 34." The `id` stays the same, so a reader replaces the line.

## What may be claimed

Each state is tied to one fact.

| State | The fact | Where it is |
|---|---|---|
| `received` | The socket took the cue. | A cue row exists. The socket writes one only for a cue it accepted. |
| `applied` | The cue reached the band at a boundary. | A decision row carries the cue in `form.applied_cue` (`FormPlan.applied_cue`). |
| `dropped` | The cue ran out of time. | A decision row carries the cue in `form.dropped_cue` (`FormPlan.dropped_cue`). |
| `not_understood` | The parser does not know the words. | `parse_cue` answers `unknown`, or the looper answers `unclear`. |
| `answered` | The looper or the player's sound replied at once. | The cue row carries the reply in `loop`. |

`answered` is the fifth state. A loop cue and a sound cue never reach the form,
so no decision row can name them. Their reply is the only fact, and it is in the
cue row. Folding them into `applied` would claim more than the reply says.

### Reaching the band is not changing it

The form lists every due cue in `applied_cue` before it checks what the cue is.
So `applied` means "reached the band", and `outcome` says what happened next.

| Outcome | Meaning | A sentence may say the music changed |
|---|---|---|
| `landed` | A matching effect is in the row. | Yes |
| `ended` | The form ended the take on this cue. | Yes |
| `held` | A loop holds the tempo, so the cue waits. | No |
| `waiting` | The bar that would show the effect is not written yet. | No |
| `nothing` | The row shows no effect, or the value was already at its limit. | No |
| `unsure` | The form jumped, and the row does not say this cue did it. | No |
| `tone` | The row logs a tone change that cannot be confirmed. | No |

The outcome comes from `record.translate`, the reader the take page already
uses. The story feed does not decide outcomes a second way.

## Intent and fact are separate sentences

A received cue has two sentences, in two fields.

- `sentence`: "Claude asked for busier." This is the fact that the cue arrived.
- `intent`: "Queued for the next phrase." This is where the cue waits. It comes
  from the cue row's `at`, and it promises nothing.

The result is a different event, and it appears only after the row that proves it.

- "Busier, from bar 33."

The page shows `intent` only while `settled` is false. Once the result is in, the
intent line goes away.

## The sentences

These are the words for the owner's read. Each line is fixed text plus a bar, a
cue name or a sender.

| When | Sentence |
|---|---|
| Cue received | "Claude asked for busier." |
| Cue received, end cue | "Claude asked to end." |
| Cue received, words that are not exactly a cue | "Claude sent a cue." |
| Intent | "Queued for the next phrase." / "Queued for the next section." / "Queued for the next bar." |
| Not understood | "Claude sent a cue I did not understand." |
| Landed | "Busier, from bar 33." |
| Landed, words that are not exactly a cue | "A cue landed at bar 33." |
| Ended | "Ending, from bar 40." |
| Ended a bar later | "Ending, one bar after bar 40." |
| Held | "Faster is on hold. A loop is holding the tempo." |
| Waiting | "Faster reached the band at bar 33. What it did is not written yet." |
| Changed nothing | "Busier reached the band at bar 33 and changed nothing." |
| Unsure | "The band jumped at bar 33. The log does not say this cue did it." |
| Tone | "Darker reached the band at bar 33. I can't find that the tone changed." |
| Dropped | "Busier ran out of time at bar 49." |
| Never used, take over | "Busier was not used. The take ended first." |
| Loop cue, yes | "You asked to record a loop. The looper said yes." |
| Loop cue, no | "You asked to stop a loop. The looper said no." |
| Loop cue, not followed | "You asked for a loop. The looper could not follow the words." |
| Sound cue | "You set your sound to Dreamy." / "You asked for a sound I do not have." |
| Band in | "Drums came in on your count, at 96 bpm." / "Bass and chords in, from bar 8." / "Melody in, from bar 12." |
| Fade, came back, came down | The take page's own sentences, unchanged. |
| End of a finished take | The take page's own ending sentence, unchanged. |

A cue's raw words never appear. "stop calling me that" holds a cue word, and it
could be anything anyone typed. Only words that are exactly a known cue are shown.

## Which events speak

These speak, live and after:

- A cue received, applied, dropped, not understood, answered.
- The band coming in: drums, bass and chords, melody.
- The band coming down or coming back with the player.
- The fade starting, and the band coming back from it.
- How a finished take ended.

These stay out until the owner signs off the direct-input test:

- The key.
- The chords.
- The style plume guessed.

The take page's "Bass and chords in, in F major." names the key, so the feed
uses its own line for that event and leaves the key out.

Section changes, the break, the outro and "I was sure I could hear you by here"
also stay out of the live line. They are on the finished take's page.

## Where live events come from

A new `events` op on the cue socket.

```
-> {"op": "events"}
<- {"ok": true, "protocol": 1, "session": "lead", "bar": 37,
    "events": [ ... the newest 50 events ... ]}
```

The take keeps two small lists in memory:

- the last 256 decision rows, added where a decision row is made;
- the last 64 cue rows, added where the socket accepts a cue.

The op runs `story.story()` over those two lists and returns the newest events.
No file is read. A take without `--record` has a story.

Rules for the ring:

- A cue row is added the moment the socket accepts the cue. It does not wait
  for a recorder, a bar, or the take folder.
- Each cue row gets a `seq`, counting up from the first cue of the take. Event
  ids use it, so they stay steady when old rows leave the ring.
- Once a decision row has left the ring, the take passes `partial=True`. The
  reader then says nothing about how the take began, because the oldest row it
  holds is not the first bar.
- The op is read-only. It never touches the cue queue and never blocks the band.
- The reply is whole each time. A reader replaces its list, keyed by `id`.
  No cursor, no "since".

## Where finished-take events come from

The same function, run over `decisions.ndjson` and `cues.ndjson`, with
`ending.json` and `live=False`. One function, two inputs.

## Held cues

A cue can arrive the moment the socket listens. The take folder is not open yet,
so the recorder holds such a cue and writes it once the folder opens.

The ring does not wait. The cue row goes into the ring at once, with no bar, and
the feed shows it as received at once. Its `bar` is `null`. When the take opens
and a row applies the cue, the result joins the list under the same `id`.

## The sender

A sender names itself. Nothing checks the name.

- Free text.
- Trimmed. Runs of space become one space.
- Printable characters only.
- At most 24 characters.
- A name that is one of plume's own words for a sender ("You", "Someone",
  "Another terminal", "I") counts as no name.

The page renders the name with `textContent`, never as markup.

The name is a claim. Any program on the machine can send a cue with any name.
The page says "Claude asked" the way a guest book says who signed. It is useful
for following a take. It is not proof of who sent a cue.

Names:

- `--from NAME` on the command line.
- `sender` in code and in JSON. `from` is a Python keyword.
- `source` stays what it is: which door the cue came through (`gui`, `cli`,
  `voice`, and `agent` once named senders are built). A cue with a `sender` and
  no known source reads as the sender's claimed name.

The take page's longer sentence (`detail`) uses `record.py`'s own word for the
sender. It names the source until the named-sender change teaches it the name.

## Protocol compatibility

- A reader ignores fields it does not know. Rows and cue rows may gain fields.
- A server ignores request fields it does not know. An old command line can cue
  a new take. A new command line can cue an old take; the old take drops `sender`.
- The `status` reply gains `"protocol": 1`. A client that needs `events` checks
  it. No number means a take from before the op.
- An unknown op answers `{"ok": false, ...}`, as it does now. A client treats
  that as "no story here" and falls back to `status`.

## Placement on the page

One quiet line under the headline. It shows the newest event's `sentence`, and
the `intent` beside it while the cue waits. The full list is in Details, newest
first, using `detail` where there is one.

No new control.

Alternative for the owner to pick: the story line is the headline, and key and
style move to the small line.

## External takes

A take started from a terminal or by an agent gets a live view: who started it,
the bar, the story line, and Stop. All four come from the socket.

The page reads no folder that the session registry names. It shows what the
running take says about itself over the socket and nothing else. A fuller view
waits.

## No take found

`plume cue` and `plume sessions` say which folder they looked in and give one
next step.

## Decided, so it is not reopened

- `plume cue` stays. It is the way in.
- `plume tell` is retired. Its verbs are cues.
- No MCP server in v1. An agent uses `plume cue` and `plume watch`.

## Fallback

If the `events` op costs more than two days, an external take shows who started
it, the bar, the cue count and Stop from the `status` op. The story line then
appears only on takes the page started.

## The reader

`src/plume/story.py`:

```python
story.story(rows, cues=None, ending=None, *, live=True, partial=False, route="unknown")
story.clean_sender(value)
```

Pure. No file, no audio, no device, no network, no clock.

It reads cue rows with the same limits as `record.py`. A cue log past those
limits reads as no cue log. A row cue with no cue row behind it still speaks,
with no sender: "Busier, from bar 8."

`record.translate` does not say which row and which cue each of its events came
from. The reader works that out again from the bar, the cue name, the sender
word and the bar the cue was asked at. Two events it cannot tell apart are about
the same words, from the same sender, asked at the same bar, on the same row, so
either sentence is true of either cue.

## Fixtures

`tests/story_fixtures.py`, checked by `tests/test_story.py`.

| Fixture | What it proves |
|---|---|
| Held cue, before the take opens | A cue with no bar and no row shows as received at once. |
| Held cue, after the take opens | The same cue lands, under the same id. |
| No record folder | Lists in memory are enough. No ending, no file. |
| Unknown source | An unknown source is "Someone". A bad name is no name. |
| Dropped cue | Dropped comes only from the row that says so. |
| Two cues in one boundary | Each keeps its own sender and its own bar. |
| Not understood | The parser's reply, and the words are never repeated. |
| Never used | A cue no row took is only ever received. |
| Row without a cue log | The row speaks, and names no sender. |

The test reads every event in every fixture and checks each claim against the
fact the event points at: the state, every number, the sender, the cue name and
the intent. A second set of tests breaks a sentence on purpose and confirms the
check fails. A seeded run of 400 random takes checks the same rule on cues that
collide.

## Open questions for the owner

1. The sentences above. Any line that reads wrong.
2. Placement: one line under the headline, or the story line as the headline.
3. The fifth state, `answered`, for loop and sound cues. The other choice is to
   leave those cues out of the story line and keep them in Details.
4. "Claude asked to end." for an end cue, or the cue's own word ("Claude asked
   for stop.").
