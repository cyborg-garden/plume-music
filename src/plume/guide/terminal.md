# From a terminal or a coding agent

Everything the page does, a terminal can do, and so can a coding agent that has access to
this machine. The commands below work alongside the page: a cue typed in a terminal
reaches a take started from Start.

## The page itself

- `plume gui` opens this page.
- `plume gui --no-open` prints the link instead of opening a browser.
- `plume gui url` prints a fresh link to the page that is already running.
- `plume gui --takes DIR` reads and saves takes in another folder.
- `plume gui --take DIR` opens one take's Record first.

## Steering a take that is playing

`plume cue "..."` sends a few words to the running take. It lands at the next phrase.
Add `--at now`, `--at next-bar` or `--at next-section` to change that. A cue that has not
landed within 16 bars is dropped, and `--ttl-bars N` changes that number.

These change a live take:

- **More or less going on:** `busier`, `sparser`, or words like `fill it up`, `denser`,
  `strip it back`, `thin it out` and `simplify`.
- **Tempo:** `faster`, `slower`, `half time`, `double time`, and `ritard`, which slows a
  little more each bar.
- **Sections:** `next section`, `previous section`, or a name after "go to the", such as
  `go to the chorus`. A name plume can't place in the song is logged and ignored.
- **Energy:** `louder`, `bigger` or `lift it` jumps to a hotter section, and `quieter`,
  `calmer` or `bring it down` to a calmer one, when the song has one.
- **Ending:** `take it out`, `wrap it up`, `last chorus` or `stop` ends the tune.

Plume understands these but does not change a live take with them: a style name (the
style is chosen when the take starts), `darker`, `brighter`, `warmer`, `dry` and `wet`.
Anything else is logged as not understood, and nothing changes. Add `--json` to see how
plume read a cue.

Loops are cues too, such as `make a loop of this`, `that's the loop`, `mute the bass loop`
and `list loops`. They have only run against a simulated device so far.

## Watching and stopping

- `plume watch DIR` prints a JSON line every 4 bars about the band and the player. Add
  `--every 8` for every 8 bars, and `--changes-only` to skip lines where nothing changed.
- `plume sessions` lists what is running.
- `plume stop` ends it gracefully, from any terminal. `plume stop --all` ends every take.

`DIR` is the take's folder. A take started from this page has a new one under
`~/Music/Plume`. In each `watch` line, trust `section`, `stage`, `key` and `chord`. The
`you_*` fields describe the player only when `you_trusted` names them: over a room mic
with the band in, it names none.

## Starting a take without the page

```
plume lead --tap-start --detect --record takes/
plume lead --tap-start "lofi in F major" --record takes/
```

From a terminal this opens your audio device and waits for your count. A script or agent running without a terminal
must also set `PLUME_ALLOW_AUDIO=1`. That is on purpose: nothing makes sound by accident.

A live take makes real sound in whatever room the machine is in. An agent checking its own
work should use the tests, which use a fake audio device, never a live take.

## Talking to it

```
plume voice "lofi in F major"
```

This starts a take you can talk to: "plume, stop", "plume, restart", "plume, busier" and
"plume, sparser". Speech is turned into text on this machine and goes nowhere else. It
needs the `voice` extra, and it reacts a few seconds late. It has not yet been proven on a
live voice, so keep a keyboard nearby.
