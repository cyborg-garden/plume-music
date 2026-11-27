# Logic after v1: record control and a plugin

v1 gives Logic a template and a recipe. Plume cannot press record, stop, or arm a
track in Logic. This page lists the routes that could add that, and what each costs.
None is built. None is tested.

**Every statement about Logic on this page is unverified.** Nothing here has been
read from Logic's documentation or run in Logic. Each one carries `[untested]`, as
in [logic.md](logic.md).

## What is missing

Record and arm control in Ableton Live is planned for v1 and is not in plume yet. The
plan is for plume to talk to a remote script inside Live, find tracks by name, arm
them, and start and stop recording. `[untested]` Logic has no such script interface.
Three routes could stand in for it.

## Route 1: Mackie Control emulation

`[untested]` Logic supports control surfaces that speak the Mackie Control protocol
over MIDI. Plume could pretend to be one.

What it gives, `[untested]`:

- Play, stop and record.
- Record-arm for each track, eight at a time, with bank buttons to reach the rest.
- Track names come back to the surface's display, so plume can look for the
  `Plume` tracks by name rather than by position.

What it costs:

- Two ports, one each way. Plume makes one outgoing port today.
- `[untested]` A one-time setup step in Logic: add a Mackie Control device and point
  it at plume's ports. Logic may only offer this while plume is running.
- The protocol's handshake and display messages must be implemented and tested
  against a real copy of Logic. Any mistake arms or records the wrong track, so
  plume must read back the state before it acts.
- A person who already owns a control surface has two. Banking can interact.

Why it is the likely first step: it is plain MIDI, it stays in Python, and the same
work serves other DAWs that accept Mackie Control.

Size: about two weeks with a Logic user testing, most of it the handshake and the
track-finding.

## Route 2: MIDI Machine Control

`[untested]` Logic has a setting to listen to MMC. MMC covers play, stop and record.
It has no way to arm a named track or to read anything back.

- Whether the setting still exists, and whether Logic obeys play, stop and record
  through it, is a question on the [test sheet](logic-checklist.md). Unanswered.
- Even if it works, arming stays manual. It is a small gain over v1 and is covered
  by Route 1.

Plume v1 sends no MMC.

## Route 3: a "Plume Part" plugin

An Audio Unit that sits on a track and plays one part of the band. You pick the part
in the plugin. It gets the notes from plume over a local connection.

What it gives:

- No MIDI port and no channel setup. Drop the plugin on a track and choose "bass".
- The same plugin could load in other DAWs.

What it does not give:

- A plugin cannot press record or arm a track in its host. It does not replace
  Route 1.

What it costs:

- A second codebase in C++, with its own build, signing and notarisation.
- Testing in each DAW and each plugin format.
- A decision on whether it is an instrument slot that passes notes on, or a MIDI
  effect in front of the instrument. Hosts differ on which they allow.

Size: weeks, not days.

## A proposed order

This is a proposal. Nothing here is decided beyond Mackie Control being the first
thing looked at after v1.

1. Read the test sheet answers.
2. Mackie Control emulation. What command it sits behind is open.
3. The plugin, perhaps only once Route 1 shows that people use plume with Logic.
