# Settings and sound

**Settings** is in the header, next to Takes. It hides while a take is playing, since a
running take keeps what it started with. Nothing in Settings opens a device or makes a
sound. Sound only starts from Start.

## What you play into

**Device** and **Channel** are what plume listens to. With no choices made, it uses the
device setup heard you on, or else the only audio interface it can find, or else the
computer's own mic. The small line under Start says what it will use.

Plume treats each input one of two ways:

- **A room mic** hears you and the band together. Once the band is in, plume cannot tell
  your notes from its own, so it shows less about you and cannot always hear you stop.
  Press Stop when you are done.
- **A plugged-in instrument** (a direct input, through an audio interface) hears only you.
  Plume can follow you for the whole take, and the band finishes by itself when you stop.

Plume decides which from the device's name, and a name can mislead it. If a take hears
the band coming back through a direct input, the live screen says so and suggests another
device.

## Where the band comes out

**Output** is where the band plays. It starts on **Same as input**, or **System output**
for a room mic. Pick another output to let a room hear the band while your instrument
still records through your interface.

With a plugged-in instrument and another output chosen, one more tick appears: **Send my
instrument to this output**. It arrives slightly late. Leave it off for a mic, which would
howl.

Wireless speakers and headphones arrive late, which makes playing along hard. Wired is
better.

## Hear me

**Hear me** plays your input back to you as you play. It is off until you tick it, and it
needs headphones: through speakers it would feed back. Ticking it is your yes to
headphones for that device.

It adds the delay your device reports, which Details shows. If your interface has a
Direct or monitor button, that has no delay. Use one or the other: both at once plays you
twice. Hear me never goes into the recording.

With another Output chosen, Hear me is not offered, because plume treats that output as
speakers.

### Sound

Under Hear me, **Sound** gives your instrument a finished sound: Clean room, Dreamy (chorus,
an echo on the band's tempo, a long shimmering reverb), Lofi tape, or Crunch. Match the
style picks one from the style. It is Off until you choose one. It changes only what you
hear: your recording stays clean, because that is what plume listens to. Lofi tape adds
about 2 ms; the others add no delay. A buzzing cable is still muted.

In a take you can say `sound dreamy`, `sound off` and so on.

## Set up again

Plume remembers what setup learned for each device, so later Starts skip the questions.
**Set up again** forgets all of it, and the next Start begins from "Play something."

Settings also shows the folder your takes go in.
