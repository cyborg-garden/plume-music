# Playing with plume

## How the band follows you

Plume plays the tempo you count, and holds it. Count the beat you want to feel: it will
not decide you meant twice or half as fast.

The band waits until it is sure enough to join. The drums play alone until you start.
Then, unless you named a key, plume listens for one. Once it has a key, the bass and chords come in, about
eight bars after you started, and the melody follows four bars later. If it hears no
clear key after 16 bars, it takes its latest reading. If it has no reading at all, the
drums keep playing alone and the band keeps waiting.

## Key and style

In Settings, Key and Style say **Detect** until you type over them. With Detect, plume
works both out as you play, and the live screen marks what it chose as a guess, for
example "F major, midwest (a guess)". A key or style you typed is never called a guess,
and a key you named never moves. Plume plays major and minor keys.

The styles are `lofi`, `midwest`, `trap`, `synthwave`, `jazz`, `house`, `hiphop` and `rnb`.
You can also type `boom bap` for hiphop, or `r&b` or `neo soul` for rnb.

A style plays at whatever tempo you count, so house at 90 is slow house. Each has a usual
range: lofi 70 to 90, rnb 60 to 90, hiphop 80 to 100, house 118 to 130 and jazz 70 to 220.
Midwest changes with tempo: below 70 it plays a swung breakbeat with the snare on 2 and 4,
and from 70 up it plays half-time with the snare on 3.

A guessed style is only a guess from your tempo, key and how busy you are. Plume never
guesses jazz below 150 bpm, so name it if that is what you want.

## Busier and sparser

While a take plays, open **Details** under Stop. **Busier** adds hats, keys and melody, and
**Sparser** takes them away. A cue lands at the start of the next phrase, not at once.

Details also shows the bar, the tempo, the stage, and what plume hears from you. Over a
room mic, once the band is in, most of that is hidden, because plume cannot tell your
notes from its own. It would rather show nothing than guess.

There are many more cues, typed from a terminal. See
[From a terminal or a coding agent](#/guide/terminal).

## If you stop

On a plugged-in instrument, a rest under 6 seconds never dips the band. Stop for longer
and the band fades, and playing again during the fade brings it straight back. If you
stay quiet, the take ends by itself.

Until you have played a full bar, plume waits 30 seconds of silence before it fades, so
you have time to get your hands back on the strings after the count.

A take ends by itself after 60 minutes, whatever is happening.
