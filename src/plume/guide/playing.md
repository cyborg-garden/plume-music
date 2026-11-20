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

Under Start, **Style** says **Let plume guess** until you pick one. Pick a style and the
band plays it from the first bar, and nothing you play changes it. Plume remembers your
pick for next time. To go back to guessing, pick **Let plume guess** again.

Key is in Settings and says **Detect** until you type over it. Plume works out whatever
you left to it as you play, and the screens say which is which, for example "F major (a
guess) · Folk, plume's guess", or "F major · Folk, chosen". A key you named never moves.
Plume plays major and minor keys.

While plume is guessing the style, it does not know it yet when you start, so it plays
only a quiet click on each beat, a little stronger on the one, like a metronome. The
style's own drums come in with the bass and chords, at the start of a bar. A style you
picked plays its own drums from the first bar.

The styles are Folk, Hip hop, House, Jazz, Lo-fi, Midwest emo, R&B, Synthwave and Trap.
From a terminal, name one in the prompt: `lofi`, `midwest`, `trap`, `synthwave`, `jazz`,
`house`, `hiphop` (or `boom bap`), `rnb` (or `r&b`, `neo soul`) or `folk`.

Folk is a light acoustic band: brushes and a cajon, an upright bass on the root and fifth,
and a piano playing plain open chords. It plays in 4/4 only for now, so a waltz is not
there yet.

A style plays at whatever tempo you count, so house at 90 is slow house. Each has a usual
range: lofi 70 to 90, rnb 60 to 90, hiphop 80 to 100, house 118 to 130, folk 60 to 132
and jazz 70 to 220.
Midwest changes with tempo: below 70 it plays a swung breakbeat with the snare on 2 and 4,
and from 70 up it plays half-time with the snare on 3.

A guessed style is only a guess. Your tempo says which styles it could be, and on a direct
input how you play picks one: whether your eighth notes swing, how many strokes you play to
a beat, whether they fall on the beat or between, how even they are, and whether you strum.
Swung eighths lean to jazz, stabs between the beats to house, a busy even pulse to
synthwave, busy uneven playing to midwest, and a few notes with space to trap, lofi, hip hop
or r&b. How loud you play hardly changes the guess, unless the input is turned so low that your
quietest notes are lost in its hiss. A room mic hears plume's own drums too,
so there the guess comes from your tempo and key alone.

Plume guesses folk for plain playing on the beat, between 60 and 132 bpm, and a strum counts
toward it. A strum drags the pick across the strings of a chord, and plume hears the attack
last longer than a picked note's. Very quick strums are not heard as strums. Name the style
you want whenever the guess is wrong: a named style always wins.

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
