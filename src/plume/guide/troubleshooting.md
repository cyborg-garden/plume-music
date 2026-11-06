# Troubleshooting

**Nothing happens after I count.** Knock harder or closer to the mic, or clap. Plume needs
four clear hits or six soft ones. On a plugged-in guitar, play the count on muted strings.

**The drums started by themselves.** Something in the room sounded like a count. Press
Stop, start again, and stay quiet until you count.

**The tempo feels twice or half as fast.** You counted a different beat than you meant.
Count the one you want to feel.

**The band never joins.** Play steady, held notes over the drums. With Key on Detect, the
band waits until it hears a key, so typing one in Settings helps.

**It doesn't fade when I stop.** On a room mic, plume can't hear you stop over the band.
Press Stop. A plugged-in instrument fixes this.

**"Nothing came in."** Check the cable, the gain and the channel. Setup offers any other
device it can try.

**The page says it heard the band in my input.** Something is sending plume's sound back
into your input, such as a loopback or monitor setting on your interface. Turn that off,
or choose another device in Settings.

**Settings has gone.** It hides while a take is playing. It comes back when the take ends.

**Hear me isn't there.** With another Output chosen, plume treats it as speakers and does
not offer Hear me. Set Output back to Same as input or System output.

**The page says its link has stopped working.** Run `plume gui url` in a terminal and open
the new link. The key lives in the tab that opened the link, so a second tab can't use it.

**I can't stop it from the page.** Run `plume stop` in any terminal. It always works.

**A take is still running after I closed the page.** Reloading finds it again. A take
waiting for a count, or on a room mic, ends about 3 minutes after the last page stopped
looking at it.

Plume's live parts have mostly run on simulated devices, and little real gear has been
tried. If something here is wrong for your setup, that is worth reporting.
