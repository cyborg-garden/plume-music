# Trying plume on your own gear

This is a 20-minute test for Windows and Linux. You need no programming background. You
copy a few lines into a terminal, play one short take, and send back what happened.

Your report says whether plume makes usable sound on a real computer like yours. "It did
not work, and here is what it said" is as useful as "it worked".

## What you need

- A computer running Windows 10 or 11, or a recent Linux desktop.
- Something to play into: the computer's own mic, or an instrument through an audio
  interface. Any make is fine.
- Somewhere for the band to come out: headphones or speakers. Wired is better than
  wireless, which arrives late.
- An instrument, or your hands to clap.

Plume runs on your computer. It sends nothing anywhere, and this test asks for no
recordings.

## Part 1: install (about 5 minutes)

Open a terminal.

- **Windows:** open the Start menu, type `PowerShell`, and press Enter.
- **Linux:** open the app called Terminal.

Copy each line below into it and press Enter. Wait for each one to finish before the next.

**1. Get the two helper tools.** `uv` installs plume. `git` fetches it.

Windows:

```
winget install --id Git.Git -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux (Debian, Ubuntu and their relatives):

```
sudo apt install git curl libportaudio2
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On another Linux, install `git`, `curl` and the PortAudio library with your own package
manager, then run the `curl` line.

**2. Close the terminal and open a new one.** The new one knows where the tools are.

**3. Install plume.** This is one long line. Copy all of it.

```
uv tool install --python 3.12 'plume-music[play,midi,notes] @ git+https://github.com/cyborg-garden/plume-music'
```

It takes a minute or two and ends with `Installed 1 executable: plume`.

**4. Check it.**

```
plume doctor
```

The last line says `all good`. If it says `problems found`, or the terminal says it does
not know `plume`, stop here and send back everything on the screen.

## Part 2: your devices (about 3 minutes)

**5. List your devices.**

```
plume inputs
```

Each line is a device: how many inputs it has, how many outputs, and its name. Copy the
whole list for your report. If a name has your own name in it, change that part first.

**6. Check your input opens.** Pick the device you play into, and put a word from its name
between the quotes:

```
plume inputs --meter "a word from the name"
```

Plume listens for 5 seconds and makes no sound. Play or clap while it does. It prints a
level for each channel and says which one is loudest.

If the levels print, your input opens at 48 kHz, the rate plume uses.

If this step prints an error instead, your device may be set to another rate. Copy the
error for your report. Then open your system's sound settings, or the device's own
control panel, set the device to 48000 Hz, and try the line again.

Windows may also ask whether apps can use the microphone. Say yes, and note that it
asked.

## Part 3: one take (about 8 minutes)

**7. Open the page.**

```
plume gui
```

A page opens in your browser. Leave this terminal open: the page needs it. Nothing plays
yet.

**8. Press Start and follow the page.** It says "Play something", then asks about
headphones. Answer what is true for you.

If the page says it could not open the device, copy its words for your report.

**9. Count in.** Knock or clap 8 even beats. With an instrument plugged straight in, play
the 8 beats on muted strings. The drums come in where the ninth beat would be.

**10. Play for two or three minutes.** Keep it simple and steady. The bass and chords
join about eight bars after you start. While you play, notice one thing above all: does
the band sound **on time** with you, **a little behind**, or **clearly behind**?

**11. Send one cue from a second terminal.** Leave everything running. Open one more
terminal, the same way as before, and run:

```
plume cue "busier"
```

The terminal prints a short reply. A few bars later the band plays more notes. Note
whether you heard the change.

**12. Stop from the page.** Press **Stop**. The band stops and the page shows the take
you just played. Press Play there if you want to hear it back.

**13. Close up.** Close the browser tab. Click the first terminal, hold Ctrl and press C.
It says `plume gui closed.`

If sound ever keeps going when it should not, run `plume stop` in any terminal.

## What to send back

Copy this, fill it in, and reply to whoever sent you this page.

```
System (for example Windows 11, or Ubuntu 24.04):
Laptop or desktop:
Device list from step 5 (names as shown, your own name removed):
Device you played into, and what was plugged into it:
Where the band came out (headphones or speakers, wired or wireless):
Step 4, plume doctor: all good / problems found
Step 6, the 5-second listen: worked / error (paste it)
Did you have to change the device to 48000 Hz: yes / no / could not
Step 8, Start: worked / the page said (paste its words)
Step 9, the drums came in after the count: yes / no
Step 10, the band felt: on time / a little behind / clearly behind
Step 11, the cue: reply printed (paste it) / error; heard the change: yes / no
Step 12, Stop from the page: worked / did not
Any warning or permission window from the system, a browser or an antivirus (which
step, and its words):
Anything else that surprised you:
```

## Removing plume

```
uv tool uninstall plume-music
```

Your takes stay where plume saved them. The page's Settings shows the folder.

## Second pass: the experimental download

Do this pass only when you are sent a download to try. It needs no terminal.

1. Unpack the download.
2. Double-click Plume inside it.
3. Watch for a warning before it opens. Windows may show a blue "Windows protected your
   PC" window, and an antivirus may hold the file. Write down the exact words, and what
   you pressed to carry on. If there is no way to carry on, stop and report that.
4. Press **Start**, count in, and play for a minute.
5. Press **Stop**.
6. Quit Plume. Check that no sound is still playing and nothing of Plume is left open.

Send back:

```
System:
Name of the download:
Warnings before it opened (exact words, and what you pressed):
Start: worked / the page said (paste its words)
Stop: worked / did not
Quit left nothing running: yes / no
The band felt: on time / a little behind / clearly behind
```
