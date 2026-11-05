"""Prompt -> Session composer. SPEC.md section 6.

parse_prompt() is rule-based and never errors on free text. compose() turns a
PromptSpec into a full Session: style blend, key/progression choice, track
construction from the style's voice map, pattern generation (drum grids,
bass/pad expansion against the progression, motif-engine melody), and an
energy-curve arrangement in 8-bar-multiple sections.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

from plume import figures
from plume.score import Movement, SectionSpec, Session, Track
from plume.styles_lib import load_style
from plume.theory import NOTE_NAMES, Chord, Key, parse_key, roman_to_chord, voice_chord

# ---------------------------------------------------------------------------
# prompt parsing


@dataclass
class PromptSpec:
    styles: list[str] = field(default_factory=list)  # ordered, first = primary
    bpm: float | None = None
    key: str | None = None
    movements: int = 3
    mood: list[str] = field(default_factory=list)
    name: str = "untitled"


# "house" is also just a word.  After a determiner or in a set phrase it is a building,
# unless a music word follows it: "my house" is not a style, "a house beat" is.
_HOUSE_NOT_AFTER = ("the", "a", "an", "my", "our", "your", "his", "her", "their", "this",
                    "that", "full", "haunted", "coffee", "tree", "safe", "trap")
_HOUSE_RE = (
    r"(?:\b(?:" + "|".join(_HOUSE_NOT_AFTER) + r")\s+house\b"
    r"(?=\s+(?:music|beat|track|groove|song|tune|jam|loop|vibes?|style)\b)"
    r"|" + "".join(rf"(?<!\b{w}\s)" for w in _HOUSE_NOT_AFTER) + r"\bhouse\b"
    r"(?![\s\-]+(?:of|party|plants?|cat|keys?|call|fire|rules|hold)\b))")
_STYLE_PATTERNS = [
    (re.compile(r"\b(?:midwest|emo)\b", re.I), "midwest"),
    (re.compile(r"\btrap\b", re.I), "trap"),
    (re.compile(r"\blo[\s\-]?fi\b", re.I), "lofi"),
    (re.compile(r"\bsynth[\s\-]?wave\b", re.I), "synthwave"),
    (re.compile(r"\bretrowave\b", re.I), "synthwave"),
    (re.compile(r"\boutrun\b", re.I), "synthwave"),
    (re.compile(r"\bjazz(?:y)?\b", re.I), "jazz"),
    (re.compile(r"\bhip[\s\-]?hop\b", re.I), "hiphop"),
    (re.compile(r"\bboom[\s\-]?bap\b", re.I), "hiphop"),
    (re.compile(_HOUSE_RE, re.I), "house"),
    (re.compile(r"\br\s*&\s*b\b|\brnb\b", re.I), "rnb"),
    (re.compile(r"\bneo[\s\-]?soul\b", re.I), "rnb"),
]
#: The styles a prompt could always name.  A prompt naming one of them keeps the meaning
#: it always had, whatever else it says: "lofi hip hop" is still lofi, at lofi's tempo.
_ORIGINAL_STYLES = frozenset({"midwest", "trap", "lofi", "synthwave"})
_MOOD_WORDS = ("chill", "dark", "eerie", "hopeful", "aggressive", "ambient",
               "dreamy", "mellow", "sad", "moody", "warm", "spooky")
_BPM_RE = re.compile(r"\b(\d{2,3}(?:\.\d+)?)\s*bpm\b", re.I)
# v2.1 AMENDMENT.  THE DEFECT: a tempo was silently dropped unless the literal
# token "bpm" followed the number — and BOTH lead-mode help texts documented
# examples that hit it.  `plume roles` printed
# `example: plume simlead take.wav --prompt 'chill trap at 120'` and that
# composed at 140; `plume lead --help` documented
# `PROMPT ... e.g. 'chill trap at 120, 3 movements'` and that composed at 140
# too.  Nothing in stdout, stderr or --json said the number had been ignored —
# the manifest's `composed_bpm` was the only tell, and only to someone who
# already suspected it.  A cold user follows the printed example and gets a
# tempo they did not ask for.
#
# So "at 120", "@120", "around 96", "tempo 140" are read as tempi.  Guarded on
# both sides: the number must be a musically possible tempo (this is what
# stops "at 3 movements" and a year), and it must not be the head of a phrase
# that means something else ("at 4 bars", "at 3 movements", "at 2 semitones").
_TEMPO_UNITS = r"(?:movements?|movement|bars?|beats?|semitones?|st|steps?|s|seconds?|db|hz|k)"
_TEMPO_AT_RE = re.compile(
    r"(?:\b(?:at|around|about|near|approx|tempo)\s*|@\s*)"
    r"(\d{2,3}(?:\.\d+)?)\b(?!\s*" + _TEMPO_UNITS + r"\b)", re.I)
#: The range a bare number is allowed to mean a tempo in.
TEMPO_MIN_BPM, TEMPO_MAX_BPM = 40.0, 300.0
_KEY_RE = re.compile(
    r"\b(?:in|key of)\s+([a-g][#b]?)\s+"
    r"(major|minor|maj|min|dorian|phrygian|harmonic[\s_-]minor)\b", re.I)
_MOV_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six)\s+(?:major\s+|big\s+)?movements?\b", re.I)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_NAME_STOPWORDS = {
    "a", "an", "and", "at", "beat", "bpm", "in", "major", "make", "me",
    "minor", "movement", "movements", "of", "over", "some", "the", "with",
}


def parse_prompt(text: str) -> PromptSpec:
    """Rule-based prompt parser. Unknowns -> None/defaults; never raises."""
    spec = PromptSpec()
    t = text or ""

    found: dict[str, int] = {}
    for rx, canon in _STYLE_PATTERNS:
        m = rx.search(t)
        if m and (canon not in found or m.start() < found[canon]):
            found[canon] = m.start()
    spec.styles = [c for c, _ in sorted(found.items(), key=lambda kv: kv[1])]
    if _ORIGINAL_STYLES & set(spec.styles):
        spec.styles = [c for c in spec.styles if c in _ORIGINAL_STYLES]
    # "lofi midwest emo" describes midwest emo with a lofi colour, not lofi.
    if "midwest" in spec.styles:
        spec.styles.remove("midwest")
        spec.styles.insert(0, "midwest")

    m = _BPM_RE.search(t)
    if m:
        spec.bpm = float(m.group(1))
    else:
        m = _TEMPO_AT_RE.search(t)
        if m and TEMPO_MIN_BPM <= float(m.group(1)) <= TEMPO_MAX_BPM:
            spec.bpm = float(m.group(1))

    m = _KEY_RE.search(t)
    if m:
        tonic = m.group(1)
        tonic = tonic[0].upper() + tonic[1:]
        mode = re.sub(r"[\s_-]+", " ", m.group(2).lower())
        spec.key = f"{tonic} {mode}"

    m = _MOV_RE.search(t)
    if m:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _WORD_NUM[raw]
        spec.movements = max(1, n)

    low = t.lower()
    spec.mood = [w for w in _MOOD_WORDS if re.search(rf"\b{w}\b", low)]

    words = [w for w in re.findall(r"[a-z0-9#]+", low)
             if w not in _NAME_STOPWORDS and not w.isdigit()]
    spec.name = "-".join(words[:4]) if words else "untitled"
    return spec


def describe_prompt(text: str) -> dict:
    """What :func:`parse_prompt` took from ``text``, and what it did not.

    v2.1 AMENDMENT.  THE DEFECT: the prompt path had **no feedback channel at
    all**.  ``plume simlead human.wav --prompt "asdfjkl qqq zzz"`` exited 0 and
    rendered a full session at the default style, tempo and key, with no
    warning and no ``ignored``/``understood`` field in ``--json``; combined
    with the dropped-tempo bug above there was no way to tell a
    fully-understood prompt from a completely-ignored one short of diffing the
    composed manifest against what you asked for.

    Returns ``{"understood": {field: value}, "ignored": [word, ...],
    "understood_any": bool}``.  ``ignored`` lists the content words that no
    rule claimed, which is what makes a typo visible.  Never raises: this is a
    reporting function on a parser that never raises either.
    """
    raw = "" if text is None else str(text)
    spec = parse_prompt(raw)
    understood: dict[str, object] = {}
    if spec.styles:
        understood["style"] = list(spec.styles)
    if spec.bpm is not None:
        understood["bpm"] = float(spec.bpm)
    if spec.key is not None:
        understood["key"] = str(spec.key)
    if _MOV_RE.search(raw):
        understood["movements"] = int(spec.movements)
    if spec.mood:
        understood["mood"] = list(spec.mood)

    claimed = set(_MOOD_WORDS)
    for rx, canon in _STYLE_PATTERNS:
        if canon not in spec.styles:
            continue
        for m in rx.finditer(raw):
            claimed.update(re.findall(r"[a-z0-9#]+", m.group(0).lower()))
    for rx in (_BPM_RE, _TEMPO_AT_RE, _KEY_RE, _MOV_RE):
        for m in rx.finditer(raw):
            claimed.update(re.findall(r"[a-z0-9#]+", m.group(0).lower()))
    ignored = [w for w in re.findall(r"[a-z0-9#]+", raw.lower())
               if w not in claimed and w not in _NAME_STOPWORDS
               and not w.isdigit() and len(w) > 1]
    return {"understood": understood, "ignored": ignored,
            "understood_any": bool(understood)}


# ---------------------------------------------------------------------------
# small musical helpers

_DRUM_PRESETS = {"808kick", "hat", "openhat", "clap", "snare", "crackle",
                 "kit_kick", "kit_snare", "kit_rim", "kit_hat", "kit_halfhat",
                 "kit_openhat", "kit_ride", "kit_crash", "kit_htom", "kit_ltom",
                 "housekick", "boomkick", "offhat", "brush"}
_SAD_TONICS = (("F", 3.0), ("A", 3.0), ("C", 2.0), ("D", 2.0), ("G", 1.0), ("E", 1.0))
_GAIN_DB = {"kick": 0.0, "snare": -2.0, "clap": -4.0, "hat": -8.0, "openhat": -9.0,
            "bass": -1.0, "pad": -7.0, "lead": -6.0, "arp": -9.0, "crackle": -18.0,
            "rolls": -10.0}
_PAN = {"hat": 0.25, "openhat": 0.2, "clap": -0.1, "lead": -0.15, "arp": 0.2,
        "rolls": -0.2}
_TIER1_ROLES = {"kick", "bass", "pad"}          # enter below energy 0.3 (sparse)
_TIER3_ROLES = {"snare", "clap", "lead"}        # enter above energy 0.6 (full)
_ALWAYS_ROLES = {"crackle"}                     # textural bed, always on


def _note_name(midi: int) -> str:
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def _alt(tokens: list[str]) -> str:
    """One token per bar of the progression -> single token or <...> group."""
    return tokens[0] if len(set(tokens)) == 1 else "<" + " ".join(tokens) + ">"


def _grid(events: list[tuple[int, int, str]], slots: int = 16) -> str:
    """(slot, dur, token) hits -> flat mini-notation grid using @ stretches."""
    out: list[str] = []
    i = 0
    for slot, dur, tok in events:
        while i < slot:
            out.append("~")
            i += 1
        out.append(tok + (f"@{dur}" if dur > 1 else ""))
        i += dur
    while i < slots:
        out.append("~")
        i += 1
    return " ".join(out)


def _thin(pattern: str, rng: np.random.Generator) -> str:
    """Sparse variant of a flat grid: keep the first hit, drop ~half the rest."""
    if "[" in pattern or "<" in pattern:
        return pattern
    toks = pattern.split()
    out: list[str] = []
    seen_hit = False
    for tok in toks:
        if tok == "~":
            out.append(tok)
        elif not seen_hit:
            out.append(tok)
            seen_hit = True
        else:
            out.append(tok if float(rng.random()) < 0.5 else "~")
    return " ".join(out)


def _merge_bars(bars: list[str]) -> str:
    """Per-bar variants sharing one rhythm -> single pattern via <...> alts."""
    if len(set(bars)) == 1:
        return bars[0]
    rows = [b.split() for b in bars]
    out: list[str] = []
    for i in range(len(rows[0])):
        toks = [r[i] for r in rows]
        out.append(toks[0] if len(set(toks)) == 1 else "<" + " ".join(toks) + ">")
    return " ".join(out)


def _raise_dominant(numeral: str) -> str:
    """dark mood: minor v -> major V (harmonic-minor dominant)."""
    core = numeral.lstrip("b")
    i = 0
    while i < len(core) and core[i] in "iIvV":
        i += 1
    if core[:i] == "v":
        return numeral.replace("v", "V", 1)
    return numeral


def _scale_midis(scale_pcs: list[int], lo: int, hi: int) -> list[int]:
    pcs = set(scale_pcs)
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def _nearest_in(pool: list[int], target: int) -> int:
    return min(pool, key=lambda m: (abs(m - target), m))


def _scale_step(midi: int, steps: int, scale_pcs: list[int], lo: int, hi: int) -> int:
    pool = _scale_midis(scale_pcs, lo - 24, hi + 24)
    idx = pool.index(_nearest_in(pool, midi))
    idx = max(0, min(len(pool) - 1, idx + steps))
    return pool[idx]


# ---------------------------------------------------------------------------
# motif engine


def make_motif(key: Key, chords: list[Chord], style_melody: dict,
               rng: np.random.Generator) -> list[str]:
    """1-bar motif on scale tones (chord tones on strong beats, range <= 1.5
    octaves, leaps > 6 st resolve stepwise opposite), then per-chord diatonic
    transposition across the progression. One mini-notation string per bar;
    all bars share the motif's rhythm."""
    grid = int(str(style_melody.get("rhythm_grid", "8")))
    density = float(style_melody.get("density", 0.4))
    octave = int(style_melody.get("octave", 5))
    scale = key.scale()
    center = 12 * (octave + 1) + key.tonic
    lo, hi = center - 9, center + 9          # 1.5-octave window
    strong = {0, grid // 2}

    onsets = [0]
    for s in range(1, grid):
        p = density * (1.35 if s % 2 == 0 else 0.65)
        if float(rng.random()) < p:
            onsets.append(s)

    base = chords[0]
    base_pcs = [(base.root + iv) % 12 for iv in base.intervals]
    cur = _nearest_in(_scale_midis(base_pcs, lo, hi), center)
    pitches: list[int] = []
    prev_leap = 0
    for idx, s in enumerate(onsets):
        if idx == 0:
            p = cur
        elif abs(prev_leap) > 6:
            p = _scale_step(cur, -1 if prev_leap > 0 else 1, scale, lo, hi)
        else:
            leap = float(rng.random()) < 0.15
            mag = int(rng.integers(3, 5)) if leap else int(rng.integers(1, 3))
            sign = 1 if float(rng.random()) < 0.5 else -1
            p = _scale_step(cur, sign * mag, scale, lo, hi)
        p = max(lo, min(hi, p))
        if s in strong:
            p = _nearest_in(_scale_midis(base_pcs, lo, hi), p)
        prev_leap = p - cur if idx else 0
        pitches.append(p)
        cur = p

    bars: list[str] = []
    for ch in chords:
        shift = ((ch.root - base.root + 6) % 12) - 6
        chord_pcs = [(ch.root + iv) % 12 for iv in ch.intervals]
        toks = ["~"] * grid
        for s, p in zip(onsets, pitches):
            q = _nearest_in(_scale_midis(scale, lo, hi), p + shift)
            if s in strong:
                q = _nearest_in(_scale_midis(chord_pcs, lo, hi), q)
            toks[s] = _note_name(q) + ("" if s in strong else ":v0.75")
        bars.append(" ".join(toks))
    return bars


# ---------------------------------------------------------------------------
# pattern builders per role


def _drum_pattern_set(role: str, style: dict, rng: np.random.Generator) -> dict[str, str]:
    variants = style["drum_patterns"][role]
    i = int(rng.integers(0, len(variants)))
    main = str(variants[i])
    alt = str(variants[(i + 1) % len(variants)])
    sparse = _thin(main, rng)
    fills = style.get("fills", {}).get(role)
    fill = str(fills[int(rng.integers(0, len(fills)))]) if fills else main
    return {"main": main, "alt": alt, "sparse": sparse, "fill": fill}


def _walking_pattern_set(chords: list[Chord],
                         rng: np.random.Generator) -> dict[str, str]:
    """One line per chord, each leaning into the chord after it, in quarter notes."""
    def line() -> str:
        bars = [figures.walking_line(ch.root, ch.intervals,
                                     chords[(i + 1) % len(chords)].root, rng)
                for i, ch in enumerate(chords)]
        return " ".join(_alt([_note_name(b[k]) for b in bars]) for k in range(4))

    main = line()
    roots = [_note_name(33 + ((ch.root - 33) % 12)) for ch in chords]
    fifths = [_note_name(33 + ((ch.root - 33) % 12)
                         + figures.degree_semitones(ch.intervals, 5))
              for ch in chords]
    sparse = f"{_alt(roots)} ~ {_alt(fifths)} ~"       # a two feel
    return {"main": main, "alt": line(), "sparse": sparse, "fill": main}


def _figure_bass_pattern_set(bass_style: str, chords: list[Chord],
                             octave: int) -> dict[str, str]:
    """A bass figure on the 16th grid, with its notes following the chords."""
    low = 24 + 12 * int(octave)
    events = []
    for beat, length, vel, degree in figures.BASS_FIGURES[bass_style]:
        notes = [_note_name(low + ch.root + figures.degree_semitones(ch.intervals, degree))
                 + f":v{vel}" for ch in chords]
        events.append((round(beat * 4), max(1, round(length * 4)), _alt(notes)))
    main = _grid(events)
    sparse = _grid([(0, 4, _alt([_note_name(low + ch.root) for ch in chords]))])
    return {"main": main, "alt": main, "sparse": sparse, "fill": main}


def _comp_layers(figure, voicings: list[list[int]], slots: int) -> str:
    """A chord figure as one layer per voice, since a pattern's note token is one note.

    A chord with fewer voices doubles its top one, rather than the layers disagreeing
    about how many notes there are.
    """
    per_beat = slots // 4
    layers = []
    for v in range(max(len(notes) for notes in voicings)):
        events = []
        for beat, length, vel in figure:
            toks = [_note_name(notes[min(v, len(notes) - 1)]) + f":v{vel}" for notes in voicings]
            events.append((round(beat * per_beat), max(1, round(length * per_beat)), _alt(toks)))
        layers.append(_grid(events, slots))
    return " , ".join(layers)


def _comp_pattern_set(pad_style: str, chords: list[Chord],
                      rng: np.random.Generator) -> dict[str, str]:
    """Jazz shells, neo soul stacks and house stabs.  Jazz sits on a triplet grid."""
    comps = figures.COMP_FIGURES[pad_style]
    slots = 12 if pad_style == "shell_comp" else 16
    if pad_style == "house_stabs":
        voicings = [[m + 12 for m in voice_chord(ch, low_midi=48)] for ch in chords]
    else:
        voicings = [figures.rootless_voicing(ch, shell=pad_style == "shell_comp")
                    for ch in chords]
    i = int(rng.integers(0, len(comps)))
    main = _comp_layers(comps[i], voicings, slots)
    alt = _comp_layers(comps[(i + 1) % len(comps)], voicings, slots)
    sparse = _comp_layers(comps[i][:1], voicings, slots)
    return {"main": main, "alt": alt, "sparse": sparse, "fill": main}


def _bass_pattern_set(bass_style: str, chords: list[Chord],
                      rng: np.random.Generator, *, octave: int = 0) -> dict[str, str]:
    if bass_style == figures.WALKING:
        return _walking_pattern_set(chords, rng)
    if bass_style in figures.BASS_FIGURES:
        return _figure_bass_pattern_set(bass_style, chords, octave)
    roots = [_note_name(24 + ch.root) for ch in chords]           # octave 1
    slides = [_note_name(24 + ch.root + 3) for ch in chords]      # slide from m3 above
    octs = [_note_name(36 + ch.root) for ch in chords]
    R, F, O = _alt(roots), _alt(slides), _alt(octs)

    def rv(v: float) -> str:
        return _alt([r + f":v{v}" for r in roots])

    if bass_style == "pump_eighths":
        main = " ".join([R if i % 2 == 0 else rv(0.85) for i in range(8)])
        sparse = " ".join([R, "~"] * 4)
        return {"main": main, "alt": main, "sparse": sparse, "fill": main}
    if bass_style == "roots_eighths":
        main = " ".join([R if i % 4 == 0 else rv(0.7) for i in range(8)])
        sparse = f"{R} ~ ~ ~ {R} ~ ~ ~"
        return {"main": main, "alt": main, "sparse": sparse, "fill": main}
    # sub_hits (trap 808): sparse hits with @ sustains and legato slide pairs
    templates = [
        [(0, 3, R), (6, 2, R), (10, 1, F), (11, 2, R), (14, 2, O)],
        [(0, 4, R), (7, 1, R), (10, 2, R), (13, 1, F), (14, 2, R)],
        [(0, 3, R), (5, 1, O), (8, 2, R), (12, 1, F), (13, 3, R)],
    ]
    i = int(rng.integers(0, len(templates)))
    main = _grid(templates[i])
    alt = _grid(templates[(i + 1) % len(templates)])
    sparse = _grid([(0, 4, R), (8, 4, R)])
    return {"main": main, "alt": alt, "sparse": sparse, "fill": alt}


def _pad_pattern_set(pad_style: str, chords: list[Chord],
                     rng: np.random.Generator) -> dict[str, str]:
    syms = [ch.sym for ch in chords]
    C = _alt(syms)
    if pad_style in figures.COMP_FIGURES:
        return _comp_pattern_set(pad_style, chords, rng)
    if pad_style == "arp16":
        return _arp_pattern_set(chords, rng)
    if pad_style == "stabs":
        templates = [
            [(0, 1, C), (7, 1, C), (10, 1, C)],
            [(3, 1, C), (7, 1, C), (11, 1, C), (14, 1, C)],
        ]
        i = int(rng.integers(0, len(templates)))
        main = _grid(templates[i])
        alt = _grid(templates[(i + 1) % len(templates)])
        return {"main": main, "alt": alt, "sparse": C, "fill": main}
    # held
    return {"main": C, "alt": C, "sparse": C, "fill": C}


def _arp_pattern_set(chords: list[Chord], rng: np.random.Generator,
                     octave: int = 4) -> dict[str, str]:
    low = 12 * (octave + 1)
    per_chord = [voice_chord(ch, low_midi=low) for ch in chords]

    def tok(i: int) -> str:
        alts = []
        for notes in per_chord:
            n = notes[i % len(notes)] + 12 * ((i // len(notes)) % 2)
            alts.append(_note_name(min(n, low + 31)))
        return _alt(alts)

    main = " ".join(tok(i) for i in range(16))
    sparse = " ".join(tok(2 * i) for i in range(8))
    return {"main": main, "alt": main, "sparse": sparse, "fill": main}


def _lead_pattern_set(motif_key: Key, chords: list[Chord], melody_cfg: dict,
                      rng: np.random.Generator) -> dict[str, str]:
    bars_main = make_motif(motif_key, chords, melody_cfg, rng)
    bars_alt = make_motif(motif_key, chords, melody_cfg, rng)
    grid = len(bars_main[0].split())
    strong = {0, grid // 2}
    bars_sparse = []
    for b in bars_main:
        toks = b.split()
        bars_sparse.append(" ".join(
            t if (i in strong or t == "~") else "~" for i, t in enumerate(toks)))
    return {"main": _merge_bars(bars_main), "alt": _merge_bars(bars_alt),
            "sparse": _merge_bars(bars_sparse), "fill": _merge_bars(bars_main)}


# ---------------------------------------------------------------------------
# track construction


def _build_track(role: str, preset: str, style: dict, chords: list[Chord],
                 motif_key: Key, melody_cfg: dict, rng: np.random.Generator,
                 spread: bool) -> Track:
    kind = "drum" if preset in _DRUM_PRESETS else "tonal"
    if role in style.get("drum_patterns", {}):
        patterns = _drum_pattern_set(role, style, rng)
    elif role == "crackle" or preset == "crackle":
        patterns = {"main": "x", "alt": "x", "sparse": "x", "fill": "x"}
    elif role == "bass":
        patterns = _bass_pattern_set(style["bass_style"], chords, rng,
                                     octave=int(style.get("bass_octave", 0)))
    elif role == "pad":
        patterns = _pad_pattern_set(style["pad_style"], chords, rng)
    elif role == "keys" and style.get("keys_style") in figures.COMP_FIGURES:
        patterns = _pad_pattern_set(style["keys_style"], chords, rng)
    elif role == "arp":
        patterns = _arp_pattern_set(chords, rng)
    elif role == "lead":
        patterns = _lead_pattern_set(motif_key, chords, melody_cfg, rng)
    else:  # unknown tonal role: hold the chords
        patterns = _pad_pattern_set("held", chords, rng)

    fx = {str(k): float(v) for k, v in style.get("fx_defaults", {}).get(role, {}).items()}
    params: dict[str, float | str] = {}
    if role == "pad" and spread:
        params["voicing"] = "spread"   # lofi voicing hint (engine may use it)
    gain_db = float(style.get("gain_db", {}).get(role, _GAIN_DB.get(role, -6.0)))
    return Track(name=role, voice=preset, kind=kind,
                 gain_db=gain_db,
                 pan=float(_PAN.get(role, 0.0)),
                 params=params, fx=fx, patterns=patterns)


def _plays_for_energy(energy: float, tracks: list[Track]) -> dict[str, str]:
    plays: dict[str, str] = {}
    for t in tracks:
        role = t.name
        if role in _ALWAYS_ROLES:
            plays[role] = "main"
        elif role in _TIER1_ROLES:
            plays[role] = "sparse" if energy < 0.3 else "main"
        elif role in _TIER3_ROLES:
            if energy > 0.6:
                plays[role] = "main"
        else:  # hats, keys, arps, rolls: the 0.3-0.6 tier
            if energy >= 0.3:
                plays[role] = "main"
    return plays


def _generic_template(n: int) -> list[dict]:
    n = max(1, n)
    out = []
    for m in range(n):
        frac = (m + 0.5) / n
        peak = 0.35 + 0.55 * max(0.0, 1.0 - abs(frac - 0.66) * 1.6)
        peak = min(0.9, max(0.3, peak))
        out.append({"name": f"movement-{m + 1}",
                    "energy_curve": [round(max(0.25, peak - 0.15), 3), round(peak, 3)],
                    "bars": 32})
    return out


def _stable_seed(spec: PromptSpec) -> int:
    payload = json.dumps(
        [spec.name, spec.styles, spec.bpm, spec.key, spec.movements, spec.mood],
        sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


# ---------------------------------------------------------------------------
# compose


def compose(spec: PromptSpec, *, seed: int | None = None) -> Session:
    """PromptSpec -> Session. Deterministic for a given (spec, seed)."""
    if seed is None:
        seed = _stable_seed(spec)
    seed = int(seed)
    rng = np.random.default_rng(seed)

    style_names = list(dict.fromkeys(spec.styles)) or ["lofi"]
    styles = [load_style(n) for n in style_names]
    primary, secondaries = styles[0], styles[1:]

    mood = [m.lower() for m in spec.mood]
    dark = bool({"dark", "eerie", "spooky", "moody"} & set(mood))
    chill = bool({"chill", "mellow"} & set(mood))
    aggressive = "aggressive" in mood
    ambient = bool({"ambient", "dreamy"} & set(mood))

    # 1-2. bpm / key / progression
    if spec.bpm:
        bpm = float(spec.bpm)
    else:
        mids = [(s["bpm_range"][0] + s["bpm_range"][1]) / 2.0 for s in styles]
        bpm = float(round(sum(mids) / len(mids)))

    if spec.key:
        key_str = spec.key
        key = parse_key(key_str)
    else:
        names = [n for n, _ in _SAD_TONICS]
        w = np.array([x for _, x in _SAD_TONICS], dtype=float)
        tonic_name = str(rng.choice(names, p=w / w.sum()))
        mode = str(primary.get("default_mode", "minor"))
        mode_word = {"minor": "min", "major": "maj"}.get(mode, mode)
        key_str = f"{tonic_name} {mode_word}"
        key = parse_key(key_str)

    # mode-aware progression pool: minor-family modes use the "minor" pool,
    # major uses "major" (fall back to whichever exists)
    pools = primary["progressions"]
    if isinstance(pools, dict):
        pool_name = "major" if key.mode == "major" else "minor"
        progs = pools.get(pool_name) or pools.get("minor") or pools.get("major")
    else:  # legacy flat list
        progs = pools
    pw = np.array([p.get("weight", 1) for p in progs], dtype=float)
    numerals = list(progs[int(rng.choice(len(progs), p=pw / pw.sum()))]["numerals"])
    if dark:
        numerals = [_raise_dominant(n) for n in numerals]
    chords = [roman_to_chord(n, key) for n in numerals]

    swing = float(primary.get("swing", 0.5))
    if chill:
        swing += 0.02
    swing = float(min(0.66, max(0.5, round(swing, 4))))

    melody_cfg = dict(primary.get("melody", {}))
    if chill or ambient:
        melody_cfg["density"] = float(melody_cfg.get("density", 0.4)) * 0.7
    motif_key = Key(key.tonic, "harmonic_minor") if (dark and key.mode == "minor") else key

    # 3. tracks from the primary style's voice map
    spread = primary["name"] == "lofi"
    tracks: list[Track] = []
    used_roles: set[str] = set()
    for role, preset in primary["voices"].items():
        if role in primary.get("live_only_roles", ()):
            continue
        tracks.append(_build_track(str(role), str(preset), primary, chords,
                                   motif_key, melody_cfg, rng, spread))
        used_roles.add(str(role))

    # style blend: secondary styles contribute their signature layer
    for s in secondaries:
        sname = s["name"]
        if sname == "synthwave" and "arp" not in used_roles:
            t = _build_track("arp", str(s["voices"].get("arp", "pluck")), s,
                             chords, motif_key, melody_cfg, rng, False)
            tracks.append(t)
            used_roles.add("arp")
        elif sname == "lofi" and "crackle" not in used_roles:
            t = _build_track("crackle", "crackle", s, chords,
                             motif_key, melody_cfg, rng, False)
            tracks.append(t)
            used_roles.add("crackle")
        elif sname == "trap" and "rolls" not in used_roles:
            pats = _drum_pattern_set("hat", s, rng)
            tracks.append(Track(name="rolls", voice=str(s["voices"].get("hat", "hat")),
                                kind="drum", gain_db=float(_GAIN_DB["rolls"]),
                                pan=float(_PAN["rolls"]), params={}, fx={},
                                patterns=pats))
            used_roles.add("rolls")

    # 5. mood: brightness / fx adjustments
    if dark:
        for t in tracks:
            if "lpf_hz" in t.fx:
                t.fx["lpf_hz"] = float(round(t.fx["lpf_hz"] * 0.75, 1))

    # master chain: primary wins, secondaries merge hints
    master = {
        "gain_db": 0.0,
        "reverb": float(primary["master"].get("reverb", 0.1)),
        "drive": float(primary["master"].get("drive", 0.15)),
        "lofi": float(primary["master"].get("lofi", 0.0)),
        "limit_db": -1.0,
    }
    for s in secondaries:
        m = s.get("master", {})
        master["reverb"] = max(master["reverb"], float(m.get("reverb", 0.0)))
        master["lofi"] = max(master["lofi"], float(m.get("lofi", 0.0)))
    if dark:
        master["reverb"] += 0.1
    if ambient:
        master["reverb"] += 0.15
    if aggressive:
        master["drive"] += 0.15
    master["reverb"] = float(round(min(1.0, master["reverb"]), 4))
    master["drive"] = float(round(min(1.0, master["drive"]), 4))

    # 4. arrangement: movements from the style template
    n_movements = max(1, int(spec.movements or 3))
    tmpl = primary.get("arrangement_templates", {}).get(str(n_movements))
    if tmpl is None:
        tmpl = _generic_template(n_movements)

    chord_syms = [ch.sym for ch in chords]
    movements: list[Movement] = []
    for mv in tmpl:
        curve = list(mv["energy_curve"])
        total = int(mv["bars"])
        per = max(8, (total // len(curve)) // 8 * 8)
        sections: list[SectionSpec] = []
        for i, e in enumerate(curve):
            e = float(e)
            if aggressive:
                e = max(e, 0.5)
            sections.append(SectionSpec(
                name=f"{mv['name']}-{chr(97 + i)}",
                bars=int(per),
                energy=float(round(min(1.0, max(0.0, e)), 3)),
                plays=_plays_for_energy(e, tracks),
                chords=list(chord_syms),
                fill=True,
            ))
        movements.append(Movement(name=str(mv["name"]), sections=sections))

    # outro: last section of last movement fades and thins to pad + crackle
    track_names = {t.name for t in tracks}
    outro = movements[-1].sections[-1]
    outro.name = outro.name + "-outro"
    outro.energy = float(min(outro.energy, 0.2))
    outro.plays = {n: "main" for n in ("pad", "crackle") if n in track_names}
    outro.fill = False

    return Session(
        name=spec.name or "untitled",
        bpm=float(bpm),
        key=str(key_str),
        swing=swing,
        seed=seed,
        style=str(primary["name"]),
        timing_bias_ms={str(k): float(v)
                        for k, v in primary.get("timing_bias_ms", {}).items()},
        humanize_ms=4.0,
        humanize_vel=0.08,
        tracks=tracks,
        movements=movements,
        master=master,
    )
