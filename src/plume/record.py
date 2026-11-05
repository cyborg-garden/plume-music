"""The Record: a take folder turned into sentences a player can read.

One job: rows, ``ending.json`` and the cue log in, events out.  Any of the three
may be missing, and zero rows is normal: a stop during the count leaves a folder
with a mic recording and nothing else.

THE RULE THIS MODULE EXISTS TO KEEP.  Once the band is in on a room mic plume
cannot hear the player, yet real rows still carry text that sounds as if it did:
``form.reason`` can say "human still mid-phrase", and ``role_reason``,
``role_events``, the human offsets and ``entry_reason`` come from the same
listening.  So this code works from an allowlist, field by field and value by
value.  A field added to the log later stays hidden until someone adds it here.
Nothing from those fields is copied into anything this module returns.

Evidence classes for the roast (plume GUI scope, section 6):

- A: plume's own actions (bar, bpm, entry, section, cues, ending).  Always fine.
- B: heard over drums alone (``playing`` in drums bars).  Gently, bars cited.
- C: direct input.  Not until it has been tested on a real interface.
- D: anything about the player on a room mic with the band in.  Never.

Pure apart from :func:`read_take`.  No audio, no device, no network.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
from bisect import bisect_right
from pathlib import Path

from .mind import CUE_BOUNDARIES, cue_is_exact, parse_cue
from .realtime import (DEFAULT_MAX_MINUTES, END_FORM, END_INTERRUPT, END_MAX_MINUTES, END_SILENCE,
                       END_SONG, END_STOPPED, END_TERMINATED, END_WEDGED,
                       REC_CUES, REC_DECISIONS, REC_ENDING, REC_HUMAN, REC_KEEP,
                       REC_MACHINE, REC_MANIFEST)
from .theory import _QUALITIES

#: Rows bigger than this are not a take this page can usefully draw.
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024

#: No real time, tempo or level is near this.  A log can hold any number at all.
_MAX_NUMBER = 1e12
_MAX_BAR = 10_000_000

#: Far past any take a person could cue.  The byte limits above still let a planted
#: log ask for minutes of work, so these limit the work.  A cue log over its limit
#: reads as no cue log: the cues still show, without who sent them or when.
MAX_CUE_LINES = 2000
MAX_CUE_EVENTS = 1000
MAX_CUES_PER_BAR = 16
MAX_CUE_CHARS = 200
_MAX_NOTES_PER_BAR = 64

MAX_NAME_CHARS = 120

#: The most a cue's mark can say and still fit on the sheet at its lowest room.  A
#: longer one reads "a cue, ...", so what happened to the cue is never the part cut off.
MAX_CUE_LABEL_CHARS = 24

_ENTRIES = ("drums", "band", "full")
_SECTIONS = ("intro", "verse", "lift", "break", "outro")
_KEY_RE = re.compile(r"[A-G][#b]? (?:major|minor)")
_CHORD_RE = re.compile(r"[A-G][#b]?(?:" + "|".join(sorted(map(re.escape, _QUALITIES), key=len, reverse=True))
                       + r")(?:/[A-G][#b]?)?")
_ROLE_RE = re.compile(r"[a-z_]{1,16}")
_STYLE_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_ACTION_RE = re.compile(r"[a-z_]{1,16}")
_CUE_NAME_RE = re.compile(r"[A-Za-z0-9' -]{1,32}")
_DENSITY_NOTE_RE = re.compile(r"density x(\d+(?:\.\d+)?) -> x(\d+(?:\.\d+)?)")
_TEMPO_NOTE_RE = re.compile(r"tempo x(\d+(?:\.\d+)?) -> x(\d+(?:\.\d+)?)")

#: How many roles a full band has, for the one quiet line.
_FULL_BAND_ROLES = 11

_END_SENTENCE = {
    END_STOPPED: "Stopped with plume stop. The last bar and the tail played out.",
    # Spoken "plume, stop" sends the same signal as Ctrl-C.
    END_INTERRUPT: "Cut short with Ctrl-C or a spoken stop.",
    END_TERMINATED: "Something outside plume told it to quit. It was not plume stop.",
    # This one is about hearing, so _ending_sentence swaps it once the band is in.
    END_SILENCE: "The mic went silent for 30 seconds, so I faded out.",
    END_SONG: "The length you set ran out, and the song wound down as planned.",
    END_MAX_MINUTES: "The time cap ended this take.",
    END_WEDGED: "The ending got stuck, and the hard deadline closed the take. "
                "You did not choose this.",
}
_END_SILENCE_ALONE = "I heard nothing from you for 30 seconds, so I faded out."
#: Once a direct input or keys have heard the player, the wait depends on the tempo.
_END_SILENCE_STOPPED = "I stopped hearing you, so I faded out."
#: What a cue to the looper asked for.  Fixed words: a loop's name can be anything
#: anyone typed, so the text itself is never echoed.
_LOOP_ASKS = {"record": "to record a loop", "close": "to close a loop take",
              "cancel": "to cancel a loop take", "stop": "to stop a loop",
              "mute": "to mute a loop", "unmute": "to bring a loop back",
              "drop": "to drop a loop", "swap": "to swap two loops",
              "list": "for the list of loops", "unclear": "for a loop"}
#: Only a cue sent for the next phrase waited for one.  With no log, nobody can say.
_END_CUE_LEAD = {"next-phrase": "A cue ended the take at a phrase line",
                 "next-section": "A cue ended the take at a section line",
                 "next-bar": "A cue ended the take without waiting for a phrase line",
                 "now": "A cue ended the take without waiting for a phrase line"}
_END_CUE_ROW = {"next-phrase": "The band stopped at this phrase line.",
                "next-section": "The band stopped at this section line.",
                "next-bar": "The band stopped here, without waiting for a phrase line.",
                "now": "The band stopped here, without waiting for a phrase line."}
_WAITED_FOR = {"next-phrase": "a phrase line", "next-section": "the next section",
               "next-bar": "the next bar", "now": "the next bar"}
_NOT_YET = {"next-phrase": "No phrase line has come yet.",
            "next-section": "The next section has not come yet.",
            "next-bar": "The next bar has not come yet.",
            "now": "The next bar has not come yet."}
_END_FORM_TOO_MANY = ("The take ended here. So many cues were logged that I can't say "
                      "whether one of them did it.")
_END_FORM_MAP = "The arrangement ran out and the take ended. You did not choose this."
_END_UNKNOWN = "This take ended for a reason this page does not know yet."
_NO_ENDING = ("No ending record. plume was killed or the machine stopped. "
              "The recording holds what reached the disk.")
_LIVE = "This take is still being recorded."
_TOO_BIG = "This take's log is too big for this page to read. It is still in the folder."

_END_LABEL = {
    END_STOPPED: "stopped", END_INTERRUPT: "cut", END_TERMINATED: "told to quit",
    END_FORM: "ended", END_SILENCE: "faded out", END_SONG: "wound down",
    END_MAX_MINUTES: "time cap", END_WEDGED: "hard deadline",
}


def _number(value) -> float | None:
    """A finite number, or None.  ``json.loads`` accepts NaN and bools count as int."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        # Bounded, so sums and rounding further down cannot overflow.  isfinite
        # itself raises on an int too big for a float.
        return float(value) if math.isfinite(value) and abs(value) <= _MAX_NUMBER else None
    except OverflowError:
        return None


def _is_row(doc) -> bool:
    return (isinstance(doc, dict) and isinstance(doc.get("bar"), int)
            and not isinstance(doc.get("bar"), bool) and 0 <= doc["bar"] <= _MAX_BAR
            and _number(doc.get("t_s")) is not None)


def _form(row: dict) -> dict:
    form = row.get("form")
    return form if isinstance(form, dict) else {}


def _entry(row: dict) -> str | None:
    value = row.get("entry")
    return value if value in _ENTRIES else None


def _section(row: dict) -> str | None:
    value = row.get("section")
    return value if value in _SECTIONS else None


def _key(row: dict) -> str | None:
    value = row.get("key")
    return value if isinstance(value, str) and _KEY_RE.fullmatch(value) else None


def _key_words(row: dict) -> str | None:
    """The key as a sentence says it: one plume worked out is never stated as a fact."""
    key = _key(row)
    if key and row.get("key_guess") is True:
        return f"{key} (a guess)"
    return key


def _style(row: dict) -> str | None:
    value = row.get("style")
    return value if isinstance(value, str) and _STYLE_RE.fullmatch(value) else None


def _tempo_from_loop(row: dict) -> bool:
    """A take can start from a loop's tempo with no count at all."""
    return row.get("tempo_from") == "loop"


def _fade(row: dict) -> float:
    return _number(row.get("fade_db")) or 0.0


def _too_many_cues(value) -> bool:
    """More cues on this row than :func:`_cue_texts` reads."""
    return isinstance(value, str) and value.count("; ") >= MAX_CUES_PER_BAR


def _cue_texts(value) -> list[str]:
    """Split the way the engine does (``_LeadCues.apply``), so both agree."""
    if not isinstance(value, str):
        return []
    parts = value.split("; ", MAX_CUES_PER_BAR)[:MAX_CUES_PER_BAR]
    return [text[:MAX_CUE_CHARS] for text in parts if text]


def cue_name(text) -> str | None:
    """A cue's words, if they are exactly a cue plume knows.

    Anything else is never echoed.  ``parse_cue`` finds a known word anywhere in the
    text, so "stop calling me that" parses as an ending, and it could be anything
    anyone typed.  Section names a map made up stay out too, since a name is all
    "go to" needs and a name could be a person's.
    """
    if not isinstance(text, str) or not _CUE_NAME_RE.fullmatch(text.strip()):
        return None
    if not cue_is_exact(text):
        return None
    return text.strip().lower()


def _cue_label(name: str | None, suffix: str) -> str:
    """A mark on the sheet.  The sentence below it always has the whole cue."""
    label = f"{name or 'a cue'}{suffix}"
    return label if len(label) <= MAX_CUE_LABEL_CHARS else f"a cue{suffix}"


def safe_row(row: dict | None, *, cue_names: bool = True) -> dict | None:
    """The log row a player may see: what plume did, on a list of safe keys.

    Naming a cue means parsing it, so ``cue_names`` lets a caller stop paying for that.
    """
    if not row:
        return None
    out: dict = {"bar": row.get("bar"), "t_s": _number(row.get("t_s"))}
    for name in ("bpm", "density", "gain_db", "fade_db"):
        value = _number(row.get(name))
        if value is not None:
            out[name] = value
    if _entry(row):
        out["entry"] = _entry(row)
    if _section(row):
        out["section"] = _section(row)
    if _key(row):
        out["key"] = _key(row)
    chord = row.get("chord")
    if isinstance(chord, str) and _CHORD_RE.fullmatch(chord):
        out["chord"] = chord
    roles = row.get("roles")
    if isinstance(roles, dict):
        out["roles"] = sorted(r for r in roles if isinstance(r, str) and _ROLE_RE.fullmatch(r))
    # Over drums alone this is pitch the player played.  After that it only means the
    # mic is not silent, so it does not leave this module.
    if _entry(row) == "drums" and isinstance(row.get("playing"), bool):
        out["playing"] = row["playing"]
    form = _form(row)
    action = form.get("action")
    safe_form: dict = {}
    # Tap-start plays its own song shape over the composed map, so there a jump or
    # an extend changes nothing the player can hear, and it may have come from the mic.
    # Only the ending is still something plume did.
    if isinstance(action, str) and _ACTION_RE.fullmatch(action) and (
            _entry(row) is None or action == "end"):
        safe_form["action"] = action
    for field in ("applied_cue", "dropped_cue") if cue_names else ():
        texts = _cue_texts(form.get(field))
        if texts:
            safe_form[field] = [cue_name(t) or "(not repeated)" for t in texts]
    if safe_form:
        out["form"] = safe_form
    return out


def _effects_for(row: dict, texts: list[str]) -> list[str | None]:
    """The renderer's note for each cue, in order.  Read for shape, never shown.

    The same cue twice on one bar has two notes.  Each note is used only once.
    """
    notes = row.get("cue_effects")
    left = ([n for n in notes[:_MAX_NOTES_PER_BAR] if isinstance(n, str)]
            if texts and isinstance(notes, list) else [])
    out: list[str | None] = []
    for text in texts:
        prefix = f"cue {text!r} "
        found = next((n for n in left if n.startswith(prefix)), None)
        if found is not None:
            left.remove(found)
        out.append(found)
    return out


def _jump_fates(row: dict, texts: list[str]) -> list[str | None]:
    """Did each cue move the form?  "moved", "ignored", or None if nothing says.

    Several cues share one row's action, so only the form's note on each cue can say
    which of them it belongs to.  Read for shape, never shown.
    """
    form = _form(row)
    reason = form.get("reason")
    if form.get("action") != "jump" or not isinstance(reason, str):
        return [None for _ in texts]
    start: dict[str, int] = {}
    out: list[str | None] = []
    for text in texts:
        prefix = f"cue {text!r} ("
        found = reason.find(prefix, start.get(text, 0))
        close = reason.find(")", found) if found >= 0 else -1
        if close < 0:
            out.append(None)
            continue
        start[text] = close
        out.append("moved" if reason.startswith(" -> section", close + 1) else "ignored")
    return out


def _density_heard(prev: dict | None, row: dict, effects: list[str | None]) -> dict[int, bool]:
    """Did each density cue change what was played?  By index into ``effects``.

    The renderer's scale can keep moving after the density it multiplies has hit
    its limits (``clamp(density * scale, 0.05, 1.0)``), and then the note reports
    a change nobody can hear.  The row holds the clamped result.
    """
    steps = {}
    for i, effect in enumerate(effects):
        found = _DENSITY_NOTE_RE.search(effect) if effect else None
        if found:
            steps[i] = (float(found.group(1)), float(found.group(2)))
    density = _number(row.get("density"))
    if not steps or density is None or min(min(v) for v in steps.values()) <= 0:
        return {i: True for i in steps}
    first_before = steps[min(steps)][0]
    final_after = steps[max(steps)][1]
    if 0.05 < density < 1.0:
        base = density / final_after
    else:
        # At a limit the row cannot say what the density would have been, so the
        # bar before has to.  With no bar before, there is no proof of a change.
        before = _number(prev.get("density")) if prev else None
        base = (before if before is not None else density) / first_before
        base = max(base, 1.0 / final_after) if density >= 1.0 else min(base, 0.05 / final_after)

    def played(scale: float) -> float:
        return min(1.0, max(0.05, base * scale))

    return {i: abs(played(a) - played(b)) > 1e-6 for i, (b, a) in steps.items()}


def _changed_nothing(effect: str | None) -> bool:
    """The note itself says so, and no later bar can change that."""
    if effect is None:
        return False
    found = _TEMPO_NOTE_RE.search(effect)
    return "NOT applied" in effect or bool(found and found.group(1) == found.group(2))


def _loop_held_tempo(row: dict) -> bool:
    """The engine moves its ratio but plays a loop's fixed tempo, and adds this note."""
    notes = row.get("cue_effects")
    return isinstance(notes, list) and any(
        isinstance(n, str) and n.startswith("tempo cue NOT applied yet")
        for n in notes[:_MAX_NOTES_PER_BAR])


def _outcome(row: dict, text: str, effect: str | None, *, next_row: dict | None = None,
             heard: bool = True, live: bool = False, fate: str | None = None) -> tuple[str, str]:
    """What a cue named in ``applied_cue`` really did, and a note we wrote.

    Being in that field proves nothing: the form appends every due cue before it
    checks the kind, unknown text included.  "Landed" needs a matching effect.
    """
    kind = parse_cue(text).get("kind")
    action = _form(row).get("action")
    if kind == "tempo" and _loop_held_tempo(row):
        return "held", ""
    if kind == "end":
        if action == "end":
            return "ended", ""
        # Another cue due on the same line can overwrite the action, but the end
        # cue has already set the ending, and the next row carries it out.
        if next_row is not None and _form(next_row).get("action") == "end":
            return "ended_next", ""
    # These two act on the next bar, and on a running take it may not exist yet.
    if live and next_row is None and kind in ("end", "tempo") and not _changed_nothing(effect):
        return "waiting", ""
    # Plain lead moves its form on these.  Tap-start plans its own song shape on
    # top, so there the same jump changes nothing the player can hear.
    if kind in ("energy", "section") and action == "jump" and _entry(row) is None:
        if fate == "moved":
            return "landed", ""
        if fate is None:
            return "unsure", ""
    if effect is None or "NOT applied" in effect:
        return "nothing", ""
    if "(texture:darker)" in effect or "(texture:brighter)" in effect:
        return "tone", ""
    # A tempo cue sets the next bar's tempo.  With no next bar, nothing sounded.
    if kind == "tempo" and next_row is None:
        return "nothing", ""
    for pattern, what in ((_DENSITY_NOTE_RE, "Density"), (_TEMPO_NOTE_RE, "Tempo")):
        found = pattern.search(effect)
        if found:
            if found.group(1) == found.group(2) or not heard:
                return "nothing", ""      # at the limit: no change the player could hear
            return "landed", f"{what} x{found.group(1)} to x{found.group(2)}."
    return "landed", ""


def _expired_after(row: dict, text: str) -> tuple[int | None, str | None]:
    """The bar count and boundary from the expiry note.  Nothing else leaves this."""
    reason = _form(row).get("reason")
    if not isinstance(reason, str):
        return None, None
    found = re.search(re.escape(f"cue {text!r} expired unapplied after ")
                      + r"(\d{1,5}) bars(?: \(ttl \d{1,5}, at ([a-z-]{1,16})\))?", reason)
    if not found:
        return None, None
    return int(found.group(1)), _boundary(found.group(2))


def _boundary(at) -> str | None:
    """One of the engine's cue boundaries, or None if nothing says which."""
    return at if at in CUE_BOUNDARIES else None


def _loop_answer(doc: dict) -> dict | None:
    """What the looper said to this cue, or None if it was a cue for the band."""
    loop = doc.get("loop")
    if not isinstance(loop, dict):
        return None
    op = loop.get("op")
    return {"op": op if isinstance(op, str) and op in _LOOP_ASKS else None,
            "ok": loop.get("ok") is True}


def _clean_cue_log(cue_log) -> list[dict] | None:
    if cue_log is None:
        return None
    out = []
    for doc in cue_log:
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
            continue
        bar = doc.get("bar")
        fine = isinstance(bar, int) and not isinstance(bar, bool) and 0 <= bar <= _MAX_BAR
        loop = _loop_answer(doc)
        if loop is not None:
            # The words stay out: they can name anything, and no row may claim this one.
            out.append({"text": "", "bar": bar if fine else None,
                        "t_s": _number(doc.get("t_s")), "source": doc.get("source"),
                        "at": None, "loop": loop})
            if len(out) > MAX_CUE_LINES:
                return None
            continue
        # One cue with "; " in it comes back from the form as several, so each part
        # has to be claimable by itself.
        for text in _cue_texts(doc["text"]) or [doc["text"][:MAX_CUE_CHARS]]:
            out.append({"text": text, "bar": bar if fine else None,
                        "t_s": _number(doc.get("t_s")), "source": doc.get("source"),
                        "at": _boundary(doc.get("at"))})
        if len(out) > MAX_CUE_LINES:
            return None
    return out


def _loop_sentence(cue: dict, at: str) -> str:
    """Only what the looper answered.  A take it agreed to can still fail to make a
    loop when it closes, and that is not in the cue log."""
    op, ok = cue["loop"]["op"], cue["loop"]["ok"]
    asked = f"{_who(cue['source'])} asked {_LOOP_ASKS.get(op, 'about a loop')} {at}."
    if not ok:
        return (f"{asked} The looper could not follow the words, so no take started."
                if op == "unclear" else f"{asked} The looper said no, so nothing changed.")
    return f"{asked} The looper said yes" + (", though that does not promise a loop came out of it."
                                             if op == "record" else ".")


def _who(source) -> str:
    # Only a person at the laptop or at the mic is "You".
    if source in ("gui", "voice"):
        return "You"
    if source == "cli":
        return "Another terminal"
    return "Someone"


def _row_clock(rows: list[dict]) -> tuple[list[float], list[int]]:
    """For :func:`_sounding_bar`, once per take.  A log's times need not be in order,
    so this keeps the latest time so far, which always is."""
    latest: list[float] = []
    for row in rows:
        t = float(row["t_s"])
        latest.append(max(latest[-1], t) if latest else t)
    return latest, [row["bar"] for row in rows]


def _sounding_bar(clock: tuple[list[float], list[int]], cue: dict) -> int | None:
    """The bar the player heard when asking.  Rows are written about a bar ahead, so
    the bar logged with a cue has often not sounded yet."""
    t = cue["t_s"]
    if t is None:
        return cue["bar"]
    latest, bars = clock
    at = bisect_right(latest, t)
    return bars[at - 1] if at else None


def _asked_at(bar: int | None) -> str:
    return "before the first bar" if bar is None else f"at bar {bar}"


def fmt_time(seconds: float) -> str:
    whole = int(max(0.0, seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def _length_s(rows: list[dict], ending: dict | None) -> float:
    elapsed = _number(ending.get("elapsed_s")) if isinstance(ending, dict) else None
    if elapsed is not None:
        return elapsed
    return float(rows[-1]["t_s"]) if rows else 0.0


def _end_cue_row(rows: list[dict]) -> tuple[dict | None, bool]:
    """The row that took an end cue, and whether too many cues hid the answer.

    From the last row back, because an end cue acts within a bar or so, so cues
    sent long before the end cannot use up the limit on reading them.
    """
    too_many = False
    read = 0
    for row in reversed(rows):
        applied = _form(row).get("applied_cue")
        texts = _cue_texts(applied)
        too_many = too_many or _too_many_cues(applied)
        if any(parse_cue(t).get("kind") == "end" for t in texts):
            return row, False
        read += len(texts)
        if read >= MAX_CUE_EVENTS:
            return None, True
    return None, too_many


def _cue_ending(rows: list[dict], at: str | None) -> str:
    """The sentence for a take that ended on its form."""
    if not rows or _form(rows[-1]).get("action") != "end":
        return _END_FORM_MAP
    row, too_many = _end_cue_row(rows)
    if row is None:
        return _END_FORM_TOO_MANY if too_many else _END_FORM_MAP
    fading = _fade(rows[-1]) < 0
    # The end cue may have landed a row early, with another cue on that line.
    gap = rows[-1]["bar"] - row["bar"]
    if gap > 0:
        bars = "one bar" if gap == 1 else f"{gap} bars"
        tail = "while it was fading." if fading else "with no fade."
        return f"A cue ended the take. The band stopped {bars} later, {tail}"
    lead = _END_CUE_LEAD.get(at, "A cue ended the take")
    return f"{lead}, while the band was fading." if fading else f"{lead}. The band stopped there, with no fade."


def _capped_sentence(ending: dict) -> str:
    """Only the length the cap was set to is known, not who set it."""
    minutes = _number(ending.get("max_minutes"))
    if minutes is None or minutes <= 0:
        return _END_SENTENCE[END_MAX_MINUTES]
    if minutes == DEFAULT_MAX_MINUTES:
        return f"The {minutes:g}-minute time cap ended this take. That is the default length."
    return f"This take had a {minutes:g}-minute time cap, set when it started, and it ran out."


def _ending_sentence(rows: list[dict], ending: dict | None, *, live: bool = False,
                     end_at: str | None = None, route: str = "unknown") -> str:
    if not isinstance(ending, dict):
        return _LIVE if live else _NO_ENDING
    reason = ending.get("reason")
    if reason == END_FORM:
        return _cue_ending(rows, end_at)
    if reason == END_SILENCE and _heard_them_stop(rows, route):
        return _END_SILENCE_STOPPED
    if reason == END_SILENCE and (_hears_player(route) or (rows and _entry(rows[-1]) == "drums")):
        return _END_SILENCE_ALONE
    if reason == END_MAX_MINUTES:
        return _capped_sentence(ending)
    if isinstance(reason, str) and reason in _END_SENTENCE:
        return _END_SENTENCE[reason]
    return _END_UNKNOWN


def _unused_words(at: str | None, *, live: bool, unread: bool) -> str:
    """Why a cue in the log has no row, as far as anyone can tell."""
    if unread:
        return "This log holds more cues than I read, so I can't say what happened to it."
    waited = _WAITED_FOR.get(at)
    if live:
        return _NOT_YET.get(at, "It has not been used yet.")
    return f"The take ended before {waited} came." if waited else "The take ended before it was used."


def _nothing_played(ending: dict | None, has_human: bool, *, live: bool = False,
                    route: str = "unknown", recorded_from: str | None = None) -> str:
    what = recorded_from or ("input" if route == "direct" else "mic")
    kept = (f"The {what} recording is still in the folder." if has_human
            else f"No {what} recording was kept.")
    if live and not isinstance(ending, dict):
        return f"{_LIVE} Nothing has played yet."
    if not isinstance(ending, dict):
        return f"No ending record, and nothing played. {kept}"
    if ending.get("reason") in (END_STOPPED, END_INTERRUPT):
        return f"Stopped before the first bar, so nothing played. {kept}"
    return f"{_ending_sentence([], ending, route=route)} Nothing played. {kept}"


def input_route(manifest) -> str:
    """How plume heard this take: midi, direct, room, or unknown for an older take.

    A ``--midi`` take listens to the keys whatever its audio route says, as the
    engine does.  Unknown gets the most cautious copy, the same as a room mic.
    """
    if not isinstance(manifest, dict):
        return "unknown"
    if manifest.get("midi_port"):
        return "midi"
    audio = manifest.get("input")
    source = audio.get("source") if isinstance(audio, dict) else None
    return source if source in ("direct", "room") else "unknown"


def recorded_from(manifest) -> str:
    """What ``human.wav`` was recorded from.  Not :func:`input_route`: a ``--midi`` take
    listens to the keys, and the file is still whichever input was open."""
    audio = manifest.get("input") if isinstance(manifest, dict) else None
    return "input" if isinstance(audio, dict) and audio.get("source") == "direct" else "mic"


def _hears_player(route: str) -> bool:
    """The band's own sound does not reach these, so silence there really is the player."""
    return route in ("direct", "midi")


def _heard_them_stop(rows: list[dict], route: str) -> bool:
    """The short wait applies once such an input has heard one bar of playing."""
    return _hears_player(route) and any(r.get("playing") is True for r in rows)


def translate(rows: list[dict], ending: dict | None = None,
              cue_log: list[dict] | None = None, *, has_human: bool = True,
              live: bool = False, unread: bool = False, route: str = "unknown",
              recorded: str | None = None) -> list[dict]:
    """Rows, ending and cue log in; events out, in bar order.

    ``live`` says the take is still recording, so a missing ending means nothing
    yet.  ``unread`` says the rows exist but were too big to read.  ``route`` is
    from :func:`input_route`, ``recorded`` from :func:`recorded_from`.

    Each event has ``bar``, ``t_s``, ``tier`` (plan, cue or end), ``label``,
    ``sentence``, ``note``, ``dashed`` and the filtered ``row``.  ``cue_outcome``
    and ``asked`` are for :func:`roast`, and :func:`public_event` drops them.
    """
    live = live and not isinstance(ending, dict)
    rows = [r for r in rows if _is_row(r)]
    log = _clean_cue_log(cue_log)
    clock = _row_clock(rows)
    cues_left = MAX_CUE_EVENTS
    used: set[int] = set()
    events: list[dict] = []
    shown_rows: dict[int, dict | None] = {}
    names_left = MAX_CUE_EVENTS

    def shown_row(row):
        """Every event on one row shares its filtered copy."""
        nonlocal names_left
        if id(row) not in shown_rows:
            shown_rows[id(row)] = safe_row(row, cue_names=names_left > 0)
            form = _form(row)
            names_left -= len(_cue_texts(form.get("applied_cue"))) + len(_cue_texts(form.get("dropped_cue")))
        return shown_rows[id(row)]

    def add(row, tier, label, sentence, *, note="", dashed=False, **extra):
        events.append({"bar": row["bar"], "t_s": float(row["t_s"]), "tier": tier,
                       "label": label, "sentence": f"Bar {row['bar']}. {sentence}",
                       "note": note, "dashed": bool(dashed), "row": shown_row(row), **extra})

    def claim(text: str, bar: int) -> dict | None:
        """Only the cue log can say when a cue was sent, and who sent it."""
        for i, cue in enumerate(log or ()):
            if i not in used and cue["text"] == text and (cue["bar"] is None or cue["bar"] <= bar):
                used.add(i)
                return cue
        return None

    prev_entry = prev_section = None
    heard = fading = False
    # Cues past the limits are not read, so the log's cues may have landed unseen.
    unread_cues = False
    end_at = None
    for i, row in enumerate(rows):
        entry, section = _entry(row), _section(row)
        bpm = _number(row.get("bpm"))
        speed = f", at {round(bpm)} bpm" if bpm is not None else ""
        if i == 0:
            if entry:
                add(row, "plan", "drums",
                    f"Drums came in on {'your loop' if _tempo_from_loop(row) else 'your count'}{speed}.")
            else:
                add(row, "plan", "start", f"The band started{speed}.")
        if entry == "drums" and not heard and row.get("playing") is True:
            heard = True
            # Three bars is what a room mic measured. A direct input or keys take less.
            add(row, "plan", "heard you",
                "I was sure I could hear you by here."
                + ("" if _hears_player(route) else " That takes me about three bars."))
        if entry and entry != prev_entry and i > 0:
            key = _key_words(row)
            if entry == "band":
                add(row, "plan", "bass and chords",
                    f"Bass and chords in, in {key}." if key else "Bass and chords in.")
            elif entry == "full":
                add(row, "plan", "melody", f"Melody in, over {key}." if key else "Melody in.")
        if (entry and section and section != prev_section and entry == prev_entry
                and ("intro" not in (section, prev_section) or section == "outro")
                and prev_section is not None):
            if section == "outro":
                # Not every style has a crash.  The row lists what this one has.
                roles = row.get("roles")
                crash = isinstance(roles, dict) and "crash" in roles
                add(row, "plan", "outro",
                    "The length you set ran out. Crash cymbal, then the kit thins out." if crash
                    else "The length you set ran out, and the kit thins out.")
            elif section == "break":
                add(row, "plan", "break",
                    "Break. Kick and snare drop out so the return reads as an event.")
            else:
                add(row, "plan", section, f"Section changes to {section}.")
        if entry:
            # With the band in, "heard" only ever meant "the mic is not silent".
            silent = _fade(row) < 0 and section != "outro"
            if silent and not fading:
                add(row, "plan", "fade started",
                    "I stopped hearing you, so I began to fade."
                    if _heard_them_stop(rows, route) else
                    "Nothing from you for 30 seconds, so I began to fade."
                    if entry == "drums" or _hears_player(route) else
                    "The mic went silent for 30 seconds, so I began to fade.")
            if not silent and fading and _fade(row) == 0:
                add(row, "plan", "came back",
                    "I heard you again, so the band came straight back."
                    if entry == "drums" or _hears_player(route) else
                    "The mic stopped being silent, so the band came straight back.")
            fading = silent
        form = _form(row)
        next_row = rows[i + 1] if i + 1 < len(rows) else None
        all_texts = _cue_texts(form.get("applied_cue"))
        texts = all_texts[:cues_left]
        cues_left -= len(texts)
        unread_cues = (unread_cues or len(texts) < len(all_texts)
                       or _too_many_cues(form.get("applied_cue")) or _too_many_cues(form.get("dropped_cue")))
        effects = _effects_for(row, texts)
        fates = _jump_fates(row, texts)
        density_heard = _density_heard(rows[i - 1] if i else None, row, effects)
        for n, text in enumerate(texts):
            kind, note = _outcome(row, text, effects[n], next_row=next_row,
                                  heard=density_heard.get(n, True), live=live, fate=fates[n])
            cue = claim(text, row["bar"])
            name = cue_name(text)
            shown = name or "a cue"
            who = _who(cue["source"]) if cue else None
            asked_bar = _sounding_bar(clock, cue) if cue else None
            at = _asked_at(asked_bar)
            asked = {"who": who, "bar": asked_bar, "name": name} if cue else None
            if kind == "landed":
                # A tempo cue sounds from the next bar.  This row's bpm is the old one.
                sounding = next_row if parse_cue(text).get("kind") == "tempo" and next_row else row
                if asked:
                    asked["landed_bar"] = sounding["bar"]
                wait = float(sounding["t_s"]) - cue["t_s"] if cue and cue["t_s"] is not None else None
                late = f" It landed {wait:.1f} seconds later." if wait is not None and wait >= 0 else ""
                sentence = (f"{who} asked for {shown} {at}.{late or ' It landed.'}" if cue
                            else f"A cue landed: {shown}.")
                add(sounding, "cue", _cue_label(name, " landed"), sentence, note=note,
                    cue_outcome="landed", asked=asked)
            elif kind == "held":
                sentence = (f"{who} asked for {shown} {at}. " if cue else f"A cue asked for {shown}. ")
                add(row, "cue", _cue_label(name, ", on hold"),
                    sentence + "A loop was holding the tempo, so it did not change here. "
                    "It waits until the last loop is dropped.",
                    dashed=True, cue_outcome="held", asked=asked)
            elif kind == "ended":
                end_at = cue["at"] if cue else None
                stopped = _END_CUE_ROW.get(end_at, "The band stopped here.")
                sentence = (f"{who} asked to end {at}. {stopped}"
                            if cue else f"A cue ended the take here. {stopped}")
                add(row, "cue", "cue ended it", sentence, cue_outcome="ended", asked=asked)
            elif kind == "ended_next":
                sentence = (f"{who} asked to end {at}. The band stopped one bar later." if cue
                            else "A cue ended the take from here. The band stopped one bar later.")
                add(row, "cue", "cue ended it", sentence, cue_outcome="ended", asked=asked)
            elif kind == "waiting":
                sentence = (f"{who} asked for {shown} {at}. " if cue else f"A cue asked for {shown}. ")
                add(row, "cue", _cue_label(name, ", waiting"),
                    sentence + "The next bar has not been written yet, so I can't say what it did.",
                    dashed=True, cue_outcome="waiting", asked=asked)
            elif kind == "tone":
                sentence = (f"{who} asked for {shown} {at}. " if cue else f"A cue asked for {shown}. ")
                add(row, "cue", _cue_label(name, ", no effect"),
                    sentence + "The log says the tone changed. I can't find that it did.",
                    dashed=True, cue_outcome="tone", asked=asked)
            elif kind == "unsure":
                prefix = f"{who} sent this {at}. " if cue else ""
                add(row, "cue", _cue_label(name, ", not sure"),
                    f"{prefix}The band jumped here, but the log does not say this cue did it.",
                    dashed=True, cue_outcome="unsure", asked=asked)
            else:
                said = (f'"{name}" was logged' if name
                        else "A cue I did not understand was logged" if parse_cue(text).get("kind") == "unknown"
                        else "A cue was logged")
                prefix = f"{who} sent this {at}. " if cue else ""
                add(row, "cue", _cue_label(name, ", did nothing"),
                    f"{prefix}{said}, and nothing acted on it.",
                    dashed=True, cue_outcome="nothing", asked=asked)
        dropped = _cue_texts(form.get("dropped_cue"))[:cues_left]
        cues_left -= len(dropped)
        for text in dropped:
            cue = claim(text, row["bar"])
            name = cue_name(text)
            shown = name or "a cue"
            after, waited = _expired_after(row, text)
            waited = waited or (cue["at"] if cue else None)
            lead = (f"{_who(cue['source'])} asked for {shown} {_asked_at(_sounding_bar(clock, cue))}. "
                    if cue else "")
            add(row, "cue", _cue_label(name, ", expired"),
                lead + "It ran out of time"
                + (f" after {after} bars" if after is not None else "")
                + f" before {_WAITED_FOR.get(waited, 'its moment')} came.",
                dashed=True, cue_outcome="expired",
                asked={"who": _who(cue["source"]) if cue else None, "bar": None, "name": name,
                       "at": waited})
        prev_entry, prev_section = entry, section

    # A cue the socket accepted that no row ever took: the take ended first.
    if rows:
        first_at: dict[int, dict] = {}
        for row in rows:
            first_at.setdefault(row["bar"], row)
        for i, cue in enumerate(log or ()):
            if i in used:
                continue
            bar = _sounding_bar(clock, cue)
            if "loop" in cue:
                # Answered when it arrived, so "the take ended first" would be false,
                # and one sent before the first bar belongs at the start, not the end.
                add(rows[0] if bar is None else first_at.get(bar, rows[-1]), "cue",
                    "loop cue" if cue["loop"]["ok"] else "loop cue, refused",
                    _loop_sentence(cue, _asked_at(bar)), dashed=not cue["loop"]["ok"],
                    cue_outcome="loop", asked={"who": _who(cue["source"]), "bar": bar, "name": None})
                continue
            row = first_at.get(bar, rows[-1])
            name = cue_name(cue["text"])
            suffix = ", not sure" if unread_cues else ", waiting" if live else ", too late"
            add(row, "cue", _cue_label(name, suffix),
                f"{_who(cue['source'])} asked for {name or 'a cue'} {_asked_at(bar)}. "
                + _unused_words(cue["at"], live=live, unread=unread_cues),
                dashed=True, cue_outcome="unused",
                asked={"who": _who(cue["source"]), "bar": bar, "name": name})

    label = "still recording" if live else "no record"
    if isinstance(ending, dict):
        reason = ending.get("reason")
        label = _END_LABEL.get(reason, "ended") if isinstance(reason, str) else "ended"
    if rows:
        last = rows[-1]
        add(last, "end", label,
            f"{_ending_sentence(rows, ending, live=live, end_at=end_at, route=route)} "
            f"{fmt_time(_length_s(rows, ending))} "
            + ("so far." if live else "in all."),
            dashed=not isinstance(ending, dict))
    else:
        # The looper answers when a cue arrives, so it can have answered with no bar played.
        # Unread rows may have held a first bar, so then "before" is not known.
        answered = [] if unread else [c for c in log or () if "loop" in c][:cues_left]
        for cue in answered:
            ok = cue["loop"]["ok"]
            events.append({"bar": 0, "t_s": 0.0, "tier": "cue",
                           "label": "loop cue" if ok else "loop cue, refused",
                           "sentence": _loop_sentence(cue, _asked_at(None)), "note": "",
                           "dashed": not ok, "row": None, "cue_outcome": "loop",
                           "asked": {"who": _who(cue["source"]), "bar": None, "name": None}})
        events.append({"bar": 0, "t_s": 0.0, "tier": "end",
                       "label": "log too big" if unread else label,
                       "sentence": _TOO_BIG if unread else _nothing_played(ending, has_human, live=live,
                                                                           route=route, recorded_from=recorded),
                       "note": "", "dashed": not isinstance(ending, dict), "row": None})
    # Stable, so events on one bar keep the order they happened in.
    events.sort(key=lambda e: e["bar"])
    for n, event in enumerate(events):
        event["id"] = f"ev-{n}"
    return events


def public_event(event: dict) -> dict:
    return {k: v for k, v in event.items() if k not in ("cue_outcome", "asked")}


def _first(rows: list[dict], test) -> dict | None:
    return next((r for r in rows if test(r)), None)


def roast(rows: list[dict], ending: dict | None = None, cue_log: list[dict] | None = None,
          *, events: list[dict] | None = None, route: str = "unknown") -> list[dict]:
    """Class A and B lines only, in a fixed order, at most five.

    Since plume cannot hear the player for most of a take, it mostly roasts itself about
    that.  Each line carries the bar it came from and that row, filtered.
    """
    rows = [r for r in rows if _is_row(r)]
    if not rows:
        return []
    if events is None:
        events = translate(rows, ending, cue_log, route=route)
    lines: list[dict] = []

    def event_id(row: dict, *labels: str) -> str | None:
        """The event this line is about, so its link opens that and not a neighbour."""
        return next((e.get("id") for e in events
                     if e["bar"] == row["bar"] and e["label"] in labels), None)

    def say(text: str, row: dict, *labels: str) -> None:
        lines.append({"text": text, "bar": row["bar"], "row": safe_row(row),
                      "event": event_id(row, *labels)})

    band = _first(rows, lambda r: _entry(r) == "band")
    tap = _entry(rows[0]) is not None
    # About three bars of confirmation are a room mic's lag, so they are never the player's fault.
    heard = _first(rows, lambda r: _entry(r) == "drums" and r.get("playing") is True)
    hears = _hears_player(route)
    if band and hears:
        # Evidence class C: true that it could hear, and still no verdict from it.
        say(f"Band came in at bar {band['bar']}. I could still hear you, but I keep it to "
            "myself until this input has been tried for real.", band, "bass and chords", "drums", "start")
    elif band:
        say(f"Band came in at bar {band['bar']}. From here on I have no idea what you did. "
            "Hope it was good.", band, "bass and chords", "drums", "start")
    elif isinstance(ending, dict) and ending.get("reason") == END_SILENCE and tap and not heard:
        last = rows[-1]
        alone = round(float(last["t_s"]) - float(rows[0]["t_s"]))
        # A loop means the player played and it may still be sounding, so they did come in,
        # and nothing was alone.
        asked = any("loop" in c and c["loop"]["ok"] and c["loop"]["op"] == "record"
                    for c in _clean_cue_log(cue_log) or ())
        say(f"Your loop set the tempo, and I never heard you after it. I played for {alone} "
            "seconds, then left." if _tempo_from_loop(rows[0]) else
            f"You asked for a loop, and I was never sure I heard you. I played for {alone} "
            "seconds, then left." if asked else
            f"You never came in. I played drums alone for {alone} seconds, then left.", last,
            _END_LABEL[END_SILENCE])
    # A direct input or keys are a verdict on the playing (class C), and the three bars
    # are not theirs. Counted from the first bar, which --from-bar moves.
    waited = heard["bar"] - rows[0]["bar"] if heard and not hears else 0
    if waited >= 8:
        say(f"You let the drums run about {waited - 3} bars before you came in. "
            "I took off the three I need to be sure.", heard, "heard you")
    bpm = _number(rows[0].get("bpm"))
    if tap and bpm is not None:
        looped = _tempo_from_loop(rows[0])
        set_by = "Your loop set" if looped else "You counted"
        speeds = [v for v in (_number(r.get("bpm")) for r in rows) if v is not None]
        held = max(speeds) - min(speeds) <= 0.01 * bpm
        minutes = (float(rows[-1]["t_s"]) - float(rows[0]["t_s"])) / 60.0
        span = "over a minute" if minutes < 1.5 else f"{round(minutes)} minutes"
        start = ("drums", "start")
        if not held:
            # The tempo can also move with no cue, under the meet tempo policy.
            moved = any(e.get("cue_outcome") == "landed" and "Tempo" in (e.get("note") or "")
                        for e in events)
            say(f"{set_by} {round(bpm)} bpm. Then a cue moved it, and I went along." if moved
                else f"{set_by} {round(bpm)} bpm. I did not stay there: I moved between "
                     f"{min(speeds):.1f} and {max(speeds):.1f}.", rows[0], *start)
        elif round(bpm) < 70 and minutes >= 1.0:
            say(f"{set_by} {round(bpm)} bpm. That is a very brave tempo and I held it for {span}.",
                rows[0], *start)
        else:
            how = "I played exactly that." if max(speeds) == min(speeds) else "I stayed within 1% of that."
            say(f"{set_by} {round(bpm)} bpm. {how} "
                f"If it feels wrong, that was your {'loop' if looped else 'count'}.", rows[0], *start)
    if cue_log is not None:
        for event in events:
            outcome, asked = event.get("cue_outcome"), event.get("asked")
            if not asked or not event.get("row") or not asked.get("name"):
                continue
            who, name = asked.get("who") or "Someone", asked["name"]
            got = asked.get("landed_bar", event["bar"])
            if outcome == "landed" and asked.get("bar") is not None and got - asked["bar"] >= 2:
                lines.append({"text": f"{who} asked for {name} at bar {asked['bar']}. I got there at "
                                      f"bar {got}. A real drummer would have sighed too.",
                              "bar": event["bar"], "row": event["row"], "event": event.get("id")})
            elif outcome == "expired":
                lines.append({"text": f"Somebody asked for {name}. I waited for "
                                      f"{_WAITED_FOR.get(asked.get('at'), 'its moment')}, "
                                      "ran out of time and dropped it.",
                              "bar": event["bar"], "row": event["row"], "event": event.get("id")})
            elif outcome == "nothing":
                lines.append({"text": f"Somebody asked for {name}. I wrote it down and did nothing "
                                      "about it.",
                              "bar": event["bar"], "row": event["row"], "event": event.get("id")})
    if band and not hears:
        say("Did you drag? I can't tell on this mic. Neither of us will ever know.", band,
            "bass and chords", "drums", "start")
    return lines[:5]


def band_line(rows: list[dict]) -> list[dict]:
    """How much band plume planned, per bar: the one quiet line on the sheet.

    Roles are always 1.0 and density comes from a fixed arc plus cues, so this is
    mostly the same shape in every take.  It is a timeline, not a fingerprint.
    """
    out = []
    for row in rows:
        if not _is_row(row):
            continue
        roles = row.get("roles")
        count = len(roles) if isinstance(roles, dict) else 0
        density = _number(row.get("density")) or 0.0
        level = (min(1.0, count / _FULL_BAND_ROLES) * (0.55 + 0.45 * min(1.0, max(0.0, density)))
                 * 10.0 ** (min(0.0, _fade(row)) / 40.0))
        out.append({"bar": row["bar"], "t_s": float(row["t_s"]), "level": round(level, 4)})
    return out


def portrait(rows: list[dict], ending: dict | None, events: list[dict], seed: int) -> dict:
    """The stone's shape, from what differs between takes.  The seed is identity."""
    rows = [r for r in rows if _is_row(r)]
    minutes = _length_s(rows, ending) / 60.0
    bpm = _number(rows[0].get("bpm")) if rows else None
    band = _first(rows, lambda r: _entry(r) == "band")
    cut = None
    if band is not None:
        cut = max(0.14, min(0.82, rows.index(band) / max(1, len(rows))))
    key = _key(band) if band else next((_key(r) for r in rows if _key(r)), None)
    return {
        "seed": int(seed) & 0x7FFFFFFF,
        "height": round(0.42 + 0.56 * min(1.0, minutes / 5.0), 4),
        "lean": round(max(-0.06, min(0.2, (90.0 - (bpm if bpm is not None else 90.0)) / 170.0)), 4),
        "cut": None if cut is None else round(cut, 4),
        "minor": bool(key and key.endswith("minor")),
        "nicks": min(8, sum(1 for e in events if e.get("cue_outcome") == "landed")),
    }


# --------------------------------------------------------------- the take folder

class _TooBig(Exception):
    """The file is real and bigger than this page will read."""


def _read_bytes(path: Path, limit: int) -> bytes | None:
    """A regular file's bytes, or None.  Raises :class:`_TooBig` over ``limit``.

    Anything can be in a take folder.  A symlink could point anywhere, a FIFO would
    block this thread for ever, and a device reports size 0 and never ends, so this
    refuses all three before it reads, and reads no more than the limit.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_size > limit:
            raise _TooBig
        chunks = []
        left = limit + 1
        while left > 0:
            chunk = os.read(fd, min(left, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        if left <= 0:
            raise _TooBig
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


def _read_ndjson(path: Path, *, keep) -> list | None:
    """Every whole, wanted line.  None when there is no such regular file."""
    data = _read_bytes(path, MAX_LOG_BYTES)
    if data is None:
        return None
    out = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            continue                  # a kill can leave half a line
        if keep(doc):
            out.append(doc)
    return out


def _read_json(path: Path) -> dict | None:
    try:
        data = _read_bytes(path, MAX_JSON_BYTES)
        doc = None if data is None else json.loads(data.decode("utf-8", errors="replace"))
    except (_TooBig, ValueError, RecursionError):
        return None
    return doc if isinstance(doc, dict) else None


def _is_regular(path: Path) -> bool:
    """Without following a symlink, which is not part of the take."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def is_take_dir(path: Path) -> bool:
    return any(_is_regular(path / name) for name in
               (REC_MANIFEST, REC_DECISIONS, REC_ENDING, REC_HUMAN))


def clean_name(name) -> str:
    """A take's name: printable, one line, bounded."""
    if not isinstance(name, str):
        return ""
    return "".join(c for c in name if c.isprintable()).strip()[:MAX_NAME_CHARS]


def read_keep(take_dir: Path) -> dict:
    doc = _read_json(take_dir / REC_KEEP) or {}
    return {"kept": doc.get("kept") is True, "name": clean_name(doc.get("name"))}


def _audio_size(path: Path) -> int | None:
    """Size of a regular file in the folder.  A symlink is not part of the take."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return info.st_size if stat.S_ISREG(info.st_mode) else None


def read_take(take_dir, *, fallback_seed: int = 0, live: bool = False) -> dict:
    """Everything The Record shows about one folder.  Never raises on bad contents.

    ``live`` says a running session is recording here.  The caller knows; this does not look.
    """
    take_dir = Path(take_dir)
    unread = False
    try:
        rows = _read_ndjson(take_dir / REC_DECISIONS, keep=_is_row) or []
    except _TooBig:
        rows, unread = [], True
    ending = _read_json(take_dir / REC_ENDING)
    try:
        cue_log = _read_ndjson(take_dir / REC_CUES, keep=lambda d: isinstance(d, dict))
    except _TooBig:
        cue_log = None
    manifest = _read_json(take_dir / REC_MANIFEST) or {}
    seed = manifest.get("seed")
    seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else fallback_seed
    sizes = {"human": _audio_size(take_dir / REC_HUMAN),
             "machine": _audio_size(take_dir / REC_MACHINE)}
    keep = read_keep(take_dir)
    args = (take_dir, ending, manifest, seed, sizes, keep, live, unread)
    try:
        return _take_doc(rows, cue_log, *args)
    except (ArithmeticError, ValueError, TypeError, KeyError, RecursionError):
        # One bad take must not take the shelf down with it.
        return _take_doc([], None, *args)


def _take_doc(rows, cue_log, take_dir, ending, manifest, seed, sizes, keep, live, unread) -> dict:
    live = live and not isinstance(ending, dict)
    route = input_route(manifest)
    events = translate(rows, ending, cue_log, has_human=sizes["human"] is not None,
                       live=live, unread=unread, route=route, recorded=recorded_from(manifest))
    lines = roast(rows, ending, cue_log, events=events, route=route)
    started = _number(manifest.get("started_at"))
    bpm = _number(rows[0].get("bpm")) if rows else None
    key_row = next((r for r in rows if _key(r)), None)
    key = None if key_row is None else _key(key_row)
    # The last one: a style picked as the band comes in replaces the one from the count.
    style_row = next((r for r in reversed(rows) if _style(r)), None)
    tap = bool(rows) and _entry(rows[0]) is not None
    band = _first(rows, lambda r: _entry(r) == "band")
    if unread:
        roast_empty = "This log is too big for me to read, so I have nothing on you."
    elif rows:
        roast_empty = "I have nothing on you this time."
    elif live:
        roast_empty = "Nothing has played yet, so I have nothing on you."
    else:
        roast_empty = "Nothing played, so I have nothing on you."
    return {
        "folder": take_dir.name,
        "kept": keep["kept"],
        "name": keep["name"],
        "started_at": started,
        "key": key,
        "key_guess": key_row is not None and key_row.get("key_guess") is True,
        "style": None if style_row is None else _style(style_row),
        "style_guess": style_row is not None and style_row.get("style_guess") is True,
        "bpm": None if bpm is None else round(bpm),
        "seconds": round(_length_s(rows, ending), 2),
        "bars": len(rows),
        "bytes": sum(v for v in sizes.values() if v),
        "audio": sorted(name for name, size in sizes.items() if size is not None),
        "has_cue_log": cue_log is not None,
        "tap_start": tap,
        "live": live,
        "unread": unread,
        "route": route,
        "recorded_from": recorded_from(manifest),
        "band_t_s": None if band is None else float(band["t_s"]),
        "start_t_s": float(rows[0]["t_s"]) if rows else None,
        "end_t_s": float(rows[-1]["t_s"]) if rows else None,
        "portrait": portrait(rows, ending, events, seed),
        "events": [public_event(e) for e in events],
        "roast": lines,
        "roast_empty": roast_empty,
        "band": band_line(rows),
    }


def summary(take: dict) -> dict:
    """The shelf's view of :func:`read_take`."""
    names = ("folder", "kept", "name", "started_at", "key", "key_guess", "style", "style_guess",
             "bpm", "seconds", "bars",
             "bytes", "portrait")
    return {name: take[name] for name in names}
