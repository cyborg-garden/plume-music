"""The Story: one ordered list of true events for a take.

One job: decision rows and cue rows in, events out.  The same function reads a
running take (rows held in memory) and a finished one (``decisions.ndjson`` and
``cues.ndjson``), so the page and an agent read one list.  The design is in
``docs/design/story-feed.md``.

THE RULE.  Every event names the fact in its input that it stands on (``facts``),
and its ``state`` says which kind of fact that is:

- ``received``: a cue row exists.  The socket writes one only for a cue it took.
- ``applied``: a decision row carries the cue in ``form.applied_cue``.
- ``dropped``: a decision row carries the cue in ``form.dropped_cue``.
- ``not_understood``: the cue parser's own reply to the cue's words.
- ``answered``: the cue row carries a reply given at once (a loop, or a sound).

Reaching the band is not the same as changing it, so an applied event also has an
``outcome``, and only ``landed`` and ``ended`` say the music changed.  What a cue
did is decided by :func:`plume.record.translate`, the reader the take page
already trusts.  This module pairs each of its cue events with a fact and adds a
short sentence.  A cue's own words are shown only when they are exactly a cue
plume knows (:func:`plume.record.cue_name`).

Pure.  No file, no audio, no device, no network, no clock.
"""
from __future__ import annotations

from . import record
from .inputfx import LABELS as _SOUND_LABELS
from .mind import parse_cue

#: What a cue event may claim.  Each is tied to a fact; see the module docstring.
STATES = ("received", "applied", "dropped", "not_understood", "answered")

#: What an applied cue did.  Only the first two say the music changed.
OUTCOMES = ("landed", "ended", "held", "waiting", "tone", "unsure", "nothing")

#: A sender's name is a claim, and it is cut to this many characters.
MAX_SENDER_CHARS = 24

#: Words plume uses for a sender it knows.  A claimed name may not be one of them.
_OWN_WORDS = ("you", "i", "someone", "another terminal", "plume", "a cue")

#: Where a queued cue waits.  Intent only: it says nothing about what happens next.
_INTENT = {"next-phrase": "Queued for the next phrase.",
           "next-section": "Queued for the next section.",
           "next-bar": "Queued for the next bar.",
           "now": "Queued for the next bar."}

#: translate's plan events that speak here, and the row field each stands on.  The
#: band's key, chords and style are not in this list, so those sentences stay out.
_BAND_FIELDS = {"drums": "entry", "start": "bar", "bass and chords": "entry",
                "melody": "entry", "fade started": "fade_db", "came back": "fade_db",
                "band followed": "follow"}
#: Events that are only true of a take's first row.
_FIRST_ROW_ONLY = ("drums", "start")
#: Said in this module's words, because translate's sentence names the key.
_BAND_OWN = {"bass and chords": "Bass and chords in, from bar {bar}.",
             "melody": "Melody in, from bar {bar}."}

#: translate's outcome for a cue a row applied, and the end of the label it gives it.
#: The label is the only place its event names a cue nobody is on record as sending.
_APPLIED_LABELS = {"landed": " landed", "held": ", on hold", "ended": None, "ended_next": None,
                   "waiting": ", waiting", "tone": ", no effect", "unsure": ", not sure",
                   "nothing": ", did nothing"}


def clean_sender(value) -> str | None:
    """A claimed sender name, safe to show as text, or None.

    Free text: printable characters only, runs of space made one, trimmed, and cut
    to :data:`MAX_SENDER_CHARS`.  Any local program can claim any name, so this
    makes a name showable and proves nothing about who sent the cue.  One of
    plume's own words for a sender ("You", "Someone") is not a name.
    """
    if not isinstance(value, str):
        return None
    kept = "".join(" " if ch.isspace() else ch for ch in value[:8 * MAX_SENDER_CHARS]
                   if ch.isspace() or ch.isprintable())
    name = " ".join(kept.split())[:MAX_SENDER_CHARS].rstrip()
    if not name or name.lower() in _OWN_WORDS:
        return None
    return name


def _who(doc: dict, cue: dict) -> tuple[str, str | None]:
    """(the word for who sent this, the claimed name if there is one)."""
    sender = clean_sender(doc.get("sender"))
    return sender or record._who(cue["source"]), sender


def _seqs(docs: list) -> list[int]:
    """A steady number for each cue row: its own ``seq`` if every row has a
    different one, else its place in the list."""
    own = [d.get("seq") if isinstance(d, dict) else None for d in docs]
    fine = all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in own)
    return own if fine and len(set(own)) == len(own) else list(range(len(docs)))


def _clean_cues(cues) -> tuple[list, list[dict], list[int]]:
    """(cue rows, the parts record reads from them, which row each part came from).

    A cue log past record's limits reads as no cue log there, so it does here too.
    """
    docs = list(cues) if isinstance(cues, (list, tuple)) else []
    clean = record._clean_cue_log(docs)
    if clean is None:
        return [], [], []
    origin: list[int] = []
    for i, doc in enumerate(docs):
        origin += [i] * len(record._clean_cue_log([doc]) or ())
    if len(origin) != len(clean):       # cannot happen; never guess whose cue it is
        return [], [], []
    return docs, clean, origin


def _claims(rows: list[dict], clean: list[dict]) -> list[tuple[int, str, int, str, int | None]]:
    """Each cue a row names: (row, field, place in the field, text, cue part or None).

    The same walk as ``record.translate``: rows in order, applied before dropped,
    and each cue part claimed once by the first row that names its text.
    """
    used: set[int] = set()
    left = record.MAX_CUE_EVENTS
    out = []

    def claim(text: str, bar: int) -> int | None:
        for k, cue in enumerate(clean):
            if k not in used and cue["text"] == text and (cue["bar"] is None or cue["bar"] <= bar):
                used.add(k)
                return k
        return None

    for j, row in enumerate(rows):
        form = record._form(row)
        for field in ("applied_cue", "dropped_cue"):
            texts = record._cue_texts(form.get(field))[:left]
            left -= len(texts)
            for n, text in enumerate(texts):
                out.append((j, field, n, text, claim(text, row["bar"])))
    return out


def _pair(claim, told: list[dict], taken: set[int], rows: list[dict], clean: list[dict],
          clock) -> dict | None:
    """translate's event for one claimed cue, or None if it has none.

    Two events this cannot tell apart are about the same cue words, from the same
    sender, asked at the same bar, on the same row, so either is true of either.
    """
    j, field, _, text, k = claim
    row = rows[j]
    after = rows[j + 1] if j + 1 < len(rows) else None
    name = record.cue_name(text)
    who = record._who(clean[k]["source"]) if k is not None else None
    kind = parse_cue(text).get("kind")
    tempo = kind == "tempo"
    for n, event in enumerate(told):
        if n in taken or event.get("tier") != "cue":
            continue
        outcome, asked = event.get("cue_outcome"), event.get("asked")
        if field == "dropped_cue":
            if (outcome != "expired" or event["bar"] != row["bar"] or not isinstance(asked, dict)
                    or asked.get("name") != name or asked.get("who") != who):
                continue
        else:
            if outcome not in _APPLIED_LABELS:
                continue
            suffix = _APPLIED_LABELS[outcome]
            if suffix is None:
                if kind != "end":
                    continue
            elif event.get("label") != record._cue_label(name, suffix):
                continue
            # Two cues with no name to show differ only in whether the parser knew them.
            if outcome == "nothing" and name is None and (
                    ("did not understand" in event["sentence"]) != (kind == "unknown")):
                continue
            # A tempo cue that landed is told on the bar it sounds from.
            at = after["bar"] if outcome == "landed" and tempo and after else row["bar"]
            if event["bar"] != at:
                continue
            if k is None:
                if asked is not None:
                    continue
            elif (not isinstance(asked, dict) or asked.get("who") != who
                  or asked.get("name") != name
                  or asked.get("bar") != record._sounding_bar(clock, clean[k])):
                continue
        taken.add(n)
        return event
    return None


def _applied_sentence(outcome: str, name: str | None, bar: int, row_bar: int) -> str:
    said = name.capitalize() if name else "A cue"
    if outcome == "landed":
        return f"{said}, from bar {bar}." if name else f"A cue landed at bar {bar}."
    if outcome == "ended":
        return f"Ending, from bar {row_bar}."
    if outcome == "ended_next":
        return f"Ending, one bar after bar {row_bar}."
    if outcome == "held":
        return f"{said} is on hold. A loop is holding the tempo."
    if outcome == "waiting":
        return f"{said} reached the band at bar {row_bar}. What it did is not written yet."
    if outcome == "tone":
        return f"{said} reached the band at bar {row_bar}. I can't find that the tone changed."
    if outcome == "unsure":
        return f"The band jumped at bar {row_bar}. The log does not say this cue did it."
    return f"{said} reached the band at bar {row_bar} and changed nothing."


def _answer(who: str, loop: dict) -> tuple[str, str]:
    """(state, sentence) for a cue answered at once.  Only what the reply says."""
    op, ok = loop["op"], loop["ok"]
    if op == record._SOUND_OP:
        if not ok:
            return "answered", f"{who} asked for a sound I do not have."
        label = _SOUND_LABELS.get(loop.get("sound"))
        return "answered", (f"{who} set your sound to {label}." if label
                            else f"{who} changed your sound.")
    if not ok and op == "unclear":
        return "not_understood", f"{who} asked for a loop. The looper could not follow the words."
    asked = f"{who} asked {record._LOOP_ASKS.get(op, 'about a loop')}."
    return "answered", f"{asked} The looper said {'yes' if ok else 'no'}."


def story(rows, cues=None, ending: dict | None = None, *, live: bool = True,
          partial: bool = False, route: str = "unknown") -> list[dict]:
    """Decision rows and cue rows in; events out, in the order they happened.

    ``rows`` are a take's decision rows and ``cues`` its cue rows, as written to
    ``decisions.ndjson`` and ``cues.ndjson`` or as held in memory.  Either may be
    empty or missing.  ``ending`` is ``ending.json`` once there is one.  ``live``
    says the take is still running.  ``partial`` says ``rows`` no longer start at
    the take's first bar, so nothing is said about how the take began.  ``route``
    is from :func:`plume.record.input_route`.

    Each event has:

    - ``id``: steady across calls while its facts stay in the input.
    - ``seq``: its place in this list.
    - ``kind``: ``cue``, ``band`` or ``end``.
    - ``state``: one of :data:`STATES` for a cue, else None.
    - ``outcome``: one of :data:`OUTCOMES` for an applied cue, else None.
    - ``bar``: the bar it belongs to, or None if no bar had sounded yet.
    - ``sentence``: one short line.
    - ``intent``: where a received cue waits, or None.  Never a result.
    - ``settled``: on a cue's own event, whether its result is known: told by a
      later event, or by the event itself.  None on every other event.
    - ``detail``: the take page's longer sentence for the same fact, or None.
    - ``who``, ``sender``: the word for who sent a cue, and the claimed name.
    - ``name``: the cue's words, if they are exactly a cue plume knows.
    - ``cue``: the cue row's place in ``cues``, or None.
    - ``facts``: where in the input each claim stands.

    A sentence never holds a cue's raw words, a key, a chord or a style it guessed.
    """
    live = bool(live) and not isinstance(ending, dict)
    rows = [r for r in (rows if isinstance(rows, (list, tuple)) else ()) if record._is_row(r)]
    docs, clean, origin = _clean_cues(cues)
    seqs = _seqs(docs)
    clock = record._row_clock(rows)
    told = record.translate(rows, ending, docs, live=live, route=route)
    claims = _claims(rows, clean)
    over = ({"in": "ending", "field": "reason"} if isinstance(ending, dict)
            else {"in": "call", "field": "live"})
    events: list[dict] = []

    def add(key, ident, kind, sentence, *, bar, facts, floor=None, state=None, outcome=None,
            intent=None, settled=None, detail=None, who=None, sender=None, name=None,
            cue=None) -> dict:
        # A result is never told before the cue it is the result of.
        if floor is not None and key < floor:
            key = floor
        event = {"id": ident, "kind": kind, "state": state, "outcome": outcome, "bar": bar,
                 "sentence": sentence, "intent": intent, "settled": settled, "detail": detail,
                 "who": who, "sender": sender, "name": name, "cue": cue, "facts": facts,
                 "_key": (*key, len(events))}
        events.append(event)
        return event

    # ---- every cue the socket took, the moment it took it
    parts_of = [origin.count(i) for i in range(len(docs))]
    part_no: dict[int, int] = {}
    receipts: dict[int, dict] = {}
    floors: dict[int, tuple] = {}
    for k, cue in enumerate(clean):
        i = origin[k]
        n = part_no[i] = part_no.get(i, -1) + 1
        ident = f"c{seqs[i]}" if parts_of[i] == 1 else f"c{seqs[i]}.{n}"
        who, sender = _who(docs[i], cue)
        bar = record._sounding_bar(clock, cue)
        t = cue["t_s"]
        key = floors[k] = (-1 if bar is None else bar, float("inf") if t is None else t)
        if "loop" in cue:
            state, sentence = _answer(who, cue["loop"])
            receipts[k] = add(key, ident, "cue", sentence, bar=bar, state=state, settled=True,
                              detail=record._loop_sentence(cue, record._asked_at(bar)),
                              who=who, sender=sender, cue=i,
                              facts=[{"in": "cues", "index": i, "field": "loop"}])
            continue
        name = record.cue_name(cue["text"])
        kind = parse_cue(cue["text"]).get("kind")
        fact = {"in": "cues", "index": i, "field": "text", "part": n}
        if kind == "unknown":
            receipts[k] = add(key, ident, "cue", f"{who} sent a cue I did not understand.",
                              bar=bar, state="not_understood", settled=True, who=who,
                              sender=sender, cue=i, facts=[fact])
            continue
        # Words that only hold a cue somewhere inside them are not repeated or read
        # back as a wish: the result, if one comes, says what the band made of them.
        sentence = (f"{who} sent a cue." if not name
                    else f"{who} asked to end." if kind == "end" else f"{who} asked for {name}.")
        intent = _INTENT.get(cue["at"])
        receipts[k] = add(key, ident, "cue", sentence, bar=bar, state="received", intent=intent,
                          settled=False, who=who, sender=sender, name=name, cue=i,
                          facts=[fact] + ([{"in": "cues", "index": i, "field": "at"}]
                                          if intent else []))

    # ---- what the rows say became of each cue
    taken: set[int] = set()
    for claim in claims:
        j, field, n, text, k = claim
        row = rows[j]
        event = _pair(claim, told, taken, rows, clean, clock)
        name = record.cue_name(text)
        facts = [{"in": "rows", "index": j, "field": f"form.{field}", "part": n}]
        who = sender = None
        if k is not None:
            facts.append({"in": "cues", "index": origin[k], "field": "text"})
            who, sender = _who(docs[origin[k]], clean[k])
            receipts[k]["settled"] = True
        ident = (f"{receipts[k]['id']}-{field[:-4]}" if k is not None
                 else f"b{row['bar']}-{field[:-4]}{n}")
        key = (row["bar"], float(row["t_s"]))
        shared = dict(who=who, sender=sender, name=name, floor=floors.get(k),
                      cue=origin[k] if k is not None else None,
                      detail=event["sentence"] if event else None)
        if field == "dropped_cue":
            said = name.capitalize() if name else "A cue"
            add(key, ident, "cue", f"{said} ran out of time at bar {row['bar']}.",
                bar=row["bar"], state="dropped", facts=facts, **shared)
        elif parse_cue(text).get("kind") == "unknown":
            # Its receipt already says so.  With no receipt, this row is all there is.
            if k is None:
                add(key, ident, "cue",
                    f"A cue I did not understand reached the band at bar {row['bar']}. "
                    "Nothing acted on it.", bar=row["bar"], state="not_understood",
                    facts=facts, **shared)
        elif event is None:
            add(key, ident, "cue", f"A cue reached the band at bar {row['bar']}.",
                bar=row["bar"], state="applied", facts=facts, **shared)
        else:
            outcome = event["cue_outcome"]
            if event["bar"] != row["bar"]:
                sounding = rows[j + 1]
                key = (sounding["bar"], float(sounding["t_s"]))
                facts.append({"in": "rows", "index": j + 1, "field": "bar"})
            if outcome == "ended_next":
                facts.append({"in": "rows", "index": j + 1, "field": "form.action"})
            add(key, ident, "cue", _applied_sentence(outcome, name, event["bar"], row["bar"]),
                bar=event["bar"], state="applied",
                outcome="ended" if outcome == "ended_next" else outcome, facts=facts, **shared)

    # ---- a cue no row ever took, once the take is over
    last = (rows[-1]["bar"], float("inf")) if rows else None
    for k, cue in enumerate(clean):
        if live or not rows or "loop" in cue or receipts[k]["settled"]:
            continue
        name = record.cue_name(cue["text"])
        asked = {"who": record._who(cue["source"]), "bar": record._sounding_bar(clock, cue),
                 "name": name}
        found = next((n for n, e in enumerate(told)
                      if n not in taken and e.get("cue_outcome") == "unused"
                      and e.get("asked") == asked), None)
        if found is None:
            continue
        taken.add(found)
        # translate says "too late" only when it read every cue the rows hold.
        if not told[found]["label"].endswith(", too late"):
            continue
        receipts[k]["settled"] = True
        add(last, f"{receipts[k]['id']}-unused", "cue",
            f"{name.capitalize() if name else 'A cue'} was not used. The take ended first.",
            bar=rows[-1]["bar"], state="received", detail=told[found]["sentence"],
            who=receipts[k]["who"], sender=receipts[k]["sender"], name=name, cue=origin[k],
            facts=[{"in": "cues", "index": origin[k], "field": "text"}, over])

    # ---- what the band did, and how the take ended
    first_at: dict[int, int] = {}
    for j, row in enumerate(rows):
        first_at.setdefault(row["bar"], j)
    seen: dict[str, int] = {}
    for event in told:
        tier, label, bar = event.get("tier"), event.get("label"), event["bar"]
        j = first_at.get(bar)
        if j is None:
            continue
        plain = event["sentence"].removeprefix(f"Bar {bar}. ")
        if tier == "plan" and label in _BAND_FIELDS:
            if partial and (j == 0 or label in _FIRST_ROW_ONLY):
                continue
            ident = f"b{bar}-{label.replace(' ', '-')}"
            seen[ident] = seen.get(ident, 0) + 1
            if seen[ident] > 1:
                ident = f"{ident}-{seen[ident]}"
            add((bar, float(rows[j]["t_s"])), ident, "band",
                _BAND_OWN[label].format(bar=bar) if label in _BAND_OWN else plain, bar=bar,
                facts=[{"in": "rows", "index": j, "field": _BAND_FIELDS[label]}])
        elif tier == "end" and not live:
            add(last, "end", "end", plain, bar=bar, facts=[over])

    events.sort(key=lambda e: e["_key"])
    for n, event in enumerate(events):
        del event["_key"]
        event["seq"] = n
    return events
