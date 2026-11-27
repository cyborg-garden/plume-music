"""The Story (``plume.story``): one ordered list of true events for a take.

The acceptance rule is the first test: every sentence in every fixture maps to a
fact in its input.  ``check_facts`` is the reader of that rule, and the tests
after it prove the reader can fail.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import re

import pytest

from plume import record, story
from plume.mind import parse_cue
from record_fixture import POISON, cue_line, ending_doc, landed, song
from story_fixtures import ALL, EXPIRY_NOTE, POISON_CUES, held_cue, two_cues_one_boundary

#: Fixed words in sentences ``record.translate`` wrote.  Their numbers are not bars.
_FIXED = ("30 seconds",)
_CLOCK_RE = re.compile(r"\b\d+:\d\d\b")
_WHO_WORDS = ("You", "Another terminal", "Someone")
_INTENTS = {"next-phrase": "Queued for the next phrase.",
            "next-section": "Queued for the next section.",
            "next-bar": "Queued for the next bar.", "now": "Queued for the next bar."}
#: Every cue word a test sends.  A sentence may hold one only as its own cue's name.
_CUE_WORDS = ("busier", "sparser", "faster", "darker", "brighter", "louder", "softer",
              "next section", "stop")
_KEY_OR_CHORD = re.compile(r"\b[A-G][#b]? (?:major|minor)\b|\bFmaj7\b")


def tell(fx: dict, **kw) -> list[dict]:
    return story.story(fx["rows"], fx["cues"], fx["ending"], live=fx["live"], **kw)


def _dig(doc: dict, path: str):
    for part in path.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None
    return doc


def check_facts(events: list[dict], fx: dict) -> None:
    """Every claim an event makes is in the input it points at, or this raises."""
    rows = [r for r in fx["rows"] if record._is_row(r)]
    cues = fx["cues"] or []
    for event in events:
        where = f"{event['id']}: {event['sentence']!r}"
        assert event["facts"], f"no fact behind {where}"
        bars, texts, fields, cue_docs = set(), [], set(), []
        for fact in event["facts"]:
            if fact["in"] == "rows":
                row = rows[fact["index"]]
                value = _dig(row, fact["field"])
                assert value is not None, f"{fact} is empty, behind {where}"
                bars.add(row["bar"])
                fields.add(fact["field"])
                if fact["field"] in ("form.applied_cue", "form.dropped_cue"):
                    texts.append(value.split("; ")[fact["part"]])
                if fact["field"] == "follow":
                    bars.add(value.get("bars"))
                if isinstance(row.get("bpm"), (int, float)):
                    bars.add(round(row["bpm"]))
            elif fact["in"] == "cues":
                doc = cues[fact["index"]]
                assert doc.get(fact["field"]) is not None, f"{fact} is empty, behind {where}"
                fields.add(f"cue.{fact['field']}")
                cue_docs.append(doc)
                if fact["field"] == "text" and "loop" not in doc:
                    parts = doc["text"].split("; ")
                    texts.append(parts[fact.get("part", 0)])
            elif fact["in"] == "ending":
                assert isinstance(fx["ending"], dict) and "reason" in fx["ending"], where
            else:
                assert fact == {"in": "call", "field": "live"} and not fx["live"], where

        # the state names the kind of fact
        state = event["state"]
        if state == "received":
            assert "cue.text" in fields, where
        elif state == "applied":
            assert "form.applied_cue" in fields, where
        elif state == "dropped":
            assert "form.dropped_cue" in fields, where
        elif state == "answered":
            loop = cue_docs[0]["loop"]
            assert "cue.loop" in fields and (loop["op"] != "unclear" or loop["ok"]), where
        elif state == "not_understood":
            unclear = [d for d in cue_docs if isinstance(d.get("loop"), dict)]
            if unclear:
                assert unclear[0]["loop"] == {"op": "unclear", "ok": False}, where
            else:
                assert texts and all(parse_cue(t)["kind"] == "unknown" for t in texts), where
        else:
            assert event["kind"] in ("band", "end") and not cue_docs, where
        if event["outcome"] is not None:
            assert state == "applied" and event["outcome"] in story.OUTCOMES, where

        # every number in the words is a bar (or tempo) of a row it points at
        for words in (event["sentence"], event["intent"] or ""):
            for fixed in _FIXED:
                words = words.replace(fixed, "")
            words = _CLOCK_RE.sub("", words)
            for number in re.findall(r"\d+", words):
                assert int(number) in bars, f"{number} is in no fact behind {where}"

        # who asked comes from the cue row, and only from it
        if cue_docs:
            doc = cue_docs[0]
            allowed = {story.clean_sender(doc.get("sender")), record._who(doc.get("source"))}
            assert event["who"] in allowed - {None}, where
            assert event["sender"] == story.clean_sender(doc.get("sender")), where
            if event["settled"] is not None:        # the cue's own event, not a result
                assert event["sentence"].startswith(f"{event['who']} "), where
        else:
            assert event["who"] is None and event["sender"] is None, where
            assert not event["sentence"].startswith(_WHO_WORDS), where
            assert " asked " not in event["sentence"], where

        # a cue's words are shown only when they are exactly a cue plume knows
        if event["name"] is not None:
            assert texts and all(record.cue_name(t) == event["name"] for t in texts), where
        if event["kind"] == "cue" and "cue.loop" not in fields:     # a loop's ask is fixed words
            for word in _CUE_WORDS:
                if word != event["name"]:
                    assert word not in event["sentence"].lower(), where
        # the take page's longer sentence is about the same cue, from the same sender
        if event["detail"] is not None and event["settled"] is None:
            detail = re.sub(r"^Bar \d+\. ", "", event["detail"])
            if cue_docs:
                assert detail.startswith(f"{record._who(cue_docs[0].get('source'))} "), where
            else:
                assert not detail.startswith(_WHO_WORDS), where
            for word in _CUE_WORDS[:-2]:        # the last two are also translate's own words
                if word != event["name"]:
                    assert word not in detail.lower(), where
            if event["name"] and event["outcome"] != "ended" and (cue_docs or state == "applied"):
                assert event["name"] in detail.lower(), where
        if event["intent"] is not None:
            assert state == "received" and "cue.at" in fields, where
            assert event["intent"] == _INTENTS[cue_docs[0]["at"]], where


# ------------------------------------------------------------ the acceptance rule

@pytest.mark.parametrize("name", sorted(ALL))
def test_every_sentence_in_every_fixture_maps_to_a_fact_in_its_input(name):
    fx = ALL[name]()
    events = tell(fx)
    assert events, "a fixture with nothing to say checks nothing"
    check_facts(events, fx)


@pytest.mark.parametrize("name", sorted(ALL))
def test_no_sentence_repeats_raw_words_or_names_a_key(name):
    fx = ALL[name]()
    for event in tell(fx):
        shown = json.dumps({k: v for k, v in event.items() if k != "facts"})
        for poison in (*POISON, *POISON_CUES):
            assert poison not in shown, (poison, event["id"])
        assert not _KEY_OR_CHORD.search(shown), event["id"]


def _tamper_bar(event):
    event["sentence"] = event["sentence"].replace("bar 16", "bar 15")


def _tamper_who(event):
    event["who"], event["sentence"] = "You", event["sentence"].replace("Claude", "You")


def _tamper_state(event):
    event["state"] = "applied"


def _tamper_name(event):
    event["name"], event["sentence"] = "faster", event["sentence"].replace("busier", "faster")


def _tamper_intent(event):
    event["intent"] = "Queued for the next bar."


def _tamper_fact(event):
    event["facts"] = []


@pytest.mark.parametrize("ident, tamper", [
    ("c0-applied", _tamper_bar), ("c0", _tamper_who), ("c0", _tamper_state),
    ("c0", _tamper_name), ("c0", _tamper_intent), ("c0-applied", _tamper_fact)])
def test_the_checker_fails_on_a_sentence_no_fact_supports(ident, tamper):
    fx = two_cues_one_boundary()
    events = tell(fx)
    check_facts(events, fx)
    tamper(next(e for e in events if e["id"] == ident))
    with pytest.raises(AssertionError):
        check_facts(events, fx)


def test_a_fact_taken_out_of_the_input_takes_its_sentence_with_it():
    fx = two_cues_one_boundary()
    assert "Busier, from bar 16." in [e["sentence"] for e in tell(fx)]
    fx["rows"][16]["form"]["applied_cue"] = None
    said = [e["sentence"] for e in tell(fx)]
    assert not [s for s in said if "from bar" in s and "in," not in s]
    assert "Claude asked for busier." in said


# ------------------------------------------------------------------ the fixtures

def lines(events):
    return [(e["id"], e["state"], e["sentence"]) for e in events]


def test_a_held_cue_shows_as_received_at_once_before_any_row():
    events = tell(held_cue(opened=False))
    assert lines(events) == [
        ("c0", "answered", "You asked to record a loop. The looper said yes."),
        ("c1", "received", "Claude asked for busier.")]
    assert [e["bar"] for e in events] == [None, None]
    assert events[1]["intent"] == "Queued for the next phrase."
    assert events[1]["settled"] is False


def test_a_held_cue_lands_once_the_take_is_open_and_keeps_its_id():
    events = tell(held_cue(opened=True))
    assert lines(events) == [
        ("c0", "answered", "You asked to record a loop. The looper said yes."),
        ("c1", "received", "Claude asked for busier."),
        ("b0-drums", None, "Drums came in on your count, at 53 bpm."),
        ("c1-applied", "applied", "Busier, from bar 4.")]
    assert events[1]["settled"] is True
    assert events[3]["outcome"] == "landed"


def test_a_take_with_no_record_folder_has_a_story():
    """Lists in memory are the whole input.  Nothing is read from anywhere."""
    events = tell(ALL["no_record_folder"]())
    assert lines(events) == [
        ("b0-drums", None, "Drums came in on your count, at 53 bpm."),
        ("b8-bass-and-chords", None, "Bass and chords in, from bar 8."),
        ("c0", "received", "drum-helper asked for sparser."),
        ("c0-applied", "applied", "Sparser, from bar 12."),
        ("b12-melody", None, "Melody in, from bar 12.")]
    assert not [e for e in events if e["kind"] == "end"]


def test_rows_that_no_longer_start_at_the_first_bar_say_nothing_about_the_start():
    fx = ALL["no_record_folder"]()
    fx["rows"] = fx["rows"][8:]
    wrong = [e["sentence"] for e in tell(fx)]
    assert wrong[0].startswith("Drums came in"), "without the flag the oldest row reads as bar one"
    events = tell(fx, partial=True)
    check_facts(events, fx)
    assert lines(events) == [
        ("c0", "received", "drum-helper asked for sparser."),
        ("c0-applied", "applied", "Sparser, from bar 12."),
        ("b12-melody", None, "Melody in, from bar 12.")]


def test_an_unknown_source_is_someone_and_a_bad_name_is_no_name():
    events = [e for e in tell(ALL["unknown_source"]()) if e["kind"] == "cue"]
    assert [e["sentence"] for e in events] == [
        "Someone asked for busier.", "Someone asked for sparser.", "Someone asked for darker.",
        "Someone asked for brighter.", "Someone asked for louder.",
        "drum-helper with a very asked for softer."]
    assert [e["sender"] for e in events] == [None] * 5 + ["drum-helper with a very"]
    assert {e["state"] for e in events} == {"received"}


def test_a_dropped_cue_is_dropped_only_by_the_row_that_says_so():
    fx = ALL["dropped_cue"]()
    events = tell(fx)
    assert lines(events) == [
        ("b0-drums", None, "Drums came in on your count, at 53 bpm."),
        ("c0", "received", "Claude asked for next section."),
        ("c0-dropped", "dropped", "Next section ran out of time at bar 20."),
        ("end", None, "Stopped with plume stop. The last bar and the tail played out. 4:00 in all.")]
    assert events[1]["intent"] == "Queued for the next section."
    assert events[2]["detail"].endswith("It ran out of time after 16 bars before the next section came.")
    fx["rows"][20]["form"]["dropped_cue"] = None
    assert "dropped" not in [e["state"] for e in tell(fx)]


def test_two_cues_in_one_boundary_keep_their_own_sender_and_bar():
    events = [e for e in tell(two_cues_one_boundary()) if e["kind"] == "cue"]
    assert lines(events) == [
        ("c0", "received", "Claude asked for busier."),
        ("c1", "received", "You asked for faster."),
        ("c0-applied", "applied", "Busier, from bar 16."),
        ("c1-applied", "applied", "Faster, from bar 17.")]
    assert [(e["who"], e["cue"]) for e in events] == [("Claude", 0), ("You", 1), ("Claude", 0), ("You", 1)]
    # The long sentence is the take page's own, for the same cue.
    assert events[3]["detail"].startswith("Bar 17. You asked for faster at bar 13.")


def test_the_same_cue_twice_in_one_boundary_is_two_results():
    effects = ["cue 'busier' (texture:busier): density x1.00 -> x1.20",
               "cue 'busier' (texture:busier): density x1.20 -> x1.43"]
    rows = song(20, band_bar=None, cues={16: {"applied": "busier; busier", "effects": effects}})
    cues = [{**cue_line("busier", 13, rows[12]["t_s"] + 1.0, "cli"), "sender": "one"},
            {**cue_line("busier", 14, rows[13]["t_s"] + 1.0, "cli"), "sender": "two"}]
    fx = {"rows": rows, "cues": cues, "ending": None, "live": True}
    events = tell(fx)
    check_facts(events, fx)
    assert [(e["id"], e["who"]) for e in events if e["state"] == "applied"] == [
        ("c0-applied", "one"), ("c1-applied", "two")]


def test_not_understood_comes_from_the_parser_and_never_repeats_the_words():
    events = [e for e in tell(ALL["not_understood"]()) if e["kind"] == "cue"]
    assert lines(events) == [
        ("c0", "not_understood", "Claude sent a cue I did not understand."),
        ("c1", "received", "Another terminal sent a cue."),
        ("c2", "not_understood", "You asked for a loop. The looper could not follow the words."),
        ("c1-applied", "applied", "Ending, from bar 8.")]
    assert events[0]["intent"] is None


def test_a_cue_no_row_took_is_only_ever_received():
    fx = ALL["never_used"]()
    assert [e["sentence"] for e in tell(fx) if e["kind"] == "cue"] == [
        "Claude asked for busier.", "Busier was not used. The take ended first."]
    fx.update(ending=None, live=True)
    running = [e for e in tell(fx) if e["kind"] == "cue"]
    assert [e["sentence"] for e in running] == ["Claude asked for busier."]
    assert running[0]["settled"] is False


def test_a_row_with_no_cue_log_names_no_sender():
    events = tell(ALL["row_without_a_cue_log"]())
    cue = next(e for e in events if e["kind"] == "cue")
    assert (cue["id"], cue["sentence"], cue["who"]) == ("b8-applied0", "Busier, from bar 8.", None)


# --------------------------------------------------- reaching is not changing

def test_a_cue_that_reached_the_band_and_changed_nothing_says_so():
    rows = song(12, band_bar=None, cues={8: {"applied": "busier", "effects": [
        "cue 'busier' (texture:busier): density x2.00 -> x2.00"]}})
    cues = [cue_line("busier", 5, rows[4]["t_s"] + 1.0, "gui")]
    fx = {"rows": rows, "cues": cues, "ending": None, "live": True}
    events = tell(fx)
    check_facts(events, fx)
    result = next(e for e in events if e["state"] == "applied")
    assert result["outcome"] == "nothing"
    assert result["sentence"] == "Busier reached the band at bar 8 and changed nothing."


def test_a_running_take_does_not_say_what_a_tempo_cue_did_before_the_next_bar():
    rows = song(9, band_bar=None, cues={8: {"applied": "faster", "effects": [
        "cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"]}})
    fx = {"rows": rows, "cues": [cue_line("faster", 5, rows[4]["t_s"] + 1.0, "gui")],
          "ending": None, "live": True}
    waiting = next(e for e in tell(fx) if e["state"] == "applied")
    assert (waiting["outcome"], waiting["id"]) == ("waiting", "c0-applied")
    assert "from bar" not in waiting["sentence"]
    fx["rows"] = song(10, band_bar=None, cues={8: {"applied": "faster", "effects": [
        "cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"]}})
    landed_now = next(e for e in tell(fx) if e["state"] == "applied")
    assert (landed_now["outcome"], landed_now["id"]) == ("landed", "c0-applied")
    assert landed_now["sentence"] == "Faster, from bar 9."


def _random_take(rng) -> dict:
    """Cues that collide on purpose: few words, few senders, shared boundaries."""
    words = ("busier", "faster", "sparser", "stop", "next section", "purple monkey dishwasher")
    notes = {"busier": "cue 'busier' (texture:busier): density x1.00 -> x1.43",
             "sparser": "cue 'sparser' (texture:sparser): density x0.05 -> x0.05",
             "faster": "cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"}
    bars = rng.randint(1, 24)
    marks = {}
    for bar in rng.sample(range(bars), k=min(bars, rng.randint(0, 5))):
        applied = [rng.choice(words) for _ in range(rng.randint(0, 3))]
        dropped = [rng.choice(words) for _ in range(rng.randint(0, 2))]
        marks[bar] = {"applied": "; ".join(applied) or None, "dropped": "; ".join(dropped) or None,
                      "effects": [notes[w] for w in applied if w in notes and rng.random() < 0.8],
                      "action": rng.choice(("hold", "end", "jump"))}
    rows = song(bars, band_bar=rng.choice((None, 4)), cues=marks)
    cues = []
    for _ in range(rng.randint(0, 8)):
        bar = rng.choice((None, *range(bars)))
        cue = cue_line(rng.choice(words), bar, rng.choice((None, rng.uniform(0, 120))),
                       rng.choice(("gui", "cli", "voice", "agent", "root")))
        cue["at"] = rng.choice(("now", "next-bar", "next-phrase", "next-section", "soon"))
        if rng.random() < 0.4:
            cue["sender"] = rng.choice(("Claude", "helper", "You", ""))
        if rng.random() < 0.15:
            cue["loop"] = {"op": rng.choice(("record", "stop", "unclear", "sound")),
                           "ok": rng.random() < 0.5}
        cues.append(cue)
    over = rng.random() < 0.5
    return {"rows": rows, "cues": cues, "ending": ending_doc(bars=bars) if over else None,
            "live": not over and rng.random() < 0.8}


def test_random_takes_never_say_a_thing_their_input_does_not_hold():
    import random
    rng = random.Random(20261127)
    said = set()
    for _ in range(400):
        fx = _random_take(rng)
        events = tell(fx)
        check_facts(events, fx)
        assert len({e["id"] for e in events}) == len(events)
        said.update((e["state"], e["outcome"]) for e in events)
        # every result found the take page's own sentence for it
        assert not [e["id"] for e in events
                    if e["kind"] == "cue" and e["settled"] is None and e["detail"] is None]
        # every cue row the reader kept has exactly one event of its own
        own = [e["cue"] for e in events if e["settled"] is not None]
        assert sorted(own) == list(range(len(fx["cues"])))
    # the generator reached the cases that matter
    assert {("applied", "landed"), ("applied", "nothing"), ("applied", "ended"), ("dropped", None),
            ("not_understood", None), ("answered", None), ("received", None)} <= said


# --------------------------------------------------------------- order and ids

def test_events_are_in_order_and_a_result_never_comes_before_its_cue():
    for name, make in ALL.items():
        events = tell(make())
        assert [e["seq"] for e in events] == list(range(len(events))), name
        assert len({e["id"] for e in events}) == len(events), name
        bars = [-1 if e["bar"] is None else e["bar"] for e in events if e["id"] != "end"]
        place = {e["id"]: e["seq"] for e in events}
        for e in events:
            asked = e["id"].split("-")[0]
            if "-" in e["id"] and asked in place:
                assert place[asked] < e["seq"], (name, e["id"])
        if not any(e["id"].endswith("-unused") for e in events):
            assert bars == sorted(bars), name


def test_a_result_logged_with_an_earlier_clock_still_follows_its_cue():
    rows = song(12, band_bar=None, cues={8: landed("busier")})
    cue = {**cue_line("busier", 5, 9e9, "gui"), "bar": 5}        # a clock no row reaches
    events = tell({"rows": rows, "cues": [cue], "ending": None, "live": True})
    ids = [e["id"] for e in events]
    assert ids.index("c0") < ids.index("c0-applied")


def test_ids_follow_the_cue_rows_own_seq_when_they_have_one():
    fx = two_cues_one_boundary()
    for seq, cue in zip((7, 9), fx["cues"]):
        cue["seq"] = seq
    assert [e["id"] for e in tell(fx) if e["kind"] == "cue"] == ["c7", "c9", "c7-applied", "c9-applied"]
    fx["cues"][1]["seq"] = 7                         # not different: fall back to places
    assert [e["id"] for e in tell(fx) if e["kind"] == "cue"][:2] == ["c0", "c1"]


def test_the_same_input_tells_the_same_story_and_is_left_alone():
    fx = two_cues_one_boundary()
    before = copy.deepcopy(fx)
    assert tell(fx) == tell(fx)
    assert fx == before


# --------------------------------------------------------------------- senders

@pytest.mark.parametrize("claimed, shown", [
    ("Claude", "Claude"), ("  Claude  ", "Claude"), ("a\tb\n c", "a b c"),
    ("x" * 40, "x" * 24), ("ab " * 20, "ab ab ab ab ab ab ab ab"),
    ("\x1b[31mred", "[31mred"), ("dr‮um", "drum"), ("na​me", "name"),
    ("", None), ("   ", None), ("\x00\x07", None), (7, None), (None, None), (["Claude"], None),
    ("You", None), ("you", None), ("Someone", None), ("ANOTHER TERMINAL", None), ("I", None)])
def test_a_sender_name_is_trimmed_printable_and_at_most_24_characters(claimed, shown):
    assert story.clean_sender(claimed) == shown
    if shown is not None:
        assert len(shown) <= story.MAX_SENDER_CHARS and shown.isprintable()


# ------------------------------------------------------------- junk and purity

@pytest.mark.parametrize("rows, cues", [
    (None, None), ([], []), ("rows", "cues"), ([7, None, {}, {"bar": "x"}], [7, None, {}]),
    ([{"bar": 1, "t_s": 1.0, "form": "nope"}], [{"text": 7}, {"text": "busier", "bar": "x"}]),
    ([{"bar": 1, "t_s": 1.0, "form": {"applied_cue": 7, "dropped_cue": ["x"]}}],
     [{"text": "busier", "loop": "nope", "seq": True, "sender": {"a": 1}}])])
def test_junk_in_never_raises(rows, cues):
    for live in (True, False):
        for event in story.story(rows, cues, None, live=live):
            assert isinstance(event["sentence"], str)


def test_fields_this_reader_does_not_know_are_ignored():
    fx = two_cues_one_boundary()
    plain = tell(fx)
    for row in fx["rows"]:
        row["a_field_from_a_newer_plume"] = {"x": 1}
    for cue in fx["cues"]:
        cue["a_field_from_a_newer_plume"] = "anything"
    assert tell(fx) == plain


def test_a_cue_log_past_the_limit_reads_as_no_cue_log():
    rows = song(12, band_bar=None, cues={8: landed("busier")})
    cues = [cue_line("busier", 5, rows[4]["t_s"] + 1.0)] * (record.MAX_CUE_LINES + 1)
    events = story.story(rows, cues)
    assert [e["id"] for e in events if e["kind"] == "cue"] == ["b8-applied0"]


def test_a_finished_take_with_rows_ends_with_how_it_ended():
    rows = song(12, band_bar=None)
    events = story.story(rows, [], ending_doc(bars=12), live=False)
    assert events[-1]["kind"] == "end" and events[-1]["facts"] == [{"in": "ending", "field": "reason"}]
    assert story.story(rows, [], ending_doc(bars=12), live=True)[-1]["kind"] == "end", \
        "an ending record outranks the caller's live flag"
    assert not [e for e in story.story(rows, [], None, live=True) if e["kind"] == "end"]


def test_the_expiry_note_fixture_matches_what_the_form_writes():
    assert record._expired_after({"form": {"reason": EXPIRY_NOTE}}, "next section") == (16, "next-section")


def test_the_module_is_pure():
    """No file, device, network or clock: only record, the parser and the sound names."""
    tree = ast.parse(inspect.getsource(story))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(f"{'.' * node.level}{node.module or ''}")
    assert imported == {"__future__", ".", ".inputfx", ".mind"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in ("open", "print")]
