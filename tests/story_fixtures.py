"""Synthetic takes for The Story (``plume.story``): rows and cue rows, no audio.

Each fixture is one case the story feed must tell truly.  Rows come from
``record_fixture``, so they carry the mic-derived text a real row can carry, and
the cue rows have the shape ``realtime.lead`` logs.  ``sender`` is the field a
named sender sets; the cue socket does not write it yet, and a reader must
already be safe with it.  Nothing here came from a real take.
"""
from __future__ import annotations

from plume import realtime
from record_fixture import cue_line, ending_doc, landed, song

EXPIRY_NOTE = "cue 'next section' expired unapplied after 16 bars (ttl 16, at next-section)"
#: Words a cue can hold that are not exactly a cue.  They must never be repeated.
LOOSE_CUE = "stop calling me that"
UNKNOWN_CUE = "purple monkey dishwasher"
POISON_CUES = (LOOSE_CUE, UNKNOWN_CUE, "calling", "purple", "dishwasher", "riff-one")


def _named(doc: dict, sender, source: str = "agent") -> dict:
    return {**doc, "source": source, "sender": sender}


def held_cue(*, opened: bool) -> dict:
    """Two cues arrive before the take is open: no bar yet, and no row.

    The looper answers one at once.  The other waits for the band.  ``opened``
    adds the rows written once the take runs, where the waiting cue lands.
    """
    cues = [{**cue_line("make a 4 bar loop called riff-one", None, 0.2, "gui"),
             "at": "now", "ttl_bars": 0, "loop": {"op": "record", "ok": True}},
            _named(cue_line("busier", None, 0.5), "Claude")]
    rows = song(8, band_bar=None, cues={4: landed("busier")}) if opened else []
    return {"rows": rows, "cues": cues, "ending": None, "live": True}


def no_record_folder() -> dict:
    """A take with no folder: rows and cues exist only in memory, and never end.

    The lists are what a running take holds, so there is no ``ending.json`` and
    no ``received_at``.
    """
    rows = song(14, band_bar=8, cues={12: landed("sparser", 1.0, 0.7)})
    cue = _named(cue_line("sparser", 10, rows[9]["t_s"] + 1.0), "drum-helper")
    cue.pop("received_at")
    return {"rows": rows, "cues": [cue], "ending": None, "live": True}


def unknown_source() -> dict:
    """Cues whose source is not one plume knows, and names that are not names."""
    rows = song(6, band_bar=None)
    t = rows[2]["t_s"] + 0.5
    cues = [{**cue_line("busier", 3, t), "source": "root"},
            {k: v for k, v in cue_line("sparser", 3, t + 0.1).items() if k != "source"},
            {**cue_line("darker", 3, t + 0.2), "source": 7, "sender": 7},
            _named(cue_line("brighter", 3, t + 0.3), "You", source="unknown"),
            _named(cue_line("louder", 3, t + 0.4), "\x1b\u202e \n", source="unknown"),
            _named(cue_line("softer", 3, t + 0.5),
                   "  dr\u202eum\x07-helper \n with a very long name ", source="unknown")]
    return {"rows": rows, "cues": cues, "ending": None, "live": True}


def dropped_cue() -> dict:
    """A cue waits for a section line that never comes, and the form drops it."""
    rows = song(24, band_bar=None, cues={20: {"dropped": "next section", "note": EXPIRY_NOTE}})
    cue = {**_named(cue_line("next section", 4, rows[3]["t_s"] + 1.0), "Claude"),
           "at": "next-section"}
    return {"rows": rows, "cues": [cue], "ending": ending_doc(bars=24), "live": False}


def two_cues_one_boundary() -> dict:
    """Two senders, two cues, one phrase line.  The tempo cue sounds a bar later."""
    effects = ["cue 'busier' (texture:busier): density x1.00 -> x1.43",
               "cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"]
    rows = song(20, band_bar=8, cues={16: {"applied": "busier; faster", "effects": effects}})
    cues = [_named(cue_line("busier", 13, rows[12]["t_s"] + 1.0), "Claude"),
            cue_line("faster", 14, rows[13]["t_s"] + 1.0, "gui")]
    return {"rows": rows, "cues": cues, "ending": None, "live": True}


def not_understood() -> dict:
    """Words the parser does not know, words it half knows, and a loop it cannot follow."""
    rows = song(12, band_bar=None, cues={8: {"applied": f"{UNKNOWN_CUE}; {LOOSE_CUE}",
                                             "action": "end"}})
    t = rows[4]["t_s"] + 0.5
    cues = [_named(cue_line(UNKNOWN_CUE, 5, t), "Claude"),
            cue_line(LOOSE_CUE, 5, t + 0.1, "cli"),
            {**cue_line("loop the riff-one thing maybe", 5, t + 0.2, "voice"),
             "at": "now", "ttl_bars": 0, "loop": {"op": "unclear", "ok": False}}]
    return {"rows": rows[:9], "cues": cues, "ending": ending_doc(realtime.END_FORM, bars=9), "live": False}


def never_used() -> dict:
    """A cue sent near the end.  The take stops before its phrase line."""
    rows = song(10, band_bar=None)
    cues = [_named(cue_line("busier", 9, rows[8]["t_s"] + 1.0), "Claude")]
    return {"rows": rows, "cues": cues, "ending": ending_doc(bars=10), "live": False}


def row_without_a_cue_log() -> dict:
    """A finished take whose cue log is gone: the rows are all anyone has."""
    rows = song(12, band_bar=None, cues={8: landed("busier")})
    return {"rows": rows, "cues": None, "ending": ending_doc(bars=12), "live": False}


def old_cue_new_cue(first: dict) -> dict:
    """A take longer than a ring holds.  Two senders ask for the same thing, 290
    bars apart, and ``first`` is what the row at bar 12 did with the first cue."""
    rows = song(320, band_bar=8, cues={12: first, 304: landed("busier")})
    cues = [_named(cue_line("busier", 10, rows[9]["t_s"] + 1.0), "Claude"),
            cue_line("busier", 300, rows[299]["t_s"] + 1.0, "gui")]
    return {"rows": rows, "cues": cues, "ending": None, "live": True}


def waiting_across_the_cut() -> dict:
    """Two senders ask for the same thing, and each waits for its own phrase line."""
    rows = song(13, band_bar=None, cues={8: landed("busier"), 12: landed("busier", 1.43, 1.6)})
    cues = [{**cue_line("busier", 3, rows[2]["t_s"] + 1.0, "cli"), "sender": "one"},
            {**cue_line("busier", 5, rows[5]["t_s"] + 1.0, "cli"), "sender": "two"}]
    return {"rows": rows, "cues": cues, "ending": None, "live": True}


def same_words_two_boundaries() -> dict:
    """Two senders ask for the same thing, and each waits for a different line.

    The first cue waits for the next section and the second for the next bar, so
    the form takes the second one first.  ``truth`` is which cue each row took:
    (bar, row field, place in the field) to the cue row's place in ``cues``.
    """
    rows = song(13, band_bar=None, cues={4: landed("busier"), 8: landed("busier", 1.43, 1.6)})
    cues = [{**cue_line("busier", 1, rows[1]["t_s"] + 1.0, "cli"), "sender": "one",
             "at": "next-section"},
            {**cue_line("busier", 3, rows[3]["t_s"] + 1.0, "cli"), "sender": "two",
             "at": "next-bar"}]
    return {"rows": rows, "cues": cues, "ending": None, "live": True,
            "truth": {(4, "form.applied_cue", 0): 1, (8, "form.applied_cue", 0): 0}}


ALL = {
    "held_cue_before_open": lambda: held_cue(opened=False),
    "held_cue_after_open": lambda: held_cue(opened=True),
    "no_record_folder": no_record_folder,
    "unknown_source": unknown_source,
    "dropped_cue": dropped_cue,
    "two_cues_one_boundary": two_cues_one_boundary,
    "not_understood": not_understood,
    "never_used": never_used,
    "row_without_a_cue_log": row_without_a_cue_log,
    "same_words_two_boundaries": same_words_two_boundaries,
}
