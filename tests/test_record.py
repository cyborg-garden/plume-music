"""The translator and the roast: what The Record is allowed to say.

These are the honesty rules of plume GUI scope section 7, as tests.  The fixture
rows carry the mic-derived text that real rows carry, because a clean fixture
would prove nothing.
"""
from __future__ import annotations

import json

import pytest

from plume import realtime, record
from plume.mind import parse_cue
from record_fixture import (MIC_ENTRY_REASON, POISON, cue_line, ending_doc, landed, make_row,
                            song, write_take)

ALL_END = sorted(v for k, v in vars(realtime).items() if k.startswith("END_"))


def words(events=(), lines=()) -> str:
    """Everything these would put on the page, as one string."""
    return json.dumps([record.public_event(e) for e in events]) + json.dumps(list(lines))


def by_label(events, label):
    return next(e for e in events if e["label"] == label)


# ------------------------------------------------------- mic-derived text stays out

def assert_clean(text: str):
    for bad in POISON:
        assert bad not in text, f"{bad!r} reached the page"


def test_no_mic_derived_text_comes_out_of_the_translator_or_the_roast():
    rows = song(40, cues={20: landed("busier"), 24: {"applied": "louder"},
                          28: {"dropped": "sparser", "note":
                               "cue 'sparser' expired unapplied after 2 bars (ttl 2, at next-phrase)"}})
    ending = ending_doc()
    cues = [cue_line("busier", 18, 90.0), cue_line("louder", 22, 100.0, "cli")]
    events = record.translate(rows, ending, cues)
    lines = record.roast(rows, ending, cues)
    assert_clean(words(events, lines))
    # The check has teeth: this text really is in the input.
    assert any(bad in json.dumps(rows) for bad in POISON)


def test_nothing_from_the_folder_but_the_allowlist_comes_out_of_read_take(tmp_path):
    rows = song(40, cues={20: landed("busier")})
    folder = write_take(tmp_path / "t", rows, ending=ending_doc(), cues=[cue_line("busier", 18, 90.0)])
    out = record.read_take(folder)
    assert_clean(json.dumps(out))


def test_the_filtered_row_is_built_from_safe_keys_only():
    row = make_row(30, applied="busier")
    row["a_new_log_field_nobody_has_reviewed"] = "secret"
    row["form"]["another_new_field"] = "secret"
    safe = record.safe_row(row)
    assert "secret" not in json.dumps(safe)
    assert set(safe) <= {"bar", "t_s", "bpm", "density", "gain_db", "fade_db", "entry", "section",
                         "key", "chord", "roles", "playing", "form"}
    assert set(safe["form"]) <= {"action", "applied_cue", "dropped_cue"}


@pytest.mark.parametrize("field,value", [
    ("key", "human is flat"), ("chord", "<img src=x onerror=alert(1)>"),
    ("entry", "human is playing"), ("section", "human solo")])
def test_a_value_in_a_safe_field_must_also_look_like_that_field(field, value):
    row = make_row(30)
    row[field] = value
    assert value not in json.dumps(record.safe_row(row))


def test_playing_is_shown_over_drums_and_dropped_once_the_band_is_in():
    assert record.safe_row(make_row(3, entry="drums", section="intro", playing=True))["playing"] is True
    assert "playing" not in record.safe_row(make_row(30, entry="full", playing=True))


def test_entry_reason_is_not_shown_even_though_it_looks_harmless():
    rows = song(10, band_bar=None)
    assert MIC_ENTRY_REASON in json.dumps(rows)
    assert MIC_ENTRY_REASON not in words(record.translate(rows, ending_doc()))


# ----------------------------------------------------------- "landed" needs proof

def test_a_cue_with_a_matching_effect_landed():
    events = record.translate(song(40, cues={20: landed("busier")}), ending_doc())
    assert by_label(events, "busier landed")["sentence"] == "Bar 20. A cue landed: busier."


def test_no_cue_log_means_no_timing_and_no_sender():
    text = words(record.translate(song(40, cues={20: landed("busier")}), ending_doc()))
    assert "A cue landed" in text
    for claim in ("seconds later", "You asked", "Another terminal", "Someone asked"):
        assert claim not in text


def test_with_a_cue_log_it_says_who_when_and_how_long_it_took():
    rows = song(40, bpm=60.0, cues={32: landed("busier")})
    bar_s = 4.0
    # Sent while bar 29 sounded, though bar 30 was already planned.
    cue = cue_line("busier", 30, rows[29]["t_s"] + 1.0)
    assert rows[29]["t_s"] + bar_s == rows[30]["t_s"]
    event = by_label(record.translate(rows, ending_doc(), [cue]), "busier landed")
    wait = rows[32]["t_s"] - cue["t_s"]
    assert event["sentence"] == f"Bar 32. You asked for busier at bar 29. It landed {wait:.1f} seconds later."
    assert not event["dashed"]


@pytest.mark.parametrize("source,who", [("gui", "You"), ("voice", "You"),
                                        ("cli", "Another terminal"), ("unknown", "Someone"),
                                        ("nonsense", "Someone")])
def test_who_asked_comes_from_the_log(source, who):
    rows = song(40, cues={32: landed("busier")})
    event = by_label(record.translate(rows, ending_doc(), [cue_line("busier", 30, rows[30]["t_s"], source)]),
                     "busier landed")
    assert f"{who} asked for busier" in event["sentence"]


def test_a_cue_before_any_bar_does_not_claim_a_bar():
    rows = song(40, cues={4: landed("busier")})
    event = by_label(record.translate(rows, ending_doc(), [cue_line("busier", None, 1.0)]), "busier landed")
    assert "before the first bar" in event["sentence"]


def test_unknown_text_in_applied_cue_does_not_read_as_landed():
    assert parse_cue("wrap up")["kind"] == "unknown"
    events = record.translate(song(40, cues={20: {"applied": "wrap up"}}), ending_doc())
    cue = next(e for e in events if e["tier"] == "cue")
    assert cue["dashed"] and "landed" not in cue["label"]
    assert "nothing acted on it" in cue["sentence"]


def test_text_plume_did_not_understand_is_not_repeated_back():
    typed = "the neighbours are complaining"
    assert parse_cue(typed)["kind"] == "unknown"
    rows = song(40, cues={20: {"applied": typed}})
    out = words(record.translate(rows, ending_doc(), [cue_line(typed, 18, rows[18]["t_s"], "cli")]))
    assert typed not in out and "did not understand" in out


def test_markup_in_a_cue_never_reaches_the_page_even_with_a_known_word_in_it():
    text = "louder <img src=x onerror=alert(1)>"
    assert parse_cue(text)["kind"] != "unknown"
    rows = song(40, cues={20: {"applied": text}})
    assert "<img" not in words(record.translate(rows, ending_doc(), [cue_line(text, 18, rows[18]["t_s"])]))
    assert record.cue_name(text) is None


def test_a_cue_logged_with_no_effect_is_dashed_with_the_specs_sentence():
    event = by_label(record.translate(song(40, cues={20: {"applied": "louder"}}), ending_doc()),
                     "louder, did nothing")
    assert event["dashed"] and event["sentence"] == 'Bar 20. "louder" was logged, and nothing acted on it.'


def test_not_applied_is_not_landed():
    rows = song(40, cues={20: {"applied": "more reverb", "effects": [
        "cue 'more reverb' (texture:reverb): understood but NOT applied — a mix-bus effect"]}})
    events = record.translate(rows, ending_doc())
    assert not [e for e in events if e["label"].endswith(" landed")]
    assert [e for e in events if e["dashed"] and e["tier"] == "cue"]


def test_a_density_cue_at_its_limit_changed_nothing_so_it_did_not_land():
    rows = song(40, cues={20: landed("busier", 2.0, 2.0)})
    assert not [e for e in record.translate(rows, ending_doc()) if e["label"].endswith(" landed")]


def test_the_note_is_ours_and_never_the_engines_text():
    event = by_label(record.translate(song(40, cues={20: landed("busier")}), ending_doc()), "busier landed")
    assert event["note"] == "Density x1.00 to x1.43."


def test_darker_says_the_log_claims_it_and_nobody_can_find_it():
    rows = song(40, cues={20: {"applied": "darker", "effects": [
        "cue 'darker' (texture:darker): tone +0.0 -> -1.0 from here on"]}})
    event = by_label(record.translate(rows, ending_doc()), "darker, no effect")
    assert event["dashed"] and "I can't find that it did" in event["sentence"]


def test_only_an_end_cue_gets_the_credit_for_ending():
    rows = song(40, cues={20: {"applied": "louder; wrap it up", "action": "end"}})[:21]
    events = record.translate(rows, ending_doc(realtime.END_FORM))
    assert "You" not in by_label(events, "cue ended it")["sentence"]
    assert by_label(events, "louder, did nothing")["dashed"]


def test_a_jump_counts_on_plain_lead_but_not_on_tap_start():
    moved = "cue 'louder' (energy:up) -> section 2 'peak-a' (energy 0.90)"
    plain = [make_row(b, entry=None, applied="louder" if b == 8 else None,
                      action="jump" if b == 8 else "hold", note=moved if b == 8 else "") for b in range(12)]
    assert by_label(record.translate(plain, ending_doc()), "louder landed")
    tap = song(30, cues={20: {"applied": "louder", "action": "jump"}})
    assert by_label(record.translate(tap, ending_doc()), "louder, did nothing")


def test_an_expired_cue_is_dashed_and_blames_the_real_cause():
    rows = song(40, cues={21: {"dropped": "sparser", "note":
                               "cue 'sparser' expired unapplied after 2 bars (ttl 2, at next-phrase)"}})
    event = by_label(record.translate(rows, ending_doc()), "sparser, expired")
    assert event["dashed"]
    assert event["sentence"] == "Bar 21. It ran out of time after 2 bars before a phrase line came."


def test_a_cue_the_take_ended_before_applying_is_shown_as_not_happening():
    rows = song(40)
    event = by_label(record.translate(rows, ending_doc(), [cue_line("busier", 38, rows[38]["t_s"])]),
                     "busier, too late")
    assert event["dashed"] and "The take ended before a phrase line came" in event["sentence"]


def test_the_same_text_twice_matches_each_log_line_once():
    rows = song(50, bpm=60.0, cues={20: landed("busier"), 40: landed("busier", 1.43, 2.0)})
    cues = [cue_line("busier", 18, rows[18]["t_s"]), cue_line("busier", 38, rows[38]["t_s"])]
    sentences = [e["sentence"] for e in record.translate(rows, ending_doc(), cues)
                 if e["label"] == "busier landed"]
    assert "at bar 18" in sentences[0] and "at bar 38" in sentences[1]


# ---------------------------------------------------------------- plume's own plan

def test_entry_changes_read_as_what_the_band_did():
    events = record.translate(song(40, band_bar=9), ending_doc())
    assert by_label(events, "drums")["sentence"] == "Bar 0. Drums came in on your count, at 53 bpm."
    assert by_label(events, "bass and chords")["sentence"] == "Bar 9. Bass and chords in, in F major."
    assert by_label(events, "melody")["bar"] == 13


def test_the_outro_says_the_length_ran_out():
    events = record.translate(song(50, outro_bar=44), ending_doc(realtime.END_SONG))
    assert "The length you set ran out" in by_label(events, "outro")["sentence"]


def test_a_silence_fade_and_its_return():
    rows = song(40, cues={b: {"fade_db": -9.0} for b in range(20, 24)})
    events = record.translate(rows, ending_doc())
    assert by_label(events, "fade started")["bar"] == 20
    back = by_label(events, "came back")
    assert back["bar"] == 24
    # With the band in that is all plume knows.
    assert "The mic stopped being silent" in back["sentence"]


def test_an_outro_fade_is_not_a_silence_fade():
    rows = song(50, outro_bar=44, cues={b: {"fade_db": -12.0} for b in range(46, 50)})
    assert not [e for e in record.translate(rows, ending_doc(realtime.END_SONG))
                if e["label"] in ("fade started", "came back")]


def test_plain_lead_rows_do_not_claim_a_count():
    rows = [make_row(b, entry=None) for b in range(10)]
    text = words(record.translate(rows, ending_doc()), record.roast(rows, ending_doc()))
    assert "on your count" not in text and "You counted" not in text
    assert "The band started" in text


# ------------------------------------------------------------------- the endings

def test_every_end_constant_is_covered():
    assert len(ALL_END) == 8
    assert sorted(record._END_LABEL) == ALL_END
    assert sorted(record._END_SENTENCE) == sorted(set(ALL_END) - {realtime.END_FORM})


@pytest.mark.parametrize("reason", ALL_END)
def test_each_ending_has_a_sentence_and_a_label(reason):
    events = record.translate(song(20), ending_doc(reason, elapsed_s=125.0))
    end = events[-1]
    assert end["tier"] == "end" and end["sentence"].endswith("2:05 in all.")
    assert end["label"] != "ended" or reason == realtime.END_FORM


@pytest.mark.parametrize("reason,needle", [
    (realtime.END_STOPPED, "Stopped with plume stop"),
    (realtime.END_INTERRUPT, "Cut short"),
    (realtime.END_TERMINATED, "It was not plume stop"),
    (realtime.END_SILENCE, "The mic went silent for 30 seconds"),
    (realtime.END_SONG, "wound down as planned"),
    (realtime.END_MAX_MINUTES, "60-minute time cap"),
    (realtime.END_WEDGED, "You did not choose this"),
    (realtime.END_FORM, "The arrangement ran out"),
])
def test_ending_sentences(reason, needle):
    assert needle in record.translate(song(20), ending_doc(reason))[-1]["sentence"]


def test_a_cue_ending_reads_differently_from_the_map_running_out():
    rows = song(21, cues={20: {"applied": "wrap it up", "action": "end"}})
    assert "A cue ended the take" in record.translate(rows, ending_doc(realtime.END_FORM))[-1]["sentence"]
    rows = song(21, cues={20: {"action": "end"}})
    assert "The arrangement ran out" in record.translate(rows, ending_doc(realtime.END_FORM))[-1]["sentence"]


def test_a_reason_this_page_does_not_know_is_not_echoed():
    end = record.translate(song(20), {"reason": "<b>human was bad</b>", "elapsed_s": 5})[-1]
    assert "human" not in end["sentence"] and "does not know" in end["sentence"]
    assert end["label"] == "ended"


def test_missing_ending_json_does_not_throw_and_says_so():
    end = record.translate(song(20), None)[-1]
    assert "No ending record" in end["sentence"] and end["dashed"]
    assert end["label"] == "no record"


@pytest.mark.parametrize("ending", [[], "stopped", 7, {"reason": 7}, {"reason": None, "elapsed_s": "x"}])
def test_a_malformed_ending_does_not_throw(ending):
    assert record.translate(song(5), ending)[-1]["tier"] == "end"


def test_zero_rows_reads_as_stopped_during_the_count():
    events = record.translate([], ending_doc(realtime.END_STOPPED, bars=0))
    assert len(events) == 1 and events[0]["row"] is None
    assert events[0]["sentence"] == ("Stopped before the first bar, so nothing played. "
                                     "The mic recording is still in the folder.")
    assert record.roast([], ending_doc()) == []


def test_zero_rows_with_no_ending_and_no_recording_claims_neither():
    text = record.translate([], None, has_human=False)[0]["sentence"]
    assert "You stopped" not in text and "No mic recording" in text


def test_zero_rows_at_the_cap_does_not_say_she_stopped():
    text = record.translate([], ending_doc(realtime.END_MAX_MINUTES))[0]["sentence"]
    assert "Stopped" not in text and "time cap ended this take" in text


@pytest.mark.parametrize("junk", [
    {"bar": "8", "t_s": 1.0}, {"bar": 8}, {"bar": True, "t_s": 1.0}, {"bar": -1, "t_s": 1.0},
    {"bar": 8, "t_s": float("nan")}, {"bar": 8, "t_s": float("inf")}, [], None, "x"])
def test_a_junk_row_is_ignored(junk):
    assert record.translate([junk], ending_doc())[0]["row"] is None


def test_junk_inside_a_good_row_does_not_throw():
    row = make_row(3)
    row.update(form="nope", roles="nope", cue_effects=7, density="x", fade_db="x", bpm=None)
    record.translate([row], ending_doc())
    record.roast([row], ending_doc())
    record.band_line([row])
    assert record.portrait([row], ending_doc(), [], 1)["seed"] == 1


# --------------------------------------------------------------------- the roast

def test_the_roast_is_short_and_every_line_cites_a_row():
    rows = song(45, bpm=53.0, band_bar=9, cues={
        b: landed("busier") for b in (16, 20, 24, 28, 32, 36)})
    cues = [cue_line("busier", b - 4, rows[b - 4]["t_s"]) for b in (16, 20, 24, 28, 32, 36)]
    lines = record.roast(rows, ending_doc(), cues)
    assert 1 <= len(lines) <= 5
    for line in lines:
        assert isinstance(line["bar"], int) and line["row"]["bar"] == line["bar"]
    assert_clean(json.dumps(lines))


def test_it_roasts_itself_about_not_hearing_once_the_band_is_in():
    text = words(lines=record.roast(song(40, band_bar=17), ending_doc()))
    assert "Band came in at bar 17. From here on I have no idea what you did." in text


def test_a_player_who_starts_at_once_is_not_blamed_for_the_lag():
    # first heard on bar 3, which is what starting immediately looks like
    assert "You let the drums run" not in words(lines=record.roast(song(40, heard_bar=3), ending_doc()))
    # bar 7 is still under the line
    assert "You let the drums run" not in words(lines=record.roast(song(40, heard_bar=7), ending_doc()))


def test_a_late_entry_takes_off_the_three_bars_of_confirmation():
    text = words(lines=record.roast(song(40, heard_bar=12, band_bar=17), ending_doc()))
    assert "You let the drums run about 9 bars" in text


def test_the_lag_line_only_reads_drums_bars():
    # playing is true from bar 0 on, band bars included: only a drums bar counts
    rows = song(40, heard_bar=None, band_bar=10)
    rows[20]["playing"] = True
    assert "You let the drums run" not in words(lines=record.roast(rows, ending_doc()))


def test_silence_with_no_player_gets_the_never_came_in_line():
    text = words(lines=record.roast(song(9, heard_bar=None, band_bar=None), ending_doc(realtime.END_SILENCE)))
    assert "You never came in" in text
    assert "You never came in" not in words(lines=record.roast(song(9, band_bar=None), ending_doc()))


def test_cue_lines_need_the_cue_log():
    rows = song(40, cues={32: landed("busier")})
    assert "I got there" not in words(lines=record.roast(rows, ending_doc()))
    # Sent during bar 28, landed on 32: a real wait.
    text = words(lines=record.roast(rows, ending_doc(), [cue_line("busier", 29, rows[28]["t_s"])]))
    assert "You asked for busier at bar 28. I got there at bar 32." in text


def test_a_prompt_cue_is_not_mocked_for_being_slow():
    rows = song(40, cues={32: landed("busier")})
    text = words(lines=record.roast(rows, ending_doc(), [cue_line("busier", 31, rows[31]["t_s"])]))
    assert "I got there" not in text


def test_a_tempo_change_does_not_say_it_held_the_tempo():
    rows = song(40, bpm=60.0)
    for row in rows[20:]:
        row["bpm"] = 70.0
    rows[19]["form"]["applied_cue"] = "faster"
    rows[19]["cue_effects"] = ["cue 'faster' (tempo:faster): tempo x1.00 -> x1.17"]
    text = words(lines=record.roast(rows, ending_doc()))
    assert "held it" not in text and "a cue moved it" in text


def test_a_tempo_that_moved_with_no_cue_does_not_blame_a_cue():
    rows = song(40, bpm=53.0, band_bar=8)
    rows[30]["bpm"] = 54.5
    text = words(lines=record.roast(rows, ending_doc(), []))
    assert "a cue moved it" not in text and "between 53.0 and 54.5" in text


def test_a_short_take_does_not_brag_about_holding_a_tempo():
    assert "very brave" not in words(lines=record.roast(song(4, bpm=60.0, band_bar=None), ending_doc()))


def test_the_roast_promises_nothing_about_plugging_in():
    assert "plug" not in words(lines=record.roast(song(40), ending_doc())).lower()


def test_the_roast_never_rates_her_timing():
    rows = song(40, heard_bar=12, band_bar=17, cues={20: landed("busier")})
    text = words(lines=record.roast(rows, ending_doc())).lower()
    for claim in ("you dragged", "you rushed", "behind the beat", "ahead of the beat", " ms"):
        assert claim not in text


# ------------------------------------------------------------- the line, the stone

def test_the_quiet_line_is_only_what_plume_planned():
    rows = song(20, band_bar=8)
    line = record.band_line(rows)
    assert [p["bar"] for p in line] == list(range(20))
    assert line[12]["level"] > line[3]["level"]        # more band, higher line
    faded = record.band_line(song(20, cues={15: {"fade_db": -36.0}}))
    assert faded[15]["level"] < faded[14]["level"]


def test_same_take_same_stone():
    rows = song(40, cues={20: landed("busier")})
    events = record.translate(rows, ending_doc())
    assert record.portrait(rows, ending_doc(), events, 9) == record.portrait(rows, ending_doc(), events, 9)
    assert record.portrait(rows, ending_doc(), events, 9)["nicks"] == 1


def test_take_facts_set_the_big_shapes():
    def stone(rows, ending=None, seed=1):
        ending = ending or ending_doc()
        return record.portrait(rows, ending, record.translate(rows, ending), seed)
    assert stone(song(40), ending_doc(elapsed_s=60)).get("height") < stone(song(40), ending_doc(elapsed_s=280))["height"]
    assert stone(song(40, bpm=50.0))["lean"] > stone(song(40, bpm=120.0))["lean"]
    assert stone(song(40, band_bar=8))["cut"] < stone(song(40, band_bar=30))["cut"]
    assert stone(song(40, key="A minor"))["minor"] is True
    assert stone(song(40, key="F major"))["minor"] is False
    assert stone(song(40, band_bar=None))["cut"] is None
    assert stone([])["cut"] is None


# ----------------------------------------------------------------- the folder

def test_an_older_take_with_no_cue_log_is_read_as_one(tmp_path):
    assert record.read_take(write_take(tmp_path / "old", song(5)))["has_cue_log"] is False
    assert record.read_take(write_take(tmp_path / "new", song(5), cues=[]))["has_cue_log"] is True


def test_a_half_written_last_line_is_skipped(tmp_path):
    folder = write_take(tmp_path / "t", song(5), cues=[cue_line("busier", 2, 9.0)])
    with (folder / realtime.REC_DECISIONS).open("a") as f:
        f.write('{"bar": 5, "t_s": 3')
    with (folder / realtime.REC_CUES).open("a") as f:
        f.write('{"text": "bus')
    assert record.read_take(folder)["bars"] == 5


def test_a_stop_during_the_count_is_still_a_take(tmp_path):
    folder = tmp_path / "t"
    folder.mkdir()
    (folder / realtime.REC_HUMAN).write_bytes(b"RIFF")
    assert record.is_take_dir(folder)
    take = record.read_take(folder)
    assert take["bars"] == 0 and take["audio"] == ["human"]
    assert "nothing played" in take["events"][0]["sentence"]


def test_an_empty_folder_is_not_a_take(tmp_path):
    assert not record.is_take_dir(tmp_path)


def test_a_symlinked_recording_is_not_part_of_the_take(tmp_path):
    folder = write_take(tmp_path / "t", song(3))
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    (folder / realtime.REC_HUMAN).symlink_to(outside)
    assert record.read_take(folder)["audio"] == []


@pytest.mark.parametrize("doc,kept,name", [
    ('{"kept": true, "name": "slow one"}', True, "slow one"),
    ('{"kept": "yes", "name": 7}', False, ""),
    ("not json", False, ""),
    ('{"kept": true, "name": "line\\nbreak\\u0007"}', True, "linebreak"),
])
def test_the_keep_sidecar_is_not_trusted(tmp_path, doc, kept, name):
    folder = write_take(tmp_path / "t", song(3))
    (folder / record.REC_KEEP).write_text(doc)
    assert record.read_keep(folder) == {"kept": kept, "name": name}


# --------------------------------------------- cue text, against the real parser

def test_gui_cue_texts_mean_what_the_buttons_say():
    assert parse_cue("wrap it up")["kind"] == "end"
    assert parse_cue("busier")["kind"] == "texture"
    assert parse_cue("sparser")["kind"] == "texture"


# ---------------------------------------------- what a cue really did, in the harder cases

def _take_end(rows, cues=None, reason=realtime.END_FORM):
    return record.translate(rows, ending_doc(reason, elapsed_s=60.0, bars=len(rows)), cues)


def test_an_end_cue_overridden_on_its_line_still_ended_the_take():
    """Real Mind: the cue sets the ending, then another due cue overwrites the action."""
    from test_form import MAP_LONG, run_form, steady

    from plume.mind import Cue, Mind, MindParams
    from plume.sim import _LeadCues

    steps = steady(12)
    steps[1]["cue"] = Cue("wrap it up")
    steps[2]["cue"] = Cue("next section")
    rows = []
    for i, plan in enumerate(run_form(Mind(MindParams()), MAP_LONG, steps)):
        rows.append(make_row(10 + i, action=plan.action, applied=plan.applied_cue,
                             effects=_LeadCues().apply(plan.applied_cue) or None))
        if plan.action == "end":
            break
    overridden = rows[-2]["form"]
    assert overridden["action"] == "jump" and "wrap it up" in overridden["applied_cue"]
    assert rows[-1]["form"]["applied_cue"] is None
    events = _take_end(rows)
    assert "nothing acted on it" not in by_label(events, "cue ended it")["sentence"]
    assert "one bar later" in by_label(events, "cue ended it")["sentence"]
    assert "You did not choose this" not in events[-1]["sentence"]
    assert "A cue ended the take" in events[-1]["sentence"]


def test_a_second_busier_at_the_ceiling_did_not_land():
    """Real _LeadCues and the engine's own clamp: the scale moved, the density did not."""
    from plume.sim import _LeadCues

    cues = _LeadCues()
    cues.apply("busier")
    notes = cues.apply("busier")
    assert " -> x" in notes[0]
    rows = [make_row(19, density=1.0), make_row(20, density=1.0, applied="busier", effects=notes)]
    events = _take_end(rows, reason=realtime.END_STOPPED)
    event = next(e for e in events if e["tier"] == "cue")
    assert event["cue_outcome"] == "nothing" and event["dashed"] and event["note"] == ""
    assert record.portrait(rows, None, events, 1)["nicks"] == 0
    text = words(lines=record.roast(rows, ending_doc(), [cue_line("busier", 10, rows[0]["t_s"])]))
    assert "I got there" not in text


def test_a_busier_that_reached_the_ceiling_did_land():
    rows = [make_row(19, density=0.7),
            make_row(20, density=1.0, applied="busier",
                     effects=["cue 'busier' (texture:busier): density x1.00 -> x1.54"])]
    assert next(e for e in _take_end(rows) if e["tier"] == "cue")["cue_outcome"] == "landed"


def test_a_sparser_still_above_the_ceiling_did_not_land():
    rows = [make_row(19, density=1.0),
            make_row(20, density=1.0, applied="sparser",
                     effects=["cue 'sparser' (texture:sparser): density x2.00 -> x1.54"])]
    assert next(e for e in _take_end(rows) if e["tier"] == "cue")["cue_outcome"] == "nothing"


def test_a_sparser_at_the_floor_did_not_land():
    rows = [make_row(19, density=0.05),
            make_row(20, density=0.05, applied="sparser",
                     effects=["cue 'sparser' (texture:sparser): density x0.30 -> x0.20"])]
    assert next(e for e in _take_end(rows) if e["tier"] == "cue")["cue_outcome"] == "nothing"


def test_the_same_cue_twice_on_one_bar_reads_its_own_note():
    rows = [make_row(19, density=0.3),
            make_row(20, density=0.6, applied="busier; busier", effects=[
                "cue 'busier' (texture:busier): density x1.00 -> x1.54",
                "cue 'busier' (texture:busier): density x1.54 -> x2.00"])]
    notes = [e["note"] for e in _take_end(rows) if e["tier"] == "cue"]
    assert notes == ["Density x1.00 to x1.54.", "Density x1.54 to x2.00."]


def test_the_second_of_two_cues_at_the_limit_reads_as_nothing():
    rows = [make_row(19, density=0.5),
            make_row(20, density=1.0, applied="busier; busier", effects=[
                "cue 'busier' (texture:busier): density x1.54 -> x2.00",
                "cue 'busier' (texture:busier): density x2.00 -> x2.00"])]
    events = [e for e in _take_end(rows) if e["tier"] == "cue"]
    assert [e["cue_outcome"] for e in events] == ["landed", "nothing"]
    assert record.portrait(rows, None, _take_end(rows), 1)["nicks"] == 1


def test_a_cue_holding_a_separator_is_claimed_by_both_parts():
    """Real _LeadCues on one cue that came back from the form as two."""
    from plume.sim import _LeadCues

    text = "busier; sparser"
    rows = [make_row(19, density=0.4),
            make_row(20, density=0.45, applied=text, effects=_LeadCues().apply(text))]
    events = _take_end(rows, [cue_line(text, 18, rows[0]["t_s"])], realtime.END_STOPPED)
    assert not any(e["label"].endswith("too late") for e in events)
    assert all("You asked" in e["sentence"] for e in events if e["tier"] == "cue")


# ------------------------------------------------------------- a take still running

def test_a_take_still_recording_is_not_described_as_dead(tmp_path):
    rows = song(20)
    folder = write_take(tmp_path / "t", rows, cues=[cue_line("busier", 19, rows[19]["t_s"] + 0.1)])
    dead = json.dumps(record.read_take(folder)["events"])
    assert "was killed" in dead and "ended before a phrase line" in dead
    out = record.read_take(folder, live=True)
    text = json.dumps(out["events"])
    assert out["live"] is True
    assert "killed" not in text and "ended before" not in text
    assert "still being recorded" in text and "No phrase line has come yet" in text


def test_an_ending_on_disk_outranks_the_registry(tmp_path):
    folder = write_take(tmp_path / "t", song(20), ending=ending_doc(), cues=[])
    out = record.read_take(folder, live=True)
    assert out["live"] is False and "still being recorded" not in json.dumps(out["events"])


def test_a_live_take_with_no_rows_says_nothing_has_played_yet(tmp_path):
    out = record.read_take(write_take(tmp_path / "t"), live=True)
    assert "Nothing has played yet" in out["events"][0]["sentence"]
    assert out["roast_empty"].startswith("Nothing has played yet")


# ------------------------------------------------------------------ sentences, again

@pytest.mark.parametrize("roles,crash", [(("kick", "snare", "crash"), True), (("kick", "snare"), False)])
def test_the_outro_only_names_a_crash_the_style_has(roles, crash):
    rows = song(30, band_bar=8, outro_bar=24)
    for row in rows:
        row["roles"] = {r: 1.0 for r in roles}
    sentence = by_label(_take_end(rows, [], realtime.END_SONG), "outro")["sentence"]
    assert ("Crash cymbal" in sentence) is crash


def test_an_outro_starting_from_the_drums_is_an_event():
    rows = [make_row(b, entry="drums", section="intro", roles=("kick", "crash")) for b in range(24)]
    rows += [make_row(b, entry="drums", section="outro", roles=("kick", "crash")) for b in range(24, 28)]
    assert by_label(_take_end(rows, [], realtime.END_SONG), "outro")["bar"] == 24


def test_never_came_in_is_not_said_to_someone_i_heard():
    rows = song(10, heard_bar=3, band_bar=None)
    text = words(lines=record.roast(rows, ending_doc(realtime.END_SILENCE), []))
    assert "You never came in" not in text


def test_never_came_in_states_the_real_length():
    rows = song(10, bpm=60.0, heard_bar=None, band_bar=None)
    text = words(lines=record.roast(rows, ending_doc(realtime.END_SILENCE), []))
    assert "alone for 36 seconds" in text


def test_a_tap_start_row_does_not_show_an_inaudible_action():
    for action in ("jump", "extend", "hold"):
        form = record.safe_row(make_row(30, action=action, applied="louder"))["form"]
        assert "action" not in form
    assert record.safe_row(make_row(30, action="end"))["form"]["action"] == "end"
    assert record.safe_row(make_row(30, entry=None, action="jump"))["form"]["action"] == "jump"


def test_an_end_cue_during_a_fade_does_not_say_there_was_no_fade():
    rows = song(30, band_bar=8)
    rows[-1].update(fade_db=-18.0, section="outro")
    rows[-1]["form"].update(action="end", applied_cue="wrap it up")
    sentence = _take_end(rows)[-1]["sentence"]
    assert "no fade" not in sentence and "while the band was fading" in sentence


def test_a_tempo_cue_lands_on_the_bar_it_changes():
    rows = song(30, bpm=60.0)
    rows[20].update(cue_effects=["cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"])
    rows[20]["form"]["applied_cue"] = "faster"
    cues = [cue_line("faster", 19, rows[20]["t_s"] - 2.0)]
    events = _take_end(rows, cues, realtime.END_STOPPED)
    sentence = next(e for e in events if e["tier"] == "cue")["sentence"]
    assert "It landed 6.0 seconds later" in sentence      # 2 s to this row, plus its 4 s bar


def test_a_tempo_cue_on_the_last_bar_never_sounded():
    rows = song(30, bpm=60.0)
    rows[-1].update(cue_effects=["cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"])
    rows[-1]["form"]["applied_cue"] = "faster"
    assert next(e for e in _take_end(rows, [], realtime.END_STOPPED)
                if e["tier"] == "cue")["cue_outcome"] == "nothing"


@pytest.mark.parametrize("reason", [realtime.END_STOPPED, realtime.END_INTERRUPT, realtime.END_TERMINATED])
def test_endings_name_only_controls_that_exist(reason):
    sentence = _take_end(song(10), [], reason)[-1]["sentence"]
    assert "this page" not in sentence and "Cut now" not in sentence and "pressed" not in sentence


# ------------------------------------------------------------- roast links and blanks

def test_a_roast_link_opens_the_event_it_is_about():
    rows = song(40, band_bar=8, cues={25: landed("busier")})
    rows[25]["section"], rows[24]["section"] = "lift", "verse"
    cues = [cue_line("busier", 20, rows[20]["t_s"])]
    ending = ending_doc()
    events = record.translate(rows, ending, cues)
    first_on_bar = next(e for e in events if e["bar"] == 25)
    line = next(x for x in record.roast(rows, ending, cues, events=events) if "I got there" in x["text"])
    target = next(e for e in events if e["id"] == line["event"])
    assert target["cue_outcome"] == "landed" and target["bar"] == 25
    assert first_on_bar["id"] != line["event"], "this test needs a neighbour on the same bar"


def test_every_band_and_heard_line_links_to_its_own_event():
    rows = song(40, heard_bar=12, band_bar=20)
    events = record.translate(rows, ending_doc(), [])
    labels = {e["id"]: e["label"] for e in events}
    lines = record.roast(rows, ending_doc(), [], events=events)
    assert labels[lines[0]["event"]] == "bass and chords"
    assert any(labels.get(x["event"]) == "heard you" for x in lines)


def test_an_empty_roast_on_a_take_that_played_does_not_say_nothing_played(tmp_path):
    plain = [make_row(b, entry=None, playing=False) for b in range(40)]
    out = record.read_take(write_take(tmp_path / "plain", plain, ending=ending_doc(), cues=[]))
    assert out["bars"] == 40 and out["roast"] == [] and "Nothing played" not in out["roast_empty"]
    empty = record.read_take(write_take(tmp_path / "empty", ending=ending_doc(bars=0), cues=[]))
    assert "Nothing played" in empty["roast_empty"]


def test_a_log_too_big_to_read_says_so(tmp_path, monkeypatch):
    folder = write_take(tmp_path / "t", song(40), ending=ending_doc(elapsed_s=999.0), cues=[])
    monkeypatch.setattr(record, "MAX_LOG_BYTES", 100)
    out = record.read_take(folder)
    text = json.dumps(out)
    assert out["unread"] is True and "too big" in out["events"][0]["sentence"]
    assert "nothing played" not in out["events"][0]["sentence"].lower()
    assert "too big" in out["roast_empty"] and out["seconds"] == 999.0 and "before the count" not in text


# ------------------------------------------------------------- hostile take folders

HUGE = 10 ** 400


@pytest.mark.parametrize("where", ["t_s", "bpm", "density", "fade_db", "bar"])
def test_a_huge_integer_in_a_row_does_not_raise(tmp_path, where):
    rows = song(10)
    rows[3][where] = HUGE
    out = record.read_take(write_take(tmp_path / "t", rows, ending=ending_doc()))
    json.dumps(out, allow_nan=False)


def test_huge_numbers_everywhere_else_do_not_raise(tmp_path):
    folder = write_take(tmp_path / "t", song(10), cues=[{"text": "busier", "bar": HUGE, "t_s": HUGE}],
                        ending={"reason": realtime.END_STOPPED, "elapsed_s": HUGE, "bars": HUGE})
    (folder / realtime.REC_MANIFEST).write_text(json.dumps({"started_at": HUGE, "seed": HUGE}))
    json.dumps(record.read_take(folder), allow_nan=False)
    assert record.read_take(folder)["started_at"] is None


def test_extreme_floats_cannot_overflow_the_arithmetic(tmp_path):
    rows = song(10)
    rows[0]["t_s"] = -1e308
    rows[-1]["t_s"] = 1e308
    rows[1]["bpm"] = 1e308
    json.dumps(record.read_take(write_take(tmp_path / "t", rows, ending=ending_doc())), allow_nan=False)


def test_read_take_survives_a_bug_in_its_own_translation(tmp_path, monkeypatch):
    folder = write_take(tmp_path / "t", song(10), ending=ending_doc())
    real = record.roast

    def broken(rows, *a, **kw):
        if rows:
            raise ValueError("boom")
        return real(rows, *a, **kw)

    monkeypatch.setattr(record, "roast", broken)
    assert record.read_take(folder)["bars"] == 0


@pytest.mark.parametrize("name", [realtime.REC_DECISIONS, realtime.REC_MANIFEST, realtime.REC_CUES,
                                  record.REC_KEEP, realtime.REC_ENDING])
def test_a_fifo_in_a_take_folder_does_not_block_the_reader(tmp_path, name):
    import os
    import threading

    folder = write_take(tmp_path / "t", song(5), ending=ending_doc(), cues=[])
    (folder / name).unlink(missing_ok=True)
    os.mkfifo(folder / name)
    box = []
    reader = threading.Thread(target=lambda: box.append(record.read_take(folder)), daemon=True)
    reader.start()
    reader.join(5.0)
    assert box, f"a FIFO at {name} hung the reader"


@pytest.mark.parametrize("name", [realtime.REC_DECISIONS, realtime.REC_CUES])
def test_a_symlink_to_a_device_is_not_read(tmp_path, name):
    folder = write_take(tmp_path / "t", song(5), ending=ending_doc(), cues=[])
    (folder / name).unlink()
    (folder / name).symlink_to("/dev/zero")
    assert record.read_take(folder)["events"]


def test_a_symlinked_keep_file_is_not_believed(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"kept": True, "name": "LEAKED"}))
    folder = write_take(tmp_path / "t", song(5))
    (folder / record.REC_KEEP).symlink_to(outside)
    out = record.read_take(folder)
    assert out["name"] == "" and out["kept"] is False


def test_a_folder_holding_only_symlinks_or_fifos_is_not_a_take(tmp_path):
    import os

    folder = tmp_path / "t"
    folder.mkdir()
    real = tmp_path / "real"
    real.write_text("{}")
    (folder / realtime.REC_MANIFEST).symlink_to(real)
    os.mkfifo(folder / realtime.REC_DECISIONS)
    assert record.is_take_dir(folder) is False


# ------------------------------------------------------------- a take still recording

def _newest_row_cue(text, effects, *, live=True, ending=None, cues=None):
    rows = song(30, bpm=60.0)
    rows[-1]["form"]["applied_cue"] = text
    rows[-1]["cue_effects"] = effects
    events = record.translate(rows, ending, cues, live=live)
    return rows, events, next(e for e in events if e["tier"] == "cue")


FASTER_NOTE = ["cue 'faster' (tempo:faster): tempo x1.00 -> x1.08"]


@pytest.mark.parametrize("text,effects", [("faster", FASTER_NOTE), ("wrap it up", None)])
def test_a_cue_on_the_newest_row_of_a_live_take_is_waiting(text, effects):
    rows, events, cue = _newest_row_cue(text, effects)
    assert cue["cue_outcome"] == "waiting" and cue["dashed"]
    assert cue["label"] == f"{text}, waiting"
    assert "nothing acted" not in cue["sentence"]
    assert "next bar has not been written" in cue["sentence"]
    assert "did nothing" not in words(lines=record.roast(rows, None, [], events=events))


@pytest.mark.parametrize("text,effects", [("faster", FASTER_NOTE), ("wrap it up", None)])
def test_the_same_cue_on_the_last_row_of_a_finished_take_did_nothing(text, effects):
    _, _, cue = _newest_row_cue(text, effects, live=False, ending=ending_doc())
    assert cue["cue_outcome"] == "nothing"


def test_a_live_tempo_cue_that_changed_nothing_does_not_wait():
    _, _, cue = _newest_row_cue("faster", ["cue 'faster' (tempo:faster): tempo x1.50 -> x1.50"])
    assert cue["cue_outcome"] == "nothing"


# ------------------------------------------------------------- one bar, one link

def test_a_tempo_roast_line_and_its_link_name_the_same_bar():
    rows = song(40, bpm=60.0)
    rows[25]["form"]["applied_cue"] = "faster"
    rows[25]["cue_effects"] = FASTER_NOTE
    for row in rows[26:]:
        row["bpm"] = 65.0
    cues = [cue_line("faster", 20, rows[20]["t_s"] + 0.1)]
    events = record.translate(rows, ending_doc(), cues)
    line = next(ln for ln in record.roast(rows, ending_doc(), cues, events=events)
                if "I got there" in ln["text"])
    event = next(e for e in events if e["id"] == line["event"])
    assert "I got there at bar 26." in line["text"]
    assert line["bar"] == 26 and line["row"]["bar"] == 26
    assert event["bar"] == 26 and event["sentence"].startswith("Bar 26.")
    assert event["row"]["bpm"] == 65.0, "the row shown should be the one at the new speed"


# ------------------------------------------------------------- work, not just size

class _CountedRow(dict):
    reads = 0

    def __getitem__(self, key):
        type(self).reads += 1
        return super().__getitem__(key)


def test_reading_rows_for_each_cue_does_not_scale_with_the_take(monkeypatch):
    class Row(_CountedRow):
        reads = 0
    rows = [Row(r) for r in song(3000, band_bar=None)]
    cues = [cue_line(f"cue number {i}", 5, 100.0 + i) for i in range(1500)]
    record.translate(rows, ending_doc(), cues)
    assert Row.reads < 40 * (len(rows) + len(cues))


def test_a_planted_cue_log_is_not_worked_through():
    rows = song(50)
    cues = [cue_line("louder", 5, 10.0) for _ in range(record.MAX_CUE_LINES + 1)]
    rows[20]["form"]["applied_cue"] = "busier"
    events = record.translate(rows, ending_doc(), cues)
    assert not [e for e in events if e["label"].startswith("louder")]


def test_one_cue_line_cannot_split_into_more_than_the_cap():
    rows = song(50)
    cues = [cue_line("; ".join(["louder"] * (record.MAX_CUE_LINES + 1)), 5, 10.0)]
    assert record.translate(rows, ending_doc(), cues)


def test_cues_on_rows_cost_no_more_than_the_cap(monkeypatch):
    calls = []
    real = record.parse_cue
    monkeypatch.setattr(record, "parse_cue", lambda text: calls.append(text) or real(text))
    rows = song(400)
    many = "; ".join(f"louder {i}" for i in range(500))
    for i, row in enumerate(rows):
        # Changing entry and section on each row adds plan events, and each shows its row.
        row["entry"], row["section"] = ("drums", "band", "full")[i % 3], ("verse", "lift", "break")[i % 3]
        row["form"]["applied_cue"] = many
        row["form"]["dropped_cue"] = many
        row["cue_effects"] = [f"cue 'louder {i}' x" for i in range(2000)]
    events = record.translate(rows, ending_doc(), [])
    assert len([e for e in events if e["tier"] == "cue"]) <= record.MAX_CUE_EVENTS
    assert len(calls) < 6 * record.MAX_CUE_EVENTS


def test_a_huge_cue_text_is_cut_before_anything_parses_it(monkeypatch):
    seen = []
    real = record.parse_cue
    monkeypatch.setattr(record, "parse_cue", lambda text: seen.append(len(text)) or real(text))
    rows = song(30)
    rows[20]["form"]["applied_cue"] = "x" * 2_000_000
    record.translate(rows, ending_doc(), [cue_line("y" * 2_000_000, 5, 10.0)])
    assert max(seen, default=0) <= record.MAX_CUE_CHARS


# ------------------------------------------------------- a sentence has to be true

def _plain(bars=12, **at_bar):
    """Plain lead rows.  ``at_bar`` maps b<bar> to extra make_row arguments."""
    return [make_row(b, entry=None, **at_bar.get(f"b{b}", {})) for b in range(bars)]


def test_silence_with_the_band_in_only_knows_about_the_mic():
    rows = song(40, cues={b: {"fade_db": -9.0} for b in range(34, 40)})
    events = record.translate(rows, ending_doc(realtime.END_SILENCE))
    assert "The mic went silent" in by_label(events, "fade started")["sentence"]
    assert "The mic went silent for 30 seconds, so I faded out" in events[-1]["sentence"]
    assert "I heard nothing" not in words(events)


def test_silence_over_drums_alone_really_was_hearing_nothing():
    rows = song(9, heard_bar=None, band_bar=None)
    assert "I heard nothing from you" in record.translate(rows, ending_doc(realtime.END_SILENCE))[-1]["sentence"]


def test_silence_with_no_rows_claims_no_hearing():
    assert "I heard nothing" not in record.translate([], ending_doc(realtime.END_SILENCE))[0]["sentence"]


@pytest.mark.parametrize("at,row_says,end_says", [
    ("next-phrase", "stopped at this phrase line", "ended the take at a phrase line"),
    ("next-section", "stopped at this section line", "ended the take at a section line"),
    ("now", "without waiting for a phrase line", "without waiting for a phrase line"),
    ("next-bar", "without waiting for a phrase line", "without waiting for a phrase line"),
])
def test_an_end_cue_says_what_it_waited_for(at, row_says, end_says):
    rows = song(21, cues={20: {"applied": "wrap it up", "action": "end"}})
    cue = cue_line("wrap it up", 18, rows[18]["t_s"]) | {"at": at}
    events = record.translate(rows, ending_doc(realtime.END_FORM), [cue])
    assert row_says in by_label(events, "cue ended it")["sentence"]
    assert end_says in events[-1]["sentence"]
    if at != "next-phrase":
        assert "at this phrase line" not in words(events) and "at a phrase line" not in words(events)


def test_an_end_cue_with_no_log_claims_no_phrase_line():
    rows = song(21, cues={20: {"applied": "wrap it up", "action": "end"}})
    events = record.translate(rows, ending_doc(realtime.END_FORM))
    assert "phrase line" not in by_label(events, "cue ended it")["sentence"]
    assert "phrase line" not in events[-1]["sentence"] and "A cue ended the take" in events[-1]["sentence"]


def test_an_end_cue_that_landed_a_row_early_says_the_band_stopped_later():
    rows = song(22, cues={20: {"applied": "wrap it up; go to the chorus", "action": "jump"},
                          21: {"action": "end"}})
    sentence = record.translate(rows, ending_doc(realtime.END_FORM))[-1]["sentence"]
    assert "stopped one bar later, with no fade" in sentence and "phrase line" not in sentence


@pytest.mark.parametrize("at,words_for_it", [
    ("next-section", "before the next section came"), ("next-bar", "before the next bar came"),
    ("now", "before the next bar came"), ("next-phrase", "before a phrase line came"),
    (None, "before it was used"),
])
def test_a_cue_too_late_names_what_it_was_waiting_for(at, words_for_it):
    rows = song(40)
    cue = cue_line("busier", 38, rows[38]["t_s"]) | {"at": at}
    assert words_for_it in by_label(record.translate(rows, ending_doc(), [cue]), "busier, too late")["sentence"]


@pytest.mark.parametrize("at,words_for_it", [
    ("next-section", "The next section has not come yet"), ("now", "The next bar has not come yet"),
    ("next-phrase", "No phrase line has come yet"), (None, "It has not been used yet"),
])
def test_a_cue_still_waiting_names_what_it_is_waiting_for(at, words_for_it):
    rows = song(40)
    cue = cue_line("busier", 38, rows[38]["t_s"]) | {"at": at}
    events = record.translate(rows, None, [cue], live=True)
    assert words_for_it in by_label(events, "busier, waiting")["sentence"]


@pytest.mark.parametrize("at,sentence_part,roast_part", [
    ("next-section", "before the next section came", "I waited for the next section"),
    ("next-phrase", "before a phrase line came", "I waited for a phrase line"),
])
def test_an_expired_cue_names_what_it_waited_for(at, sentence_part, roast_part):
    note = f"cue 'sparser' expired unapplied after 16 bars (ttl 16, at {at})"
    rows = song(40, band_bar=None, cues={21: {"dropped": "sparser", "note": note}})
    cues = [cue_line("sparser", 5, rows[5]["t_s"]) | {"at": at}]
    assert sentence_part in by_label(record.translate(rows, ending_doc(), cues), "sparser, expired")["sentence"]
    assert any(roast_part in line["text"] for line in record.roast(rows, ending_doc(), cues))


def test_a_section_cue_the_map_ignored_did_not_land():
    note = "cue 'go to the chorus' (section:chorus): no such section in the map \u2014 ignored"
    rows = _plain(b8=dict(applied="go to the chorus", action="jump", note=note))
    events = record.translate(rows, ending_doc())
    assert not [e for e in events if e["label"].endswith(" landed")]
    assert by_label(events, "a cue, did nothing")["dashed"]
    assert '"go to the chorus" was logged' in by_label(events, "a cue, did nothing")["sentence"]


def test_an_energy_cue_with_nowhere_to_go_did_not_land_on_someone_elses_jump():
    note = ("cue 'go to the verse' (section:verse) -> section 1 'drift-a'; "
            "cue 'more energy' (energy:up): no hotter section in the map \u2014 ignored")
    rows = _plain(b8=dict(applied="go to the verse; more energy", action="jump", note=note))
    events = record.translate(rows, ending_doc())
    assert by_label(events, "go to the verse landed")
    assert by_label(events, "more energy, did nothing")


def test_the_same_jump_cue_twice_reads_each_note_once():
    note = ("cue 'go to the verse' (section:verse) -> section 1 'drift-a'; "
            "cue 'go to the verse' (section:verse): already at section 1")
    rows = _plain(b8=dict(applied="go to the verse; go to the verse", action="jump", note=note))
    events = record.translate(rows, ending_doc())
    assert len([e for e in events if e["label"] == "go to the verse landed"]) == 1
    assert len([e for e in events if e["label"] == "a cue, did nothing"]) == 1


def test_a_jump_with_no_note_about_this_cue_says_it_cannot_tell():
    rows = _plain(b8=dict(applied="louder", action="jump"))
    event = by_label(record.translate(rows, ending_doc()), "louder, not sure")
    assert event["dashed"] and "does not say this cue did it" in event["sentence"]
    assert record.portrait(rows, ending_doc(), [event], 1)["nicks"] == 0


def test_a_cap_she_set_is_not_called_something_she_did_not_choose():
    doc = ending_doc(realtime.END_MAX_MINUTES) | {"max_minutes": 5.0}
    sentence = record.translate(song(20), doc)[-1]["sentence"]
    assert "You did not choose" not in sentence and "5-minute time cap" in sentence
    assert "You did not choose" not in record.translate(song(20), ending_doc(realtime.END_MAX_MINUTES))[-1]["sentence"]
    assert "The time cap ended this take" in record.translate(song(20), {"reason": realtime.END_MAX_MINUTES})[-1]["sentence"]


def test_a_plain_lead_stopped_early_claims_no_count():
    sentence = record.translate([], ending_doc(realtime.END_STOPPED, bars=0))[0]["sentence"]
    assert "count" not in sentence and "first bar" in sentence


def test_exactly_means_exactly():
    rows = song(10, bpm=120.0, band_bar=None)
    assert any("I played exactly that" in l["text"] for l in record.roast(rows, ending_doc()))
    rows[5]["bpm"] = 121.0     # 0.8%, inside what counts as holding it
    text = json.dumps(record.roast(rows, ending_doc()))
    assert "exactly" not in text and "within 1%" in text


def test_many_cues_long_before_the_end_do_not_hide_the_one_that_ended_it():
    rows = song(1101, band_bar=None)
    for row in rows[:1100]:
        row["form"]["applied_cue"] = "busier"
    rows[-1]["form"].update(applied_cue="wrap it up", action="end")
    sentence = record.translate(rows, ending_doc(realtime.END_FORM))[-1]["sentence"]
    assert "A cue ended the take" in sentence and "The arrangement ran out" not in sentence


def test_too_many_cues_on_the_last_row_do_not_blame_the_arrangement():
    rows = song(21)
    text = "; ".join(["busier"] * 16 + ["wrap it up"])
    rows[-1]["form"].update(applied_cue=text, action="end")
    sentence = record.translate(rows, ending_doc(realtime.END_FORM))[-1]["sentence"]
    assert "The arrangement ran out" not in sentence and "can't say" in sentence


def test_cues_past_the_limit_are_not_called_too_late():
    rows = song(40)
    rows[20]["form"]["applied_cue"] = "; ".join(["busier"] * 20)
    cues = [cue_line("busier", 18, rows[18]["t_s"]) for _ in range(20)]
    text = words(record.translate(rows, ending_doc(), cues))
    assert "too late" not in text and "before a phrase line came" not in text
    assert "more cues than I read" in text


@pytest.mark.parametrize("text", [
    "stop calling me maria", "go to my exs house", "ending things with sam", "go to the sam",
    "hit the road maria", "stop it maria", "more of maria", "jazz with sam"])
def test_a_known_word_inside_other_words_is_not_echoed(text):
    assert parse_cue(text)["kind"] != "unknown"     # the parser does act on these
    assert record.cue_name(text) is None
    rows = song(21, cues={20: {"applied": text, "action": "end"}})
    out = words(record.translate(rows, ending_doc(realtime.END_FORM), [cue_line(text, 18, rows[18]["t_s"])]))
    assert text not in out and "maria" not in out and "sam" not in out
    assert text not in json.dumps(record.safe_row(rows[20]))


@pytest.mark.parametrize("text", [
    "busier", "sparser", "wrap it up", "stop", "take it out", "faster", "half time", "louder",
    "more energy", "next section", "go to the chorus", "back to the verse", "chorus", "darker",
    "more reverb", "jazz", "busier please", "a bit louder", "Wrap It Up"])
def test_a_known_cue_is_repeated(text):
    assert record.cue_name(text) == text.lower()


def test_a_cue_understood_but_not_repeated_is_not_called_not_understood():
    rows = song(40, cues={20: {"applied": "busier for maria"}})
    sentence = by_label(record.translate(rows, ending_doc()), "a cue, did nothing")["sentence"]
    assert "maria" not in sentence and "did not understand" not in sentence
    rows = song(40, cues={20: {"applied": "tell maria hello"}})
    assert "did not understand" in by_label(record.translate(rows, ending_doc()), "a cue, did nothing")["sentence"]


@pytest.mark.parametrize("chord,shown", [
    ("Fmaj7", True), ("Am", True), ("Bbm7b5", True), ("C#7sus4", True), ("G", True), ("C/E", True),
    ("CANARYTEXT", False), ("Dad", False), ("Ella", False), ("G-string", False), ("Fred", False)])
def test_a_chord_has_to_be_a_chord(chord, shown):
    assert ("chord" in record.safe_row(make_row(30, chord=chord))) is shown


def test_a_mark_on_the_sheet_always_keeps_what_happened():
    for suffix in (" landed", ", did nothing", ", expired", ", too late", ", waiting", ", no effect", ", not sure"):
        for name in ("busier", "go to the turnaround section", None):
            label = record._cue_label(name, suffix)
            assert label.endswith(suffix) and len(label) <= record.MAX_CUE_LABEL_CHARS


def test_roles_and_actions_are_filtered_by_shape_too():
    row = make_row(30, entry=None, action="<b>human</b>", roles=("kick", "<img src=x>"))
    safe = record.safe_row(row)
    assert safe["roles"] == ["kick"] and "action" not in safe.get("form", {})


def test_what_the_roast_keeps_for_itself_does_not_reach_the_page():
    event = record.translate(song(40, cues={20: landed("busier")}), ending_doc(), [])[1]
    public = record.public_event({**event, "cue_outcome": "landed", "asked": {"who": "You"}})
    assert "cue_outcome" not in public and "asked" not in public


def test_cues_on_one_row_share_its_filtered_copy(monkeypatch):
    rows = song(40, cues={20: {"applied": "busier; sparser; louder"}})
    calls = []
    real = record.safe_row
    monkeypatch.setattr(record, "safe_row", lambda row, **kw: (calls.append(row["bar"]), real(row, **kw))[1])
    record.translate(rows, ending_doc())
    assert calls.count(20) == 1


# ------------------------------------------------- what the looper and the routes add

def loop_line(text, bar, t_s, *, op, ok=True, source="gui"):
    return {**cue_line(text, bar, t_s, source), "at": "now", "ttl_bars": 0,
            "loop": {"op": op, "ok": ok}}


def test_a_loop_cue_is_answered_so_it_never_reads_as_too_late():
    rows = song(40)
    events = record.translate(rows, ending_doc(), [loop_line("make a loop", 12, rows[12]["t_s"], op="record")])
    event = by_label(events, "loop cue")
    assert event["bar"] == 12
    assert "You asked to record a loop at bar 12" in event["sentence"]
    assert "does not promise a loop came out of it" in event["sentence"]
    assert "take ended" not in event["sentence"] and not event["dashed"]


def test_a_refused_loop_cue_says_so_and_changes_nothing():
    rows = song(40)
    events = record.translate(rows, ending_doc(), [loop_line("stop the loop", 5, rows[5]["t_s"], op="stop", ok=False)])
    event = by_label(events, "loop cue, refused")
    assert "The looper said no, so nothing changed." in event["sentence"] and event["dashed"]


def test_words_the_looper_could_not_follow_started_nothing():
    rows = song(40)
    events = record.translate(rows, ending_doc(), [loop_line("loop thing", 5, rows[5]["t_s"], op="unclear", ok=False)])
    assert "could not follow the words, so no take started" in by_label(events, "loop cue, refused")["sentence"]


def test_a_loop_cue_before_the_first_bar_sits_at_the_start_not_the_end():
    rows = song(40)
    event = by_label(record.translate(rows, ending_doc(), [loop_line("make a loop", None, 0.5, op="record")]),
                     "loop cue")
    assert event["bar"] == 0 and "before the first bar" in event["sentence"]


@pytest.mark.parametrize("planted", [
    {"op": "<img src=x>", "ok": True}, {"op": ["record"], "ok": True}, {"op": {}, "ok": "yes"}, {},
])
def test_a_planted_loop_answer_cannot_reach_the_page_or_break_it(planted):
    rows = song(10)
    line = {**cue_line("secret name loop", 2, rows[2]["t_s"]), "loop": planted}
    text = words(record.translate(rows, ending_doc(), [line]))
    assert "<img" not in text and "about a loop" in text


def test_a_loops_name_is_never_echoed():
    rows = song(10)
    text = words(record.translate(rows, ending_doc(),
                                  [loop_line("make a loop called maria", 2, rows[2]["t_s"], op="record")]))
    assert "maria" not in text


def test_a_loop_cue_cannot_claim_a_band_cue_with_the_same_words():
    rows = song(40, cues={20: landed("busier")})
    log = [loop_line("busier", 1, rows[1]["t_s"], op="list"), cue_line("busier", 18, rows[18]["t_s"], "cli")]
    event = by_label(record.translate(rows, ending_doc(), log), "busier landed")
    assert "Another terminal asked for busier" in event["sentence"]


HELD = {"applied": "faster",
        "effects": ["cue 'faster' (tempo:faster): tempo x1.00 -> x1.06",
                    "tempo cue NOT applied yet: a loop is holding 120.0 bpm until the last one is dropped"]}


@pytest.mark.parametrize("live", [False, True])
def test_a_tempo_cue_under_a_held_loop_did_not_land(live):
    """The note reads as a change.  The bpm did not move, so "landed" would be false."""
    rows = song(40 if not live else 21, cues={20: HELD})
    events = record.translate(rows, None if live else ending_doc(), [cue_line("faster", 18, rows[18]["t_s"])], live=live)
    event = by_label(events, "faster, on hold")
    assert event["cue_outcome"] == "held" and event["dashed"]
    assert "A loop was holding the tempo" in event["sentence"]
    assert "landed" not in words(events)
    assert "NOT applied" not in words(events)


def test_a_take_started_from_a_loop_was_not_counted():
    rows = song(40, band_bar=8)
    for row in rows:
        row["tempo_from"] = "loop"
    events = record.translate(rows, ending_doc())
    lines = record.roast(rows, ending_doc(), events=events)
    text = words(events, lines)
    assert "Drums came in on your loop" in text and "Your loop set 53 bpm" in text
    assert "your count" not in text and "You counted" not in text


def test_a_counted_take_still_says_it_was_counted():
    events = record.translate(song(40), ending_doc())
    text = words(events, record.roast(song(40), ending_doc(), events=events))
    assert "Drums came in on your count" in text and "You counted 53 bpm" in text


@pytest.mark.parametrize("manifest,route", [
    ({"midi_port": "Fake Keys", "input": {"source": "room"}}, "midi"),
    ({"midi_port": "Fake Keys", "input": {"source": "direct"}}, "midi"),
    ({"midi_port": None, "input": {"source": "direct"}}, "direct"),
    ({"midi_port": None, "input": {"source": "room"}}, "room"),
    ({"input": {"source": "mystery"}}, "unknown"),
    ({"input": "direct"}, "unknown"),
    ({}, "unknown"),
    (None, "unknown"),
])
def test_the_route_comes_from_the_manifest(manifest, route):
    assert record.input_route(manifest) == route


def faded_song():
    return song(40, band_bar=8, cues={b: {"fade_db": -6.0} for b in range(30, 34)})


@pytest.mark.parametrize("route", ["direct", "midi"])
def test_an_input_the_band_cannot_reach_never_blames_the_mic(route):
    rows = faded_song()
    ending = ending_doc(realtime.END_SILENCE)
    events = record.translate(rows, ending, route=route)
    lines = record.roast(rows, ending, events=events, route=route)
    text = words(events, lines)
    assert "The mic" not in text and "this mic" not in text
    assert "no idea what you did" not in text
    assert "I stopped hearing you, so I began to fade" in text
    assert "I stopped hearing you, so I faded out" in text
    # That wait depends on the tempo, so no number can be true of every take.
    assert "30 seconds" not in text
    assert "I could still hear you" in text
    # Class C: being able to hear is not leave to judge.
    assert "drag" not in text


@pytest.mark.parametrize("route", ["direct", "midi"])
def test_an_input_that_never_heard_a_note_really_did_wait_thirty_seconds(route):
    rows = song(20, heard_bar=None, band_bar=None, cues={b: {"fade_db": -6.0} for b in range(16, 20)})
    ending = ending_doc(realtime.END_SILENCE)
    text = words(record.translate(rows, ending, route=route))
    assert "Nothing from you for 30 seconds" in text
    assert "I heard nothing from you for 30 seconds" in text
    assert "I stopped hearing you" not in text


@pytest.mark.parametrize("route", ["room", "unknown"])
def test_a_room_mic_and_an_older_take_keep_the_cautious_copy(route):
    rows = faded_song()
    ending = ending_doc(realtime.END_SILENCE)
    events = record.translate(rows, ending, route=route)
    text = words(events, record.roast(rows, ending, events=events, route=route))
    assert "The mic went silent for 30 seconds" in text
    assert "no idea what you did" in text and "this mic" in text


def test_a_direct_take_that_never_played_has_no_mic_to_mention():
    events = record.translate([], ending_doc(), route="direct")
    assert "input recording" in events[0]["sentence"] and "mic" not in events[0]["sentence"]


def test_read_take_reports_the_route_it_used(tmp_path):
    folder = write_take(tmp_path / "t", song(20), ending=ending_doc())
    assert record.read_take(folder)["route"] == "unknown"
    manifest = json.loads((folder / realtime.REC_MANIFEST).read_text())
    (folder / realtime.REC_MANIFEST).write_text(json.dumps({**manifest, "midi_port": "Fake Keys"}))
    take = record.read_take(folder)
    assert take["route"] == "midi"
    assert "I could still hear you" in json.dumps(take["roast"])


def loop_rows(count=30):
    rows = song(count, heard_bar=None, band_bar=None)
    for row in rows:
        row["tempo_from"] = "loop"
    return rows


def test_silence_after_a_loop_does_not_say_she_never_came_in():
    text = words(lines=record.roast(loop_rows(), ending_doc(realtime.END_SILENCE)))
    assert "never came in" not in text and "drums alone" not in text
    assert "Your loop set the tempo" in text


def test_silence_after_asking_for_a_loop_does_not_say_she_never_came_in():
    log = [{"text": "make a loop", "bar": 2, "t_s": 5.0, "source": "gui",
            "loop": {"op": "record", "ok": True}}]
    rows = song(30, heard_bar=None, band_bar=None)
    text = words(lines=record.roast(rows, ending_doc(realtime.END_SILENCE), log))
    assert "never came in" not in text and "drums alone" not in text


@pytest.mark.parametrize("route", ["direct", "midi"])
def test_only_a_room_mic_is_said_to_take_three_bars(route):
    rows = song(40, heard_bar=12, band_bar=17)
    said = words(events=record.translate(rows, ending_doc(), route=route))
    assert "I was sure I could hear you by here." in said and "three bars" not in said
    text = words(lines=record.roast(rows, ending_doc(), route=route))
    assert "You let the drums run" not in text and "the three I need" not in text


def test_a_room_mic_still_gets_its_three_bars():
    rows = song(40, heard_bar=12, band_bar=17)
    assert "about three bars" in words(events=record.translate(rows, ending_doc(), route="room"))
    assert "about 9 bars" in words(lines=record.roast(rows, ending_doc(), route="room"))


def test_a_late_entry_counts_from_the_first_bar_played():
    rows = song(40, heard_bar=20, band_bar=25)
    started_at_8 = [r for r in rows if r["bar"] >= 8]
    assert "about 9 bars" in words(lines=record.roast(started_at_8, ending_doc(), route="room"))
    short = [r for r in song(40, heard_bar=13, band_bar=25) if r["bar"] >= 8]
    assert "You let the drums run" not in words(lines=record.roast(short, ending_doc(), route="room"))


def test_midi_and_direct_together_say_input_for_the_recording():
    manifest = {"midi_port": "Keys", "input": {"source": "direct"}}
    assert record.input_route(manifest) == "midi"
    assert record.recorded_from(manifest) == "input"
    assert record.recorded_from({"midi_port": "Keys", "input": {"source": "room"}}) == "mic"
    assert record.recorded_from(None) == "mic"
    said = record.translate([], ending_doc(realtime.END_STOPPED), route="midi", recorded="input")
    assert "The input recording" in said[-1]["sentence"]


def test_a_loop_cue_with_no_bar_played_is_still_reported():
    log = [{"text": "make a loop", "bar": None, "t_s": 1.0, "source": "gui",
            "loop": {"op": "record", "ok": True}},
           {"text": "mute all loops", "bar": None, "t_s": 2.0, "source": "cli",
            "loop": {"op": "mute", "ok": False}}]
    events = record.translate([], ending_doc(realtime.END_STOPPED), log)
    assert [e["tier"] for e in events] == ["cue", "cue", "end"]
    assert "before the first bar" in events[0]["sentence"] and "said yes" in events[0]["sentence"]
    assert "said no" in events[1]["sentence"] and events[1]["dashed"] is True
    assert [e["id"] for e in events] == ["ev-0", "ev-1", "ev-2"]
    assert all(set(record.public_event(e)) >= {"bar", "row", "sentence"} for e in events)


def test_a_log_too_big_to_read_does_not_claim_before_the_first_bar():
    log = [{"text": "make a loop", "bar": None, "t_s": 1.0, "source": "gui",
            "loop": {"op": "record", "ok": True}}]
    events = record.translate([], ending_doc(), log, unread=True)
    assert [e["tier"] for e in events] == ["end"]


# ------------------------------------------------------- when the band followed

def _song_with_follow(moved, *, bar=30, bars=3):
    rows = song(40)
    for r in rows:
        if r.get("entry") != "drums":
            r["follow"] = {"mode": "follow", "trusted": True, "gain_db": 0.0, "density": 1.0,
                           "moved": [], "bars": bars}
    rows[bar]["follow"]["moved"] = moved
    return rows


@pytest.mark.parametrize("route", ["direct", "midi"])
def test_a_move_the_band_made_to_follow_is_a_moment(route):
    events = record.translate(_song_with_follow(["softer"]), ending_doc(), route=route)
    event = by_label(events, "band followed")
    assert event["bar"] == 30 and event["tier"] == "plan"
    assert "softer" in event["sentence"] and "3 bars" in event["sentence"]
    assert "came down" in event["sentence"]


def test_on_a_room_mic_it_only_says_what_the_mic_heard():
    events = record.translate(_song_with_follow(["louder"], bars=4), ending_doc(), route="room")
    sentence = by_label(events, "band followed")["sentence"]
    assert "mic" in sentence and "4 bars" in sentence and "you" not in sentence.lower().split()


def test_each_way_it_followed_gets_its_own_plain_sentence():
    events = record.translate(_song_with_follow(["softer", "sparser"]), ending_doc(), route="direct")
    said = [e["sentence"] for e in events if e["label"] == "band followed"]
    assert len(said) == 2 and len(set(said)) == 2


def test_a_moved_word_it_does_not_know_never_reaches_the_page():
    events = record.translate(_song_with_follow(["human sounds tired"], bars=99), ending_doc(),
                              route="direct")
    assert not [e for e in events if e["label"] == "band followed"]
    assert_clean(words(events))


def test_steady_following_says_nothing():
    events = record.translate(_song_with_follow([]), ending_doc(), route="direct")
    assert not [e for e in events if e["label"] == "band followed"]
