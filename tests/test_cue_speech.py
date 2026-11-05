"""parse_cue on loose speech (2026-11-06).

Found offline against a hand-labelled set of 40 instructions and 15 lines of
room chatter: "stop talking, I'm just thinking out loud" ended the song, "back
to the top" and "strip it back to just the groove" became jumps to sections
that do not exist, and "more coffee later?" made the band louder.

These are the plain misses, fixed one rule at a time.  The parser stays a
keyword parser; it does not try to understand English, and the trade-offs it
still makes are written down in ``plume.mind`` beside the rules.
"""
import pytest

from plume.mind import parse_cue


def kv(text):
    p = parse_cue(text)
    return p["kind"], p["value"]


# ------------------------------------------------------------ chatter is not a cue

@pytest.mark.parametrize("text", [
    "stop talking, I'm just thinking out loud",
    "more coffee later?",
    "I need more sleep",
    "my hands need a stop",
    "is there more?",
    "is it too loud for the neighbours?",
])
def test_talk_with_a_loose_cue_word_in_it_is_not_a_cue(text):
    assert kv(text) == ("unknown", None)


# ------------------------------------------------------------ real ends stay easy

@pytest.mark.parametrize("text", [
    "wrap it up", "take it home", "end", "end it", "stop", "ok stop", "stop now",
    "take us home after this one", "final time through", "last time", "Stop!",
])
def test_real_end_phrases_still_end(text):
    assert kv(text) == ("end", "out")


# ------------------------------------------------------------ "back to" is not a jump on its own

@pytest.mark.parametrize("text", ["back to the top", "take it to the top",
                                  "go back to the start", "from the top"])
def test_the_top_is_the_intro(text):
    assert kv(text) == ("section", "intro")


@pytest.mark.parametrize("text,want", [
    ("strip it back to just the groove", ("texture", "sparser")),
    ("listen to the drums", ("unknown", None)),
    ("back to normal", ("unknown", None)),
])
def test_a_loose_to_the_does_not_invent_a_section(text, want):
    assert kv(text) == want


@pytest.mark.parametrize("text,want", [
    ("back to the verse", ("section", "verse")),
    ("go to the arc-c", ("section", "arc c")),        # a composer's own name
    ("jump to peak-b", ("section", "peak b")),
    ("into the drop", ("section", "drop")),
])
def test_named_jumps_still_jump(text, want):
    assert kv(text) == want


# ------------------------------------------------------------ plain misses

@pytest.mark.parametrize("text,want", [
    ("let's pump it up", ("energy", "up")),
    ("let's double it up", ("tempo", "double")),
    ("ease the tempo back a touch", ("tempo", "slower")),
    ("give it some sparkle", ("texture", "brighter")),
    ("drench it in reverb", ("texture", "wet")),
    ("no reverb", ("texture", "dry")),
    ("everybody drop out except bass", ("texture", "sparser")),
    ("can you lay back, give me some room", ("texture", "sparser")),
    ("too busy, calm the hats down", ("texture", "sparser")),
    ("too loud", ("energy", "down")),
    ("too fast", ("tempo", "slower")),
    ("too dark", ("texture", "brighter")),
    ("too much reverb", ("texture", "dry")),
    ("fill it in more", ("texture", "busier")),
])
def test_plain_misses(text, want):
    assert kv(text) == want
