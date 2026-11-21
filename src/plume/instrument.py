"""What she plays, when she says: an optional choice, used only where it plainly helps.

Not set is the default and changes nothing: every take that was never told is the take
it was before this choice existed.  Told, a take uses it in two small ways:

- **The strum.**  The style guess counts a strummed chord for folk and against every
  other style (plume.playstyle).  A strum is strings sounding together; a bass line or a
  single voice or lead line plays one note at a time, so a long attack there is a slide
  or a swell.  On those the strum is not listened for (:func:`hears_strum`), and the
  guess is made on the rest of what was heard.  A named style is never guessed, so this
  never touches one.
- **Her register.**  The band's parts that sit where she plays step back a little
  (:data:`ROOM_DB`): under a bass player the band's bass and sub, under keys the band's
  keys and pad, under a lead line the band's lead and arpeggio.  Nothing is taken out
  and nothing else moves, so the band is the same band, leaving her room.

Guitar and other are recorded and change nothing yet: a guitar is what the take's ears
were written against, and other says nothing a take could use.
"""
from __future__ import annotations

import json

#: Every choice, in the order a picker lists them.  None is not set.
INSTRUMENTS = ("guitar", "bass", "keys", "lead", "other")
LABELS = {"guitar": "Guitar", "bass": "Bass", "keys": "Keys", "lead": "Voice or lead line",
          "other": "Something else"}
NOT_SET_LABEL = "Not set"
#: Other words for a choice, as someone might type them.
_ALIASES = {"piano": "keys", "keyboard": "keys", "keyboards": "keys", "synth": "keys",
            "voice": "lead", "vocals": "lead", "singing": "lead", "melody": "lead",
            "not set": None, "none": None}
#: How far the band's parts in her register step back, dB on the role's own level.
ROOM_DB = {"bass": {"bass": -6.0, "sub": -6.0},
           "keys": {"keys": -4.0, "pad": -4.0},
           "lead": {"lead": -4.0, "arp": -4.0}}
#: The instruments that play one note at a time, so never strum.
_ONE_NOTE = frozenset({"bass", "lead"})


def canonical(value) -> str | None:
    """A choice's own name from what was given, None for not set.  ValueError otherwise."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("instrument must be text")
    word = " ".join(value.lower().split())
    if not word:
        return None
    if word in INSTRUMENTS:
        return word
    if word in _ALIASES:
        return _ALIASES[word]
    raise ValueError(f"unknown instrument {value!r} (instruments: {', '.join(INSTRUMENTS)})")


def hears_strum(instrument: str | None) -> bool:
    """Whether a strum is listened for on this instrument: not on one that plays one
    note at a time."""
    return instrument not in _ONE_NOTE


def make_room(spec, instrument: str | None):
    """``spec`` (a style's name, or a style already loaded and changed) with the band's
    parts in her register stepped back (:data:`ROOM_DB`).  With nothing to step back,
    ``spec`` itself, untouched.  A dict given is copied, never changed."""
    trims = ROOM_DB.get(instrument)
    if not trims:
        return spec
    from .duet import _ROLE_GAIN_DB  # noqa: PLC0415 - the engine's own role defaults
    from .styles_lib import load_style  # noqa: PLC0415
    style = load_style(spec) if isinstance(spec, str) else json.loads(json.dumps(spec))
    gains = dict(style.get("role_gain_db") or {})
    for role, db in trims.items():
        gains[role] = float(gains.get(role, _ROLE_GAIN_DB.get(role, -6.0))) + db
    style["role_gain_db"] = gains
    return style
