"""What she plays, when she says: an optional choice, used only where it plainly helps.

Not set is the default and changes nothing: every take that was never told is the take
it was before this choice existed.  Told, a take uses it in two small ways:

- **The strum.**  The style guess counts a strummed chord for folk and against every
  other style (plume.playstyle).  A strum is strings sounding together; a bass line or a
  single voice or lead line mostly plays one note at a time, so a long attack there is
  more likely a slide or a swell than a strum.  On those folk costs a little in the
  guess (:data:`STYLE_PRIOR`): a close call with folk goes the other way, while a strum
  heard plainly, such as double stops strummed on a bass, still gets folk.  A named
  style is never guessed, so this never touches one.
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
#: Other words for a choice, as someone might type them.
_ALIASES = {"piano": "keys", "keyboard": "keys", "keyboards": "keys",
            "voice": "lead", "vocals": "lead", "singing": "lead", "melody": "lead",
            "not set": None, "none": None}
#: How far the band's parts in her register step back, dB on the role's own level.
ROOM_DB = {"bass": {"bass": -6.0, "sub": -6.0},
           "keys": {"keys": -4.0, "pad": -4.0},
           "lead": {"lead": -4.0, "arp": -4.0}}
#: What each choice costs a style in the guess (plume.playstyle.score_styles), on the
#: same scale as a reading a full step outside a style's band (1).  Bass and a lead
#: line mostly play one note at a time, so folk, which a strum makes, costs a little:
#: half of one reading's full misfit.  A strum share just inside folk's band then
#: loses to the next best style; one heard plainly, a few points further in, does not.
STYLE_PRIOR = {"bass": {"folk": 0.5}, "lead": {"folk": 0.5}}


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


def style_prior(instrument: str | None) -> dict | None:
    """What this choice holds against a style in the guess (:data:`STYLE_PRIOR`), None
    for not set and for a choice that holds nothing."""
    prior = STYLE_PRIOR.get(instrument)
    return dict(prior) if prior else None


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
