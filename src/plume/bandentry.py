"""When the band joins a tap-started take: drums first, then you, then them.

The order is Juniper's (2026-11-02).  She taps four, the drums start and hold
that tempo, and she plays over them for a while.  Only once plume has heard her
settle in one key do the bass and chords come in, in that key, with the melody
a few bars after them.

Pure: a function of the (bar, harmony, playing) stream, so it is testable
without a device.  Key commitment itself is :class:`HarmonyGovernor`'s job; what
this adds is that bars she sat out are no evidence of a key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .governor import HarmonyGovernor
from .listen.harmony import HarmonyState
from .theory import parse_key

HARMONY_ROLES = ("bass", "pad", "keys", "shimmer")
MELODY_ROLES = ("lead", "arp")

DRUMS, BAND, FULL = "drums", "band", "full"

#: Detect mode: after this many heard bars with no four running on one key, take the
#: tracker's latest word for it rather than leave the player on drums for the whole take.
DETECT_KEY_FALLBACK_BARS = 16
#: Onsets per beat that make a player busy, and that make one sparse enough for trap.
#: Like every number in detect_style, a first guess nobody has calibrated.
_BUSY_ONSETS_PER_BEAT = 1.0
_SPARSE_ONSETS_PER_BEAT = 0.5


def detect_style(bpm: float, mode: str | None = None, busy: float | None = None) -> str:
    """A style for a player who did not name one: a guess, and the page says so.

    The tempo does most of the work, and each style's own range sets the bands.  The
    mode and how busy the playing is (onsets per beat) only split a band.  None means
    not known yet, or on a room mic not knowable, and counts as neither minor nor busy.
    The thresholds are uncalibrated: they have never met a real player.

    A minor key with room in it is the only sign this rule has of r&b, hip hop or house,
    so those three are claims about tempo and mode and nothing more.  It hears nothing
    that could tell jazz from anything else played fast, so being fast is not enough:
    jazz only answers above 150, where no other style's range reaches, for a player a
    direct input knows is leaving a lot of room, and a sparse player at those tempos is
    still only a guess.  Everyone else fast keeps midwest, and a slow jazz player, or one
    on a room mic, has to name it.
    """
    busy_now = busy is not None and busy >= _BUSY_ONSETS_PER_BEAT
    minor_with_room = mode == "minor" and not busy_now
    if bpm < 80.0:
        return "rnb" if minor_with_room and bpm >= 60.0 else "lofi"
    if bpm < 90.0:
        if busy_now:
            return "midwest"
        return "hiphop" if minor_with_room else "lofi"
    if bpm < 100.0:
        return "hiphop" if minor_with_room else "midwest"
    if bpm < 118.0:
        return "synthwave" if minor_with_room else "midwest"
    if bpm < 130.0:
        return "house" if minor_with_room else "midwest"
    if bpm <= 150.0:
        return "midwest" if busy is not None and busy >= _SPARSE_ONSETS_PER_BEAT else "trap"
    known_sparse = busy is not None and busy < _SPARSE_ONSETS_PER_BEAT
    return "jazz" if known_sparse and bpm <= 220.0 else "midwest"


def _mode_of(key: str | None) -> str | None:
    if not key:
        return None
    try:
        return parse_key(key).mode
    except ValueError:
        return None


class StylePick:
    """Where a detected style comes from over a take.

    Drums need a style the moment the count lands, when only the tempo is known, so it
    picks once there.  It picks once more as the band enters, knowing the key and how
    busy the player has been, and the drums changing then reads as part of the entrance.
    After that it never moves.  A room mic hears the drums as onsets too, so its
    ``trust_busy`` is off.
    """

    def __init__(self, *, key: str | None = None, trust_busy: bool = False):
        self._mode = _mode_of(key)
        self._trust_busy = bool(trust_busy)
        self._onsets = 0
        self._beats = 0
        self.settled = False

    def heard_bar(self, onsets: int) -> None:
        self._onsets += int(onsets)
        self._beats += 4

    @property
    def busy(self) -> float | None:
        """Onsets per beat over the bars heard, if this input can say."""
        if not self._trust_busy or not self._beats:
            return None
        return self._onsets / self._beats

    def at_count(self, bpm: float) -> str:
        return detect_style(bpm, self._mode)

    def at_entry(self, bpm: float, key: str | None) -> str:
        self.settled = True
        return detect_style(bpm, _mode_of(key) or self._mode, self.busy)


@dataclass(frozen=True)
class EntryState:
    stage: str               # drums | band | full
    held_back: tuple[str, ...]   # roles not allowed to sound yet; the rest play
    key: str | None          # committed key, None until the band is in
    chord: str | None        # chord from the committed progression
    reason: str


_NOTHING_HEARD = HarmonyState(t_s=0.0, key=None, key_conf=0.0, chord=None,
                              chord_conf=0.0, register_low_midi=None,
                              register_high_midi=None)


#: How much of a bar's pitch energy its three strongest notes must hold for the bar
#: to count as someone playing.  Drums barely favour any note: bars measured with
#: nobody playing (a real take, 2026-11-02) held 0.29-0.37, with guitar 0.46-0.76.
TONAL_TOP3 = 0.42


class GuitarEar:
    """Hear a guitar over drums alone.  It only works before the band comes in: once
    bass and chords are in the room, the mic's pitches are mostly theirs."""

    def __init__(self):
        self._heard = np.zeros(12)
        self._since_tonal = 99
        #: This bar alone sounded played, confirmed or not.  A fade needs this: a
        #: player coming back must not wait a bar for confirmation.
        self.last_bar_tonal = False

    def bar(self, chromas) -> bool:
        """Is someone playing?  One tonal bar could be a drum ringing near the
        threshold, so it takes two within three bars.  Notes of a tonal bar are
        remembered."""
        frames = [c for c in chromas if float(np.sum(c)) > 0.0]
        mean = np.mean(frames, axis=0) if frames else None
        self.last_bar_tonal = (mean is not None
                               and float(np.sort(mean)[-3:].sum()) >= TONAL_TOP3)
        if not self.last_bar_tonal:
            self._since_tonal += 1
            return False
        confirmed = self._since_tonal <= 2
        self._since_tonal = 1
        self._heard += mean
        return confirmed

    def notes(self, count: int = 5) -> tuple[int, ...]:
        """Her most-played pitch classes, strongest first; empty if none heard."""
        if not np.any(self._heard):
            return ()
        order = np.argsort(self._heard)[::-1][:count]
        return tuple(int(i) for i in order if self._heard[i] > 0.0)


class BandEntry:
    def __init__(self, style: str, *, min_drum_bars: int = 8, key_bars: int = 4,
                 melody_delay_bars: int = 4, release_bars: int = 8,
                 known_key: str | None = None, after_first_heard_bars: int = 5,
                 key_fallback_bars: int | None = None):
        """``after_first_heard_bars`` counts from when she is confirmed, which is
        about three bars after her first note, so she gets about eight bars.

        ``key_fallback_bars``: the governor wants several bars running on one key, and
        a player who moves around may never give it that.  After this many heard bars,
        commit to the tracker's latest key instead.  None waits for as long as it takes.
        """
        self._min_drum_bars = int(min_drum_bars)
        self._melody_delay = int(melody_delay_bars)
        self._gov = HarmonyGovernor(style, SimpleNamespace(
            harmony_commit_bars=int(key_bars),
            harmony_release_bars=int(release_bars)))
        self._band_bar: int | None = None
        self._heard = 0
        self._known_key = known_key
        self._after_heard = int(after_first_heard_bars)
        self._first_heard: int | None = None
        self._key_fallback_bars = key_fallback_bars
        self._latest_key: str | None = None
        self.bars_in = 0

    @property
    def band_is_in(self) -> bool:
        return self._band_bar is not None

    def restyle(self, style: str) -> None:
        """The chords come from this style from now on, in the same key."""
        self._gov.restyle(style)

    def chord_at(self, bar: int, harmony) -> str | None:
        """Ask again for a bar already governed, after a :meth:`restyle`."""
        return self._gov.govern(bar, harmony if harmony is not None else _NOTHING_HEARD,
                                resting=True).chord

    def update(self, bar: int, harmony, playing: bool) -> EntryState:
        if playing:
            self._heard += 1
            if self._first_heard is None:
                self._first_heard = bar
        ready = (self._first_heard is not None and bar >= max(
            self._min_drum_bars, self._first_heard + self._after_heard))
        if self._known_key is not None:
            # She told us the key, so nothing heard can change it.  The band still
            # waits for her to start: drums alone are for playing over.
            if not ready and self._band_bar is None:
                return EntryState(DRUMS, HARMONY_ROLES + MELODY_ROLES, None, None,
                                  f"drums: waiting to bring in {self._known_key}")
            if self._gov.committed_key is None:
                self._gov.commit(self._known_key, bar)
            playing, harmony = False, _NOTHING_HEARD
        committed = self._gov.committed_key is not None
        # A bar she sat out says nothing about her key.  Before commitment it
        # never reaches the governor.  Afterwards it has to, because the chord
        # cycle keeps advancing while she rests, but a rest must neither let the
        # tracker's stale belief change key nor interrupt a real key change.
        if playing:
            # A bar the tracker could not name is no evidence either, the same as a
            # rest: one unsure bar must not cost a real key change all its bars.
            unsure = committed and harmony is not None and harmony.key is None
            governed = self._gov.govern(bar, harmony, resting=unsure)
        elif committed:
            governed = self._gov.govern(bar, harmony, resting=True)
        else:
            governed = None
        if playing and harmony is not None and harmony.key is not None:
            self._latest_key = harmony.key
        if (self._key_fallback_bars is not None and self._gov.committed_key is None
                and self._heard >= self._key_fallback_bars
                and self._latest_key is not None):
            self._gov.commit(self._latest_key, bar)
            governed = self._gov.govern(bar, harmony if harmony is not None else _NOTHING_HEARD,
                                        resting=True)
        key = self._gov.committed_key
        if key is None or (not ready and self._band_bar is None):
            return EntryState(DRUMS, HARMONY_ROLES + MELODY_ROLES, None, None,
                              f"drums: heard {self._heard} bars, no settled key")
        if self._band_bar is None:
            self._band_bar = bar
        self.bars_in = bar - self._band_bar
        chord = None if governed is None else governed.chord
        if bar - self._band_bar < self._melody_delay:
            return EntryState(BAND, MELODY_ROLES, key, chord,
                              f"band: bass and chords in {key}")
        return EntryState(FULL, (), key, chord, f"full: melody over {key}")


#: The shape of the song once the band is in: (name, bars, density).  Density
#: picks how busy the hats, keys and melody are; the kick and snare never change,
#: so the song grows without the groove being rewritten under her.
SONG_ARC = (("verse", 16, 0.45), ("lift", 16, 0.85), ("verse", 8, 0.5),
            ("break", 4, 0.3), ("lift", 16, 0.9), ("verse", 16, 0.45),
            ("break", 4, 0.3), ("lift", 24, 0.9))
#: Roles sitting out a section.  A break drops the kick and snare, so the band
#: coming back reads as an event; the high pad is saved for the lifts.
_SECTION_RESTS = {"verse": ("shimmer",), "break": ("kick", "snare", "shimmer", "lead"),
                  "lift": ()}


def song_section(bars_in: int) -> tuple[str, float, tuple[str, ...]]:
    """(section name, density, roles resting) ``bars_in`` bars after the band came
    in.  The arc repeats, so a long take keeps moving."""
    total = sum(bars for _, bars, _ in SONG_ARC)
    at = max(0, int(bars_in)) % total
    for name, bars, density in SONG_ARC:
        if at < bars:
            return name, density, _SECTION_RESTS[name]
        at -= bars
    raise AssertionError("unreachable: at is reduced modulo the arc's length")


# ------------------------------------------------------------ following the player
#
# Juniper, 2026-11-06: "the band should follow the player but just with the confidence
# waiting thing so that if i just make like a little mistake it doesnt assume thats the
# whole direction we're going."  Before this the band's level never moved and its
# density was the arc's, by bar count, whatever she played.

#: The band follows.  "lead", where plume steers the song itself, is on the roadmap;
#: it would be another entry here, with the arc at full depth instead of following.
FOLLOW = "follow"
FOLLOW_MODES = (FOLLOW,)

#: Bars a new way of playing has to hold before the band answers.  One bar is a
#: stumble, and two is a fill or the end of a phrase, which breathe in twos.  A third
#: bar the same way is past any fill and is a choice, and it is still inside the
#: four-bar phrase that changed, so the answer is not a section late.  Bars are read two
#: bars late on top of this, so the band moves about five bars after she does.
FOLLOW_HOLD_BARS = 3
#: A room mic hears the band as well as the player, so it is less sure what it heard.
FOLLOW_HOLD_BARS_ROOM = 4
#: Played bars it listens to before it knows how this player normally plays.
FOLLOW_LEARN_BARS = 4
#: How far a bar has to be from what the band already follows to count as different.
#: 3 dB is mind.py's energy_up_db; half an octave of busyness is about 40% more or
#: fewer notes.  Inside these a bar is more of the same.
FOLLOW_LEVEL_DEADBAND_DB = 3.0
FOLLOW_BUSY_DEADBAND = 0.5
#: Notes per beat added to both sides of the busyness ratio, so a sparse player's
#: second note is not "twice as busy".  mind.py's phrase floor.
_BUSY_FLOOR = 0.25
#: The band moves half as far as she did: it answers rather than mirrors (the idea
#: behind mind.py's delayed recovery), and comes down to find a soft player without
#: vanishing under her.
FOLLOW_GAIN_PER_DB = 0.5
#: Up is bounded tighter than down: a band that lifts with her must not bury her.
FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_MAX_DB = -6.0, 3.0
#: Density, as a multiple of the song's own.  Softer playing thins the band as well as
#: quietening it; busier playing fills it in.
FOLLOW_DENSITY_PER_DB = 0.04
FOLLOW_DENSITY_PER_OCTAVE = 0.25
FOLLOW_DENSITY_MIN, FOLLOW_DENSITY_MAX = 0.6, 1.3
#: A room mic never moves the band as far.
_ROOM_GAIN_DB = (-4.0, 2.0)
_ROOM_DENSITY = (0.75, 1.15)
#: The most it moves in one bar, so an answer is a glide.  The engine glides a gain
#: change across the bar as well, so a decibel is never heard as a step.
FOLLOW_GAIN_STEP_DB = 1.0
FOLLOW_DENSITY_STEP = 0.05
#: Notes per beat past which more band buries the player: mind.py's density_high.
#: Over it the band never fills in, however busy that is for them.
FOLLOW_TORRENT = 2.5
#: Bars a "busier" or "sparser" cue holds the band's density where the cue left it.
#: After that following moves again, under the cue's own scale, which stays.
FOLLOW_CUE_HOLD_BARS = 16
#: How much of the arc's swing survives while the band follows.  The arc stays for
#: its sections (breaks, the high pad saved for lifts, the moments the take page
#: shows), but its density by bar count alone is the band leading, so following
#: keeps half of it and lets her playing do the rest.
FOLLOW_ARC_DEPTH = 0.5
_ARC_MEAN = sum(bars * d for _, bars, d in SONG_ARC) / sum(bars for _, bars, _ in SONG_ARC)
#: A learned normal keeps adapting this slowly, and only while the band is not leaning,
#: so a long soft passage is still soft at its end.
_NORMAL_ALPHA = 0.05


@dataclass(frozen=True)
class FollowState:
    gain_db: float               # added to the band's level
    density: float               # multiplies the band's density
    moved: tuple[str, ...]       # what changed this bar: softer, louder, busier, sparser
    reason: str


class _Patience:
    """One reading's patience: commit only to what has held for ``hold`` bars.

    Each bar is compared with what is already committed.  A bar inside the deadband is
    more of the same and ends any run, and a bar on the other side starts a new one.  So
    a mistake moves nothing, and cannot be half of a real move either: the same rule
    the key governor keeps.
    """

    def __init__(self, hold: int, deadband: float):
        self._hold, self._deadband = int(hold), float(deadband)
        self.value = 0.0
        self._run: list[float] = []
        self._side = 0

    def heard(self, reading: float) -> int:
        """-1 or +1 when this bar commits a move that way, else 0."""
        off = reading - self.value
        side = 0 if abs(off) <= self._deadband else (1 if off > 0 else -1)
        if side == 0:
            self._run, self._side = [], 0
            return 0
        if side != self._side:
            self._run, self._side = [], side
        self._run.append(reading)
        if len(self._run) < self._hold:
            return 0
        self.value = sum(self._run) / len(self._run)
        self._run, self._side = [], 0
        return side


class BandFollow:
    """The band's level and density, following how she plays.

    Feed it each bar's level (dB) and busyness (onsets per beat), and whether she
    played.  Both are read against her own normal, learned from her first played bars:
    nobody is soft in absolute terms.  A bar she sat out is no evidence of anything, so
    it neither moves the band nor breaks a run.

    ``trusted`` is False on a room mic, where the band is in the signal.  Then it
    ignores busyness (the drums are onsets too), waits longer, moves less far, learns
    her normal only once the band is in, and takes its own lean back off what it hears,
    so a band that lifted cannot read as a player who lifted.
    """

    def __init__(self, *, trusted: bool, mode: str = FOLLOW):
        if mode not in FOLLOW_MODES:
            raise ValueError(f"band mode {mode!r}: only {FOLLOW!r} exists, and plume "
                             f"leading is on the roadmap")
        self.mode = mode
        self.trusted = bool(trusted)
        self.hold_bars = FOLLOW_HOLD_BARS if self.trusted else FOLLOW_HOLD_BARS_ROOM
        self._level = _Patience(self.hold_bars, FOLLOW_LEVEL_DEADBAND_DB)
        self._busy = _Patience(self.hold_bars, FOLLOW_BUSY_DEADBAND)
        self._gain_lim = ((FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_MAX_DB) if self.trusted
                          else _ROOM_GAIN_DB)
        self._dens_lim = ((FOLLOW_DENSITY_MIN, FOLLOW_DENSITY_MAX) if self.trusted
                          else _ROOM_DENSITY)
        self._normal_db: float | None = None
        self._normal_busy: float | None = None
        self._learned = 0
        self._torrent = False
        self.gain_db = 0.0
        self.density = 1.0
        self._cue_hold = 0

    def shape(self, arc_density: float) -> float:
        """The song arc's density, keeping only part of its swing while following."""
        return _ARC_MEAN + FOLLOW_ARC_DEPTH * (float(arc_density) - _ARC_MEAN)

    def _learn(self, db: float, busy: float | None) -> None:
        if self._learned < FOLLOW_LEARN_BARS:
            n = self._learned
            self._normal_db = db if n == 0 else (self._normal_db * n + db) / (n + 1)
            if busy is not None:
                self._normal_busy = (busy if self._normal_busy is None
                                     else (self._normal_busy * n + busy) / (n + 1))
            self._learned += 1
        elif self._level.value == 0.0 and self._busy.value == 0.0:
            self._normal_db += _NORMAL_ALPHA * (db - self._normal_db)
            if busy is not None and self._normal_busy is not None:
                self._normal_busy += _NORMAL_ALPHA * (busy - self._normal_busy)

    def bar(self, rms_db: float | None, onsets_per_beat: float | None, *,
            playing: bool, band_in: bool = True) -> FollowState:
        """This bar's lean.  ``band_in`` False: drums alone, so only listen."""
        busy = None if onsets_per_beat is None or not self.trusted else float(onsets_per_beat)
        if not band_in:
            # A room mic over drums alone hears another mix than it will with the band
            # in, so only a trusted input learns from these bars.
            if self.trusted and playing and rms_db is not None:
                self._learn(float(rms_db), busy)
            return FollowState(0.0, 1.0, (), "follow: listening until the band is in")
        moved: list[str] = []
        if playing and rms_db is not None:
            # Off a room mic, take back what the band's own lean put into it.
            db = float(rms_db) - (0.0 if self.trusted else self.gain_db)
            if self._learned >= FOLLOW_LEARN_BARS:
                step = self._level.heard(db - self._normal_db)
                if step:
                    moved.append("louder" if step > 0 else "softer")
                if busy is not None and self._normal_busy is not None:
                    self._torrent = busy > FOLLOW_TORRENT
                    ratio = math.log2((busy + _BUSY_FLOOR) / (self._normal_busy + _BUSY_FLOOR))
                    step = self._busy.heard(ratio)
                    if step:
                        moved.append("busier" if step > 0 else "sparser")
            self._learn(db, busy)
        gain_t, dens_t = self._targets()
        self.gain_db = _toward(self.gain_db, gain_t, FOLLOW_GAIN_STEP_DB)
        if self._cue_hold > 0:
            self._cue_hold -= 1
        else:
            self.density = _toward(self.density, dens_t, FOLLOW_DENSITY_STEP)
        return self._state(tuple(moved))

    def _targets(self) -> tuple[float, float]:
        lvl, busy = self._level.value, self._busy.value
        gain = _clip(FOLLOW_GAIN_PER_DB * lvl, *self._gain_lim)
        dens = 1.0 + FOLLOW_DENSITY_PER_DB * lvl + FOLLOW_DENSITY_PER_OCTAVE * busy
        hi = min(1.0, self._dens_lim[1]) if self._torrent else self._dens_lim[1]
        return gain, _clip(dens, self._dens_lim[0], hi)

    def cue(self, value: str) -> FollowState:
        """A "busier" or "sparser" cue is the player's own word, and beats following.

        Any lean against it goes at once, and the density holds there for
        :data:`FOLLOW_CUE_HOLD_BARS`.  A lean the same way as the cue stays."""
        if value not in ("busier", "sparser"):
            return self._state(())
        self.density = (min(self.density, 1.0) if value == "sparser"
                        else max(self.density, 1.0))
        self._cue_hold = FOLLOW_CUE_HOLD_BARS
        return self._state(())

    def _state(self, moved: tuple[str, ...]) -> FollowState:
        why = (f"follow: level {self.gain_db:+.1f}dB density x{self.density:.2f}"
               + (f", {' and '.join(moved)} for {self.hold_bars} bars" if moved else ""))
        return FollowState(round(self.gain_db, 3), round(self.density, 4), moved, why)


def _toward(now: float, target: float, step: float) -> float:
    return now + max(-step, min(step, target - now))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


OUTRO_SIGNAL_BARS = 4
OUTRO_FADE_BARS = 6


def outro(seconds_in: float, wind_down_s: float | None, bar_s: float):
    """Where a timed song is in its ending, or None before it.

    (bars into the outro, gain offset in dB, whether the take ends).  It signals
    first, with a crash and the drums thinning out, so the player can wind down
    with the band; only then does it fade.
    """
    if wind_down_s is None or seconds_in < wind_down_s:
        return None
    bars = int((seconds_in - wind_down_s) / bar_s)
    fading = max(0, bars - OUTRO_SIGNAL_BARS + 1)
    gain = -36.0 * min(fading, OUTRO_FADE_BARS) / OUTRO_FADE_BARS
    return bars, gain, fading >= OUTRO_FADE_BARS


#: How long after the player stops a direct or MIDI take should be over.
END_AFTER_S = 20.0
#: Roughly how long a fade should take, whatever the tempo.  In whole bars, so this
#: is only ever as close as the bar lets it be.
_FADE_TARGET_S = 6.0
_FADE_MIN_BARS, _FADE_MAX_BARS = 2, 6
#: What reading costs before anything can react.  Bars are read two bars late and a
#: player stops somewhere inside one, so a stop is noticed one to two bars after it
#: happens.  Waiting in whole bars can add up to one more, so a take lands a little over.
_READ_LAG_BARS = 1.5
#: A rest shorter than this never dips the band, at any tempo, however late that makes the
#: ending at a slow one.  Bars are read late, so a dip for a rest would land after she is back.
MIN_REST_S = 6.0


class SilenceEnding:
    """Nobody playing: wait, then fade out and end (Juniper, 2026-11-02).

    Times are bar starts in stream seconds.  A bar of playing during the fade
    brings the band straight back, so a rest costs at most a short dip.  Bars are read
    two bars late, so the take stays open ``listen_bars`` past the fade: a return
    in the last bars of the fade is still heard.

    ``end_after_s``: for an input that really can tell the player stopped (direct,
    MIDI), size the wait and fade so the take ends about that long after they stop,
    given ``bar_s`` on each update.  The wait never drops below :data:`MIN_REST_S` or two
    bars, and reading and fading take six more, so below about 90 bpm it ends later than
    asked: that is the price of a phrase rest not ducking a slow song.  Until they have
    played a bar it keeps ``silence_s`` and ``fade_bars``: they have only just knocked, and
    their hands are still on the way back to the strings.  A room mic leaves it unset
    and keeps those numbers throughout, because it cannot hear a player stop over the
    band.
    """

    def __init__(self, *, silence_s: float = 30.0, fade_bars: int = 4,
                 floor_db: float = -36.0, listen_bars: int = 2,
                 end_after_s: float | None = None):
        self._silence_s = float(silence_s)
        self._fade_bars = int(fade_bars)
        self._floor_db = float(floor_db)
        self._listen_bars = int(listen_bars)
        self._end_after_s = None if end_after_s is None else float(end_after_s)
        self._last_heard: float | None = None
        self._played = False
        self._fade = 0
        self._bars_of_fade = self._fade_bars

    def _timing(self, bar_s: float | None) -> tuple[float, int]:
        """(seconds to wait, bars to fade over)."""
        if self._end_after_s is None or bar_s is None or not self._played:
            return self._silence_s, self._fade_bars
        fade = max(_FADE_MIN_BARS, min(_FADE_MAX_BARS, round(_FADE_TARGET_S / bar_s)))
        spent = (fade + self._listen_bars + _READ_LAG_BARS) * bar_s
        return max(self._end_after_s - spent, MIN_REST_S, 2.0 * bar_s), fade

    def update(self, t_bar: float, playing: bool,
               bar_s: float | None = None) -> tuple[float, bool]:
        """(gain offset in dB for this bar, whether the take should end)."""
        self._played = self._played or bool(playing)
        if playing or self._last_heard is None:
            self._last_heard = float(t_bar)
            self._fade = 0
            if playing:
                return 0.0, False
        if self._fade == 0:           # once it is fading, a tempo cue cannot reshape it
            wait, self._bars_of_fade = self._timing(bar_s)
            # 1e-6: a wait of one bar must not lose to rounding in the clock's sums.
            if t_bar - self._last_heard < wait - 1e-6:
                return 0.0, False
        self._fade += 1
        gain = self._floor_db * min(self._fade, self._bars_of_fade) / self._bars_of_fade
        return gain, self._fade >= self._bars_of_fade + self._listen_bars


def count_start(taps, soft_hits, *, tol_frac: float = 0.15,
                max_back: int = 12) -> list[float]:
    """The whole count, including knocks too quiet to have confirmed it.

    A count is confirmed by its four clearest knocks, which need not be its
    first four: on 2026-11-02 the first three of a count were quiet, the band
    counted from the fourth, and the drums came in three beats late.  So walk
    back along the confirmed spacing through quieter hits, until a beat is
    missing.  The first knock found is where she started counting.
    """
    knocks = sorted(float(t) for t in taps)
    gaps = sorted(b - a for a, b in zip(knocks, knocks[1:]))
    period = gaps[len(gaps) // 2]
    soft = sorted(float(t) for t in soft_hits)
    anchor = knocks[0]
    for beats_back in range(1, max_back + 1):
        # Measured from the confirmed knock each time: stepping from each soft hit
        # would let stray room noise walk the grid away.
        want = anchor - beats_back * period
        near = [t for t in soft if abs(t - want) <= tol_frac * period]
        if not near:
            break
        knocks.insert(0, want)     # the grid's time: a hit only proves the beat
    return knocks


def tap_tempo(taps) -> tuple[float, float]:
    """(bpm, next beat time) from a confirmed count-in's tap times.

    Median interval, so one rushed first knock does not set the tempo.
    """
    ts = sorted(float(t) for t in taps)
    if len(ts) < 2:
        raise ValueError(f"tap_tempo needs at least 2 taps, got {len(ts)}")
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    n = len(gaps)
    period = gaps[n // 2] if n % 2 else 0.5 * (gaps[n // 2 - 1] + gaps[n // 2])
    return 60.0 / period, ts[-1] + period


def first_bar_time(next_beat_s: float, bar_s: float, playhead_s: float,
                   comp_s: float, *, margin_s: float = 0.15) -> float:
    """The first downbeat still far enough ahead to place, on the tapped grid.

    Placing early by ``comp_s`` at a fast tempo can land the next beat in the
    past.  Rather than clip it, or start on a beat that puts the "one" somewhere
    else, come in a whole bar later.
    """
    t = float(next_beat_s)
    while t - comp_s < playhead_s + margin_s:
        t += bar_s
    return t
