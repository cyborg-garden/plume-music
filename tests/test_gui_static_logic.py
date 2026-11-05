"""The page's pure logic, run in Node.  No browser, no page, no sound.

Skipped where Node is missing, since plume itself has no JavaScript toolchain.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from plume import gui

STATIC = Path(gui.__file__).with_name("gui_static")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node")


def run_js(tmp_path, *names, body: str):
    """Import static modules, under .mjs names so Node reads them as modules."""
    imports = []
    for name in names:
        shutil.copy(STATIC / f"{name}.js", tmp_path / f"{name}.mjs")
        imports.append(f"import * as {name} from './{name}.mjs';")
    script = tmp_path / "run.mjs"
    script.write_text("\n".join(imports) + f"\nconsole.log(JSON.stringify((() => {{ {body} }})()));\n")
    done = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(done.stdout)


def boxes(tmp_path, xs):
    return run_js(tmp_path, "layout", body=f"return layout.hitBoxes({json.dumps(xs)});")


def test_events_on_one_bar_each_get_their_own_target(tmp_path):
    out = boxes(tmp_path, [100.0, 500.0, 500.0, 500.0, 900.0])
    same = out[1:4]
    assert all(b["width"] >= 8 for b in same)
    for a, b in zip(same, same[1:]):
        assert a["left"] + a["width"] <= b["left"] + 1e-6, "one target covers another"
    assert same[0]["left"] <= 500.0 <= same[-1]["left"] + same[-1]["width"]


def test_targets_never_overlap_in_a_crowd(tmp_path):
    xs = [float(i * 3) for i in range(30)]
    out = boxes(tmp_path, xs)
    for a, b in zip(out, out[1:]):
        assert a["left"] + a["width"] <= b["left"] + 1e-6


def test_a_lone_event_has_a_usable_target_around_its_line(tmp_path):
    only = boxes(tmp_path, [300.0])[0]
    assert only["width"] == 24 and only["left"] == 288


@pytest.mark.parametrize("key,at,count,want", [
    ("ArrowRight", 0, 3, 1), ("ArrowRight", 2, 3, 2), ("ArrowLeft", 0, 3, 0), ("ArrowLeft", 2, 3, 1),
    ("Home", 2, 3, 0), ("End", 0, 3, 2), ("a", 1, 3, -1), ("Tab", 1, 3, -1)])
def test_arrow_keys_walk_the_strip(tmp_path, key, at, count, want):
    assert run_js(tmp_path, "layout", body=f"return layout.nextIndex('{key}', {at}, {count});") == want


def test_no_stone_hovers_over_its_own_ground(tmp_path):
    """Seeds that once floated, and many more, at the heights where it showed."""
    out = run_js(tmp_path, "stone", body="""
      const bad = [];
      const seeds = [291, 1455, 2037, 3686]; for (let s = 0; s < 1500; s++) seeds.push(s);
      for (const seed of seeds) for (const height of [0.42, 0.7, 0.98]) {
        const r = stone.rng(seed * 13 + 5);
        const b = stone.blockPlanes({ height, seed }, r);
        for (let f = -0.4; f <= 0.4; f += 0.05) {
          const x = f * b.w, y = 0.01;
          let z = Infinity;
          for (const P of b.planes) z = Math.min(z, P.a * x + P.b * y + P.c);
          if (z <= 0) { bad.push([seed, height, f]); break; }
        }
      }
      return bad;""")
    assert out == []


def test_a_hairline_breaks_only_where_a_label_sits(tmp_path):
    label = {"left": 100, "right": 200, "top": 40, "bottom": 54}
    def spans(x, top=0, bottom=300):
        return run_js(tmp_path, "layout",
                      body=f"return layout.lineSpans({top}, {bottom}, {x}, {json.dumps([label])});")
    assert spans(50) == [[0, 300]]
    assert spans(150) == [[0, 40], [54, 300]]
    assert spans(150, top=45) == [[54, 300]]
    assert spans(150, top=60) == [[60, 300]]
    assert spans(150, bottom=30) == [[0, 30]]


def test_two_labels_can_both_break_one_hairline(tmp_path):
    labels = [{"left": 0, "right": 9, "top": 10, "bottom": 20}, {"left": 0, "right": 9, "top": 30, "bottom": 40}]
    out = run_js(tmp_path, "layout", body=f"return layout.lineSpans(0, 50, 5, {json.dumps(labels)});")
    assert out == [[0, 10], [20, 30], [40, 50]]


@pytest.mark.parametrize("came_by_link,pinned,address,want", [
    (True, "abc", "", "#/take/abc"),     # the link the player opened
    (False, "abc", "", None),            # a reload of the shelf stays on the shelf
    (True, "abc", "#/take/def", None),   # an address wins
    (True, None, "", None),
])
def test_only_the_first_load_jumps_to_the_pinned_take(tmp_path, came_by_link, pinned, address, want):
    args = ", ".join(json.dumps(v) for v in (came_by_link, pinned, address))
    assert run_js(tmp_path, "session", body=f"return session.pinnedRoute({args});") == want


@pytest.mark.parametrize("address,want", [
    ("", {"name": "home"}), ("#/", {"name": "home"}), ("#/takes", {"name": "takes"}),
    ("#/live", {"name": "live"}), ("#/take/0123456789abcdef", {"name": "take", "id": "0123456789abcdef"}),
    ("#/take/nothex", {"name": "home"}), ("#/take/0123456789abcdef/x", {"name": "home"}), ("#k=ab", {"name": "home"}),
])
def test_an_address_names_one_screen(tmp_path, address, want):
    assert run_js(tmp_path, "session", body=f"return session.routeOf({json.dumps(address)});") == want


def live_js(tmp_path, expr):
    return run_js(tmp_path, "live", body=f"return {expr};")


@pytest.mark.parametrize("row,doc,want", [
    ({"key": "F major", "style": "midwest", "key_guess": True, "style_guess": True}, {}, "F major, midwest (a guess)"),
    ({"key": "F major", "style": "midwest", "key_guess": False, "style_guess": True}, {"key": "F major"}, "F major, midwest (the style is a guess)"),
    ({"key": "F major", "style": "midwest", "key_guess": True, "style_guess": False}, {"style": "midwest"}, "F major, midwest (the key is a guess)"),
    ({"key": "F major", "style": "midwest", "key_guess": False, "style_guess": False}, {"key": "F major", "style": "midwest"}, "F major, midwest"),
    ({"key": None, "style": "lofi", "key_guess": True, "style_guess": True}, {}, "Listening for the key, lofi (the style is a guess)"),
    # A typed value wins, even over a row that calls it a guess.
    ({"key": "F major", "style": "lofi", "key_guess": True, "style_guess": True}, {"key": "F major", "style": "lofi"}, "F major, lofi"),
    # A key named by the player but held back until the band comes in.
    ({"key": None, "style": "lofi", "key_guess": False, "style_guess": True}, {"key": "F major"}, "F major, lofi (the style is a guess)"),
    (None, {}, "Listening for the key"),
])
def test_the_key_line_calls_a_guess_a_guess(tmp_path, row, doc, want):
    assert live_js(tmp_path, f"live.keyLine({json.dumps(row)}, {json.dumps(doc)})") == want


@pytest.mark.parametrize("trusted,notes,level", [
    (["you_db", "you_busy", "you_notes"], ["F"], -18.0),
    (["you_notes"], ["F"], None),
    ([], None, None),
    (None, None, None),
])
def test_a_field_the_bridge_does_not_trust_is_not_there(tmp_path, trusted, notes, level):
    row = {"input": "room", "you_notes": ["F"], "you_db": -30.0, "you_trusted": trusted}
    out = live_js(tmp_path, f"live.youView({json.dumps(row)}, {{level_db: -18.0}})")
    assert out["notes"] == notes and out["level"] == level
    assert bool(out["why"]) == (trusted != ["you_db", "you_busy", "you_notes"])


def test_the_row_level_stands_in_when_the_recording_has_not_said(tmp_path):
    row = {"input": "direct", "you_notes": [], "you_db": -30.0, "you_trusted": ["you_db", "you_notes"]}
    assert live_js(tmp_path, f"live.youView({json.dumps(row)}, {{level_db: null}})")["level"] == -30.0


def test_with_no_row_the_level_is_the_recordings(tmp_path):
    assert live_js(tmp_path, "live.youView(null, {level_db: -44.0})") == {"notes": None, "level": -44.0, "why": ""}
    assert live_js(tmp_path, "live.youView(null, {})")["level"] is None


def test_a_direct_input_is_not_blamed_on_a_room_mic(tmp_path):
    out = live_js(tmp_path, "live.youView({input: 'direct', you_trusted: []}, {level_db: -18})")
    assert "room mic" not in out["why"] and out["why"]


@pytest.mark.parametrize("db,want", [(-120.0, 0), (-60.0, 0), (-30.0, 0.5), (0.0, 1), (6.0, 1), (None, 0)])
def test_the_meter_stays_on_its_scale(tmp_path, db, want):
    assert live_js(tmp_path, f"live.meterShare({json.dumps(db)})") == want


@pytest.mark.parametrize("doc,want", [
    ({"state": "done", "take": "t1"}, {"take": "t1"}),
    ({"state": "failed", "message": "words"}, {"home": True, "message": "words", "settings": False}),
    ({"state": "idle"}, {"home": True, "message": "", "settings": False}),
    ({"state": "idle", "message": "try another", "settings": True}, {"home": True, "message": "try another", "settings": True}),
    ({"state": "elsewhere", "message": "not a failure"}, {"home": True, "message": "", "settings": False}),
])
def test_where_a_take_lands(tmp_path, doc, want):
    assert live_js(tmp_path, f"live.landing({json.dumps(doc)})") == want


@pytest.mark.parametrize("state,running", [("starting", True), ("waiting", True), ("playing", True), ("stopping", True),
                                           ("idle", False), ("done", False), ("failed", False), ("elsewhere", False)])
def test_what_counts_as_running(tmp_path, state, running):
    assert live_js(tmp_path, f"live.isRunning({{state: '{state}'}})") is running


def test_no_number_means_no_claimed_delay(tmp_path):
    words = live_js(tmp_path, "live.delayLine(null)")
    assert "headphones" in words and "ms" not in words.split()


def test_the_line_on_home_says_what_start_will_use(tmp_path):
    direct = {"name": "Box", "mode": "direct"}
    room = {"name": "Mic", "mode": "room"}
    assert live_js(tmp_path, f"live.routeLine({json.dumps(direct)}, 2, '', '')") == "Box, channel 2"
    assert live_js(tmp_path, f"live.routeLine({json.dumps(room)}, 1, 'F major', 'lofi')") == "Mic, room mic, F major, lofi"
    assert live_js(tmp_path, "live.routeLine(null, 1, '', '')") == "System default, room mic"


def test_a_tab_with_no_key_is_told_why_and_how_to_get_one(tmp_path):
    words = run_js(tmp_path, "session", body="return session.lostWords(false);")
    assert "session" not in words["title"].lower()
    assert "tab" in words["text"] and "plume gui url" in words["text"]


def test_a_stale_key_is_told_why_and_how_to_get_one(tmp_path):
    words = run_js(tmp_path, "session", body="return session.lostWords(true);")
    assert "restarted" in words["text"] and "plume gui url" in words["text"]


def test_only_day_and_night_are_offered():
    static = Path(__file__).resolve().parents[1] / "src" / "plume" / "gui_static"
    for name in ("index.html", "app.js", "theme.js", "plume.css"):
        text = (static / name).read_text(encoding="utf-8")
        assert "theme-dim" not in text and "'dim'" not in text
        assert 'data-theme="dim"' not in text
    assert 'id="theme-day"' in (static / "index.html").read_text(encoding="utf-8")
    assert 'id="theme-night"' in (static / "index.html").read_text(encoding="utf-8")
