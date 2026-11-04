"""The page's script itself, run under Node with a small stand-in for the browser.

No browser, no sound.  The audio element in the stub only counts calls, and the
take documents come from a real bridge over real take folders.  Skipped where Node
is missing, since plume itself has no JavaScript toolchain.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from plume import gui, realtime
from record_fixture import cue_line, ending_doc, landed, make_row, song, write_take
from test_gui_bridge import Client

STATIC = Path(gui.__file__).with_name("gui_static")
STUB = Path(__file__).with_name("gui_page_stub.mjs")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node")
KEY = "ab" * 32


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """Real bridge answers for a shelf: kept "tap", kept "other", plain "lead", drums-only "alone"."""
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    root = tmp_path / "takes"
    wav = b"RIFF" + bytes(64)
    write_take(root / "tap", song(40, band_bar=8, cues={20: landed("busier")}), ending=ending_doc(),
               cues=[cue_line("busier", 18, 90.0)], wav_bytes=wav, started_at=1800000004.0)
    write_take(root / "other", song(30, band_bar=10), ending=ending_doc(), wav_bytes=wav, started_at=1800000003.0)
    write_take(root / "lead", [make_row(b, entry=None) for b in range(20)], ending=ending_doc(),
               started_at=1800000002.0)
    write_take(root / "alone", song(9, heard_bar=None, band_bar=None), ending=ending_doc(realtime.END_SILENCE),
               started_at=1800000001.0)
    for folder, name in (("tap", "tap name"), ("other", "other name")):
        (root / folder / "keep.json").write_text(json.dumps({"kept": True, "name": name}))
    srv = gui.GuiServer(roots=[root], log=lambda line: None).start()
    try:
        client = Client(srv).login()
        takes = json.loads(client.get("/api/takes")[1])["takes"]
        full = {t["id"]: json.loads(client.get(f"/api/take/{t['id']}")[1]) for t in takes}
    finally:
        srv.close()
    return {"takes": takes, "docs": full, "ids": {t["folder"]: t["id"] for t in takes}}


def run_page(tmp_path, docs, body, **options):
    """Run ``body`` against the real app.js.  Only the import paths change, for Node."""
    work = tmp_path / "page"
    work.mkdir(exist_ok=True)
    for name in ("app", "stone", "layout", "session"):
        source = (STATIC / f"{name}.js").read_text()
        (work / f"{name}.mjs").write_text(re.sub(r"'/static/(\w+)\.js'", r"'./\1.mjs'", source))
    assert "/static/" not in (work / "app.mjs").read_text()
    shutil.copy(STUB, work / "stub.mjs")
    settings = {"takes": docs["takes"], "docs": docs["docs"], **options}
    (work / "html.txt").write_text((STATIC / "index.html").read_text())
    (work / "settings.json").write_text(json.dumps(settings))
    (work / "ids.json").write_text(json.dumps(docs["ids"]))
    (work / "run.mjs").write_text("""
import { readFileSync } from 'node:fs';
import { start } from './stub.mjs';
const here = new URL('.', import.meta.url);
const read = (name) => readFileSync(new URL(name, here), 'utf8');
const ids = JSON.parse(read('ids.json'));
const page = await start(read('html.txt'), JSON.parse(read('settings.json')));
const $ = page.$;
const result = await (async () => {
""" + body + """
})();
console.log(JSON.stringify(result));
process.exit(0);
""")
    done = subprocess.run([NODE, str(work / "run.mjs")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------- sound

SOUND = "{plays: $('player').plays, src: $('player').src, hidden: $('player').hidden, pauses: $('player').pauses, audio: page.calls.filter((c) => c.path.includes('/audio/')).length}"


def test_opening_a_record_plays_nothing_and_asks_for_no_audio(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        return {{ screen: document.body.getAttribute('data-screen'), buttons: !!$('listen-human'), ...{SOUND} }};""")
    assert out["screen"] == "record" and out["buttons"], "the check needs a take that has audio"
    assert out["plays"] == 0 and out["src"] == "" and out["hidden"] is True and out["audio"] == 0


def test_only_the_listen_button_sets_a_source(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-machine').click();
        await page.settle();
        return {SOUND};""")
    assert out["plays"] == 1 and out["hidden"] is False
    assert out["src"].endswith(f"/audio/machine?k={KEY}")


@pytest.mark.parametrize("leave", ["await page.shelf();", "page.history.go(-1); await page.settle();"])
def test_leaving_the_record_stops_the_sound(tmp_path, docs, leave):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-human').click();
        await page.settle();
        const before = $('player').pauses;
        {leave}
        return {{ screen: document.body.getAttribute('data-screen'), before, ...{SOUND} }};""")
    assert out["screen"] == "shelf"
    assert out["pauses"] > out["before"] and out["src"] == "" and out["hidden"] is True


def test_moving_to_another_record_stops_the_sound(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        await page.shelf();
        await page.open(ids.other);
        $('listen-human').click();
        page.history.go(-2);
        await page.settle();
        return {{ title: $('rec-title').textContent, ...{SOUND} }};""")
    assert out["title"] == "tap name" and out["src"] == "" and out["hidden"] is True


# ------------------------------------------------------------------ hearing words

HEARING = """{
  yes: $('hear-yes').textContent, no: $('hear-no').textContent,
  noHidden: $('hear-no').hidden, lineHidden: $('hear-line').hidden,
  width: parseFloat($('hear-line').style.width), left: parseFloat($('hear-line').style.left) }"""


def test_the_hearing_words_are_exactly_these(tmp_path, docs):
    out = run_page(tmp_path, docs, f"await page.open(ids.tap); return {HEARING};")
    assert out["yes"] == "plume can hear"
    assert out["no"] == "can't hear you from here on"
    assert out["noHidden"] is False and out["lineHidden"] is False


def test_the_hearing_line_stops_at_the_bar_the_band_came_in(tmp_path, docs):
    take = docs["docs"][docs["ids"]["tap"]]
    out = run_page(tmp_path, docs, f"await page.open(ids.tap); return {HEARING};")
    points = take["band"]
    share = (take["band_t_s"] - points[0]["t_s"]) / (points[-1]["t_s"] - points[0]["t_s"])
    # The stub's sheet is 1280 wide with 46 of margin each side.
    assert out["width"] == pytest.approx(share * (1280 - 92), abs=0.01)
    assert 0 < share < 0.5


def test_with_no_band_there_is_no_claim_about_after_the_band(tmp_path, docs):
    out = run_page(tmp_path, docs, f"await page.open(ids.alone); return {HEARING};")
    assert out["noHidden"] is True and out["lineHidden"] is False


def test_plain_lead_draws_no_hearing_line(tmp_path, docs):
    out = run_page(tmp_path, docs, f"await page.open(ids.lead); return {HEARING};")
    assert out["lineHidden"] is True and out["noHidden"] is True


# ------------------------------------------------------------------- the page key

def test_the_key_leaves_the_address_at_once_and_stays_out_of_local_storage(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        return { first: page.stateLog[0], hash: page.location.hash, log: page.stateLog,
                 local: page.local.written, session: page.session.written,
                 keys: [...new Set(page.calls.map((c) => c.key))] };""")
    assert out["first"] == "/" and KEY not in json.dumps(out["log"]) and KEY not in out["hash"]
    assert KEY not in json.dumps(out["local"]), "the key must not outlive the tab"
    assert out["session"] == [["plume-key", KEY]]
    assert out["keys"] == [KEY]


def test_a_reload_finds_its_key_in_the_tab(tmp_path, docs):
    out = run_page(tmp_path, docs, "return page.calls[0].key;", address="", stored=KEY)
    assert out == KEY


# ------------------------------------------------------------------- a typed name

def test_a_name_typed_for_one_take_is_saved_on_that_take_when_back_jumps_two(tmp_path, docs):
    """The page moves to another record while the box still has focus."""
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        await page.shelf();
        await page.open(ids.other);
        page.type('meant for other');
        page.history.go(-2);
        await page.settle();
        return { posts: page.posts('rename'), title: $('rec-title').textContent, box: $('take-name').value };""")
    assert out["posts"] == [{"take": docs["ids"]["other"], "name": "meant for other"}]
    assert out["title"] == "tap name" and out["box"] == "tap name"


def test_a_name_typed_while_going_to_the_shelf_goes_to_its_own_take(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        await page.shelf();
        await page.open(ids.other);
        page.type('meant for other');
        page.history.go(-1);
        await page.settle();
        await page.open(ids.tap);
        return { posts: page.posts('rename'), title: $('rec-title').textContent, box: $('take-name').value };""")
    assert out["posts"] == [{"take": docs["ids"]["other"], "name": "meant for other"}]
    assert out["title"] == "tap name" and out["box"] == "tap name"


def test_moving_on_with_nothing_typed_renames_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        $('take-name').focus();
        await page.shelf();
        await page.open(ids.other);
        $('record').focus();
        return page.posts('rename');""")
    assert out == []


def test_a_name_saved_normally_still_saves(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.type('better name');
        $('record').focus();
        await page.settle();
        return { posts: page.posts('rename'), title: $('rec-title').textContent };""")
    assert out["posts"] == [{"take": docs["ids"]["tap"], "name": "better name"}]
    assert out["title"] == "better name"


def test_a_slow_rename_answer_does_not_land_on_the_next_take(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const open = page.hold('rename');
        page.type('late name');
        $('record').focus();
        await page.shelf();
        await page.open(ids.other);
        open();
        await page.settle();
        return { title: $('rec-title').textContent, box: $('take-name').value };""")
    assert out == {"title": "other name", "box": "other name"}


def test_a_slow_keep_answer_does_not_land_on_the_next_take(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const open = page.hold('keep');
        $('keep').click();
        await page.shelf();
        await page.open(ids.lead);
        open();
        await page.settle();
        return { title: $('rec-title').textContent, kept: !$('kept-status').hidden,
                 button: $('keep').textContent };""")
    assert out == {"title": "lead", "kept": False, "button": "Keep"}


def test_an_old_rename_answer_cannot_overwrite_a_newer_one(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const open = page.hold('rename');
        page.type('first try');
        $('record').focus();
        await page.settle();
        page.type('second try');
        $('record').focus();
        await page.settle();
        open();
        await page.settle();
        return $('rec-title').textContent;""")
    assert out == "second try"


def test_a_keep_answer_does_not_undo_a_rename_sent_after_it(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const open = page.hold('keep');
        $('keep').click();
        await page.settle();
        page.type('newer name');
        $('record').focus();
        await page.settle();
        open();
        await page.settle();
        return { title: $('rec-title').textContent, status: !$('kept-status').hidden };""")
    assert out == {"title": "newer name", "status": False}


def test_a_rename_that_works_clears_the_failure_before_it(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.failNext.rename = 1;
        page.type('will fail');
        $('record').focus();
        await page.settle();
        const said = [$('keep-error').hidden, $('keep-error').textContent];
        page.type('will work');
        $('record').focus();
        await page.settle();
        return { said, hidden: $('keep-error').hidden, title: $('rec-title').textContent };""")
    assert out["said"] == [False, "That name was not saved."]
    assert out["hidden"] is True and out["title"] == "will work"


def test_a_failed_save_for_a_take_left_behind_names_that_take(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.failNext.rename = 1;
        const open = page.hold('rename');
        page.type('lost name');
        $('record').focus();
        await page.shelf();
        await page.open(ids.other);
        open();
        await page.settle();
        return { hidden: $('keep-error').hidden, words: $('keep-error').textContent,
                 title: $('rec-title').textContent };""")
    assert out == {"hidden": False, "words": "The name was not saved for tap name.", "title": "other name"}


def test_a_failed_save_stays_on_screen_however_far_the_page_is_scrolled():
    """The stub has no layout, so this reads the rule a browser would apply."""
    rule = re.search(r"#keep-error \{([^}]*)\}", (STATIC / "plume.css").read_text())
    assert rule and "position: sticky" in rule[1] and "top: 0" in rule[1]
    assert "background" in rule[1], "the page would show through the words"


def test_a_failed_keep_says_so_on_the_take_it_was_for(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.lead);
        page.failNext.keep = 1;
        $('keep').click();
        await page.settle();
        return { hidden: $('keep-error').hidden, words: $('keep-error').textContent,
                 kept: !$('kept-status').hidden };""")
    assert out["hidden"] is False and "not saved" in out["words"] and out["kept"] is False


def test_a_failed_name_shows_on_the_shelf_it_was_left_for(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.failNext.rename = 1;
        page.type('lost name');
        await page.shelf();
        await page.settle();
        const onShelf = [document.body.getAttribute('data-screen'), $('keep-error').hidden,
                         $('keep-error').textContent];
        await page.open(ids.other);
        return { onShelf, after: $('keep-error').hidden };""")
    assert out["onShelf"] == ["shelf", False, "The name was not saved for tap name."]
    assert out["after"] is True, "it has had its screen, and does not stay for ever"


def test_a_failed_name_survives_going_straight_to_another_take(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.other);
        await page.shelf();
        await page.open(ids.tap);
        page.failNext.rename = 1;
        page.type('lost name');
        page.history.go(-2);
        await page.settle();
        return { title: $('rec-title').textContent, hidden: $('keep-error').hidden,
                 words: $('keep-error').textContent };""")
    assert out == {"title": "other name", "hidden": False, "words": "The name was not saved for tap name."}


def test_a_name_answered_after_the_shelf_opened_still_reaches_the_shelf(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const open = page.hold('rename');
        page.type('new name');
        await page.shelf();
        const before = $('shelf-' + ids.tap).textContent;
        open();
        await page.settle();
        return { before, after: $('shelf-' + ids.tap).textContent };""")
    assert "tap name" in out["before"]
    assert "new name" in out["after"] and "tap name" not in out["after"]


def test_the_shelf_keeps_a_name_answered_while_it_was_loading(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const takes = page.hold('takes');
        page.type('new name');
        page.$('nav-takes').click();
        await page.settle();
        takes();
        await page.settle();
        return $('shelf-' + ids.tap).textContent;""")
    assert "new name" in out and "tap name" not in out


def test_the_recording_is_named_for_what_it_was_recorded_from(tmp_path, docs):
    docs["docs"][docs["ids"]["other"]].update(route="midi", recorded_from="input")
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        const plain = $('listen-human').textContent;
        await page.shelf();
        await page.open(ids.other);
        return { plain, keys_over_direct: $('listen-human').textContent };""")
    assert out == {"plain": "Your mic", "keys_over_direct": "Your input"}
