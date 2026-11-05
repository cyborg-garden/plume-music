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

from plume import gui, guitake, realtime
from record_fixture import cue_line, ending_doc, landed, make_row, song, write_take
from test_gui_bridge import Client

STATIC = Path(gui.__file__).with_name("gui_static")
STUB = Path(__file__).with_name("gui_page_stub.mjs")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node")
KEY = "ab" * 32


#: Made-up devices, in the shape ``plume inputs --json`` writes.
DEVICES = [{"name": "Test Mic", "inputs": 1, "outputs": 0},
           {"name": "Test Speakers", "inputs": 0, "outputs": 2},
           {"name": "Test Interface", "inputs": 4, "outputs": 4}]


def _no_spawner(argv, log_path):  # noqa: ARG001
    raise AssertionError("nothing in these tests starts a take")


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
    srv = gui.GuiServer(roots=[root], log=lambda line: None, lister=lambda: DEVICES,
                        spawner=_no_spawner, sessions=lambda: []).start()
    try:
        client = Client(srv).login()
        takes = json.loads(client.get("/api/takes")[1])["takes"]
        full = {t["id"]: json.loads(client.get(f"/api/take/{t['id']}")[1]) for t in takes}
        home = json.loads(client.get("/api/home")[1])
    finally:
        srv.close()
    return {"takes": takes, "docs": full, "ids": {t["folder"]: t["id"] for t in takes}, "home": home}


def run_page(tmp_path, docs, body, **options):
    """Run ``body`` against the real app.js.  Only the import paths change, for Node."""
    work = tmp_path / "page"
    work.mkdir(exist_ok=True)
    for name in ("app", "stone", "layout", "session", "live"):
        source = (STATIC / f"{name}.js").read_text()
        (work / f"{name}.mjs").write_text(re.sub(r"'/static/(\w+)\.js'", r"'./\1.mjs'", source))
    assert "/static/" not in (work / "app.mjs").read_text()
    shutil.copy(STUB, work / "stub.mjs")
    settings = {"takes": docs["takes"], "docs": docs["docs"], "home": docs["home"], **options}
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


# ------------------------------------------------------------------- screens

LOOK = """{ screen: page.screen(), hash: page.location.hash }"""


def test_the_first_screen_is_home_and_asks_for_no_takes(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        return {{ ...{LOOK}, start: !$('start').hidden, paths: page.calls.map((c) => c.path) }};""")
    assert out["screen"] == "home" and out["start"] is True
    assert "/api/takes" not in out["paths"]


def test_the_wordmark_goes_home_and_takes_goes_to_the_shelf(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = [];
        $('nav-takes').click(); await page.settle(); seen.push({LOOK});
        $('wordmark').click(); await page.settle(); seen.push({LOOK});
        $('wordmark').click(); await page.settle(); seen.push({LOOK});
        $('nav-takes').click(); await page.settle();
        $('nav-takes').click(); await page.settle(); seen.push({LOOK});
        page.history.go(-1); await page.settle(); seen.push({LOOK});
        return seen;""")
    assert [s["screen"] for s in out] == ["shelf", "home", "home", "shelf", "home"]
    assert out[0]["hash"] == "#/takes" and out[1]["hash"] == "#/"


def test_a_reload_of_the_shelf_stays_on_the_shelf(tmp_path, docs):
    address = "#/takes"
    out = run_page(tmp_path, docs, f"return {LOOK};", address=address, stored=KEY)
    assert out == {"screen": "shelf", "hash": address}


def test_a_pinned_take_still_opens_first_when_nothing_is_running(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {{ ...{LOOK}, title: $('rec-title').textContent }};",
                   pinned=docs["ids"]["tap"])
    assert out["screen"] == "record" and out["title"] == "tap name"


def test_the_shelf_says_how_to_make_a_first_take(tmp_path, docs):
    assert "Start" in re.search(r'id="shelf-empty"[^>]*>([^<]*)', (STATIC / "index.html").read_text())[1]


# ---------------------------------------------------------------------- start

def test_start_is_the_only_thing_that_asks_for_a_take(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.settle();
        await page.poll();
        const before = page.posts('start').length;
        $('nav-takes').click(); await page.settle();
        $('wordmark').click(); await page.settle();
        const after = page.posts('start').length;
        return {{ before, after, focus: document.activeElement.id }};""")
    assert out["before"] == 0 and out["after"] == 0
    assert out["focus"] != "start", "a stray key must not be able to press it"


def test_start_posts_once_and_touches_no_audio(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.settle();
        const before = page.calls.length;
        $('start').click();
        $('start').click();
        await page.settle();
        return {{ posts: page.calls.slice(before).filter((c) => c.method === 'POST'),
                  ...{LOOK}, ...{SOUND} }};""")
    assert len(out["posts"]) == 1 and out["posts"][0]["path"] == "/api/start"
    assert out["posts"][0]["key"] == KEY
    assert out["plays"] == 0 and out["src"] == "" and out["audio"] == 0
    assert out["screen"] == "live" and out["hash"] == "#/live"


def test_with_nothing_changed_start_leaves_the_choice_to_the_bridge(tmp_path, docs):
    """It knows what setup learned, and a page that never changed anything must not overrule it."""
    out = run_page(tmp_path, docs, """
        $('start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{}]


def test_start_works_with_no_device_list_yet(tmp_path, docs):
    docs["home"] = None
    out = run_page(tmp_path, docs, """
        const line = $('home-summary').textContent, hear = $('hear-row').hidden;
        $('start').click(); await page.settle();
        return { line, hear, body: page.posts('start') };""")
    assert out == {"line": "", "hear": True, "body": [{}]}


def test_typed_key_and_style_and_chosen_channel_go_in_the_body(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        $('set-key')._value = '  F major ';
        $('set-style')._value = 'midwest';
        $('set-key').fire('input');
        $('set-channel').value = '3';
        $('set-channel').fire('change');
        const summary = $('home-summary').textContent;
        $('start').click(); await page.settle();
        return { summary, body: page.posts('start')[0] };""")
    assert out["body"]["key"] == "F major" and out["body"]["style"] == "midwest" and out["body"]["channel"] == 3
    assert out["summary"] == "Test Interface, channel 3, F major, midwest"


def test_a_refused_field_comes_back_home_says_so_and_opens_the_menu(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [400, {{ error: {json.dumps(guitake.BAD_KEY)} }}];
        $('set-key')._value = '-rf';
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('home-note').textContent, shown: !$('home-note').hidden,
                  menu: $('home-menu').open, key: $('set-key').value }};""")
    assert out["screen"] == "home" and out["hash"] == "#/"
    assert out["note"] == guitake.BAD_KEY and out["shown"] is True and out["menu"] is True
    assert out["key"] == "-rf", "what was typed stays, to be fixed"


@pytest.mark.parametrize("status", [500, 403])
def test_any_other_refusal_says_nothing_it_was_sent(tmp_path, docs, status):
    trace = "Traceback (most recent call last): File /nowhere"
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [{status}, {{ error: {json.dumps(trace)} }}];
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('home-note').textContent, all: document.body.textContent }};""")
    assert out["screen"] == "home" and out["note"] == guitake.DID_NOT_START
    assert "Traceback" not in out["note"]


def test_a_second_start_elsewhere_lands_on_the_running_take(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [409, {{ error: 'A take is already running.',
                                       live: {{ state: 'playing', row: null }} }}];
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, stop: !$('live-stop').hidden, starts: page.posts('start').length }};""")
    assert out["screen"] == "live" and out["stop"] is True and out["starts"] == 1


def test_a_start_that_cannot_reach_the_bridge_says_so(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.down = true;
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('home-note').textContent }};""")
    assert out["screen"] == "home" and "still running" in out["note"]


def test_a_lost_key_on_start_says_how_to_get_a_new_link(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [401, {{ error: 'no' }}];
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, text: $('problem-text').textContent }};""")
    assert out["screen"] == "problem" and "plume gui url" in out["text"]


def test_an_answer_from_before_start_cannot_land_on_the_old_take(tmp_path, docs):
    old = docs["ids"]["tap"]
    out = run_page(tmp_path, docs, f"""
        page.live = {{ state: 'done', take: {json.dumps(old)}, note: null }};
        const held = page.hold('live');
        await page.poll();                 // this look at the bridge is now stuck in the air
        $('start').click(); await page.settle();
        held();                            // and comes back saying the old take is done
        await page.settle();
        return {LOOK};""")
    assert out == {"screen": "live", "hash": "#/live"}


# ---------------------------------------------------------- settings and hear me

HEAR = "{ row: !$('hear-row').hidden, on: !!$('hear-me').checked, words: !$('hear-words').hidden, delay: $('hear-delay').textContent }"
ROOM_ONLY = [{"name": "Test Mic", "inputs": 1, "outputs": 0}, {"name": "Test Speakers", "inputs": 0, "outputs": 2}]


def home_for(devices):
    described = guitake.describe_devices(devices)
    chosen = guitake.default_device(described)
    return {"devices": described, "device": chosen and chosen["id"],
            "mode": chosen["mode"] if chosen else "room", "channel": 1,
            "hear_me": {"allowed": False, "default": False, "reported_ms": None},
            "folder": "~/Music/Plume", "styles": ["lofi", "midwest"], "problem": None}


def _saved_hear_me(docs, name="Test Interface"):
    for device in docs["home"]["devices"]:
        if device["name"] == name:
            device["hear_me"], device["headphones"] = True, True


def test_the_first_screen_is_one_start_with_hear_me_in_the_menu():
    html = (STATIC / "index.html").read_text()
    first = html[html.index('<main id="home"'):html.index('<details id="home-menu"')]
    menu = html[html.index('<details id="home-menu"'):html.index("</details>")]
    assert "hear" not in first.lower() and "headphones" not in first.lower()
    assert 'id="hear-row"' in menu and 'id="hear-direct"' in menu


def test_hear_me_is_off_until_someone_says_otherwise(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = {HEAR};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["row"] is True and out["on"] is False and out["words"] is False
    assert out["body"] == {}


def test_hear_me_is_offered_on_a_built_in_mic_too(tmp_path, docs):
    docs["home"] = home_for(ROOM_ONLY)
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["row"] is True and out["on"] is False


def test_a_saved_hear_me_shows_ticked_and_is_left_to_the_bridge(tmp_path, docs):
    _saved_hear_me(docs)
    out = run_page(tmp_path, docs, f"""
        const seen = {HEAR};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["on"] is True and out["words"] is True and out["body"] == {}


TICK = "$('hear-me').checked = true; $('hear-me').fire('change'); await page.settle();"


def test_ticking_hear_me_tells_the_bridge_then_and_start_says_nothing_about_it(tmp_path, docs):
    """A tick that only lives in the page outlasts the gear it was true of."""
    interface = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "hear_me": True})}];
        {TICK}
        const ticked = $('hear-me').checked;
        $('start').click(); await page.settle();
        return {{ ticked, said: page.posts('setup/answer'), start: page.posts('start') }};""")
    assert out["said"] == [{"device": interface["id"], "step": "hear_me", "answer": "on"}]
    assert out["ticked"] is True and out["start"] == [{}]


def test_a_tick_left_in_the_page_never_reaches_start_anyway(tmp_path, docs):
    nothing = {**setup_doc("listen", "Barely hearing you.", "again", "start"),
               "heard": {"result": "low", "channel": 1, "peak_db": -45.0}, "others": []}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "hear_me": True})}];
        {TICK}
        page.replies['setup/listen'] = [200, {json.dumps(nothing)}];
        $('start').click(); await page.settle();
        $('setup-start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{}]


def test_a_tick_the_bridge_refuses_does_not_stay_ticked(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [400, {{ error: {json.dumps(guitake.DEVICE_MISSING)} }}];
        {TICK}
        return {{ ticked: $('hear-me').checked, note: $('hear-note').textContent, shown: !$('hear-note').hidden }};""")
    assert out == {"ticked": False, "note": guitake.DEVICE_MISSING, "shown": True}


def test_a_tick_that_cannot_reach_the_bridge_does_not_stay_ticked(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.down = true;
        {TICK}
        return {{ ticked: $('hear-me').checked, note: $('hear-note').textContent }};""")
    assert out == {"ticked": False, "note": "That did not reach plume gui."}


def test_each_device_shows_its_own_tick(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    interface = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = run_page(tmp_path, docs, f"""
        const pick = (id) => {{ $('set-device').value = id; $('set-device').fire('change'); }};
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "hear_me": True})}];
        {TICK}
        pick({json.dumps(mic["id"])});
        const onMic = {{ ...{HEAR}, line: $('home-summary').textContent, channel: !$('set-channel').hidden }};
        pick({json.dumps(interface["id"])});
        const back = {{ ...{HEAR}, line: $('home-summary').textContent, channel: !$('set-channel').hidden }};
        $('start').click(); await page.settle();
        return {{ onMic, back, body: page.posts('start')[0] }};""")
    assert out["onMic"]["on"] is False and out["onMic"]["line"] == "Test Mic, room mic" and out["onMic"]["channel"] is False
    assert out["back"]["on"] is True and out["back"]["channel"] is True
    assert out["back"]["line"] == "Test Interface, channel 1"
    assert out["body"] == {"device": interface["id"]}, "the channel is setup's to fill in, unless she changed it"


def test_turning_hear_me_off_is_told_to_the_bridge_and_stays_off(tmp_path, docs):
    _saved_hear_me(docs)
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "hear_me": False})}];
        $('hear-me').checked = false; $('hear-me').fire('change'); await page.settle();
        $('nav-takes').click(); await page.settle();
        $('wordmark').click(); await page.settle();
        const ticked = $('hear-me').checked;
        $('start').click(); await page.settle();
        return {{ ticked, said: page.posts('setup/answer')[0].answer, body: page.posts('start')[0] }};""")
    assert out == {"ticked": False, "said": "off", "body": {}}


def test_the_channel_setup_found_shows_and_is_left_to_the_bridge(tmp_path, docs):
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["channel"] = 3
    out = run_page(tmp_path, docs, """
        const seen = { line: $('home-summary').textContent, value: $('set-channel').value };
        $('start').click(); await page.settle();
        return { ...seen, body: page.posts('start')[0] };""")
    assert out == {"line": "Test Interface, channel 3", "value": "3", "body": {}}


def test_the_delay_is_called_what_it_is(tmp_path, docs):
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["reported_ms"] = 12.4
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["delay"] == "Use headphones. Hear me adds about 12 ms, by your device's own count, and it could be more."


def test_the_delay_shown_is_the_chosen_devices_own(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    other = {"name": "Test Interface Two", "inputs": 2, "outputs": 2}
    docs["home"]["devices"].append({**guitake.describe_devices([other])[0]})
    mic["reported_ms"] = 8.3
    docs["home"]["hear_me"]["reported_ms"] = 8.3     # not what a device says about another
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["delay"] == "Use headphones. Plume shows the delay your device reports once a take starts."


DIRECT_LINE = "A Direct or monitor button on your interface has no delay. Using both doubles the sound."


def test_the_direct_button_line_waits_in_the_menu_for_hear_me(tmp_path, docs):
    _saved_hear_me(docs)
    out = run_page(tmp_path, docs, "return { text: $('hear-direct').textContent, hidden: $('hear-direct').hidden, words: $('hear-words').hidden, menu: $('home-menu').open };")
    assert out == {"text": DIRECT_LINE, "hidden": False, "words": False, "menu": False}


def test_the_direct_button_line_is_not_offered_on_a_built_in_mic(tmp_path, docs):
    docs["home"] = home_for(ROOM_ONLY)
    assert run_page(tmp_path, docs, "return $('hear-direct').hidden;") is True


def test_the_direct_button_line_is_there_while_it_is_happening(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, "return { text: $('live-direct').textContent, hidden: $('live-direct').hidden };")
    assert out == {"text": DIRECT_LINE, "hidden": False}
    off = live_page(tmp_path, docs, {**PLAYING, "hear_me": False}, "return $('live-direct').hidden;")
    assert off is True


def test_the_direct_button_line_says_nothing_to_a_mic_hearing_itself(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "hear_me": True}
    assert live_page(tmp_path, docs, room, "return $('live-direct').hidden;") is True


def test_stop_never_lands_under_a_second_click_on_start():
    """Both screens start at the same place, so Stop has to start below where Start ended."""
    css = (STATIC / "plume.css").read_text()
    start = int(re.search(r"#start \{ height: (\d+)px", css).group(1))
    say = int(re.search(r"#live-say \{[^}]*min-height: (\d+)px", css).group(1))
    gap = int(re.search(r"\.column \{[^}]*gap: (\d+)px", css).group(1))
    assert say + gap > start


def test_a_machine_with_no_default_can_still_start_on_the_system_input(tmp_path, docs):
    docs["home"] = home_for([{"name": "Test Speakers", "inputs": 0, "outputs": 2}, {"name": "Odd One", "inputs": 1, "outputs": 0}])
    out = run_page(tmp_path, docs, """
        const line = $('home-summary').textContent;
        $('start').click(); await page.settle();
        return { line, body: page.posts('start')[0] };""")
    assert out["line"] == "System default, room mic" and "device" not in out["body"]


def test_a_device_that_went_away_falls_back_to_the_default(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    fewer = home_for([d for d in DEVICES if d["name"] != "Test Mic"])
    out = run_page(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        page.home = {json.dumps(fewer)};
        $('home-menu').open = true; $('home-menu').fire('toggle');
        await page.settle();
        return $('home-summary').textContent;""")
    assert out == "Test Interface, channel 1"


@pytest.mark.parametrize("problem", [guitake.NO_INPUT, guitake.AUDIO_PART_MISSING, guitake.NO_LISTING])
def test_a_device_problem_is_said_on_home(tmp_path, docs, problem):
    docs["home"].update(devices=[], device=None, problem=problem)
    out = run_page(tmp_path, docs, "return { note: $('home-note').textContent, shown: !$('home-note').hidden };")
    assert out == {"note": problem, "shown": True}


# ------------------------------------------------------------------- a live take

ROW = {"bar": 12, "t_s": 30.2, "bpm": 114.6, "section": "verse", "stage": "band", "key": "F major",
       "chord": "Fmaj7", "style": "midwest", "style_guess": True, "key_guess": True, "you_db": -21.4,
       "band_db": -14.0, "you_busy": 1.2, "you_notes": ["F", "A", "C"], "you_source": "mic",
       "input": "direct", "you_trusted": ["you_db", "you_busy", "you_notes"]}
PLAYING = {"state": "playing", "device": "Test Interface", "mode": "direct", "channel": 1, "hear_me": True,
           "key": None, "style": None, "level_db": -18.2, "row": ROW,
           "stream": {"input_ms": 3.0, "output_ms": 5.0, "reported_ms": 8.0, "block_ms": 5.333, "calibrated_ms": None},
           "hint": None}
WAITING = {**PLAYING, "state": "waiting", "row": None, "level_db": -60.0}
FACTS = """{ say: $('live-say').textContent, bar: $('live-bar').textContent, tempo: $('live-tempo').textContent,
  stage: $('live-stage').textContent, notes: $('live-notes').textContent, level: $('live-level').textContent,
  notesShown: !$('live-notes-box').hidden, levelShown: !$('live-level-box').hidden, meter: !$('meter').hidden,
  why: $('live-why').hidden ? '' : $('live-why').textContent, facts: !$('live-facts').hidden,
  delay: $('live-delay').hidden ? '' : $('live-delay').textContent, cues: !$('live-cues').hidden,
  stop: $('live-stop').textContent, stopDisabled: !!$('live-stop').disabled,
  hint: $('live-hint').hidden ? '' : $('live-hint').textContent }"""


def live_page(tmp_path, docs, live, body):
    return run_page(tmp_path, docs, body, live=live, address="#/live", stored=KEY)


def test_the_row_stays_inside_what_the_bridge_sends():
    assert set(ROW) <= set(guitake._ROW_FIELDS)


def test_a_reload_finds_a_running_take_and_starts_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {{ ...{LOOK}, posts: page.calls.filter((c) => c.method === 'POST') }};",
                   live=PLAYING, address="", stored=KEY)
    assert out["screen"] == "live" and out["hash"] == "#/live" and out["posts"] == []


def test_a_running_take_beats_a_pinned_one_and_an_old_address(tmp_path, docs):
    pinned = run_page(tmp_path, docs, f"return {LOOK};", live=PLAYING, pinned=docs["ids"]["tap"])
    old = run_page(tmp_path, docs, f"return {LOOK};", live=PLAYING, address=f"#/take/{docs['ids']['tap']}", stored=KEY)
    assert pinned["screen"] == "live" and old["screen"] == "live"


def test_the_first_view_of_a_take_is_one_line_a_sign_and_stop():
    """Everything else waits in one menu, cues and the delay included."""
    html = (STATIC / "index.html").read_text()
    live = html[html.index('<main id="live"'):html.index('<main id="record"')]
    first = live[:live.index('<details id="live-details"')]
    details = live[live.index('<details id="live-details"'):]
    assert 'id="live-say"' in first and 'id="live-stop"' in first and 'id="meter"' in first
    for name in ("live-facts", "live-why", "live-delay", "live-direct", "live-cues", "cue-busier", "cue-sparser"):
        assert f'id="{name}"' in details and f'id="{name}"' not in first
    assert live.count("<details") == 1 and ">Details</summary>" in details


def test_details_stays_shut_on_its_own_and_is_there_once_there_is_something_in_it(tmp_path, docs):
    room = {**WAITING, "mode": "room", "hear_me": False}
    out = live_page(tmp_path, docs, room, f"""
        const waiting = !$('live-details').hidden;
        page.live = {json.dumps(PLAYING)};
        await page.poll();
        return {{ waiting, playing: !$('live-details').hidden, open: $('live-details').open }};""")
    assert out == {"waiting": False, "playing": True, "open": False}


def test_the_delay_can_be_found_while_waiting_when_it_is_known(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, f"return {{ details: !$('live-details').hidden, ...{FACTS} }};")
    assert out["details"] is True and "about 8 ms" in out["delay"]


# Her own playing, as the engine saw it, is what the words about the ending go by.
PLAYED = {**PLAYING, "row": {**ROW, "playing": True, "playing_trusted": True, "fade_db": 0.0}}


def _rest(bar, **more):
    return {**PLAYED, "row": {**PLAYED["row"], "bar": bar, "playing": False, **more}}


END = "$('live-end').hidden ? '' : $('live-end').textContent"


def test_it_says_how_the_take_ends_when_she_has_stopped_playing(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYED, f"""
        const playing = {END};
        page.live = {json.dumps(_rest(13))};
        await page.poll();
        const oneBar = {END};
        page.live = {json.dumps(_rest(14))};
        await page.poll();
        const twoBars = {END};
        page.live = {json.dumps({**PLAYED, "row": {**PLAYED["row"], "bar": 15}})};
        await page.poll();
        return {{ playing, oneBar, twoBars, back: {END} }};""")
    assert out["playing"] == "" and out["oneBar"] == ""
    assert out["twoBars"] == "Stop playing and the band will finish on its own."
    assert out["back"] == ""


def test_the_drums_waiting_for_her_are_not_the_end_of_anything(tmp_path, docs):
    """Nothing has been played yet, so telling her how to finish would be nonsense."""
    out = live_page(tmp_path, docs, _rest(1), f"""
        page.live = {json.dumps(_rest(4))};
        await page.poll();
        return {END};""")
    assert out == ""


def test_a_room_mic_never_promises_the_band_will_finish(tmp_path, docs):
    """It cannot hear her stop, so that is said in Details and Stop is how it ends."""
    room = {**PLAYING, "mode": "room", "hear_me": False,
            "row": {**ROW, "input": "room", "you_trusted": [], "playing": False, "playing_trusted": False, "fade_db": 0.0}}
    played = {**room, "row": {**room["row"], "playing": True, "bar": 3}}
    out = live_page(tmp_path, docs, played, f"""
        page.live = {json.dumps({**room, "row": {**room["row"], "bar": 9}})};
        await page.poll();
        return {{ end: {END}, how: $('live-how').hidden ? '' : $('live-how').textContent }};""")
    assert "finish on its own" not in out["end"]
    assert out["how"] == "A room mic may not hear you stop, so press Stop when you are done."


ROOM_BAND_IN = {**PLAYING, "mode": "room", "hear_me": False,
                "row": {**ROW, "input": "room", "you_trusted": [], "playing": False, "playing_trusted": False, "fade_db": 0.0}}


def _room_at(bar):
    return {**ROOM_BAND_IN, "row": {**ROOM_BAND_IN["row"], "bar": bar}}


def test_a_room_mic_says_how_it_ends_on_the_first_view_once_the_band_is_in_and_then_lets_go(tmp_path, docs):
    out = live_page(tmp_path, docs, _room_at(12), f"""
        const first = {END};
        page.live = {json.dumps(_room_at(19))};
        await page.poll();
        const later = {END};
        page.live = {json.dumps(_room_at(20))};
        await page.poll();
        return {{ first, later, gone: {END}, details: $('live-how').textContent }};""")
    line = "A room mic may not hear you stop, so press Stop when you are done."
    assert out == {"first": line, "later": line, "gone": "", "details": line}


def test_the_room_line_waits_until_plume_really_cannot_hear_her_stop(tmp_path, docs):
    drums = {**ROOM_BAND_IN, "row": {**ROOM_BAND_IN["row"], "stage": "drums", "playing_trusted": True}}
    assert live_page(tmp_path, docs, drums, f"return {END};") == ""


def test_an_old_row_with_no_word_about_trust_says_nothing_about_how_it_ends(tmp_path, docs):
    old = {**PLAYING, "mode": "room", "row": {**ROW, "input": "room", "you_trusted": []}}
    assert live_page(tmp_path, docs, old, f"return {END};") == ""


def test_the_room_line_is_not_repeated_while_the_band_fades(tmp_path, docs):
    fading = {**ROOM_BAND_IN, "row": {**ROOM_BAND_IN["row"], "fade_db": -3.0}}
    assert live_page(tmp_path, docs, fading, f"return {END};") == ""


def test_a_direct_input_is_not_told_to_press_stop(tmp_path, docs):
    assert live_page(tmp_path, docs, PLAYED, "return $('live-how').hidden;") is True


def test_the_fade_is_one_line_that_says_she_can_bring_it_back(tmp_path, docs):
    out = live_page(tmp_path, docs, _rest(20, fade_db=-3.0), f"return {{ end: {END}, ...{FACTS} }};")
    assert out["say"] == "The band is finishing. Play and it comes back."
    assert out["end"] == "" and out["stop"] == "Stop"


def test_a_take_after_this_one_starts_with_a_clean_slate(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYED, f"""
        page.live = {json.dumps(_rest(13))};
        await page.poll();
        page.live = {json.dumps({"state": "idle"})};
        await page.poll();
        $('start').click(); await page.settle();
        page.live = {json.dumps(_rest(1))};
        await page.poll();
        page.live = {json.dumps(_rest(3))};
        await page.poll();
        return {END};""")
    assert out == ""


@pytest.mark.parametrize("level,word", [(-18.0, "Hearing you"), (-60.0, "Listening")])
def test_the_small_sign_says_in_a_word_whether_it_hears_her(tmp_path, docs, level, word):
    out = live_page(tmp_path, docs, {**WAITING, "level_db": level},
                    "return { word: $('hearing-word').textContent, shown: !$('hearing').hidden };")
    assert out == {"word": word, "shown": True}


def test_a_room_mic_with_the_band_in_still_shows_it_hears_the_room_and_claims_nothing_of_her(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "hear_me": False, "row": {**ROW, "input": "room", "you_trusted": []}}
    out = live_page(tmp_path, docs, room, f"""
        return {{ word: $('hearing-word').textContent, shown: !$('hearing').hidden, meter: !$('meter').hidden,
                 fill: $('meter-fill').style.width, level: {FACTS}.levelShown }};""")
    assert out == {"word": "Hearing the room", "shown": True, "meter": True, "fill": "70%", "level": False}


def test_a_quiet_room_mic_does_not_claim_to_hear_the_room(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "level_db": -60.0, "row": {**ROW, "input": "room", "you_trusted": []}}
    assert live_page(tmp_path, docs, room, "return $('hearing-word').textContent;") == "Listening"


def test_a_room_mic_with_no_recording_level_yet_has_no_sign(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "level_db": None, "row": {**ROW, "input": "room", "you_trusted": []}}
    assert live_page(tmp_path, docs, room, "return $('hearing').hidden;") is True


BLEED = {**PLAYING, "row": {**ROW, "you_trusted": []}}
BLEED_LINE = "Plume hears the band on this input. Press Stop and try another device in the menu."


def test_the_band_in_a_direct_input_is_said_on_the_first_view(tmp_path, docs):
    """A wrong guess at what a device is gets caught here, so it cannot sit in a closed menu."""
    out = live_page(tmp_path, docs, BLEED, f"return {{ ...{FACTS}, open: $('live-details').open, sign: !$('hearing').hidden }};")
    assert out["hint"] == BLEED_LINE
    assert out["why"] == "" and out["open"] is False and out["sign"] is False
    html = (STATIC / "index.html").read_text()
    live = html[html.index('<main id="live"'):html.index('<main id="record"')]
    assert 'id="live-hint"' in live[:live.index('<details id="live-details"')]


def test_a_word_from_the_bridge_comes_before_the_routing_line(tmp_path, docs):
    out = live_page(tmp_path, docs, {**BLEED, "hint": guitake.STALLED}, f"return {FACTS};")
    assert out["hint"] == guitake.STALLED


def test_a_room_mic_is_not_told_the_band_is_in_its_input(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "row": {**ROW, "input": "room", "you_trusted": []}}
    assert live_page(tmp_path, docs, room, f"return {FACTS};")["hint"] == ""


def test_a_direct_input_it_trusts_gets_no_routing_line(tmp_path, docs):
    assert live_page(tmp_path, docs, PLAYING, f"return {FACTS};")["hint"] == ""


def test_while_waiting_it_says_to_count_and_shows_a_level(tmp_path, docs):
    out = live_page(tmp_path, docs, {**WAITING, "mode": "room", "hear_me": False}, f"return {FACTS};")
    assert out["say"] == "Knock or clap 8 even beats."
    assert out["meter"] is True and out["facts"] is False and out["cues"] is False
    assert out["stop"] == "Stop" and out["stopDisabled"] is False


def test_a_clap_cannot_reach_a_pickup_so_a_direct_input_is_told_to_play_them(tmp_path, docs):
    out = live_page(tmp_path, docs, {**WAITING, "mode": "direct"}, f"return {FACTS};")
    assert out["say"] == "Play 8 even beats. Muted strings work well."


def test_after_start_the_page_looks_again_soon(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        $('start').click(); await page.settle();
        return { waiting: page.later.size, delay: page.lastDelay };""")
    assert out == {"waiting": 1, "delay": 1000}, "one poll waiting, and soon, however a take began"


def test_the_page_moves_from_waiting_to_playing_on_its_own(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, f"""
        const first = $('live-say').textContent;
        page.live = {json.dumps(PLAYING)};
        await page.poll();
        return {{ first, ...{FACTS} }};""")
    assert out["first"] != out["say"]
    assert out["say"] == "F major, midwest (a guess)"
    assert out["bar"] == "12" and out["tempo"] == "115 bpm" and out["stage"] == "Bass and chords"
    assert out["notes"] == "F A C" and out["level"] == "-18 dB"
    assert out["cues"] is True and out["why"] == ""


def test_a_key_the_player_typed_is_not_called_a_guess(tmp_path, docs):
    typed = {**PLAYING, "key": "F major", "style": "midwest", "row": {**ROW, "key_guess": False, "style_guess": False}}
    assert live_page(tmp_path, docs, typed, f"return {FACTS};")["say"] == "F major, midwest"


def test_on_a_room_mic_it_hides_what_it_cannot_trust_and_says_so(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "hear_me": False, "row": {**ROW, "input": "room", "you_trusted": []}}
    out = live_page(tmp_path, docs, room, f"return {FACTS};")
    assert out["notesShown"] is False and out["levelShown"] is False
    assert out["why"] == "Plume can't tell over the band on a room mic."
    assert out["delay"] == ""


def test_over_drums_on_a_room_mic_only_the_notes_are_shown(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "row": {**ROW, "input": "room", "stage": "drums", "you_trusted": ["you_notes"]}}
    out = live_page(tmp_path, docs, room, f"return {FACTS};")
    assert out["notesShown"] is True and out["levelShown"] is False


def test_the_live_delay_shows_only_when_monitoring_and_reported(tmp_path, docs):
    on = live_page(tmp_path, docs, PLAYING, f"return {FACTS};")
    off = live_page(tmp_path, docs, {**PLAYING, "hear_me": False}, f"return {FACTS};")
    unknown = live_page(tmp_path, docs, {**PLAYING, "stream": {**PLAYING["stream"], "reported_ms": None}}, f"return {FACTS};")
    assert "about 8 ms" in on["delay"] and "could be more" in on["delay"]
    assert off["delay"] == "" and unknown["delay"] == ""


@pytest.mark.parametrize("hint", [guitake.NO_COUNT, guitake.OPENING, "Nothing on channel 1. Try another channel in the menu."])
def test_a_hint_from_the_bridge_is_shown(tmp_path, docs, hint):
    out = live_page(tmp_path, docs, {**WAITING, "hint": hint}, f"return {FACTS};")
    assert out["hint"] == hint


def test_stop_sends_one_request_however_often_it_is_clicked(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, f"""
        $('live-stop').click();
        $('live-stop').click();
        await page.settle();
        $('live-stop').click();
        await page.settle();
        return {{ stops: page.posts('stop').length, ...{FACTS} }};""")
    assert out["stops"] == 1
    assert out["stop"] == "Stopping" and out["stopDisabled"] is True and out["say"] == "Stopping."


def test_a_wedged_stop_says_what_to_run(tmp_path, docs):
    out = live_page(tmp_path, docs, {"state": "stopping", "message": guitake.WEDGED, "row": None}, f"return {FACTS};")
    assert out["hint"] == guitake.WEDGED


def test_a_stop_clicked_during_start_still_reaches_the_take(tmp_path, docs):
    """The click can arrive before the bridge has any take to stop."""
    out = run_page(tmp_path, docs, """
        const open = page.hold('start');
        $('start').click(); await page.settle();
        $('live-stop').click(); await page.settle();
        const during = page.posts('stop').length;
        open(); await page.settle();
        return { during, after: page.posts('stop').length };""")
    assert out == {"during": 1, "after": 2}


@pytest.mark.parametrize("note", [None, guitake.STALLED, guitake.ENDED_EARLY, guitake.NO_PAGE])
def test_a_take_that_ends_lands_on_its_record(tmp_path, docs, note):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": note}
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps(done)};
        await page.poll();
        return {{ ...{LOOK}, title: $('rec-title').textContent, shown: !$('rec-note').hidden, note: $('rec-note').textContent }};""")
    assert out["screen"] == "record" and out["hash"] == f"#/take/{docs['ids']['tap']}"
    assert out["title"] == "tap name"
    assert out["shown"] is bool(note) and out["note"] == (note or "")


def test_the_ended_note_stays_with_its_own_take(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": guitake.STALLED}
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps(done)};
        await page.poll();
        await page.open(ids.other);
        return !$('rec-note').hidden;""")
    assert out is False


def test_back_from_a_landed_record_does_not_bounce(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = run_page(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        page.live = {json.dumps(done)};
        await page.poll();
        const landed = page.screen();
        page.history.go(-1); await page.settle();
        return {{ landed, back: page.screen() }};""")
    assert out == {"landed": "record", "back": "home"}


@pytest.mark.parametrize("message", [guitake.DEVICE_MISSING, guitake.OPEN_FAILED, guitake.AUDIO_PART_MISSING,
                                     guitake.SOUND_OFF, guitake.DID_NOT_START,
                                     "Could not make a folder in ~/Music/Plume."])
def test_a_failed_take_lands_on_home_in_plain_words(tmp_path, docs, message):
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps({"state": "failed", "message": message})};
        await page.poll();
        return {{ ...{LOOK}, note: $('home-note').textContent, shown: !$('home-note').hidden, start: !$('start').hidden }};""")
    assert out == {"screen": "home", "hash": "#/", "note": message, "shown": True, "start": True}
    assert "Traceback" not in message


def _guessed_docs(docs, tmp_path, *, key_guess=True, style_guess=True):
    rows = song(20, band_bar=8)
    for row in rows:
        row.update(key_guess=key_guess, style_guess=style_guess, style="midwest")
    root = tmp_path / "guessed"
    write_take(root / "g", rows, ending=ending_doc(), started_at=1800000009.0)
    srv = gui.GuiServer(roots=[root], log=lambda line: None, lister=lambda: DEVICES,
                        spawner=_no_spawner, sessions=lambda: []).start()
    try:
        client = Client(srv).login()
        takes = json.loads(client.get("/api/takes")[1])["takes"]
        full = {t["id"]: json.loads(client.get(f"/api/take/{t['id']}")[1]) for t in takes}
    finally:
        srv.close()
    return {**docs, "takes": takes, "docs": full, "ids": {t["folder"]: t["id"] for t in takes}}


def test_a_detected_key_and_style_stay_guesses_on_the_record_and_the_shelf(tmp_path, docs):
    guessed = _guessed_docs(docs, tmp_path)
    out = run_page(tmp_path, guessed, """
        $('nav-takes').click(); await page.settle();
        const shelf = page.$('shelf-' + ids.g).children[2].textContent;
        page.$('shelf-' + ids.g).click(); await page.settle();
        return { shelf, meta: $('rec-meta').textContent, sentence: $('ev-sentence').textContent };""")
    assert "F major, midwest (a guess)" in out["shelf"]
    assert "F major, midwest (a guess)" in out["meta"]
    assert out["sentence"] == "Bar 8. Bass and chords in, in F major (a guess)."


def test_a_typed_key_and_style_are_stated_plainly_afterwards(tmp_path, docs):
    guessed = _guessed_docs(docs, tmp_path, key_guess=False, style_guess=False)
    out = run_page(tmp_path, guessed, """
        $('nav-takes').click(); await page.settle();
        page.$('shelf-' + ids.g).click(); await page.settle();
        return { meta: $('rec-meta').textContent, sentence: $('ev-sentence').textContent };""")
    assert "F major, midwest" in out["meta"] and "guess" not in out["meta"]
    assert out["sentence"] == "Bar 8. Bass and chords in, in F major."


def test_a_false_start_lands_on_home_with_the_menu_open_and_says_why(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, """
        page.live = { state: 'idle', menu: true, message: 'Nothing came in on channel 1. Try another channel.' };
        await page.poll();
        return { screen: document.body.getAttribute('data-screen'), note: $('home-note').textContent,
                 shown: !$('home-note').hidden, menu: $('home-menu').open, start: !$('start').hidden };""")
    assert out == {"screen": "home", "note": "Nothing came in on channel 1. Try another channel.",
                   "shown": True, "menu": True, "start": True}


def test_a_start_answered_with_failure_lands_on_home_too(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [200, {{ state: 'failed', message: {json.dumps(guitake.DEVICE_MISSING)} }}];
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('home-note').textContent }};""")
    assert out["screen"] == "home" and out["note"] == guitake.DEVICE_MISSING


def test_a_note_about_the_last_try_goes_when_the_next_one_starts(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [200, {{ state: 'failed', message: {json.dumps(guitake.OPEN_FAILED)} }}];
        $('start').click(); await page.settle();
        $('start').click(); await page.settle();
        page.live = {{ state: 'idle' }};
        await page.poll();
        return {{ ...{LOOK}, shown: !$('home-note').hidden }};""")
    assert out["screen"] == "home" and out["shown"] is False


def test_a_complaint_about_a_field_goes_when_it_is_fixed_or_left(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const refuse = () => {{ page.replies.start = [400, {{ error: {json.dumps(guitake.BAD_KEY)} }}]; $('start').click(); }};
        refuse(); await page.settle();
        const first = !$('home-note').hidden;
        $('set-key')._value = 'F major'; $('set-key').fire('input');
        const typed = !$('home-note').hidden;
        refuse(); await page.settle();
        $('nav-takes').click(); await page.settle();
        $('wordmark').click(); await page.settle();
        return {{ first, typed, left: !$('home-note').hidden }};""")
    assert out == {"first": True, "typed": False, "left": False}


def test_an_old_failure_does_not_greet_a_new_visit(tmp_path, docs):
    out = run_page(tmp_path, docs, "return !$('home-note').hidden;",
                   live={"state": "failed", "message": guitake.OPEN_FAILED}, address="", stored=KEY)
    assert out is False


def test_stopping_before_any_sound_goes_home_quietly(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, f"""
        page.live = {{ state: 'idle' }};
        await page.poll();
        return {{ ...{LOOK}, shown: !$('home-note').hidden }};""")
    assert out == {"screen": "home", "hash": "#/", "shown": False}


def test_a_lost_bridge_is_said_and_recovers(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.down = true;
        await page.poll();
        const lost = {FACTS};
        page.down = false;
        await page.poll();
        return {{ lost: lost.hint, now: {FACTS}.hint }};""")
    assert "plume stop" in out["lost"] and out["now"] == ""


def test_the_take_can_be_left_and_found_again(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, f"""
        const onLive = $('nav-live').hidden;
        $('nav-takes').click(); await page.settle();
        const label = $('nav-live').textContent, shown = !$('nav-live').hidden, screen = page.screen();
        await page.poll();
        const stayed = page.screen();
        $('wordmark').click(); await page.settle();
        const home = page.screen();
        $('nav-takes').click(); await page.settle();
        $('nav-live').click(); await page.settle();
        return {{ onLive, label, shown, screen, stayed, home, back: page.screen(), polls: page.calls.filter((c) => c.path === '/api/live').length }};""")
    assert out["onLive"] is True and out["shown"] is True and out["label"] == "Playing"
    assert out["screen"] == "shelf" and out["stayed"] == "shelf"
    assert out["home"] == "live", "with a take running there is nothing to start"
    assert out["back"] == "live" and out["polls"] >= 2


def test_polling_goes_on_from_the_shelf_so_the_bridge_knows_a_page_is_there(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, """
        $('nav-takes').click(); await page.settle();
        const before = page.calls.filter((c) => c.path === '/api/live').length;
        await page.poll();
        await page.poll();
        return page.calls.filter((c) => c.path === '/api/live').length - before;""")
    assert out == 2


def test_a_take_ending_while_on_the_shelf_does_not_pull_the_page_away(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = live_page(tmp_path, docs, PLAYING, f"""
        $('nav-takes').click(); await page.settle();
        page.live = {json.dumps(done)};
        await page.poll();
        const still = page.screen(), label = $('nav-live').textContent;
        $('nav-live').click(); await page.settle();
        return {{ still, label, then: page.screen(), hash: page.location.hash, shown: !$('nav-live').hidden }};""")
    assert out["still"] == "shelf" and out["label"] == "Last take"
    assert out["then"] == "record" and out["hash"] == f"#/take/{docs['ids']['tap']}" and out["shown"] is False


def test_cues_go_through_the_bridge_and_refusals_are_said(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, """
        $('cue-busier').click(); await page.settle();
        const said = $('cue-note').textContent;
        page.replies.cue = [409, { error: 'The band did not answer that cue.' }];
        $('cue-sparser').click(); await page.settle();
        return { said, then: $('cue-note').textContent, posts: page.posts('cue') };""")
    assert out["posts"] == [{"cue": "busier"}, {"cue": "sparser"}]
    assert out["said"] == "Asked for busier." and out["then"] == "The band did not answer that cue."


# ---------------------------------------------------------- a take from a terminal

ELSEWHERE = {"state": "elsewhere", "pid": 4242, "message": guitake.TERMINAL_TAKE, "stopping": False}
OTHER = """{ screen: page.screen(), start: !$('start').hidden, box: !$('home-elsewhere').hidden,
  text: $('elsewhere-text').textContent, stop: $('elsewhere-stop').textContent, off: !!$('elsewhere-stop').disabled }"""


def test_a_terminal_take_replaces_start_and_can_be_stopped(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = {OTHER};
        page.replies.stop = [200, {json.dumps({**ELSEWHERE, "stopping": True})}];
        $('elsewhere-stop').click(); await page.settle();
        return {{ seen, stops: page.posts('stop'), now: {OTHER}, starts: page.posts('start').length }};""",
                   live=ELSEWHERE, address="", stored=KEY)
    assert out["seen"]["screen"] == "home" and out["seen"]["start"] is False and out["seen"]["box"] is True
    assert out["seen"]["text"] == guitake.TERMINAL_TAKE and out["seen"]["off"] is False
    assert out["stops"] == [{}] and out["starts"] == 0
    assert out["now"]["off"] is True and out["now"]["stop"] == "Stopping"


def test_start_comes_back_when_the_terminal_take_has_gone(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.live = {{ state: 'idle' }};
        await page.poll();
        return {OTHER};""", live=ELSEWHERE, address="", stored=KEY)
    assert out["start"] is True and out["box"] is False


def test_a_terminal_take_is_not_blamed_on_a_leftover_complaint(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [400, {{ error: {json.dumps(guitake.BAD_KEY)} }}];
        $('start').click(); await page.settle();
        page.live = {json.dumps(ELSEWHERE)};
        await page.poll();
        return !$('home-note').hidden;""")
    assert out is False


def test_start_refused_for_a_terminal_take_shows_that_take(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [409, {{ error: {json.dumps(guitake.TERMINAL_TAKE)}, live: {json.dumps(ELSEWHERE)} }}];
        $('start').click(); await page.settle();
        return {OTHER};""")
    assert out["screen"] == "home" and out["box"] is True and out["start"] is False


# ------------------------------------------------------------------- the first press

def steps(*names):
    return [{"answer": n, "label": n.title()} for n in names]


def setup_doc(step, line, *choices, device="dev0000000000000", device_name="Test Interface", **more):
    return {"step": step, "line": line, "note": None, "choices": steps(*choices), "done": step == "count",
            "seconds": 4.0, "device": device, "device_name": device_name, "mode": "direct", "channel": 1,
            "hear_me": False, **more}


LISTEN = setup_doc("listen", guitake.PLAY_SOMETHING)
HEADPHONES = setup_doc("headphones", "Headphones on?", "yes", "no")
HEARD = {"result": "ok", "channel": 2, "peak_db": -12.0}
COUNT = setup_doc("count", guitake.COUNT_LINE)
SEEN = """{ screen: page.screen(), say: $('setup-say').textContent, more: $('setup-more').hidden ? '' : $('setup-more').textContent, note: $('setup-note').hidden ? '' : $('setup-note').textContent,
  choices: $('setup-choices').children.map((b) => b.textContent), others: $('setup-others').children.map((b) => b.textContent),
  listening: !$('setup-listening').hidden }"""
MUTATIONS = "page.calls.filter((c) => c.method === 'POST').map((c) => c.path.replace('/api/', ''))"


def first_run(tmp_path, docs, body, setup=LISTEN, **options):
    return run_page(tmp_path, docs, body, setup=setup, **options)


def test_loading_the_page_listens_to_nothing(tmp_path, docs):
    out = first_run(tmp_path, docs, f"return {MUTATIONS};")
    assert out == []


def test_the_click_on_start_is_what_starts_the_listening(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        const open = page.hold('setup/listen');
        $('start').click(); await page.settle();
        const during = {SEEN}, posts = {MUTATIONS};
        open();
        return {{ during, posts, body: page.posts('setup/listen') }};""")
    assert out["during"]["screen"] == "setup" and out["during"]["say"] == guitake.PLAY_SOMETHING
    assert out["during"]["choices"] == [] and out["during"]["listening"] is True
    assert out["posts"] == ["setup/listen"] and out["body"] == [{}]


def test_two_clicks_on_start_listen_once(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.hold('setup/listen');
        $('start').click(); $('start').click(); await page.settle();
        return {MUTATIONS};""")
    assert out == ["setup/listen"]


def test_a_reload_on_the_setup_address_listens_to_nothing(tmp_path, docs):
    out = first_run(tmp_path, docs, f"return {{ screen: page.screen(), posts: {MUTATIONS} }};", address="#/setup", stored=KEY)
    assert out == {"screen": "home", "posts": []}


def test_a_good_level_goes_straight_to_the_next_question_and_says_who_heard(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps({**HEADPHONES, "channel": 2, "heard": HEARD, "others": []})}];
        $('start').click(); await page.settle();
        return {SEEN};""")
    assert out["screen"] == "setup" and out["say"] == "Headphones on?" and out["choices"] == ["Yes", "No"]
    assert out["note"] == "Heard you on Test Interface, channel 2." and out["listening"] is False


def test_a_level_that_needs_a_hand_says_so_and_offers_both_ways_on(tmp_path, docs):
    hot = {**setup_doc("listen", guitake.HOT_GAIN, "again", "start"),
           "heard": {"result": "hot", "channel": 1, "peak_db": -0.2}, "others": []}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps(hot)}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        page.hold('setup/listen');
        $('setup-again').click(); await page.settle();
        return {{ seen, again: {SEEN}, posts: {MUTATIONS} }};""")
    assert out["seen"]["say"] == "A bit hot." and out["seen"]["choices"] == ["Again", "Start"]
    assert out["seen"]["more"] == "Turn the gain down a little and play again."
    assert out["again"]["say"] == guitake.PLAY_SOMETHING and out["again"]["listening"] is True
    assert out["posts"] == ["setup/listen", "setup/listen"]


def test_start_anyway_never_traps_her_and_never_turns_on_hear_me(tmp_path, docs):
    nothing = {**setup_doc("listen", "Nothing came in on Test Interface.", "again", "start"),
               "heard": {"result": "nothing", "channel": None, "peak_db": None}, "others": []}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps(nothing)}];
        $('start').click(); await page.settle();
        $('setup-start').click(); await page.settle();
        return {{ screen: page.screen(), hash: page.location.hash, body: page.posts('start') }};""")
    assert out == {"screen": "live", "hash": "#/live", "body": [{}]}


def test_when_nothing_arrives_another_device_can_be_tried(tmp_path, docs):
    nothing = {**setup_doc("listen", "Nothing came in on Test Interface.", "again", "start"),
               "heard": {"result": "nothing", "channel": None, "peak_db": None},
               "others": [{"id": "b" * 16, "name": "Second Box"}]}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps(nothing)}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        page.replies['setup/listen'] = [200, {json.dumps({**HEADPHONES, "device": "b" * 16, "device_name": "Second Box", "heard": HEARD, "others": []})}];
        $('setup-others').children[0].click(); await page.settle();
        const then = {SEEN};
        $('setup-no').click(); await page.settle();
        return {{ seen, then, listens: page.posts('setup/listen'), answers: page.posts('setup/answer') }};""")
    assert out["seen"]["others"] == ["Try Second Box"]
    assert out["listens"] == [{}, {"device": "b" * 16}]
    assert out["then"]["say"] == "Headphones on?" and out["then"]["note"].startswith("Heard you on Second Box")
    assert out["answers"] == [{"device": "b" * 16, "step": "headphones", "answer": "no"}]


def test_the_channel_setup_just_heard_is_not_overruled_by_a_page_that_knew_less(tmp_path, docs):
    """The page's device rows are from before she was heard, on channel 1."""
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = first_run(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        page.replies['setup/listen'] = [200, {json.dumps({**COUNT, "channel": 3, "heard": HEARD, "others": []})}];
        $('start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{"device": mic["id"]}]


def test_a_device_picked_in_the_menu_is_the_one_listened_to(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    out = first_run(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        page.hold('setup/listen');
        $('start').click(); await page.settle();
        return {{ asked: page.calls.filter((c) => c.path.startsWith('/api/setup?')).map((c) => c.path),
                  body: page.posts('setup/listen'), note: {SEEN}.note }};""")
    assert out["asked"] == [f"/api/setup?device={mic['id']}"]
    assert out["body"] == [{"device": mic["id"]}]
    assert out["note"] == "Listening on Test Mic."


def test_a_long_wait_says_the_computer_may_be_asking(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.hold('setup/listen');
        $('start').click(); await page.settle();
        const before = {SEEN}.note;
        await page.poll();
        return {{ before, after: {SEEN}.note }};""")
    assert out["before"] == "" and out["after"] == "This computer may be asking to use the microphone."


def test_an_answer_goes_to_the_bridge_and_a_done_setup_starts_with_no_click(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        page.replies['setup/answer'] = [200, {json.dumps(COUNT)}];
        $('setup-no').click(); await page.settle();
        return {{ screen: page.screen(), hash: page.location.hash, answers: page.posts('setup/answer'), starts: page.posts('start') }};""",
                    setup=HEADPHONES)
    assert out["answers"] == [{"device": HEADPHONES["device"], "step": "headphones", "answer": "no"}]
    assert out["screen"] == "live" and out["hash"] == "#/live" and out["starts"] == [{"device": COUNT["device"]}]


def test_a_question_already_answered_is_not_asked_and_nothing_is_listened_to(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        return {{ ...{SEEN}, posts: {MUTATIONS} }};""", setup=HEADPHONES)
    assert out["say"] == "Headphones on?" and out["posts"] == [] and out["listening"] is False


def test_a_finished_setup_goes_straight_to_the_count(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        return {{ screen: page.screen(), posts: {MUTATIONS} }};""", setup=COUNT)
    assert out == {"screen": "live", "posts": ["start"]}


def test_a_finished_setup_starts_on_the_device_it_heard_not_one_picked_again(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        $('start').click(); await page.settle();
        return page.posts('start');""", setup=COUNT)
    assert out == [{"device": COUNT["device"]}]


def test_start_anyway_is_not_held_to_a_device_nothing_was_heard_on(tmp_path, docs):
    """No device name means the bridge's own safe choice."""
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps({**setup_doc("listen", "Nothing came in.", "again", "start"), "others": []})}];
        $('start').click(); await page.settle();
        $('setup-start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{}]


def test_a_device_gone_from_a_finished_setup_goes_home_and_says_so(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['start'] = [400, {{ error: {json.dumps(guitake.DEVICE_MISSING)} }}];
        $('start').click(); await page.settle();
        return {{ screen: page.screen(), note: $('home-note').textContent }};""", setup=COUNT)
    assert out == {"screen": "home", "note": guitake.DEVICE_MISSING}


def test_a_double_click_on_an_answer_does_not_answer_the_next_question(tmp_path, docs):
    hears = setup_doc("hears_self", "Can you hear yourself?", "yes", "no")
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        page.replies['setup/answer'] = [200, {json.dumps(hears)}];
        $('setup-yes').click();
        await page.settle();
        $('setup-yes').fire('click', {{ detail: 2 }});
        await page.settle();
        return {{ answers: page.posts('setup/answer').length, say: {SEEN}.say }};""", setup=HEADPHONES)
    assert out == {"answers": 1, "say": "Can you hear yourself?"}


def test_two_quick_answers_send_one(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        $('start').click(); await page.settle();
        const open = page.hold('setup/answer');
        $('setup-yes').click(); $('setup-no').click(); await page.settle();
        open();
        return page.posts('setup/answer').length;""", setup=HEADPHONES)
    assert out == 1


@pytest.mark.parametrize("leave", ["$('setup-leave').click();", "$('wordmark').click();", "$('nav-takes').click();",
                                   "page.history.go(-1);"])
def test_leaving_while_it_listens_lets_go_of_the_device_and_keeps_nothing_running(tmp_path, docs, leave):
    out = first_run(tmp_path, docs, f"""
        const open = page.hold('setup/listen');
        $('start').click(); await page.settle();
        {leave}
        await page.settle();
        const screen = page.screen(), posts = {MUTATIONS};
        page.replies['setup/listen'] = undefined;
        open(); await page.settle();
        return {{ screen, posts, now: page.screen(), starts: page.posts('start').length }};""")
    assert out["screen"] != "setup" and out["now"] == out["screen"] and out["starts"] == 0
    assert out["posts"] == ["setup/listen", "setup/cancel"]


def test_a_reload_or_a_closed_tab_calls_off_a_listen_in_progress(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        page.hold('setup/listen');
        $('start').click(); await page.settle();
        page.fireWindow('pagehide');
        await page.settle();
        const sent = page.calls.filter((c) => c.path === '/api/setup/cancel');
        return { sent: sent.length, outlives: sent[0] && sent[0].keepalive, key: sent[0] && sent[0].key, body: sent[0] && sent[0].body };""")
    assert out == {"sent": 1, "outlives": True, "key": KEY, "body": {}}


@pytest.mark.parametrize("setup", ["HEADPHONES", "COUNT"])
def test_a_tab_going_away_with_nothing_listening_sends_nothing(tmp_path, docs, setup):
    doc = {"HEADPHONES": HEADPHONES, "COUNT": COUNT}[setup]
    out = first_run(tmp_path, docs, """
        $('start').click(); await page.settle();
        page.fireWindow('pagehide');
        await page.settle();
        return page.posts('setup/cancel').length;""", setup=doc)
    assert out == 0


def test_leaving_at_a_question_sends_nothing_it_does_not_need_to(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        $('setup-leave').click(); await page.settle();
        return {{ screen: page.screen(), posts: {MUTATIONS} }};""", setup=HEADPHONES)
    assert out == {"screen": "home", "posts": []}


def test_an_answer_that_comes_back_after_she_left_starts_nothing(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        const open = page.hold('setup/answer');
        page.replies['setup/answer'] = [200, {json.dumps(COUNT)}];
        $('setup-no').click(); await page.settle();
        $('setup-leave').click(); await page.settle();
        open(); await page.settle();
        return {{ screen: page.screen(), starts: page.posts('start').length }};""", setup=HEADPHONES)
    assert out == {"screen": "home", "starts": 0}


def test_when_setup_cannot_be_asked_start_still_starts(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        page.failNext.setup = 1;
        $('start').click(); await page.settle();
        return { screen: page.screen(), body: page.posts('start') };""")
    assert out == {"screen": "live", "body": [{}]}


@pytest.mark.parametrize("op,click", [("setup/listen", ""), ("setup/answer", "$('setup-yes').click(); await page.settle();")])
def test_a_failed_step_leaves_a_way_on(tmp_path, docs, op, click):
    setup = LISTEN if op == "setup/listen" else HEADPHONES
    out = first_run(tmp_path, docs, f"""
        page.failNext[{json.dumps(op)}] = 1;
        $('start').click(); await page.settle();
        {click}
        const seen = {SEEN};
        $('setup-start').click(); await page.settle();
        return {{ seen, screen: page.screen(), body: page.posts('start') }};""", setup=setup)
    assert out["seen"]["choices"] == ["Play again", "Start anyway"] and out["seen"]["listening"] is False
    assert out["screen"] == "live" and out["body"] == [{}]


def test_a_refused_step_says_the_bridges_words(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [400, {{ error: {json.dumps(guitake.DEVICE_MISSING)} }}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        return seen.say + ' ' + seen.more;""")
    assert out == guitake.DEVICE_MISSING


def test_a_low_level_offers_another_device_and_listens_on_it(tmp_path, docs):
    low = {**setup_doc("listen", guitake.LOW_GAIN, "again", "start"),
           "heard": {"result": "low", "channel": 1, "peak_db": -50.0}, "others": [{"id": "b" * 16, "name": "Second Box"}]}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps(low)}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        page.hold('setup/listen');
        $('setup-others').children[0].click(); await page.settle();
        return {{ seen, listens: page.posts('setup/listen') }};""")
    assert out["seen"]["others"] == ["Try Second Box"]
    assert out["listens"] == [{}, {"device": "b" * 16}]


def test_with_nothing_plugged_in_there_is_one_thing_to_press(tmp_path, docs):
    none = {**setup_doc("listen", guitake.NO_INPUT_NEXT), "choices": [{"answer": "again", "label": "Try again"}],
            "device": None, "device_name": None, "heard": {"result": "failed", "channel": None, "peak_db": None}, "others": []}
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps(none)}];
        $('start').click(); await page.settle();
        const seen = {SEEN};
        page.hold('setup/listen');
        $('setup-again').click(); await page.settle();
        return {{ seen, listens: page.posts('setup/listen').length, starts: page.posts('start').length }};""")
    assert out["seen"]["say"] == "Plume could not find a microphone or an audio interface."
    assert out["seen"]["more"] == "Plug one in, then try again."
    assert out["seen"]["choices"] == ["Try again"] and out["listens"] == 2 and out["starts"] == 0


def test_a_lost_key_during_setup_says_how_to_get_a_new_link(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        page.replies['setup/listen'] = [401, { error: 'no' }];
        $('start').click(); await page.settle();
        return page.screen();""")
    assert out == "problem"


def test_a_terminal_take_found_while_listening_goes_home_and_says_so(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [409, {{ error: {json.dumps(guitake.TERMINAL_TAKE)}, live: {json.dumps(ELSEWHERE)} }}];
        $('start').click(); await page.settle();
        return {OTHER};""")
    assert out["screen"] == "home" and out["box"] is True and out["start"] is False


def test_a_take_that_starts_elsewhere_during_setup_takes_the_page_there(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        const open = page.hold('setup/listen');
        $('start').click(); await page.settle();
        page.live = {json.dumps(WAITING)};
        await page.poll();
        const screen = page.screen(), posts = {MUTATIONS};
        open(); await page.settle();
        return {{ screen, posts, now: page.screen() }};""")
    assert out["screen"] == "live" and out["now"] == "live" and "setup/cancel" in out["posts"]


def test_no_answer_in_the_doc_still_leaves_a_way_on(tmp_path, docs):
    """What is plugged in changed under her: the bridge asks for step 1 again, with no click yet."""
    out = first_run(tmp_path, docs, f"""
        $('start').click(); await page.settle();
        page.replies['setup/answer'] = [200, {json.dumps(LISTEN)}];
        $('setup-yes').click(); await page.settle();
        return {{ ...{SEEN}, listens: page.posts('setup/listen').length }};""", setup=HEADPHONES)
    assert out["say"] == guitake.PLAY_SOMETHING and out["listens"] == 0 and out["listening"] is False
    assert out["choices"] == ["Play again", "Start anyway"]


def test_the_choices_never_land_under_a_second_click_on_start():
    css = (STATIC / "plume.css").read_text()
    start = int(re.search(r"#start \{ height: (\d+)px", css).group(1))
    say = int(re.search(r"#setup-words \{[^}]*min-height: (\d+)px", css).group(1))
    gap = int(re.search(r"\.column \{[^}]*gap: (\d+)px", css).group(1))
    assert say + gap > start


def test_nothing_gives_a_choice_focus_so_a_stray_key_cannot_answer(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        $('start').click(); await page.settle();
        return page.activeElement.id;""", setup=HEADPHONES)
    assert out == "setup"


def test_a_double_click_on_start_selects_no_words_where_it_lands():
    css = (STATIC / "plume.css").read_text()
    rule = re.search(r"#setup, #live-say \{([^}]*)\}", css).group(1)
    assert "user-select: none" in rule and "-webkit-user-select: none" in rule


def test_a_choice_looks_pressed_while_its_answer_is_on_the_way(tmp_path, docs):
    out = first_run(tmp_path, docs, """
        $('start').click(); await page.settle();
        const open = page.hold('setup/answer');
        $('setup-yes').click(); await page.settle();
        const off = [$('setup-yes').disabled, $('setup-no').disabled];
        open();
        return off;""", setup=HEADPHONES)
    assert out == [True, True]


def test_set_up_again_is_one_link_in_the_menu_and_starts_nothing(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    menu = html[html.index('<details id="home-menu"'):html.index("</details>")]
    assert ">Set up again</button>" in menu and html.count("Set up again") == 1
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "hear_me": True})}];
        {TICK}
        const before = $('hear-me').checked;
        $('set-again').click(); await page.settle();
        const seen = {{ note: $('set-again-note').textContent, ticked: $('hear-me').checked }};
        return {{ ...seen, before, posts: {MUTATIONS}, bodies: page.posts('setup/reset'), screen: page.screen() }};""")
    assert out["before"] is True
    assert out["posts"] == ["setup/answer", "setup/reset"] and out["bodies"] == [{}] and out["screen"] == "home"
    assert out["note"] == "Start will ask again." and out["ticked"] is False


def test_a_set_up_again_that_fails_says_so(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        page.failNext['setup/reset'] = 1;
        $('set-again').click(); await page.settle();
        return $('set-again-note').textContent;""")
    assert out == "That did not reach plume gui."


def test_real_bridge_answers_walk_the_whole_flow_into_the_take(tmp_path, docs, monkeypatch):
    """The page against documents a real bridge wrote, with a fake meter and a spawner that refuses."""
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "walk-state"))
    meter = lambda name, seconds, cancelled: {"channels": [{"channel": 1, "peak_db": -90.0}, {"channel": 2, "peak_db": -12.0},  # noqa: E731
                                                           {"channel": 3, "peak_db": -90.0}, {"channel": 4, "peak_db": -90.0}]}
    srv = gui.GuiServer(roots=[tmp_path / "walk"], log=lambda line: None, lister=lambda: DEVICES, spawner=_no_spawner,
                        sessions=lambda: [], meter=meter, write_state=False).start()
    try:
        client = Client(srv).login()
        site = {"Sec-Fetch-Site": "same-origin", "Origin": client.origin, "Content-Type": "application/json"}

        def post(op, body):
            return json.loads(client.request("POST", f"/api/{op}", headers=site, body=body)[1])
        first = json.loads(client.get("/api/setup", headers={"Sec-Fetch-Site": "same-origin"})[1])
        heard = post("setup/listen", {})
        hears = post("setup/answer", {"step": "headphones", "answer": "yes"})
        volume = post("setup/answer", {"step": "hears_self", "answer": "no"})
        done = post("setup/answer", {"step": "volume", "answer": "still"})
    finally:
        srv.close()
    assert first["step"] == "listen" and heard["step"] == "headphones" and done["done"] is True
    out = first_run(tmp_path, docs, f"""
        const said = [];
        page.replies['setup/listen'] = [200, {json.dumps(heard)}];
        $('start').click(); await page.settle();
        said.push({SEEN});
        page.replies['setup/answer'] = [200, {json.dumps(hears)}];
        $('setup-yes').click(); await page.settle();
        said.push({SEEN});
        page.replies['setup/answer'] = [200, {json.dumps(volume)}];
        $('setup-no').click(); await page.settle();
        said.push({SEEN});
        page.replies['setup/answer'] = [200, {json.dumps(done)}];
        $('setup-still').click(); await page.settle();
        return {{ said, screen: page.screen(), body: page.posts('start'), answers: page.posts('setup/answer') }};""", setup=first)
    assert [s["say"] for s in out["said"]] == ["Headphones on?", "Can you hear yourself?", "Check your headphone volume."]
    assert out["said"][0]["note"] == "Heard you on Test Interface, channel 2."
    assert out["said"][2]["choices"] == ["That was it", "Still nothing"]
    assert out["screen"] == "live", "Hear me comes from what the bridge saved"
    assert out["body"] == [{"device": done["device"]}], "and nothing but the device she was heard on is named"
    assert [a["step"] for a in out["answers"]] == ["headphones", "hears_self", "volume"]


# --------------------------------------------------------------------- honest words

def test_no_em_dash_in_anything_a_person_reads():
    for name in ("index.html", "app.js", "live.js", "session.js"):
        assert "\u2014" not in (STATIC / name).read_text(encoding="utf-8"), name
