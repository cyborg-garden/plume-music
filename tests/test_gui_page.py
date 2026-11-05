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


def test_with_nothing_changed_start_sends_the_bridges_own_choice(tmp_path, docs):
    interface = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = run_page(tmp_path, docs, """
        $('start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{"device": interface["id"], "channel": 1, "hear_me": True}]


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
            "hear_me": {"allowed": bool(chosen and chosen["hear_me"]),
                        "default": bool(chosen and chosen["hear_me"]), "reported_ms": None},
            "folder": "~/Music/Plume", "styles": ["lofi", "midwest"], "problem": None}


def test_hear_me_is_on_by_default_for_an_interface(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["row"] is True and out["on"] is True and out["words"] is True


def test_hear_me_is_not_offered_on_a_built_in_mic(tmp_path, docs):
    docs["home"] = home_for(ROOM_ONLY)
    out = run_page(tmp_path, docs, f"""
        const seen = {HEAR};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["row"] is False and out["words"] is False and out["on"] is False
    assert out["body"]["hear_me"] is False


def test_choosing_the_mic_takes_hear_me_away_and_back(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    interface = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = run_page(tmp_path, docs, f"""
        const pick = (id) => {{ $('set-device').value = id; $('set-device').fire('change'); }};
        pick({json.dumps(mic["id"])});
        const onMic = {{ ...{HEAR}, line: $('home-summary').textContent, channel: !$('set-channel').hidden }};
        pick({json.dumps(interface["id"])});
        const back = {{ ...{HEAR}, line: $('home-summary').textContent, channel: !$('set-channel').hidden }};
        return {{ onMic, back }};""")
    assert out["onMic"]["row"] is False and out["onMic"]["line"] == "Test Mic, room mic" and out["onMic"]["channel"] is False
    assert out["back"]["row"] is True and out["back"]["on"] is True and out["back"]["channel"] is True
    assert out["back"]["line"] == "Test Interface, channel 1"


def test_turning_hear_me_off_is_sent_and_kept_across_a_visit_to_the_shelf(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        $('hear-me').checked = false; $('hear-me').fire('change');
        $('nav-takes').click(); await page.settle();
        $('wordmark').click(); await page.settle();
        const short = $('hear-short').hidden;
        $('start').click(); await page.settle();
        return { short, body: page.posts('start')[0] };""")
    assert out["body"]["hear_me"] is False and out["short"] is True


def test_the_delay_is_called_what_it_is(tmp_path, docs):
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["reported_ms"] = 12.4
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["delay"] == ("Use headphones. Plume adds about 12 ms, as your device reports it. "
                            "This is not a measured round trip.")


def test_the_delay_shown_is_the_chosen_devices_own(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    other = {"name": "Test Interface Two", "inputs": 2, "outputs": 2}
    docs["home"]["devices"].append({**guitake.describe_devices([other])[0]})
    mic["reported_ms"] = 8.3
    docs["home"]["hear_me"]["reported_ms"] = 8.3     # not what a device says about another
    out = run_page(tmp_path, docs, f"return {HEAR};")
    assert out["delay"] == "Use headphones. Plume shows the delay your device reports once a take starts."


def test_the_direct_button_line_is_there_once(tmp_path):
    html = (STATIC / "index.html").read_text()
    assert html.count("A Direct button on your interface has no latency. Using both doubles the sound.") == 1


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


def test_while_waiting_it_says_to_count_and_shows_a_level(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, f"return {FACTS};")
    assert out["say"] == "Knock or clap 8 even beats."
    assert out["meter"] is True and out["facts"] is False and out["cues"] is False
    assert out["stop"] == "Stop" and out["stopDisabled"] is False


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
    assert out["notesShown"] is False and out["levelShown"] is False and out["meter"] is False
    assert out["why"] == "Plume can't tell over the band on a room mic."
    assert out["delay"] == ""


def test_over_drums_on_a_room_mic_only_the_notes_are_shown(tmp_path, docs):
    room = {**PLAYING, "mode": "room", "row": {**ROW, "input": "room", "stage": "drums", "you_trusted": ["you_notes"]}}
    out = live_page(tmp_path, docs, room, f"return {FACTS};")
    assert out["notesShown"] is True and out["levelShown"] is False and out["meter"] is False


def test_the_live_delay_shows_only_when_monitoring_and_reported(tmp_path, docs):
    on = live_page(tmp_path, docs, PLAYING, f"return {FACTS};")
    off = live_page(tmp_path, docs, {**PLAYING, "hear_me": False}, f"return {FACTS};")
    unknown = live_page(tmp_path, docs, {**PLAYING, "stream": {**PLAYING["stream"], "reported_ms": None}}, f"return {FACTS};")
    assert "about 8 ms" in on["delay"] and "not a measured round trip" in on["delay"]
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


# --------------------------------------------------------------------- honest words

def test_no_em_dash_in_anything_a_person_reads():
    for name in ("index.html", "app.js", "live.js", "session.js"):
        assert "\u2014" not in (STATIC / name).read_text(encoding="utf-8"), name
