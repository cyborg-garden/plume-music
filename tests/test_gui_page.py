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
    # One take whose device said how long the band took to come back, in the file a take keeps for it.
    (root / "other" / "stream.json").write_text(json.dumps({"calibrated_ms": 250.0, "reported_ms": 480.0}))
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
    for name in ("app", "stone", "layout", "session", "live", "guide", "video"):
        source = (STATIC / f"{name}.js").read_text()
        (work / f"{name}.mjs").write_text(re.sub(r"'/static/(\w+)\.js'", r"'./\1.mjs'", source))
    assert "/static/" not in (work / "app.mjs").read_text()
    shutil.copy(STUB, work / "stub.mjs")
    settings = {"takes": docs["takes"], "docs": docs["docs"], "home": docs["home"],
                "guide": {"pages": gui.guide_pages()}, **options}
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
    assert out["plays"] == 0 and out["src"] == "" and out["audio"] == 0


def test_only_the_listen_button_sets_a_source(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-machine').click();
        await page.settle();
        return {SOUND};""")
    assert out["plays"] == 1
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
    assert out["pauses"] > out["before"] and out["src"] == ""


def test_moving_to_another_record_stops_the_sound(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        await page.shelf();
        await page.open(ids.other);
        $('listen-human').click();
        page.history.go(-2);
        await page.settle();
        return {{ title: $('rec-title').textContent, ...{SOUND} }};""")
    assert out["title"] == "tap name" and out["src"] == ""


def test_a_direct_take_starts_on_together_and_play_is_what_starts_it(tmp_path, docs):
    docs["docs"][docs["ids"]["tap"]].update(together=True, recorded_from="input")
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        const before = {{ ...{SOUND}, label: $('listen-play').textContent,
                          chosen: $('listen-together').getAttribute('aria-pressed') }};
        $('listen-play').click(); await page.settle();
        const on = {{ ...{SOUND}, label: $('listen-play').textContent }};
        $('listen-play').click(); await page.settle();
        return {{ before, on, off: {SOUND}, label: $('listen-play').textContent }};""")
    assert out["before"]["plays"] == 0 and out["before"]["src"] == "" and out["before"]["label"] == "Play"
    assert out["before"]["chosen"] == "true"
    assert out["on"]["plays"] == 1 and out["on"]["src"].endswith(f"/audio/together?k={KEY}") and out["on"]["label"] == "Pause"
    assert out["off"]["pauses"] > out["on"]["pauses"] and out["label"] == "Play"


def test_a_take_with_no_together_offers_only_what_it_has(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        return { together: !!$('listen-together'), human: !!$('listen-human'),
                 chosen: $('listen-human').getAttribute('aria-pressed') };""")
    assert out == {"together": False, "human": True, "chosen": "true"}


def test_all_moments_and_the_log_row_sit_together_under_the_player():
    html = (STATIC / "index.html").read_text()
    assert html.index('id="sheet"') < html.index('id="listen"') < html.index('id="rec-details"')
    box = html[html.index('id="rec-details"'):html.index('id="rec-lower"')]
    assert 'id="ev-more"' in box and 'id="raw"' in box


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


def test_home_stands_plumes_own_stone_at_the_size_it_was_carved():
    html = (STATIC / "index.html").read_text()
    home = html[html.index('<main id="home"'):html.index("</main>", html.index('<main id="home"'))]
    assert '<canvas id="home-stone" width="480" height="540"' in home
    assert "#home-stone { position: absolute; left: 0; bottom: 28px; width: 220px;" in (STATIC / "plume.css").read_text()


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
    """It knows what setup learned, and a page that never changed anything must not overrule it.
    The style is the one choice always said, since the page shows it: "" is Let plume guess."""
    out = run_page(tmp_path, docs, """
        $('start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{"style": ""}]


def test_start_works_with_no_device_list_yet(tmp_path, docs):
    """Before home has loaded the page does not know her saved style, so it shows no style and
    sends none, and the bridge uses the one it saved."""
    docs["home"] = None
    out = run_page(tmp_path, docs, """
        const line = $('home-gear').textContent, hear = $('hear-row').hidden, style = $('home-style-row').hidden;
        $('start').click(); await page.settle();
        return { line, hear, style, body: page.posts('start') };""")
    assert out == {"line": "", "hear": True, "style": True, "body": [{}]}


def test_typed_key_and_chosen_style_and_channel_go_in_the_body(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        $('set-key')._value = '  F major ';
        $('set-key').fire('input');
        $('home-style').value = 'midwest';
        $('home-style').fire('change'); await page.settle();
        $('set-channel').value = '3';
        $('set-channel').fire('change');
        const summary = $('home-gear').textContent;
        $('start').click(); await page.settle();
        return { summary, body: page.posts('start')[0] };""")
    assert out["body"]["key"] == "F major" and out["body"]["style"] == "midwest" and out["body"]["channel"] == 3
    # The style is said once, by its own control, not again in the line under it.
    assert out["summary"] == "Test Interface, channel 3, F major"


# ---------------------------------------------------------------------- the style

STYLE = """{ value: $('home-style').value, labels: $('home-style').children.map((o) => o.textContent),
  values: $('home-style').children.map((o) => o.value), shown: !$('home-style-row').hidden }"""


def test_the_style_starts_on_let_plume_guess_and_start_sends_none(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = {STYLE};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0], saved: page.posts('style') }};""")
    assert out["value"] == "" and out["shown"] is True
    assert out["labels"] == ["Let plume guess", "Folk", "Hip hop", "House", "Jazz", "Lo-fi", "Midwest emo",
                             "R&B", "Synthwave", "Trap"]
    assert out["values"][1:] == docs["home"]["styles"]
    assert out["body"] == {"style": ""} and out["saved"] == []


def test_a_remembered_style_is_shown_and_used(tmp_path, docs):
    docs["home"]["style"] = "folk"
    out = run_page(tmp_path, docs, f"""
        const seen = {STYLE};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["value"] == "folk" and out["body"] == {"style": "folk"}


def test_choosing_a_style_is_saved_at_once_and_going_back_to_guessing_too(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        $('home-style').value = 'jazz'; $('home-style').fire('change'); await page.settle();
        $('home-style').value = ''; $('home-style').fire('change'); await page.settle();
        $('start').click(); await page.settle();
        return {{ ...{STYLE}, saved: page.posts('style'), body: page.posts('start')[0] }};""")
    assert out["saved"] == [{"style": "jazz"}, {"style": ""}]
    assert out["value"] == "" and out["body"] == {"style": ""}


def test_a_refused_style_is_said_kindly_and_the_choice_goes_back(tmp_path, docs):
    docs["home"]["style"] = "folk"
    out = run_page(tmp_path, docs, f"""
        page.replies.style = [400, {{ error: {json.dumps(guitake.UNKNOWN_STYLE)} }}];
        $('home-style').value = 'polka'; $('home-style').fire('change'); await page.settle();
        return {{ ...{STYLE}, ...{LOOK}, note: $('home-note').textContent, noteShown: !$('home-note').hidden,
                  disabled: !!$('home-style').disabled }};""")
    assert out["value"] == "folk" and out["disabled"] is False
    assert out["note"] == guitake.UNKNOWN_STYLE and out["noteShown"] is True
    assert out["screen"] == "home"


def test_a_style_refused_at_start_is_said_at_home_where_it_can_be_fixed(tmp_path, docs):
    """A page older than the bridge's list: the choice goes back to what the bridge remembers."""
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [400, {{ error: {json.dumps(guitake.UNKNOWN_STYLE)}, field: 'style' }}];
        $('home-style').value = 'polka';
        $('start').click(); await page.settle();
        return {{ ...{STYLE}, ...{LOOK}, note: $('home-note').textContent }};""")
    assert out["screen"] == "home" and out["hash"] == "#/"
    assert out["note"] == guitake.UNKNOWN_STYLE and out["value"] == ""


def test_a_style_that_did_not_reach_the_bridge_is_said_and_put_back(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.failNext.style = 1;
        $('home-style').value = 'trap'; $('home-style').fire('change'); await page.settle();
        return {{ ...{STYLE}, note: $('home-note').textContent }};""")
    assert out["value"] == "" and out["note"]


def test_a_refused_field_goes_to_settings_and_says_so_there(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.replies.start = [400, {{ error: {json.dumps(guitake.BAD_KEY)} }}];
        $('set-key')._value = '-rf';
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('settings-note').textContent, shown: !$('settings-note').hidden,
                  key: $('set-key').value }};""")
    assert out["screen"] == "settings" and out["hash"] == "#/settings"
    assert out["note"] == guitake.BAD_KEY and out["shown"] is True
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
            "outputs": guitake.describe_outputs(devices), "output": None, "send_channel": None, "output_problem": None,
            "folder": "~/Music/Plume", "styles": ["lofi", "midwest"], "problem": None}


def _saved_hear_me(docs, name="Test Interface"):
    for device in docs["home"]["devices"]:
        if device["name"] == name:
            device["hear_me"], device["headphones"] = True, True


def _settings_html():
    html = (STATIC / "index.html").read_text()
    return html[html.index('<main id="settings"'):html.index('<main id="setup"')]


def test_the_first_screen_is_one_start_with_hear_me_in_settings():
    html = (STATIC / "index.html").read_text()
    first = html[html.index('<main id="home"'):html.index('<main id="settings"')]
    assert "hear" not in first.lower() and "headphones" not in first.lower()
    # One quiet choice beside Start, the style, and nothing else to fill in.
    assert first.count("<select") == 1 and 'id="home-style"' in first
    assert "<input" not in first and "<details" not in first
    assert "set-style" not in html, "one place to choose a style, not two"
    assert 'id="hear-row"' in _settings_html() and 'id="hear-direct"' in _settings_html()


def test_hear_me_is_off_until_someone_says_otherwise(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = {HEAR};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["row"] is True and out["on"] is False and out["words"] is False
    assert out["body"] == {"style": ""}


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
    assert out["on"] is True and out["words"] is True and out["body"] == {"style": ""}


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
    assert out["ticked"] is True and out["start"] == [{"style": ""}]


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
    assert out == [{"style": ""}]


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
        const onMic = {{ ...{HEAR}, line: $('home-gear').textContent, channel: !$('set-channel').hidden }};
        pick({json.dumps(interface["id"])});
        const back = {{ ...{HEAR}, line: $('home-gear').textContent, channel: !$('set-channel').hidden }};
        $('start').click(); await page.settle();
        return {{ onMic, back, body: page.posts('start')[0] }};""")
    assert out["onMic"]["on"] is False and out["onMic"]["line"] == "Test Mic, room mic" and out["onMic"]["channel"] is False
    assert out["back"]["on"] is True and out["back"]["channel"] is True
    assert out["back"]["line"] == "Test Interface, channel 1"
    assert out["body"] == {"style": "", "device": interface["id"]}, "the channel is setup's to fill in, unless she changed it"


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
    assert out == {"ticked": False, "said": "off", "body": {"style": ""}}


def test_the_channel_setup_found_shows_and_is_left_to_the_bridge(tmp_path, docs):
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["channel"] = 3
    out = run_page(tmp_path, docs, """
        const seen = { line: $('home-gear').textContent, value: $('set-channel').value };
        $('start').click(); await page.settle();
        return { ...seen, body: page.posts('start')[0] };""")
    assert out == {"line": "Test Interface, channel 3", "value": "3", "body": {"style": ""}}


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


def test_the_direct_button_line_waits_in_settings_for_hear_me(tmp_path, docs):
    _saved_hear_me(docs)
    out = run_page(tmp_path, docs, "return { text: $('hear-direct').textContent, hidden: $('hear-direct').hidden, words: $('hear-words').hidden, screen: page.screen() };")
    assert out == {"text": DIRECT_LINE, "hidden": False, "words": False, "screen": "home"}


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
        const line = $('home-gear').textContent;
        $('start').click(); await page.settle();
        return { line, body: page.posts('start')[0] };""")
    assert out["line"] == "System default, room mic" and "device" not in out["body"]


def test_a_device_that_went_away_falls_back_to_the_default(tmp_path, docs):
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    fewer = home_for([d for d in DEVICES if d["name"] != "Test Mic"])
    out = run_page(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        page.home = {json.dumps(fewer)};
        $('nav-settings').click();
        await page.settle();
        return $('home-gear').textContent;""")
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
    """Everything else waits in one Details, cues and the delay included."""
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
BLEED_LINE = "Plume hears the band on this input. Press Stop and try another device in Settings."


def test_the_band_in_a_direct_input_is_said_on_the_first_view(tmp_path, docs):
    """A wrong guess at what a device is gets caught here, so it cannot sit in closed Details."""
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
    assert out["say"] == "F major (a guess) \u00b7 Midwest emo, plume's guess"
    assert out["bar"] == "12" and out["tempo"] == "115 bpm" and out["stage"] == "Bass and chords"
    assert out["notes"] == "F A C" and out["level"] == "-18 dB"
    assert out["cues"] is True and out["why"] == ""


def test_a_key_the_player_typed_is_not_called_a_guess(tmp_path, docs):
    typed = {**PLAYING, "key": "F major", "style": "midwest", "row": {**ROW, "key_guess": False, "style_guess": False}}
    assert live_page(tmp_path, docs, typed, f"return {FACTS};")["say"] == "F major \u00b7 Midwest emo, chosen"


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


@pytest.mark.parametrize("hint", [guitake.NO_COUNT, guitake.OPENING, "Nothing on channel 1. Try another channel in Settings."])
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


def test_the_take_that_just_ended_offers_to_play_again_and_that_click_is_the_consent(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps(done)};
        await page.poll();
        const seen = {{ screen: page.screen(), shown: !$('rec-again').hidden,
                        focus: page.activeElement.id, posts: {MUTATIONS} }};
        $('rec-again').click(); await page.settle();
        return {{ seen, now: page.screen(), starts: page.posts('start').length }};""")
    assert out["seen"] == {"screen": "record", "shown": True, "focus": "record", "posts": []}
    assert ">New take</button>" in (STATIC / "index.html").read_text()
    assert out["now"] == "live" and out["starts"] == 1


def test_an_old_take_on_the_shelf_does_not_offer_to_play_again(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        return { screen: page.screen(), shown: !$('rec-again').hidden };""")
    assert out == {"screen": "record", "shown": False}


def test_play_again_gives_way_to_a_take_running_from_a_terminal(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps(done)};
        await page.poll();
        page.live = {json.dumps(ELSEWHERE)};
        await page.poll();
        return {{ screen: page.screen(), shown: !$('rec-again').hidden }};""")
    assert out == {"screen": "record", "shown": False}


def test_a_double_click_on_play_again_is_one_press(tmp_path, docs):
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = live_page(tmp_path, docs, PLAYING, f"""
        page.live = {json.dumps(done)};
        await page.poll();
        $('rec-again').fire('click', {{ detail: 2 }});
        await page.settle();
        return {{ screen: page.screen(), posts: {MUTATIONS} }};""")
    assert out == {"screen": "record", "posts": []}


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
        row.update(key_guess=key_guess, style_guess=style_guess, style_chosen=not style_guess, style="midwest")
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
    # On the shelf the key and style get a line of their own, apart from the date and the tempo.
    assert out["shelf"].split("\n")[1] == "F major (a guess) \u00b7 Midwest emo, plume's guess"
    assert out["shelf"].split("\n")[2].startswith("53 bpm \u00b7 ")
    assert "F major (a guess) \u00b7 Midwest emo, plume's guess" in out["meta"]
    assert out["sentence"] == "Bar 8. Bass and chords in, in F major (a guess)."


def test_a_typed_key_and_style_are_stated_plainly_afterwards(tmp_path, docs):
    guessed = _guessed_docs(docs, tmp_path, key_guess=False, style_guess=False)
    out = run_page(tmp_path, guessed, """
        $('nav-takes').click(); await page.settle();
        page.$('shelf-' + ids.g).click(); await page.settle();
        return { meta: $('rec-meta').textContent, sentence: $('ev-sentence').textContent };""")
    assert "F major \u00b7 Midwest emo, chosen" in out["meta"] and "guess" not in out["meta"]
    assert out["sentence"] == "Bar 8. Bass and chords in, in F major."


def test_a_false_start_lands_on_settings_and_says_why(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, """
        page.live = { state: 'idle', settings: true, message: 'Nothing came in on channel 1. Try another channel.' };
        await page.poll();
        return { screen: document.body.getAttribute('data-screen'), note: $('settings-note').textContent,
                 shown: !$('settings-note').hidden, hash: page.location.hash, starts: page.posts('start').length };""")
    assert out == {"screen": "settings", "note": "Nothing came in on channel 1. Try another channel.",
                   "shown": True, "hash": "#/settings", "starts": 0}


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
        const first = !$('settings-note').hidden;
        $('set-key')._value = 'F major'; $('set-key').fire('input');
        const typed = !$('settings-note').hidden;
        refuse(); await page.settle();
        $('nav-takes').click(); await page.settle();
        page.history.go(-1); await page.settle();
        return {{ first, typed, left: !$('settings-note').hidden, screen: page.screen() }};""")
    assert out == {"first": True, "typed": False, "left": False, "screen": "settings"}


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
    assert out == {"screen": "live", "hash": "#/live", "body": [{"style": ""}]}


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
    assert out == [{"style": "", "device": mic["id"]}]


def test_a_channel_from_settings_gives_way_to_the_one_setup_just_heard(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        $('set-channel').value = '3'; $('set-channel').fire('change');
        page.replies['setup/listen'] = [200, {json.dumps({**COUNT, "channel": 2, "heard": HEARD, "others": []})}];
        $('start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{"style": "", "device": COUNT["device"]}]


def test_a_channel_from_settings_stands_when_there_was_nothing_to_listen_to(tmp_path, docs):
    interface = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    out = first_run(tmp_path, docs, """
        $('set-channel').value = '3'; $('set-channel').fire('change');
        $('start').click(); await page.settle();
        return page.posts('start');""", setup={**COUNT, "device": interface["id"]})
    assert out == [{"style": "", "device": interface["id"], "channel": 3}]


def test_a_channel_she_chose_survives_a_listen_that_heard_nothing(tmp_path, docs):
    nothing = {**setup_doc("listen", "Nothing came in.", "again", "start"),
               "heard": {"result": "nothing", "channel": None, "peak_db": None}, "others": []}
    out = first_run(tmp_path, docs, f"""
        $('set-channel').value = '3'; $('set-channel').fire('change');
        page.replies['setup/listen'] = [200, {json.dumps(nothing)}];
        $('start').click(); await page.settle();
        $('setup-start').click(); await page.settle();
        return page.posts('start')[0].channel;""")
    assert out == 3


def test_a_device_picked_in_settings_is_the_one_listened_to(tmp_path, docs):
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
    assert out["screen"] == "live" and out["hash"] == "#/live" and out["starts"] == [{"style": "", "device": COUNT["device"]}]


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
    assert out == [{"style": "", "device": COUNT["device"]}]


def test_start_anyway_is_not_held_to_a_device_nothing_was_heard_on(tmp_path, docs):
    """No device name means the bridge's own safe choice."""
    out = first_run(tmp_path, docs, f"""
        page.replies['setup/listen'] = [200, {json.dumps({**setup_doc("listen", "Nothing came in.", "again", "start"), "others": []})}];
        $('start').click(); await page.settle();
        $('setup-start').click(); await page.settle();
        return page.posts('start');""")
    assert out == [{"style": ""}]


def test_a_device_gone_from_a_finished_setup_goes_to_settings_and_says_so(tmp_path, docs):
    out = first_run(tmp_path, docs, f"""
        page.replies['start'] = [400, {{ error: {json.dumps(guitake.DEVICE_MISSING)} }}];
        $('start').click(); await page.settle();
        return {{ screen: page.screen(), note: $('settings-note').textContent }};""", setup=COUNT)
    assert out == {"screen": "settings", "note": guitake.DEVICE_MISSING}


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
    assert out == {"screen": "live", "body": [{"style": ""}]}


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
    assert out["screen"] == "live" and out["body"] == [{"style": ""}]


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


def test_set_up_again_is_one_link_in_settings_and_starts_nothing(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    assert ">Set up again</button>" in _settings_html() and html.count("Set up again") == 1
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
    assert out["body"] == [{"style": "", "device": done["device"]}], "and nothing but the device she was heard on is named"
    assert [a["step"] for a in out["answers"]] == ["headphones", "hears_self", "volume"]


# --------------------------------------------------------------------- honest words

def test_no_em_dash_in_anything_a_person_reads():
    for name in ("index.html", "app.js", "live.js", "session.js"):
        assert "\u2014" not in (STATIC / name).read_text(encoding="utf-8"), name


# --------------------------------------------------------------------- settings as a screen

ROOM_SPEAKERS = {"name": "Room Speakers", "inputs": 0, "outputs": 2}
WITH_SPEAKERS = [*DEVICES, ROOM_SPEAKERS]
SPEAKERS_ID = guitake.describe_outputs([ROOM_SPEAKERS])[0]["id"]
OUTPUT = """{ row: !$('set-output').hidden, value: $('set-output').value, labels: $('set-output').children.map((o) => o.textContent),
              note: $('output-note').hidden ? '' : $('output-note').textContent, gear: $('home-gear').textContent,
              hearRow: !$('hear-row').hidden, sendRow: !$('send-row').hidden, sendOn: !!$('send-me').checked,
              sendWords: $('send-words').hidden ? '' : $('send-words').textContent,
              room: $('room-output').hidden ? '' : $('room-output').textContent }"""
SEND_LINE = "It arrives slightly late. Only for an instrument plugged in: a mic on this channel would howl."
WIRELESS = "Wireless speakers arrive late, which makes playing along hard."


def _with_output(docs, saved=None, sends=None, name="Test Interface", devices=None):
    docs["home"] = home_for(devices or WITH_SPEAKERS)
    for device in docs["home"]["devices"]:
        device.setdefault("output", None), device.setdefault("send_channel", None), device.setdefault("output_problem", None)
        if device["name"] == name:
            device["output"], device["send_channel"] = saved, sends


def test_home_is_start_a_stone_and_a_line(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    home = html[html.index('<main id="home"'):html.index('<main id="settings"')]
    assert set(re.findall(r'\bid="([^"]+)"', home)) == {"home", "start", "home-elsewhere", "elsewhere-text", "elsewhere-stop",
                                                        "home-note", "home-gear", "home-stone", "home-style-row",
                                                        "home-style"}
    out = run_page(tmp_path, docs, "return { screen: page.screen(), gear: $('home-gear').textContent, note: $('home-note').hidden };")
    assert out == {"screen": "home", "gear": "Test Interface, channel 1", "note": True}


def test_the_settings_link_sits_next_to_takes_and_lives_in_the_address(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    assert html.index('id="nav-takes"') < html.index('id="nav-settings"') < html.index('id="nav-live"')
    out = run_page(tmp_path, docs, f"""
        $('nav-settings').click(); await page.settle();
        const there = {LOOK};
        $('settings-done').click(); await page.settle();
        const done = {LOOK};
        page.history.go(-1); await page.settle();
        return {{ there, done, back: {LOOK}, posts: page.calls.filter((c) => c.method === 'POST').length }};""")
    assert out == {"there": {"screen": "settings", "hash": "#/settings"}, "done": {"screen": "home", "hash": "#/"},
                   "back": {"screen": "settings", "hash": "#/settings"}, "posts": 0}


def test_a_reload_on_settings_stays_there(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {LOOK};", address="#/settings", stored=KEY)
    assert out == {"screen": "settings", "hash": "#/settings"}


def test_settings_asks_after_devices_plugged_in_since(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        const before = page.calls.filter((c) => c.path === '/api/home').length;
        $('nav-settings').click(); await page.settle();
        return page.calls.filter((c) => c.path === '/api/home').length - before;""")
    assert out == 1


def test_settings_is_not_there_during_a_take(tmp_path, docs):
    out = live_page(tmp_path, docs, PLAYING, f"""
        const hidden = $('nav-settings').hidden;
        page.location.hash = '#/settings';
        page.history.pushState(null, '', '#/settings');
        page.fireWindow('popstate');
        await page.settle();
        return {{ hidden, ...{LOOK} }};""")
    assert out == {"hidden": True, "screen": "live", "hash": "#/live"}


def test_settings_comes_back_when_the_take_is_over(tmp_path, docs):
    out = live_page(tmp_path, docs, WAITING, """
        page.live = { state: 'idle' };
        await page.poll();
        return $('nav-settings').hidden;""")
    assert out is False


def test_a_take_started_elsewhere_takes_settings_over(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        $('nav-settings').click(); await page.settle();
        page.live = {json.dumps(PLAYING)};
        await page.poll();
        return {LOOK};""")
    assert out == {"screen": "live", "hash": "#/live"}


def test_a_device_problem_is_said_in_settings_too(tmp_path, docs):
    docs["home"].update(devices=[], device=None, problem=guitake.NO_INPUT)
    out = run_page(tmp_path, docs, """
        $('nav-settings').click(); await page.settle();
        return $('settings-note').textContent;""")
    assert out == guitake.NO_INPUT


# ------------------------------------------------------------------------------- the output

def test_the_output_defaults_to_the_same_as_the_input(tmp_path, docs):
    _with_output(docs)
    out = run_page(tmp_path, docs, f"return {OUTPUT};")
    assert out["value"] == "" and out["labels"] == ["Same as input", "Test Speakers", "Room Speakers"]
    assert out["note"] == "" and out["hearRow"] is True and out["sendRow"] is False and out["room"] == ""
    assert out["gear"] == "Test Interface, channel 1"


def test_a_room_mic_is_not_told_its_sound_goes_to_the_mic(tmp_path, docs):
    _with_output(docs)
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    out = run_page(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        return {OUTPUT};""")
    assert out["labels"] == ["System output", "Test Speakers", "Test Interface", "Room Speakers"]


def test_choosing_an_output_is_saved_says_the_wireless_line_and_offers_the_opt_in_off(tmp_path, docs):
    _with_output(docs)
    answer = {**COUNT, "output": SPEAKERS_ID}
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps(answer)}];
        $('set-output').value = {json.dumps(SPEAKERS_ID)}; $('set-output').fire('change');
        await page.settle();
        return {{ ...{OUTPUT}, posts: page.posts('setup/answer'), starts: page.posts('start').length,
                  meters: page.posts('setup/listen').length }};""")
    device = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    assert out["posts"] == [{"step": "output", "answer": SPEAKERS_ID, "device": device["id"]}]
    assert out["starts"] == 0 and out["meters"] == 0, "choosing opens nothing"
    assert out["note"] == WIRELESS and out["gear"] == "Test Interface, channel 1, out to Room Speakers"
    assert out["sendRow"] is True and out["sendOn"] is False and out["sendWords"] == SEND_LINE
    assert out["hearRow"] is False, "a yes about headphones means nothing out there"


def test_the_opt_in_is_told_to_the_bridge_with_its_channel(tmp_path, docs):
    _with_output(docs, saved=SPEAKERS_ID)
    out = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [200, {json.dumps(COUNT)}];
        $('send-me').checked = true; $('send-me').fire('change'); await page.settle();
        return {{ ...{OUTPUT}, posts: page.posts('setup/answer') }};""")
    device = next(d for d in docs["home"]["devices"] if d["name"] == "Test Interface")
    assert out["posts"] == [{"step": "send", "answer": "on", "channel": 1, "device": device["id"]}]
    assert out["sendOn"] is True


def test_the_opt_in_shows_only_for_the_channel_it_was_given_on(tmp_path, docs):
    _with_output(docs, saved=SPEAKERS_ID, sends=1)
    out = run_page(tmp_path, docs, f"""
        const first = {OUTPUT}.sendOn;
        $('set-channel').value = '2'; $('set-channel').fire('change');
        return {{ first, then: {OUTPUT}.sendOn }};""")
    assert out == {"first": True, "then": False}


def test_a_room_mic_on_another_output_is_offered_no_way_to_hear_itself(tmp_path, docs):
    _with_output(docs, saved=SPEAKERS_ID, name="Test Mic")
    mic = next(d for d in docs["home"]["devices"] if d["name"] == "Test Mic")
    out = run_page(tmp_path, docs, f"""
        $('set-device').value = {json.dumps(mic["id"])}; $('set-device').fire('change');
        return {OUTPUT};""")
    assert out["hearRow"] is False and out["sendRow"] is False
    assert out["room"] == "Hear me stays off here: a room mic through speakers would howl."


def test_an_output_that_has_gone_is_said_and_not_swapped(tmp_path, docs):
    _with_output(docs, saved="f" * 16)
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["output_problem"] = guitake.OUTPUT_MISSING
    out = run_page(tmp_path, docs, f"return {OUTPUT};")
    assert out["value"] == "f" * 16 and out["labels"][-1] == "Not plugged in"
    assert out["note"] == guitake.OUTPUT_MISSING and out["sendRow"] is False
    assert "has gone" in out["gear"]


def test_a_refused_output_goes_to_settings_from_start(tmp_path, docs):
    _with_output(docs)
    out = run_page(tmp_path, docs, f"""
        page.replies.setup = [400, {{ error: {json.dumps(guitake.OUTPUT_MISSING)} }}];
        page.replies.start = [400, {{ error: {json.dumps(guitake.OUTPUT_MISSING)} }}];
        $('start').click(); await page.settle();
        return {{ ...{LOOK}, note: $('settings-note').textContent }};""")
    assert out == {"screen": "settings", "hash": "#/settings", "note": guitake.OUTPUT_MISSING}


def test_a_take_refused_by_the_pair_lands_on_settings_in_plain_words(tmp_path, docs):
    _with_output(docs, saved=SPEAKERS_ID)
    out = live_page(tmp_path, docs, WAITING, f"""
        page.live = {{ state: 'failed', settings: true, message: {json.dumps(guitake.OUTPUT_REFUSED)} }};
        await page.poll();
        return {{ ...{LOOK}, note: $('settings-note').textContent }};""")
    assert out == {"screen": "settings", "hash": "#/settings", "note": guitake.OUTPUT_REFUSED}


def test_a_failed_save_says_so_and_a_take_started_since_takes_over(tmp_path, docs):
    _with_output(docs)
    failed = run_page(tmp_path, docs, f"""
        $('nav-settings').click(); await page.settle();
        page.failNext['setup/answer'] = 1;
        $('set-output').value = {json.dumps(SPEAKERS_ID)}; $('set-output').fire('change');
        await page.settle();
        return {{ note: $('settings-note').textContent, value: $('set-output').value }};""")
    assert failed == {"note": "no", "value": ""}, "the bridge's word, and the choice is as it was"
    taken = run_page(tmp_path, docs, f"""
        page.replies['setup/answer'] = [409, {{ error: 'A take is already running.', live: {json.dumps(PLAYING)} }}];
        $('nav-settings').click(); await page.settle();
        $('set-output').value = {json.dumps(SPEAKERS_ID)}; $('set-output').fire('change');
        await page.settle();
        return {LOOK};""")
    assert taken == {"screen": "live", "hash": "#/live"}


def test_the_take_says_what_it_really_sends(tmp_path, docs):
    sending = {**PLAYING, "output": "Room Speakers", "hear_me": True, "stream": {"reported_ms": 24.2}}
    out = live_page(tmp_path, docs, sending, "return { delay: $('live-delay').textContent, direct: $('live-direct').hidden };")
    assert out["delay"] == "Your instrument reaches that output about 24 ms late, by your device's own count, and it could be more."
    assert out["direct"] is True, "that button only reaches the interface's own outs"


SEND_LATE = ("Your instrument reaches that output about 85 ms late, by your device's own count, and it "
             "could be more. To hear yourself on time, send the sound out through the same device as "
             "your input, or turn off Hear me and use that device's own direct monitoring.")


def test_a_long_way_round_also_says_the_two_ways_out(tmp_path, docs):
    sending = {**PLAYING, "output": "Room Speakers", "hear_me": True, "stream": {"reported_ms": 85.3}}
    out = live_page(tmp_path, docs, sending, f"return {FACTS};")
    assert out["delay"] == SEND_LATE
    assert out["hint"] == "", "the delay has its own line, never the hint's"


def test_her_own_output_and_a_take_without_hear_me_get_no_ways_out(tmp_path, docs):
    own = {**PLAYING, "hear_me": True, "stream": {"reported_ms": 85.3}}
    assert "same device" not in live_page(tmp_path, docs, own, "return $('live-delay').textContent;")
    off = {**PLAYING, "output": "Room Speakers", "hear_me": False, "stream": {"reported_ms": 85.3}}
    assert live_page(tmp_path, docs, off, "return $('live-delay').hidden;") is True


def test_a_long_way_round_never_hides_the_band_in_her_input_or_a_fault(tmp_path, docs):
    late = {"output": "Room Speakers", "hear_me": True, "stream": {"reported_ms": 85.3}}
    out = live_page(tmp_path, docs, {**BLEED, **late}, f"return {FACTS};")
    assert out["hint"] == BLEED_LINE and out["delay"] == SEND_LATE
    in_mic = {**PLAYING, **late, "row": {**ROW, "band_in_mic": {"bar": 3}}}
    out = live_page(tmp_path, docs, in_mic, f"return {FACTS};")
    assert out["hint"].startswith("Plume hears the band in this mic") and out["delay"] == SEND_LATE
    fault = "Your input is buzzing. Check the cable."
    out = live_page(tmp_path, docs, {**PLAYING, **late, "hint": fault}, f"return {FACTS};")
    assert out["hint"] == fault and out["delay"] == SEND_LATE


# ------------------------------------------------- setup moving off the input the output was for

BUILT_IN_MIC = {"name": "Laptop Mic", "inputs": 1, "outputs": 0}
TWO_BOXES = [BUILT_IN_MIC, DEVICES[2], {"name": "Other Box", "inputs": 2, "outputs": 2}, ROOM_SPEAKERS]
BOX = guitake.describe_devices([DEVICES[2]])[0]["id"]
MIC = guitake.describe_devices([BUILT_IN_MIC])[0]["id"]


def _on_the_mic(docs):
    """Two interfaces, so home shows the mic, and Room Speakers is saved on it."""
    _with_output(docs, saved=SPEAKERS_ID, name="Laptop Mic", devices=TWO_BOXES)
    assert docs["home"]["device"] == guitake.describe_devices([BUILT_IN_MIC])[0]["id"]


def _heard_on_box(step="count"):
    line = guitake.COUNT_LINE if step == "count" else "Headphones on?"
    return setup_doc(step, line, *(() if step == "count" else ("yes", "no")), device=BOX, output=None,
                     heard=HEARD, others=[])


def _press(reply):
    return f"""
        page.replies['setup/listen'] = [200, {json.dumps(reply)}];
        $('start').click(); await page.settle();"""


def test_setup_moving_to_another_input_asks_before_any_band_plays(tmp_path, docs):
    """Home said the band goes to Room Speakers.  Starting on the box without a word would break that."""
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        return {{ ...{SEEN}, starts: page.posts('start').length, saved: page.posts('setup/answer') }};""")
    assert out["starts"] == 0 and out["saved"] == []
    assert out["say"] == "Plume heard you on Test Interface."
    assert out["more"] == "Send the band to Room Speakers? If not, it comes out of Test Interface."
    assert out["choices"] == ["Yes", "No"], "short, because a long name overflowed the button on a phone"


def test_a_yes_saves_the_output_on_the_new_input_and_then_starts(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "device": BOX, "output": SPEAKERS_ID})}];
        $('setup-yes').click(); await page.settle();
        return {{ saved: page.posts('setup/answer'), starts: page.posts('start') }};""")
    assert out["saved"] == [{"step": "output", "answer": SPEAKERS_ID, "device": BOX}]
    assert out["starts"] == [{"style": "", "device": BOX}]


def test_a_no_changes_nothing_and_carries_on_with_setup(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box("headphones"))}
        $('setup-no').click(); await page.settle();
        return {{ ...{SEEN}, saved: page.posts('setup/answer'), starts: page.posts('start') }};""")
    assert out["saved"] == [] and out["starts"] == [] and out["say"] == "Headphones on?"


def test_a_yes_keeps_what_setup_had_to_say_about_the_level(tmp_path, docs):
    _on_the_mic(docs)
    low = {**_heard_on_box(), "step": "listen", "line": guitake.LOW_GAIN, "done": False,
           "choices": steps("again", "start"), "heard": {**HEARD, "result": "low"}}
    out = first_run(tmp_path, docs, f"""{_press(low)}
        const asked = $('setup-more').textContent;
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "device": BOX, "output": SPEAKERS_ID})}];
        $('setup-yes').click(); await page.settle();
        return {{ asked, say: $('setup-say').textContent, starts: page.posts('start') }};""")
    assert out["asked"].startswith("Send the band to Room Speakers?")
    assert out["starts"] == [] and out["say"] in guitake.LOW_GAIN


def _nothing(device=BOX, **more):
    return {**_heard_on_box(), "step": "listen", "line": "Nothing there.", "done": False, "device": device,
            "choices": steps("again", "start"), "heard": {**HEARD, "result": "nothing", "channel": None}, **more}


def test_start_anyway_after_hearing_nothing_starts_what_home_showed(tmp_path, docs):
    """It tried the box first and heard nothing, so nothing has moved and nobody was heard."""
    def press(home):
        home(docs)
        return first_run(tmp_path, docs, f"""{_press(_nothing())}
            const first = $('setup-say').textContent;
            $('setup-start').click(); await page.settle();
            return {{ first, asked: !!$('setup-yes'), screen: page.screen(), starts: page.posts('start') }};""")
    saved = press(_on_the_mic)
    assert saved["first"] == "Nothing there."
    assert saved["asked"] is False and saved["screen"] == "live"
    assert saved["starts"] == [{"style": ""}], "no device named, so the bridge starts what home showed"
    assert saved == press(lambda d: _with_output(d, devices=TWO_BOXES)), "the same as with no output saved"


def test_start_anyway_on_an_input_she_named_goes_there_without_saying_it_heard_her(tmp_path, docs):
    _on_the_mic(docs)
    others = [{"id": BOX, "name": "Test Interface"}]
    out = first_run(tmp_path, docs, f"""{_press(_nothing(MIC, others=others))}
        page.replies['setup/listen'] = [200, {json.dumps(_nothing())}];
        $('setup-others').children[0].click(); await page.settle();
        const said = $('setup-say').textContent;
        $('setup-start').click(); await page.settle();
        return {{ said, asked: !!$('setup-yes'), starts: page.posts('start') }};""")
    assert out == {"said": "Nothing there.", "asked": False, "starts": [{"style": "", "device": BOX}]}


def test_an_input_nobody_listened_to_this_time_is_not_called_heard(tmp_path, docs):
    """A saved setup that is ready goes straight to the band.  Nothing was heard to move off home."""
    _on_the_mic(docs)
    ready = {**COUNT, "device": BOX, "output": None}
    out = first_run(tmp_path, docs, """$('start').click(); await page.settle();
        return { asked: !!$('setup-yes'), starts: page.posts('start') };""", setup=ready)
    assert out == {"asked": False, "starts": [{"style": "", "device": BOX}]}


def test_a_no_on_a_low_level_is_not_asked_again_by_start_anyway_or_play_again(tmp_path, docs):
    _on_the_mic(docs)
    low = {**_heard_on_box(), "step": "listen", "line": guitake.LOW_GAIN, "done": False,
           "choices": steps("again", "start"), "heard": {**HEARD, "result": "low"}}
    out = first_run(tmp_path, docs, f"""{_press(low)}
        $('setup-no').click(); await page.settle();
        page.replies['setup/listen'] = [200, {json.dumps(low)}];
        $('setup-again').click(); await page.settle();
        const again = !!$('setup-yes');
        $('setup-start').click(); await page.settle();
        return {{ again, asked: !!$('setup-yes'), starts: page.posts('start'), saved: page.posts('setup/answer') }};""")
    assert out == {"again": False, "asked": False, "starts": [{"style": ""}], "saved": []}


def test_a_no_is_remembered_on_the_next_press_of_new_take(tmp_path, docs):
    """After a No the bridge's home says the box, with nothing saved on it.  The page has to read that."""
    _on_the_mic(docs)
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        const asked = !!$('setup-yes');
        page.home = {{ ...page.home, device: {json.dumps(BOX)} }};
        const before = page.calls.filter((c) => c.path === '/api/home').length;
        $('setup-no').click(); await page.settle();
        const reads = page.calls.filter((c) => c.path === '/api/home').length - before;
        const seen = [];
        for (let take = 0; take < 2; take++) {{
          page.live = {json.dumps(done)};
          await page.poll();
          seen.push(page.screen());
          page.replies['setup/listen'] = [200, {json.dumps(_heard_on_box())}];
          $('rec-again').click(); await page.settle();
          seen.push(!!$('setup-yes'));
        }}
        return {{ asked, reads, seen, starts: page.posts('start'), saved: page.posts('setup/answer') }};""")
    assert out["asked"] is True and out["reads"] == 1
    assert out["seen"] == ["record", False, "record", False], "one answer, not one question per take"
    assert out["starts"] == [{"style": "", "device": BOX}] * 3 and out["saved"] == []


def test_a_yes_reads_home_again_only_once_the_output_is_saved(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "device": BOX, "output": SPEAKERS_ID})}];
        const reads = () => page.calls.filter((c) => c.path === '/api/home').length;
        const before = reads(), open = page.hold('setup/answer');
        $('setup-yes').click(); await page.settle();
        const during = reads() - before;
        open(); await page.settle();
        return {{ during, after: reads() - before, starts: page.posts('start') }};""")
    assert out == {"during": 0, "after": 1, "starts": [{"style": "", "device": BOX}]}


def test_a_yes_that_failed_reads_nothing_again(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        const before = page.calls.filter((c) => c.path === '/api/home').length;
        page.failNext['setup/answer'] = 1;
        $('setup-yes').click(); await page.settle();
        return {{ reads: page.calls.filter((c) => c.path === '/api/home').length - before, starts: page.posts('start').length }};""")
    assert out == {"reads": 0, "starts": 0}


def test_a_yes_that_failed_keeps_the_question_so_nothing_starts_without_her_answer(tmp_path, docs):
    """Start anyway on a trouble screen would start on the box she was heard on, out of the box."""
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        page.replies['setup/answer'] = [409, {{ error: 'Plume could not list the devices.' }}];
        $('setup-yes').click(); await page.settle();
        const failed = {SEEN};
        page.replies['setup/answer'] = [200, {json.dumps({**COUNT, "device": BOX, "output": SPEAKERS_ID})}];
        const before = page.posts('start').length;
        $('setup-yes').click(); await page.settle();
        return {{ failed, before, saved: page.posts('setup/answer'), starts: page.posts('start') }};""")
    assert out["failed"]["choices"] == ["Yes", "No"] and out["failed"]["note"] == "Plume could not list the devices."
    assert out["failed"]["more"].startswith("Send the band to Room Speakers?")
    assert out["before"] == 0
    assert out["saved"] == [{"step": "output", "answer": SPEAKERS_ID, "device": BOX}] * 2
    assert out["starts"] == [{"style": "", "device": BOX}]


def test_a_yes_that_never_arrived_can_still_be_a_no(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        page.down = true;
        $('setup-yes').click(); await page.settle();
        const note = $('setup-note').textContent;
        page.down = false;
        $('setup-no').click(); await page.settle();
        return {{ note, saved: page.posts('setup/answer').length, starts: page.posts('start') }};""")
    assert out["note"] == "That did not reach plume gui. Try again."
    assert out["starts"] == [{"style": "", "device": BOX}]


def _saved_on_box(docs):
    """Two interfaces, so home shows the mic with no output, and Room Speakers is saved on the box."""
    _with_output(docs, saved=SPEAKERS_ID, devices=TWO_BOXES)
    assert docs["home"]["device"] == MIC and docs["home"]["output"] is None


def _heard_with_speakers(**more):
    return {**_heard_on_box(), "output": SPEAKERS_ID, **more}


def test_an_output_saved_on_the_input_it_moved_to_is_said_before_any_band_plays(tmp_path, docs):
    """Home named no output, so the band going to the box's saved one would be news."""
    _saved_on_box(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_with_speakers())}
        return {{ ...{SEEN}, starts: page.posts('start').length }};""")
    assert out["starts"] == 0
    assert out["say"] == "Plume heard you on Test Interface."
    assert out["more"] == "Send the band to Room Speakers? If not, it comes out of Test Interface."
    assert out["choices"] == ["Yes", "No"]


def test_a_yes_to_an_output_already_saved_saves_nothing_and_starts(tmp_path, docs):
    _saved_on_box(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_with_speakers())}
        $('setup-yes').click(); await page.settle();
        return {{ saved: page.posts('setup/answer'), starts: page.posts('start') }};""")
    assert out == {"saved": [], "starts": [{"style": "", "device": BOX}]}


def test_a_no_to_an_output_already_saved_clears_it_and_asks_what_her_own_output_needs(tmp_path, docs):
    """Off the saved output the next questions are about her own, so the bridge's answer is followed."""
    _saved_on_box(docs)
    reply = _heard_on_box("headphones")
    out = first_run(tmp_path, docs, f"""{_press(_heard_with_speakers())}
        page.replies['setup/answer'] = [200, {json.dumps(reply)}];
        $('setup-no').click(); await page.settle();
        return {{ ...{SEEN}, saved: page.posts('setup/answer'), starts: page.posts('start') }};""")
    assert out["saved"] == [{"step": "output", "answer": "same", "device": BOX}]
    assert out["starts"] == [] and out["say"] == "Headphones on?"


def test_a_no_that_cleared_the_output_is_not_asked_again_on_a_low_level(tmp_path, docs):
    _saved_on_box(docs)
    low = _heard_with_speakers(step="listen", line=guitake.LOW_GAIN, done=False, choices=steps("again", "start"),
                               heard={**HEARD, "result": "low"})
    cleared = {**low, "output": None}
    out = first_run(tmp_path, docs, f"""{_press(low)}
        page.replies['setup/answer'] = [200, {json.dumps(_heard_on_box("headphones"))}];
        $('setup-no').click(); await page.settle();
        const say = $('setup-say').textContent;
        page.replies['setup/listen'] = [200, {json.dumps(cleared)}];
        $('setup-again').click(); await page.settle();
        return {{ say, asked: !!$('setup-yes'), saved: page.posts('setup/answer').length }};""")
    assert out["say"] in guitake.LOW_GAIN and out["asked"] is False and out["saved"] == 1


def test_a_no_that_failed_to_clear_keeps_the_question(tmp_path, docs):
    _saved_on_box(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_with_speakers())}
        page.failNext['setup/answer'] = 1;
        $('setup-no').click(); await page.settle();
        return {{ asked: !!$('setup-yes'), starts: page.posts('start').length }};""")
    assert out == {"asked": True, "starts": 0}


def test_an_answer_on_a_saved_output_is_not_asked_again_on_new_take(tmp_path, docs):
    _saved_on_box(docs)
    done = {"state": "done", "take": docs["ids"]["tap"], "note": None}
    out = first_run(tmp_path, docs, f"""{_press(_heard_with_speakers())}
        page.home = {{ ...page.home, device: {json.dumps(BOX)}, output: {json.dumps(SPEAKERS_ID)} }};
        $('setup-yes').click(); await page.settle();
        page.live = {json.dumps(done)};
        await page.poll();
        page.replies['setup/listen'] = [200, {json.dumps(_heard_with_speakers())}];
        $('rec-again').click(); await page.settle();
        return {{ asked: !!$('setup-yes'), starts: page.posts('start') }};""")
    assert out == {"asked": False, "starts": [{"style": "", "device": BOX}] * 2}


def test_no_question_when_the_saved_output_is_where_the_input_home_showed_plays(tmp_path, docs):
    """Home's interface plays out of itself, and the box's saved output is that same interface."""
    other = {"name": "Other Box", "inputs": 2, "outputs": 2}
    other_out = guitake.describe_outputs([other])[0]["id"]
    other_in = guitake.describe_devices([other])[0]["id"]
    _with_output(docs, saved=other_out, devices=TWO_BOXES)
    docs["home"]["device"] = other_in
    out = first_run(tmp_path, docs, f"{_press(_heard_with_speakers(output=other_out))} return page.posts('start');")
    assert out == [{"style": "", "device": BOX}]


def test_nothing_heard_on_an_input_with_its_own_output_still_starts_what_home_showed(tmp_path, docs):
    _saved_on_box(docs)
    out = first_run(tmp_path, docs, f"""{_press(_nothing(output=SPEAKERS_ID))}
        $('setup-start').click(); await page.settle();
        return {{ asked: !!$('setup-yes'), starts: page.posts('start') }};""")
    assert out == {"asked": False, "starts": [{"style": ""}]}


def test_no_question_when_the_band_goes_where_home_said(tmp_path, docs):
    _on_the_mic(docs)
    agrees = {**_heard_on_box(), "output": SPEAKERS_ID}
    out = first_run(tmp_path, docs, f"{_press(agrees)} return page.posts('start');")
    assert out == [{"style": "", "device": BOX}]


def test_no_question_when_the_input_it_moved_to_is_that_output(tmp_path, docs):
    box_out = guitake.describe_outputs([DEVICES[2]])[0]["id"]
    _with_output(docs, saved=box_out, name="Laptop Mic", devices=TWO_BOXES)
    out = first_run(tmp_path, docs, f"{_press(_heard_on_box())} return page.posts('start');")
    assert out == [{"style": "", "device": BOX}]


def test_no_question_with_no_output_chosen(tmp_path, docs):
    _with_output(docs, devices=TWO_BOXES)
    out = first_run(tmp_path, docs, f"{_press(_heard_on_box())} return page.posts('start');")
    assert out == [{"style": "", "device": BOX}]


def test_an_output_gone_by_the_answer_goes_to_settings_and_starts_nothing(tmp_path, docs):
    _on_the_mic(docs)
    out = first_run(tmp_path, docs, f"""{_press(_heard_on_box())}
        page.replies['setup/answer'] = [400, {{ error: {json.dumps(guitake.OUTPUT_MISSING)} }}];
        $('setup-yes').click(); await page.settle();
        return {{ ...{LOOK}, note: $('settings-note').textContent, shown: !$('settings-note').hidden, starts: page.posts('start').length }};""")
    assert out["screen"] == "settings" and out["starts"] == 0
    assert out["note"] == guitake.OUTPUT_MISSING and out["shown"] is True


# ------------------------------------------------------------- saying a missing output only once

def test_an_output_that_has_gone_is_said_once_and_not_sent_to_where_she_already_is(tmp_path, docs):
    _with_output(docs, saved="f" * 16)
    for device in docs["home"]["devices"]:
        if device["name"] == "Test Interface":
            device["output_problem"] = guitake.OUTPUT_MISSING
    out = run_page(tmp_path, docs, f"""
        page.replies.setup = [400, {{ error: {json.dumps(guitake.OUTPUT_MISSING)} }}];
        page.replies.start = [400, {{ error: {json.dumps(guitake.OUTPUT_MISSING)} }}];
        $('start').click(); await page.settle();
        return {{ ...{OUTPUT}, screen: page.screen(), top: !$('settings-note').hidden }};""")
    assert out["screen"] == "settings"
    assert out["top"] is False, "the same red sentence twice"
    assert out["note"] == guitake.OUTPUT_MISSING
    assert "in Settings" not in guitake.OUTPUT_MISSING


# ------------------------------------------------------------------- the header on a narrow phone

def test_the_header_cannot_push_a_phone_page_sideways():
    """The stub has no layout, so this reads the rules.  Measured in headless Chrome at 320 too."""
    css = (STATIC / "plume.css").read_text()
    assert "flex-wrap: wrap" in re.search(r"(?m)^header \{([^}]*)\}", css)[1]
    assert "white-space: nowrap" in re.search(r"(?m)^\.navlink \{([^}]*)\}", css)[1]
    assert "margin-left: auto" in re.search(r"(?m)^\.seg \{([^}]*)\}", css)[1], "wrapped, the light switch stays on the right"


# ---------------------------------------------------------------------- the guide

GUIDE = """{ ...""" + LOOK + """, title: ($('guide-body').children.find((c) => c.tag === 'h1') || {}).textContent,
  tabs: $('guide-pages').children.map((c) => c.textContent),
  current: $('guide-pages').children.filter((c) => c.getAttribute('aria-current') === 'page').map((c) => c.textContent),
  next: $('guide-next').textContent, asked: page.calls.filter((c) => c.path === '/api/guide').length }"""


def test_the_guide_sits_next_to_settings_and_opens_on_its_first_page(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    assert html.index('id="nav-settings"') < html.index('id="nav-guide"') < html.index('id="nav-live"')
    out = run_page(tmp_path, docs, f"$('nav-guide').click(); await page.settle(); return {GUIDE};")
    assert out["screen"] == "guide" and out["hash"] == "#/guide"
    assert out["title"] == "Getting started" and out["current"] == ["Getting started"]
    assert out["tabs"] == ["Getting started", "Playing with plume", "Settings and sound",
                           "From a terminal or a coding agent", "Troubleshooting"]
    assert out["next"] == "Next: Playing with plume"


def test_the_guide_asks_for_no_takes_and_starts_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        $('nav-guide').click(); await page.settle();
        $('guide-next').children[0].click(); await page.settle();
        return page.calls.map((c) => c.method + ' ' + c.path.split('?')[0]);""")
    assert not [c for c in out if c.startswith("POST")]
    assert not [c for c in out if "/api/take" in c]
    assert out.count("GET /api/guide") == 1, "the words are fetched once"


def test_a_link_inside_a_page_moves_to_that_page_and_back_comes_back(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = [];
        $('nav-guide').click(); await page.settle();
        const link = $('guide-body').walk().find((e) => e.tag === 'a' && e.getAttribute('href') === '#/guide/sound');
        link.click(); await page.settle(); seen.push({GUIDE});
        page.history.go(-1); await page.settle(); seen.push({GUIDE});
        return seen;""")
    assert out[0]["hash"] == "#/guide/sound" and out[0]["title"] == "Settings and sound"
    assert out[1]["hash"] == "#/guide" and out[1]["title"] == "Getting started"


def test_a_reload_of_a_guide_page_stays_on_it(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {GUIDE};", address="#/guide/terminal", stored=KEY)
    assert out["screen"] == "guide" and out["title"] == "From a terminal or a coding agent"
    assert out["next"] == "Next: Troubleshooting"


def test_an_unknown_guide_page_shows_the_first_one(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {GUIDE};", address="#/guide/nope", stored=KEY)
    assert out["screen"] == "guide" and out["title"] == "Getting started"


def test_the_last_guide_page_offers_no_next(tmp_path, docs):
    out = run_page(tmp_path, docs, f"return {GUIDE};", address="#/guide/troubleshooting", stored=KEY)
    assert out["title"] == "Troubleshooting" and out["next"] == ""


def test_settings_has_one_quiet_way_to_the_terminal_page(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        $('nav-settings').click(); await page.settle();
        $('settings-terminal').click(); await page.settle();
        return {GUIDE};""")
    assert out["hash"] == "#/guide/terminal" and out["title"] == "From a terminal or a coding agent"


def test_the_guide_can_be_read_during_a_take_and_leaves_it_running(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        $('nav-guide').click(); await page.settle();
        return {{ ...{GUIDE}, guide: !$('nav-guide').hidden, back: !$('nav-live').hidden,
                 stops: page.posts('stop').length }};""", live=PLAYING, address="", stored=KEY)
    assert out["screen"] == "guide" and out["guide"] and out["back"] and out["stops"] == 0


def test_a_guide_that_cannot_load_says_so_in_words(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        page.failNext['guide'] = 1;
        $('nav-guide').click(); await page.settle();
        return { screen: page.screen(), text: $('guide-body').textContent };""")
    assert out["screen"] == "guide" and "did not load" in out["text"]


# ------------------------------------------------------------------ the playhead

#: Where the line is, and what the player was told.
LINE = ("{ x: parseFloat((/translateX\\(([-\\d.]+)px\\)/.exec($('playhead').style.transform || '') || [0, 'NaN'])[1]),"
        " hidden: $('playhead').hidden, at: $('player').currentTime, frames: page.frames.length,"
        " plays: $('player').plays, src: $('player').src }")
#: A recording whose length the browser knows, playing.
LOADED = """
    await page.open(ids.tap);
    $('listen-' + (typeof track === 'undefined' ? 'human' : track)).click();
    $('player').duration = 400;
    $('player').fire('loadedmetadata');
    await page.settle();
"""
DOWN = "{ clientX: %s, button: 0, isPrimary: true, preventDefault() {} }"


def _tap(docs, ahead=0.0):
    take = docs["docs"][docs["ids"]["tap"]]
    take["band_ahead_s"] = ahead
    return take


def _moment(take, share):
    points = take["band"]
    return points[0]["t_s"] + share * (points[-1]["t_s"] - points[0]["t_s"])


def _x(share):
    # The stub's sheet is 1280 wide with 46 of margin each side.
    return 46 + share * (1280 - 92)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_the_line_sits_on_the_moment_that_is_playing(tmp_path, docs, share):
    t = _moment(_tap(docs), share)
    out = run_page(tmp_path, docs, LOADED + f"$('player').currentTime = {t}; page.frame(); return {LINE};")
    assert out["hidden"] is False
    assert out["x"] == pytest.approx(_x(share), abs=0.01)


def test_the_line_moves_on_every_frame_while_it_plays_and_rests_when_it_stops(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        const seen = [page.frames.length];
        $('player').currentTime = {_moment(take, 0.25)}; page.frame();
        const first = {LINE};
        $('player').currentTime = {_moment(take, 0.26)}; page.frame();
        const second = {LINE};
        $('listen-play').click(); page.frame(); page.frame();
        return {{ seen, first, second, after: page.frames.length }};""")
    assert out["seen"] == [1], "one frame asked for, however many things redraw the player"
    assert out["first"]["x"] == pytest.approx(_x(0.25), abs=0.01) and out["first"]["frames"] == 1
    assert out["second"]["x"] == pytest.approx(_x(0.26), abs=0.01)
    assert out["after"] == 0, "nothing runs while nothing plays"


@pytest.mark.parametrize("track, ahead", [("human", 0.0), ("machine", 0.5)])
def test_the_line_knows_the_band_was_recorded_ahead_of_the_player(tmp_path, docs, track, ahead):
    take = _tap(docs, ahead=0.5)
    span = take["band"][-1]["t_s"] - take["band"][0]["t_s"]
    out = run_page(tmp_path, docs, f"const track = '{track}';" + LOADED
                   + f"$('player').currentTime = {_moment(take, 0.5)}; page.frame(); return {LINE};")
    assert out["x"] == pytest.approx(_x(0.5 + ahead / span), abs=0.01)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_a_click_on_the_chart_moves_the_sound_to_that_moment(tmp_path, docs, share):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {DOWN % _x(share)});
        page.fireWindow('pointerup', {{}});
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, share), abs=0.001)
    assert out["x"] == pytest.approx(_x(share), abs=0.01)


def test_a_click_on_the_chart_while_the_band_plays_lands_on_the_same_moment(tmp_path, docs):
    take = _tap(docs, ahead=0.5)
    out = run_page(tmp_path, docs, "const track = 'machine';" + LOADED + f"""
        $('sheet').fire('pointerdown', {DOWN % _x(0.5)});
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, 0.5) - 0.5, abs=0.001)
    assert out["x"] == pytest.approx(_x(0.5), abs=0.01)


def test_the_line_can_be_dragged_and_lets_go(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {DOWN % _x(0.2)});
        page.fireWindow('pointermove', {{ clientX: {_x(0.7)} }});
        const held = {LINE};
        page.fireWindow('pointerup', {{}});
        page.fireWindow('pointermove', {{ clientX: {_x(0.1)} }});
        return {{ held, after: {LINE} }};""")
    assert out["held"]["at"] == pytest.approx(_moment(take, 0.7), abs=0.001)
    assert out["after"]["at"] == pytest.approx(_moment(take, 0.7), abs=0.001), "a pointer that was let go moves nothing"


def test_a_pointer_that_never_pressed_the_chart_moves_nothing(tmp_path, docs):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"page.fireWindow('pointermove', {{ clientX: 600 }}); return {LINE};")
    assert out["at"] == 0


def test_only_the_main_button_moves_the_sound(tmp_path, docs):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {{ clientX: 600, button: 2, isPrimary: true, preventDefault() {{}} }});
        return {LINE};""")
    assert out["at"] == 0


def test_left_of_the_first_bar_is_the_start_of_the_recording(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').currentTime = {_moment(take, 0.5)};
        $('sheet').fire('pointerdown', {DOWN % 10});
        return {LINE};""")
    assert out["at"] == 0 and out["x"] == pytest.approx(_x(0.0), abs=0.01)


def test_before_any_sound_a_drag_is_safe_and_play_starts_from_it(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('sheet').fire('pointerdown', {DOWN % _x(0.5)});
        page.fireWindow('pointermove', {{ clientX: {_x(0.6)} }});
        page.fireWindow('pointerup', {{}});
        const before = {LINE};
        $('listen-play').click();
        const unknown = {LINE};
        $('sheet').fire('pointerdown', {DOWN % _x(0.4)});
        page.fireWindow('pointerup', {{}});
        $('player').duration = 400;
        $('player').fire('loadedmetadata');
        return {{ before, unknown, after: {LINE} }};""")
    assert out["before"]["plays"] == 0 and out["before"]["src"] == "", "a drag is not a press on Play"
    assert out["before"]["x"] == pytest.approx(_x(0.6), abs=0.01)
    assert out["unknown"]["plays"] == 1 and out["unknown"]["x"] == pytest.approx(_x(0.6), abs=0.01)
    assert out["after"]["at"] == pytest.approx(_moment(take, 0.4), abs=0.001)
    assert out["after"]["x"] == pytest.approx(_x(0.4), abs=0.01)


def test_another_recording_is_heard_from_the_same_moment(tmp_path, docs):
    take = _tap(docs, ahead=0.5)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').currentTime = {_moment(take, 0.5)};
        $('listen-machine').click();
        $('player').duration = 400;
        $('player').fire('loadedmetadata');
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, 0.5) - 0.5, abs=0.001)
    assert out["x"] == pytest.approx(_x(0.5), abs=0.01)


def test_leaving_the_take_puts_the_line_back_and_stops_asking_for_frames(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').currentTime = {_moment(take, 0.5)}; page.frame();
        await page.shelf(); page.frame(); page.frame();
        const away = page.frames.length;
        await page.open(ids.other);
        return {{ away, ...{LINE} }};""")
    assert out["away"] == 0
    assert out["x"] == pytest.approx(_x(0.0), abs=0.01) and out["at"] == 0


def test_a_take_with_no_recording_has_no_line_and_a_press_does_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.lead);
        $('sheet').fire('pointerdown', {DOWN % 600});
        page.fireWindow('pointermove', {{ clientX: 700 }});
        page.fireWindow('pointerup', {{}});
        return {LINE};""")
    assert out["hidden"] is True and out["plays"] == 0 and out["src"] == "" and out["frames"] == 0


def test_the_line_adds_nothing_to_press_or_read_and_takes_its_colour_from_the_theme():
    html = (STATIC / "index.html").read_text()
    assert re.search(r'<div id="playhead" aria-hidden="true"( hidden)?></div>', html)
    css = (STATIC / "plume.css").read_text()
    rules = re.findall(r"^#playhead[^{]*\{[^}]*\}", css, flags=re.M)
    assert rules and any("var(--now)" in rule for rule in rules)
    assert not any(re.search(r"#[0-9a-fA-F]{3,8}\b|rgb|hsl", rule.split("{", 1)[1]) for rule in rules)
    # A press on a moment still reaches the moment: the line itself is never in the way.
    assert any("pointer-events: none" in rule for rule in rules)


# ------------------------------------------------- the playhead: only the hand that took it

def _ev(x, pointer=1, **more):
    """A pointer event, as a browser shapes one."""
    fields = {"clientX": x, "clientY": 200, "button": 0, "buttons": 1, "isPrimary": True, "pointerId": pointer,
              "pointerType": "mouse", **more}
    return json.dumps(fields).replace("}", ", preventDefault() {} }")


def test_a_second_finger_does_not_move_the_sound(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.2))});
        $('sheet').fire('pointerdown', {_ev(_x(0.9), pointer=2, isPrimary=False)});
        page.fireWindow('pointermove', {_ev(_x(0.8), pointer=2, isPrimary=False)});
        const other = {LINE};
        page.fireWindow('pointerup', {_ev(_x(0.8), pointer=2, isPrimary=False, buttons=0)});
        page.fireWindow('pointermove', {_ev(_x(0.4))});
        return {{ other, own: {LINE} }};""")
    assert out["other"]["at"] == pytest.approx(_moment(take, 0.2), abs=0.001)
    assert out["own"]["at"] == pytest.approx(_moment(take, 0.4), abs=0.001), "another finger lifting is not this one letting go"


def test_a_press_that_is_not_the_first_pointer_moves_nothing(tmp_path, docs):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.5), pointer=2, isPrimary=False)});
        page.fireWindow('pointermove', {_ev(_x(0.6), pointer=2, isPrimary=False)});
        return {LINE};""")
    assert out["at"] == 0


@pytest.mark.parametrize("lost", [
    "page.fireWindow('pointercancel', %s);" % _ev(0, buttons=0),
    "page.fireWindow('blur', {});",
    "page.fireWindow('contextmenu', {});"])
def test_a_drag_whose_release_never_comes_still_ends(tmp_path, docs, lost):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.2))});
        {lost}
        page.fireWindow('pointermove', {_ev(_x(0.7))});
        return {{ scrub: $('sheet').classList.contains('scrub'), ...{LINE} }};""")
    assert out["at"] == pytest.approx(_moment(take, 0.2), abs=0.001)
    assert out["scrub"] is False


def test_a_move_with_no_button_held_is_a_hover_and_ends_the_drag(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.2))});
        page.fireWindow('pointermove', {_ev(_x(0.7), buttons=0)});
        page.fireWindow('pointermove', {_ev(_x(0.8))});
        return {{ scrub: $('sheet').classList.contains('scrub'), ...{LINE} }};""")
    assert out["at"] == pytest.approx(_moment(take, 0.2), abs=0.001)
    assert out["scrub"] is False


def test_leaving_the_take_in_the_middle_of_a_drag_lets_go(tmp_path, docs):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.2))});
        await page.shelf();
        await page.open(ids.tap);
        page.fireWindow('pointermove', {_ev(_x(0.7))});
        return {{ scrub: $('sheet').classList.contains('scrub'), ...{LINE} }};""")
    assert out["scrub"] is False
    assert out["x"] == pytest.approx(_x(0.0), abs=0.01)


def test_a_press_is_measured_from_the_charts_own_left_edge(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, f"""
        page.rects.sheet = {{ left: 30, top: 0, width: 1280, height: 360, right: 1310, bottom: 360 }};
        {LOADED}
        $('sheet').fire('pointerdown', {_ev(30 + _x(0.5))});
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, 0.5), abs=0.001)


def test_the_line_keeps_its_moment_when_the_window_changes_width(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').currentTime = {_moment(take, 0.5)}; page.frame();
        $('listen-play').click(); page.frame();
        page.rects.sheet = {{ left: 0, top: 0, width: 1000, height: 360, right: 1000, bottom: 360 }};
        page.fireWindow('resize');
        await new Promise((r) => setTimeout(r, 250));
        return {LINE};""")
    assert out["x"] == pytest.approx(46 + 0.5 * (1000 - 92), abs=0.01)


# ------------------------------------------------- the playhead: a finger that is scrolling

def test_a_finger_going_down_the_page_does_not_move_the_sound(tmp_path, docs):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.5), pointerType="touch")});
        const down = {LINE};
        page.fireWindow('pointermove', {_ev(_x(0.5) + 3, pointerType="touch", clientY=230)});
        page.fireWindow('pointermove', {_ev(_x(0.6), pointerType="touch", clientY=300)});
        page.fireWindow('pointerup', {_ev(_x(0.6), pointerType="touch", clientY=300, buttons=0)});
        return {{ down, after: {LINE} }};""")
    assert out["down"]["at"] == 0, "a finger landing is not yet a press"
    assert out["after"]["at"] == 0


def test_a_finger_going_along_the_chart_drags_the_line(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.5), pointerType="touch")});
        page.fireWindow('pointermove', {_ev(_x(0.5) + 20, pointerType="touch", clientY=203)});
        page.fireWindow('pointermove', {_ev(_x(0.7), pointerType="touch", clientY=260)});
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, 0.7), abs=0.001), "once it is a drag it stays one"


def test_a_tap_moves_the_sound_when_the_finger_lifts(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('sheet').fire('pointerdown', {_ev(_x(0.5), pointerType="touch")});
        page.fireWindow('pointerup', {_ev(_x(0.5) + 1, pointerType="touch", buttons=0)});
        return {LINE};""")
    assert out["at"] == pytest.approx(_moment(take, 0.5) + 1 * (take["band"][-1]["t_s"] - take["band"][0]["t_s"]) / (1280 - 92), abs=0.001)


def test_the_chart_leaves_scrolling_down_the_page_to_the_browser():
    css = (STATIC / "plume.css").read_text()
    rule = re.search(r"^#sheet \{[^}]*\}", css, flags=re.M).group(0)
    assert "touch-action: pan-y" in rule


# ------------------------------------------------- the playhead: edges

def test_changing_what_you_hear_at_the_start_stays_at_the_start(tmp_path, docs):
    _tap(docs, ahead=0.5)
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-machine').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        $('listen-human').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        const first = $('player').currentTime;
        $('player').currentTime = 400; $('player').fire('ended');
        $('listen-machine').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        $('listen-human').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        return {{ first, afterEnd: $('player').currentTime }};""")
    assert out == {"first": 0, "afterEnd": 0}


@pytest.mark.parametrize("band", [
    [],
    [{"bar": 0, "t_s": 3.0, "level": 0.5}],
    [{"bar": 0, "t_s": 3.0, "level": 0.5}, {"bar": 1, "t_s": 3.0, "level": 0.5}]])
def test_a_chart_with_no_span_of_time_has_no_line_and_moves_nothing(tmp_path, docs, band):
    take = _tap(docs)
    take["band"] = band
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').currentTime = 3.5; page.frame();
        const playing = {LINE};
        $('sheet').fire('pointerdown', {_ev(640)});
        return {{ playing, after: {LINE} }};""")
    assert out["playing"]["hidden"] is True
    assert out["after"]["at"] == 3.5 and out["after"]["hidden"] is True


def test_the_far_end_of_the_chart_is_not_the_end_of_the_recording(tmp_path, docs):
    """Sent to its very end a browser calls the recording over and goes back to the start."""
    take = _tap(docs)
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-human').click();
        $('player').duration = {_moment(take, 0.9)};
        $('player').fire('loadedmetadata');
        $('sheet').fire('pointerdown', {_ev(_x(1.0))});
        return {{ at: $('player').currentTime, length: $('player').duration }};""")
    assert out["length"] - 0.2 < out["at"] < out["length"]


@pytest.mark.parametrize("how", ["pause", "error"])
def test_a_stop_the_page_did_not_ask_for_still_stops_the_frames(tmp_path, docs, how):
    _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').paused = true;
        $('player').fire('{how}');
        page.frame(); page.frame();
        return {{ frames: page.frames.length, word: $('listen-play').textContent }};""")
    assert out == {"frames": 0, "word": "Play"}


def test_a_start_the_page_did_not_ask_for_starts_the_frames_again(tmp_path, docs):
    take = _tap(docs)
    out = run_page(tmp_path, docs, LOADED + f"""
        $('player').paused = true; $('player').fire('pause'); page.frame();
        $('player').paused = false; $('player').fire('play');
        const frames = page.frames.length;
        $('player').currentTime = {_moment(take, 0.5)}; page.frame();
        return {{ frames, word: $('listen-play').textContent, ...{LINE} }};""")
    assert out["frames"] == 1 and out["word"] == "Pause"
    assert out["x"] == pytest.approx(_x(0.5), abs=0.01)


# ------------------------------------------------- the playhead: what the bridge says

def test_the_bridge_tells_the_page_how_far_ahead_the_band_was_recorded(tmp_path, docs):
    take = docs["docs"][docs["ids"]["other"]]
    assert take["band_ahead_s"] == 0.25
    points = take["band"]
    span = points[-1]["t_s"] - points[0]["t_s"]
    mid = points[0]["t_s"] + span / 2
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.other);
        $('listen-machine').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        $('player').currentTime = {mid}; page.frame();
        return {LINE};""")
    assert out["x"] == pytest.approx(_x(0.5 + 0.25 / span), abs=0.01)


def test_a_take_that_cannot_say_how_far_ahead_the_band_was_draws_both_on_one_clock(tmp_path, docs):
    take = docs["docs"][docs["ids"]["tap"]]
    assert take["band_ahead_s"] is None
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('listen-machine').click();
        $('player').duration = 400; $('player').fire('loadedmetadata');
        $('player').currentTime = {_moment(take, 0.5)}; page.frame();
        return {LINE};""")
    assert out["x"] == pytest.approx(_x(0.5), abs=0.01)


# ---------------------------------------------------------------------- files

def test_show_files_names_the_take_and_nothing_else(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.replies.reveal = [200, { shown: true, mix: true }];
        $('show-files').click();
        await page.settle();
        return { posts: page.posts('reveal'), key: page.calls.find((c) => c.path === '/api/reveal').key,
                 quiet: $('keep-error').hidden, off: $('show-files').disabled,
                 note: [$('files-note').hidden, $('files-note').textContent], ...""" + SOUND + """ };""")
    assert out["posts"] == [{"take": docs["ids"]["tap"]}] and out["key"] == KEY
    assert out["quiet"] is True and out["off"] is False and out["note"] == [True, ""]
    assert out["plays"] == 0 and out["audio"] == 0


def test_a_folder_that_will_not_open_says_where_it_is(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.failNext.reveal = 1;
        $('show-files').click();
        await page.settle();
        return { hidden: $('keep-error').hidden, words: $('keep-error').textContent };""")
    where = docs["docs"][docs["ids"]["tap"]]["where"]
    assert out["hidden"] is False and out["words"] == "The folder did not open. It is " + where


def test_show_files_is_drawn_from_the_pages_own_colours():
    rules = re.findall(r"#show-files[^{]*\{([^}]*)\}", (STATIC / "plume.css").read_text())
    assert rules and all("#" not in rule and "rgb" not in rule for rule in rules)
    assert "var(--secondary)" in rules[0]


@pytest.mark.parametrize("answer, words", [
    ({"shown": True, "mix": False, "band_twice": False},
     "No mix was saved for this take. Your recordings are in the folder."),
    ({"shown": True, "mix": True, "band_twice": True},
     "Saved as mix.wav. Your mic may have heard the band too, so the band could be in it twice."),
    # An answer that does not say a mix was saved is not read as one.
    ({"shown": True}, "No mix was saved for this take. Your recordings are in the folder."),
    ({"shown": True, "mix": False, "band_twice": False, "why": "theirs"},
     "This folder already has a mix.wav that plume can't replace. Move or rename it to get a new one."),
    ({"shown": True, "mix": False, "band_twice": False, "why": "stale"},
     "The mix.wav in this folder is an older one, and a new one could not be saved."),
    ({"shown": True, "mix": False, "band_twice": False, "why": "in_use"},
     "mix.wav is open in another program, so it could not be updated. Close it there and try again."),
])
def test_show_files_says_one_line_when_the_mix_is_not_plain(tmp_path, docs, answer, words):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.replies.reveal = [200, """ + json.dumps(answer) + """];
        $('show-files').click();
        await page.settle();
        const here = [$('files-note').hidden, $('files-note').textContent];
        await page.shelf();
        await page.open(ids.other);
        const elsewhere = [$('files-note').hidden, $('files-note').textContent];
        page.replies.reveal = [200, { shown: true, mix: true, band_twice: false }];
        page.history.go(-2);
        await page.settle();
        const back = $('files-note').textContent;
        $('show-files').click();
        await page.settle();
        return { here, elsewhere, back, after: [$('files-note').hidden, $('files-note').textContent],
                 problem: $('keep-error').hidden };""")
    assert out["here"] == [False, words] and out["problem"] is True
    assert out["elsewhere"] == [True, ""], "the line is about one take"
    assert out["back"] == words
    assert out["after"] == [True, ""], "a plain mix clears it"


def test_a_new_click_takes_the_last_line_away_before_its_answer_comes(tmp_path, docs):
    out = run_page(tmp_path, docs, """
        await page.open(ids.tap);
        page.replies.reveal = [200, { shown: true, mix: false, band_twice: false, why: null }];
        $('show-files').click();
        await page.settle();
        const first = [$('files-note').hidden, $('files-note').textContent];
        const open = page.hold('reveal');
        $('show-files').click();
        await page.settle();
        const waiting = [$('files-note').hidden, $('files-note').textContent, $('show-files').disabled];
        page.replies.reveal = [200, { shown: true, mix: true, band_twice: false, why: null }];
        open();
        await page.settle();
        return { first, waiting, after: [$('files-note').hidden, $('show-files').disabled] };""")
    assert out["first"][0] is False and out["first"][1].startswith("No mix")
    assert out["waiting"] == [True, "", True], "an old line must not sit beside a new click"
    assert out["after"] == [True, False]


OVERLOAD_HINT = ("Your input is overloading. Turn its gain down a little. "
                 "If it is only pops or bumps, check your cable.")


def test_the_band_in_her_input_is_said_over_a_standing_overload(tmp_path, docs):
    """An overload can stand for a whole take, and a howl reads as one: Stop matters more."""
    late = {"hint": OVERLOAD_HINT}
    bleed = {**BLEED, **late, "row": {**BLEED["row"], "input_fault": "overload"}}
    assert live_page(tmp_path, docs, bleed, f"return {FACTS};")["hint"] == BLEED_LINE
    in_mic = {**PLAYING, **late, "row": {**ROW, "band_in_mic": {"bar": 3}, "input_fault": "overload"}}
    assert live_page(tmp_path, docs, in_mic, f"return {FACTS};")["hint"].startswith(
        "Plume hears the band in this mic")
    alone = {**PLAYING, **late, "row": {**ROW, "input_fault": "overload"}}
    assert live_page(tmp_path, docs, alone, f"return {FACTS};")["hint"] == OVERLOAD_HINT


def test_a_buzz_is_still_said_over_the_band_in_her_input(tmp_path, docs):
    buzz = "Your input is buzzing. Check the cable."
    doc = {**BLEED, "hint": buzz, "row": {**BLEED["row"], "input_fault": "buzz"}}
    assert live_page(tmp_path, docs, doc, f"return {FACTS};")["hint"] == buzz


@pytest.mark.parametrize("ms,shown,advice", [(30.4, 30, False), (30.6, 31, True), (45.0, 45, True),
                                             (29.6, 30, False)])
def test_the_ways_out_go_by_the_number_she_is_shown(tmp_path, docs, ms, shown, advice):
    sending = {**PLAYING, "output": "Room Speakers", "hear_me": True, "stream": {"reported_ms": ms}}
    line = live_page(tmp_path, docs, sending, "return $('live-delay').textContent;")
    assert f"about {shown} ms late" in line
    assert ("same device" in line) is advice


# ------------------------------------------------------------- the band through a DAW

MIDI = """{ section: !$('midi-more').hidden, open: !!$('midi-more').open, row: !$('midi-row').hidden,
  out: !!$('midi-out').checked, onlyRow: !$('midi-only-row').hidden, only: !!$('midi-only').checked,
  words: $('midi-words').textContent, gear: $('home-gear').textContent }"""


def _midi(docs, choice="off", problem=None):
    docs["home"]["midi"] = {"choice": choice, "problem": problem}


def test_the_daw_choice_is_tucked_away_in_settings_and_off():
    html = (STATIC / "index.html").read_text()
    first = html[html.index('<main id="home"'):html.index('<main id="settings"')]
    assert "midi" not in first.lower() and "daw" not in first.lower()
    settings = _settings_html()
    # Inside a closed details: depth, not something to fill in.
    folded = settings[settings.index('<details id="midi-more"'):]
    folded = folded[:folded.index("</details>")]
    assert 'id="midi-out"' in folded and 'id="midi-only"' in folded
    assert "checked" not in folded and " open" not in folded


def test_the_daw_choice_starts_off_and_start_says_nothing_about_it(tmp_path, docs):
    _midi(docs)
    out = run_page(tmp_path, docs, f"""
        const seen = {MIDI};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0], said: page.posts('midi') }};""")
    assert out["section"] and not out["open"] and out["row"]
    assert out["out"] is False and out["onlyRow"] is False and out["only"] is False
    assert "MIDI port called plume" in out["words"]
    assert out["body"] == {"style": ""} and out["said"] == []
    assert "DAW" not in out["gear"]


def test_ticking_the_daw_choice_saves_it_and_offers_muting(tmp_path, docs):
    _midi(docs)
    out = run_page(tmp_path, docs, f"""
        $('midi-out').checked = true; $('midi-out').fire('change'); await page.settle();
        const ticked = {MIDI};
        $('midi-only').checked = true; $('midi-only').fire('change'); await page.settle();
        const muted = {MIDI};
        $('midi-out').checked = false; $('midi-out').fire('change'); await page.settle();
        $('start').click(); await page.settle();
        return {{ ticked, muted, off: {MIDI}, said: page.posts('midi'), body: page.posts('start')[0] }};""")
    assert out["said"] == [{"midi": "out"}, {"midi": "only"}, {"midi": "off"}]
    assert out["ticked"]["out"] and out["ticked"]["onlyRow"] and not out["ticked"]["only"]
    assert out["ticked"]["gear"].endswith(", band through a DAW")
    assert out["muted"]["out"] and out["muted"]["only"]
    assert out["muted"]["gear"].endswith(", band only through a DAW")
    assert not out["off"]["out"] and not out["off"]["onlyRow"] and "DAW" not in out["off"]["gear"]
    assert out["body"] == {"style": ""}


def test_a_saved_daw_choice_shows_ticked(tmp_path, docs):
    _midi(docs, "only")
    out = run_page(tmp_path, docs, f"return {MIDI};")
    assert out["out"] and out["onlyRow"] and out["only"]
    assert out["gear"].endswith(", band only through a DAW")


def test_without_the_midi_part_the_choice_is_explained_not_offered(tmp_path, docs):
    _midi(docs, "only", guitake.MIDI_PART_MISSING)
    out = run_page(tmp_path, docs, f"return {MIDI};")
    assert out["section"] and not out["row"] and not out["onlyRow"]
    assert out["words"] == guitake.MIDI_PART_MISSING
    assert "DAW" not in out["gear"]


def test_a_refused_daw_choice_is_said_and_unticked(tmp_path, docs):
    _midi(docs)
    out = run_page(tmp_path, docs, f"""
        page.replies.midi = [409, {{ error: {json.dumps(guitake.MIDI_NO_PORT)} }}];
        $('midi-out').checked = true; $('midi-out').fire('change'); await page.settle();
        return {{ ...{MIDI}, note: $('midi-note').textContent, shown: !$('midi-note').hidden,
                  disabled: !!$('midi-out').disabled }};""")
    assert out["out"] is False and out["onlyRow"] is False and out["disabled"] is False
    # The boxes give way to the reason, said once, as if the page had known all along.
    assert out["row"] is False and out["words"] == guitake.MIDI_NO_PORT and out["shown"] is False


# ---------------------------------------------------------------- what she plays

INSTRUMENT = """{ value: $('set-instrument').value, labels: $('set-instrument').children.map((o) => o.textContent),
  values: $('set-instrument').children.map((o) => o.value), disabled: !!$('set-instrument').disabled }"""


def test_what_she_plays_is_in_settings_and_never_beside_start():
    html = (STATIC / "index.html").read_text()
    home = html[html.index('<main id="home"'):html.index('<main id="settings"')]
    assert "set-instrument" not in home
    assert 'id="set-instrument"' in _settings_html()


def test_the_instrument_starts_not_set_and_start_sends_nothing_new(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        const seen = {INSTRUMENT};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0], saved: page.posts('instrument') }};""")
    assert out["value"] == "" and out["labels"][0] == "Not set"
    assert out["values"][1:] == [c["id"] for c in docs["home"]["instruments"]]
    assert out["labels"][1:] == [c["label"] for c in docs["home"]["instruments"]]
    assert out["body"] == {"style": ""} and out["saved"] == []


def test_a_remembered_instrument_is_shown_and_start_still_says_only_the_style(tmp_path, docs):
    docs["home"]["instrument"] = "bass"
    out = run_page(tmp_path, docs, f"""
        const seen = {INSTRUMENT};
        $('start').click(); await page.settle();
        return {{ ...seen, body: page.posts('start')[0] }};""")
    assert out["value"] == "bass" and out["body"] == {"style": ""}


def test_choosing_an_instrument_is_saved_at_once_and_not_set_too(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        $('set-instrument').value = 'keys'; $('set-instrument').fire('change'); await page.settle();
        const mid = $('set-instrument').value;
        $('set-instrument').value = ''; $('set-instrument').fire('change'); await page.settle();
        return {{ ...{INSTRUMENT}, mid, saved: page.posts('instrument') }};""")
    assert out["saved"] == [{"instrument": "keys"}, {"instrument": ""}]
    assert out["mid"] == "keys" and out["value"] == "" and out["disabled"] is False


def test_a_refused_instrument_is_said_and_the_choice_goes_back(tmp_path, docs):
    docs["home"]["instrument"] = "lead"
    out = run_page(tmp_path, docs, f"""
        page.replies.instrument = [400, {{ error: 'That is not something the instrument choice understands.' }}];
        $('set-instrument').value = 'kazoo'; $('set-instrument').fire('change'); await page.settle();
        return {{ ...{INSTRUMENT}, note: $('settings-note').textContent, shown: !$('settings-note').hidden }};""")
    assert out["value"] == "lead" and out["disabled"] is False
    assert out["note"] and out["shown"] is True


def test_an_instrument_that_did_not_reach_the_bridge_is_said_and_put_back(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        page.failNext.instrument = 1;
        $('set-instrument').value = 'bass'; $('set-instrument').fire('change'); await page.settle();
        return {{ ...{INSTRUMENT}, note: $('settings-note').textContent }};""")
    assert out["value"] == "" and out["note"]
