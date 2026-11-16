"""The take page's video control, run under Node against the real page script.

No browser, no sound, no file: the "video" is a made-up object the page must hand
over untouched, and the bridge is the stub's stand-in, which a test tells what to say.
"""
from __future__ import annotations

import json

import pytest

from plume import gui
from test_gui_page import KEY, NODE, STATIC, docs, run_page  # noqa: F401 - docs is a fixture

pytestmark = pytest.mark.skipif(NODE is None, reason="needs node")

IDLE = {"state": "idle", "can": True, "ready": True, "have": False, "theirs": False, "max_bytes": 1000}
#: What the page shows of the control, in one object.
VIEW = """({ shown: !$('video').hidden, say: $('video-say').hidden ? '' : $('video-say').textContent,
  warn: $('video-say').classList.contains('warn'), add: !$('video-add').hidden && !$('video-add').disabled,
  sound: !$('video-sound-row').hidden, bar: !$('video-bar').hidden, busy: $('video-bar').classList.contains('busy'),
  fill: $('video-bar-fill').style.transform || '', nudge: !$('video-nudge').hidden, listen: !$('video-listen').hidden,
  show: !$('video-show').hidden, stop: !$('video-stop').hidden, plays: $('video-player').plays,
  uploads: page.calls.filter((c) => c.path.startsWith('/api/video/') && c.file).length })"""
#: Choosing a file, the way the file box does it.  The name is one no path should ever hold.
CHOOSE = """const film = { name: '../../evil name.mp4', size: 500, type: 'video/mp4' };
        $('video-file').files = [film]; $('video-file').fire('change'); await page.settle();"""


def video(**doc):
    return "page.video = " + json.dumps({**IDLE, **doc}) + ";"


def test_the_take_page_has_one_control_and_one_switch_in_plain_words(tmp_path, docs):
    html = (STATIC / "index.html").read_text()
    assert '<button id="video-add" type="button">Add a video</button>' in html
    assert "<input id=\"video-sound\" type=\"checkbox\" checked> Keep the video's sound</label>" in html
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        return {{ view: {VIEW}, on: $('video-sound').checked,
                 asked: page.calls.filter((c) => c.path.endsWith('/video')).map((c) => [c.method, c.key]) }};""")
    view = out["view"]
    assert view["shown"] and view["add"] and view["sound"] and out["on"] is True
    # Everything else waits until it is needed.
    assert not any(view[k] for k in ("say", "bar", "nudge", "show", "stop", "plays", "uploads"))
    assert out["asked"] == [["GET", KEY]]


def test_a_take_that_cannot_have_a_video_shows_no_control(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        {video(can=False)}
        await page.open(ids.tap);
        return {VIEW};""")
    assert out["shown"] is False


def test_choosing_a_video_hands_the_file_over_and_never_its_name(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "listening", "progress": None})}];
        {CHOOSE}
        const call = page.calls.find((c) => c.file);
        return {{ path: call.path, method: call.method, same: call.file === film, type: call.type, key: call.key,
                 named: page.calls.some((c) => JSON.stringify([c.path, c.body]).includes('evil')),
                 box: $('video-file').value, view: {VIEW} }};""")
    assert out["method"] == "POST" and out["path"] == f"/api/video/{docs['ids']['tap']}?sound=1"
    assert out["same"] is True and out["type"] == "application/octet-stream" and out["key"] == KEY
    assert out["named"] is False and out["box"] == ""
    view = out["view"]
    assert view["say"] == "Finding where you start playing." and view["bar"] and view["busy"] and view["stop"]
    assert not view["add"] and not view["nudge"] and not view["warn"]


def test_the_switch_off_says_so_and_is_remembered(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('video-sound').checked = false; $('video-sound').fire('change');
        {CHOOSE}
        return {{ path: page.calls.find((c) => c.file).path, kept: page.local.getItem('plume-video-sound') }};""")
    assert out["path"].endswith("?sound=0") and out["kept"] == "off"


def test_the_switch_starts_as_it_was_left(tmp_path, docs):
    out = run_page(tmp_path, docs, "return { on: $('video-sound').checked };", )
    assert out["on"] is True
    # Read as the page starts, from where the change above was kept.
    assert "$('video-sound').checked = load('plume-video-sound') !== 'off';" in (STATIC / "app.js").read_text()


def test_progress_is_gentle_and_ends_with_an_offer_to_show_the_file(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        const seen = [];
        const open = page.hold('video/upload');
        {CHOOSE}
        seen.push({VIEW});                                  // on its way, nothing heard back yet
        {video(state="receiving", progress=0.5)}
        await page.poll();
        seen.push({VIEW});
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "listening", "progress": None})}];
        open(); await page.settle();
        {video(state="making", progress=0.25)}
        await page.poll();
        seen.push({VIEW});
        {video(state="done", have=True, file="video.mp4", start_s=3.0)}
        await page.poll();
        seen.push({VIEW});
        $('video-show').click(); await page.settle();
        return {{ seen, shows: page.posts('video/show'), later: page.later.size }};""")
    first, receiving, making, done = out["seen"]
    assert first["say"] == "Bringing your video in." and first["fill"] == "scaleX(0)" and first["stop"]
    assert receiving["say"] == "Bringing your video in." and receiving["fill"] == "scaleX(0.5)"
    assert making["say"] == "Laying the band in." and making["fill"] == "scaleX(0.25)" and not making["busy"]
    assert done["say"] == "Your video is ready, with the band in it. It's saved as video.mp4 in this take's folder."
    assert done["show"] and done["add"] and not done["bar"] and not done["stop"] and not done["nudge"]
    assert out["shows"] == [{"take": docs["ids"]["tap"]}]
    assert all(v["uploads"] == 1 and v["plays"] == 0 for v in out["seen"])


def test_an_idle_answer_while_the_file_is_still_on_its_way_changes_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.hold('video/upload');
        {CHOOSE}
        await page.poll();                                  // the bridge has not heard of it yet
        return {VIEW};""")
    assert out["say"] == "Bringing your video in." and out["stop"] and not out["add"]


def test_the_nudge_shows_only_when_plume_is_unsure(tmp_path, docs):
    unsure = {**IDLE, "state": "unsure", "why": "unsure", "start_s": 3.25, "min_s": -1.75, "max_s": 8.25,
              "step_s": 0.01, "listen": True, "confidence": 0.1}
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps(unsure)}];
        {CHOOSE}
        const slide = $('video-slide');
        const first = {{ view: {VIEW}, range: [slide.min, slide.max, slide.step, slide.value], at: $('video-at').textContent }};
        slide.value = '4.5'; slide.fire('input');
        const moved = $('video-at').textContent;
        $('video-listen').click(); await page.settle();
        const heard = {{ src: $('video-player').src, plays: $('video-player').plays, other: $('player').plays }};
        page.replies['video/place'] = [202, {json.dumps({**IDLE, "state": "making", "progress": 0.0})}];
        $('video-save').click(); await page.settle();
        return {{ first, moved, heard, place: page.posts('video/place'), after: {VIEW},
                 paused: $('video-player').pauses, src: $('video-player').src }};""")
    view = out["first"]["view"]
    assert view["say"] == ("Plume isn't sure where you start playing. Listen, and slide until you and the band "
                           "sit together.")
    assert view["nudge"] and view["listen"] and view["stop"] and view["sound"] and not view["add"] and not view["bar"]
    assert view["plays"] == 0, "no sound until Listen is clicked"
    assert out["first"]["range"] == ["-1.75", "8.25", "0.01", "3.25"] and out["first"]["at"] == "0:03.25"
    assert out["moved"] == "0:04.50"
    tid = docs["ids"]["tap"]
    assert out["heard"] == {"src": f"/api/take/{tid}/video/preview?at=4.500&k={KEY}", "plays": 1, "other": 0}
    assert out["place"] == [{"take": tid, "start_s": 4.5, "sound": True}]
    assert out["after"]["say"] == "Laying the band in." and not out["after"]["nudge"]
    assert out["paused"] >= 1 and out["src"] == ""


def test_a_video_with_no_sound_says_so_kindly_and_offers_the_nudge_alone(tmp_path, docs):
    silent = {**IDLE, "state": "unsure", "why": "silent", "start_s": 0.0, "min_s": -8.0, "max_s": 20.0,
              "step_s": 0.05, "listen": False, "confidence": 0.0}
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps(silent)}];
        {CHOOSE}
        $('video-slide').value = '-2'; $('video-slide').fire('input');
        return {{ view: {VIEW}, at: $('video-at').textContent }};""")
    assert out["view"]["say"] == ("This video has no sound to line up with. Slide to the moment you start "
                                  "playing, then save it and watch.")
    assert out["view"]["nudge"] and not out["view"]["listen"]
    assert out["at"] == "2.00 s before the video starts"


@pytest.mark.parametrize("why,words", [
    ("not_video", "Plume couldn't read that as a video. Try another file?"),
    ("no_room", "There isn't enough room on this disk for the video."),
    ("in_use", "video.mp4 is open in another program. Close it there, then add your video again."),
    ("busy", "Plume is still working on a video for this take."),
    ("too_long", "That video is too long for plume to bring in."),
    ("failed", "That didn't work, and your take is untouched. Try again?"),
    ("something new", "That didn't work, and your take is untouched. Try again?"),
])
def test_what_went_wrong_is_said_in_one_plain_line(tmp_path, docs, why, words):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [409, {json.dumps({"error": "x", "why": why})}];
        {CHOOSE}
        return {VIEW};""")
    assert out["say"] == words and out["warn"] and out["add"] and not out["nudge"] and not out["bar"]


def test_a_failure_found_later_is_said_too(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "listening"})}];
        {CHOOSE}
        {video(state="failed", why="not_video")}
        await page.poll();
        return {VIEW};""")
    assert out["say"] == "Plume couldn't read that as a video. Try another file?" and out["warn"] and out["add"]


def test_without_ffmpeg_the_control_says_why_in_one_line_and_sends_nothing(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        {video(ready=False)}
        await page.open(ids.tap);
        const before = {VIEW};
        {CHOOSE}
        return {{ before, after: {VIEW} }};""")
    assert out["before"]["say"] == ("Adding a video needs ffmpeg, and plume couldn't find it. Install ffmpeg, "
                                    "then open plume again.")
    assert out["before"]["shown"] and not out["before"]["add"] and out["after"]["uploads"] == 0


def test_her_own_video_in_the_folder_is_said_before_anything_is_sent(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        {video(theirs=True)}
        await page.open(ids.tap);
        const before = {VIEW};
        {CHOOSE}
        return {{ before, after: {VIEW} }};""")
    assert out["before"]["say"] == "" and out["before"]["add"]
    assert out["after"]["say"] == ("This take's folder already has a video.mp4 that plume didn't make. Move or "
                                   "rename it, then add your video again.")
    assert out["after"]["uploads"] == 0 and out["after"]["warn"]


def test_a_video_too_big_is_said_before_anything_is_sent(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        $('video-file').files = [{{ name: 'big.mp4', size: 1001 }}]; $('video-file').fire('change'); await page.settle();
        return {VIEW};""")
    assert out["say"] == "That video is too big for plume to bring in." and out["uploads"] == 0


def test_a_lost_session_is_said_as_everywhere_else(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [401, {{ error: 'no session' }}];
        {CHOOSE}
        return {VIEW};""")
    assert out["say"].startswith("This page's link has stopped working.")


def test_cancel_stops_the_work(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "listening"})}];
        {CHOOSE}
        $('video-stop').click(); await page.settle();
        return {{ view: {VIEW}, cancels: page.posts('video/cancel') }};""")
    assert out["cancels"] == [{"take": docs["ids"]["tap"]}]
    assert out["view"]["add"] and not out["view"]["stop"] and not out["view"]["bar"] and out["view"]["say"] == ""


def test_a_second_video_cannot_start_while_one_is_under_way(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.hold('video/upload');
        {CHOOSE}
        {{ {CHOOSE} }}
        return {VIEW};""")
    assert out["uploads"] == 1


def test_dropping_a_video_on_the_take_starts_it_and_never_replaces_the_page(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        let stopped = 0;
        const film = {{ name: 'x.mp4', size: 10 }};
        const ev = {{ dataTransfer: {{ types: ['Files'], files: [film] }}, preventDefault() {{ stopped += 1; }} }};
        page.fireWindow('dragover', ev);
        const over = {VIEW};
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "listening"})}];
        page.fireWindow('drop', ev); await page.settle();
        const call = page.calls.find((c) => c.file);
        return {{ over, stopped, same: call && call.file === film, after: {VIEW} }};""")
    assert out["over"]["say"] == "Drop your video anywhere here."
    assert out["stopped"] == 2 and out["same"] is True and out["after"]["say"] == "Finding where you start playing."


def test_a_drop_that_is_not_a_file_or_not_on_a_take_is_left_alone(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        let stopped = 0;
        const words = {{ dataTransfer: {{ types: ['text/plain'], files: [] }}, preventDefault() {{ stopped += 1; }} }};
        const film = {{ dataTransfer: {{ types: ['Files'], files: [{{ size: 1 }}] }}, preventDefault() {{ stopped += 1; }} }};
        page.fireWindow('drop', film);                      // on the first screen
        await page.open(ids.tap);
        page.fireWindow('drop', words); await page.settle();
        return {{ stopped, view: {VIEW} }};""")
    assert out["stopped"] == 0 and out["view"]["uploads"] == 0


def test_closing_the_page_gives_back_a_video_that_was_waiting_for_a_nudge(tmp_path, docs):
    unsure = {**IDLE, "state": "unsure", "why": "unsure", "start_s": 1.0, "min_s": -4.0, "max_s": 6.0,
              "step_s": 0.01, "listen": True}
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.fireWindow('pagehide');
        const none = page.posts('video/cancel').length;
        page.replies['video/upload'] = [202, {json.dumps(unsure)}];
        {CHOOSE}
        page.fireWindow('pagehide'); await page.settle();
        const call = page.calls.find((c) => c.path === '/api/video/cancel');
        return {{ none, body: call.body, keepalive: call.keepalive }};""")
    assert out == {"none": 0, "body": {"take": docs["ids"]["tap"]}, "keepalive": True}


def test_leaving_the_take_stops_a_preview(tmp_path, docs):
    unsure = {**IDLE, "state": "unsure", "why": "unsure", "start_s": 1.0, "min_s": -4.0, "max_s": 6.0,
              "step_s": 0.01, "listen": True}
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps(unsure)}];
        {CHOOSE}
        $('video-listen').click(); await page.settle();
        const before = $('video-player').pauses;
        await page.shelf();
        return {{ more: $('video-player').pauses > before, src: $('video-player').src }};""")
    assert out == {"more": True, "src": ""}


def test_another_take_does_not_show_this_takes_video(tmp_path, docs):
    out = run_page(tmp_path, docs, f"""
        await page.open(ids.tap);
        page.replies['video/upload'] = [202, {json.dumps({**IDLE, "state": "done", "have": True})}];
        {CHOOSE}
        const here = {VIEW};
        await page.shelf();
        await page.open(ids.other);
        return {{ here, there: {VIEW} }};""")
    assert out["here"]["say"].startswith("Your video is ready") and out["there"]["say"] == ""


def test_no_maker_or_device_is_named_and_every_word_comes_from_one_place():
    words = (STATIC / "video.js").read_text()
    assert "video.js" in gui.STATIC_FILES
    page = (STATIC / "app.js").read_text()
    # The page script holds no sentence of its own about videos: they are all in video.js.
    assert "Bringing your video" not in page and "isn't sure" not in page
    assert "innerHTML" not in words and "innerHTML" not in page
