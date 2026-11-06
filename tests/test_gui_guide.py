"""The guide: the words the page shows under Guide, how they are served, and whether they are true.

The guide is shipped Markdown in the small subset ``gui_static/guide.js`` renders.  Every
command and flag it names is checked against plume's own parser, and every cue it names is
checked against the cue parser, so the guide cannot drift from the code without a red test.
No audio device, no sound, no take.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from plume import cli, gui
from plume.mind import parse_cue
from test_gui_bridge import Client

GUIDE = Path(gui.__file__).with_name("guide")
STATIC = Path(gui.__file__).with_name("gui_static")
NODE = shutil.which("node")
PAGES = {name: (GUIDE / f"{name}.md").read_text(encoding="utf-8") for name in gui.GUIDE_PAGES}
ALL = "\n".join(PAGES.values())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    srv = gui.GuiServer(roots=[tmp_path], log=lambda line: None).start()
    try:
        yield Client(srv).login()
    finally:
        srv.close()


# ---------------------------------------------------------------- serving it

def test_the_guide_is_served_in_reading_order(client):
    resp, out = client.get("/api/guide")
    assert resp.status == 200 and resp.getheader("Content-Type").startswith("application/json")
    pages = json.loads(out)["pages"]
    assert [p["id"] for p in pages] == list(gui.GUIDE_PAGES)
    assert [p["text"] for p in pages] == list(PAGES.values())


def test_the_guide_needs_the_page_key(client):
    assert client.get("/api/guide", headers={"X-Plume-Token": None})[0].status == 401
    assert client.get("/api/guide", headers={"X-Plume-Token": "0" * 64})[0].status == 401


@pytest.mark.parametrize("path", ["/api/guide/terminal", "/api/guide/../gui.py", "/api/guide?page=../../x",
                                  "/static/guide/terminal.md", "/static/../guide/terminal.md"])
def test_no_guide_file_can_be_named_by_a_client(client, path):
    resp, out = client.get(path)
    if resp.status == 200:      # the query string is ignored: it is still the whole guide
        assert [p["id"] for p in json.loads(out)["pages"]] == list(gui.GUIDE_PAGES)
    else:
        assert resp.status == 404


def test_every_page_ships_and_nothing_else_is_in_the_folder():
    assert {p.name for p in GUIDE.iterdir()} == {f"{name}.md" for name in gui.GUIDE_PAGES}
    text = (Path(gui.__file__).parents[2] / "pyproject.toml").read_text()
    assert '"guide/*.md"' in text


# ---------------------------------------------------------------- its form

def test_every_page_starts_with_its_title_and_has_only_one():
    for name, text in PAGES.items():
        assert text.startswith("# "), name
        assert len(re.findall(r"^# ", text, flags=re.M)) == 1, name


def test_the_guide_uses_only_what_the_renderer_understands():
    for name, text in PAGES.items():
        assert not re.search(r"<[a-zA-Z/!]", text), f"{name}: HTML is shown as text, not run"
        assert not re.search(r"^\s*\|", text, flags=re.M), f"{name}: tables are not rendered"
        assert not re.search(r"^#{4,} ", text, flags=re.M), f"{name}: only three heading levels"
        assert not re.search(r"^\s*>", text, flags=re.M), f"{name}: no quotes"
        assert not re.search(r"!\[", text), f"{name}: no images"


def test_every_link_goes_to_a_real_page():
    for name, text in PAGES.items():
        for target in re.findall(r"\]\(([^)]*)\)", text):
            found = re.fullmatch(r"#/guide/([a-z0-9-]+)", target)
            assert found and found.group(1) in gui.GUIDE_PAGES, f"{name}: {target}"


def test_the_guide_has_no_em_dashes_or_private_paths():
    assert "—" not in ALL and "–" not in ALL
    assert not re.search(r"/Users/|/home/[a-z]", ALL)


# ---------------------------------------------------------------- is it true

def _commands():
    """Every ``plume ...`` line in a code block, and every inline ``plume ...`` span."""
    found = []
    for block in re.findall(r"```\n(.*?)```", ALL, flags=re.S):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("plume "):
                found.append(line)
    found += [span for span in re.findall(r"`([^`\n]+)`", ALL) if span.startswith("plume ")]
    return found


def test_there_are_commands_to_check():
    assert len(_commands()) >= 12


@pytest.mark.parametrize("command", _commands())
def test_every_command_in_the_guide_parses(command):
    argv = shlex.split(command.replace('"..."', '"busier"'))[1:]
    try:
        args = cli._build_parser().parse_args(argv)
    except SystemExit as exit_:
        pytest.fail(f"plume does not take {command!r} ({exit_.code})")
    assert callable(args.func)


def _all_flags() -> set[str]:
    flags = set()

    def walk(parser):
        for action in parser._actions:
            flags.update(s for s in action.option_strings if s.startswith("--"))
            for sub in getattr(action, "choices", None) or {}:
                if isinstance(action.choices, dict):
                    walk(action.choices[sub])
    walk(cli._build_parser())
    return flags


def test_every_flag_the_guide_names_exists():
    named = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", ALL))
    assert named, "nothing to check"
    assert named <= _all_flags(), sorted(named - _all_flags())


#: What the terminal page says each cue does, against what the cue parser makes of it.
LIVE_CUES = {
    "busier": ("texture", "busier"), "sparser": ("texture", "sparser"),
    "fill it up": ("texture", "busier"), "denser": ("texture", "busier"),
    "strip it back": ("texture", "sparser"), "thin it out": ("texture", "sparser"),
    "simplify": ("texture", "sparser"),
    "faster": ("tempo", "faster"), "slower": ("tempo", "slower"), "half time": ("tempo", "half"),
    "double time": ("tempo", "double"), "ritard": ("tempo", "ritard"),
    "next section": ("section", "next"), "previous section": ("section", "prev"),
    "go to the chorus": ("section", "chorus"),
    "louder": ("energy", "up"), "bigger": ("energy", "up"), "lift it": ("energy", "up"),
    "quieter": ("energy", "down"), "calmer": ("energy", "down"), "bring it down": ("energy", "down"),
    "take it out": ("end", "out"), "wrap it up": ("end", "out"), "last chorus": ("end", "out"),
    "stop": ("end", "out"),
    "darker": ("texture", "darker"), "brighter": ("texture", "brighter"),
    "warmer": ("texture", "warmer"), "dry": ("texture", "dry"), "wet": ("texture", "wet"),
}

@pytest.mark.parametrize("cue", sorted(LIVE_CUES))
def test_each_cue_the_guide_names_means_what_it_says(cue):
    assert f"`{cue}`" in PAGES["terminal"]
    parsed = parse_cue(cue)
    assert (parsed["kind"], parsed["value"]) == LIVE_CUES[cue]


def test_the_live_take_really_changes_only_what_the_guide_says():
    """Live lead hands cues to the same accumulator ``simlead`` uses, and reads back its
    density, its tempo and its tone, which the band's tone stage applies."""
    source = (Path(gui.__file__).with_name("realtime.py")).read_text()
    assert "lead_cues.density_scale" in source and "lead_cues.bar_ratio()" in source
    assert "lead_cues.tilt" in source and "BusTone" in source


def test_the_page_offers_the_same_two_cues_the_guide_describes():
    from plume import guitake
    assert guitake.CUES == ("busier", "sparser")
    html = (STATIC / "index.html").read_text()
    assert ">Busier<" in html and ">Sparser<" in html and "**Busier**" in PAGES["playing"]


def test_where_takes_go_is_where_the_page_puts_them():
    assert gui.default_library() == Path.home() / "Music" / "Plume"
    assert ALL.count("`~/Music/Plume`") >= 2


def test_the_setup_lines_quoted_are_the_ones_plume_says():
    from plume import guitake
    assert guitake.PLAY_SOMETHING in PAGES["getting-started"]
    for words in ("Headphones on?", "Can you hear yourself?"):
        assert words in PAGES["getting-started"] and words in Path(guitake.__file__).read_text()


# ---------------------------------------------------------------- the renderer

pytestmark_node = pytest.mark.skipif(NODE is None, reason="needs node")


def parse(tmp_path, text):
    shutil.copy(STATIC / "guide.js", tmp_path / "guide.mjs")
    (tmp_path / "in.json").write_text(json.dumps(text))
    (tmp_path / "run.mjs").write_text(
        "import { readFileSync } from 'node:fs';\n"
        "import { parseGuide, pageTitle } from './guide.mjs';\n"
        "const text = JSON.parse(readFileSync(new URL('./in.json', import.meta.url), 'utf8'));\n"
        "const blocks = parseGuide(text);\n"
        "console.log(JSON.stringify({ blocks, title: pageTitle(blocks) }));\n")
    done = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True,
                          timeout=60, check=True)
    return json.loads(done.stdout)


@pytestmark_node
def test_the_subset_parses_into_plain_blocks(tmp_path):
    out = parse(tmp_path, "# Title\n\nOne line\nand its wrap.\n\n## Part\n\n- a `code` item\n  wrapped\n"
                          "- **bold** and *it*\n\n1. first\n2. [two](#/guide/sound)\n\n```\nplume cue \"x\"\n```\n")
    blocks = out["blocks"]
    assert out["title"] == "Title"
    assert [b["t"] for b in blocks] == ["h", "p", "h", "ul", "ol", "pre"]
    assert blocks[1]["inl"] == [{"t": "text", "s": "One line and its wrap."}]
    assert blocks[2]["level"] == 2
    assert blocks[3]["items"][0] == [{"t": "text", "s": "a "}, {"t": "code", "s": "code"},
                                     {"t": "text", "s": " item wrapped"}]
    assert blocks[3]["items"][1] == [{"t": "b", "c": [{"t": "text", "s": "bold"}]}, {"t": "text", "s": " and "},
                                     {"t": "i", "c": [{"t": "text", "s": "it"}]}]
    assert blocks[4]["items"][1] == [{"t": "a", "href": "#/guide/sound", "c": [{"t": "text", "s": "two"}]}]
    assert blocks[5] == {"t": "pre", "s": 'plume cue "x"'}


@pytestmark_node
def test_html_is_only_ever_words(tmp_path):
    out = parse(tmp_path, '<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n`<b>`\n')
    assert out["blocks"] == [
        {"t": "p", "inl": [{"t": "text", "s": "<script>alert(1)</script>"}]},
        {"t": "p", "inl": [{"t": "text", "s": "<img src=x onerror=alert(1)>"}]},
        {"t": "p", "inl": [{"t": "code", "s": "<b>"}]}]


@pytestmark_node
@pytest.mark.parametrize("href", ["javascript:alert", "https://example.com/", "//example.com",
                                  "#/guide/../x", "data:text/html,x", "/api/takes", "#/take/0000"])
def test_a_link_off_the_page_shows_as_its_words(tmp_path, href):
    out = parse(tmp_path, f"[go]({href})")
    assert out["blocks"] == [{"t": "p", "inl": [{"t": "span", "c": [{"t": "text", "s": "go"}]}]}]


@pytestmark_node
def test_a_script_link_with_brackets_in_it_is_still_no_link(tmp_path):
    out = parse(tmp_path, "[go](javascript:alert(1))")
    assert out["blocks"][0]["inl"][0]["t"] == "span"
    assert '"a"' not in json.dumps(out)


@pytestmark_node
def test_an_unmatched_mark_stays_a_mark(tmp_path):
    out = parse(tmp_path, "a * b ** c ` d [e](")
    assert out["blocks"] == [{"t": "p", "inl": [{"t": "text", "s": "a * b ** c ` d [e]("}]}]


@pytestmark_node
def test_every_shipped_page_parses_with_no_mark_left_over(tmp_path):
    for name, text in PAGES.items():
        out = parse(tmp_path, text)
        plain = json.dumps([n for b in out["blocks"] if b["t"] != "pre"
                            for n in (b.get("inl") or [i for item in b.get("items", []) for i in item])
                            if n["t"] == "text"])
        assert "**" not in plain and "`" not in plain and "](" not in plain, name
        assert not [n for b in out["blocks"] for n in (b.get("inl") or []) if n["t"] == "span"], name


@pytestmark_node
def test_the_same_words_always_give_the_same_page(tmp_path):
    first = parse(tmp_path, PAGES["terminal"])
    assert parse(tmp_path, PAGES["terminal"]) == first
