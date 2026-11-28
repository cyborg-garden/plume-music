"""The DAW recipes print commands a player copies.  Each one must parse.

Nothing here runs a take: the commands only go through the argument parser.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from plume import cli

DOCS = Path(__file__).resolve().parents[1] / "docs" / "daw"
_FENCE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.S)


def _commands() -> list[tuple[str, str]]:
    found = []
    for page in sorted(DOCS.glob("*.md")):
        for block in _FENCE.findall(page.read_text(encoding="utf-8")):
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("plume "):
                    found.append((page.name, line))
    return found


def test_the_daw_pages_print_commands():
    assert _commands(), "no plume command found in docs/daw: the scan is broken"


@pytest.mark.parametrize("page,line", _commands())
def test_a_printed_command_parses(page, line):
    argv = shlex.split(line, comments=True)[1:]
    try:
        cli._build_parser().parse_args(argv)
    except SystemExit as e:
        pytest.fail(f"{page}: `{line}` does not parse (exit {e.code})")


def test_the_scan_catches_a_flag_with_no_value():
    """The mistake this file exists for: --record printed with no folder."""
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["lead", "--tap-start", "lofi at 90 bpm", "--record"])


@pytest.mark.parametrize("page", sorted(p.name for p in DOCS.glob("*.md")))
def test_record_always_names_a_folder(page):
    """In prose too: `--record` followed by a backtick or a line end has no folder."""
    text = (DOCS / page).read_text(encoding="utf-8")
    bare = re.findall(r"--midi-out \S+(?: --midi-only)? --record(?=`|\s*\n)", text)
    assert not bare, f"{page}: --record with no folder in {bare}"
