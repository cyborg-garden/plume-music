"""How plume starts itself as a child: from an install, and from a frozen bundle.

Nothing here opens a device.  The children that really run only list styles.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from plume import childproc, cli, guitake, oscompat, voice

#: A stand-in for a one-folder bundle's own binary.
BUNDLE_BINARY = str(Path(os.sep, "apps", "plume-bundle", "plume"))


@pytest.fixture
def frozen(monkeypatch):
    """This process, looking like the bundle's binary."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", BUNDLE_BINARY)
    return BUNDLE_BINARY


def _route():
    return guitake.Route(name="Some Input", mode="direct", channel=1, hear_me=False)


# ------------------------------------------------------------------ the command line

def test_an_install_runs_its_own_python_with_the_safe_path_flag():
    assert childproc.plume_argv() == (sys.executable, "-P", "-m", "plume")


def test_a_frozen_bundle_runs_its_own_binary(frozen):
    assert childproc.plume_argv() == (frozen,)


def test_frozen_is_read_at_call_time(monkeypatch):
    before = childproc.plume_argv()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert childproc.plume_argv() != before


@pytest.mark.parametrize("layout", ["install", "frozen"])
def test_start_builds_a_line_the_real_parser_reads(layout, request, tmp_path):
    if layout == "frozen":
        request.getfixturevalue("frozen")
    head = list(childproc.plume_argv())
    argv = guitake.build_argv(_route(), tmp_path / "take", key=None, style=None, seed=7)
    assert argv[:len(head)] == head
    parsed = cli._build_parser().parse_args(argv[len(head):])
    assert parsed.tap_start and parsed.seed == 7


def test_a_frozen_start_has_no_interpreter_flags(frozen, tmp_path):
    argv = guitake.build_argv(_route(), tmp_path / "take", key=None, style=None, seed=7)
    assert argv[:2] == [frozen, "lead"]
    assert "-P" not in argv and "-m" not in argv


@pytest.mark.parametrize("layout", ["install", "frozen"])
def test_the_lister_and_the_meter_use_the_same_line(layout, request, monkeypatch):
    if layout == "frozen":
        request.getfixturevalue("frozen")
    head = list(childproc.plume_argv())
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b'{"devices": []}')

    class Meter:
        returncode = 0

        def __init__(self, argv, **kw):
            seen.append(argv)

        def communicate(self, timeout=None):
            return b'{"level": 0}', b""

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.delenv(guitake.AUDIO_CONSENT_ENV, raising=False)
    monkeypatch.setattr(guitake, "_under_pytest", lambda: False)
    monkeypatch.setattr(guitake.subprocess, "run", run)
    monkeypatch.setattr(guitake.subprocess, "Popen", Meter)
    guitake.default_lister()
    try:
        guitake.default_meter("Some Input", 0.1, guitake.threading.Event())
    except guitake.TakeError:
        pass                    # what the fake child said is not the point: its argv is
    assert [argv[:len(head) + 1] for argv in seen] == [[*head, "inputs"], [*head, "inputs"]]


@pytest.mark.parametrize("layout", ["install", "frozen"])
def test_voice_starts_its_band_with_the_same_line(layout, request, tmp_path):
    if layout == "frozen":
        request.getfixturevalue("frozen")
    head = list(childproc.plume_argv())
    seen = {}

    class Recording:
        def __init__(self, args, **kw):
            seen.update(kw, args=args)

        def poll(self):
            return None

    band = voice.Band(prompt="slow", base=tmp_path, write=lambda s: None, popen=Recording)
    band.start()
    band._proc = None
    assert seen["args"][:len(head) + 1] == [*head, "lead"]
    # The process-group contract is the same in both layouts: the child is started
    # directly, leads its own group, and is told who its parent is.
    assert seen["env"][oscompat.OWN_GROUP_ENV] == str(os.getpid())
    for name, value in oscompat.detached_kwargs().items():
        assert seen[name] == value


def test_voice_sends_a_cue_with_the_same_line():
    """The cue line is built from the helper, not spelled out again."""
    source = Path(voice.__file__).read_text(encoding="utf-8")
    assert '"-m", "plume"' not in source
    assert source.count("childproc.plume_argv()") == 2


def test_no_other_module_spells_the_child_line_out():
    here = Path(childproc.__file__).parent
    spelled = [p.name for p in here.glob("*.py")
               if p.name != "childproc.py" and '"-m", "plume"' in p.read_text(encoding="utf-8")]
    assert spelled == []


# ---------------------------------------------------------------- the import path

def test_the_working_folder_comes_off_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    elsewhere = str(tmp_path.parent)
    path = ["", str(tmp_path), elsewhere, os.curdir]
    childproc.drop_working_folder(path)
    assert path == [elsewhere]


def test_a_bundle_that_is_the_working_folder_keeps_its_own_code(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    inner = bundle / "_internal"
    inner.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "plume"))
    monkeypatch.chdir(bundle)
    path = ["", str(bundle), str(inner)]
    childproc.drop_working_folder(path)
    assert path == [str(bundle), str(inner)]


def test_an_install_in_the_working_folder_still_loses_it(tmp_path, monkeypatch):
    """Only a frozen bundle's own folder is kept; an install gets what -P gives."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.chdir(tmp_path)
    path = [str(tmp_path)]
    childproc.drop_working_folder(path)
    assert path == []


# ------------------------------------------------------------------- the dispatch

def test_arguments_run_the_cli_and_never_the_window(monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "main", lambda argv: ran.append(argv) or 3)
    code = childproc.main(["styles"], window=lambda: pytest.fail("the window was loaded"))
    assert code == 3 and ran == [["styles"]]


def test_no_arguments_open_the_window(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv: pytest.fail("the cli ran"))
    assert childproc.main([], window=lambda: 5) == 5
    assert childproc.main(["-psn_0_12345"], window=lambda: None) == 0


def test_no_window_means_the_cli(monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "main", lambda argv: ran.append(argv) or 0)
    assert childproc.main([]) == 0 and ran == [[]]


def test_the_module_imports_no_window_or_audio_code():
    code = ("import sys; import plume.childproc; "
            "print(sorted(m for m in sys.modules if m.startswith('plume')))")
    done = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True,
                          check=True, timeout=60)
    assert done.stdout.strip() == "['plume', 'plume.childproc']"


# ------------------------------------------------- real children, in both layouts

PLANTED = textwrap.dedent("""\
    import pathlib
    pathlib.Path(__file__).with_name("planted-ran").write_text("ran")
    raise SystemExit(0)
""")

#: What a bundle's entry point does, run by a plain python standing in for the binary:
#: marks itself frozen, leaves the working folder on the path as a bundler might, and
#: hands over to childproc.main with a window that would leave a mark.
ENTRY = textwrap.dedent("""\
    import pathlib
    import sys

    sys.frozen = True
    sys.path.append("")

    from plume.childproc import main


    def window():
        import bundle_window
        return bundle_window.run()


    code = main(window=window)
    try:
        import cwd_probe                      # only the working folder has one
    except ImportError:
        pass
    sys.exit(code)
""")

WINDOW = textwrap.dedent("""\
    import pathlib


    def run():
        pathlib.Path(__file__).with_name("window-ran").write_text("ran")
        return 0
""")


@pytest.fixture
def planted(tmp_path):
    """A working folder holding files named to be picked up by a careless import."""
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "plume.py").write_text(PLANTED, encoding="utf-8")
    (folder / "cwd_probe.py").write_text(PLANTED, encoding="utf-8")
    return folder


def _quiet_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PLUME_ALLOW_AUDIO")}
    return env


def test_an_installed_child_never_runs_a_plume_in_the_working_folder(planted):
    done = subprocess.run([*childproc.plume_argv(), "styles"], cwd=planted, env=_quiet_env(),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "lofi" in done.stdout
    assert not (planted / "planted-ran").exists()


@pytest.fixture
def bundle(tmp_path):
    """A simulated one-folder layout: an entry point beside its window code."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "entry.py").write_text(ENTRY, encoding="utf-8")
    (folder / "bundle_window.py").write_text(WINDOW, encoding="utf-8")
    # -P keeps the stand-in python from adding anything of its own; the entry point
    # then adds the bundle and the working folder itself, as the layout under test.
    boot = ("import runpy, sys; sys.path.insert(0, sys.argv[1]); entry = sys.argv[1]; "
            "del sys.argv[1]; runpy.run_path(entry + '/entry.py', run_name='__main__')")
    return folder, [sys.executable, "-P", "-c", boot, str(folder)]


def test_a_frozen_child_runs_the_cli_and_loads_no_window(bundle, planted):
    folder, binary = bundle
    done = subprocess.run([*binary, "styles"], cwd=planted, env=_quiet_env(),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "lofi" in done.stdout
    assert not (folder / "window-ran").exists()
    assert not (planted / "planted-ran").exists()


def test_a_frozen_binary_with_no_arguments_opens_its_window(bundle, planted):
    folder, binary = bundle
    done = subprocess.run(binary, cwd=planted, env=_quiet_env(),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert (folder / "window-ran").exists()
    assert not (planted / "planted-ran").exists()


def test_the_planted_file_would_run_without_the_protection(bundle, planted):
    """The teeth of the tests above: the same layout, with the protection left out."""
    folder, binary = bundle
    entry = (folder / "entry.py").read_text(encoding="utf-8")
    (folder / "entry.py").write_text(
        entry.replace("code = main(window=window)", "code = 0"), encoding="utf-8")
    subprocess.run([*binary, "styles"], cwd=planted, env=_quiet_env(),
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert (planted / "planted-ran").exists()
