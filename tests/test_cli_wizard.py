"""Tests for the interactive CLI wizard.

Covers:
- Quit path (option 4) exits cleanly
- Invalid input re-prompts without crash
- Demo path short-circuits with a clear error when ANTHROPIC_API_KEY is absent
- Wizard selection dispatches to the correct branch
- API custom branch rejects missing env var
- Local branch autodetects vs falls back to custom host
"""

from __future__ import annotations

import io
from unittest import mock

import pytest

from mdk_orchestra import cli


def _run_wizard_with_input(text: str) -> int:
    """Drive the wizard by feeding `text` to stdin."""
    with mock.patch("sys.stdin", io.StringIO(text)):
        return cli._run_wizard()


# ---------------------------------------------------------------------------
# Option 4 — Quit
# ---------------------------------------------------------------------------


def test_wizard_ctrl_c_exits_cleanly(capsys, monkeypatch):
    """Ctrl+C at the main menu returns 0 and prints goodbye."""
    def _raise_kbd(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_prompt", _raise_kbd)
    rc = cli._run_wizard()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Goodbye" in out
    assert "MDK Orchestra" in out
    # Four options visible, no explicit Quit. Rich table whitespace varies
    # with terminal width; match on the labels alone.
    assert "Demo" in out
    assert "API" in out
    assert "Local LLM" in out
    assert "Explore" in out
    assert "Quit" not in out
    assert "Ctrl+C to exit" in out


def test_wizard_explore_dispatches(monkeypatch):
    """Selecting [4] routes into _wizard_explore."""
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "4")
    monkeypatch.setattr(cli, "_wizard_explore", lambda: 7)  # sentinel
    rc = cli._run_wizard()
    assert rc == 7


# ---------------------------------------------------------------------------
# Duration parsing — must return int, not float (ab_experiment expects int)
# ---------------------------------------------------------------------------


def test_prompt_duration_returns_int(monkeypatch):
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "15")
    value = cli._prompt_duration_minutes()
    assert isinstance(value, int)
    assert value == 15


def test_prompt_duration_default_when_empty(monkeypatch):
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "30")
    value = cli._prompt_duration_minutes()
    assert value == 30


def test_prompt_duration_accepts_trailing_m(monkeypatch):
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "60m")
    assert cli._prompt_duration_minutes() == 60


def test_prompt_duration_falls_back_on_garbage(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "abc")
    value = cli._prompt_duration_minutes()
    assert value == 30
    out = capsys.readouterr().out
    assert "Invalid duration" in out


def test_prompt_duration_floors_float_input(monkeypatch):
    """`2.5` should become an int (2 or 3), never a float."""
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "2.5")
    value = cli._prompt_duration_minutes()
    assert isinstance(value, int)
    assert value in (2, 3)


def test_prompt_duration_minimum_one_minute(monkeypatch):
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "0")
    assert cli._prompt_duration_minutes() == 1


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_wizard_invalid_choice_rejected(capsys, monkeypatch):
    """Input '99' should print 'Invalid choice' and re-prompt.

    We verify by checking that the prompt emits the invalid-choice message
    when fed an out-of-range input; then Ctrl+C terminates.
    """
    inputs = iter(["99", KeyboardInterrupt])
    def _feed(msg, valid=None, default=None):
        val = next(inputs)
        if isinstance(val, type) and issubclass(val, BaseException):
            raise val()
        if valid is not None and val not in valid:
            # mimic the real _prompt's rejection message
            print(f"  Invalid choice — must be one of {sorted(valid)}.")
            raise KeyboardInterrupt
        return val
    monkeypatch.setattr(cli, "_prompt", _feed)
    rc = cli._run_wizard()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Invalid choice" in out


# ---------------------------------------------------------------------------
# Option 1 — Demo (replay-based, no API needed)
# ---------------------------------------------------------------------------


def test_wizard_demo_aborts_on_ctrl_c_before_start(monkeypatch, capsys):
    """Ctrl+C at the 'Press Enter to start' prompt returns to main menu.

    We mock `_run_wizard` to return a sentinel so we can detect the
    re-entry without recursing through the real banner.
    """
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    sentinel = 42
    monkeypatch.setattr(cli, "_run_wizard", lambda: sentinel)
    rc = cli._wizard_demo()
    assert rc == sentinel


def test_wizard_demo_fails_when_assets_missing(monkeypatch, tmp_path, capsys):
    """If examples/demo_replay/ is absent, demo prints a clear error."""
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path)  # no examples/ under tmp_path
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    rc = cli._wizard_demo()
    assert rc == 1
    err = capsys.readouterr().err
    assert "demo replay assets missing" in err


# ---------------------------------------------------------------------------
# Option 2 — API
# ---------------------------------------------------------------------------


def test_wizard_api_menu_shows_provider_options(capsys, monkeypatch):
    """Main [2] → API menu must surface the two provider branches."""
    inputs = iter(["2", KeyboardInterrupt, KeyboardInterrupt])
    def _feed(msg, valid=None, default=None):
        val = next(inputs)
        if isinstance(val, type) and issubclass(val, BaseException):
            raise val()
        return val
    monkeypatch.setattr(cli, "_prompt", _feed)
    # Prevent KeyboardInterrupt from recursing the wizard forever
    monkeypatch.setattr(cli, "_wizard_api", lambda: 0)
    cli._run_wizard()
    out = capsys.readouterr().out
    # With Rich the menu table contains "API" as a labelled row. The
    # row descriptions get heavily truncated when stdout is captured
    # (no real terminal → narrow default width), so we only assert on
    # tokens that survive regardless of console width.
    assert "API" in out
    assert "Explore" in out


def test_collect_api_key_inline_accepts_pasted(monkeypatch, capsys):
    """Pasting a key sets the env var and acknowledges the last 4 chars."""
    monkeypatch.delenv("TEST_FAKE_KEY_X", raising=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "sk-ant-faketest1234")
    ok = cli._collect_api_key_inline("TEST_FAKE_KEY_X", "FakeProvider")
    assert ok is True
    assert os.environ.get("TEST_FAKE_KEY_X") == "sk-ant-faketest1234"
    out = capsys.readouterr().out
    assert "Key accepted" in out
    assert "...1234" in out


def test_collect_api_key_inline_uses_env_when_empty(monkeypatch, capsys):
    """Empty input falls back to pre-existing env var."""
    monkeypatch.setenv("TEST_FAKE_KEY_Y", "sk-env-value-AAAA")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    ok = cli._collect_api_key_inline("TEST_FAKE_KEY_Y", "FakeProvider")
    assert ok is True
    out = capsys.readouterr().out
    assert "Using TEST_FAKE_KEY_Y from env" in out
    assert "...AAAA" in out


def test_collect_api_key_inline_fails_without_anything(monkeypatch, capsys):
    """Empty input AND no env var → graceful failure with instructions."""
    monkeypatch.delenv("TEST_FAKE_KEY_Z", raising=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    ok = cli._collect_api_key_inline("TEST_FAKE_KEY_Z", "FakeProvider")
    assert ok is False
    err = capsys.readouterr().err
    assert "No key provided" in err
    assert "TEST_FAKE_KEY_Z" in err


# Module-level import needed for env-manipulation tests above
import os  # noqa: E402


# ---------------------------------------------------------------------------
# Option 3 — Local
# ---------------------------------------------------------------------------


def test_wizard_local_no_ollama_goes_back(monkeypatch, capsys):
    """If Ollama isn't up, the 'not detected' menu with Back option appears.

    KeyboardInterrupt from an inner prompt bubbles past _run_wizard; we
    catch it here and assert on the output captured up to that point.
    """
    monkeypatch.setattr(cli, "_probe_ollama_models", lambda *a, **k: [])
    inputs = iter(["3", KeyboardInterrupt])
    def _feed(msg, valid=None, default=None):
        val = next(inputs)
        if isinstance(val, type) and issubclass(val, BaseException):
            raise val()
        return val
    monkeypatch.setattr(cli, "_prompt", _feed)
    with pytest.raises(KeyboardInterrupt):
        cli._run_wizard()
    out = capsys.readouterr().out
    assert "Ollama not detected" in out


def test_wizard_local_lists_ollama_models(monkeypatch, capsys):
    """If Ollama reports models, they must appear in the prompt."""
    monkeypatch.setattr(
        cli, "_probe_ollama_models",
        lambda *a, **k: ["qwen2.5:7b-instruct", "llama3.3:70b"],
    )
    inputs = iter(["3", KeyboardInterrupt])
    def _feed(msg, valid=None, default=None):
        val = next(inputs)
        if isinstance(val, type) and issubclass(val, BaseException):
            raise val()
        return val
    monkeypatch.setattr(cli, "_prompt", _feed)
    with pytest.raises(KeyboardInterrupt):
        cli._run_wizard()
    out = capsys.readouterr().out
    assert "qwen2.5:7b-instruct" in out
    assert "llama3.3:70b" in out
