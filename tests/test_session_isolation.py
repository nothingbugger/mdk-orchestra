"""Tests for per-run directory isolation.

Each `mdk-orchestra` run (wizard or explicit) must create its own timestamped
directory under `runs/`, so the dashboard starts with a clean event log and
prior runs remain untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mdk_orchestra import cli


def test_create_run_dir_is_timestamped(tmp_path, monkeypatch):
    """The new resolver honours MDK_RUNS_DIR, so tests point it at tmp_path."""
    monkeypatch.setenv("MDK_RUNS_DIR", str(tmp_path))
    d1 = cli._create_run_dir(prefix="demo")
    assert d1.parent == tmp_path
    assert d1.name.startswith("demo_")
    assert d1.exists() and d1.is_dir()
    # Standard subdirs created
    assert (d1 / "logs").exists()
    assert (d1 / "memory_snapshot_start").exists()


def test_two_runs_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MDK_RUNS_DIR", str(tmp_path))
    d1 = cli._create_run_dir(prefix="test")
    # Sleep > 1 sec so the timestamp in the second directory differs
    time.sleep(1.1)
    d2 = cli._create_run_dir(prefix="test")
    assert d1 != d2
    assert d1.exists() and d2.exists()
    # Both are siblings under tmp_path (the overridden runs root)
    assert d1.parent == d2.parent == tmp_path


def test_output_dir_argument_overrides_env(tmp_path, monkeypatch):
    """Explicit override arg wins over MDK_RUNS_DIR."""
    env_dir = tmp_path / "from_env"
    arg_dir = tmp_path / "from_arg"
    monkeypatch.setenv("MDK_RUNS_DIR", str(env_dir))
    d = cli._create_run_dir(prefix="test", override=str(arg_dir))
    assert d.parent == arg_dir
    # env dir is NOT used
    assert not env_dir.exists() or not list(env_dir.iterdir())


def test_backend_summary_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MDK_RUNS_DIR", str(tmp_path))
    run_dir = cli._create_run_dir()
    summary = {"label": "Test", "detail": "unit", "profile": "test"}
    cli._write_backend_summary(run_dir, summary)
    written = (run_dir / "backend_summary.json").read_text()
    assert "\"label\": \"Test\"" in written
    assert "\"profile\": \"test\"" in written


def test_build_backend_summary_full_api():
    s = cli._build_backend_summary("full_api")
    assert s["label"] == "Anthropic API"
    assert "Sonnet" in s["detail"]
    assert "Opus" in s["detail"]


def test_build_backend_summary_full_local(monkeypatch):
    monkeypatch.setattr(cli, "_resolve_standard_local_host", lambda: "http://test:11434")
    s = cli._build_backend_summary("full_local")
    assert s["label"] == "Local"
    assert "http://test:11434" in s["detail"]


def test_build_backend_summary_hybrid():
    s = cli._build_backend_summary("hybrid_economic")
    assert s["label"] == "Hybrid"
    assert "Anthropic" in s["detail"]
    assert "local" in s["detail"].lower()


def test_build_backend_summary_custom_override():
    custom = {"label": "Remote API", "detail": "groq.com · llama-3.3-70b", "profile": "full_api (custom)"}
    s = cli._build_backend_summary("full_api", custom_backend=custom)
    assert s == custom
