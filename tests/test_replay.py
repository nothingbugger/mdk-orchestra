"""Tests for the replay module (demo-replay subcommand).

Verifies:
- Timeline is loaded in chronological order across multiple streams
- Sleep durations are scaled by `speed`
- Each event lands in the correct destination file
- `replay_meta.json` is emitted with counts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdk_orchestra import replay


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_load_timeline_sorts_across_streams(tmp_path):
    _write_jsonl(tmp_path / "flags.jsonl", [
        {"ts": "2026-01-01T00:00:00Z", "event": "flag_raised", "data": {"i": 1}},
        {"ts": "2026-01-01T00:00:10Z", "event": "flag_raised", "data": {"i": 3}},
    ])
    _write_jsonl(tmp_path / "decisions.jsonl", [
        {"ts": "2026-01-01T00:00:05Z", "event": "orchestrator_decision", "data": {"i": 2}},
    ])
    timeline = replay._load_timeline(tmp_path)
    # Three events, strictly chronological
    assert [ev[1] for ev in timeline] == ["flags", "decisions", "flags"]
    indices = [ev[2]["data"]["i"] for ev in timeline]
    assert indices == [1, 2, 3]


def test_run_replay_respects_speed(tmp_path, monkeypatch):
    """A 20-second span at 4× should trigger ~5 seconds of scheduled sleep."""
    _write_jsonl(tmp_path / "flags.jsonl", [
        {"ts": "2026-01-01T00:00:00Z", "data": {"flag_id": "flg_1"}},
        {"ts": "2026-01-01T00:00:20Z", "data": {"flag_id": "flg_2"}},
    ])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sleeps: list[float] = []
    meta = replay._run_replay(
        source=tmp_path,
        run_dir=run_dir,
        speed=4.0,
        sleep_fn=lambda s: sleeps.append(s),
    )
    # First event fires immediately (sleep == 0); second scheduled at
    # 20/4 = 5 wall seconds from start.
    assert any(4.5 <= s <= 5.5 for s in sleeps), f"sleeps={sleeps}"
    assert meta["total_events"] == 2
    assert meta["counts"]["flags"] == 2


def test_run_replay_writes_events_per_stream(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "flags.jsonl", [
        {"ts": "2026-01-01T00:00:00Z", "data": {"flag_id": "flg_1"}},
    ])
    _write_jsonl(tmp_path / "decisions.jsonl", [
        {"ts": "2026-01-01T00:00:01Z", "data": {"decision_id": "dec_1"}},
    ])
    _write_jsonl(tmp_path / "actions.jsonl", [
        {"ts": "2026-01-01T00:00:02Z", "data": {"action_id": "act_1"}},
    ])
    _write_jsonl(tmp_path / "snapshots.jsonl", [
        {"ts": "2026-01-01T00:00:03Z", "data": {}},
    ])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay._run_replay(source=tmp_path, run_dir=run_dir, speed=100.0, sleep_fn=lambda s: None)
    for name in ("flags", "decisions", "actions", "snapshots"):
        out = run_dir / f"{name}.jsonl"
        assert out.exists(), f"{name}.jsonl missing"
        lines = out.read_text().splitlines()
        assert len(lines) == 1, f"expected 1 line in {name}.jsonl, got {len(lines)}"


def test_replay_to_run_writes_meta(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "flags.jsonl", [
        {"ts": "2026-01-01T00:00:00Z", "data": {"flag_id": "x"}},
    ])
    monkeypatch.setattr(replay, "_REPO_ROOT", tmp_path)
    run_dir = replay.replay_to_run(tmp_path, speed=1000.0, run_id="test_run")
    meta_path = run_dir / "replay_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["total_events"] == 1
    assert "started_at" in meta and "ended_at" in meta
    # Backend summary written up front for dashboard
    summary_path = run_dir / "backend_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["label"] == "Demo Replay"
    assert "(1000×)" in summary["detail"]
