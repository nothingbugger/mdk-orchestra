"""End-to-end agents CLI smoke test.

Spawns `python -m agents.main` as a subprocess, feeds two FlagRaised
envelopes into the flags stream, asserts that two OrchestratorDecision
envelopes land on decisions.jsonl and that per-specialist
reasoning_response / episodic_memory_write events appear on live.jsonl.

Runs in mock mode — no Anthropic API calls, deterministic output.

The rule engine's sustained windows are wall-clock (5 min even at the
'high' sensitivity profile), so a true simulator→detector→maestro smoke
takes > 5 real minutes. That longer run is covered by the A/B experiment
runner; this test validates the agents layer is wired to the CLI and
consumes flags end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_flag(path: Path, source: str, miner_id: str, flag_type: str, severity: str) -> None:
    envelope = {
        "event": "flag_raised",
        "ts": "2026-04-20T15:30:00.000000Z",
        "source": source,
        "data": {
            "flag_id": f"flg_e2e_{miner_id}_{flag_type}",
            "miner_id": miner_id,
            "flag_type": flag_type,
            "severity": severity,
            "confidence": 0.78,
            "source_tool": "rule_engine",
            "evidence": {
                "metric": "voltage_v" if flag_type == "voltage_drift" else "temp_chip_c",
                "window_min": 5.0,
                "z_score": -4.2,
            },
            "raw_score": 0.82,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(envelope) + "\n")


def _wait_for_n_lines(path: Path, n: int, timeout_s: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if path.exists() and sum(1 for _ in path.open()) >= n:
            return
        time.sleep(0.25)
    raise AssertionError(f"{path} never reached {n} lines in {timeout_s}s")


def test_agents_cli_consumes_flags_emits_decisions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        stream_dir = tmpdir / "stream"
        memory_dir = tmpdir / "memory"
        stream_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MDK_STREAM_DIR"] = str(stream_dir)
        env["MDK_MEMORY_DIR"] = str(memory_dir)
        env["MDK_AGENT_MOCK"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT)

        flags_path = stream_dir / "flags.jsonl"
        _write_flag(flags_path, "detector", "m007", "voltage_drift", "warn")
        _write_flag(flags_path, "detector", "m019", "thermal_runaway", "crit")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agents.main",
                "--from-start",
                "--max-flags",
                "2",
                "--stop-after",
                "20",
            ],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            stdout, _ = proc.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise

        assert proc.returncode == 0, (
            f"agents.main exited non-zero ({proc.returncode}). Output:\n{stdout.decode()}"
        )

        decisions_path = stream_dir / "decisions.jsonl"
        assert decisions_path.exists(), "decisions.jsonl was never created"
        decision_lines = decisions_path.read_text().splitlines()
        assert len(decision_lines) == 2, f"expected 2 decisions, got {len(decision_lines)}"

        for i, raw in enumerate(decision_lines):
            env_obj = json.loads(raw)
            assert env_obj["event"] == "orchestrator_decision"
            data = env_obj["data"]
            assert data["flag_id"] in (
                "flg_e2e_m007_voltage_drift",
                "flg_e2e_m019_thermal_runaway",
            )
            assert data["miner_id"] in ("m007", "m019")
            assert data["consulted_agents"], f"decision {i} has no consulted agents"
            assert data["reasoning_trace"], f"decision {i} has empty reasoning_trace"
            assert data["autonomy_level"] in (
                "L1_observe",
                "L2_suggest",
                "L3_bounded_auto",
                "L4_human_only",
            )
            assert data["total_cost_usd"] == 0.0, "mock mode should cost zero"

        # At least one episodic_memory_write per specialist consultation
        live_path = stream_dir / "live.jsonl"
        live_text = live_path.read_text() if live_path.exists() else ""
        assert '"event":"episodic_memory_write"' in live_text.replace(" ", "")
