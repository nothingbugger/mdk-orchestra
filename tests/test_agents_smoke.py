"""End-to-end smoke test for the agents layer (mock mode, no API calls).

Feeds one `flag_raised` into the bus, runs Maestro for one flag, asserts
that an `orchestrator_decision` envelope lands on decisions.jsonl with
sensible fields.

Runs offline: `MDK_AGENT_MOCK=1` short-circuits the Anthropic call and
uses the deterministic mock responses from each specialist and from
Maestro's synthesis step. This validates the wiring, not the quality of
model output.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("structlog")

from shared.event_bus import read_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import FlagEvidence, FlagRaised


def _setup_tmp_env(tmpdir: Path) -> None:
    os.environ["MDK_AGENT_MOCK"] = "1"
    os.environ["MDK_STREAM_DIR"] = str(tmpdir / "stream")
    os.environ["MDK_MEMORY_DIR"] = str(tmpdir / "memory")
    os.environ["MDK_LOG_DIR"] = str(tmpdir / "log")


def _emit_flag(miner_id: str = "m042") -> FlagRaised:
    flag = FlagRaised(
        flag_id="flg_smoke_01",
        miner_id=miner_id,
        flag_type="voltage_drift",
        severity="warn",
        confidence=0.71,
        source_tool="rule_engine",
        evidence=FlagEvidence(
            metric="voltage_v",
            window_min=30.0,
            z_score=-5.5,
            recent_mean=11.83,
            baseline_mean=12.05,
        ),
        raw_score=0.82,
    )
    write_event("flag_raised", "detector", flag)
    return flag


def test_agents_smoke_one_flag_one_decision() -> None:
    """One flag in → one orchestrator_decision out, with consulted agents and a trace."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_tmp_env(tmpdir)

        # Import after env is set so paths pick up tmp dirs.
        from agents.maestro import run_orchestrator

        _emit_flag()

        run_orchestrator(from_start=True, max_flags=1, stop_after=10.0)

        sp = stream_paths()
        decisions = [env for env in read_events(sp.decisions) if env.event == "orchestrator_decision"]
        assert decisions, "no orchestrator_decision was emitted"

        dec = decisions[0].typed_data()
        assert dec.miner_id == "m042"
        assert dec.flag_id == "flg_smoke_01"
        assert dec.consulted_agents, "expected at least one consulted specialist"
        assert dec.reasoning_trace, "reasoning_trace must be non-empty"
        assert 0.0 <= dec.confidence <= 1.0
        assert dec.autonomy_level in {"L1_observe", "L2_suggest", "L3_bounded_auto", "L4_human_only"}

        # In mock mode cost should be zero (no API calls).
        assert dec.total_cost_usd == 0.0
