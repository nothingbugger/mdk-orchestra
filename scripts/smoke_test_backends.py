"""Smoke test for the 3 llm_routing profiles — full_api, hybrid_economic, full_local.

Runs one synthetic flag per profile, measures cost/latency, reports
which backend each agent_slot actually resolved to.

Invocation:
    MDK_LLM_PROFILE=full_api       python scripts/smoke_test_backends.py
    MDK_LLM_PROFILE=hybrid_economic python scripts/smoke_test_backends.py
    MDK_LLM_PROFILE=full_local     python scripts/smoke_test_backends.py

Output: a compact per-run summary + writes `scripts/smoke_test_backends.log`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def make_flag() -> dict:
    return {
        "flag_id": "flg_smoke_backends_01",
        "miner_id": "m042",
        "flag_type": "hashrate_degradation",
        "severity": "warn",
        "confidence": 0.78,
        "source_tool": "rule_engine",
        "evidence": {
            "metric": "hashrate_th",
            "window_min": 30.0,
            "recent_mean": 95.3,
            "baseline_mean": 104.0,
            "z_score": -3.4,
        },
        "raw_score": 0.82,
    }


def run_once() -> dict:
    from agents.llm_backend import resolve_routing
    from agents.maestro import Maestro

    profile = os.environ.get("MDK_LLM_PROFILE", "(none)")
    print(f"\n=== profile: {profile} ===", flush=True)
    slots = [
        "maestro.dispatch",
        "maestro.escalation",
        "maestro.curation",
        "specialists.voltage",
        "specialists.hashrate",
        "specialists.environment",
        "specialists.power",
    ]
    print("[routing] slot → (backend, model)")
    for s in slots:
        r = resolve_routing(s)
        print(f"  {s:28s} → ({r['backend']}, {r['model']})")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MDK_STREAM_DIR"] = str(Path(tmp) / "stream")
        os.environ["MDK_MEMORY_DIR"] = str(Path(tmp) / "memory")
        t0 = time.monotonic()
        m = Maestro()
        decision = m.dispatch_flag(make_flag())
        wall_s = time.monotonic() - t0

    summary = {
        "profile": profile,
        "decision_id": decision.decision_id,
        "action": decision.action,
        "autonomy_level": decision.autonomy_level,
        "confidence": decision.confidence,
        "total_cost_usd": round(decision.total_cost_usd, 6),
        "total_latency_ms": round(decision.total_latency_ms, 2),
        "wall_seconds": round(wall_s, 2),
        "consulted": list(decision.consulted_agents),
        "reasoning_trace_len": len(decision.reasoning_trace),
        "reasoning_trace_head": decision.reasoning_trace[:200],
    }
    print("\n--- result ---")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run_once()
    log = REPO / "scripts" / "smoke_test_backends.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(s) + "\n")
    print(f"\nlogged → {log}")
