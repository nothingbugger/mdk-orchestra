"""Smoke test for cost-optimization changes to agents/maestro.py.

Verifies:
1. Primary-only path when specialist returns definitive verdict (no fallback call)
2. Primary + fallback path when specialist returns inconclusive
3. Tiered synthesis — L3/L4 triggers Opus second opinion, L1/L2 does not

Runs in MDK_AGENT_MOCK=1 — no real API calls. Cost accounting still tests the
*code path* (which call_structured invocations fired), and the mock result's
`cost_usd=0` is expected.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("pydantic")

os.environ.setdefault("MDK_AGENT_MOCK", "1")


def _setup_tmp_env(tmpdir: Path) -> None:
    os.environ["MDK_AGENT_MOCK"] = "1"
    os.environ["MDK_STREAM_DIR"] = str(tmpdir / "stream")
    os.environ["MDK_MEMORY_DIR"] = str(tmpdir / "memory")


def _make_flag(
    flag_type: str,
    severity: str = "warn",
    miner_id: str = "m042",
    flag_id: str = "flg_test_01",
) -> dict:
    return {
        "flag_id": flag_id,
        "miner_id": miner_id,
        "flag_type": flag_type,
        "severity": severity,
        "confidence": 0.85,
        "source_tool": "rule_engine",
        "evidence": {"metric": "x", "window_min": 5.0},
        "raw_score": 0.80,
    }


def test_primary_definitive_skips_fallback() -> None:
    """hashrate_degradation → primary hashrate_agent with real_signal → no fallback call."""
    from agents.maestro import Maestro

    calls: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        orig_consult = m._consult

        def tracking(agent_names, flag):
            calls.append(",".join(agent_names))
            return orig_consult(agent_names, flag)

        with patch.object(m, "_consult", side_effect=tracking):
            m.dispatch_flag(_make_flag("hashrate_degradation", severity="warn"))

        # Primary only — mock_response returns assessment based on flag confidence;
        # at 0.85 it returns "real_signal" (>= 0.7 threshold). No fallback expected.
        assert calls == ["hashrate_agent"], f"expected primary-only, got {calls}"


def test_primary_inconclusive_triggers_fallback() -> None:
    """Force primary to inconclusive by monkeypatching and verify fallback fires."""
    from agents.base_specialist import BaseSpecialist
    from agents.maestro import Maestro

    calls: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        orig_consult = m._consult

        def tracking(agent_names, flag):
            calls.append(",".join(agent_names))
            return orig_consult(agent_names, flag)

        # Force the primary specialist to return "inconclusive" by monkeypatching
        # its _mock_response to override the canned output.
        primary = m.specialists["hashrate_agent"]

        def forced_inconclusive(request, episodes):
            flag = request.context.flag
            return {
                "assessment": "inconclusive",
                "confidence": 0.45,
                "severity_estimate": flag.get("severity", "warn"),
                "reasoning": "[forced test] inconclusive",
                "recommended_action_hint": "observe",
            }

        with patch.object(primary, "_mock_response", side_effect=forced_inconclusive):
            with patch.object(m, "_consult", side_effect=tracking):
                m.dispatch_flag(_make_flag("hashrate_degradation", severity="warn"))

        # Primary first, then fallback because primary was inconclusive.
        assert len(calls) == 2, f"expected primary + fallback, got {calls}"
        assert calls[0] == "hashrate_agent"
        assert calls[1] == "voltage_agent"


def test_fan_anomaly_has_no_fallback() -> None:
    """fan_anomaly primary is environment_agent with no fallback — even if
    environment returns inconclusive, no second call should fire."""
    from agents.maestro import Maestro

    calls: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        primary = m.specialists["environment_agent"]

        def forced_inconclusive(request, episodes):
            return {
                "assessment": "inconclusive",
                "confidence": 0.4,
                "severity_estimate": "warn",
                "reasoning": "[forced]",
                "recommended_action_hint": "observe",
            }

        orig_consult = m._consult

        def tracking(agent_names, flag):
            calls.append(",".join(agent_names))
            return orig_consult(agent_names, flag)

        with patch.object(primary, "_mock_response", side_effect=forced_inconclusive):
            with patch.object(m, "_consult", side_effect=tracking):
                m.dispatch_flag(_make_flag("fan_anomaly", severity="warn"))

        assert calls == ["environment_agent"], (
            f"fan_anomaly has no fallback; got {calls}"
        )


def test_tiered_synthesis_no_escalation_on_l1_l2() -> None:
    """Mock Maestro's first-pass to L1 — no second Opus call."""
    from agents.maestro import Maestro

    structured_calls: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        import agents.maestro as maestro_mod

        real_call = maestro_mod.call_structured

        def tracking_call(**kwargs):
            structured_calls.append(kwargs["model"])
            return real_call(**kwargs)

        # The mock_decision defaults to L1/L2 when specialists report low confidence.
        # With a warn flag and mocked real_signal conf=0.85 from specialists,
        # the mock synthesis returns L2_suggest alert_operator. Should NOT escalate.
        with patch.object(maestro_mod, "call_structured", side_effect=tracking_call):
            decision = m.dispatch_flag(_make_flag("hashrate_degradation", severity="warn"))

        # Exactly one synthesis call (the first-pass Sonnet); specialist calls
        # also go through call_structured. We only assert the Opus model doesn't
        # appear for maestro synthesis.
        opus_calls = [c for c in structured_calls if c.startswith("claude-opus")]
        assert len(opus_calls) == 0, (
            f"expected 0 Opus calls for L1/L2 outcome, got {opus_calls}"
        )
        assert decision.autonomy_level in {"L1_observe", "L2_suggest"}


def test_tiered_synthesis_escalates_on_l3() -> None:
    """Force first-pass to return L3_throttle — second Opus call fires."""
    from agents.maestro import Maestro

    structured_calls: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        import agents.maestro as maestro_mod

        real_call = maestro_mod.call_structured

        # Fake first-pass result — L3_throttle to force escalation.
        from agents._client import LLMResult

        fake_first = LLMResult(
            tool_input={
                "action": "throttle",
                "action_params": {"target_hashrate_pct": 0.80, "duration_min": 60},
                "autonomy_level": "L3_bounded_auto",
                "confidence": 0.82,
                "reasoning_trace": "[forced] first-pass proposes L3 throttle",
            },
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cost_usd=0.001,  # fake Sonnet cost
            latency_ms=50.0,
            model="claude-sonnet-4-6",
            is_mock=False,
        )
        call_count = {"n": 0}

        def tracking_call(**kwargs):
            structured_calls.append(
                {"model": kwargs["model"], "tool": kwargs.get("tool_name")}
            )
            if kwargs.get("tool_name") == "submit_decision":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return fake_first
                # Second call (Opus) — return a fake confirming L3
                return LLMResult(
                    tool_input={
                        "action": "throttle",
                        "action_params": {"target_hashrate_pct": 0.80, "duration_min": 60},
                        "autonomy_level": "L3_bounded_auto",
                        "confidence": 0.88,
                        "reasoning_trace": "[forced] Opus confirms L3",
                    },
                    input_tokens=150,
                    output_tokens=60,
                    cache_read_tokens=0,
                    cost_usd=0.005,
                    latency_ms=80.0,
                    model="claude-opus-4-7",
                    is_mock=False,
                )
            return real_call(**kwargs)

        with patch.object(maestro_mod, "call_structured", side_effect=tracking_call):
            decision = m.dispatch_flag(_make_flag("hashrate_degradation", severity="crit"))

        submit_models = [c["model"] for c in structured_calls if c["tool"] == "submit_decision"]
        assert submit_models == ["claude-sonnet-4-6", "claude-opus-4-7"], (
            f"expected Sonnet then Opus, got {submit_models}"
        )
        assert decision.autonomy_level == "L3_bounded_auto"
        # total_cost_usd = first ($0.001) + second ($0.005) + specialists (mock = 0)
        assert decision.total_cost_usd == pytest.approx(0.006, abs=1e-6)


def test_anomaly_composite_still_uses_flow3() -> None:
    """anomaly_composite must dispatch to all 4 specialists even under the new scheme."""
    from agents.maestro import Maestro

    calls: list[list[str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        _setup_tmp_env(Path(tmp))
        m = Maestro()

        orig_consult = m._consult

        def tracking(agent_names, flag):
            calls.append(list(agent_names))
            return orig_consult(agent_names, flag)

        with patch.object(m, "_consult", side_effect=tracking):
            m.dispatch_flag(_make_flag("anomaly_composite", severity="warn"))

        assert calls, "no consult calls recorded"
        assert set(calls[0]) == {
            "voltage_agent",
            "hashrate_agent",
            "environment_agent",
            "power_agent",
        }, f"flow3 mismatch: {calls[0]}"
