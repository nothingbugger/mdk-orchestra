"""Agent configuration — model assignment, memory paths, feature flags.

Defaults per BRIEFING_FOR_MAESTRO.md Appendix C and
shared/specs/interfaces.md §4. Models are env-overridable so a run can be
redirected (e.g., to Sonnet across the board for a cheap smoke, or to
mock-mode for offline testing).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """One row of the agent fleet. Read at startup, immutable at runtime.

    Attributes:
        name: canonical key matching `shared.schemas.events.AgentName`.
        personality_md_path: path to the `.md` identity file loaded as
            system prompt.
        model: Anthropic model ID. Env var `MDK_<UPPER>_MODEL` overrides.
        memory_dir: per-agent episodic memory directory (JSONL log lives
            at `<memory_dir>/events.jsonl`).
        enabled: if False the Maestro will skip this agent even if its
            domain matches the flag dispatch table.
    """

    name: str
    personality_md_path: str
    model: str
    memory_dir: str
    enabled: bool = True


def _env_model(var: str, default: str) -> str:
    return os.environ.get(var, default)


DEFAULT_AGENT_CONFIGS: dict[str, AgentConfig] = {
    "orchestrator": AgentConfig(
        name="orchestrator",
        personality_md_path="agents/maestro.md",
        model=_env_model("MDK_ORCHESTRATOR_MODEL", "claude-opus-4-7"),
        memory_dir="memory/agent_episodes/orchestrator/",
    ),
    "voltage_agent": AgentConfig(
        name="voltage_agent",
        personality_md_path="agents/voltage_agent.md",
        model=_env_model("MDK_VOLTAGE_MODEL", "claude-sonnet-4-6"),
        memory_dir="memory/agent_episodes/voltage_agent/",
    ),
    "hashrate_agent": AgentConfig(
        name="hashrate_agent",
        personality_md_path="agents/hashrate_agent.md",
        model=_env_model("MDK_HASHRATE_MODEL", "claude-sonnet-4-6"),
        memory_dir="memory/agent_episodes/hashrate_agent/",
    ),
    "environment_agent": AgentConfig(
        name="environment_agent",
        personality_md_path="agents/environment_agent.md",
        model=_env_model("MDK_ENVIRONMENT_MODEL", "claude-haiku-4-5-20251001"),
        memory_dir="memory/agent_episodes/environment_agent/",
    ),
    "power_agent": AgentConfig(
        name="power_agent",
        personality_md_path="agents/power_agent.md",
        model=_env_model("MDK_POWER_MODEL", "claude-sonnet-4-6"),
        memory_dir="memory/agent_episodes/power_agent/",
    ),
}


# Approximate per-million pricing (USD) as of April 2026. Used only for
# rough cost-tracking in event payloads and A/B reporting. Not a
# contractual pricing source — confirm with Anthropic docs for billing.
MODEL_PRICING_USD_PER_M: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0, "cache_read": 1.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25, "cache_read": 0.03},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
    """Best-effort cost estimate. Unknown models fall back to Sonnet pricing."""
    pricing = MODEL_PRICING_USD_PER_M.get(model, MODEL_PRICING_USD_PER_M["claude-sonnet-4-6"])
    non_cached_input = max(0, input_tokens - cache_read_tokens)
    return (
        non_cached_input * pricing["input"] / 1_000_000
        + cache_read_tokens * pricing["cache_read"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


def mock_mode_enabled() -> bool:
    """Return True when the agents layer should short-circuit the real API.

    Active when `MDK_AGENT_MOCK=1` or when `ANTHROPIC_API_KEY` is unset.
    Mock mode lets the end-to-end smoke run without burning API credits and
    without the dashboard having to wait for real model latency.
    """
    if os.environ.get("MDK_AGENT_MOCK") in ("1", "true", "yes"):
        return True
    return not os.environ.get("ANTHROPIC_API_KEY")
