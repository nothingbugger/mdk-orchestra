"""Memory curator — Maestro-driven, simulator-time scheduled.

Architecture: **curator-periodic**. Every 30 simulated minutes Maestro
looks back at the decisions it made since the previous curation and
updates the five memory files (one for itself + four for specialists)
via the `write_memory_pattern` tool.

- Simulator time is taken from `OrchestratorDecision.ts` on the bus.
  A fresh decision advances a running `sim_time_cursor`; when the cursor
  has jumped ≥ `CURATION_INTERVAL_MIN` since the last pass we trigger.
- The curation call is a single Opus message with `tool_choice="auto"`,
  so Maestro may reply with zero, one, or many `write_memory_pattern`
  tool uses, or with a plain text saying "nothing worth remembering".
- Passes with no new decisions since last curation are silent no-ops —
  we don't even make the API call.
- Specialists do not curate. They are passive readers; the memory files
  are loaded into their system prompts (ephemeral cache) at every
  invocation so they see the latest curator output.

The curator is invoked from the Maestro dispatch loop:

    maestro.dispatch_flag(...)  # standard flag → decision
    curator.maybe_curate(maestro, new_decision)  # cheap; fires API only when due

This module has no hidden timers; all triggers come from decision
timestamps flowing through. That keeps it testable (drive with fake
decisions) and avoids wall-clock drift on fast-speed A/B runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from agents._client import call_structured_multi_tool
from agents.tools import (
    WRITE_MEMORY_PATTERN_SCHEMA,
    load_memory_file,
    write_memory_pattern,
)
from shared.schemas.events import OrchestratorDecision

_LOG = structlog.get_logger(__name__)

CURATION_INTERVAL_MIN: float = float(os.environ.get("MDK_CURATION_INTERVAL_MIN", "30.0"))
"""Wall-clock minutes between curation passes (the curator uses
``datetime.now()`` as its timestamp source). 30-minute default is
tuned for real-time deployments; fast-speed A/B runs should override
via ``MDK_CURATION_INTERVAL_MIN`` to keep cadence aligned with simulated
time (e.g. 3.0 at 10× speed ≈ 30 sim-minutes per pass)."""

CURATION_SYSTEM_APPENDIX = """

# Memory curation role

You curate the memory files for yourself and for the four specialists.
Every 30 simulated minutes you review recent decisions and update the
memory files with patterns worth remembering.

Rules:
- Write a pattern ONLY if you've seen it ≥ 2 times with a similar
  outcome. One-off events are NOT memory material.
- Write domain-specific patterns to the specialist's file
  (voltage_memory.md for voltage patterns, etc.) so the specialist
  reads them on its next invocation.
- Write cross-domain patterns (e.g. a combination of specialist verdicts
  that implied a specific action) to maestro_memory.md.
- Patterns are LESSONS, not event logs. Phrase them as reusable
  knowledge ("fan_rpm collapse with chip temp climb → throttle L3,
  override alert hint") not as incident reports.
- Do NOT write trivial observations, restatements of the .md
  personality files, or vague generalities.
- If nothing in this window is worth remembering, reply with plain
  text "nothing to curate" and emit zero tool calls.

Use `write_memory_pattern(target_file, pattern_name, signature,
verdict_or_action, confidence, reasoning, example_dec_id,
increment_if_exists)` for each pattern you want to persist. You may
make multiple tool calls in the same response.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class CuratorState:
    """Per-Maestro curator state — rolling buffer of decisions since the last
    curation pass, plus the simulator-time cursor that fires the next one."""

    last_curation_sim_ts: datetime | None = None
    pending_decisions: list[dict[str, Any]] = field(default_factory=list)
    total_passes: int = 0
    total_patterns_written: int = 0
    total_patterns_incremented: int = 0
    total_cost_usd: float = 0.0

    def record_decision(self, decision_env: dict[str, Any]) -> None:
        """Append a decision envelope (either the typed data dict, or the
        full envelope with ts) to the pending buffer."""
        self.pending_decisions.append(decision_env)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_curate(
    maestro_config: Any,
    maestro_personality: str,
    state: CuratorState,
    decision_ts: datetime,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """Trigger a curation pass if ≥ CURATION_INTERVAL_MIN simulated minutes
    have elapsed since the last pass. Returns a summary dict when a pass
    fired, or None when skipped."""

    if state.last_curation_sim_ts is None:
        # First decision ever — set the clock, don't curate (no history).
        state.last_curation_sim_ts = decision_ts
        return None

    delta = decision_ts - state.last_curation_sim_ts
    if delta < timedelta(minutes=CURATION_INTERVAL_MIN):
        return None

    if not state.pending_decisions:
        # Window elapsed but no new decisions — silent no-op, advance cursor.
        _LOG.info(
            "curator.silent_no_decisions",
            since=state.last_curation_sim_ts.isoformat(),
            now=decision_ts.isoformat(),
        )
        state.last_curation_sim_ts = decision_ts
        return None

    return _run_curation_pass(
        maestro_config=maestro_config,
        maestro_personality=maestro_personality,
        state=state,
        decision_ts=decision_ts,
        repo_root=repo_root,
    )


def _run_curation_pass(
    maestro_config: Any,
    maestro_personality: str,
    state: CuratorState,
    decision_ts: datetime,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Make one API call with recent decisions, apply any write_memory_pattern
    tool uses, return a summary dict."""

    memory_snapshot = _snapshot_all_memories(repo_root)
    user_prompt = _build_curation_prompt(state.pending_decisions, memory_snapshot)

    # Full system prompt = Maestro personality + curation appendix.
    # We deliberately do NOT cache this separately: maestro_personality is
    # already ephemeral-cached upstream, and the appendix is short.
    system_prompt = maestro_personality + CURATION_SYSTEM_APPENDIX

    result = call_structured_multi_tool(
        model=maestro_config.model,
        system_prompt=system_prompt,
        user_content=user_prompt,
        tools=[
            {
                "name": "write_memory_pattern",
                "description": (
                    "Persist (or increment) a learned pattern in one of the "
                    "five memory files. Use sparingly — only for patterns "
                    "with ≥ 2 occurrences and reusable framing."
                ),
                "input_schema": WRITE_MEMORY_PATTERN_SCHEMA,
            }
        ],
        max_tokens=2048,
        mock_tool_calls=_mock_curation_calls(state.pending_decisions),
        agent_slot="maestro.curation",
    )

    n_written = 0
    n_incremented = 0
    failures: list[str] = []
    for call in result.tool_calls:
        if call.get("name") != "write_memory_pattern":
            continue
        try:
            ret = write_memory_pattern(**call["input"], repo_root=repo_root)
            if ret["action"] == "written":
                n_written += 1
            elif ret["action"] == "incremented":
                n_incremented += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
            _LOG.error("curator.tool_call_failed", exc=str(exc), call=call)

    state.total_passes += 1
    state.total_patterns_written += n_written
    state.total_patterns_incremented += n_incremented
    state.total_cost_usd += result.cost_usd

    summary = {
        "pass_idx": state.total_passes,
        "sim_ts": decision_ts.isoformat(),
        "decisions_analyzed": len(state.pending_decisions),
        "patterns_written": n_written,
        "patterns_incremented": n_incremented,
        "failures": failures,
        "cost_usd": round(result.cost_usd, 6),
        "latency_ms": round(result.latency_ms, 2),
        "is_mock": result.is_mock,
    }

    _LOG.info("curator.pass_complete", **summary)

    # Reset buffer + advance cursor
    state.pending_decisions.clear()
    state.last_curation_sim_ts = decision_ts
    return summary


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _snapshot_all_memories(repo_root: Path | None) -> str:
    targets = ["maestro", "voltage", "hashrate", "environment", "power"]
    chunks: list[str] = []
    for t in targets:
        content = load_memory_file(t, repo_root=repo_root)
        chunks.append(f"### {t}_memory.md (current)\n\n```markdown\n{content.strip()}\n```")
    return "\n\n".join(chunks)


def _build_curation_prompt(
    pending_decisions: list[dict[str, Any]], memory_snapshot: str
) -> str:
    """Render the user-side of the curation message."""
    # Summarize each decision compactly. The full reasoning_trace is included
    # verbatim since that is where the patterns live.
    decision_blocks: list[str] = []
    for d in pending_decisions:
        decision_blocks.append(
            json.dumps(
                {
                    "decision_id": d.get("decision_id"),
                    "flag_id": d.get("flag_id"),
                    "miner_id": d.get("miner_id"),
                    "action": d.get("action"),
                    "autonomy_level": d.get("autonomy_level"),
                    "confidence": d.get("confidence"),
                    "consulted_agents": d.get("consulted_agents"),
                    "reasoning_trace": d.get("reasoning_trace"),
                },
                indent=2,
            )
        )

    decisions_text = "\n\n---\n\n".join(decision_blocks)

    return (
        "You are running a curation pass. Review the decisions below and "
        "decide which ones, if any, represent recurring patterns worth "
        "writing to memory.\n\n"
        f"### Decisions in this window ({len(pending_decisions)} total)\n\n"
        f"{decisions_text}\n\n"
        "### Current memory files (for context — do not re-emit existing patterns "
        "unless you are incrementing)\n\n"
        f"{memory_snapshot}\n\n"
        "Apply the rules from your system prompt. Emit zero or more "
        "`write_memory_pattern` tool calls. If nothing warrants a memory "
        "entry, reply with plain text 'nothing to curate' and no tool calls."
    )


# ---------------------------------------------------------------------------
# Mock-mode fallback — deterministic canned curation for offline smoke
# ---------------------------------------------------------------------------


def _mock_curation_calls(
    pending_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic tool-use mocks for offline smoke tests.

    Emits one canned write per unique (action, autonomy_level) pair it sees,
    so the memory files get populated without hitting the API. Real Claude
    curation produces richer entries."""
    if not pending_decisions:
        return []

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for d in pending_decisions:
        key = (d.get("action", ""), d.get("autonomy_level", ""))
        if key in seen:
            continue
        seen[key] = d

    calls: list[dict[str, Any]] = []
    for (action, autonomy), ex in seen.items():
        name = f"mock_{action}_{autonomy.lower()}"
        calls.append(
            {
                "name": "write_memory_pattern",
                "input": {
                    "target_file": "maestro",
                    "pattern_name": name,
                    "signature": (
                        f"[mock] Recurring {action} decisions at {autonomy} — "
                        "canned entry from curator mock mode."
                    ),
                    "verdict_or_action": autonomy,
                    "confidence": 0.5,
                    "reasoning": (
                        "[mock] This entry exists so the smoke test can observe "
                        "memory files being populated without an API call."
                    ),
                    "example_dec_id": ex.get("decision_id", "dec_mock"),
                    "increment_if_exists": True,
                },
            }
        )
    return calls


__all__ = [
    "CURATION_INTERVAL_MIN",
    "CURATION_SYSTEM_APPENDIX",
    "CuratorState",
    "maybe_curate",
]
