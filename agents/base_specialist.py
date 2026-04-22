"""Base class for the four specialist agents.

Each specialist (voltage/hashrate/environment/power):
- loads its `.md` personality file once as the system prompt
- handles a `reasoning_request` by retrieving its own recent episodes,
  building the user-facing context, calling the Anthropic API with forced
  tool-use for structured output, and returning a typed
  `reasoning_response`
- writes an `episodic_memory_write` entry to its own JSONL log

Retrieval is subclass-specific: voltage and hashrate key on the same
miner's past shape/trajectory, environment keys on site/zone, power keys
on rack/PDU + tariff window. The base class provides a sensible default
(recent-by-miner) that subclasses override.

The tool-use schema is identical across specialists — all four emit the
same `reasoning_response.data` shape per `shared/specs/event_schemas.md`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import structlog

from agents._client import LLMResult, call_structured
from agents.config import AgentConfig
from shared.event_bus import read_events, write_event
from shared.paths import ensure_memory_dir
from shared.schemas.events import (
    Assessment,
    EpisodicMemoryWrite,
    ReasoningRequest,
    ReasoningResponse,
    Severity,
)

_LOG = structlog.get_logger(__name__)


# Forced tool-use schema that every specialist uses. Mirrors
# shared/specs/event_schemas.md#reasoning_response so the output is
# directly Pydantic-validatable.
SPECIALIST_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["real_signal", "noise", "inconclusive"],
            "description": "Your judgement on whether the flag reflects a real pre-failure signal.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Calibrated confidence in the assessment.",
        },
        "severity_estimate": {
            "type": "string",
            "enum": ["info", "warn", "crit"],
            "description": "How severe you judge this to be.",
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Compact 2-4 sentence narrative naming the pattern, the comparison to baseline, "
                "and the matching memory if any. Write like an SRE Slack message, not a report."
            ),
        },
        "recommended_action_hint": {
            "type": "string",
            "enum": [
                "observe",
                "alert_operator",
                "throttle",
                "migrate_workload",
                "schedule_maintenance",
                "human_review",
                "shutdown",
            ],
            "description": "Action suggestion for the orchestrator. It may override.",
        },
    },
    "required": [
        "assessment",
        "confidence",
        "severity_estimate",
        "reasoning",
        "recommended_action_hint",
    ],
}


class BaseSpecialist:
    """Abstract base for the four domain specialists.

    Subclasses typically only override `_build_user_prompt` (to surface
    the domain-relevant slice of context) and `_retrieve_episodes` (to
    weight retrieval by domain-specific similarity). Everything else —
    LLM call, schema validation, memory write — is shared.
    """

    # Map agent-config name → memory file target key (for agents.tools).
    # Non-specialist agents (orchestrator) are handled by Maestro class.
    _MEMORY_TARGETS: dict[str, str] = {
        "voltage_agent": "voltage",
        "hashrate_agent": "hashrate",
        "environment_agent": "environment",
        "power_agent": "power",
    }

    def __init__(
        self,
        config: AgentConfig,
        max_history_events: int = 3,
        max_tokens: int = 1024,
    ) -> None:
        self.config = config
        self.max_history_events = max_history_events
        self.max_tokens = max_tokens

        md_path = Path(config.personality_md_path)
        self._personality = md_path.read_text(encoding="utf-8")

        self._memory_dir = ensure_memory_dir(config.name)
        self._events_path = self._memory_dir / "events.jsonl"
        self._memory_target = self._MEMORY_TARGETS.get(config.name)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def handle_request(self, request: ReasoningRequest) -> ReasoningResponse:
        """Process a reasoning_request and return a reasoning_response."""
        episodes = self._retrieve_episodes(request)
        user_prompt = self._build_user_prompt(request, episodes)

        # Load the domain memory fresh at every call — ephemeral cache on
        # the Anthropic side keeps this cheap, and a post-curation update
        # is immediately visible here.
        system_prompt = self._personality
        if self._memory_target:
            from agents.tools import load_memory_file

            memory_md = load_memory_file(self._memory_target)
            if memory_md.strip():
                system_prompt = (
                    self._personality
                    + "\n\n---\n\n# Curated memory (updated by Maestro)\n\n"
                    + memory_md.strip()
                )

        # Map agent name → routing slot. Specialists use
        # `specialists.<domain>` so the YAML can route them independently
        # (hybrid_economic profile sends specialists to Ollama while Maestro
        # stays on Anthropic).
        _SPEC_SLOT = {
            "voltage_agent": "specialists.voltage",
            "hashrate_agent": "specialists.hashrate",
            "environment_agent": "specialists.environment",
            "power_agent": "specialists.power",
        }
        slot = _SPEC_SLOT.get(self.config.name)

        result = call_structured(
            model=self.config.model,
            system_prompt=system_prompt,
            user_content=user_prompt,
            tool_name="submit_assessment",
            tool_description=(
                "Submit your structured assessment of the flag. "
                "All fields required. Be concise and calibrated."
            ),
            tool_schema=SPECIALIST_TOOL_SCHEMA,
            max_tokens=self.max_tokens,
            mock_fallback=self._mock_response(request, episodes),
            agent_slot=slot,
        )

        response = self._build_response(request, result)
        self._write_episodic(request, response)

        _LOG.info(
            "specialist_response",
            agent=self.config.name,
            miner_id=request.miner_id,
            flag_id=request.flag_id,
            assessment=response.assessment,
            confidence=response.confidence,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            is_mock=result.is_mock,
        )
        return response

    # ------------------------------------------------------------------
    # overridable hooks
    # ------------------------------------------------------------------

    def _retrieve_episodes(self, request: ReasoningRequest) -> list[dict[str, Any]]:
        """Default retrieval: most recent episodes for the same miner.

        Subclasses override to weight by domain-specific similarity.
        """
        if not self._events_path.exists():
            return []
        collected: list[dict[str, Any]] = []
        for env in read_events(self._events_path):
            data = env.data
            if data.get("miner_id") == request.miner_id:
                collected.append(data)
        return collected[-self.max_history_events :]

    def _build_user_prompt(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> str:
        """Compose the user-facing prompt. Override to tune domain framing.

        Default includes: the orchestrator's question, the flag payload, the
        context block as-is, and any retrieved episodes.
        """
        ctx_dict = request.context.model_dump(mode="json")
        episodes_list = list(episodes)
        episodes_block = (
            json.dumps(episodes_list, indent=2, default=str)
            if episodes_list
            else "No matching past episodes."
        )
        return (
            f"Orchestrator question: {request.question}\n\n"
            f"Flag and context (authoritative):\n"
            f"{json.dumps(ctx_dict, indent=2, default=str)}\n\n"
            f"Your episodic memory — top {self.max_history_events} most relevant past events:\n"
            f"{episodes_block}\n\n"
            f"Submit your assessment via the `submit_assessment` tool."
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_response(
        self, request: ReasoningRequest, result: LLMResult
    ) -> ReasoningResponse:
        ti = result.tool_input or {}
        assessment: Assessment = ti.get("assessment") or "inconclusive"  # type: ignore[assignment]
        confidence = float(ti.get("confidence", 0.0))
        severity: Severity = ti.get("severity_estimate") or "info"  # type: ignore[assignment]
        reasoning = ti.get("reasoning") or "(no reasoning returned)"
        action_hint = ti.get("recommended_action_hint") or "observe"

        return ReasoningResponse(
            request_id=request.request_id,
            miner_id=request.miner_id,
            assessment=assessment,
            confidence=max(0.0, min(1.0, confidence)),
            severity_estimate=severity,
            reasoning=reasoning,
            recommended_action_hint=action_hint,
            cost_usd=result.cost_usd,
            model_used=result.model,
            latency_ms=result.latency_ms,
        )

    def _write_episodic(self, request: ReasoningRequest, response: ReasoningResponse) -> None:
        memory = EpisodicMemoryWrite(
            memory_id=f"mem_{self.config.name}_{request.request_id}",
            miner_id=request.miner_id,
            trigger_flag_id=request.flag_id,
            request_id=request.request_id,
            snapshot={"flag": request.context.flag},
            assessment=response.assessment,
            reasoning=response.reasoning,
            outcome_followup=None,
        )
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(memory.model_dump_json() + "\n")
        # Also stream to the live bus so the dashboard can see agent activity.
        write_event(
            "episodic_memory_write",
            self.config.name,  # type: ignore[arg-type]
            memory,
        )

    def _mock_response(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> dict[str, Any]:
        """Deterministic canned response for mock mode.

        Subclasses may override to inject domain flavor. The default does a
        reasonable best-effort based on flag severity + confidence so the
        end-to-end smoke produces sensible traces without any API key.
        """
        flag = request.context.flag
        flag_conf = float(flag.get("confidence", 0.5))
        flag_sev = flag.get("severity", "info")
        assessment = "real_signal" if flag_conf >= 0.7 else ("inconclusive" if flag_conf >= 0.4 else "noise")
        action = {
            "info": "observe",
            "warn": "alert_operator",
            "crit": "throttle",
        }.get(flag_sev, "observe")
        episodes_list = list(episodes)
        return {
            "assessment": assessment,
            "confidence": round(max(0.3, min(0.9, flag_conf)), 2),
            "severity_estimate": flag_sev,
            "reasoning": (
                f"[mock/{self.config.name}] flag={flag.get('flag_type')} on {request.miner_id} "
                f"confidence={flag_conf:.2f}, severity={flag_sev}. "
                f"{len(episodes_list)} past episode(s) retrieved."
            ),
            "recommended_action_hint": action,
        }
