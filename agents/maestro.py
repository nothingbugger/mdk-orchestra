"""Maestro — the conductor of Orchestra.

Consumes `flag_raised` events, dispatches each to the relevant specialist
subset (per the table in `agents/maestro.md`), calls them in parallel,
synthesizes their responses into an `orchestrator_decision` via one Opus
call with forced tool-use, and emits the decision to the bus.

The dispatch table here is the Python mirror of the Markdown table in
`maestro.md`. Keep them in sync — the `.md` is the narrative, this dict
is the enforcement point. When a flag_type is absent from the table we
fall back to `all_specialists`; the orchestrator LLM is responsible for
downweighting that via the usual synthesis rules.

Cost profile (rough): for one consultation we spend
- 1 Opus synthesis call (system = maestro.md, cache hit after first)
- N specialist calls (N = 2 to 4 depending on flag type)
Total is 3–5 model calls per flag.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import structlog

from agents._client import call_structured
from agents._history import FleetHistoryBuffer
from agents.base_specialist import BaseSpecialist
from agents.config import DEFAULT_AGENT_CONFIGS, AgentConfig
from agents.environment_agent import EnvironmentAgent
from agents.hashrate_agent import HashrateAgent
from agents.power_agent import PowerAgent
from agents.voltage_agent import VoltageAgent
from shared.event_bus import tail_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import (
    ActionKind,
    AgentName,
    AutonomyLevel,
    FlagType,
    OrchestratorDecision,
    ReasoningContext,
    ReasoningRequest,
    ReasoningResponse,
)

_LOG = structlog.get_logger(__name__)


# Dispatch table — keep in sync with maestro.md.
#
# Cost-optimized schema: each flag_type maps to (primary, fallback).
# - PRIMARY is always consulted first.
# - FALLBACK is consulted ONLY when primary returns `inconclusive`. When primary
#   returns `real_signal` or `noise` we skip fallback entirely — one specialist
#   call is enough for a definitive verdict.
# - `anomaly_composite` is the exception: it dispatches to all four specialists
#   (Flow 3) because the origin is ambiguous by construction.
DISPATCH_TABLE: dict[str, tuple[str, str | None]] = {
    "voltage_drift":                ("voltage_agent",      "power_agent"),
    "hashrate_degradation":         ("hashrate_agent",     "voltage_agent"),
    "chip_instability_precursor":   ("hashrate_agent",     "voltage_agent"),
    "hashboard_failure_precursor":  ("hashrate_agent",     "voltage_agent"),
    "thermal_runaway":              ("environment_agent",  "voltage_agent"),
    "fan_anomaly":                  ("environment_agent",  None),
    "power_instability":            ("power_agent",        "voltage_agent"),
    "chip_variance_high":           ("voltage_agent",      "hashrate_agent"),
}

_ALL_SPECIALISTS = [
    "voltage_agent",
    "hashrate_agent",
    "environment_agent",
    "power_agent",
]

# Flow 3 — anomaly_composite consults every specialist in parallel.
_FLOW3_FLAGS = {"anomaly_composite"}


# Model routing for tiered synthesis.
# First-pass synthesis runs on a cheap model. If the first pass concludes an
# action that needs L3 or L4 authority, we re-synthesize on Opus with the
# first-pass reasoning as context (second-opinion pattern). L1/L2 decisions
# do NOT escalate — they're cheap-to-get-right cases.
_MAESTRO_FIRST_PASS_MODEL: str = os.environ.get(
    "MDK_MAESTRO_FIRST_PASS_MODEL", "claude-sonnet-4-6"
)
_MAESTRO_ESCALATION_AUTONOMY = {"L3_bounded_auto", "L4_human_only"}
_MAESTRO_ESCALATION_ACTIONS = {"throttle", "migrate_workload", "shutdown", "human_review"}


# Forced tool-use schema for Maestro's synthesis call.
DECISION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
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
        },
        "action_params": {
            "type": "object",
            "description": (
                "Free-form params for the action. Recommended keys for throttle: "
                "target_hashrate_pct (0.70-1.00), duration_min. For observe/alert: "
                "can be empty."
            ),
        },
        "autonomy_level": {
            "type": "string",
            "enum": ["L1_observe", "L2_suggest", "L3_bounded_auto", "L4_human_only"],
            "description": (
                "L1: log only. L2: alert operator, no auto action. "
                "L3: execute (reversible, rate-limited only). L4: queue for human approval."
            ),
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_trace": {
            "type": "string",
            "description": (
                "Compact SRE-style narrative: miner_id, flag summary, per-specialist verdicts, "
                "synthesis, action justification. No filler."
            ),
        },
    },
    "required": ["action", "autonomy_level", "confidence", "reasoning_trace"],
}


def _decision_sim_ts(decision: OrchestratorDecision) -> datetime:
    """Simulator timestamp for a decision. OrchestratorDecision doesn't carry
    its own envelope ts, so we derive from decision_id-adjacent data: the
    decision is emitted "now" in sim time, matching when the flag arrived.
    For the curator it's enough to use utcnow() since the flag→decision gap
    is << 30 min sim. If running at 10x speed the wall→sim drift is 1:10,
    which still keeps pass alignment within a handful of seconds sim-time."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Maestro
# ---------------------------------------------------------------------------


class Maestro:
    """Orchestrator: dispatch → consult → synthesize → emit."""

    def __init__(
        self,
        agent_configs: dict[str, AgentConfig] | None = None,
        history_buffer: FleetHistoryBuffer | None = None,
        max_parallel_agents: int = 4,
    ) -> None:
        self.configs = agent_configs or DEFAULT_AGENT_CONFIGS
        self.maestro_config = self.configs["orchestrator"]
        self._personality = Path(self.maestro_config.personality_md_path).read_text(
            encoding="utf-8"
        )
        self.specialists: dict[str, BaseSpecialist] = _build_specialists(self.configs)
        self.history = history_buffer or FleetHistoryBuffer()
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_agents)

        # Curator state — tracks simulator time + decision buffer since the
        # last curation pass. Fires `maybe_curate` after every dispatch.
        from agents.curator import CuratorState  # local import to avoid cycle

        self._curator_state = CuratorState()

    def start(self) -> None:
        self.history.start()

    def stop(self) -> None:
        self.history.stop()
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def dispatch_flag(self, flag: dict[str, Any]) -> OrchestratorDecision:
        """Handle one flag end-to-end. Returns the decision (also emitted to the bus).

        Dispatch strategy (cost-optimized):
        - `anomaly_composite`: Flow 3 — all four specialists in parallel.
        - All other flag types: primary specialist first. Consult fallback
          ONLY if primary returns `inconclusive` — definitive `real_signal`
          or `noise` from the primary is enough to skip the second call.
        """
        t0 = time.monotonic()
        flag_type = flag.get("flag_type", "anomaly_composite")

        if flag_type in _FLOW3_FLAGS:
            responses = self._consult(_ALL_SPECIALISTS, flag)
        else:
            primary, fallback = DISPATCH_TABLE.get(flag_type, (_ALL_SPECIALISTS[0], None))
            primary_responses = self._consult([primary], flag)
            responses = primary_responses

            primary_verdict = primary_responses[0].assessment if primary_responses else "inconclusive"
            if primary_verdict == "inconclusive" and fallback:
                fallback_responses = self._consult([fallback], flag)
                responses = primary_responses + fallback_responses

        decision = self._synthesize(flag, responses, t0)

        write_event("orchestrator_decision", "orchestrator", decision)
        _LOG.info(
            "decision_emitted",
            decision_id=decision.decision_id,
            flag_id=decision.flag_id,
            miner_id=decision.miner_id,
            action=decision.action,
            autonomy=decision.autonomy_level,
            confidence=decision.confidence,
            consulted=decision.consulted_agents,
            cost_usd=decision.total_cost_usd,
            latency_ms=decision.total_latency_ms,
        )

        # Record the decision in the curator buffer and fire a curation pass
        # if the sim-time window has elapsed.
        self._curator_state.record_decision(decision.model_dump(mode="json"))
        from agents.curator import maybe_curate

        try:
            maybe_curate(
                maestro_config=self.maestro_config,
                maestro_personality=self._personality,
                state=self._curator_state,
                decision_ts=_decision_sim_ts(decision),
            )
        except Exception as exc:  # noqa: BLE001 — never let curation kill the main loop
            _LOG.error("curator.pass_failed", exc=str(exc))

        return decision

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _consult(
        self, agent_names: list[str], flag: dict[str, Any]
    ) -> list[ReasoningResponse]:
        """Fire requests to multiple specialists in parallel, collect responses."""
        futures = []
        for name in agent_names:
            specialist = self.specialists.get(name)
            if not specialist or not self.configs[name].enabled:
                continue
            request = self._build_request(name, flag)
            futures.append((name, self._executor.submit(specialist.handle_request, request)))

        responses: list[ReasoningResponse] = []
        for name, fut in futures:
            try:
                responses.append(fut.result(timeout=60))
            except Exception as exc:  # noqa: BLE001 — log, continue, do not poison consultation
                _LOG.error("specialist_failed", agent=name, exc=str(exc))
        return responses

    def _build_request(self, target_agent: str, flag: dict[str, Any]) -> ReasoningRequest:
        miner_id = flag["miner_id"]
        recent_telemetry = self.history.recent_telemetry(miner_id, minutes=30.0)
        recent_kpis = self.history.recent_kpis(miner_id, minutes=30.0)
        context_extras: dict[str, Any] = {
            "miner_recent_telemetry_30min": recent_telemetry[-60:],  # cap size
            "miner_recent_kpis_30min": recent_kpis[-60:],
        }
        if target_agent in ("environment_agent", "power_agent"):
            context_extras["zone_peers_30min"] = {
                mid: samples[-12:]
                for mid, samples in self.history.zone_peers(miner_id, minutes=30.0).items()
            }

        question = _QUESTIONS.get(
            target_agent,
            "Assess whether this flag reflects a real pre-failure signal given the context.",
        )

        return ReasoningRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            flag_id=flag["flag_id"],
            target_agent=target_agent,  # type: ignore[arg-type]
            miner_id=miner_id,
            question=question,
            context=ReasoningContext.model_validate({"flag": flag, **context_extras}),
        )

    def _needs_tiebreaker(self, responses: list[ReasoningResponse]) -> bool:
        if not responses:
            return False
        if any(r.assessment == "inconclusive" for r in responses):
            return True
        confs = [r.confidence for r in responses]
        if not confs:
            return False
        # Proxy for disagreement: spread > 0.25 triggers optional consult.
        return (max(confs) - min(confs)) > 0.25

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------

    def _maestro_system_prompt(self) -> str:
        """Full Maestro system prompt = personality + curated memory.

        The memory file is re-read at every call so that newly-curated
        patterns are immediately visible in the next dispatch. Ephemeral
        prompt cache on the Anthropic side keeps the cost low — the cache
        only invalidates when the memory file actually changes.
        """
        from agents.tools import load_memory_file

        memory_md = load_memory_file("maestro")
        if memory_md.strip():
            return (
                self._personality
                + "\n\n---\n\n# Curated memory (self-authored)\n\n"
                + memory_md.strip()
            )
        return self._personality

    def _synthesize(
        self,
        flag: dict[str, Any],
        responses: list[ReasoningResponse],
        t0: float,
    ) -> OrchestratorDecision:
        """Tiered synthesis: Sonnet first-pass; Opus second-opinion only if
        the first pass concludes L3/L4 (consequential action).

        Cost profile per decision:
        - L1/L2 outcome: one Sonnet call  (~$0.03-0.05 + specialist cost)
        - L3/L4 outcome: Sonnet + Opus second opinion  (~$0.10-0.15 + specialist cost)

        The Opus pass receives the Sonnet synthesis as context and either
        confirms or revises. Both traces are logged; the returned decision
        carries the Opus trace (or the Sonnet trace if no escalation ran).
        """
        system_prompt = self._maestro_system_prompt()
        user_prompt = self._build_synthesis_prompt(flag, responses)

        first = call_structured(
            model=_MAESTRO_FIRST_PASS_MODEL,
            system_prompt=system_prompt,
            user_content=user_prompt,
            tool_name="submit_decision",
            tool_description=(
                "Submit the final orchestrator decision. Respect the autonomy ladder: "
                "L3 only for reversible/rate-limited actions; L4 for anything "
                "consequential. Be concise in reasoning_trace."
            ),
            tool_schema=DECISION_TOOL_SCHEMA,
            max_tokens=1024,
            mock_fallback=self._mock_decision(flag, responses),
            agent_slot="maestro.dispatch",
        )
        first_ti = first.tool_input or {}
        first_autonomy = first_ti.get("autonomy_level") or "L2_suggest"
        first_action = first_ti.get("action") or "alert_operator"

        needs_escalation = (
            first_autonomy in _MAESTRO_ESCALATION_AUTONOMY
            or first_action in _MAESTRO_ESCALATION_ACTIONS
        )

        chosen = first
        escalation_cost = 0.0
        escalation_latency = 0.0
        if needs_escalation:
            _LOG.info(
                "maestro.escalating_to_opus",
                first_autonomy=first_autonomy,
                first_action=first_action,
                first_model=first.model,
                flag_id=flag.get("flag_id"),
                miner_id=flag.get("miner_id"),
            )
            second_user_prompt = (
                user_prompt
                + "\n\n---\n\n"
                + "## First-pass synthesis (Sonnet) — awaiting your second opinion\n\n"
                + f"The first-pass model proposed action=`{first_action}`, autonomy="
                + f"`{first_autonomy}`, confidence={first_ti.get('confidence', 'n/a')}.\n\n"
                + "Its reasoning trace:\n\n> "
                + (first_ti.get("reasoning_trace") or "(empty)").replace("\n", "\n> ")
                + "\n\nReview this first-pass decision. If you agree, emit the same "
                + "`submit_decision` tool call (you may tighten the reasoning_trace). "
                + "If you disagree — especially on the autonomy level for this miner's "
                + "current state — override with your own call and justify the change. "
                + "Remember: L3 is for reversible capped actions; L4 is mandatory for "
                + "anything the operator should approve."
            )
            second = call_structured(
                model=self.maestro_config.model,  # Opus
                system_prompt=system_prompt,
                user_content=second_user_prompt,
                tool_name="submit_decision",
                tool_description=(
                    "Second-opinion synthesis. Confirm or revise the first-pass decision."
                ),
                tool_schema=DECISION_TOOL_SCHEMA,
                max_tokens=1024,
                mock_fallback=self._mock_decision(flag, responses),
                agent_slot="maestro.escalation",
            )
            chosen = second
            escalation_cost = second.cost_usd
            escalation_latency = second.latency_ms

        ti = chosen.tool_input or {}
        action: ActionKind = ti.get("action") or "alert_operator"  # type: ignore[assignment]
        autonomy: AutonomyLevel = ti.get("autonomy_level") or "L2_suggest"  # type: ignore[assignment]
        confidence = float(ti.get("confidence", 0.5))
        trace = ti.get("reasoning_trace") or "(no trace returned)"
        action_params = ti.get("action_params") or {}
        if not isinstance(action_params, dict):
            action_params = {}

        total_cost = (
            first.cost_usd + escalation_cost + sum(r.cost_usd for r in responses)
        )
        total_latency = (time.monotonic() - t0) * 1000.0
        consulted: list[AgentName] = [r.request_id and "voltage_agent" for r in responses]  # placeholder
        # Reassign consulted properly based on which responses we got —
        # the response object carries model_used but not the agent name,
        # so we infer from order of consultation.
        consulted = _infer_consulted(responses, self.specialists)

        return OrchestratorDecision(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            flag_id=flag["flag_id"],
            miner_id=flag["miner_id"],
            action=action,
            action_params=action_params,
            autonomy_level=autonomy,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning_trace=trace,
            consulted_agents=consulted,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            pending_human_approval=(autonomy == "L4_human_only"),
        )

    def _build_synthesis_prompt(
        self, flag: dict[str, Any], responses: list[ReasoningResponse]
    ) -> str:
        responses_block = [
            {
                "specialist_model": r.model_used,
                "assessment": r.assessment,
                "confidence": r.confidence,
                "severity_estimate": r.severity_estimate,
                "reasoning": r.reasoning,
                "recommended_action_hint": r.recommended_action_hint,
            }
            for r in responses
        ]
        return (
            "You are Maestro. A new flag has arrived. The specialists have reported. "
            "Apply your operating principles (reversibility first, autonomy ladder, "
            "listen to the specialists, cost-aware).\n\n"
            f"Flag:\n{json.dumps(flag, indent=2, default=str)}\n\n"
            f"Specialist responses ({len(responses)}):\n"
            f"{json.dumps(responses_block, indent=2, default=str)}\n\n"
            "Synthesize and submit your decision via the `submit_decision` tool. "
            "Your reasoning_trace must cite each specialist's assessment in one "
            "compact narrative."
        )

    def _mock_decision(
        self, flag: dict[str, Any], responses: list[ReasoningResponse]
    ) -> dict[str, Any]:
        """Deterministic synthesis for mock mode — mirrors maestro.md rules."""
        if not responses:
            return {
                "action": "observe",
                "action_params": {},
                "autonomy_level": "L1_observe",
                "confidence": 0.4,
                "reasoning_trace": (
                    f"[mock] No specialist responses available for {flag.get('miner_id')}. "
                    "Defaulting to observe."
                ),
            }

        real_count = sum(1 for r in responses if r.assessment == "real_signal")
        total = len(responses)
        avg_conf = sum(r.confidence for r in responses) / total

        if real_count / total >= 0.67 and any(r.severity_estimate == "crit" for r in responses):
            action = "throttle"
            autonomy = "L3_bounded_auto"
            params = {"target_hashrate_pct": 0.80, "duration_min": 60}
        elif real_count / total >= 0.5:
            action = "alert_operator"
            autonomy = "L2_suggest"
            params = {}
        else:
            action = "observe"
            autonomy = "L1_observe"
            params = {}

        trace_parts = [
            f"{r.model_used} {r.assessment} conf={r.confidence:.2f} ({r.reasoning[:80]})"
            for r in responses
        ]
        trace = (
            f"[mock] {flag.get('miner_id')} {flag.get('flag_type')} "
            f"{flag.get('severity')}/{flag.get('confidence'):.2f}. "
            + " | ".join(trace_parts)
            + f" Synthesis: {real_count}/{total} real, avg_conf={avg_conf:.2f} → {action}."
        )
        return {
            "action": action,
            "action_params": params,
            "autonomy_level": autonomy,
            "confidence": round(avg_conf, 2),
            "reasoning_trace": trace,
        }


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


_QUESTIONS: dict[str, str] = {
    "voltage_agent": (
        "Assess whether the voltage pattern reflects a real pre-failure signal "
        "or noise, given the miner's own history."
    ),
    "hashrate_agent": (
        "Classify the hashrate trajectory shape and judge whether this is a real "
        "degradation or normal variance."
    ),
    "environment_agent": (
        "Is this event miner-local or site-wide? Check ambient, time-of-day, and "
        "cross-miner correlation."
    ),
    "power_agent": (
        "Is this event scoped to one miner or to a rack / site? Check tariff alignment "
        "and recurring patterns in your memory."
    ),
}


def _build_specialists(configs: dict[str, AgentConfig]) -> dict[str, BaseSpecialist]:
    specialists: dict[str, BaseSpecialist] = {}
    if "voltage_agent" in configs and configs["voltage_agent"].enabled:
        specialists["voltage_agent"] = VoltageAgent(configs["voltage_agent"])
    if "hashrate_agent" in configs and configs["hashrate_agent"].enabled:
        specialists["hashrate_agent"] = HashrateAgent(configs["hashrate_agent"])
    if "environment_agent" in configs and configs["environment_agent"].enabled:
        specialists["environment_agent"] = EnvironmentAgent(configs["environment_agent"])
    if "power_agent" in configs and configs["power_agent"].enabled:
        specialists["power_agent"] = PowerAgent(configs["power_agent"])
    return specialists


def _infer_consulted(
    responses: list[ReasoningResponse], specialists: dict[str, BaseSpecialist]
) -> list[AgentName]:
    """Map ReasoningResponse.model_used back to the agent name.

    Two specialists on the same model (voltage + hashrate + power all run
    Sonnet) are not distinguishable by model alone, so we walk the
    specialists dict and attribute responses in creation order. Good
    enough for the prototype — a richer implementation would stamp the
    agent name onto the response itself.
    """
    by_model: dict[str, list[str]] = {}
    for name, spec in specialists.items():
        by_model.setdefault(spec.config.model, []).append(name)
    out: list[AgentName] = []
    cursor: dict[str, int] = {}
    for r in responses:
        pool = by_model.get(r.model_used, [])
        idx = cursor.get(r.model_used, 0)
        if idx < len(pool):
            out.append(pool[idx])  # type: ignore[arg-type]
            cursor[r.model_used] = idx + 1
        else:
            out.append("orchestrator")  # fallback
    return out


def run_orchestrator(
    flag_stream: str | Path | None = None,
    from_start: bool = False,
    stop_after: float | None = None,
    max_flags: int | None = None,
) -> None:
    """Main loop. Tails `flag_stream`, dispatches each flag, emits decisions.

    Args:
        flag_stream: path to the flags JSONL. Defaults to canonical stream.
        from_start: if True, replay existing flags before following.
        stop_after: seconds — stop after this wall-clock duration (None = forever).
        max_flags: stop after processing this many flags (None = no cap).
    """
    path = Path(flag_stream) if flag_stream else stream_paths().flags
    maestro = Maestro()
    maestro.start()
    processed = 0
    try:
        for env in tail_events(path, from_start=from_start, stop_after=stop_after):
            if env.event != "flag_raised":
                continue
            maestro.dispatch_flag(env.data)
            processed += 1
            if max_flags is not None and processed >= max_flags:
                return
    finally:
        maestro.stop()
