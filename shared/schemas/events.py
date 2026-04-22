"""Pydantic models for every event flowing through the MDK Fleet bus.

Authoritative contract: `shared/specs/event_schemas.md`.
If this file drifts from the spec, the spec wins — fix this file.

Every event serializes as a common envelope:

    {"event": "<name>", "ts": "<iso8601>", "source": "<producer>", "data": {...}}

Producers build the concrete `<Name>` model (e.g. `TelemetryTick`) and hand it
to `shared.event_bus.write_event`, which wraps it in the envelope and appends
to the right JSONL file. Consumers use `parse_event` to dispatch raw dicts back
to their typed model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared literals
# ---------------------------------------------------------------------------

Severity = Literal["info", "warn", "crit"]
"""Severity ladder used across flags, specialist assessments, and decisions."""

MinerStatus = Literal["ok", "warn", "imm", "shut"]
"""Dashboard miner status tokens. See shared/design/tokens.json#semantic.miner_status."""

OperatingMode = Literal["turbo", "balanced", "eco"]

FlagType = Literal[
    "voltage_drift",
    "hashrate_degradation",
    "thermal_runaway",
    "fan_anomaly",
    "power_instability",
    "chip_variance_high",
    "anomaly_composite",
    "chip_instability_precursor",
    "hashboard_failure_precursor",
]

SourceTool = Literal[
    "xgboost_predictor",
    "isolation_forest_v2",
    "rule_engine",
    "xgboost_rolling_variance",
    "xgboost_thermal_electrical",
]

AgentName = Literal[
    "orchestrator",
    "voltage_agent",
    "hashrate_agent",
    "environment_agent",
    "power_agent",
]

Assessment = Literal["real_signal", "noise", "inconclusive"]

ActionKind = Literal[
    "observe",
    "alert_operator",
    "throttle",
    "migrate_workload",
    "schedule_maintenance",
    "human_review",
    "shutdown",
]

AutonomyLevel = Literal[
    "L1_observe",
    "L2_suggest",
    "L3_bounded_auto",
    "L4_human_only",
]

EventName = Literal[
    "telemetry_tick",
    "kpi_update",
    "fleet_snapshot",
    "flag_raised",
    "reasoning_request",
    "reasoning_response",
    "orchestrator_decision",
    "action_taken",
    "episodic_memory_write",
]

Source = Literal[
    "simulator",
    "ingest",
    "detector",
    "orchestrator",
    "voltage_agent",
    "hashrate_agent",
    "environment_agent",
    "power_agent",
    "action",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


MinerId = Annotated[
    str,
    Field(pattern=r"^m\d{3}$", description="Lowercase miner id, e.g. 'm042'."),
]


class _Base(BaseModel):
    """Common config: forbid extras so typos surface immediately."""

    model_config = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# Event payloads (data only — wrapped in Envelope below)
# ---------------------------------------------------------------------------


class EnvBlock(_Base):
    site_temp_c: float
    site_humidity_pct: float
    elec_price_usd_kwh: float
    hashprice_usd_per_th_day: float


class TelemetryTick(_Base):
    miner_id: MinerId
    miner_model: str
    hashrate_th: float
    hashrate_expected_th: float
    temp_chip_c: float
    temp_amb_c: float
    power_w: float
    voltage_v: float
    fan_rpm: list[int] = Field(min_length=4, max_length=4)
    operating_mode: OperatingMode
    uptime_s: float
    env: EnvBlock
    fault_injected: str | None = None
    """Ground-truth fault tag. Only the simulator writes this. Consumers must
    only read it for A/B labeling — never for detection logic."""


class TeComponents(_Base):
    """TE (v2) is a purely economic ratio; cooling and stability penalties
    moved to HSI. Schema below reflects the simplified formula."""

    model_config = ConfigDict(extra="allow")

    value_usd_day: float
    cost_usd_day: float
    h_eff_th: float
    p_hashprice: float
    p_asic_w: float
    rho_elec: float


class HsiComponents(_Base):
    """HSI (v2) blends four stress components: thermal, voltage, operating
    mode, and hashrate instability. Legacy fields (`thermal_stress`,
    `hashrate_variability`, `hot_time_frac`) are preserved for readability
    and backwards-compat with the canonical replay dataset."""

    model_config = ConfigDict(extra="allow")

    thermal_stress: float
    voltage_stress: float
    mode_stress: float
    instability_stress: float
    hot_time_frac: float


class KpiUpdate(_Base):
    miner_id: MinerId
    te: float
    hsi: float
    te_components: TeComponents
    hsi_components: HsiComponents


class FleetSnapshotMiner(_Base):
    te: float
    hsi: float
    status: MinerStatus
    hashrate_th: float
    temp_chip_c: float


class FleetSnapshot(_Base):
    miner_count: int
    miners: dict[str, FleetSnapshotMiner]
    fleet_te: float
    fleet_hsi: float
    env: EnvBlock


class FlagEvidence(_Base):
    """Free-form evidence block. Per spec most fields are numeric but we leave
    the dict shape flexible because different flaggers surface different
    metrics. Required keys are `metric` and `window_min`."""

    model_config = ConfigDict(extra="allow")

    metric: str
    window_min: float


class FlagRaised(_Base):
    flag_id: str
    miner_id: MinerId
    flag_type: FlagType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    source_tool: SourceTool
    evidence: FlagEvidence
    raw_score: float


class ReasoningContext(_Base):
    """The orchestrator packages everything the specialist needs here. Shape is
    flexible because different specialists want different slices of history.

    The `flag` key mirrors the FlagRaised.data object; the rest is best-effort
    extraction from the relevant streams."""

    model_config = ConfigDict(extra="allow")

    flag: dict[str, Any]


class ReasoningRequest(_Base):
    request_id: str
    flag_id: str
    target_agent: AgentName
    miner_id: MinerId
    question: str
    context: ReasoningContext


class ReasoningResponse(_Base):
    request_id: str
    miner_id: MinerId
    assessment: Assessment
    confidence: float = Field(ge=0.0, le=1.0)
    severity_estimate: Severity
    reasoning: str
    recommended_action_hint: ActionKind
    cost_usd: float = Field(ge=0.0)
    model_used: str
    latency_ms: float = Field(ge=0.0)


class OrchestratorDecision(_Base):
    decision_id: str
    flag_id: str
    miner_id: MinerId
    action: ActionKind
    action_params: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: AutonomyLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trace: str
    consulted_agents: list[AgentName]
    total_cost_usd: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    pending_human_approval: bool = False

    @field_validator("pending_human_approval", mode="after")
    @classmethod
    def _l4_must_pend(cls, v: bool, info: Any) -> bool:  # noqa: ARG003
        # Can't cross-validate in a single-field validator cleanly; enforced in
        # the executor module. Kept here as explicit field for downstream use.
        return v


class ActionTaken(_Base):
    action_id: str
    decision_id: str
    miner_id: MinerId
    action: ActionKind
    status: Literal["executed", "queued_for_human", "rejected", "failed"]
    outcome_expected: str
    outcome_observed: str | None = None
    rollback_ts_scheduled: datetime | None = None


class EpisodicMemoryWrite(_Base):
    memory_id: str
    miner_id: MinerId
    trigger_flag_id: str
    request_id: str
    snapshot: dict[str, Any]
    assessment: Assessment
    reasoning: str
    outcome_followup: str | None = None


# ---------------------------------------------------------------------------
# Envelope + dispatch
# ---------------------------------------------------------------------------


PayloadModel = Union[
    TelemetryTick,
    KpiUpdate,
    FleetSnapshot,
    FlagRaised,
    ReasoningRequest,
    ReasoningResponse,
    OrchestratorDecision,
    ActionTaken,
    EpisodicMemoryWrite,
]


_EVENT_TO_MODEL: dict[str, type[BaseModel]] = {
    "telemetry_tick": TelemetryTick,
    "kpi_update": KpiUpdate,
    "fleet_snapshot": FleetSnapshot,
    "flag_raised": FlagRaised,
    "reasoning_request": ReasoningRequest,
    "reasoning_response": ReasoningResponse,
    "orchestrator_decision": OrchestratorDecision,
    "action_taken": ActionTaken,
    "episodic_memory_write": EpisodicMemoryWrite,
}


class Envelope(_Base):
    """Common wire format. Every JSONL line on every stream is an Envelope."""

    event: EventName
    ts: datetime
    source: Source
    data: dict[str, Any]

    @field_validator("ts", mode="before")
    @classmethod
    def _coerce_ts(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Accept "...Z" and "...+00:00" forms.
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            return datetime.fromisoformat(v)
        return v

    def typed_data(self) -> PayloadModel:
        """Parse `data` into the concrete model matching `event`."""
        model = _EVENT_TO_MODEL[self.event]
        return model.model_validate(self.data)  # type: ignore[return-value]

    @classmethod
    def wrap(
        cls,
        event: EventName,
        source: Source,
        payload: BaseModel | dict[str, Any],
        ts: datetime | None = None,
    ) -> "Envelope":
        """Build an envelope around a payload model (or raw dict)."""
        if isinstance(payload, BaseModel):
            data = payload.model_dump(mode="json")
        else:
            data = dict(payload)
        return cls(
            event=event,
            ts=ts or datetime.now(tz=timezone.utc),
            source=source,
            data=data,
        )


def parse_event(raw: dict[str, Any]) -> tuple[Envelope, PayloadModel]:
    """Parse a raw JSON dict into (envelope, typed_payload).

    Raises pydantic.ValidationError on contract violations.
    """
    env = Envelope.model_validate(raw)
    return env, env.typed_data()
