"""Pydantic event schemas for MDK Fleet.

See `shared/specs/event_schemas.md` for the authoritative contract. Every
module that produces or consumes events should import models from
`shared.schemas.events` rather than building raw dicts.
"""

from shared.schemas.events import (
    ActionTaken,
    Envelope,
    EpisodicMemoryWrite,
    EventName,
    FleetSnapshot,
    FlagRaised,
    KpiUpdate,
    OrchestratorDecision,
    ReasoningRequest,
    ReasoningResponse,
    Severity,
    Source,
    TelemetryTick,
    parse_event,
)

__all__ = [
    "ActionTaken",
    "Envelope",
    "EpisodicMemoryWrite",
    "EventName",
    "FleetSnapshot",
    "FlagRaised",
    "KpiUpdate",
    "OrchestratorDecision",
    "ReasoningRequest",
    "ReasoningResponse",
    "Severity",
    "Source",
    "TelemetryTick",
    "parse_event",
]
