"""Power specialist — rack/PDU scope, tariff window alignment, long memory."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agents.base_specialist import BaseSpecialist
from shared.event_bus import read_events
from shared.schemas.events import ReasoningRequest


class PowerAgent(BaseSpecialist):
    """90-day memory retention, rack/PDU scope (prototype: fleet-wide stand-in)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Bump default retrieval budget: grid events are sparse but
        # recurring, so we want to see more of our own history than the
        # other specialists do.
        kwargs.setdefault("max_history_events", 5)
        super().__init__(*args, **kwargs)

    def _retrieve_episodes(self, request: ReasoningRequest) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        # Prototype: return most recent events regardless of miner — the
        # point is to surface recurring patterns at the site/tariff-window
        # level, not per-miner specificity. Replace with a site_zone +
        # tariff_window key once topology lands.
        events = [env.data for env in read_events(self._events_path)]
        return events[-self.max_history_events :]

    def _build_user_prompt(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> str:
        base = super()._build_user_prompt(request, episodes)
        return (
            "Focus: scope first, magnitude second. Is this one miner or the whole rack? "
            "Is the event aligned to a peak-tariff window? Have I seen this exact pattern "
            "recurring at this site? Read the peer telemetry in the context block.\n\n"
            f"{base}\n\n"
            "Reminder — per-chip voltage mechanics are voltage_agent's territory. "
            "Your whole value is preventing wrong miner-level throttles for upstream events."
        )
