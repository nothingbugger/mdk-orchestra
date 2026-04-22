"""Environment specialist — site/zone scope, cluster correlation, HVAC signatures."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agents.base_specialist import BaseSpecialist
from shared.event_bus import read_events
from shared.schemas.events import ReasoningRequest


class EnvironmentAgent(BaseSpecialist):
    """Retrieval is site-scoped, not miner-scoped. Different from others."""

    def _retrieve_episodes(self, request: ReasoningRequest) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        # Prototype: we don't yet have explicit zones, so recent fleet-wide
        # env events stand in. Replace with hour-of-day bucket + zone key
        # once the simulator exposes topology.
        all_events = [env.data for env in read_events(self._events_path)]
        return all_events[-self.max_history_events :]

    def _build_user_prompt(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> str:
        base = super()._build_user_prompt(request, episodes)
        return (
            "Focus: zoom out. Is ambient temp drifting vs time-of-day baseline? "
            "Are other miners in the zone moving together (cluster score)? "
            "Is the event aligned to a tariff peak window?\n\n"
            f"{base}\n\n"
            "Keep it brief. You're on Haiku — two sentences of reasoning is usually enough. "
            "Your verdict bounds at alert_operator; miner-level throttling is not your call."
        )
