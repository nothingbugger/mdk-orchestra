"""Hashrate specialist — reads trajectory shape against miner's own history."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agents.base_specialist import BaseSpecialist
from shared.event_bus import read_events
from shared.schemas.events import ReasoningRequest


class HashrateAgent(BaseSpecialist):
    """Retrieval keyed on same-miner past trajectory events."""

    def _retrieve_episodes(self, request: ReasoningRequest) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        collected: list[dict[str, Any]] = []
        for env in read_events(self._events_path):
            if env.data.get("miner_id") == request.miner_id:
                collected.append(env.data)
        return collected[-self.max_history_events :]

    def _build_user_prompt(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> str:
        base = super()._build_user_prompt(request, episodes)
        return (
            "Focus: classify the trajectory shape (step / ramp / sawtooth / stall / stationary). "
            "Think in percentages vs the miner's own baseline, not absolute TH/s. "
            "Check temp correlation before calling sawtooth; check shareback before blaming chip.\n\n"
            f"{base}\n\n"
            "Reminder — voltage mechanics belong to voltage_agent; site-level power to power_agent; "
            "ambient causal stories to environment_agent. You tell the shape of the curve."
        )
