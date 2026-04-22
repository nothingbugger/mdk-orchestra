"""Voltage specialist — reads voltage patterns against the miner's own baseline."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from agents.base_specialist import BaseSpecialist
from shared.event_bus import read_events
from shared.schemas.events import ReasoningRequest


class VoltageAgent(BaseSpecialist):
    """Same-miner-first retrieval: voltage patterns repeat on the same PSU."""

    def _retrieve_episodes(self, request: ReasoningRequest) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []

        same_miner: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []
        flag_type = request.context.flag.get("flag_type")

        for env in read_events(self._events_path):
            data = env.data
            snapshot = data.get("snapshot") or {}
            episode_flag = snapshot.get("flag") or {}
            if data.get("miner_id") == request.miner_id:
                same_miner.append(data)
            elif episode_flag.get("flag_type") == flag_type:
                other.append(data)

        ranked = same_miner[::-1][: self.max_history_events] or other[::-1][: self.max_history_events]
        return ranked

    def _build_user_prompt(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> str:
        ctx = request.context.model_dump(mode="json")
        base = super()._build_user_prompt(request, episodes)
        return (
            "Focus: disentangle thermal drift, capacitor wear, and upstream grid effects "
            "on voltage. Name the shape (step / ramp / oscillation / stationary) and cite "
            "mV figures when possible.\n\n"
            f"{base}\n\n"
            f"Reminder — your domain is the PSU and the 12V rail as delivered to the hashboards. "
            f"Fleet-wide/rack-level voltage context belongs to the power_agent."
        )

    def _mock_response(
        self, request: ReasoningRequest, episodes: Iterable[dict[str, Any]]
    ) -> dict[str, Any]:
        base = super()._mock_response(request, episodes)
        flag = request.context.flag
        metric = flag.get("evidence", {}).get("metric", "voltage_v")
        base["reasoning"] = (
            f"[mock/voltage] {request.miner_id} {metric}: "
            f"flag_confidence={flag.get('confidence'):.2f}. "
            "Pattern shape unknown (mock mode). Passing through severity."
        )
        return base
