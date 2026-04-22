"""Agents layer — Maestro (conductor) + specialist fleet.

Public API for the orchestrator and its specialists. Pydantic events come
from `shared.schemas.events`; the bus comes from `shared.event_bus`. This
package wires those to the Anthropic API and to per-agent episodic memory.
"""

from agents.base_specialist import BaseSpecialist
from agents.config import DEFAULT_AGENT_CONFIGS, AgentConfig
from agents.environment_agent import EnvironmentAgent
from agents.hashrate_agent import HashrateAgent
from agents.maestro import Maestro, run_orchestrator
from agents.power_agent import PowerAgent
from agents.voltage_agent import VoltageAgent

__all__ = [
    "AgentConfig",
    "BaseSpecialist",
    "DEFAULT_AGENT_CONFIGS",
    "EnvironmentAgent",
    "HashrateAgent",
    "Maestro",
    "PowerAgent",
    "VoltageAgent",
    "run_orchestrator",
]
