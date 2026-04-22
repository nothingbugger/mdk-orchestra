"""deterministic_tools — pre-failure flaggers for MDK Fleet.

Public API (matches shared/specs/interfaces.md §3):
    run_detector(...)          — main detector loop (runner.py)
    Flagger                    — Protocol (base.py)
    FlagResult                 — dataclass (base.py)
    MinerHistory               — per-miner rolling state (base.py)
    RuleEngineFlagger          — always-on hard-threshold flagger
    IsolationForestFlagger     — fleet-wide IF anomaly detector
    XGBoostFlagger             — streaming degradation predictor
"""

from deterministic_tools.base import FlagResult, Flagger, MinerHistory
from deterministic_tools.isolation_forest_flagger import IsolationForestFlagger
from deterministic_tools.rule_engine_flagger import RuleEngineFlagger
from deterministic_tools.runner import run_detector
from deterministic_tools.xgboost_flagger import XGBoostFlagger

__all__ = [
    "run_detector",
    "Flagger",
    "FlagResult",
    "MinerHistory",
    "RuleEngineFlagger",
    "IsolationForestFlagger",
    "XGBoostFlagger",
]
