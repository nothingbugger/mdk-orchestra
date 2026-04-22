"""Action executor — consumes orchestrator_decision, emits action_taken.

Separation of concerns: Maestro never reaches into the simulator. All
miner-facing commands pass through this module's `apply_action` hook so
that (a) the action log is centralized in one place, (b) the A/B
experiment's Track B (deterministic-only, no Maestro) can reuse the
same hook with a rule-based action mapping.
"""

from action.executor import Executor, apply_to_fleet, run_action_executor

__all__ = ["Executor", "apply_to_fleet", "run_action_executor"]
