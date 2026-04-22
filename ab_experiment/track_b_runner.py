"""Entry point for Track B rule-based mapper subprocess.

Launched by ab_experiment/runner.py as:
    python -m ab_experiment.track_b_runner \
        --flags /path/flags.jsonl \
        --decisions /path/decisions.jsonl \
        --actions /path/actions.jsonl

Replaces Maestro for Track B: reads flags, applies severity→action rules,
emits orchestrator_decision + action_taken (zero LLM calls, zero cost).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ab_experiment.track_b_runner",
        description="MDK Fleet A/B Track B — deterministic rule-based action mapper.",
    )
    p.add_argument(
        "--flags",
        required=True,
        help="Path to flags.jsonl for this track.",
    )
    p.add_argument(
        "--decisions",
        required=True,
        help="Path to write orchestrator_decision events.",
    )
    p.add_argument(
        "--actions",
        required=True,
        help="Path to write action_taken events.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    log = structlog.get_logger(__name__)
    log.info(
        "track_b_runner.start",
        flags=args.flags,
        decisions=args.decisions,
        actions=args.actions,
    )

    from ab_experiment.track_b import run_track_b_mapper

    try:
        run_track_b_mapper(
            flags_path=Path(args.flags),
            decisions_path=Path(args.decisions),
            actions_path=Path(args.actions),
            fleet_handle=None,  # no in-process fleet handle in subprocess mode
            stop_when=None,
        )
    except KeyboardInterrupt:
        log.info("track_b_runner.interrupted")
        return 130

    log.info("track_b_runner.stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
