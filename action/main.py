"""Entry point: `python -m action.main`.

Tails `decisions.jsonl`, emits `action_taken` envelopes to the bus.
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog

from action.executor import run_action_executor
from shared.paths import stream_paths


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MDK Fleet — action executor.")
    parser.add_argument("--decision-stream", default=None, help="Path to decisions JSONL.")
    parser.add_argument("--from-start", action="store_true", help="Replay existing decisions.")
    parser.add_argument("--stop-after", type=float, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = structlog.get_logger(__name__)
    path = args.decision_stream or str(stream_paths().decisions)
    log.info(
        "action_executor_starting",
        decision_stream=path,
        from_start=args.from_start,
        stop_after=args.stop_after,
        max_decisions=args.max_decisions,
    )

    try:
        run_action_executor(
            decision_stream=path,
            from_start=args.from_start,
            stop_after=args.stop_after,
            max_decisions=args.max_decisions,
        )
    except KeyboardInterrupt:
        log.info("action_executor_interrupted")
        return 130
    log.info("action_executor_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
