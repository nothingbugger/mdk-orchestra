"""Entry point: `python -m agents.main`.

Tails the deterministic_tools flag stream and dispatches every `flag_raised`
to Maestro. Emits `orchestrator_decision` and per-specialist
`reasoning_response` / `episodic_memory_write` events to the bus.

Environment:
  ANTHROPIC_API_KEY   — if absent, agents run in deterministic mock mode
                        so the end-to-end smoke works without credits.
  MDK_AGENT_MOCK=1    — force mock mode even when a key is set.
  MDK_ORCHESTRATOR_MODEL, MDK_VOLTAGE_MODEL, ... — per-agent overrides.
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog

from agents.config import mock_mode_enabled
from agents.maestro import run_orchestrator
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
    parser = argparse.ArgumentParser(description="MDK Fleet — Maestro orchestrator loop.")
    parser.add_argument(
        "--flag-stream",
        default=None,
        help="Path to flags JSONL (default: canonical stream).",
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Replay existing flags before following (default: seek to EOF).",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        default=None,
        help="Stop after N wall-clock seconds.",
    )
    parser.add_argument(
        "--max-flags",
        type=int,
        default=None,
        help="Stop after processing N flags.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = structlog.get_logger(__name__)

    flag_path = args.flag_stream or str(stream_paths().flags)
    log.info(
        "maestro_starting",
        flag_stream=flag_path,
        mock_mode=mock_mode_enabled(),
        from_start=args.from_start,
        stop_after=args.stop_after,
        max_flags=args.max_flags,
    )

    try:
        run_orchestrator(
            flag_stream=flag_path,
            from_start=args.from_start,
            stop_after=args.stop_after,
            max_flags=args.max_flags,
        )
    except KeyboardInterrupt:
        log.info("maestro_interrupted")
        return 130
    log.info("maestro_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
