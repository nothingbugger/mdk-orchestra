"""Entry point: python -m ingest.main

Starts the ingest pipeline, tailing the telemetry stream and emitting
kpi_update and fleet_snapshot events.

Usage
-----
    # Default paths (from shared/paths.py, env-overridable):
    python -m ingest.main

    # Override paths:
    python -m ingest.main \\
        --input-stream /run/mdk_fleet/stream/telemetry.jsonl \\
        --kpi-output   /run/mdk_fleet/stream/kpis.jsonl \\
        --snap-output  /run/mdk_fleet/stream/snapshots.jsonl \\
        --snap-interval 1.0

Environment overrides
---------------------
    MDK_STREAM_DIR    root dir for all streams (see shared/paths.py)
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog


def _configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON output to stderr."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDK Fleet ingest module — computes TE/HSI from telemetry stream.",
    )
    p.add_argument(
        "--input-stream",
        default=None,
        help="Path to telemetry.jsonl (default: canonical stream_paths().telemetry)",
    )
    p.add_argument(
        "--kpi-output",
        default=None,
        help="Path for kpi_update events (default: canonical stream_paths().kpis)",
    )
    p.add_argument(
        "--snap-output",
        default=None,
        help="Path for fleet_snapshot events (default: canonical stream_paths().snapshots)",
    )
    p.add_argument(
        "--snap-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between fleet_snapshot emissions (default: 1.0)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the ingest pipeline."""
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    from ingest.runner import run_ingest

    run_ingest(
        input_stream=args.input_stream,
        kpi_output=args.kpi_output,
        snapshot_output=args.snap_output,
        snapshot_interval_s=args.snap_interval,
    )


if __name__ == "__main__":
    main()
