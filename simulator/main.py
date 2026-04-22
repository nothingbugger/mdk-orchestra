"""Entry point: python -m simulator.main

Examples:
    # Default: 50 miners, 5s ticks, run forever
    python -m simulator.main

    # 50 miners, seed 42, run 1 simulated hour at 10x wall speed
    python -m simulator.main --n-miners 50 --seed 42 --duration 3600 --speed 10

    # Custom output, no faults
    python -m simulator.main --output /tmp/my_ticks.jsonl --no-faults
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog

from simulator.runner import run_simulator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON output to stderr."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simulator.main",
        description="MDK Fleet real-time miner telemetry simulator.",
    )
    p.add_argument("--n-miners", type=int, default=50, help="Number of miners (default: 50)")
    p.add_argument(
        "--tick",
        dest="tick_interval_s",
        type=float,
        default=5.0,
        help="Simulated seconds per tick (default: 5.0)",
    )
    p.add_argument(
        "--duration",
        dest="duration_s",
        type=float,
        default=None,
        help="Stop after N simulated seconds. Omit to run forever.",
    )
    p.add_argument(
        "--speed",
        dest="wall_speed_factor",
        type=float,
        default=1.0,
        help="Wall-clock speed multiplier. 10 = 10x faster (default: 1.0)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument(
        "--output",
        dest="output_stream",
        type=str,
        default=None,
        help="Output JSONL path. Default: auto-routed via shared.paths",
    )
    p.add_argument(
        "--no-faults",
        dest="fault_injection_enabled",
        action="store_false",
        default=True,
        help="Disable fault injection (clean telemetry only)",
    )
    p.add_argument(
        "--fault-mix",
        choices=["random", "balanced"],
        default="random",
        help=(
            "Fault-type assignment strategy. 'random' (default) uses the master "
            "RNG for each chosen miner's fault type, which can cluster on one "
            "type. 'balanced' round-robins over the four fault types so each "
            "gets ~an equal share — useful for demo/pitch runs needing variety."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    run_simulator(
        n_miners=args.n_miners,
        tick_interval_s=args.tick_interval_s,
        duration_s=args.duration_s,
        fault_injection_enabled=args.fault_injection_enabled,
        output_stream=args.output_stream,
        seed=args.seed,
        wall_speed_factor=args.wall_speed_factor,
        fault_mix=args.fault_mix,
    )


if __name__ == "__main__":
    main()
