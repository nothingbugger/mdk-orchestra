"""CLI entry point for the A/B experiment module.

Usage:
    # Quick smoke test (1 min simulated = 6s real, mock mode)
    MDK_AGENT_MOCK=1 python -m ab_experiment.runner --scenario smoke --duration-min 1

    # Full 60-min run with real Claude API
    python -m ab_experiment.runner --scenario full --duration-min 60 --api-mode

    # Full run, custom seed and output
    python -m ab_experiment.runner --scenario prod --duration-min 60 --seed 1234 \\
        --output /tmp/ab_runs/ --n-miners 50

Also importable as:
    from ab_experiment.runner import run_ab_experiment
"""

from __future__ import annotations

import argparse
import logging
import os
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ab_experiment.runner",
        description=(
            "MDK Fleet A/B experiment — compare multi-agent fleet vs deterministic tools."
        ),
    )
    p.add_argument(
        "--scenario",
        default="default",
        help="Scenario name (used in output directory). Default: 'default'.",
    )
    p.add_argument(
        "--duration-min",
        dest="duration_min",
        type=int,
        default=60,
        help=(
            "Simulated duration in minutes (real wall time = duration / 10x speed). "
            "Default: 60 (12 real minutes)."
        ),
    )
    p.add_argument(
        "--output",
        dest="output_dir",
        default=None,
        help=(
            "Output root directory. Default: <repo>/events/ab_runs/ or "
            "/run/mdk_fleet/ab_runs/ if writable."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Simulator RNG seed (same for both tracks). Default: 42.",
    )
    p.add_argument(
        "--n-miners",
        dest="n_miners",
        type=int,
        default=20,
        help="Number of miners per track (default: 20 for speed; 50 for full run).",
    )
    p.add_argument(
        "--api-mode",
        dest="api_mode",
        action="store_true",
        default=False,
        help=(
            "Enable real Claude API calls (requires ANTHROPIC_API_KEY). "
            "Default: mock mode (MDK_AGENT_MOCK=1)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--fault-mix",
        dest="fault_mix",
        choices=["random", "balanced"],
        default="random",
        help=(
            "Simulator fault-scheduling strategy. 'random' is per-miner "
            "independent draws (default); 'balanced' round-robins over "
            "the 4 fault types for maximal type variety (recommended for "
            "pitch-canonical runs requiring autonomy-ladder diversity)."
        ),
    )
    return p


def _default_output_dir() -> str:
    """Resolve the default ab_experiment output directory.

    Honours `shared.paths.get_runs_dir`, which picks in priority order:
      1. `MDK_RUNS_DIR` env var (if set)
      2. `~/.mdk-orchestra/runs/` (default)

    The previous implementation fell back to a repo-relative
    `events/ab_runs/` path which, under pipx-installed users, landed
    inside `site-packages/` and got wiped by every `pipx uninstall`.
    """
    from shared.paths import get_runs_dir
    return str(get_runs_dir())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    log = structlog.get_logger(__name__)

    # Resolve mock mode
    mock_mode = not args.api_mode
    if mock_mode:
        os.environ["MDK_AGENT_MOCK"] = "1"

    output_dir = args.output_dir or _default_output_dir()

    log.info(
        "ab_experiment.cli.start",
        scenario=args.scenario,
        duration_min=args.duration_min,
        seed=args.seed,
        n_miners=args.n_miners,
        mock_mode=mock_mode,
        output_dir=output_dir,
    )

    from ab_experiment.runner import run_ab_experiment

    try:
        results = run_ab_experiment(
            scenario=args.scenario,
            duration_min=args.duration_min,
            output_dir=output_dir,
            seed=args.seed,
            n_miners=args.n_miners,
            mock_mode=mock_mode,
            fault_mix=args.fault_mix,
        )
    except KeyboardInterrupt:
        log.info("ab_experiment.cli.interrupted")
        return 130

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(f"  A/B EXPERIMENT COMPLETE — {results.scenario} / {results.run_id}")
    print("=" * 60)
    print(f"  Track A (Fleet ON):")
    print(f"    flags: {results.run_a_flags_raised}  actions: {results.run_a_actions_taken}"
          f"  faults_caught: {results.run_a_faults_caught_pre}")
    print(f"  Track B (Det. only):")
    print(f"    flags: {results.run_b_flags_raised}  actions: {results.run_b_actions_taken}"
          f"  faults_caught: {results.run_b_faults_caught_pre}")
    print(f"  Total API cost: ${results.total_cost_usd:.4f}")
    print(f"  Results: {results.report_path}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
