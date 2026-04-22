"""Entry point: python -m deterministic_tools.main

Starts the detector loop — tails the telemetry stream and emits flag_raised
events whenever a pre-failure pattern is detected.

Usage
-----
    python -m deterministic_tools.main [--sensitivity LEVEL]

Environment overrides (see shared/paths.py):
    MDK_STREAM_DIR   — override stream directory (default: /run/mdk_fleet/stream)

The rule engine activates immediately on first tick.
The Isolation Forest and XGBoost flaggers activate only if their model files
exist (``models/if_v2.pkl`` and ``models/xgb_predictor.pkl`` respectively).
Run ``python -m deterministic_tools.train`` to train them.
"""

from __future__ import annotations

import argparse
import sys

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="deterministic_tools.main",
        description="MDK Fleet — deterministic pre-failure detector",
    )
    p.add_argument(
        "--sensitivity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Flagger sensitivity profile (default: medium)",
    )
    p.add_argument(
        "--telemetry-stream",
        default=None,
        help="Path to telemetry JSONL stream (overrides MDK_STREAM_DIR)",
    )
    p.add_argument(
        "--flag-output",
        default=None,
        help="Path to write flag_raised events (overrides MDK_STREAM_DIR)",
    )
    p.add_argument(
        "--xgb-model",
        default="models/xgb_predictor.pkl",
        help="Path to XGBoost model pkl (default: models/xgb_predictor.pkl)",
    )
    p.add_argument(
        "--if-model",
        default="models/if_v2.pkl",
        help="Path to Isolation Forest model pkl (default: models/if_v2.pkl)",
    )
    p.add_argument(
        "--chip-instability-model",
        default="models/xgb_chip_instability.pkl",
        help="Path to chip_instability XGBoost pkl",
    )
    p.add_argument(
        "--hashboard-failure-model",
        default="models/xgb_hashboard_failure.pkl",
        help="Path to hashboard_failure XGBoost pkl",
    )
    p.add_argument(
        "--disable-chip-instability",
        action="store_true",
        default=False,
        help="Disable ChipInstabilityFlagger (ablation flag)",
    )
    # hashboard_failure predictor is DISABLED by default — see runner.py
    # docstring for why (n=2-miner sample, cross-miner AUC 0.745 < 0.80 target).
    # Use --enable-hashboard-failure to opt in when you have more data.
    hb_group = p.add_mutually_exclusive_group()
    hb_group.add_argument(
        "--disable-hashboard-failure",
        dest="disable_hashboard_failure",
        action="store_true",
        default=True,
        help="Disable HashboardFailureFlagger (DEFAULT — pending more training data)",
    )
    hb_group.add_argument(
        "--enable-hashboard-failure",
        dest="disable_hashboard_failure",
        action="store_false",
        help="Force-enable HashboardFailureFlagger (advanced, pending more training data)",
    )

    # IsolationForestFlagger is DISABLED by default — zero flags across three
    # real runs and no validation metrics. See runner.py docstring and
    # models/ensemble_summary.md for the full rationale.
    if_group = p.add_mutually_exclusive_group()
    if_group.add_argument(
        "--disable-isolation-forest",
        dest="disable_isolation_forest",
        action="store_true",
        default=True,
        help="Disable IsolationForestFlagger (DEFAULT — pending validation study)",
    )
    if_group.add_argument(
        "--enable-isolation-forest",
        dest="disable_isolation_forest",
        action="store_false",
        help="Force-enable IsolationForestFlagger (advanced, unvalidated)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the detector loop (blocks forever)."""
    args = _parse_args(argv)

    log.info(
        "detector.main.start",
        sensitivity=args.sensitivity,
        xgb_model=args.xgb_model,
        if_model=args.if_model,
        chip_instability_model=args.chip_instability_model,
        hashboard_failure_model=args.hashboard_failure_model,
        disable_chip_instability=args.disable_chip_instability,
        disable_hashboard_failure=args.disable_hashboard_failure,
        disable_isolation_forest=args.disable_isolation_forest,
    )

    from deterministic_tools.runner import run_detector

    try:
        run_detector(
            input_stream=args.telemetry_stream,
            flag_output=args.flag_output,
            predictor_model_path=args.xgb_model,
            anomaly_model_path=args.if_model,
            chip_instability_model_path=args.chip_instability_model,
            hashboard_failure_model_path=args.hashboard_failure_model,
            sensitivity=args.sensitivity,
            disable_chip_instability=args.disable_chip_instability,
            disable_hashboard_failure=args.disable_hashboard_failure,
            disable_isolation_forest=args.disable_isolation_forest,
        )
    except KeyboardInterrupt:
        log.info("detector.main.stopped", reason="KeyboardInterrupt")
        sys.exit(0)


if __name__ == "__main__":
    main()
