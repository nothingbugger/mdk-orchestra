"""Outer training loop with automatic retry if metrics are weak.

Calls generate_training_data.py → train.py → reads xgb_metrics.json.
If metrics don't meet targets, retries up to MAX_RETRIES times with a larger
dataset and stronger hyperparameters.

Targets (minimum passing bar):
  - precision_at_top_10pct >= 0.40
  - roc_auc >= 0.70

Retry escalation:
  Retry 1: 8 hours of simulated data, same HPs
  Retry 2: 8 hours, HP boost (n_estimators=500, max_depth=7, learning_rate=0.03)

Writes decision log to models/training_log.md.

Usage:
    python3 scripts/train_xgb_iterate.py
    python3 scripts/train_xgb_iterate.py --skip-generate  # reuse existing data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = MODELS_DIR / "xgb_metrics.json"
TRAINING_LOG_PATH = MODELS_DIR / "training_log.md"

# Metric targets
TARGET_PRECISION_TOP10 = 0.40
TARGET_ROC_AUC = 0.70

MAX_RETRIES = 2

# Iteration configs: (hours, hparams_override or None)
_ITERATION_CONFIGS = [
    # Iteration 0: baseline – 4h balanced, default HPs
    {
        "hours": 4.0,
        "seed": 7,
        "hparams": None,
        "label": "4h_balanced_default_hp",
    },
    # Retry 1: bigger dataset, same HPs
    {
        "hours": 8.0,
        "seed": 7,
        "hparams": None,
        "label": "8h_balanced_default_hp",
    },
    # Retry 2: bigger dataset + boosted HPs
    {
        "hours": 8.0,
        "seed": 7,
        "hparams": {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03},
        "label": "8h_balanced_boosted_hp",
    },
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative XGBoost training with auto-retry")
    p.add_argument(
        "--skip-generate",
        action="store_true",
        default=False,
        help="Skip data generation (reuse existing stream files). "
        "Useful for re-running training on existing data.",
    )
    p.add_argument(
        "--stream-dir",
        default="/tmp/mdk_train_data",
        help="Directory for generated telemetry files (default: /tmp/mdk_train_data)",
    )
    return p.parse_args(argv)


def _meets_targets(metrics: dict[str, Any]) -> bool:
    """Return True if metrics satisfy both passing criteria."""
    return (
        metrics.get("precision_at_top_10pct", 0.0) >= TARGET_PRECISION_TOP10
        and metrics.get("roc_auc", 0.0) >= TARGET_ROC_AUC
    )


def _run_generate(hours: float, seed: int, stream_path: Path) -> None:
    """Run data generation subprocess."""
    script = REPO_ROOT / "scripts" / "generate_training_data.py"
    cmd = [
        sys.executable,
        str(script),
        "--hours", str(hours),
        "--miners", "50",
        "--seed", str(seed),
        "--fault-mix", "balanced",
        "--out", str(stream_path),
    ]
    log.info("iterate.generate_start", cmd=" ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False, check=True)
    elapsed = round(time.monotonic() - t0, 1)
    log.info("iterate.generate_done", wall_s=elapsed, returncode=result.returncode)


def _run_train(stream_path: Path, hparams: dict[str, Any] | None, dataset_source: str) -> dict[str, Any]:
    """Run training subprocess and return parsed metrics."""
    script_module = "deterministic_tools.train"
    cmd = [
        sys.executable, "-m", script_module,
        "--stream", str(stream_path),
        "--metrics-out", str(METRICS_PATH),
        "--xgb-model", str(MODELS_DIR / "xgb_predictor.pkl"),
        "--if-model", str(MODELS_DIR / "if_v2.pkl"),
        "--dataset-source", dataset_source,
    ]
    # duration-min: set large so all data in file is consumed
    # The stop_when callback fires on EOF rather than duration, but duration is
    # a safety valve — 99999 ensures we consume the whole file
    cmd += ["--duration-min", "99999"]

    if hparams:
        cmd += ["--hparams", json.dumps(hparams)]

    log.info("iterate.train_start", cmd=" ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False, check=True, cwd=str(REPO_ROOT))
    elapsed = round(time.monotonic() - t0, 1)
    log.info("iterate.train_done", wall_s=elapsed, returncode=result.returncode)

    if not METRICS_PATH.exists():
        log.error("iterate.metrics_missing", path=str(METRICS_PATH))
        return {}

    with METRICS_PATH.open() as f:
        return json.load(f)


def _append_log(
    log_path: Path,
    iteration: int,
    config_label: str,
    hours: float,
    hparams: dict[str, Any] | None,
    metrics: dict[str, Any],
    targets_met: bool,
    decision: str,
    wall_s: float,
) -> None:
    """Append a training iteration entry to training_log.md."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hparams_str = json.dumps(hparams) if hparams else "default"

    entry_lines = [
        f"\n## Iteration {iteration} — {config_label} ({ts})\n",
        f"- **Dataset:** {hours}h simulated @ 50 miners, balanced fault mix\n",
        f"- **Hyperparameters:** `{hparams_str}`\n",
        f"- **n_train:** {metrics.get('n_train', 'N/A')}  |  **n_val:** {metrics.get('n_val', 'N/A')}\n",
        f"- **Positive rate (train/val):** "
        f"{metrics.get('positive_rate_train', 'N/A')} / {metrics.get('positive_rate_val', 'N/A')}\n",
        f"\n### Results\n",
        f"| Metric | Value | Target |\n",
        f"|--------|-------|--------|\n",
        f"| roc_auc | {metrics.get('roc_auc', 'N/A')} | ≥ {TARGET_ROC_AUC} |\n",
        f"| pr_auc | {metrics.get('pr_auc', 'N/A')} | — |\n",
        f"| precision_at_top_10pct | {metrics.get('precision_at_top_10pct', 'N/A')} | ≥ {TARGET_PRECISION_TOP10} |\n",
        f"| recall_at_top_10pct | {metrics.get('recall_at_top_10pct', 'N/A')} | — |\n",
        f"| log_loss | {metrics.get('log_loss', 'N/A')} | — |\n",
        f"| brier_score | {metrics.get('brier_score', 'N/A')} | — |\n",
    ]

    fi = metrics.get("feature_importance_top5", [])
    if fi:
        entry_lines.append(f"\n### Feature Importance Top-5\n")
        for item in fi:
            entry_lines.append(f"- `{item['name']}`: {item['importance']}\n")

    cm = metrics.get("confusion_at_default_threshold", {})
    if cm:
        entry_lines.append(
            f"\n### Confusion Matrix @ threshold=0.5\n"
            f"TN={cm.get('tn')}  FP={cm.get('fp')}  FN={cm.get('fn')}  TP={cm.get('tp')}\n"
        )

    entry_lines.append(f"\n**Wall time:** {wall_s}s\n")

    if targets_met:
        entry_lines.append(f"\n**TARGETS MET** — shipping this model.\n")
    else:
        entry_lines.append(f"\n**TARGETS NOT MET** — {decision}\n")

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# XGBoost Training Log\n\nGenerated by `scripts/train_xgb_iterate.py`.\n")
            f.write(f"\n**Target thresholds:** roc_auc ≥ {TARGET_ROC_AUC}, "
                    f"precision_at_top_10pct ≥ {TARGET_PRECISION_TOP10}\n")
        for line in entry_lines:
            f.write(line)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stream_dir = Path(args.stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)

    best_metrics: dict[str, Any] = {}
    best_iteration = -1
    final_targets_met = False

    wall_start = time.monotonic()

    for iteration, cfg in enumerate(_ITERATION_CONFIGS):
        iter_label = cfg["label"]
        hours = cfg["hours"]
        seed = cfg["seed"]
        hparams = cfg["hparams"]
        stream_path = stream_dir / f"telemetry_iter{iteration}.jsonl"

        log.info(
            "iterate.iteration_start",
            iteration=iteration,
            label=iter_label,
            hours=hours,
            hparams=hparams,
        )

        iter_wall_start = time.monotonic()

        # Generate data (skip if --skip-generate and file exists for iter 0)
        if args.skip_generate and iteration == 0 and stream_path.exists():
            log.info("iterate.skipping_generate", path=str(stream_path))
        else:
            _run_generate(hours=hours, seed=seed, stream_path=stream_path)

        # Train
        metrics = _run_train(
            stream_path=stream_path,
            hparams=hparams,
            dataset_source=f"{iter_label} ({stream_path})",
        )

        iter_wall_s = round(time.monotonic() - iter_wall_start, 1)

        if not metrics:
            log.error("iterate.no_metrics", iteration=iteration)
            decision = "Training produced no metrics; aborting."
            _append_log(
                TRAINING_LOG_PATH, iteration, iter_label, hours, hparams,
                metrics={}, targets_met=False, decision=decision, wall_s=iter_wall_s
            )
            break

        best_metrics = metrics
        best_iteration = iteration
        targets_met = _meets_targets(metrics)
        final_targets_met = targets_met

        roc = metrics.get("roc_auc", 0.0)
        prec = metrics.get("precision_at_top_10pct", 0.0)

        if targets_met:
            decision = "Targets met — no further iterations needed."
            _append_log(
                TRAINING_LOG_PATH, iteration, iter_label, hours, hparams,
                metrics=metrics, targets_met=True, decision=decision, wall_s=iter_wall_s
            )
            log.info(
                "iterate.targets_met",
                iteration=iteration,
                roc_auc=roc,
                precision_at_top_10pct=prec,
            )
            break
        else:
            if iteration + 1 < len(_ITERATION_CONFIGS):
                next_cfg = _ITERATION_CONFIGS[iteration + 1]
                decision = (
                    f"roc_auc={roc} (target {TARGET_ROC_AUC}), "
                    f"precision_top10={prec} (target {TARGET_PRECISION_TOP10}). "
                    f"Retrying with {next_cfg['label']}."
                )
            else:
                decision = (
                    f"roc_auc={roc} (target {TARGET_ROC_AUC}), "
                    f"precision_top10={prec} (target {TARGET_PRECISION_TOP10}). "
                    f"MAX RETRIES REACHED — shipping best-so-far model. "
                    f"TARGETS NOT MET — see note below."
                )

            _append_log(
                TRAINING_LOG_PATH, iteration, iter_label, hours, hparams,
                metrics=metrics, targets_met=False, decision=decision, wall_s=iter_wall_s
            )
            log.warning(
                "iterate.targets_not_met",
                iteration=iteration,
                roc_auc=roc,
                precision_at_top_10pct=prec,
                decision=decision,
            )

            if iteration + 1 >= len(_ITERATION_CONFIGS):
                break

    total_wall = round(time.monotonic() - wall_start, 1)

    if not final_targets_met and best_metrics:
        # Append a loud final warning to training_log.md
        with TRAINING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"\n---\n"
                f"## FINAL STATUS — TARGETS NOT MET\n\n"
                f"After {best_iteration + 1} iteration(s), targets were not reached.\n"
                f"Best model is from iteration {best_iteration}.\n\n"
                f"**Best roc_auc:** {best_metrics.get('roc_auc', 'N/A')} (target ≥ {TARGET_ROC_AUC})\n"
                f"**Best precision_at_top_10pct:** {best_metrics.get('precision_at_top_10pct', 'N/A')} "
                f"(target ≥ {TARGET_PRECISION_TOP10})\n\n"
                f"Possible next steps:\n"
                f"- More training data (>8h) or higher miner count\n"
                f"- Label smoothing / different look-ahead window\n"
                f"- Feature engineering (lag features, rate-of-change)\n"
                f"- Hyperparameter search (grid or Bayesian)\n"
            )
        log.warning(
            "iterate.final_targets_not_met",
            best_iteration=best_iteration,
            best_roc_auc=best_metrics.get("roc_auc"),
            best_precision_top10=best_metrics.get("precision_at_top_10pct"),
            note="Shipping best-so-far model",
        )
    else:
        log.info(
            "iterate.final_success",
            iteration=best_iteration,
            roc_auc=best_metrics.get("roc_auc"),
            precision_at_top_10pct=best_metrics.get("precision_at_top_10pct"),
        )

    log.info(
        "iterate.complete",
        total_iterations=best_iteration + 1,
        total_wall_s=total_wall,
        final_targets_met=final_targets_met,
        metrics_path=str(METRICS_PATH),
        log_path=str(TRAINING_LOG_PATH),
    )


if __name__ == "__main__":
    main()
