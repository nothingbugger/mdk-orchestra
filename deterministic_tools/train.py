"""Bootstrap training for Isolation Forest and XGBoost flaggers.

Entry point: python -m deterministic_tools.train

This script replays the live telemetry stream (or a recorded one) to collect
labelled samples, then fits both ML models and saves them to disk.

Bootstrap protocol
------------------
1. Tail the telemetry stream.
2. Feed each tick into both the IF bootstrap buffer and the XGBoost label
   accumulator.
3. For the Isolation Forest: collect ``BOOTSTRAP_TICKS`` clean ticks
   (fault_injected is None). Reading fault_injected HERE IS ALLOWED because
   this is the training path, not the inference path. The model itself only
   sees feature vectors, never the labels.
4. For XGBoost: build a per-miner history buffer. After collecting enough
   ticks, label each tick with a 30-min look-ahead: label=1 if any tick in
   the next 360 ticks (30 min at 5 s) has fault_injected != None.
5. Both models are saved once fitted. The main detector loop will load them
   on next startup.

Usage
-----
    python -m deterministic_tools.train [--stream PATH] [--duration-min N]
        [--xgb-model PATH] [--if-model PATH] [--hparams '{"n_estimators":500}']
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)

# How long (seconds) to collect data before training.
DEFAULT_COLLECT_DURATION_MIN: int = 20

# XGBoost label look-ahead in ticks.
_LABEL_LOOKAHEAD_TICKS: int = 360

# Minimum positive samples required for XGBoost fit.
_MIN_POSITIVE_SAMPLES: int = 50

# Default XGBoost hyperparameters
_DEFAULT_XGB_HPARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="deterministic_tools.train",
        description="Bootstrap training for IF and XGBoost flaggers",
    )
    p.add_argument(
        "--stream",
        default=None,
        help="Path to telemetry JSONL stream (replayed from start). "
        "Defaults to live stream from MDK_STREAM_DIR.",
    )
    p.add_argument(
        "--duration-min",
        type=int,
        default=DEFAULT_COLLECT_DURATION_MIN,
        help=f"Minutes to collect data (default: {DEFAULT_COLLECT_DURATION_MIN})",
    )
    p.add_argument(
        "--xgb-model",
        default="models/xgb_predictor.pkl",
        help="Output path for the XGBoost model pkl",
    )
    p.add_argument(
        "--if-model",
        default="models/if_v2.pkl",
        help="Output path for the Isolation Forest model pkl",
    )
    p.add_argument(
        "--from-start",
        action="store_true",
        default=True,
        help="Read stream from the beginning (default: True)",
    )
    p.add_argument(
        "--hparams",
        default=None,
        help='Optional JSON string to override XGBoost hyperparameters. '
        'E.g. \'{"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03}\'',
    )
    p.add_argument(
        "--metrics-out",
        default="models/xgb_metrics.json",
        help="Output path for XGBoost validation metrics JSON (default: models/xgb_metrics.json)",
    )
    p.add_argument(
        "--dataset-source",
        default=None,
        help="Label describing the dataset source (e.g. stream file path or description)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the training bootstrap."""
    args = _parse_args(argv)

    from shared.event_bus import read_events, tail_events
    from shared.paths import stream_paths

    from deterministic_tools.isolation_forest_flagger import (
        BOOTSTRAP_TICKS as IF_BOOTSTRAP,
        IsolationForestFlagger,
        _extract_features,
    )
    from deterministic_tools.xgboost_flagger import _build_features, _LABEL_LOOKAHEAD_TICKS
    from deterministic_tools.base import MinerHistory

    sp = stream_paths()
    telemetry_path = Path(args.stream) if args.stream else sp.telemetry
    # When a static file is provided via --stream, use one-shot read_events so
    # the loop terminates at EOF. For the live stream (no --stream), use
    # tail_events with a duration stop.
    use_static_read = args.stream is not None
    xgb_model_path = Path(args.xgb_model)
    if_model_path = Path(args.if_model)
    metrics_out_path = Path(args.metrics_out)
    dataset_source = args.dataset_source or str(telemetry_path)

    # Parse optional hyperparameter overrides
    hparam_overrides: dict[str, Any] = {}
    if args.hparams:
        try:
            hparam_overrides = json.loads(args.hparams)
            log.info("train.hparam_overrides", overrides=hparam_overrides)
        except json.JSONDecodeError as exc:
            log.error("train.hparam_parse_error", error=str(exc))
            sys.exit(1)

    duration_s = args.duration_min * 60
    log.info(
        "train.starting",
        telemetry_path=str(telemetry_path),
        duration_min=args.duration_min,
        static_read=use_static_read,
        xgb_model=str(xgb_model_path),
        if_model=str(if_model_path),
        metrics_out=str(metrics_out_path),
        hparam_overrides=hparam_overrides,
    )

    # Data accumulators.
    if_clean_samples: list[list[float]] = []
    xgb_histories: dict[str, MinerHistory] = defaultdict(lambda: MinerHistory(miner_id=""))
    # Per-miner deque of (features, fault_injected_bool) for label look-ahead.
    xgb_tick_buf: dict[str, deque[tuple[list[float], bool]]] = defaultdict(
        lambda: deque(maxlen=_LABEL_LOOKAHEAD_TICKS + 100)
    )
    xgb_X: list[list[float]] = []
    xgb_y: list[int] = []

    start_wall = time.monotonic()
    ticks_seen = 0

    def _stop() -> bool:
        return (time.monotonic() - start_wall) >= duration_s

    log.info("train.collecting", duration_s=duration_s, static_read=use_static_read)

    event_source = (
        read_events(telemetry_path)
        if use_static_read
        else tail_events(telemetry_path, from_start=args.from_start, stop_when=_stop)
    )

    for envelope in event_source:
        if envelope.event != "telemetry_tick":
            continue

        tick = envelope.typed_data()
        mid = tick.miner_id
        fault = tick.fault_injected  # ALLOWED: this is the training path
        ticks_seen += 1

        # --- Isolation Forest bootstrap ---
        if len(if_clean_samples) < IF_BOOTSTRAP:
            if fault is None:
                features = _extract_features(tick)
                if_clean_samples.append(features)
                if len(if_clean_samples) % 200 == 0:
                    log.info(
                        "train.if_progress",
                        collected=len(if_clean_samples),
                        needed=IF_BOOTSTRAP,
                    )

        # --- XGBoost history + label accumulation ---
        history = xgb_histories[mid]
        history.miner_id = mid
        history.push_telemetry(tick, envelope.ts)

        feat = _build_features(history, tick)
        if feat is not None:
            is_fault = fault is not None
            xgb_tick_buf[mid].append((feat, is_fault))

            # Once the buffer has > LABEL_LOOKAHEAD_TICKS entries, label the
            # oldest tick with its 30-min look-ahead.
            buf = xgb_tick_buf[mid]
            if len(buf) > _LABEL_LOOKAHEAD_TICKS:
                oldest_feat, _ = buf[0]
                # Look-ahead window: do any of the next 360 entries have fault?
                label = int(
                    any(is_f for _, is_f in list(buf)[1 : _LABEL_LOOKAHEAD_TICKS + 1])
                )
                xgb_X.append(oldest_feat)
                xgb_y.append(label)

    log.info(
        "train.collection_done",
        ticks_seen=ticks_seen,
        if_clean_samples=len(if_clean_samples),
        xgb_labelled_samples=len(xgb_X),
        xgb_positive_labels=sum(xgb_y),
    )

    # --- Fit Isolation Forest ---
    _fit_isolation_forest(if_clean_samples, if_model_path)

    # --- Fit XGBoost with validation split ---
    _fit_xgboost(
        xgb_X,
        xgb_y,
        xgb_model_path,
        metrics_out_path=metrics_out_path,
        hparam_overrides=hparam_overrides,
        dataset_source=dataset_source,
    )

    log.info("train.done")


def _fit_isolation_forest(samples: list[list[float]], out_path: Path) -> None:
    """Train and save Isolation Forest."""
    import pickle

    from sklearn.ensemble import IsolationForest

    if len(samples) < 100:
        log.warning(
            "train.if_skipped",
            reason="insufficient_clean_samples",
            have=len(samples),
            need=100,
        )
        return

    X = np.array(samples, dtype=np.float32)
    log.info("train.if_fitting", n_samples=X.shape[0])
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("train.if_saved", path=str(out_path))


def _fit_xgboost(
    X_list: list[list[float]],
    y_list: list[int],
    out_path: Path,
    metrics_out_path: Path | None = None,
    hparam_overrides: dict[str, Any] | None = None,
    dataset_source: str = "unknown",
) -> dict[str, Any]:
    """Train XGBoost with 80/20 stratified split and compute honest validation metrics.

    Args:
        X_list: feature matrix rows.
        y_list: binary labels (0/1).
        out_path: path to save the fitted model pkl.
        metrics_out_path: if set, write validation metrics JSON here.
        hparam_overrides: optional hyperparameter dict to override defaults.
        dataset_source: human-readable label for the dataset source.

    Returns:
        Metrics dict (also written to metrics_out_path if set).
    """
    import json
    import pickle

    import xgboost as xgb  # type: ignore[import]
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    from deterministic_tools.xgboost_flagger import _FEATURE_NAMES

    if len(X_list) < 200:
        log.warning(
            "train.xgb_skipped",
            reason="insufficient_labelled_samples",
            have=len(X_list),
            need=200,
        )
        return {}

    positives = sum(y_list)
    if positives < _MIN_POSITIVE_SAMPLES:
        log.warning(
            "train.xgb_skipped",
            reason="insufficient_positive_samples",
            positives=positives,
            need=_MIN_POSITIVE_SAMPLES,
        )
        return {}

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # 80/20 stratified split, seed 42
    log.info("train.xgb_splitting", total=len(y), positives=int(positives))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    n_train = len(y_train)
    n_val = len(y_val)
    pos_rate_train = float(y_train.mean())
    pos_rate_val = float(y_val.mean())

    log.info(
        "train.xgb_split_done",
        n_train=n_train,
        n_val=n_val,
        positive_rate_train=round(pos_rate_train, 4),
        positive_rate_val=round(pos_rate_val, 4),
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Build hyperparameters (defaults + any overrides)
    hparams: dict[str, Any] = dict(_DEFAULT_XGB_HPARAMS)
    if hparam_overrides:
        hparams.update(hparam_overrides)
    hparams["scale_pos_weight"] = round(scale_pos_weight, 4)

    log.info(
        "train.xgb_fitting",
        n_train=n_train,
        positives=pos_count,
        negatives=neg_count,
        scale_pos_weight=round(scale_pos_weight, 2),
        hyperparameters=hparams,
    )

    t0 = time.monotonic()
    model = xgb.XGBClassifier(**hparams)
    model.fit(X_train, y_train)
    training_time_s = round(time.monotonic() - t0, 2)

    log.info("train.xgb_trained", training_time_s=training_time_s)

    # --- Validation metrics ---
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_val, y_prob))
    pr_auc = float(average_precision_score(y_val, y_prob))
    ll = float(log_loss(y_val, y_prob))
    brier = float(brier_score_loss(y_val, y_prob))

    # Precision/recall at top-10% predicted probability
    n_top = max(1, int(len(y_val) * 0.10))
    top_indices = np.argsort(y_prob)[::-1][:n_top]
    top_labels = y_val[top_indices]
    precision_top10 = float(top_labels.mean())
    # recall = positives captured in top-10% / total positives
    total_pos = int(y_val.sum())
    recall_top10 = float(top_labels.sum() / max(total_pos, 1))

    # Confusion matrix at default threshold 0.5
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Feature importance top-5
    fi = model.feature_importances_
    fi_pairs = sorted(
        zip(_FEATURE_NAMES, fi.tolist()), key=lambda x: x[1], reverse=True
    )
    feature_importance_top5 = [
        {"name": name, "importance": round(imp, 6)} for name, imp in fi_pairs[:5]
    ]

    metrics: dict[str, Any] = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_top_10pct": round(precision_top10, 4),
        "recall_at_top_10pct": round(recall_top10, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "feature_importance_top5": feature_importance_top5,
        "confusion_at_default_threshold": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n_train": n_train,
        "n_val": n_val,
        "positive_rate_train": round(pos_rate_train, 4),
        "positive_rate_val": round(pos_rate_val, 4),
        "training_time_s": training_time_s,
        "hyperparameters": hparams,
        "dataset_source": dataset_source,
    }

    log.info(
        "train.xgb_metrics",
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        precision_at_top_10pct=metrics["precision_at_top_10pct"],
        recall_at_top_10pct=metrics["recall_at_top_10pct"],
        log_loss=metrics["log_loss"],
        brier_score=metrics["brier_score"],
        feature_importance_top5=feature_importance_top5,
    )

    # Save model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("train.xgb_saved", path=str(out_path))

    # Save metrics
    if metrics_out_path is not None:
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("train.metrics_saved", path=str(metrics_out_path))

    return metrics


if __name__ == "__main__":
    main()
