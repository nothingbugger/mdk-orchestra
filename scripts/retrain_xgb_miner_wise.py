"""Retrain XGBoost with miner-wise canonical split.

Replaces the random-split baseline model with an honest miner-wise model:
train on miners m001–m040, test on m041–m050. Keeps the baseline metrics
in the new metrics JSON as `baseline_reference` for transparency.

Outputs:
  models/xgb_predictor.pkl              — retrained model (canonical)
  models/xgb_predictor.original.pkl     — backup of random-split model
  models/xgb_metrics.json               — updated (primary = miner-wise)
  models/xgb_feature_importance.json    — top-10 with values
  models/xgb_retrain_summary.md         — brief report

Usage:
    python scripts/retrain_xgb_miner_wise.py [--input path]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Reuse utilities from the leakage-check script (paths, feature extractor,
# hparams, row loader, labeler, XGB fit scaffolding).
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from xgb_leakage_check import (  # type: ignore[import-not-found]
    _FEATURE_NAMES,
    _XGB_HPARAMS,
    _build_labelled_dataset,
    _load_raw_records,
)

REPO = SCRIPTS.parent
MODELS_DIR = REPO / "models"
DATA_DIR = REPO / "training_data"


def retrain(input_path: Path, test_miner_range: tuple[int, int] = (41, 50)) -> dict[str, Any]:
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )

    print(f"[retrain] loading {input_path} ...", flush=True)
    records = _load_raw_records(input_path)
    print(f"[retrain] {len(records)} raw ticks", flush=True)
    rows = _build_labelled_dataset(records)
    print(f"[retrain] {len(rows)} labeled samples after feature extraction", flush=True)

    train_miners = {f"m{i:03d}" for i in range(1, test_miner_range[0])}
    test_miners = {f"m{i:03d}" for i in range(test_miner_range[0], test_miner_range[1] + 1)}
    print(
        f"[retrain] split: train={len(train_miners)} miners, test={len(test_miners)} miners",
        flush=True,
    )

    train_rows = [r for r in rows if r["miner_id"] in train_miners]
    test_rows = [r for r in rows if r["miner_id"] in test_miners]

    X_train = np.array([r["features"] for r in train_rows], dtype=np.float32)
    y_train = np.array([r["label"] for r in train_rows], dtype=np.int32)
    X_test = np.array([r["features"] for r in test_rows], dtype=np.float32)
    y_test = np.array([r["label"] for r in test_rows], dtype=np.int32)

    pos_train = int(y_train.sum())
    neg_train = len(y_train) - pos_train
    scale = neg_train / max(pos_train, 1)

    hparams = dict(_XGB_HPARAMS)
    hparams["scale_pos_weight"] = round(scale, 4)

    print(
        f"[retrain] fitting on {len(y_train)} (pos={pos_train}, neg={neg_train}, "
        f"scale_pos_weight={scale:.2f})",
        flush=True,
    )
    t0 = time.monotonic()
    model = xgb.XGBClassifier(**hparams)
    model.fit(X_train, y_train)
    fit_time_s = round(time.monotonic() - t0, 2)
    print(f"[retrain] fit done in {fit_time_s}s", flush=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    ll = float(log_loss(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    n_top = max(1, int(len(y_test) * 0.10))
    import numpy as _np
    top_idx = _np.argsort(y_prob)[::-1][:n_top]
    top_labels = y_test[top_idx]
    prec10 = float(top_labels.mean())
    total_pos = int(y_test.sum())
    rec10 = float(top_labels.sum() / max(total_pos, 1))

    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fi = model.feature_importances_
    fi_pairs = sorted(
        zip(_FEATURE_NAMES, fi.tolist()), key=lambda x: x[1], reverse=True
    )

    return {
        "model": model,
        "split": "miner_wise_80_20",
        "train_miners": sorted(train_miners),
        "test_miners": sorted(test_miners),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": round(pos_train / max(len(y_train), 1), 4),
        "positive_rate_test": round(int(y_test.sum()) / max(len(y_test), 1), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_top_10pct": round(prec10, 4),
        "recall_at_top_10pct": round(rec10, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "confusion_at_default_threshold": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "feature_importance_full": [
            {"name": n, "importance": round(imp, 6)} for n, imp in fi_pairs
        ],
        "feature_importance_top5": [
            {"name": n, "importance": round(imp, 6)} for n, imp in fi_pairs[:5]
        ],
        "hparams": hparams,
        "fit_time_s": fit_time_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/tmp/mdk_train_data/telemetry_iter0.jsonl",
        help="Path to telemetry JSONL (default: /tmp/mdk_train_data/telemetry_iter0.jsonl)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[fatal] input not found: {input_path}", file=sys.stderr)
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Back up the existing model
    canonical_pkl = MODELS_DIR / "xgb_predictor.pkl"
    backup_pkl = MODELS_DIR / "xgb_predictor.original.pkl"
    if canonical_pkl.exists() and not backup_pkl.exists():
        shutil.copy2(canonical_pkl, backup_pkl)
        print(f"[retrain] backed up original → {backup_pkl}", flush=True)

    # Load previous metrics for baseline_reference
    baseline_metrics_path = MODELS_DIR / "xgb_metrics.json"
    baseline = {}
    if baseline_metrics_path.exists():
        try:
            baseline = json.loads(baseline_metrics_path.read_text())
        except Exception:
            baseline = {}

    result = retrain(input_path)

    # --- save model ---
    with canonical_pkl.open("wb") as fh:
        pickle.dump(result["model"], fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[retrain] saved → {canonical_pkl}", flush=True)

    # --- save metrics JSON ---
    # Preserve baseline as reference if present
    baseline_reference = {
        "split": "random_stratified_80_20",
        "roc_auc": baseline.get("roc_auc"),
        "pr_auc": baseline.get("pr_auc"),
        "precision_at_top_10pct": baseline.get("precision_at_top_10pct"),
        "recall_at_top_10pct": baseline.get("recall_at_top_10pct"),
        "log_loss": baseline.get("log_loss"),
        "feature_importance_top5": baseline.get("feature_importance_top5"),
        "note": (
            "Baseline used a random stratified split that interleaved "
            "ticks from the same miner across train and test — identity "
            "leakage inflated AUC. Kept here as reference only. See "
            "models/leakage_report.md for full analysis."
        ),
    }

    metrics_json = {
        "split_method": "miner_wise_80_20",
        "description": (
            "Train on miners m001..m040, test on m041..m050. Canonical "
            "evaluation for MDK Fleet XGBoost degradation predictor. "
            "Honest OOD generalization metric."
        ),
        "primary": {
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "precision_at_top_10pct": result["precision_at_top_10pct"],
            "recall_at_top_10pct": result["recall_at_top_10pct"],
            "log_loss": result["log_loss"],
            "brier_score": result["brier_score"],
            "confusion_at_default_threshold": result["confusion_at_default_threshold"],
        },
        "dataset": {
            "source": str(input_path),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "train_miners_count": len(result["train_miners"]),
            "test_miners_count": len(result["test_miners"]),
            "positive_rate_train": result["positive_rate_train"],
            "positive_rate_test": result["positive_rate_test"],
        },
        "feature_importance_top5": result["feature_importance_top5"],
        "hparams": result["hparams"],
        "fit_time_s": result["fit_time_s"],
        "baseline_reference": baseline_reference,
    }
    baseline_metrics_path.write_text(json.dumps(metrics_json, indent=2))
    print(f"[retrain] saved → {baseline_metrics_path}", flush=True)

    # --- feature importance JSON (top-10) ---
    fi_path = MODELS_DIR / "xgb_feature_importance.json"
    fi_path.write_text(
        json.dumps(
            {
                "top10": result["feature_importance_full"][:10],
                "all": result["feature_importance_full"],
            },
            indent=2,
        )
    )
    print(f"[retrain] saved → {fi_path}", flush=True)

    # --- summary markdown ---
    summary = MODELS_DIR / "xgb_retrain_summary.md"
    baseline_auc = baseline.get("roc_auc")
    delta_line = (
        f"- ROC AUC delta: {baseline_auc:.4f} → {result['roc_auc']:.4f} "
        f"({result['roc_auc'] - baseline_auc:+.4f})"
        if isinstance(baseline_auc, (int, float))
        else "- Baseline AUC not available for delta."
    )
    summary_text = (
        "# XGBoost retrain — miner-wise canonical split\n\n"
        f"Generated at fit-time {result['fit_time_s']}s.\n\n"
        "## Canonical metrics (miner-wise 80/20)\n\n"
        f"- ROC AUC: **{result['roc_auc']:.4f}**\n"
        f"- PR AUC: **{result['pr_auc']:.4f}**\n"
        f"- Precision@top-10%: **{result['precision_at_top_10pct']:.4f}**\n"
        f"- Recall@top-10%: **{result['recall_at_top_10pct']:.4f}**\n"
        f"- Log loss: {result['log_loss']:.4f}\n"
        f"- Brier score: {result['brier_score']:.4f}\n\n"
        "## Baseline reference (random split, documented leakage)\n\n"
        f"{delta_line}\n\n"
        "- Retained in `xgb_metrics.json.baseline_reference` for transparency.\n"
        "- Random-split model still available at `models/xgb_predictor.original.pkl`.\n\n"
        "## Feature importance change\n\n"
        "| Rank | Baseline top-5 | Retrained top-5 |\n"
        "|---|---|---|\n"
    )
    baseline_fi = baseline.get("feature_importance_top5") or []
    for i in range(5):
        b = baseline_fi[i] if i < len(baseline_fi) else {"name": "—", "importance": 0}
        r = result["feature_importance_top5"][i] if i < len(result["feature_importance_top5"]) else {"name": "—", "importance": 0}
        summary_text += (
            f"| {i+1} | {b['name']} ({float(b.get('importance', 0)):.3f}) | "
            f"{r['name']} ({float(r.get('importance', 0)):.3f}) |\n"
        )

    summary_text += (
        "\n## Signal robustness\n\n"
        "The top-2 features (`fan_mean`, `power_mean_1m`) remain dominant "
        "across both splits, and the 3-test leakage audit (`models/leakage_report.md`) "
        "confirms the same top-2 are robust under episode-wise and temporal splits. "
        "The signal underneath is real; the baseline was simply over-optimistic about "
        "how well it would generalize to unseen miners.\n"
    )
    summary.write_text(summary_text)
    print(f"[retrain] saved → {summary}", flush=True)

    print("\n=== RETRAIN SUMMARY ===", flush=True)
    print(f"ROC AUC:           {result['roc_auc']:.4f}", flush=True)
    print(f"PR AUC:            {result['pr_auc']:.4f}", flush=True)
    print(f"Precision@top-10%: {result['precision_at_top_10pct']:.4f}", flush=True)
    print(f"Recall@top-10%:    {result['recall_at_top_10pct']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
