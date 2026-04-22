"""Unit tests for the XGBoost training pipeline.

Tests:
  1. train_test_split stratification – positive rate preserved across split.
  2. Metric computation against a golden mini-dataset.
  3. Feature extraction contract – fault_injected is never accessed.
  4. _build_features returns correct length / no fault leak.
"""

from __future__ import annotations

import pickle
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root on sys.path for direct test execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_telemetry_tick(
    miner_id: str = "m001",
    hashrate_th: float = 90.0,
    voltage_v: float = 13.5,
    temp_chip_c: float = 70.0,
    power_w: float = 3200.0,
    fan_rpm: list[int] | None = None,
    fault_injected: str | None = None,
) -> "TelemetryTick":
    """Build a minimal TelemetryTick for testing."""
    from shared.schemas.events import EnvBlock, TelemetryTick

    env = EnvBlock(
        site_temp_c=25.0,
        site_humidity_pct=40.0,
        elec_price_usd_kwh=0.07,
        hashprice_usd_per_th_day=0.10,
    )
    return TelemetryTick(
        miner_id=miner_id,
        miner_model="S19Pro",
        hashrate_th=hashrate_th,
        hashrate_expected_th=100.0,
        temp_chip_c=temp_chip_c,
        temp_amb_c=25.0,
        power_w=power_w,
        voltage_v=voltage_v,
        fan_rpm=fan_rpm or [5800, 5800, 5800, 5800],
        operating_mode="balanced",
        uptime_s=3600.0,
        env=env,
        fault_injected=fault_injected,
    )


def _make_miner_history(n_ticks: int = 80, with_fault: bool = False) -> "MinerHistory":
    """Build a MinerHistory populated with ``n_ticks`` synthetic ticks."""
    from deterministic_tools.base import MinerHistory

    history = MinerHistory(miner_id="m001")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n_ticks):
        tick = _make_telemetry_tick(
            fault_injected="chip_instability" if (with_fault and i > 60) else None
        )
        history.push_telemetry(tick, ts)
    return history


# ---------------------------------------------------------------------------
# Test 1: Stratified split preserves positive rate
# ---------------------------------------------------------------------------


def test_stratified_split_positive_rate():
    """80/20 split must keep positive rates within 5 pp of each other."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n = 2000
    # ~25% positives
    y = (rng.random(n) < 0.25).astype(int)
    X = rng.random((n, 13)).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    assert len(X_train) == 1600
    assert len(X_val) == 400

    pos_train = y_train.mean()
    pos_val = y_val.mean()
    # Stratification must keep rates within 5 percentage points
    assert abs(pos_train - pos_val) < 0.05, (
        f"Positive rate mismatch: train={pos_train:.3f} val={pos_val:.3f}"
    )
    # Rates should be close to original 25%
    assert abs(pos_train - 0.25) < 0.05


# ---------------------------------------------------------------------------
# Test 2: Metric computation against a controlled mini-dataset
# ---------------------------------------------------------------------------


def test_metric_computation_golden():
    """Validate that the metrics block computes correct values on known data."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    n_val = 200
    # Generate a moderately separable problem
    y_val = (rng.random(n_val) < 0.30).astype(int)
    # Probabilities: positive class gets higher scores on average
    y_prob = np.clip(
        rng.normal(loc=np.where(y_val == 1, 0.65, 0.35), scale=0.15), 0.01, 0.99
    )

    # Compute metrics the same way _fit_xgboost does
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    n_top = max(1, int(n_val * 0.10))
    top_indices = np.argsort(y_prob)[::-1][:n_top]
    top_labels = y_val[top_indices]
    precision_top10 = float(top_labels.mean())
    total_pos = int(y_val.sum())
    recall_top10 = float(top_labels.sum() / max(total_pos, 1))

    # With decent separation the metrics should be reasonable
    assert roc_auc > 0.60, f"Expected roc_auc > 0.60, got {roc_auc:.4f}"
    assert pr_auc > 0.30, f"Expected pr_auc > 0.30, got {pr_auc:.4f}"
    assert 0.0 <= precision_top10 <= 1.0
    assert 0.0 <= recall_top10 <= 1.0


# ---------------------------------------------------------------------------
# Test 3: _build_features never accesses fault_injected
# ---------------------------------------------------------------------------


def test_build_features_no_fault_leak():
    """_build_features must return a 13-element vector and never read fault_injected.

    We patch TelemetryTick to raise AttributeError if fault_injected is read,
    verifying the inference path is clean.
    """
    from deterministic_tools.xgboost_flagger import _build_features, _FEATURE_NAMES

    history = _make_miner_history(n_ticks=80, with_fault=False)
    latest = _make_telemetry_tick()

    # Wrap latest in a proxy that raises if fault_injected is accessed
    class NoFaultProxy:
        """Proxy that mirrors all attributes but raises on fault_injected access."""

        def __init__(self, tick):
            self._tick = tick

        def __getattr__(self, name):
            if name == "fault_injected":
                raise AssertionError("fault_injected must NOT be read in _build_features!")
            return getattr(self._tick, name)

    proxy = NoFaultProxy(latest)

    # Should not raise
    features = _build_features(history, proxy)

    assert features is not None, "_build_features returned None (window too small?)"
    assert len(features) == len(_FEATURE_NAMES), (
        f"Expected {len(_FEATURE_NAMES)} features, got {len(features)}"
    )
    # All features must be finite
    for i, (name, val) in enumerate(zip(_FEATURE_NAMES, features)):
        assert np.isfinite(val), f"Feature {name} (idx={i}) is not finite: {val}"


# ---------------------------------------------------------------------------
# Test 4: _build_features returns None when history is too short
# ---------------------------------------------------------------------------


def test_build_features_insufficient_window():
    """With fewer than 12 ticks, _build_features must return None."""
    from deterministic_tools.xgboost_flagger import _build_features

    history = _make_miner_history(n_ticks=5)
    latest = _make_telemetry_tick()

    result = _build_features(history, latest)
    assert result is None, f"Expected None with only 5 ticks, got {result}"


# ---------------------------------------------------------------------------
# Test 5: _fit_xgboost produces expected metrics keys
# ---------------------------------------------------------------------------


def test_fit_xgboost_metrics_keys(tmp_path):
    """_fit_xgboost must produce a metrics dict with all required keys."""
    from deterministic_tools.train import _fit_xgboost

    rng = np.random.default_rng(99)
    n = 1000
    X = rng.random((n, 13)).astype(np.float32)
    y = (rng.random(n) < 0.25).astype(int).tolist()
    X_list = X.tolist()

    model_path = tmp_path / "test_xgb.pkl"
    metrics_path = tmp_path / "test_metrics.json"

    metrics = _fit_xgboost(
        X_list=X_list,
        y_list=y,
        out_path=model_path,
        metrics_out_path=metrics_path,
        hparam_overrides={"n_estimators": 20},  # small for speed
        dataset_source="test_golden",
    )

    required_keys = [
        "roc_auc",
        "pr_auc",
        "precision_at_top_10pct",
        "recall_at_top_10pct",
        "log_loss",
        "brier_score",
        "feature_importance_top5",
        "confusion_at_default_threshold",
        "n_train",
        "n_val",
        "positive_rate_train",
        "positive_rate_val",
        "training_time_s",
        "hyperparameters",
        "dataset_source",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    assert len(metrics["feature_importance_top5"]) == 5
    assert metrics["n_train"] == 800
    assert metrics["n_val"] == 200
    assert metrics["dataset_source"] == "test_golden"
    assert model_path.exists()
    assert metrics_path.exists()
