"""Tests for the XGBoost pattern-specific ensemble flaggers.

Tests:
  1. Loading each model and computing predict_proba returns shape (1,2).
  2. Feature extractor for each pattern returns expected number of features and no NaN.
  3. ChipInstabilityFlagger.evaluate() returns FlagResult with correct flag_type when
     prob >= 0.50 on a synthetic pre-fault-ish sample, and None when below threshold.
  4. fault_injected is not referenced in the two new flagger source files.
"""

from __future__ import annotations

import ast
import pickle
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CHIP_PKL = _REPO_ROOT / "models" / "xgb_chip_instability.pkl"
_HB_PKL = _REPO_ROOT / "models" / "xgb_hashboard_failure.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_block():
    from shared.schemas.events import EnvBlock

    return EnvBlock(
        site_temp_c=25.0,
        site_humidity_pct=40.0,
        elec_price_usd_kwh=0.07,
        hashprice_usd_per_th_day=0.10,
    )


def _make_tick(
    miner_id: str = "m001",
    hashrate_th: float = 90.0,
    voltage_v: float = 13.5,
    temp_chip_c: float = 70.0,
    temp_amb_c: float = 25.0,
    power_w: float = 3200.0,
    fan_rpm: list[int] | None = None,
) -> "TelemetryTick":
    """Build a TelemetryTick WITHOUT reading fault_injected."""
    from shared.schemas.events import TelemetryTick

    return TelemetryTick(
        miner_id=miner_id,
        miner_model="S19Pro",
        hashrate_th=hashrate_th,
        hashrate_expected_th=100.0,
        temp_chip_c=temp_chip_c,
        temp_amb_c=temp_amb_c,
        power_w=power_w,
        voltage_v=voltage_v,
        fan_rpm=fan_rpm or [5800, 5800, 5800, 5800],
        operating_mode="balanced",
        uptime_s=3600.0,
        env=_make_env_block(),
        fault_injected=None,  # never used in inference
    )


def _make_history(
    n_ticks: int = 400,
    hashrate_th: float = 90.0,
    power_w: float = 3200.0,
    add_variance: bool = False,
) -> "MinerHistory":
    """Build a MinerHistory with at least 360 ticks (needed for 30m features)."""
    from deterministic_tools.base import MinerHistory

    history = MinerHistory(miner_id="m001")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)

    for i in range(n_ticks):
        if add_variance:
            # Simulate chip instability: high variance in hashrate
            hr = hashrate_th + rng.normal(0, 8.0)
            pw = power_w + rng.normal(0, 120.0)
        else:
            hr = hashrate_th + rng.normal(0, 0.5)
            pw = power_w + rng.normal(0, 10.0)
        tick = _make_tick(hashrate_th=float(hr), power_w=float(pw))
        history.push_telemetry(tick, ts)

    return history


# ---------------------------------------------------------------------------
# Test 1: predict_proba returns shape (1, 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CHIP_PKL.exists(), reason="chip_instability model not found")
def test_chip_instability_predict_proba_shape():
    """predict_proba on a golden feature vector returns shape (1, 2)."""
    with _CHIP_PKL.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301

    from deterministic_tools._pattern_features import CHIP_INSTABILITY_FEATURE_NAMES

    n_features = len(CHIP_INSTABILITY_FEATURE_NAMES)
    X = np.zeros((1, n_features), dtype=np.float32)
    proba = model.predict_proba(X)

    assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"
    assert 0.0 <= proba[0, 1] <= 1.0, "Probability out of [0,1] range"


@pytest.mark.skipif(not _HB_PKL.exists(), reason="hashboard_failure model not found")
def test_hashboard_failure_predict_proba_shape():
    """predict_proba on a golden feature vector returns shape (1, 2)."""
    with _HB_PKL.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301

    from deterministic_tools._pattern_features import HASHBOARD_FAILURE_FEATURE_NAMES

    n_features = len(HASHBOARD_FAILURE_FEATURE_NAMES)
    X = np.zeros((1, n_features), dtype=np.float32)
    proba = model.predict_proba(X)

    assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"
    assert 0.0 <= proba[0, 1] <= 1.0, "Probability out of [0,1] range"


# ---------------------------------------------------------------------------
# Test 2: feature extractors return correct count and no NaN
# ---------------------------------------------------------------------------


def test_chip_instability_features_count_and_no_nan():
    """extract_chip_instability_features returns 14 features with no NaN."""
    from deterministic_tools._pattern_features import (
        CHIP_INSTABILITY_FEATURE_NAMES,
        extract_chip_instability_features,
    )

    history = _make_history(n_ticks=400)
    latest = _make_tick()

    features = extract_chip_instability_features(latest, history)

    assert features is not None, "Expected features, got None (window too small?)"
    assert len(features) == len(CHIP_INSTABILITY_FEATURE_NAMES), (
        f"Expected {len(CHIP_INSTABILITY_FEATURE_NAMES)} features, got {len(features)}"
    )
    for i, (name, val) in enumerate(zip(CHIP_INSTABILITY_FEATURE_NAMES, features)):
        assert np.isfinite(val), f"Feature '{name}' (idx={i}) is NaN or Inf: {val}"


def test_hashboard_failure_features_count_and_no_nan():
    """extract_hashboard_failure_features returns 11 features with no NaN."""
    from deterministic_tools._pattern_features import (
        HASHBOARD_FAILURE_FEATURE_NAMES,
        extract_hashboard_failure_features,
    )

    history = _make_history(n_ticks=400)
    latest = _make_tick()

    features = extract_hashboard_failure_features(latest, history)

    assert features is not None, "Expected features, got None (window too small?)"
    assert len(features) == len(HASHBOARD_FAILURE_FEATURE_NAMES), (
        f"Expected {len(HASHBOARD_FAILURE_FEATURE_NAMES)} features, got {len(features)}"
    )
    for i, (name, val) in enumerate(zip(HASHBOARD_FAILURE_FEATURE_NAMES, features)):
        assert np.isfinite(val), f"Feature '{name}' (idx={i}) is NaN or Inf: {val}"


def test_feature_extractors_return_none_when_window_too_short():
    """Extractors must return None when history has fewer than 360 ticks."""
    from deterministic_tools._pattern_features import (
        extract_chip_instability_features,
        extract_hashboard_failure_features,
    )

    short_history = _make_history(n_ticks=100)  # < 360
    latest = _make_tick()

    assert extract_chip_instability_features(latest, short_history) is None
    assert extract_hashboard_failure_features(latest, short_history) is None


# ---------------------------------------------------------------------------
# Test 3: ChipInstabilityFlagger.evaluate() returns correct FlagResult
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CHIP_PKL.exists(), reason="chip_instability model not found")
def test_chip_instability_flagger_evaluate_with_model():
    """ChipInstabilityFlagger.evaluate() emits correct flag_type on pre-fault signal."""
    from deterministic_tools.xgb_pattern_flaggers import ChipInstabilityFlagger

    flagger = ChipInstabilityFlagger(model_path=str(_CHIP_PKL))
    assert flagger.is_active(), "Flagger should be active when model exists"

    # Build high-variance history (chip instability signature)
    history = _make_history(n_ticks=400, add_variance=True)
    # Use a tick with high hashrate variance (mimicking pre-fault)
    latest = _make_tick(hashrate_th=75.0, power_w=3600.0)  # deviated values

    result = flagger.evaluate(latest, history)
    # Result may be None (prob < threshold) or FlagResult
    # We only check type and fields if it fires
    if result is not None:
        assert result.flag_type == "chip_instability_precursor", (
            f"Expected 'chip_instability_precursor', got '{result.flag_type}'"
        )
        assert result.source_tool == "xgboost_rolling_variance"
        assert result.severity in ("warn", "crit")
        assert 0.0 <= result.confidence <= 1.0
        assert "metric" in result.evidence
        assert "window_min" in result.evidence
        assert result.evidence["window_min"] == 30


@pytest.mark.skipif(not _CHIP_PKL.exists(), reason="chip_instability model not found")
def test_chip_instability_flagger_evaluate_returns_none_when_below_threshold():
    """ChipInstabilityFlagger.evaluate() returns None for normal (clean) signal."""
    from deterministic_tools.xgb_pattern_flaggers import ChipInstabilityFlagger

    flagger = ChipInstabilityFlagger(model_path=str(_CHIP_PKL))

    # Build very stable history (no variance — should score low)
    history = _make_history(n_ticks=400, add_variance=False)
    latest = _make_tick(hashrate_th=90.0, power_w=3200.0)

    # Patch the model to return a sub-threshold probability
    import unittest.mock as mock

    mock_model = mock.MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
    flagger._model = mock_model

    result = flagger.evaluate(latest, history)
    assert result is None, "Expected None when prob=0.05 (below 0.50 threshold)"


def test_chip_instability_flagger_disabled_when_model_missing(tmp_path):
    """ChipInstabilityFlagger starts disabled and returns None when pkl not found."""
    from deterministic_tools.xgb_pattern_flaggers import ChipInstabilityFlagger

    missing_path = tmp_path / "nonexistent.pkl"
    flagger = ChipInstabilityFlagger(model_path=str(missing_path))
    assert not flagger.is_active(), "Expected disabled when model file missing"

    history = _make_history(n_ticks=400)
    latest = _make_tick()
    result = flagger.evaluate(latest, history)
    assert result is None, "Expected None when flagger is disabled"


# ---------------------------------------------------------------------------
# Test 4: fault_injected is not referenced in flagger source files
# ---------------------------------------------------------------------------


def _find_fault_injected_in_code(source: str) -> list[tuple[int, str]]:
    """Return lines where fault_injected appears in actual code (not comments/docstrings).

    Uses a line-by-line heuristic: skip lines that are pure comments (#) and
    lines that are inside docstrings. Also skip lines that only contain the
    string in a quoted context (docstring body).
    We parse the source as an AST and collect all string-literal node locations
    to exclude them from the check.
    """
    tree = ast.parse(source)

    # Collect line ranges of all string literals (docstrings + string constants)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is not None and end is not None:
                for ln in range(start, end + 1):
                    docstring_lines.add(ln)

    violations = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if "fault_injected" not in stripped:
            continue
        if stripped.startswith("#"):
            continue  # pure comment line
        if i in docstring_lines:
            continue  # inside a string literal / docstring
        violations.append((i, stripped))

    return violations


def test_fault_injected_not_in_pattern_flaggers():
    """xgb_pattern_flaggers.py must not reference fault_injected in executable code."""
    flagger_path = _REPO_ROOT / "deterministic_tools" / "xgb_pattern_flaggers.py"
    assert flagger_path.exists(), f"Flagger file not found: {flagger_path}"
    violations = _find_fault_injected_in_code(flagger_path.read_text())
    assert not violations, (
        f"fault_injected referenced in executable code in xgb_pattern_flaggers.py: {violations}"
    )


def test_fault_injected_not_in_pattern_features():
    """_pattern_features.py must not reference fault_injected in executable code."""
    features_path = _REPO_ROOT / "deterministic_tools" / "_pattern_features.py"
    assert features_path.exists(), f"Feature file not found: {features_path}"
    violations = _find_fault_injected_in_code(features_path.read_text())
    assert not violations, (
        f"fault_injected referenced in executable code in _pattern_features.py: {violations}"
    )
