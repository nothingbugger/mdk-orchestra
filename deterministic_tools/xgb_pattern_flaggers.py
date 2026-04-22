"""XGBoost pattern-specific flaggers for chip_instability and hashboard_failure.

Two flaggers:
  ChipInstabilityFlagger  — rolling_variance_spike pattern
  HashboardFailureFlagger — thermal_electrical_decoupling pattern

Both implement the ``Flagger`` Protocol from ``deterministic_tools/base.py``.

IMPORTANT: Neither flagger reads ``fault_injected`` at inference time. All features
come from observable telemetry and rolling history via ``_pattern_features.py``.

Severity thresholds (uniform for both):
  prob >= 0.75 → crit
  prob >= 0.50 → warn
  else         → no flag (return None)

Cooldown: 5 min per (miner_id, flag_type).
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from deterministic_tools._pattern_features import (
    CHIP_INSTABILITY_FEATURE_NAMES,
    HASHBOARD_FAILURE_FEATURE_NAMES,
    extract_chip_instability_features,
    extract_hashboard_failure_features,
)
from deterministic_tools.base import FlagResult, MinerHistory
from shared.schemas.events import TelemetryTick

log = structlog.get_logger(__name__)

# IMPORTANT: fault_injected is never read at inference. This comment is kept
# here for auditing; grep for "fault_injected" should return zero matches in
# this file and in _pattern_features.py.

_COOLDOWN_S: float = 300.0  # 5 minutes

_PROB_CRIT: float = 0.75
_PROB_WARN: float = 0.50

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_model(model_path: Path) -> Any | None:
    """Load a pkl model, returning None (disabled) if the file doesn't exist."""
    if not model_path.exists():
        return None
    with model_path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 (trusted local file)


def _score_to_severity(prob: float) -> str | None:
    """Map probability to severity string, or None if below threshold."""
    if prob >= _PROB_CRIT:
        return "crit"
    if prob >= _PROB_WARN:
        return "warn"
    return None


# ---------------------------------------------------------------------------
# ChipInstabilityFlagger
# ---------------------------------------------------------------------------


class ChipInstabilityFlagger:
    """XGBoost chip_instability precursor detector.

    Implements the ``Flagger`` Protocol. Pattern: rolling_variance_spike.
    Emits flag_type="chip_instability_precursor", source_tool="xgboost_rolling_variance".

    Starts disabled if ``models/xgb_chip_instability.pkl`` is not present.
    """

    name: str = "xgboost_chip_instability"

    def __init__(
        self,
        model_path: str | Path = "models/xgb_chip_instability.pkl",
    ) -> None:
        """Initialise the flagger.

        Args:
            model_path: path to the trained XGBoost pkl file.
        """
        self._model_path = Path(model_path)
        self._model: Any = _load_model(self._model_path)

        if self._model is None:
            log.warning(
                "chip_instability_flagger.disabled",
                model_path=str(self._model_path),
                hint="Run: python scripts/train_xgb_ensemble.py",
            )
        else:
            log.info(
                "chip_instability_flagger.loaded",
                model_path=str(self._model_path),
            )

    def is_active(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._model is not None

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
    ) -> FlagResult | None:
        """Predict chip_instability precursor probability.

        IMPORTANT: Does NOT read ``fault_injected``. All features come from
        observable telemetry and rolling history only.

        Args:
            miner_telemetry: latest tick for this miner.
            miner_history: rolling per-miner state (read-only).

        Returns:
            FlagResult if prob ≥ 0.50, else None.
        """
        if self._model is None:
            return None

        features = extract_chip_instability_features(miner_telemetry, miner_history)
        if features is None:
            return None  # window too short for 30m rolling features

        X = np.array([features], dtype=np.float32)
        prob = float(self._model.predict_proba(X)[0, 1])

        severity = _score_to_severity(prob)
        if severity is None:
            return None

        now = datetime.now(tz=timezone.utc)
        flag_type = "chip_instability_precursor"

        if miner_history.is_on_cooldown(flag_type, _COOLDOWN_S, now):
            return None

        miner_history.record_emission(flag_type, now)

        # Top feature by importance (index 0 in feature list = hashrate_th_30m_std)
        top_feat_name = CHIP_INSTABILITY_FEATURE_NAMES[0]
        top_feat_val = features[0]

        log.info(
            "chip_instability_flagger.flag_raised",
            miner_id=miner_telemetry.miner_id,
            severity=severity,
            prob=round(prob, 4),
        )

        return FlagResult(
            flag_type=flag_type,
            severity=severity,
            confidence=round(prob, 4),
            source_tool="xgboost_rolling_variance",
            raw_score=round(prob, 4),
            evidence={
                "metric": "hashrate_th_30m_std",
                "window_min": 30,
                "predicted_prob": round(prob, 4),
                "top_feature_name": top_feat_name,
                "top_feature_value": round(top_feat_val, 4),
                "features": CHIP_INSTABILITY_FEATURE_NAMES,
                "feature_values": [round(f, 4) for f in features],
            },
        )


# ---------------------------------------------------------------------------
# HashboardFailureFlagger
# ---------------------------------------------------------------------------


class HashboardFailureFlagger:
    """XGBoost hashboard_failure precursor detector.

    Implements the ``Flagger`` Protocol. Pattern: thermal_electrical_decoupling.
    Emits flag_type="hashboard_failure_precursor", source_tool="xgboost_thermal_electrical".

    Starts disabled if ``models/xgb_hashboard_failure.pkl`` is not present.
    """

    name: str = "xgboost_hashboard_failure"

    def __init__(
        self,
        model_path: str | Path = "models/xgb_hashboard_failure.pkl",
    ) -> None:
        """Initialise the flagger.

        Args:
            model_path: path to the trained XGBoost pkl file.
        """
        self._model_path = Path(model_path)
        self._model: Any = _load_model(self._model_path)

        if self._model is None:
            log.warning(
                "hashboard_failure_flagger.disabled",
                model_path=str(self._model_path),
                hint="Run: python scripts/train_xgb_ensemble.py",
            )
        else:
            log.info(
                "hashboard_failure_flagger.loaded",
                model_path=str(self._model_path),
            )

    def is_active(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._model is not None

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
    ) -> FlagResult | None:
        """Predict hashboard_failure precursor probability.

        IMPORTANT: Does NOT read ``fault_injected``. All features come from
        observable telemetry and rolling history only.

        Args:
            miner_telemetry: latest tick for this miner.
            miner_history: rolling per-miner state (read-only).

        Returns:
            FlagResult if prob ≥ 0.50, else None.
        """
        if self._model is None:
            return None

        features = extract_hashboard_failure_features(miner_telemetry, miner_history)
        if features is None:
            return None  # window too short for 30m rolling features

        X = np.array([features], dtype=np.float32)
        prob = float(self._model.predict_proba(X)[0, 1])

        severity = _score_to_severity(prob)
        if severity is None:
            return None

        now = datetime.now(tz=timezone.utc)
        flag_type = "hashboard_failure_precursor"

        if miner_history.is_on_cooldown(flag_type, _COOLDOWN_S, now):
            return None

        miner_history.record_emission(flag_type, now)

        # Top feature by importance (index 0 = temp_per_power)
        top_feat_name = HASHBOARD_FAILURE_FEATURE_NAMES[0]
        top_feat_val = features[0]

        log.info(
            "hashboard_failure_flagger.flag_raised",
            miner_id=miner_telemetry.miner_id,
            severity=severity,
            prob=round(prob, 4),
        )

        return FlagResult(
            flag_type=flag_type,
            severity=severity,
            confidence=round(prob, 4),
            source_tool="xgboost_thermal_electrical",
            raw_score=round(prob, 4),
            evidence={
                "metric": "temp_per_power",
                "window_min": 30,
                "predicted_prob": round(prob, 4),
                "top_feature_name": top_feat_name,
                "top_feature_value": round(top_feat_val, 6),
                "features": HASHBOARD_FAILURE_FEATURE_NAMES,
                "feature_values": [round(f, 4) for f in features],
            },
        )
