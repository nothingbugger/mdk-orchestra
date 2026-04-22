"""XGBoost streaming degradation predictor.

Target: predict the probability that a miner's hashrate will drop > 20% within
the next 30 minutes, given the current rolling feature window.

Features (per tick):
  - Rolling mean of hashrate_th over last 12 ticks (1 min)
  - Rolling std  of hashrate_th over last 12 ticks
  - Rolling mean of hashrate_th over last 72 ticks (6 min)
  - Rolling std  of hashrate_th over last 72 ticks
  - Rolling mean of voltage_v   over last 12 ticks
  - Rolling std  of voltage_v   over last 12 ticks
  - Rolling mean of temp_chip_c over last 12 ticks
  - Rolling std  of temp_chip_c over last 12 ticks
  - Rolling mean of power_w     over last 12 ticks
  - Rolling std  of power_w     over last 12 ticks
  - Latest HSI (from KpiUpdate if available, else 0.0)
  - Latest TE  (from KpiUpdate if available, else 50.0)
  - Fan RPM mean (latest tick)

Label construction (TRAINING ONLY):
  A tick is labelled positive (1) if ``fault_injected is not None`` at any
  point within the next 360 ticks (~30 min at 5 s/tick). This look-ahead uses
  the ``fault_injected`` ground-truth only during training.

IMPORTANT: At inference time, ``fault_injected`` is NEVER read. The ``evaluate()``
method signature accepts only ``TelemetryTick`` and ``MinerHistory``; neither
exposes the label to the prediction path. The training code is isolated in
``train.py`` (separate entry point).

Bootstrap:
  - Collect ``BOOTSTRAP_TICKS`` ticks total (across all miners) before labelling.
  - Labels require 360-tick look-ahead, so labelling starts once enough history
    is available.
  - On startup: if ``models/xgb_predictor.pkl`` exists → load and activate.
    Otherwise: accumulate ticks and wait for the ``train.py`` step.

Flag emission:
  - prob >= prob_warn → flag_type=hashrate_degradation, severity=warn
  - prob >= prob_crit → severity=crit
  - Cooldown per (miner_id, flag_type) prevents storms.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from deterministic_tools.base import FlagResult, MinerHistory
from deterministic_tools.config import xgb_cfg
from shared.schemas.events import TelemetryTick

log = structlog.get_logger(__name__)

# IMPORTANT: inference must NOT read fault_injected. This constant is defined
# for documentation only; it is only used in train.py.
_LABEL_LOOKAHEAD_TICKS: int = 360  # 30 min at 5 s/tick

# Minimum ticks in rolling window before computing features (shorter windows
# return None and the flagger silently skips).
_MIN_WINDOW: int = 12

_FEATURE_NAMES = [
    "hr_mean_1m",
    "hr_std_1m",
    "hr_mean_6m",
    "hr_std_6m",
    "v_mean_1m",
    "v_std_1m",
    "temp_mean_1m",
    "temp_std_1m",
    "power_mean_1m",
    "power_std_1m",
    "hsi",
    "te",
    "fan_mean",
]


def _build_features(
    miner_history: MinerHistory,
    latest: TelemetryTick,
) -> list[float] | None:
    """Extract feature vector from miner history.

    IMPORTANT: This function reads only observable telemetry fields.
    ``fault_injected`` is NEVER accessed.

    Args:
        miner_history: rolling per-miner history.
        latest: the most recent tick.

    Returns:
        13-element feature list, or None if window is too small.
    """
    ticks = miner_history.recent_telemetry(_MIN_WINDOW * 6)  # up to 72 ticks
    if len(ticks) < _MIN_WINDOW:
        return None

    last_12 = ticks[-12:]
    all_avail = ticks  # up to 72

    hr_1m = [t.hashrate_th for t in last_12]
    hr_6m = [t.hashrate_th for t in all_avail]
    v_1m = [t.voltage_v for t in last_12]
    temp_1m = [t.temp_chip_c for t in last_12]
    power_1m = [t.power_w for t in last_12]

    fan_mean = float(np.mean(latest.fan_rpm)) if latest.fan_rpm else 5800.0

    kpi = miner_history.last_kpi()
    hsi = kpi.hsi if kpi is not None else 0.0
    te = kpi.te if kpi is not None else 50.0

    return [
        float(np.mean(hr_1m)),
        float(np.std(hr_1m)) if len(hr_1m) > 1 else 0.0,
        float(np.mean(hr_6m)),
        float(np.std(hr_6m)) if len(hr_6m) > 1 else 0.0,
        float(np.mean(v_1m)),
        float(np.std(v_1m)) if len(v_1m) > 1 else 0.0,
        float(np.mean(temp_1m)),
        float(np.std(temp_1m)) if len(temp_1m) > 1 else 0.0,
        float(np.mean(power_1m)),
        float(np.std(power_1m)) if len(power_1m) > 1 else 0.0,
        float(hsi),
        float(te),
        fan_mean,
    ]


class XGBoostFlagger:
    """XGBoost-based hashrate degradation predictor.

    Implements the ``Flagger`` Protocol. Starts in DISABLED state until
    ``models/xgb_predictor.pkl`` is present (loaded at startup) or created by
    running ``python -m deterministic_tools.train``.
    """

    name: str = "xgboost_predictor"

    def __init__(
        self,
        model_path: str | Path = "models/xgb_predictor.pkl",
        sensitivity: str = "medium",
    ) -> None:
        """Initialise the XGBoost flagger.

        Args:
            model_path: path to load/save the trained XGBoost model.
            sensitivity: 'low' | 'medium' | 'high'.
        """
        self._model_path = Path(model_path)
        self._sensitivity = sensitivity
        self._cfg = xgb_cfg(sensitivity)
        self._cooldown_s: float = float(self._cfg["cooldown_s"])
        self._prob_warn: float = float(self._cfg["prob_warn"])
        self._prob_crit: float = float(self._cfg["prob_crit"])
        self._min_confidence: float = float(self._cfg["min_confidence"])

        self._model: Any = None
        self._active: bool = False

        if self._model_path.exists():
            self._load_model()
        else:
            log.info(
                "xgboost_flagger.training_required",
                model_path=str(self._model_path),
                hint="Run: python -m deterministic_tools.train",
            )

    # ------------------------------------------------------------------
    # Public API (Flagger Protocol)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
    ) -> FlagResult | None:
        """Predict hashrate degradation probability.

        IMPORTANT: Does NOT read ``fault_injected``. All features come from
        observable telemetry and rolling KPI history only.

        Args:
            miner_telemetry: latest tick.
            miner_history: per-miner rolling state.

        Returns:
            FlagResult if predicted degradation probability exceeds threshold,
            else None. Returns None immediately if model not yet trained.
        """
        if not self._active:
            return None

        features = _build_features(miner_history, miner_telemetry)
        if features is None:
            return None  # window too short

        X = np.array([features], dtype=np.float32)
        prob = float(self._model.predict_proba(X)[0, 1])

        if prob >= self._prob_crit:
            severity = "crit"
            confidence = min(0.98, 0.70 + prob * 0.28)
        elif prob >= self._prob_warn:
            severity = "warn"
            confidence = 0.50 + prob * 0.40
        else:
            return None

        if confidence < self._min_confidence:
            return None

        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        mid = miner_telemetry.miner_id

        if miner_history.is_on_cooldown("hashrate_degradation", self._cooldown_s, now):
            return None

        miner_history.record_emission("hashrate_degradation", now)
        log.info(
            "xgboost_flagger.flag_raised",
            miner_id=mid,
            severity=severity,
            prob=round(prob, 4),
            confidence=round(confidence, 4),
        )
        return FlagResult(
            flag_type="hashrate_degradation",
            severity=severity,
            confidence=round(confidence, 4),
            source_tool="xgboost_predictor",
            evidence={
                "metric": "hashrate_th",
                "window_min": 1,
                "predicted_prob_drop_20pct_30min": round(prob, 4),
                "features": _FEATURE_NAMES,
                "feature_values": [round(f, 4) for f in features],
                "prob_warn_threshold": self._prob_warn,
                "prob_crit_threshold": self._prob_crit,
            },
            raw_score=round(prob, 4),
        )

    def is_active(self) -> bool:
        """Return True if model is loaded and detection is live."""
        return self._active

    def load_model(self, path: Path | None = None) -> None:
        """Reload model from disk (e.g. after training completes).

        Args:
            path: override model path. Uses constructor path if None.
        """
        if path is not None:
            self._model_path = path
        self._load_model()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        with self._model_path.open("rb") as fh:
            self._model = pickle.load(fh)  # noqa: S301 (trusted local file)
        self._active = True
        log.info("xgboost_flagger.model_loaded", path=str(self._model_path))

    def _save_model(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with self._model_path.open("wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("xgboost_flagger.model_saved", path=str(self._model_path))
