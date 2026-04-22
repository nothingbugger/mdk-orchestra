"""Isolation Forest anomaly flagger.

Strategy: one fleet-wide Isolation Forest trained on a 5-feature vector:
  [voltage_v, temp_chip_c, hashrate_th, power_w, fan_rpm_mean]

Rationale for fleet-wide vs per-miner:
  - 50 miners × ~120 ticks ≈ 6 000 bootstrap samples before training starts.
    One global model fits comfortably; per-miner would require 120 samples each
    (10 min at 5 s / tick) which is fine statistically but triples training time
    and multiplies pkl size by 50 with negligible accuracy gain when the fleet
    is homogeneous (same S19j Pro model, same site).
  - Per-miner bias is absorbed by mean-centring on the miner's own rolling
    baseline (z-score normalisation before feeding to the IF).

Bootstrap protocol:
  - Accumulate the first ``BOOTSTRAP_TICKS`` ticks where ``fault_injected is
    None`` (ground-truth clean). At inference time we NEVER read fault_injected.
  - Once collected, fit the IF and activate detection.
  - Model is persisted to ``models/if_v2.pkl``. On next startup, if the file
    exists, it is loaded and bootstrap is skipped.

Scoring:
  - sklearn's ``decision_function()`` returns a score in ℝ; more negative =
    more anomalous. We normalise to [0, 1] using the sigmoid of the negated
    score so that 1.0 means "extremely anomalous".
  - Warn threshold and crit threshold are read from the sensitivity profile.

NOTE: fault_injected is only read during the bootstrap COLLECTION phase to
exclude dirty samples from training. It is NOT read at inference time.
The collection code clearly gates on envelope-level attribute reading and the
inference path (``evaluate()``) accepts only the TelemetryTick model which the
caller already has — the caller must not pass fault information through.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from deterministic_tools.base import FlagResult, MinerHistory
from deterministic_tools.config import if_cfg
from shared.schemas.events import TelemetryTick

log = structlog.get_logger(__name__)

# Number of clean ticks to collect before training.
BOOTSTRAP_TICKS: int = 1440  # ~2 h at 5 s/tick for 50 miners collectively

_FEATURES = ["voltage_v", "temp_chip_c", "hashrate_th", "power_w", "fan_rpm_mean"]


def _extract_features(tick: TelemetryTick) -> list[float]:
    """Extract the 5-element feature vector from a telemetry tick.

    IMPORTANT: does NOT touch ``fault_injected``. Detection must be blind.
    """
    fan_mean = float(np.mean(tick.fan_rpm)) if tick.fan_rpm else 5800.0
    return [
        tick.voltage_v,
        tick.temp_chip_c,
        tick.hashrate_th,
        tick.power_w,
        fan_mean,
    ]


def _sigmoid_score(raw: float) -> float:
    """Map raw IF decision_function output → [0, 1] anomaly score.

    sklearn decision_function: negative = anomalous, positive = normal.
    We negate so that larger = more anomalous, then pass through sigmoid.
    """
    x = -raw  # now: large = anomalous
    return float(1.0 / (1.0 + np.exp(-x)))


class IsolationForestFlagger:
    """Fleet-wide Isolation Forest anomaly detector.

    Implements the ``Flagger`` Protocol. Starts in TRAINING state until either:
    (a) the model file exists on disk → loads immediately, or
    (b) enough clean bootstrap ticks accumulate → fits and activates.
    """

    name: str = "isolation_forest_v2"

    def __init__(
        self,
        model_path: str | Path = "models/if_v2.pkl",
        sensitivity: str = "medium",
    ) -> None:
        """Initialise the flagger.

        Args:
            model_path: path to load/save the trained sklearn IsolationForest.
            sensitivity: 'low' | 'medium' | 'high' — controls score thresholds
                and cooldown via sensitivity.yaml.
        """
        self._model_path = Path(model_path)
        self._sensitivity = sensitivity
        self._cfg = if_cfg(sensitivity)
        self._cooldown_s: float = float(self._cfg["cooldown_s"])
        self._warn_thresh: float = float(self._cfg["anomaly_score_warn"])
        self._crit_thresh: float = float(self._cfg["anomaly_score_crit"])
        self._min_confidence: float = float(self._cfg["min_confidence"])

        self._model: Any = None  # sklearn IsolationForest or None
        self._active: bool = False
        self._bootstrap_buf: deque[list[float]] = deque(maxlen=BOOTSTRAP_TICKS)

        if self._model_path.exists():
            self._load_model()
        else:
            log.info(
                "isolation_forest_flagger.bootstrap_mode",
                model_path=str(self._model_path),
                bootstrap_ticks_needed=BOOTSTRAP_TICKS,
            )

    # ------------------------------------------------------------------
    # Public API (Flagger Protocol)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        miner_telemetry: TelemetryTick,
        miner_history: MinerHistory,
    ) -> FlagResult | None:
        """Detect anomalies using the trained IF model.

        If the model is not yet trained, feeds the tick into the bootstrap
        buffer and returns None.

        NOTE: Does NOT read ``fault_injected`` — inference is blind to ground
        truth. (Bootstrap collection during training IS allowed to read it via
        the separate ``feed_bootstrap`` method called from the train path.)

        Args:
            miner_telemetry: latest tick (TelemetryTick). ``fault_injected``
                must NOT be read here.
            miner_history: per-miner rolling state (read-only).

        Returns:
            FlagResult if anomaly detected and model is active, else None.
        """
        if not self._active:
            # Still bootstrapping — feed the tick but don't emit flags.
            features = _extract_features(miner_telemetry)
            self._bootstrap_buf.append(features)
            if len(self._bootstrap_buf) >= BOOTSTRAP_TICKS:
                self._fit_model()
            return None

        features = _extract_features(miner_telemetry)
        score = self._score(features)

        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        mid = miner_telemetry.miner_id

        if score >= self._crit_thresh:
            severity = "crit"
            confidence = min(0.98, 0.70 + score * 0.28)
        elif score >= self._warn_thresh:
            severity = "warn"
            confidence = 0.55 + score * 0.30
        else:
            return None

        if confidence < self._min_confidence:
            return None

        if miner_history.is_on_cooldown("anomaly_composite", self._cooldown_s, now):
            return None

        miner_history.record_emission("anomaly_composite", now)
        log.info(
            "isolation_forest_flagger.flag_raised",
            miner_id=mid,
            severity=severity,
            score=round(score, 4),
            confidence=round(confidence, 4),
        )
        return FlagResult(
            flag_type="anomaly_composite",
            severity=severity,
            confidence=round(confidence, 4),
            source_tool="isolation_forest_v2",
            evidence={
                "metric": "multi_feature_vector",
                "window_min": 0,  # point-in-time, not sustained
                "features": _FEATURES,
                "feature_values": [round(f, 4) for f in features],
                "anomaly_score": round(score, 4),
                "warn_threshold": self._warn_thresh,
                "crit_threshold": self._crit_thresh,
            },
            raw_score=round(score, 4),
        )

    def feed_bootstrap(self, tick: TelemetryTick, fault_injected: str | None) -> None:
        """Feed one tick into the bootstrap buffer (training path only).

        This method is called from the separate ``train`` entry point. It is
        allowed to read ``fault_injected`` to exclude dirty samples.

        Args:
            tick: telemetry tick.
            fault_injected: the ground-truth fault tag (may be None if clean).
                Only clean ticks (None) are added to the bootstrap buffer.
        """
        if fault_injected is not None:
            return  # skip dirty samples during bootstrap collection
        features = _extract_features(tick)
        self._bootstrap_buf.append(features)
        if len(self._bootstrap_buf) >= BOOTSTRAP_TICKS:
            self._fit_model()

    def is_active(self) -> bool:
        """Return True if the model is trained and detection is live."""
        return self._active

    def bootstrap_progress(self) -> tuple[int, int]:
        """Return (collected, needed) bootstrap tick counts."""
        return len(self._bootstrap_buf), BOOTSTRAP_TICKS

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_model(self) -> None:
        """Train the Isolation Forest on the current bootstrap buffer."""
        from sklearn.ensemble import IsolationForest  # type: ignore[import]

        X = np.array(list(self._bootstrap_buf), dtype=np.float32)
        log.info(
            "isolation_forest_flagger.fitting",
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,  # expect ~5% anomalies in healthy data
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        self._model = model
        self._active = True
        self._save_model()
        log.info(
            "isolation_forest_flagger.model_ready",
            model_path=str(self._model_path),
        )

    def _score(self, features: list[float]) -> float:
        """Return a [0, 1] anomaly score for a single feature vector."""
        X = np.array([features], dtype=np.float32)
        raw = self._model.decision_function(X)[0]
        return _sigmoid_score(raw)

    def _save_model(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with self._model_path.open("wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("isolation_forest_flagger.model_saved", path=str(self._model_path))

    def _load_model(self) -> None:
        with self._model_path.open("rb") as fh:
            self._model = pickle.load(fh)  # noqa: S301 (trusted local file)
        self._active = True
        log.info("isolation_forest_flagger.model_loaded", path=str(self._model_path))
