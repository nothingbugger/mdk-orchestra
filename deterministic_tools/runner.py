"""Main detector loop — tails the telemetry stream and emits flag_raised events.

Public API matches ``shared/specs/interfaces.md`` §3:

    run_detector(
        input_stream, flag_output,
        predictor_model_path, anomaly_model_path,
        sensitivity,
    ) -> None

Architecture
------------
1. Tail ``telemetry.jsonl`` for ``telemetry_tick`` events.
2. Maintain one ``MinerHistory`` per miner_id (bounded deque, ~1 h rolling).
3. On each tick, run all three flaggers (rule engine always active; ML flaggers
   skip gracefully if their model is not yet trained).
4. For each non-None FlagResult, build a ``FlagRaised`` Pydantic model and emit
   via ``write_event("flag_raised", "detector", ...)``.

KPI stream
----------
The detector also tails ``kpis.jsonl`` for ``kpi_update`` events and feeds
them into the relevant MinerHistory. KPI-enriched features improve XGBoost
accuracy. KPI tailing is non-blocking (second thread); missed KPIs gracefully
degrade to HSI=0, TE=50 fallbacks.

Flag ID
-------
Generated as a sequential counter per process run. Not globally unique across
process restarts; use the event's ``ts`` for dedup if needed.
"""

from __future__ import annotations

import itertools
import threading
from pathlib import Path
from typing import Any

import structlog

from deterministic_tools.base import Flagger, FlagResult, MinerHistory
from deterministic_tools.config import VALID_SENSITIVITIES
from deterministic_tools.isolation_forest_flagger import IsolationForestFlagger
from deterministic_tools.rule_engine_flagger import RuleEngineFlagger
from deterministic_tools.xgboost_flagger import XGBoostFlagger
from deterministic_tools.xgb_pattern_flaggers import ChipInstabilityFlagger, HashboardFailureFlagger
from shared.event_bus import tail_events, write_event
from shared.paths import stream_paths
from shared.schemas.events import FlagEvidence, FlagRaised, KpiUpdate, TelemetryTick

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Flag ID generator
# ---------------------------------------------------------------------------

_flag_counter = itertools.count(1)


def _next_flag_id() -> str:
    return f"flg_{next(_flag_counter):05d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_detector(
    input_stream: str | None = None,
    flag_output: str | None = None,
    predictor_model_path: str = "models/xgb_predictor.pkl",
    anomaly_model_path: str = "models/if_v2.pkl",
    chip_instability_model_path: str = "models/xgb_chip_instability.pkl",
    hashboard_failure_model_path: str = "models/xgb_hashboard_failure.pkl",
    sensitivity: str = "medium",
    disable_chip_instability: bool = False,
    disable_hashboard_failure: bool = True,
    disable_isolation_forest: bool = True,
    stop_when: Any = None,
) -> None:
    """Tail the telemetry stream, run all flaggers, emit flag_raised events.

    Args:
        input_stream: path to the JSONL telemetry stream. Defaults to the
            canonical ``stream_paths().telemetry`` path.
        flag_output: path to write flag_raised events. Defaults to the
            canonical ``stream_paths().flags`` path.
        predictor_model_path: path to the XGBoost pkl file.
        anomaly_model_path: path to the Isolation Forest pkl file.
        chip_instability_model_path: path to chip_instability XGBoost pkl.
        hashboard_failure_model_path: path to hashboard_failure XGBoost pkl.
        sensitivity: 'low' | 'medium' | 'high'.
        disable_chip_instability: if True, skip ChipInstabilityFlagger (ablation).
        disable_hashboard_failure: DEFAULT True — hashboard_failure predictor
            was trained on a n=2-miner pre-fault sample and cross-miner
            generalization OOD AUC 0.745 is under the 0.80 target. The pkl
            stays in the repo for re-activation once more data is available;
            until then it is disabled at startup to avoid shipping flags with
            weak confidence calibration. Pass ``--enable-hashboard-failure``
            on the CLI to opt-in.
        disable_isolation_forest: DEFAULT True — Isolation Forest flagger
            bootstraps itself from the live stream but has emitted zero flags
            across three real runs (realistic_b3b723d8, realistic_550baf33,
            rich_run_20260421_1650) and has no validation metrics (no ROC AUC,
            no OOD test). Its `anomaly_composite` flag would dispatch to all
            four specialists (Maestro Flow 3, ~$0.35-0.50 per decision) so
            firing an unvalidated classifier into that path is the wrong
            default. Pkl kept for re-activation after a dedicated validation
            study. Pass ``--enable-isolation-forest`` to opt-in.
        stop_when: optional callable → bool; returns when True (for tests).
    """
    if sensitivity not in VALID_SENSITIVITIES:
        raise ValueError(f"sensitivity must be one of {VALID_SENSITIVITIES}, got '{sensitivity}'")

    sp = stream_paths()
    telemetry_path = Path(input_stream) if input_stream else sp.telemetry
    kpi_path = sp.kpis
    flags_path = Path(flag_output) if flag_output else sp.flags

    log.info(
        "detector.starting",
        telemetry_path=str(telemetry_path),
        kpi_path=str(kpi_path),
        flags_path=str(flags_path),
        sensitivity=sensitivity,
        predictor_model=predictor_model_path,
        anomaly_model=anomaly_model_path,
    )

    # Build flaggers.
    rule_flagger = RuleEngineFlagger(sensitivity=sensitivity)
    xgb_flagger = XGBoostFlagger(
        model_path=predictor_model_path, sensitivity=sensitivity
    )
    flaggers: list[Any] = [rule_flagger, xgb_flagger]

    if not disable_isolation_forest:
        if_flagger = IsolationForestFlagger(
            model_path=anomaly_model_path, sensitivity=sensitivity
        )
        flaggers.append(if_flagger)
    else:
        if_flagger = None
        log.info("detector.isolation_forest_flagger.disabled_by_flag")

    if not disable_chip_instability:
        chip_flagger = ChipInstabilityFlagger(model_path=chip_instability_model_path)
        flaggers.append(chip_flagger)
    else:
        chip_flagger = None
        log.info("detector.chip_instability_flagger.disabled_by_flag")

    if not disable_hashboard_failure:
        hb_flagger = HashboardFailureFlagger(model_path=hashboard_failure_model_path)
        flaggers.append(hb_flagger)
    else:
        hb_flagger = None
        log.info("detector.hashboard_failure_flagger.disabled_by_flag")

    # Per-miner rolling history.
    histories: dict[str, MinerHistory] = {}

    # Start background KPI tailer thread.
    _start_kpi_tailer(kpi_path, histories)

    log.info(
        "detector.flaggers_ready",
        rule_engine="active",
        isolation_forest=(
            "disabled_by_flag" if disable_isolation_forest
            else ("active" if if_flagger and if_flagger.is_active() else "bootstrap")
        ),
        xgboost="active" if xgb_flagger.is_active() else "training_required",
        chip_instability=(
            "disabled_by_flag" if disable_chip_instability
            else ("active" if chip_flagger and chip_flagger.is_active() else "model_missing")
        ),
        hashboard_failure=(
            "disabled_by_flag" if disable_hashboard_failure
            else ("active" if hb_flagger and hb_flagger.is_active() else "model_missing")
        ),
    )

    # Main loop — tail telemetry.
    for envelope in tail_events(telemetry_path, stop_when=stop_when):
        if envelope.event != "telemetry_tick":
            continue

        tick: TelemetryTick = envelope.typed_data()  # type: ignore[assignment]
        mid = tick.miner_id

        # Get or create MinerHistory.
        if mid not in histories:
            histories[mid] = MinerHistory(miner_id=mid)
        history = histories[mid]
        history.push_telemetry(tick, envelope.ts)

        # Run each flagger.
        for flagger in flaggers:
            try:
                result: FlagResult | None = flagger.evaluate(tick, history)
            except Exception as exc:
                log.error(
                    "detector.flagger_error",
                    flagger=flagger.name,
                    miner_id=mid,
                    error=str(exc),
                )
                continue

            if result is None:
                continue

            _emit_flag(result, mid, flags_path)


def _emit_flag(result: FlagResult, miner_id: str, flags_path: Path) -> None:
    """Build a FlagRaised model and write it to the flags stream."""
    evidence = FlagEvidence(
        metric=result.evidence.get("metric", "unknown"),
        window_min=float(result.evidence.get("window_min", 0)),
        **{k: v for k, v in result.evidence.items() if k not in ("metric", "window_min")},
    )
    flag_model = FlagRaised(
        flag_id=_next_flag_id(),
        miner_id=miner_id,  # type: ignore[arg-type]
        flag_type=result.flag_type,  # type: ignore[arg-type]
        severity=result.severity,  # type: ignore[arg-type]
        confidence=result.confidence,
        source_tool=result.source_tool,  # type: ignore[arg-type]
        evidence=evidence,
        raw_score=result.raw_score,
    )
    write_event(
        event="flag_raised",
        source="detector",
        payload=flag_model,
        stream_path=flags_path,
    )
    log.info(
        "detector.flag_emitted",
        flag_id=flag_model.flag_id,
        miner_id=miner_id,
        flag_type=result.flag_type,
        severity=result.severity,
        source_tool=result.source_tool,
    )


# ---------------------------------------------------------------------------
# Background KPI tailer
# ---------------------------------------------------------------------------


def _kpi_tailer_thread(kpi_path: Path, histories: dict[str, MinerHistory]) -> None:
    """Background thread that tails the KPI stream and updates histories."""
    log.info("detector.kpi_tailer.starting", kpi_path=str(kpi_path))
    try:
        for envelope in tail_events(kpi_path):
            if envelope.event != "kpi_update":
                continue
            kpi: KpiUpdate = envelope.typed_data()  # type: ignore[assignment]
            mid = kpi.miner_id
            if mid in histories:
                histories[mid].push_kpi(kpi)
    except Exception as exc:
        log.warning("detector.kpi_tailer.error", error=str(exc))


def _start_kpi_tailer(kpi_path: Path, histories: dict[str, MinerHistory]) -> None:
    t = threading.Thread(
        target=_kpi_tailer_thread,
        args=(kpi_path, histories),
        daemon=True,
        name="kpi-tailer",
    )
    t.start()
