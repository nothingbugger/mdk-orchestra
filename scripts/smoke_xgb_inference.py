"""Smoke verification of the XGBoost inference path.

Runs the XGBoostFlagger against the generated dataset, confirms:
1. FlagRaised envelopes are emitted.
2. _build_features never touches fault_injected (assertion-based).
3. Dumps 3-5 example flag envelopes to /tmp/xgb_flag_samples.jsonl.

Usage:
    python3 scripts/smoke_xgb_inference.py [--stream PATH]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

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

OUTPUT_PATH = Path("/tmp/xgb_flag_samples.jsonl")
MAX_SAMPLES = 5
MAX_TICKS = 50000  # read first 50k events for speed


def _make_flag_envelope(flag_result, tick, ts) -> dict:
    """Build a FlagRaised envelope dict from a FlagResult."""
    import ulid

    from shared.schemas.events import Envelope, FlagEvidence, FlagRaised

    evidence = FlagEvidence(
        metric=flag_result.evidence.get("metric", "hashrate_th"),
        window_min=flag_result.evidence.get("window_min", 1),
        **{k: v for k, v in flag_result.evidence.items() if k not in ("metric", "window_min")},
    )

    flag_raised = FlagRaised(
        flag_id=str(ulid.new()),
        miner_id=tick.miner_id,
        flag_type=flag_result.flag_type,
        severity=flag_result.severity,
        confidence=flag_result.confidence,
        source_tool=flag_result.source_tool,
        evidence=evidence,
        raw_score=flag_result.raw_score,
    )

    env = Envelope.wrap(
        event="flag_raised",
        source="detector",
        payload=flag_raised,
        ts=ts,
    )
    return env.model_dump(mode="json")


def run_smoke(stream_path: Path | None = None) -> None:
    from shared.event_bus import read_events
    from deterministic_tools.base import MinerHistory
    from deterministic_tools.xgboost_flagger import XGBoostFlagger, _build_features

    model_path = Path("models/xgb_predictor.pkl")
    if not model_path.exists():
        log.error("smoke.model_not_found", path=str(model_path))
        sys.exit(1)

    flagger = XGBoostFlagger(model_path=model_path)
    if not flagger.is_active():
        log.error("smoke.flagger_not_active")
        sys.exit(1)

    log.info("smoke.flagger_loaded", model_path=str(model_path))

    if stream_path is None:
        stream_path = Path("/tmp/mdk_train_data/telemetry_iter0.jsonl")
    if not stream_path.exists():
        log.error("smoke.stream_not_found", path=str(stream_path))
        sys.exit(1)

    histories: dict[str, MinerHistory] = defaultdict(lambda: MinerHistory(miner_id=""))
    flag_envelopes: list[dict] = []
    ticks_processed = 0
    flags_raised = 0

    # Test _build_features contract: must not access fault_injected
    class FaultAccessGuard:
        """Proxy that raises if fault_injected is accessed."""
        def __init__(self, tick):
            self._tick = tick
        def __getattr__(self, name):
            if name == "fault_injected":
                raise AssertionError("FAULT LEAK: _build_features accessed fault_injected!")
            return getattr(self._tick, name)

    log.info("smoke.starting", stream=str(stream_path))

    for envelope in read_events(stream_path):
        if envelope.event != "telemetry_tick":
            continue

        tick = envelope.typed_data()
        mid = tick.miner_id

        history = histories[mid]
        history.miner_id = mid
        history.push_telemetry(tick, envelope.ts)

        # Feature contract check (first 5000 ticks)
        if ticks_processed < 5000:
            guard = FaultAccessGuard(tick)
            feat = _build_features(history, guard)
            if feat is not None:
                assert all(f == f for f in feat), "NaN in features!"  # nan != nan
                assert len(feat) == 13, f"Wrong feature count: {len(feat)}"

        # Run flagger
        flag = flagger.evaluate(tick, history)
        if flag is not None:
            flags_raised += 1
            if len(flag_envelopes) < MAX_SAMPLES:
                try:
                    env_dict = _make_flag_envelope(flag, tick, envelope.ts)
                    flag_envelopes.append(env_dict)
                    log.info(
                        "smoke.flag_captured",
                        miner_id=tick.miner_id,
                        severity=flag.severity,
                        prob=flag.raw_score,
                        confidence=flag.confidence,
                        n_captured=len(flag_envelopes),
                    )
                except Exception as exc:
                    log.warning("smoke.envelope_build_failed", error=str(exc))

        ticks_processed += 1
        if ticks_processed >= MAX_TICKS:
            log.info("smoke.max_ticks_reached", ticks=ticks_processed)
            break

    log.info(
        "smoke.done",
        ticks_processed=ticks_processed,
        flags_raised=flags_raised,
        flag_samples_captured=len(flag_envelopes),
    )

    if flags_raised == 0:
        log.warning("smoke.no_flags", message="XGBoost flagger emitted zero flags — check model or thresholds")

    # Write samples
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for env in flag_envelopes:
            f.write(json.dumps(env) + "\n")

    log.info("smoke.samples_written", path=str(OUTPUT_PATH), count=len(flag_envelopes))

    if flag_envelopes:
        print(f"\n=== First {len(flag_envelopes)} flag envelope(s) ===")
        for i, env in enumerate(flag_envelopes[:3]):
            print(f"\n--- envelope {i+1} ---")
            print(json.dumps(env, indent=2)[:800])  # truncate for readability

    # Final assertion
    assert flags_raised > 0, "Smoke test FAILED: no flags were raised!"
    assert len(flag_envelopes) > 0, "Smoke test FAILED: no flag envelopes captured!"
    print(f"\n✓ Smoke test PASSED: {flags_raised} flags raised, {ticks_processed} ticks processed")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stream", default=None)
    args = p.parse_args()
    run_smoke(Path(args.stream) if args.stream else None)
