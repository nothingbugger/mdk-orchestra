"""Agent Value-added Efficiency (AVE) — pitch v8 formula.

    AVE = (Q · V_net) / (T · (C_agent + P_miscal))

where:
  Q         : 1 if emitted (action, autonomy_level) == ground_truth, else 0
  V_net     : V_avoided − V_forgone, in USD, realized when Q=1
  T         : decision latency in seconds
  C_agent   : measured inference cost in USD
  P_miscal  : miscalibration penalty in USD, by error class

The aggregate form over a batch is:

    AVE_aggregate = sum(Q_i · V_net_i) / (sum(T_i · (C_agent_i + P_miscal_i)) + ε)

Calibration (V_net defaults by severity, P_miscal by error class) lives in
`config/ave_calibration.yaml` — edit that file, not this one. Per-flag
overrides: the ground-truth batch (e.g. `scripts/ave_test_batch/flags.json`)
may specify `miscalibration_cost_on_error` per flag to override the
calibrated P_miscal for that specific flag; and `value_if_correct` to
override V_net.

Usage:
    python -m report.compute_ave <run.json>
    python -m report.compute_ave <run.json> --batch scripts/ave_test_batch/flags.json
    python -m report.compute_ave <run.json> --calibration config/ave_calibration.yaml

Input shape: the script consumes a list of per-decision records with
{flag_id, emitted_action, emitted_autonomy, cost_usd, latency_s,
 ground_truth_action, ground_truth_autonomy, severity, quality_match}
— the format produced by `scripts/ave_comparison_runner.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Project root is two levels up from this file (report/compute_ave.py).
_REPO = Path(__file__).resolve().parent.parent

_DEFAULT_CALIBRATION: dict[str, Any] = {
    "schema_version": 1,
    "value_by_severity": {"info": 10, "warn": 50, "crit": 500},
    "miscalibration_penalty": {
        "correct": 0,
        "action_only": 5,
        "adjacent_under": 5,
        "adjacent_over": 50,
        "distant_under": 200,
        "distant_over": 100,
    },
    "epsilon": 0.001,
}


# Autonomy ladder ordering. Higher index = more consequential escalation.
_LEVEL_ORDER = {
    "L1_observe": 0,
    "L2_suggest": 1,
    "L3_bounded_auto": 2,
    "L4_human_only": 3,
}


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    """Load calibration YAML. Falls back to built-in defaults when the file
    is missing or the yaml module is unavailable."""
    if path is None:
        path = _REPO / "config" / "ave_calibration.yaml"
    if not path.exists() or yaml is None:
        return dict(_DEFAULT_CALIBRATION)
    raw = yaml.safe_load(path.read_text()) or {}
    # Shallow merge: anything missing falls back to defaults.
    merged = dict(_DEFAULT_CALIBRATION)
    merged.update(raw)
    # Nested dicts also need merging.
    for key in ("value_by_severity", "miscalibration_penalty"):
        base = dict(_DEFAULT_CALIBRATION[key])
        base.update(raw.get(key) or {})
        merged[key] = base
    return merged


# ---------------------------------------------------------------------------
# Scoring a single decision
# ---------------------------------------------------------------------------


@dataclass
class AVEScore:
    flag_id: str
    emitted_action: str
    emitted_autonomy: str
    ground_truth_action: str
    ground_truth_autonomy: str
    Q: int
    V_net: float
    T_s: float
    C_agent: float
    P_miscal: float
    error_class: str
    ave: float  # Q·V_net / (T·(C_agent+P_miscal)+ε)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "emitted_action": self.emitted_action,
            "emitted_autonomy": self.emitted_autonomy,
            "ground_truth_action": self.ground_truth_action,
            "ground_truth_autonomy": self.ground_truth_autonomy,
            "Q": self.Q,
            "V_net_usd": round(self.V_net, 4),
            "T_s": round(self.T_s, 4),
            "C_agent_usd": round(self.C_agent, 6),
            "P_miscal_usd": round(self.P_miscal, 4),
            "error_class": self.error_class,
            "ave": round(self.ave, 6),
        }


def classify_error(
    emitted_action: str,
    emitted_autonomy: str,
    ground_truth_action: str,
    ground_truth_autonomy: str,
) -> str:
    """Classify a decision's error relative to ground truth.

    Returns one of: correct | action_only | adjacent_under | adjacent_over |
    distant_under | distant_over | unknown.
    """
    action_match = emitted_action == ground_truth_action
    level_match = emitted_autonomy == ground_truth_autonomy
    if action_match and level_match:
        return "correct"
    if level_match:
        return "action_only"

    e = _LEVEL_ORDER.get(emitted_autonomy)
    g = _LEVEL_ORDER.get(ground_truth_autonomy)
    if e is None or g is None:
        return "unknown"
    diff = e - g  # positive = emitted higher (over-escalation)
    if diff == 1:
        return "adjacent_over"
    if diff == -1:
        return "adjacent_under"
    if diff >= 2:
        return "distant_over"
    if diff <= -2:
        return "distant_under"
    # diff == 0 handled above by level_match; shouldn't reach here.
    return "unknown"


def score_decision(
    record: dict[str, Any],
    calibration: dict[str, Any],
) -> AVEScore:
    """Score one decision against ground truth using the v8 formula."""
    gt_action = record.get("ground_truth_action", "")
    gt_autonomy = record.get("ground_truth_autonomy", "")
    em_action = record.get("emitted_action", "")
    em_autonomy = record.get("emitted_autonomy", "")

    error_class = classify_error(em_action, em_autonomy, gt_action, gt_autonomy)
    q = 1 if error_class == "correct" else 0

    # V_net: per-flag override first, else default by severity.
    v_override = record.get("value_if_correct")
    if v_override is None:
        v_by_sev = calibration.get("value_by_severity", {})
        severity = record.get("severity", "warn")
        v_net = float(v_by_sev.get(severity, 0))
    else:
        v_net = float(v_override)
    # V_net only realized when correct.
    if q == 0:
        v_net_realized = 0.0
    else:
        v_net_realized = v_net

    # P_miscal: per-flag override first, else default by error class.
    p_override = record.get("miscalibration_cost_on_error")
    if error_class == "correct":
        p_miscal = 0.0
    elif p_override is not None:
        p_miscal = float(p_override)
    else:
        penalties = calibration.get("miscalibration_penalty", {})
        p_miscal = float(penalties.get(error_class, penalties.get("distant_over", 100)))

    t_s = float(record.get("latency_s", 0.0))
    c_agent = float(record.get("cost_usd", 0.0))
    epsilon = float(calibration.get("epsilon", 0.001))

    denom = t_s * (c_agent + p_miscal) + epsilon
    ave = (q * v_net_realized) / denom if denom > 0 else 0.0

    return AVEScore(
        flag_id=record.get("flag_id", ""),
        emitted_action=em_action,
        emitted_autonomy=em_autonomy,
        ground_truth_action=gt_action,
        ground_truth_autonomy=gt_autonomy,
        Q=q,
        V_net=v_net_realized,
        T_s=t_s,
        C_agent=c_agent,
        P_miscal=p_miscal,
        error_class=error_class,
        ave=ave,
    )


# ---------------------------------------------------------------------------
# Aggregate across decisions
# ---------------------------------------------------------------------------


def aggregate_scores(
    scores: list[AVEScore],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate AVE over a batch of decisions (one config's 10 flags, say)."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "ave_aggregate": None}
    sum_qv = sum(s.Q * s.V_net for s in scores)
    sum_t_c_p = sum(s.T_s * (s.C_agent + s.P_miscal) for s in scores)
    epsilon = float(calibration.get("epsilon", 0.001))
    ave_agg = sum_qv / (sum_t_c_p + epsilon)

    from collections import Counter
    error_dist = Counter(s.error_class for s in scores)

    return {
        "n": n,
        "sum_Q": sum(s.Q for s in scores),
        "accuracy": sum(s.Q for s in scores) / n,
        "sum_V_net_usd": round(sum_qv, 4),
        "sum_T_x_CplusP": round(sum_t_c_p, 6),
        "sum_C_agent_usd": round(sum(s.C_agent for s in scores), 6),
        "sum_P_miscal_usd": round(sum(s.P_miscal for s in scores), 4),
        "mean_latency_s": round(sum(s.T_s for s in scores) / n, 3),
        "ave_aggregate": round(ave_agg, 6),
        "error_distribution": dict(error_dist),
    }


# ---------------------------------------------------------------------------
# Batch loaders — accept runner JSON or raw list
# ---------------------------------------------------------------------------


def _extract_records(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept either a runner JSON (with per_config map) or a flat list of
    decision dicts. Returns {config_label: [decision_dict, ...]}."""
    if isinstance(payload, dict) and "per_config" in payload:
        return payload["per_config"]  # type: ignore[return-value]
    if isinstance(payload, list):
        return {"_single": payload}  # noqa: WPS509
    raise ValueError("unrecognized payload shape — expected runner JSON or list")


def _merge_ground_truth(
    decisions: list[dict[str, Any]],
    batch_by_flag: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich decision records with ground-truth and miscalibration override
    fields from the batch definition."""
    out: list[dict[str, Any]] = []
    for d in decisions:
        merged = dict(d)
        gt = batch_by_flag.get(d.get("flag_id", ""), {})
        # Runner already embeds ground_truth_* but the batch carries the
        # optional overrides.
        for key in ("miscalibration_cost_on_error", "value_if_correct", "severity"):
            if key in gt and key not in merged:
                merged[key] = gt[key]
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="AVE v8 scorer")
    ap.add_argument("run_json", help="runner output (e.g. reports/ave_comparison_10flags.json)")
    ap.add_argument(
        "--batch",
        default=str(_REPO / "scripts" / "ave_test_batch" / "flags.json"),
        help="ground-truth batch file (for per-flag miscal overrides)",
    )
    ap.add_argument(
        "--calibration",
        default=str(_REPO / "config" / "ave_calibration.yaml"),
        help="calibration YAML",
    )
    ap.add_argument("--out", default=None, help="write JSON to this path (else stdout)")
    args = ap.parse_args()

    run_payload = json.loads(Path(args.run_json).read_text())
    batch_payload = json.loads(Path(args.batch).read_text()) if Path(args.batch).exists() else {"flags": []}
    batch_by_flag = {f["flag_id"]: f for f in batch_payload.get("flags", [])}

    calibration = load_calibration(Path(args.calibration))

    per_config = _extract_records(run_payload)
    out: dict[str, Any] = {
        "formula": "AVE = (Q · V_net) / (T · (C_agent + P_miscal))",
        "calibration": calibration,
        "per_config": {},
    }
    for cfg_label, decisions in per_config.items():
        enriched = _merge_ground_truth(decisions, batch_by_flag)
        scores = [
            score_decision(d, calibration)
            for d in enriched
            if d.get("error") is None
        ]
        out["per_config"][cfg_label] = {
            "aggregate": aggregate_scores(scores, calibration),
            "per_decision": [s.to_dict() for s in scores],
        }

    text = json.dumps(out, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
