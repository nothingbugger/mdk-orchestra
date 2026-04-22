"""AVE comparison runner — 5 configurations × 10 flags.

Reads scripts/ave_test_batch/flags.json (authoritative batch with
physically-motivated ground truth), runs each flag through the five
architectural configurations, records (action, autonomy, cost, latency)
per decision, and writes reports/ave_comparison_10flags.{md,json}.

Configurations
--------------
1. multi_full_api    — Maestro Orchestra on Anthropic (Sonnet dispatch,
                       specialists mixed).
2. multi_opus_prem   — Maestro dispatch/escalation on Opus, specialists
                       on Sonnet/Haiku (profile `opus_premium`).
3. single_sonnet     — One Sonnet call per flag with monolithic prompt.
4. single_opus       — One Opus call per flag with monolithic prompt.
5. multi_full_local  — Maestro Orchestra via Ollama (Qwen 2.5 7B on Mac B).

Output
------
reports/ave_comparison_10flags.md   — human-readable report
reports/ave_comparison_10flags.json — raw per-decision data + aggregates

Failsafes
---------
- Any single decision that crashes is logged with {"error": ...} and
  excluded from aggregates for that config; the run continues.
- Config 5 (full_local) is gated by a wall-clock budget; if elapsed time
  exceeds --full-local-budget-s, remaining flags are skipped and the
  report documents a "partial completion" note.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Single-agent monolithic prompt (shared between scenarios 3 and 4)
# ---------------------------------------------------------------------------

AUTONOMY_LADDER_DOC = """\
Autonomy ladder (from maestro.md):
- L1_observe       — log the situation, do nothing.
- L2_suggest       — alert operator with recommendation, operator approves.
- L3_bounded_auto  — execute. Only for reversible actions with magnitude cap
                     and scheduled rollback (throttle down to 70% min,
                     migrate workload up to 4h).
- L4_human_only    — queue for approval. Always for: shutdown, retire,
                     firmware update, voltage change, fleet-wide batch,
                     anything affecting > 5 miners.

Action vocabulary: observe / alert_operator / throttle / migrate_workload /
schedule_maintenance / human_review / shutdown.
"""

ROLE_MONOLITHIC = """\
You are the sole AI operator for a Bitcoin mining fleet predictive
maintenance system. You have full, integrated knowledge of voltage
patterns, hashrate trajectories, environmental/HVAC context, and
grid/power-supply dynamics. Given a pre-failure flag, decide the single
action to take — you do not have specialist agents to consult.

Reason across domains yourself:
- Voltage: is the PSU delivering nominal rail? Supply-side issues
  require operator action, not compute throttling.
- Hashrate trajectory: is the degradation real (sustained variance,
  trend) or noise-shaped (rolling features healthy)?
- Environment: is ambient / HVAC a plausible exogenous driver?
- Power: rack-level effects visible?

Apply the autonomy ladder strictly. Reversibility first. If ambiguous,
escalate to L2 — don't take irreversible action on unclear evidence.
L3 is for reversible capped actions (throttle, migrate). L4 is for
anything consequential (shutdown, human review of completed faults).

Emit your decision via the submit_decision tool. Keep reasoning_trace
compact and SRE-style.
"""


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "observe", "alert_operator", "throttle",
                "migrate_workload", "schedule_maintenance",
                "human_review", "shutdown",
            ],
        },
        "action_params": {"type": "object"},
        "autonomy_level": {
            "type": "string",
            "enum": ["L1_observe", "L2_suggest", "L3_bounded_auto", "L4_human_only"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_trace": {"type": "string"},
    },
    "required": ["action", "autonomy_level", "confidence", "reasoning_trace"],
}


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class FlagDecision:
    config: str
    flag_id: str
    flag_type: str
    severity: str
    ground_truth_action: str
    ground_truth_autonomy: str
    emitted_action: str
    emitted_autonomy: str
    confidence: float
    cost_usd: float
    latency_s: float
    n_api_calls: int
    model_used: str
    consulted_agents: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    quality_match: int = 0  # 1 if (action, autonomy) == ground_truth
    error: str | None = None

    @classmethod
    def build_error(cls, config: str, flag: dict[str, Any], err: str) -> "FlagDecision":
        return cls(
            config=config,
            flag_id=flag["flag_id"],
            flag_type=flag["flag_type"],
            severity=flag["severity"],
            ground_truth_action=flag["ground_truth_action"],
            ground_truth_autonomy=flag["ground_truth_autonomy"],
            emitted_action="(error)",
            emitted_autonomy="(error)",
            confidence=0.0,
            cost_usd=0.0,
            latency_s=0.0,
            n_api_calls=0,
            model_used="(error)",
            error=err,
        )


# ---------------------------------------------------------------------------
# Helpers for flag → Maestro dispatch
# ---------------------------------------------------------------------------


def _flag_for_maestro(flag_entry: dict[str, Any]) -> dict[str, Any]:
    """Strip ground-truth annotations so the Orchestra sees only the
    evidence and decides blind."""
    return {
        "flag_id": flag_entry["flag_id"],
        "miner_id": flag_entry["miner_id"],
        "flag_type": flag_entry["flag_type"],
        "severity": flag_entry["severity"],
        "confidence": flag_entry["confidence"],
        "source_tool": flag_entry["source_tool"],
        "evidence": flag_entry["evidence"],
        "raw_score": flag_entry["raw_score"],
    }


def _reload_maestro_module():
    """Force a fresh import of agents.maestro after env var flips so the
    `_load_config` YAML cache and `_BACKEND_INSTANCES` factory are rebuilt
    against the current routing profile."""
    # Blow caches inside llm_backend.
    try:
        import agents.llm_backend as llm_backend  # noqa: WPS433
        llm_backend._reset_config_cache()
        llm_backend._BACKEND_INSTANCES.clear()
    except Exception:
        pass
    # Re-import maestro to rebuild specialists under the new profile.
    for mod_name in list(sys.modules):
        if mod_name.startswith("agents."):
            del sys.modules[mod_name]
    import agents.maestro  # noqa: F401
    return importlib.import_module("agents.maestro")


# ---------------------------------------------------------------------------
# Config runners
# ---------------------------------------------------------------------------


def _run_maestro_profile(
    profile: str,
    flag_entry: dict[str, Any],
    config_label: str,
) -> FlagDecision:
    """Generic multi-agent runner — parameterized by MDK_LLM_PROFILE."""
    os.environ["MDK_LLM_PROFILE"] = profile
    os.environ.pop("MDK_AGENT_MOCK", None)

    maestro_mod = _reload_maestro_module()
    Maestro = maestro_mod.Maestro

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MDK_STREAM_DIR"] = str(Path(tmp) / "stream")
        os.environ["MDK_MEMORY_DIR"] = str(Path(tmp) / "memory")

        t0 = time.monotonic()
        maestro = Maestro()
        decision = maestro.dispatch_flag(_flag_for_maestro(flag_entry))
        latency_s = time.monotonic() - t0

    n_api = 1 + len(decision.consulted_agents)
    gt_action = flag_entry["ground_truth_action"]
    gt_autonomy = flag_entry["ground_truth_autonomy"]
    q = 1 if (decision.action == gt_action and decision.autonomy_level == gt_autonomy) else 0

    return FlagDecision(
        config=config_label,
        flag_id=flag_entry["flag_id"],
        flag_type=flag_entry["flag_type"],
        severity=flag_entry["severity"],
        ground_truth_action=gt_action,
        ground_truth_autonomy=gt_autonomy,
        emitted_action=decision.action,
        emitted_autonomy=decision.autonomy_level,
        confidence=decision.confidence,
        cost_usd=decision.total_cost_usd,
        latency_s=latency_s,
        n_api_calls=n_api,
        model_used=f"profile={profile}",
        consulted_agents=list(decision.consulted_agents),
        reasoning_trace=decision.reasoning_trace,
        quality_match=q,
    )


def run_config_multi_full_api(flag: dict[str, Any]) -> FlagDecision:
    return _run_maestro_profile("full_api", flag, "multi_full_api")


def run_config_multi_opus_premium(flag: dict[str, Any]) -> FlagDecision:
    return _run_maestro_profile("opus_premium", flag, "multi_opus_premium")


def run_config_multi_full_local(flag: dict[str, Any]) -> FlagDecision:
    return _run_maestro_profile("full_local", flag, "multi_full_local")


def _single_agent_prompt(flag_entry: dict[str, Any]) -> str:
    ev = flag_entry["evidence"]
    feature_block = ""
    if "features" in ev and "feature_values" in ev:
        feature_block = "\nRolling feature snapshot:\n" + "\n".join(
            f"  {n} = {v}" for n, v in zip(ev["features"], ev["feature_values"])
        )
    flag_public = _flag_for_maestro(flag_entry)
    return (
        f"Flag received:\n{json.dumps(flag_public, indent=2)}\n"
        f"{feature_block}\n\n"
        f"{AUTONOMY_LADDER_DOC}\n"
        "Decide and emit via submit_decision."
    )


def _run_single_agent(model: str, config_label: str, flag: dict[str, Any]) -> FlagDecision:
    from agents._client import call_structured

    os.environ.pop("MDK_LLM_PROFILE", None)
    os.environ.pop("MDK_AGENT_MOCK", None)

    t0 = time.monotonic()
    r = call_structured(
        model=model,
        system_prompt=ROLE_MONOLITHIC,
        user_content=_single_agent_prompt(flag),
        tool_name="submit_decision",
        tool_description="Submit the single operator decision.",
        tool_schema=DECISION_SCHEMA,
        max_tokens=1024,
        mock_fallback=None,
        agent_slot=None,
    )
    latency_s = time.monotonic() - t0

    ti = r.tool_input or {}
    emitted_action = ti.get("action", "(missing)")
    emitted_autonomy = ti.get("autonomy_level", "(missing)")
    gt_action = flag["ground_truth_action"]
    gt_autonomy = flag["ground_truth_autonomy"]
    q = 1 if (emitted_action == gt_action and emitted_autonomy == gt_autonomy) else 0

    return FlagDecision(
        config=config_label,
        flag_id=flag["flag_id"],
        flag_type=flag["flag_type"],
        severity=flag["severity"],
        ground_truth_action=gt_action,
        ground_truth_autonomy=gt_autonomy,
        emitted_action=emitted_action,
        emitted_autonomy=emitted_autonomy,
        confidence=float(ti.get("confidence", 0.0)),
        cost_usd=r.cost_usd,
        latency_s=latency_s,
        n_api_calls=1,
        model_used=r.model,
        consulted_agents=[],
        reasoning_trace=ti.get("reasoning_trace", "(empty)"),
        quality_match=q,
    )


def run_config_single_sonnet(flag: dict[str, Any]) -> FlagDecision:
    return _run_single_agent("claude-sonnet-4-6", "single_sonnet", flag)


def run_config_single_opus(flag: dict[str, Any]) -> FlagDecision:
    return _run_single_agent("claude-opus-4-7", "single_opus", flag)


# ---------------------------------------------------------------------------
# AVE aggregation
# ---------------------------------------------------------------------------


EPSILON = 1e-3


def _aggregate_config(decisions: list[FlagDecision]) -> dict[str, Any]:
    ok = [d for d in decisions if d.error is None]
    n = len(ok)
    n_err = len(decisions) - n
    if n == 0:
        return {
            "n_ok": 0,
            "n_err": n_err,
            "accuracy": None,
            "sum_Q": 0,
            "mean_latency_s": None,
            "mean_cost_usd": None,
            "total_cost_usd": 0.0,
            "AVE_aggregate_eps": None,
            "AVE_aggregate_no_eps": None,
        }
    sum_q = sum(d.quality_match for d in ok)
    sum_cost = sum(d.cost_usd for d in ok)
    sum_lat = sum(d.latency_s for d in ok)
    sum_tc = sum(d.latency_s * d.cost_usd for d in ok)

    ave_eps = sum_q / (sum_tc + EPSILON)
    ave_no_eps = (sum_q / sum_tc) if sum_tc > 0 else None

    return {
        "n_ok": n,
        "n_err": n_err,
        "accuracy": sum_q / n,
        "sum_Q": sum_q,
        "mean_latency_s": sum_lat / n,
        "mean_cost_usd": sum_cost / n,
        "total_cost_usd": sum_cost,
        "sum_T_x_C": sum_tc,
        "AVE_aggregate_eps": ave_eps,
        "AVE_aggregate_no_eps": ave_no_eps,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


_CONFIG_ORDER = [
    "multi_full_api",
    "multi_opus_premium",
    "single_sonnet",
    "single_opus",
    "multi_full_local",
]

_CONFIG_LABEL = {
    "multi_full_api": "Multi full_api (Sonnet dispatch + specialists)",
    "multi_opus_premium": "Multi opus_premium (Opus dispatch + Sonnet/Haiku specialists)",
    "single_sonnet": "Single Sonnet (monolithic)",
    "single_opus": "Single Opus (monolithic)",
    "multi_full_local": "Multi full_local (Qwen 2.5 7B via Ollama)",
}


def render_markdown(
    batch: dict[str, Any],
    per_config: dict[str, list[FlagDecision]],
    aggregates: dict[str, dict[str, Any]],
    partial_note: str | None,
) -> str:
    lines: list[str] = []
    lines.append("# AVE Comparison — 5 Configurations × 10 Flags\n")
    lines.append(
        "Comparative evaluation of five architectural configurations on a "
        "fixed batch of 10 flags with physically-motivated ground-truth "
        "decisions. The metric is AVE (Agent Value-added Efficiency):\n"
    )
    lines.append(
        "```\n"
        "AVE_aggregate = sum(Q_i) / (sum(T_i · C_i) + ε)\n"
        "  Q_i = 1 if (emitted_action, autonomy_level) == ground_truth, else 0\n"
        "  T_i = latency in seconds\n"
        "  C_i = cost in USD\n"
        "  ε   = 1e-3 (guards against div-by-zero for cost-free configs)\n"
        "```\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"- Batch: `scripts/ave_test_batch/flags.json` "
        f"({len(batch['flags'])} flags, batch_id `{batch['batch_id']}`)."
    )
    lines.append(
        "- Ground truth: physically-motivated per-flag (see batch `ground_truth_reasoning` "
        "field for each flag — summarized derivation in the pitch appendix)."
    )
    lines.append(
        "- Quality metric Q is binary exact-match on (action, autonomy_level). "
        "No partial credit. The autonomy ladder is the operational decision "
        "the operator acts on, so level matters as much as action."
    )
    if partial_note:
        lines.append(f"\n> **Note:** {partial_note}\n")

    lines.append("\n## Aggregate results\n")
    lines.append(
        "| Config | Accuracy | Mean Latency | Mean Cost | Total Cost | AVE Aggregate |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )
    for key in _CONFIG_ORDER:
        agg = aggregates.get(key)
        if agg is None or agg["n_ok"] == 0:
            lines.append(f"| {_CONFIG_LABEL[key]} | — | — | — | — | — |")
            continue
        acc = agg["accuracy"]
        ml = agg["mean_latency_s"]
        mc = agg["mean_cost_usd"]
        tc = agg["total_cost_usd"]
        ave = agg["AVE_aggregate_eps"]

        if mc == 0.0:
            ave_str = f"**{ave:,.1f}** (cost-zero — ranked separately)"
            mc_str = "$0.0000"
            tc_str = "$0.0000"
        else:
            ave_str = f"{ave:,.2f}"
            mc_str = f"${mc:.4f}"
            tc_str = f"${tc:.4f}"

        err_suffix = f" ({agg['n_err']} err)" if agg["n_err"] else ""
        lines.append(
            f"| {_CONFIG_LABEL[key]} | "
            f"{acc:.0%} ({agg['sum_Q']}/{agg['n_ok']}){err_suffix} | "
            f"{ml:.1f}s | {mc_str} | {tc_str} | {ave_str} |"
        )
    lines.append("")
    lines.append(
        "_Cost-zero configurations (Qwen local) are ranked separately: the "
        "AVE formula with ε=1e-3 gives them an effectively unbounded value "
        "that does not commensurate with API-cost configurations on a "
        "cost-sensitive axis. Compare them on (accuracy, latency) instead._"
    )

    lines.append("\n## Per-flag decision matrix\n")
    lines.append(
        "Each cell shows the emitted `(action / autonomy_level)` with ✓ on exact match to ground truth, ✗ otherwise."
    )
    lines.append("")
    lines.append(
        "| Flag | Type / Severity | Ground truth | "
        + " | ".join(_CONFIG_LABEL[k].split(" (")[0] for k in _CONFIG_ORDER)
        + " |"
    )
    lines.append("|---|---|---|" + "---|" * len(_CONFIG_ORDER))
    flag_id_to_gt = {f["flag_id"]: f for f in batch["flags"]}
    # Per-flag row — need to locate each config's decision by flag_id.
    config_decision_idx: dict[str, dict[str, FlagDecision]] = {
        k: {d.flag_id: d for d in per_config.get(k, [])} for k in _CONFIG_ORDER
    }
    for f in batch["flags"]:
        gt = f"`{f['ground_truth_action']}` / `{f['ground_truth_autonomy']}`"
        type_sev = f"{f['flag_type']} / {f['severity']}"
        cells: list[str] = []
        for k in _CONFIG_ORDER:
            d = config_decision_idx[k].get(f["flag_id"])
            if d is None:
                cells.append("—")
            elif d.error:
                cells.append(f"⚠ err")
            else:
                mark = "✓" if d.quality_match else "✗"
                cells.append(f"{mark} `{d.emitted_action}` / `{d.emitted_autonomy}`")
        lines.append(f"| `{f['flag_id']}` | {type_sev} | {gt} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("\n## Wrong-action distribution\n")
    lines.append(
        "For each config, the flags it got wrong and what it emitted instead:"
    )
    lines.append("")
    for k in _CONFIG_ORDER:
        decs = per_config.get(k, [])
        wrongs = [d for d in decs if d.error is None and d.quality_match == 0]
        lines.append(f"**{_CONFIG_LABEL[k]}** — {len(wrongs)} wrong")
        for d in wrongs:
            gt = flag_id_to_gt.get(d.flag_id, {})
            lines.append(
                f"- `{d.flag_id}` ({d.flag_type}/{d.severity}): "
                f"emitted `{d.emitted_action}`/`{d.emitted_autonomy}`, "
                f"expected `{gt.get('ground_truth_action','?')}`/`{gt.get('ground_truth_autonomy','?')}`"
            )
        if not wrongs:
            lines.append("- _(all correct)_")
        lines.append("")

    lines.append("## Interpretazione\n")
    lines.append(
        "_Editorialize the numbers above. Key comparisons to surface in the "
        "pitch:_\n"
    )
    lines.append(
        "- **Multi-agent vs single-agent on the same model family.** "
        "`multi_full_api` vs `single_sonnet`: if accuracy is equivalent, "
        "the specialist layer pays for itself only in auditability "
        "(traces) — the architectural thesis weakens. If `multi_full_api` "
        "wins on accuracy the thesis is supported."
    )
    lines.append(
        "- **Does premium Opus Maestro justify the cost uplift?** "
        "`multi_opus_premium` ≈ 3-4× the cost of `multi_full_api` by "
        "construction (Opus dispatch + escalation on every flag). "
        "If accuracy moves by +0–1 decision, the uplift is not earned."
    )
    lines.append(
        "- **Local Orchestra viability.** `multi_full_local` at zero API "
        "cost is the ablation: if it gets within 1-2 flags of the API "
        "configurations it is a credible fallback for cost-free ops."
    )
    lines.append(
        "- **Wrong-action character.** Check the wrong-action table for "
        "pattern: do the single-agent configs tend to under-escalate "
        "(observing when ground truth is alert/throttle) or over-react "
        "(throttling on noise-shaped signals)? The multi-agent traces "
        "include the specialist verdicts that would have surfaced the "
        "right framing."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch",
        default=str(REPO / "scripts" / "ave_test_batch" / "flags.json"),
    )
    ap.add_argument(
        "--out-md",
        default=str(REPO / "reports" / "ave_comparison_10flags.md"),
    )
    ap.add_argument(
        "--out-json",
        default=str(REPO / "reports" / "ave_comparison_10flags.json"),
    )
    ap.add_argument(
        "--only",
        default="",
        help="CSV of config names to run (skip others).",
    )
    ap.add_argument(
        "--full-local-budget-s",
        type=float,
        default=1800.0,
        help="Wall-clock budget for the full_local config in seconds (default 30 min).",
    )
    args = ap.parse_args()

    want = set(s.strip() for s in args.only.split(",") if s.strip()) or set(_CONFIG_ORDER)

    batch_path = Path(args.batch)
    batch = json.loads(batch_path.read_text())
    flags = batch["flags"]
    print(f"[setup] loaded {len(flags)} flags from {batch_path}", flush=True)
    print(f"[setup] configs to run: {sorted(want)}", flush=True)

    if any(c in want for c in ("multi_full_api", "multi_opus_premium", "single_sonnet", "single_opus")):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY missing — cannot run API configs", file=sys.stderr)
            return 2

    per_config: dict[str, list[FlagDecision]] = {k: [] for k in _CONFIG_ORDER}
    partial_note: str | None = None

    runners = {
        "multi_full_api": run_config_multi_full_api,
        "multi_opus_premium": run_config_multi_opus_premium,
        "single_sonnet": run_config_single_sonnet,
        "single_opus": run_config_single_opus,
        "multi_full_local": run_config_multi_full_local,
    }

    for cfg in _CONFIG_ORDER:
        if cfg not in want:
            continue
        print(f"\n=== config: {cfg} ===", flush=True)
        cfg_start = time.monotonic()
        for i, flag in enumerate(flags, start=1):
            if cfg == "multi_full_local":
                elapsed = time.monotonic() - cfg_start
                if elapsed > args.full_local_budget_s:
                    remaining = len(flags) - (i - 1)
                    msg = (
                        f"full_local budget exhausted after {elapsed:.0f}s; "
                        f"skipped last {remaining} flags (partial completion)"
                    )
                    print(f"[{cfg}] {msg}", flush=True)
                    partial_note = msg
                    break
            print(
                f"[{cfg}] flag {i}/{len(flags)} {flag['flag_id']} "
                f"({flag['flag_type']}/{flag['severity']})…",
                flush=True,
            )
            try:
                d = runners[cfg](flag)
                per_config[cfg].append(d)
                mark = "✓" if d.quality_match else "✗"
                print(
                    f"  → {mark} action={d.emitted_action} "
                    f"autonomy={d.emitted_autonomy} "
                    f"cost=${d.cost_usd:.4f} lat={d.latency_s:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc(limit=3)
                err = f"{type(exc).__name__}: {exc}"
                print(f"  ⚠ error: {err}\n{tb}", flush=True)
                per_config[cfg].append(
                    FlagDecision.build_error(cfg, flag, err)
                )

        # Incremental JSON dump after each config so we don't lose data
        # if a later config crashes.
        _dump_partial(args.out_json, per_config, batch, partial_note)

    aggregates = {k: _aggregate_config(per_config.get(k, [])) for k in _CONFIG_ORDER}

    # Write markdown report
    md = render_markdown(batch, per_config, aggregates, partial_note)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    print(f"\n[done] wrote {args.out_md}", flush=True)

    # Write final JSON
    _dump_partial(args.out_json, per_config, batch, partial_note, aggregates=aggregates)
    print(f"[done] wrote {args.out_json}", flush=True)
    return 0


def _dump_partial(
    path: str,
    per_config: dict[str, list[FlagDecision]],
    batch: dict[str, Any],
    partial_note: str | None,
    aggregates: dict[str, dict[str, Any]] | None = None,
) -> None:
    out = {
        "batch_id": batch["batch_id"],
        "created_at": batch["created_at"],
        "partial_note": partial_note,
        "per_config": {
            k: [asdict(d) for d in per_config.get(k, [])]
            for k in _CONFIG_ORDER
        },
    }
    if aggregates is not None:
        out["aggregates"] = aggregates
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
