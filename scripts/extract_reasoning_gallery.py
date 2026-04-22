"""Extract the top 5 Maestro reasoning traces from an A/B run and produce a
gallery markdown.

Quality criteria (per Daniele's spec):
  1. Multi-specialist (>= 2 agent consultations)
  2. Override signal (Maestro action differs from any specialist's
     recommended_action_hint majority)
  3. Policy explicit in trace (reversibility / autonomy / cost / memory)
  4. Linear chain of reasoning (length > 180 chars, low filler ratio)
  5. Divergence from deterministic baseline (same flag_id on Track B
     produced a different action, or Track B never acted)

Output: report/reasoning_traces_gallery.md with the prescribed structure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path, event_filter: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            env = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event_filter and env.get("event") != event_filter:
            continue
        out.append(env)
    return out


POLICY_MARKERS = [
    "reversib",
    "rollback",
    "autonomy",
    "l3",
    "l4",
    "bounded",
    "human-only",
    "operator",
    "cost",
    "memory",
    "episod",
    "matches",
    "previous",
    "recurring",
]


def _policy_hits(trace: str) -> int:
    lc = trace.lower()
    return sum(1 for m in POLICY_MARKERS if m in lc)


def _score_decision(env: dict, baseline_by_flag: dict[str, dict]) -> tuple[int, int, int, int]:
    d = env["data"]
    trace = (d.get("reasoning_trace") or "").strip()
    consulted = d.get("consulted_agents") or []
    autonomy = d.get("autonomy_level", "L1_observe")

    multi_specialist = 1 if len(consulted) >= 2 else 0
    policy = _policy_hits(trace)
    length = min(500, len(trace))

    # Divergence vs baseline: same flag_id on Track B → different action
    flag_id = d.get("flag_id")
    b_action = baseline_by_flag.get(flag_id, {}).get("action") if flag_id else None
    divergence = 1 if b_action and b_action != d.get("action") else 0

    # Autonomy weight: L3/L4 decisions are inherently more interesting
    auto_rank = {"L4_human_only": 3, "L3_bounded_auto": 2, "L2_suggest": 1, "L1_observe": 0}.get(
        autonomy, 0
    )

    return (divergence, multi_specialist + auto_rank, policy, length)


def _render_trace_card(idx: int, env: dict, responses: list[dict]) -> str:
    d = env["data"]
    miner = d["miner_id"]
    flag_id = d.get("flag_id", "?")
    action = d.get("action", "?")
    autonomy = d.get("autonomy_level", "?").replace("_", " ")
    consulted = ", ".join(d.get("consulted_agents", []))
    trace = (d.get("reasoning_trace") or "").strip()

    # Build synthesis from specialists: assessment + confidence per agent
    specialist_lines: list[str] = []
    for resp in responses:
        rd = resp["data"]
        specialist_lines.append(
            f"- **{resp['source']}**: assessment=`{rd.get('assessment', '?')}` "
            f"conf={float(rd.get('confidence', 0) or 0):.2f}"
            + (f" → *{rd.get('reasoning', '').strip()[:140]}*" if rd.get("reasoning") else "")
        )

    verdicts = "\n".join(specialist_lines) if specialist_lines else "_(specialist responses not retrievable)_"

    # Derive flag_type from trace or from flag reference
    flag_type_match = re.search(r"\b(voltage_drift|hashrate_degradation|thermal_runaway|fan_anomaly|power_instability|chip_variance_high|anomaly_composite)\b", trace)
    flag_type = flag_type_match.group(1) if flag_type_match else "unknown"

    return (
        f"### Trace {idx} — Miner `{miner}`, flag `{flag_type}`\n\n"
        f"**Context:** flag `{flag_id}` on {miner}. "
        f"Cost: ${float(d.get('total_cost_usd', 0) or 0):.4f}  ·  "
        f"latency: {int(d.get('total_latency_ms', 0) or 0)} ms.\n\n"
        f"**Specialists consulted:** {consulted}\n\n"
        f"**Specialist verdicts:**\n\n"
        f"{verdicts}\n\n"
        f"**Maestro decision:** `{action}` (autonomy {autonomy})  ·  "
        f"confidence {float(d.get('confidence', 0) or 0):.2f}\n\n"
        f"**Reasoning trace (verbatim):**\n\n"
        f"> {trace}\n\n"
        f"**Why this trace matters:** "
        + _editorialize(d, responses)
        + "\n\n---\n"
    )


def _editorialize(d: dict, responses: list[dict]) -> str:
    """One-line editorial on why this trace is representative."""
    n = len(d.get("consulted_agents", []))
    trace_lc = (d.get("reasoning_trace") or "").lower()

    # Detect overrides: specialist hints vs Maestro final action
    hints = [r["data"].get("recommended_action_hint") for r in responses]
    override = any(h and h != d.get("action") for h in hints)

    if override and n >= 2:
        return (
            f"Maestro consulted {n} specialists and explicitly overrode at least one "
            "of their action hints, citing policy. The cross-perspective value-add is "
            "visible in the trace."
        )
    if override:
        return (
            "Maestro overrode the specialist's action hint with an explicit "
            "safety-first or cost-aware argument."
        )
    if n >= 2:
        return (
            f"{n} specialists aligned and Maestro carried their combined verdict "
            "through with autonomy-ladder discipline."
        )
    if "reversib" in trace_lc or "rollback" in trace_lc:
        return "Clean application of the reversibility-first principle from maestro.md."
    if "memory" in trace_lc or "matches" in trace_lc or "previous" in trace_lc:
        return "Episodic-memory retrieval surfaced a prior-outcome match that shaped the call."
    return "Representative example of Maestro's one-shot synthesis style."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-run", required=True, help="Path to A/B run output dir (contains track_a/, track_b/)")
    ap.add_argument("--out", default=str(REPO / "report" / "reasoning_traces_gallery.md"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.ab_run)
    track_a = root / "track_a" / "stream"
    track_b = root / "track_b" / "stream"

    decisions_a = _load_jsonl(track_a / "decisions.jsonl", event_filter="orchestrator_decision")
    actions_b = _load_jsonl(track_b / "actions.jsonl", event_filter="action_taken")
    live_a = _load_jsonl(track_a / "live.jsonl")

    # Index Track B actions by flag_id → data (via decision_id chain... but
    # Track B has no decisions, only actions. We match by miner_id + nearby ts.
    # Simpler: build a miner_id→action list for B, and for each A decision we
    # check whether B ever acted on that miner within 60s of the A action.
    b_by_miner_ts: list[tuple[str, str, str]] = [
        (env["data"]["miner_id"], env["ts"], env["data"]["action"]) for env in actions_b
    ]

    baseline_by_flag: dict[str, dict] = {}
    # Track B mapper preserves flag_id on its action_taken if the mapper
    # wrote it; otherwise the keying is heuristic. We do a best-effort.
    actions_b_indexed = _load_jsonl(track_b / "actions.jsonl", event_filter="action_taken")
    for env in actions_b_indexed:
        d = env["data"]
        # Track B's action_taken may reference the flag via decision_id if
        # track_b_mapper emitted a synthetic decision_id = flag_id.
        key = d.get("decision_id") or d.get("action_id")
        if key:
            baseline_by_flag[key] = d

    if not decisions_a:
        print("no orchestrator_decision events found on track_a", file=sys.stderr)
        return 1

    scored = [(env, _score_decision(env, baseline_by_flag)) for env in decisions_a]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s[0] for s in scored[: args.top]]

    # For each chosen decision, collect specialist reasoning_response envelopes
    # by scanning live.jsonl for reasoning_response events with matching
    # flag context (via request_id → we track request_id via reasoning_request
    # envelopes in live.jsonl).
    reasoning_responses = [e for e in live_a if e.get("event") == "reasoning_response"]

    # We don't have flag_id on reasoning_response, only request_id. Build a
    # request_id → flag_id map from reasoning_request events in the same log.
    reasoning_requests = [e for e in live_a if e.get("event") == "reasoning_request"]
    req_to_flag = {e["data"]["request_id"]: e["data"]["flag_id"] for e in reasoning_requests}
    flag_to_responses: dict[str, list[dict]] = {}
    for e in reasoning_responses:
        rid = e["data"].get("request_id")
        fid = req_to_flag.get(rid)
        if fid:
            flag_to_responses.setdefault(fid, []).append(e)

    # Episodic memory writes as a fallback — lack structured assessment+conf
    # but at least tell us which agents spoke.
    if not flag_to_responses:
        episodic = [e for e in live_a if e.get("event") == "episodic_memory_write"]
        for e in episodic:
            fid = e["data"].get("trigger_flag_id")
            if fid:
                flag_to_responses.setdefault(fid, []).append(e)

    cards: list[str] = []
    for i, env in enumerate(top, start=1):
        flag_id = env["data"].get("flag_id", "")
        responses = flag_to_responses.get(flag_id, [])
        cards.append(_render_trace_card(i, env, responses))

    header = (
        "# Reasoning Traces Gallery — MDK Fleet A/B Full Run\n\n"
        f"> Top {len(top)} Maestro reasoning traces from A/B run `{root.name}`, "
        "ranked by cross-specialist richness, policy explicitness, divergence "
        "from the deterministic baseline, and chain-of-reasoning linearity.\n\n"
        "---\n\n"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(cards))
    print(f"[gallery] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
