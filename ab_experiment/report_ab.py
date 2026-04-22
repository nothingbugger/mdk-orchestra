"""Report generator — writes ab_results.json + markdown summary + figures.

Generates into: <run_dir>/figures/

Figures (all styled via shared/design/tokens.py):
  1. action_timeline.png   — stems per action, both tracks side-by-side
  2. catch_rate.png        — faults caught A vs B per fault type
  3. cost_vs_catch.png     — cost vs catch-rate scatter
  4. reasoning_trace.png   — card-style rendering of one reasoning trace

Also writes:
  ab_results.json          — structured metrics dict (machine-readable)
  summary.md               — short markdown narrative
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_results_json(summary: Any, run_dir: Path) -> Path:
    """Serialize ABMetricSummary to ab_results.json.

    Args:
        summary: ABMetricSummary instance.
        run_dir: root of this A/B run.

    Returns:
        Path to the written file.
    """
    a = summary.track_a
    b = summary.track_b

    results: dict[str, Any] = {
        "run_id": summary.run_id,
        "scenario": summary.scenario,
        "seed": summary.seed,
        "duration_min": summary.duration_min,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "track_a": {
            "flags_raised": a.flags_raised,
            "faults_injected": a.faults_injected,
            "faults_caught": a.faults_caught,
            "catch_rate": round(a.faults_caught / max(a.faults_injected, 1), 4),
            "total_actions": a.total_actions,
            "action_counts": a.action_counts,
            "false_positives": a.false_positives,
            "false_positive_rate": round(a.false_positive_rate, 4),
            "total_cost_usd": round(a.total_cost_usd, 6),
            "median_latency_s": round(a.median_latency_s, 3),
            "reasoning_snapshots": a.reasoning_snapshots,
        },
        "track_b": {
            "flags_raised": b.flags_raised,
            "faults_injected": b.faults_injected,
            "faults_caught": b.faults_caught,
            "catch_rate": round(b.faults_caught / max(b.faults_injected, 1), 4),
            "total_actions": b.total_actions,
            "action_counts": b.action_counts,
            "false_positives": b.false_positives,
            "false_positive_rate": round(b.false_positive_rate, 4),
            "total_cost_usd": 0.0,
            "median_latency_s": round(b.median_latency_s, 3),
        },
        "comparison": {
            "catch_rate_delta": round(
                (a.faults_caught - b.faults_caught) / max(a.faults_injected, 1), 4
            ),
            "false_positive_delta": round(a.false_positive_rate - b.false_positive_rate, 4),
            "cost_per_extra_catch_usd": _cost_per_extra_catch(a, b),
            "latency_ratio_a_vs_b": round(
                a.median_latency_s / max(b.median_latency_s, 0.001), 3
            ),
        },
    }

    out_path = run_dir / "ab_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _LOG.info("report.json_written", path=str(out_path))
    return out_path


def _cost_per_extra_catch(a: Any, b: Any) -> float | str:
    extra = a.faults_caught - b.faults_caught
    if extra <= 0:
        return "N/A (no extra catches)"
    return round(a.total_cost_usd / extra, 4)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def write_markdown_summary(summary: Any, run_dir: Path) -> Path:
    """Write a concise markdown narrative comparing both tracks."""
    a = summary.track_a
    b = summary.track_b

    catch_rate_a = a.faults_caught / max(a.faults_injected, 1) * 100
    catch_rate_b = b.faults_caught / max(b.faults_injected, 1) * 100
    extra_catches = a.faults_caught - b.faults_caught
    cost_str = f"${a.total_cost_usd:.4f}" if a.total_cost_usd > 0 else "mock-mode ($0)"

    # Top actions for each track
    def top_actions(track: Any) -> str:
        if not track.action_counts:
            return "(none)"
        return ", ".join(
            f"{v} {k}" for k, v in sorted(track.action_counts.items(), key=lambda x: -x[1])
        )

    snapshot_section = ""
    if a.reasoning_snapshots:
        snapshot_section = "\n## Reasoning Trace Snapshots (Track A)\n\n"
        for i, snap in enumerate(a.reasoning_snapshots[:3], 1):
            trace = textwrap.shorten(snap.get("reasoning_trace", ""), width=300, placeholder="…")
            snapshot_section += (
                f"**Snapshot {i}** — miner `{snap['miner_id']}` → `{snap['action']}`"
                f" (conf={snap['confidence']:.2f}, cost=${snap['cost_usd']:.4f})\n\n"
                f"> {trace}\n\n"
            )

    md = f"""# A/B Experiment Results — {summary.scenario}

**Run ID**: `{summary.run_id}`
**Seed**: `{summary.seed}`
**Duration**: {summary.duration_min} min
**Generated**: {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

## Summary

| Metric | Track A (Fleet ON) | Track B (Det. only) |
|---|---|---|
| Flags raised | {a.flags_raised} | {b.flags_raised} |
| Faults injected | {a.faults_injected} | {b.faults_injected} |
| Faults caught | {a.faults_caught} | {b.faults_caught} |
| Catch rate | {catch_rate_a:.1f}% | {catch_rate_b:.1f}% |
| Total actions | {a.total_actions} | {b.total_actions} |
| False positives | {a.false_positives} | {b.false_positives} |
| False positive rate | {a.false_positive_rate:.1%} | {b.false_positive_rate:.1%} |
| Median latency | {a.median_latency_s:.2f}s | {b.median_latency_s:.2f}s |
| Total API cost | {cost_str} | $0.00 |

## Action Profiles

- **Track A**: {top_actions(a)}
- **Track B**: {top_actions(b)}

## Key Findings

- Track A caught **{a.faults_caught}** faults vs **{b.faults_caught}** for Track B
  ({'+' if extra_catches >= 0 else ''}{extra_catches} extra catches).
- Track B responded in **{b.median_latency_s:.2f}s** median; Track A in **{a.median_latency_s:.2f}s**
  (longer due to LLM consultation, but richer reasoning).
- False positive rate: A={a.false_positive_rate:.1%}, B={b.false_positive_rate:.1%}.
- Total agent layer cost: {cost_str}.
{snapshot_section}
---

*Generated by `ab_experiment/report_ab.py`*
"""

    out_path = run_dir / "summary.md"
    out_path.write_text(md, encoding="utf-8")
    _LOG.info("report.markdown_written", path=str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _import_matplotlib() -> tuple[Any, Any]:
    """Lazy import matplotlib and apply design tokens."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from shared.design.tokens import matplotlib_rcparams
    plt.rcParams.update(matplotlib_rcparams())
    return matplotlib, plt


def plot_action_timeline(summary: Any, figures_dir: Path) -> Path:
    """Figure 1: action timeline — two rows, stems per action colored by type."""
    matplotlib, plt = _import_matplotlib()
    from shared.design import tokens as T

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Action Timeline — Fleet ON vs Deterministic Only", x=0.02, ha="left")

    action_colors = {
        "observe": T.TEXT_MUTE,
        "alert_operator": T.YELLOW,
        "throttle": T.RED,
        "shutdown": T.RED,
        "migrate_workload": T.AMBER,
        "schedule_maintenance": T.MINT,
        "human_review": T.BORDER,
    }

    for ax, track, track_label in [
        (axes[0], summary.track_a, "Track A (Fleet ON)"),
        (axes[1], summary.track_b, "Track B (Det. only)"),
    ]:
        ax.set_title(track_label, fontsize=11)
        ax.set_ylabel("action", fontsize=9)

        if not track.action_counts:
            ax.text(0.5, 0.5, "no actions", ha="center", va="center", transform=ax.transAxes)
            continue

        # Use action index as x-axis since we don't have timestamps in TrackMetrics
        # (full timeline would need raw event scan; this is the profile view)
        actions = list(track.action_counts.keys())
        counts = [track.action_counts[a] for a in actions]
        colors = [action_colors.get(a, T.BORDER) for a in actions]

        bars = ax.bar(actions, counts, color=colors, edgecolor=T.BORDER_DIM, linewidth=0.5)
        ax.set_ylabel("count", fontsize=9)

        # Label bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
                color=T.TEXT_DIM,
            )

    plt.tight_layout()
    out = figures_dir / "action_timeline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("figure.saved", name="action_timeline", path=str(out))
    return out


def plot_catch_rate(summary: Any, figures_dir: Path) -> Path:
    """Figure 2: catch-rate bar chart — faults caught A vs B."""
    matplotlib, plt = _import_matplotlib()
    from shared.design import tokens as T
    import numpy as np

    a = summary.track_a
    b = summary.track_b

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Fault Catch Rate — Track A vs Track B", x=0.02, ha="left")

    categories = ["Faults Injected", "Faults Caught", "False Positives"]
    a_vals = [a.faults_injected, a.faults_caught, a.false_positives]
    b_vals = [b.faults_injected, b.faults_caught, b.false_positives]

    x = np.arange(len(categories))
    width = 0.35

    bars_a = ax.bar(x - width / 2, a_vals, width, label="Track A (Fleet)", color=T.MINT,
                    edgecolor=T.BORDER_DIM, linewidth=0.5)
    bars_b = ax.bar(x + width / 2, b_vals, width, label="Track B (Det. only)", color=T.YELLOW,
                    edgecolor=T.BORDER_DIM, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("count")
    ax.legend(loc="upper right")

    # Label bars
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=9,
                color=T.TEXT,
            )

    # Catch rate annotations
    cr_a = a.faults_caught / max(a.faults_injected, 1) * 100
    cr_b = b.faults_caught / max(b.faults_injected, 1) * 100
    ax.text(
        0.98, 0.95,
        f"Catch rate A: {cr_a:.1f}%\nCatch rate B: {cr_b:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=T.TEXT_DIM,
        bbox=dict(boxstyle="square,pad=0.4", facecolor=T.BG_HERO, edgecolor=T.BORDER_DIM),
    )

    plt.tight_layout()
    out = figures_dir / "catch_rate.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("figure.saved", name="catch_rate", path=str(out))
    return out


def plot_cost_vs_catch(summary: Any, figures_dir: Path) -> Path:
    """Figure 3: cost vs catch-rate scatter."""
    matplotlib, plt = _import_matplotlib()
    from shared.design import tokens as T

    a = summary.track_a
    b = summary.track_b

    cr_a = a.faults_caught / max(a.faults_injected, 1) * 100
    cr_b = b.faults_caught / max(b.faults_injected, 1) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Cost vs Catch Rate", x=0.02, ha="left")

    ax.scatter([a.total_cost_usd], [cr_a], s=200, color=T.MINT, zorder=5, label="Track A (Fleet)")
    ax.scatter([0.0], [cr_b], s=200, color=T.YELLOW, marker="s", zorder=5,
               label="Track B (Det. only)")

    ax.annotate(
        f"A: ${a.total_cost_usd:.4f}\n{cr_a:.1f}% catch",
        (a.total_cost_usd, cr_a),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=9,
        color=T.MINT,
    )
    ax.annotate(
        f"B: $0.00\n{cr_b:.1f}% catch",
        (0.0, cr_b),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
        color=T.YELLOW,
    )

    ax.set_xlabel("API cost (USD)")
    ax.set_ylabel("catch rate (%)")
    ax.legend(loc="lower right")

    # x-axis: ensure origin is visible
    ax.set_xlim(left=-max(a.total_cost_usd * 0.1, 0.0002))
    ax.set_ylim(bottom=0, top=105)

    plt.tight_layout()
    out = figures_dir / "cost_vs_catch.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("figure.saved", name="cost_vs_catch", path=str(out))
    return out


def plot_reasoning_trace(summary: Any, figures_dir: Path) -> Path:
    """Figure 4: render one reasoning trace as a card-style figure."""
    matplotlib, plt = _import_matplotlib()
    import matplotlib.patches as mpatches
    from shared.design import tokens as T

    snapshots = summary.track_a.reasoning_snapshots
    if not snapshots:
        # Placeholder figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No reasoning traces available\n(mock mode or no flags processed)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=T.TEXT_DIM)
        out = figures_dir / "reasoning_trace.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    snap = snapshots[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Card background
    card = mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor=T.BORDER,
        facecolor=T.BG_HERO,
        zorder=1,
    )
    ax.add_patch(card)

    # Header bar
    header = mpatches.FancyBboxPatch(
        (0.02, 0.82), 0.96, 0.13,
        boxstyle="square,pad=0",
        linewidth=0,
        facecolor=T.BG_SOFT,
        zorder=2,
    )
    ax.add_patch(header)

    # Header text
    ax.text(
        0.05, 0.88,
        f"ORCHESTRATOR DECISION — miner {snap['miner_id']}",
        fontsize=11, fontweight="bold", color=T.TEXT, zorder=3,
        fontfamily="monospace",
    )

    action_color = {
        "throttle": T.RED,
        "alert_operator": T.YELLOW,
        "observe": T.TEXT_MUTE,
        "shutdown": T.RED,
    }.get(snap["action"], T.MINT)

    ax.text(
        0.95, 0.88,
        f"ACTION: {snap['action'].upper()}",
        fontsize=10, fontweight="bold", color=action_color, zorder=3,
        ha="right", fontfamily="monospace",
    )

    # Meta row
    consulted = ", ".join(snap.get("consulted_agents", [])) or "none"
    ax.text(
        0.05, 0.77,
        f"confidence={snap['confidence']:.2f}  |  agents={consulted}"
        f"  |  cost=${snap['cost_usd']:.4f}  |  latency={snap['latency_ms']:.0f}ms",
        fontsize=9, color=T.TEXT_DIM, zorder=3, fontfamily="monospace",
    )

    # Reasoning text (wrapped)
    trace = snap.get("reasoning_trace", "(no trace)")
    wrapped = textwrap.fill(trace, width=110)
    lines = wrapped.split("\n")[:8]  # cap at 8 lines

    for i, line in enumerate(lines):
        ax.text(
            0.05, 0.68 - i * 0.075,
            line,
            fontsize=8.5, color=T.TEXT, zorder=3, fontfamily="monospace",
        )

    fig.suptitle("Reasoning Trace — Track A Example", x=0.02, ha="left", fontsize=12)
    plt.tight_layout()
    out = figures_dir / "reasoning_trace.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("figure.saved", name="reasoning_trace", path=str(out))
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(summary: Any, run_dir: Path) -> dict[str, str]:
    """Generate all report artifacts for a completed A/B run.

    Args:
        summary: ABMetricSummary.
        run_dir: root directory of this A/B run.

    Returns:
        Dict mapping artifact name → file path (as strings).
    """
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    # JSON results
    artifacts["ab_results_json"] = str(write_results_json(summary, run_dir))

    # Markdown summary
    artifacts["summary_md"] = str(write_markdown_summary(summary, run_dir))

    # Figures (matplotlib may not be installed — fail gracefully per artifact)
    for name, fn in [
        ("action_timeline", plot_action_timeline),
        ("catch_rate", plot_catch_rate),
        ("cost_vs_catch", plot_cost_vs_catch),
        ("reasoning_trace", plot_reasoning_trace),
    ]:
        try:
            path = fn(summary, figures_dir)
            artifacts[name] = str(path)
        except ImportError as exc:
            _LOG.warning("figure.skipped_no_dep", name=name, exc=str(exc))
            artifacts[name] = "skipped (matplotlib not installed)"
        except Exception as exc:
            _LOG.error("figure.failed", name=name, exc=str(exc))
            artifacts[name] = f"failed: {exc}"

    _LOG.info("report.complete", artifacts=list(artifacts.keys()), run_dir=str(run_dir))
    return artifacts
