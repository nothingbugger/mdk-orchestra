"""Generate `report/figures/fig_ave_explained.png` — the AVE v8 formula
explainer diagram for the pitch deck.

Layout: formula banner on top, five symbol cards below with a short
description and unit. Pure matplotlib, no external assets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Design tokens — mactop TUI palette, module-level constants.
try:
    from shared.design import tokens as T  # type: ignore[import-not-found]
except Exception:
    T = None  # type: ignore[assignment]


def _tok(attr: str, fallback: str) -> str:
    return getattr(T, attr, fallback) if T is not None else fallback


# Palette — match the rest of the MDK figures (dark slate background).
_PALETTE = {
    "bg":          _tok("BG",         "#1b2430"),
    "banner":      _tok("BG_HERO",    "#243242"),
    "card_fill":   _tok("BG_SOFT",    "#222d3b"),
    "card_border": _tok("BORDER_DIM", "#8a8366"),
    "text":        _tok("TEXT",       "#e8e4d4"),
    "text_muted":  _tok("TEXT_DIM",   "#9da89a"),
    "accent_Q":    _tok("MINT_GLOW",  "#b8dab2"),
    "accent_V":    _tok("MINT",       "#9dc797"),
    "accent_T":    _tok("BORDER",     "#d4c998"),
    "accent_C":    _tok("YELLOW",     "#e4c571"),
    "accent_P":    _tok("RED",        "#e26d6d"),
}
_MONO = "Menlo"
_SANS = "Helvetica"


def _card(
    ax, x: float, y: float, w: float, h: float,
    symbol: str, symbol_color: str,
    title: str, body: str,
) -> None:
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=_PALETTE["card_border"],
        facecolor=_PALETTE["card_fill"],
    )
    ax.add_patch(box)

    cx = x + w / 2.0
    # Title at the top of the card
    ax.text(
        cx, y + h - 0.25,
        title,
        fontsize=10.5,
        fontweight="bold",
        color=_PALETTE["text"],
        family=_SANS,
        va="center", ha="center",
    )
    # Symbol centered in the upper third
    ax.text(
        cx, y + h - 0.82,
        symbol,
        fontsize=36,
        fontweight="bold",
        color=symbol_color,
        family=_MONO,
        va="center", ha="center",
    )
    # Body text below — left-aligned for readability
    ax.text(
        x + 0.10 * w, y + 0.08 * h,
        body,
        fontsize=8.5,
        color=_PALETTE["text_muted"],
        family=_SANS,
        va="bottom", ha="left",
        linespacing=1.35,
    )


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.0, 7.2), dpi=160)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor(_PALETTE["bg"])

    # Title strip
    ax.text(
        6.5, 6.85,
        "AVE — Agent Value-added Efficiency (pitch v8)",
        fontsize=17, fontweight="bold",
        color=_PALETTE["text"], family=_SANS,
        ha="center", va="center",
    )
    ax.text(
        6.5, 6.45,
        "how much operational value each decision realizes per dollar-second spent on it",
        fontsize=10.5,
        color=_PALETTE["text_muted"], family=_SANS,
        ha="center", va="center", style="italic",
    )

    # Formula banner
    banner = mpatches.FancyBboxPatch(
        (0.7, 4.75), 11.6, 1.35,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.4,
        edgecolor=_PALETTE["card_border"],
        facecolor=_PALETTE["banner"],
    )
    ax.add_patch(banner)
    ax.text(
        6.5, 5.43,
        r"$\mathbf{AVE} \;=\; \dfrac{Q \,\cdot\, V_{\mathrm{net}}}{T \,\cdot\, (C_{\mathrm{agent}} \,+\, P_{\mathrm{miscal}})}$",
        fontsize=28,
        color=_PALETTE["text"],
        ha="center", va="center",
    )

    # Cards: 5 symbols, one row
    card_y = 0.7
    card_h = 3.85
    # 5 cards across 13 wide with 0.2 gaps and some padding.
    card_w = 2.35
    gap = 0.15
    left = (13.0 - (5 * card_w + 4 * gap)) / 2.0

    cards = [
        (
            "Q",
            _PALETTE["accent_Q"],
            "Quality (0 or 1)",
            "1 when emitted\n(action, autonomy_level)\nexactly matches the\nphysically-motivated\nground truth; else 0.",
        ),
        (
            r"$V_\mathrm{net}$",
            _PALETTE["accent_V"],
            "Net value (USD)",
            "V_avoided − V_forgone.\nUSD realized by getting\nthe call right: downtime\nand hardware damage\naverted, net of the\nthrottle / migrate cost.",
        ),
        (
            "T",
            _PALETTE["accent_T"],
            "Latency (s)",
            "Wall-clock seconds\nfrom flag arrival to\ndecision emission.\nIncludes specialist\nparallel calls +\nMaestro synthesis.",
        ),
        (
            r"$C_\mathrm{agent}$",
            _PALETTE["accent_C"],
            "Inference cost (USD)",
            "Measured API / compute\ncost of this decision:\nmodel tokens priced\nat provider rates,\ncache reads included.\nZero for local Ollama.",
        ),
        (
            r"$P_\mathrm{miscal}$",
            _PALETTE["accent_P"],
            "Miscal. penalty (USD)",
            "Penalty when wrong.\nCalibration ladder:\n  $0 correct\n  $5 adjacent under\n  $50 adjacent over\n  $100-400 distant/\n  physics-severe",
        ),
    ]

    for i, (sym, colr, title, body) in enumerate(cards):
        x = left + i * (card_w + gap)
        _card(ax, x, card_y, card_w, card_h, sym, colr, title, body)

    # Footer note
    ax.text(
        6.5, 0.32,
        "aggregate:  AVE = Σ(Q·V_net) / Σ(T·(C_agent + P_miscal)) + ε        calibration: config/ave_calibration.yaml",
        fontsize=8.5,
        color=_PALETTE["text_muted"], family=_MONO,
        ha="center", va="center",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(REPO / "report" / "figures" / "fig_ave_explained.png")
