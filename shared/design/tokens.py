"""MDK TUI · design tokens (Python).

Single source of truth for colors, fonts, spacing, sizes used across
the MDK 2.0 dashboards and figures. Paired with ``tokens.json`` so
non-Python tools (CSS, SVG, JS, shell) can read the same values.

Usage:

    from design.tokens import T
    style = {"backgroundColor": T.BG, "color": T.TEXT, "fontFamily": T.MONO}

    # plotly
    fig.update_layout(**T.plotly_layout())

    # matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update(T.matplotlib_rcparams())

Keep in sync with ``tokens.json`` — if you change a value here, update
it there too.

Style: TUI mactop (dark slate + cream borders + mint accent). Palette
chosen for high contrast on monospaced typography and low visual noise
when scanning compressed operational data.
"""
from __future__ import annotations


# ─── palette ─────────────────────────────────────────────────────────────────

BG          = "#1b2430"   # page background — deep slate, warm blue undertone
BG_SOFT     = "#222d3b"   # panel interior surface
BG_HERO     = "#243242"   # hero panels (optimizer / predicter / highlighted)
BG_DOT      = "rgba(212, 201, 152, 0.08)"  # row hover tint

BORDER      = "#d4c998"   # panel border — warm cream, mactop-style
BORDER_DIM  = "#8a8366"   # muted divider inside panels

TEXT        = "#e8e4d4"   # primary text — off-white cream
TEXT_DIM    = "#9da89a"   # secondary / labels
TEXT_MUTE   = "#6b7280"   # tertiary / timestamps / inactive

MINT        = "#9dc797"   # ok / healthy / gain — primary accent
MINT_GLOW   = "#b8dab2"   # brighter mint for hero numbers
YELLOW      = "#e4c571"   # warning tier
AMBER       = "#d97706"   # operational alert / marker
AMBER_SOFT  = "#fde8c7"   # highlighter wash (rare)
RED         = "#e26d6d"   # critical / shutdown / failure
SHUT        = TEXT_MUTE   # shutdown blocks (grey)

# Semantic aliases (so callsites read naturally)
STATUS_OK   = MINT
STATUS_WARN = YELLOW
STATUS_IMM  = RED
STATUS_SHUT = SHUT

# Plotly / area fills (low-alpha wash colors)
FILL_OK_SOFT    = "rgba(157, 199, 151, 0.10)"
CALLOUT_OK      = "rgba(157, 199, 151, 0.14)"
CALLOUT_WARN    = "rgba(228, 197, 113, 0.14)"
CALLOUT_CRIT    = "rgba(226, 109, 109, 0.18)"
CALLOUT_ALERT   = "rgba(217, 119, 6, 0.10)"

TRANSPARENT = "rgba(0,0,0,0)"
PLOT_GRID   = "rgba(212, 201, 152, 0.18)"


# ─── typography ──────────────────────────────────────────────────────────────

MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
HAND = "'Kalam', 'Caveat', cursive"  # wireframe annotations only

FS_TINY   = "10px"
FS_LABEL  = "11px"
FS_SMALL  = "11px"
FS_BODY   = "13px"
FS_HEADER = "13px"
FS_BIG    = "22px"
FS_KPI    = "28px"
FS_HERO   = "34px"

FW_NORMAL = 400
FW_MEDIUM = 500
FW_BOLD   = 700

LH_TIGHT  = 1.3
LH_NORMAL = 1.5

LS_TINY   = "0.3px"
LS_SMALL  = "0.8px"
LS_LABEL  = "1.2px"
LS_KPI    = "1.4px"
LS_STRONG = "1.8px"


# ─── spacing scale (4-pt grid) ───────────────────────────────────────────────

SP_0 = "0"
SP_1 = "4px"
SP_2 = "8px"
SP_3 = "12px"
SP_4 = "16px"
SP_5 = "24px"
SP_6 = "32px"


# ─── borders / radii ─────────────────────────────────────────────────────────

B_PANEL      = f"1px solid {BORDER}"
B_HERO       = f"1.5px solid {MINT}"
B_DIM_DASHED = f"1px dashed {BORDER_DIM}"
B_ROW_DOTTED = "1px dotted rgba(212, 201, 152, 0.22)"
RADIUS       = "0"   # TUI feel — sharp corners everywhere


# ─── component sizes ─────────────────────────────────────────────────────────

PANEL_PAD       = "16px 18px 14px 18px"
PANEL_PAD_HERO  = "22px 22px 20px"
BAR_HEIGHT      = "24px"
CELL_HEIGHT     = "24px"
CELL_GAP        = "4px"
MAX_MINER_COLS  = 10   # fleet heatmap — slots per model row


# ─── plotly defaults ─────────────────────────────────────────────────────────

PLOT_FONT   = dict(family=MONO, size=11, color=TEXT_DIM)
PLOT_MARGIN = dict(l=4, r=4, t=4, b=4)
PLOT_BG     = BG_SOFT  # legacy alias — prefer TRANSPARENT in new code


def plotly_layout(*, height: int = 240, show_axes: bool = False) -> dict:
    """Base ``fig.update_layout(**plotly_layout())`` dict.

    Transparent backgrounds make the chart inherit whatever panel bg it
    sits inside. ``show_axes=False`` hides ticks/gridlines (sparkline
    idiom); set True for real analytical charts.
    """
    axis_hidden = dict(showgrid=False, showticklabels=False,
                       zeroline=False, showline=False, tickfont=PLOT_FONT)
    axis_visible = dict(axis_hidden, showgrid=True, gridcolor=PLOT_GRID,
                        showticklabels=True)
    axis = axis_visible if show_axes else axis_hidden
    return dict(
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        margin=PLOT_MARGIN,
        height=height,
        font=PLOT_FONT,
        xaxis=axis,
        yaxis=dict(axis, gridcolor=PLOT_GRID, showgrid=show_axes),
        showlegend=False,
        transition=dict(duration=500, easing="cubic-in-out"),
    )


# ─── matplotlib defaults ─────────────────────────────────────────────────────

def matplotlib_rcparams() -> dict:
    """Drop-in for ``plt.rcParams.update(...)``.

    Applies MDK TUI palette + JetBrains Mono to every figure. Call once
    at the top of a plotting script::

        import matplotlib.pyplot as plt
        from design.tokens import matplotlib_rcparams
        plt.rcParams.update(matplotlib_rcparams())
    """
    from cycler import cycler   # imported lazily so the token module
                                 # stays free of matplotlib at import time
    return {
        "figure.facecolor":       BG,
        "axes.facecolor":         BG_SOFT,
        "savefig.facecolor":      BG,
        "text.color":             TEXT,
        "axes.labelcolor":        TEXT_DIM,
        "axes.edgecolor":         BORDER_DIM,
        "xtick.color":            TEXT_DIM,
        "ytick.color":            TEXT_DIM,
        "grid.color":             BORDER_DIM,
        "grid.linestyle":         ":",
        "grid.alpha":             0.35,
        "axes.grid":              True,
        "axes.prop_cycle":        cycler("color",
                                         [MINT, YELLOW, RED, BORDER,
                                          MINT_GLOW, AMBER]),
        "font.family":            "monospace",
        "font.monospace":         ["JetBrains Mono", "SF Mono",
                                   "Menlo", "Courier New"],
        "font.size":              11,
        "axes.titlesize":         13,
        "axes.titleweight":       "bold",
        "axes.titlelocation":     "left",
        "axes.titlepad":          12,
        "axes.labelsize":         11,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "legend.facecolor":       BG_SOFT,
        "legend.edgecolor":       BORDER_DIM,
        "legend.labelcolor":      TEXT,
        "figure.titlesize":       14,
        "figure.titleweight":     "bold",
    }


# ─── semantic helpers ────────────────────────────────────────────────────────

KIND_COLOR = {
    "fleet":     MINT,
    "state":     YELLOW,
    "optimizer": MINT,
    "predicter": RED,
    "dataset":   BORDER,
}

SEVERITY_COLOR = {
    "info": TEXT_DIM,
    "warn": YELLOW,
    "crit": RED,
}

MINER_STATUS = {
    "ok":   {"color": MINT,      "label": "safe",     "range": "TTF > 24h"},
    "warn": {"color": YELLOW,    "label": "warning",  "range": "TTF 6-24h"},
    "imm":  {"color": RED,       "label": "imminent", "range": "TTF < 6h"},
    "shut": {"color": TEXT_MUTE, "label": "shutdown", "range": "locked"},
}


# ─── convenience style builders ──────────────────────────────────────────────

def style_panel(hero: bool = False) -> dict:
    """Inline style dict for a TUI panel (Dash html.Div)."""
    return {
        "border":     B_HERO if hero else B_PANEL,
        "background": BG_HERO if hero else BG_SOFT,
        "padding":    PANEL_PAD_HERO if hero else PANEL_PAD,
        "position":   "relative",
    }


def style_text(dim: bool = False, mute: bool = False) -> dict:
    return {
        "color":      TEXT_MUTE if mute else (TEXT_DIM if dim else TEXT),
        "fontFamily": MONO,
    }


# ─── class interface (alternative import style) ──────────────────────────────

class T:
    """Namespace wrapper — ``from design.tokens import T; T.BG`` syntax."""
    BG          = BG
    BG_SOFT     = BG_SOFT
    BG_HERO     = BG_HERO
    BORDER      = BORDER
    BORDER_DIM  = BORDER_DIM
    TEXT        = TEXT
    TEXT_DIM    = TEXT_DIM
    TEXT_MUTE   = TEXT_MUTE
    MINT        = MINT
    MINT_GLOW   = MINT_GLOW
    YELLOW      = YELLOW
    AMBER       = AMBER
    RED         = RED
    SHUT        = SHUT
    STATUS_OK   = STATUS_OK
    STATUS_WARN = STATUS_WARN
    STATUS_IMM  = STATUS_IMM
    STATUS_SHUT = STATUS_SHUT
    MONO        = MONO
    HAND        = HAND
    FS_TINY     = FS_TINY
    FS_LABEL    = FS_LABEL
    FS_SMALL    = FS_SMALL
    FS_BODY     = FS_BODY
    FS_HEADER   = FS_HEADER
    FS_BIG      = FS_BIG
    FS_KPI      = FS_KPI
    FS_HERO     = FS_HERO
    FW_NORMAL   = FW_NORMAL
    FW_MEDIUM   = FW_MEDIUM
    FW_BOLD     = FW_BOLD
    SP_0        = SP_0
    SP_1        = SP_1
    SP_2        = SP_2
    SP_3        = SP_3
    SP_4        = SP_4
    SP_5        = SP_5
    SP_6        = SP_6
    B_PANEL     = B_PANEL
    B_HERO      = B_HERO
    PANEL_PAD   = PANEL_PAD
    PANEL_PAD_HERO = PANEL_PAD_HERO
    MAX_MINER_COLS = MAX_MINER_COLS
    plotly_layout = staticmethod(plotly_layout)
    matplotlib_rcparams = staticmethod(matplotlib_rcparams)
    style_panel = staticmethod(style_panel)
    style_text  = staticmethod(style_text)
