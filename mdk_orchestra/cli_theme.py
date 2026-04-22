"""CLI color theme mirrored from `dashboard/static/css/dashboard.css`.

The dashboard is dark-themed (mactop TUI: deep slate + cream borders +
mint accent). These tokens exactly match the CSS variables defined in
the dashboard. If you change a value here, update it in the CSS too
(or vice-versa) — they need to stay in sync so the terminal wizard and
the browser dashboard feel like the same product.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme


# ---------------------------------------------------------------------------
# Raw palette (hex values from dashboard.css)
# ---------------------------------------------------------------------------

DASHBOARD_COLORS: dict[str, str] = {
    # Surfaces
    "bg_base":        "#1b2430",  # page background — deep slate
    "bg_soft":        "#222d3b",  # panel interior
    "bg_hero":        "#243242",  # hero panels
    # Borders
    "border_base":    "#d4c998",  # warm cream border (primary)
    "border_dim":     "#8a8366",  # muted divider
    # Text
    "text_primary":   "#e8e4d4",  # off-white cream
    "text_dim":       "#9da89a",  # secondary / labels
    "text_mute":      "#6b7280",  # tertiary / timestamps
    # Semantic / status
    "status_ok":      "#9dc797",  # mint — primary accent (OK / healthy)
    "status_ok_glow": "#b8dab2",  # brighter mint — hero numbers
    "status_warn":    "#e4c571",  # yellow — warn tier
    "status_alert":   "#d97706",  # amber — alert / L2
    "status_crit":    "#e26d6d",  # soft red — crit / L4
    "status_shut":    "#6b7280",  # grey — shutdown / inactive
}


# ---------------------------------------------------------------------------
# Rich theme — named styles used throughout the CLI
# ---------------------------------------------------------------------------

MDK_THEME = Theme({
    # Structural
    "brand":        f"bold {DASHBOARD_COLORS['status_ok']}",
    "title":        f"bold {DASHBOARD_COLORS['text_primary']}",
    "subtitle":     f"italic {DASHBOARD_COLORS['text_dim']}",
    "dim":          f"{DASHBOARD_COLORS['text_mute']}",
    "muted":        f"{DASHBOARD_COLORS['text_dim']}",
    "accent":       f"bold {DASHBOARD_COLORS['status_ok']}",
    "glow":         f"bold {DASHBOARD_COLORS['status_ok_glow']}",
    # Borders (for Panel)
    "border":       DASHBOARD_COLORS['border_base'],
    "border.dim":   DASHBOARD_COLORS['border_dim'],
    "border.brand": DASHBOARD_COLORS['status_ok'],
    "border.danger":DASHBOARD_COLORS['status_crit'],
    # Menu rows
    "menu.num":     f"bold {DASHBOARD_COLORS['status_ok']}",
    "menu.name":    f"bold {DASHBOARD_COLORS['text_primary']}",
    "menu.desc":    f"{DASHBOARD_COLORS['text_dim']}",
    # Prompts
    "prompt":       f"bold {DASHBOARD_COLORS['status_ok']}",
    "input":        f"{DASHBOARD_COLORS['text_primary']}",
    # Semantic
    "success":      f"bold {DASHBOARD_COLORS['status_ok']}",
    "warning":      f"bold {DASHBOARD_COLORS['status_warn']}",
    "alert":        f"bold {DASHBOARD_COLORS['status_alert']}",
    "danger":       f"bold {DASHBOARD_COLORS['status_crit']}",
    "info":         f"{DASHBOARD_COLORS['text_dim']}",
    # Autonomy-level badges (for progress panels)
    "l1":           f"{DASHBOARD_COLORS['text_dim']}",
    "l2":           f"bold {DASHBOARD_COLORS['status_warn']}",
    "l3":           f"bold {DASHBOARD_COLORS['status_alert']}",
    "l4":           f"bold {DASHBOARD_COLORS['status_crit']}",
    # Progress bars
    "bar.done":     DASHBOARD_COLORS['status_ok'],
    "bar.remain":   DASHBOARD_COLORS['border_dim'],
})


def get_console() -> Console:
    """Construct a themed Console. Safe to call multiple times; each call
    returns a fresh instance so tests can monkey-patch without bleed-through.

    Falls back to plain output when stdout is not a tty (CI / pipe) so
    log files stay readable.
    """
    import sys
    return Console(
        theme=MDK_THEME,
        # Use rich formatting whenever we're attached to a real terminal.
        # Redirected output gets plain text — log-friendly.
        force_terminal=None,
        # `highlight=False` stops Rich from auto-coloring numbers, URLs,
        # etc. We want deliberate styling via markup tags only.
        highlight=False,
    )
