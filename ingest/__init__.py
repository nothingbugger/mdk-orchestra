"""MDK Fleet — ingest module.

Consumes telemetry_tick events, computes True Efficiency (TE) and
Health State Index (HSI) per miner, and emits kpi_update and fleet_snapshot
events.

Public API (as per shared/specs/interfaces.md §2):

    from ingest.runner import run_ingest
    from ingest.kpi import compute_te, compute_hsi, compute_miner_status
"""
