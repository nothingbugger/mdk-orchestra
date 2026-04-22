"""simulator — MDK Fleet real-time miner telemetry generator.

Public API (matches shared/specs/interfaces.md §1):
    from simulator.runner import run_simulator, simulate_one_tick, make_simulator_state

Internal modules:
    simulator.miner_sim       — single-miner physics model
    simulator.fleet_sim       — fleet orchestrator + fault injection scheduler
    simulator.environmental   — ambient + price feed evolution
"""
