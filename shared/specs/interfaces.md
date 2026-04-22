# Module Interfaces — MDK Fleet

**This file defines the Python function signatures that modules expose to one another.**
Combined with `event_schemas.md`, this is the full contract across modules.

All modules follow the same pattern: a **thin public API** at `module/__init__.py` or `module/<module>_public.py`, with internals free to be organized however the module author prefers.

---

## 1. simulator

**Purpose**: stream realistic miner telemetry continuously, with optional fault injection.

**Public API**:

```python
# simulator/runner.py

def run_simulator(
    n_miners: int = 50,
    tick_interval_s: float = 5.0,
    duration_s: float | None = None,   # None = run forever
    fault_injection_enabled: bool = True,
    output_stream: str = "/run/mdk_fleet/stream/telemetry.jsonl",
    seed: int | None = None,
) -> None:
    """Run the simulator loop, emitting telemetry_tick events to output_stream."""

def simulate_one_tick(
    state: SimulatorState,
    tick_time: datetime,
) -> list[dict]:
    """Compute one tick for all miners, return list of telemetry_tick event dicts."""
```

**Internals not constrained**. Simulator can be physics-based, stochastic, hybrid — author's choice.

**Output**: writes JSONL events conforming to `telemetry_tick` schema.

---

## 2. ingest

**Purpose**: consume `telemetry_tick` stream, compute live KPIs (TE, HSI), emit `kpi_update` and `fleet_snapshot`.

**Public API**:

```python
# ingest/runner.py

def run_ingest(
    input_stream: str = "/run/mdk_fleet/stream/telemetry.jsonl",
    kpi_output: str = "/run/mdk_fleet/stream/kpis.jsonl",
    snapshot_output: str = "/run/mdk_fleet/stream/snapshots.jsonl",
    snapshot_interval_s: float = 1.0,
) -> None:
    """Tail input_stream, compute KPIs, emit outputs."""
```

```python
# ingest/kpi.py

def compute_te(telemetry: dict, alpha: float = 0.5,
               fleet_r_p5: float, fleet_r_p95: float) -> tuple[float, dict]:
    """
    Compute True Efficiency for one miner's telemetry.

    Returns (te_value, components_dict). See event_schemas.md:kpi_update.
    """

def compute_hsi(telemetry: dict, window_30min: list[dict]) -> tuple[float, dict]:
    """Compute Health State Index. Returns (hsi_value, components_dict)."""

def compute_miner_status(te: float, hsi: float) -> str:
    """Classify miner into 'ok' | 'warn' | 'imm' | 'shut' based on KPI thresholds."""
```

**Thresholds live in**: `ingest/thresholds.py` (reviewable config).

---

## 3. deterministic_tools

**Purpose**: consume telemetry stream, raise flags when pre-failure patterns appear.

**Public API**:

```python
# deterministic_tools/runner.py

def run_detector(
    input_stream: str = "/run/mdk_fleet/stream/telemetry.jsonl",
    flag_output: str = "/run/mdk_fleet/stream/flags.jsonl",
    predictor_model_path: str = "models/xgb_predictor.pkl",
    anomaly_model_path: str = "models/if_v2.pkl",
    sensitivity: str = "medium",   # "low" | "medium" | "high"
) -> None:
    """Tail telemetry, emit flag_raised events."""
```

```python
# deterministic_tools/flaggers.py

class Flagger(Protocol):
    name: str

    def evaluate(self, miner_telemetry: dict,
                 miner_history: "MinerHistory") -> FlagResult | None:
        """Return a FlagResult if a pre-failure pattern is detected, else None."""

@dataclass
class FlagResult:
    flag_type: str
    severity: str           # "info" | "warn" | "crit"
    confidence: float       # [0.0, 1.0]
    source_tool: str
    evidence: dict
    raw_score: float
```

**Implementations**: `XGBoostFlagger`, `IsolationForestFlagger`, `RuleEngineFlagger` — all conforming to `Flagger` protocol.

**Sensitivity knob**: adjusts confidence threshold for emission. Low = emit only high-confidence flags (fewer false positives, fewer catches). High = emit everything above noise (more catches, more false positives, more LLM cost downstream). Documented and exposed in A/B experiment.

---

## 4. agents

**Purpose**: the LLM agent fleet. Orchestrator + 4 specialist sub-agents.

**Public API**:

```python
# agents/maestro.py

def run_orchestrator(
    flag_stream: str = "/run/mdk_fleet/stream/flags.jsonl",
    decision_output: str = "/run/mdk_fleet/stream/decisions.jsonl",
    action_output: str = "/run/mdk_fleet/stream/actions.jsonl",
    maestro_md_path: str = "agents/maestro.md",
    agent_configs: dict[str, "AgentConfig"] = ...,   # defaults per agent below
) -> None:
    """Main orchestrator loop. On every flag, dispatch to relevant sub-agents,
    aggregate responses, emit orchestrator_decision."""
```

```python
# agents/base_specialist.py

class BaseSpecialist:
    def __init__(
        self,
        name: str,
        personality_md_path: str,
        model: str,                       # e.g., "claude-sonnet-4-6"
        memory_dir: str,                  # e.g., "/run/mdk_fleet/memory/voltage_agent/"
        max_history_events: int = 10,     # top-K retrieval budget
    ): ...

    def handle_request(
        self, request: dict,    # reasoning_request event
    ) -> dict:                   # reasoning_response event
        """Load personality + retrieve relevant past events + call LLM + return response."""

    def write_episodic(self, event_data: dict) -> None:
        """Persist one episode to this agent's memory dir."""
```

**Each specialist is a subclass** with its own retrieval logic:
- `VoltageAgent(BaseSpecialist)` — retrieval: past voltage-related events on same miner
- `HashrateAgent(BaseSpecialist)` — retrieval: past hashrate-trajectory events
- `EnvironmentAgent(BaseSpecialist)` — retrieval: recent site-wide env events
- `PowerAgent(BaseSpecialist)` — retrieval: past power-supply events across fleet

**Configuration**:

```python
# agents/config.py

@dataclass
class AgentConfig:
    name: str
    personality_md_path: str
    model: str
    memory_dir: str
    enabled: bool = True

DEFAULT_AGENT_CONFIGS = {
    "orchestrator": AgentConfig("orchestrator", "agents/maestro.md",
                                "claude-opus-4-7", "/run/mdk_fleet/memory/orchestrator/"),
    "voltage_agent": AgentConfig("voltage_agent", "agents/voltage_agent.md",
                                  "claude-sonnet-4-6", "/run/mdk_fleet/memory/voltage_agent/"),
    "hashrate_agent": AgentConfig("hashrate_agent", "agents/hashrate_agent.md",
                                   "claude-sonnet-4-6", "/run/mdk_fleet/memory/hashrate_agent/"),
    "environment_agent": AgentConfig("environment_agent", "agents/environment_agent.md",
                                      "claude-haiku-4-5-20251001", "/run/mdk_fleet/memory/environment_agent/"),
    "power_agent": AgentConfig("power_agent", "agents/power_agent.md",
                                "claude-sonnet-4-6", "/run/mdk_fleet/memory/power_agent/"),
}
```

---

## 5. dashboard

**Purpose**: live web UI showing fleet state, KPIs, flags, and agent reasoning trace.

**Public API**:

```python
# dashboard/app.py

def create_app(
    stream_dir: str = "/run/mdk_fleet/stream/",
    design_tokens_path: str = "shared/design/tokens.json",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Flask:
    """Build and return the Flask app. Serve at http://host:port/."""

def run_dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the dashboard server (convenience entry point)."""
```

**URL routes** (human-facing):
- `/` — main dashboard (fleet map + KPIs + live feed)
- `/miner/<id>` — detail view of one miner
- `/decisions` — reasoning trace log (searchable)
- `/api/stream/<channel>` — SSE endpoint for live updates (channels: `telemetry`, `kpi`, `flag`, `decision`)

**Design**: must use tokens from `shared/design/tokens.json` — no hardcoded colors/fonts/spacing.

---

## 6. ab_experiment

**Purpose**: run two configurations side-by-side and compare.

**Public API**:

```python
# ab_experiment/runner.py

def run_ab_experiment(
    scenario: str,                        # scenario config name
    duration_min: int = 60,
    output_dir: str = "/run/mdk_fleet/ab_runs/",
    seed: int = 42,
) -> ABResults:
    """
    Run two simulations with identical seed:
      - Run A: simulator + ingest + deterministic_tools + agents (full system)
      - Run B: simulator + ingest + deterministic_tools (no agent layer)

    Return comparative metrics.
    """

@dataclass
class ABResults:
    run_a_flags_raised: int
    run_b_flags_raised: int         # should equal run_a_flags_raised (same flags)
    run_a_actions_taken: int
    run_b_actions_taken: int        # = 0 (no agent = no auto actions)
    run_a_faults_caught_pre: int
    run_b_faults_caught_pre: int
    total_cost_usd: float
    cost_per_flag_usd: float
    per_agent_breakdown: dict
    report_path: str
```

**Deliverable**: `ab_experiment/output/results_<scenario>.md` + figures.

---

## Cross-cutting: design tokens

Every module with any visual output (figures, HTML, dashboards) imports from:

```python
# Python
from mdk_fleet.shared.design.tokens import T
# or
import mdk_fleet.shared.design.tokens as tokens

# Matplotlib figures
import matplotlib.pyplot as plt
plt.rcParams.update(tokens.matplotlib_rcparams())

# Plotly
fig.update_layout(**T.plotly_layout())
```

HTML / JS:

```html
<!-- import tokens via fetch or build step -->
<script>
  fetch('/shared/design/tokens.json').then(r => r.json()).then(t => { /* apply */ });
</script>
```

## Changes to this file

Same rule as event_schemas.md: **don't change unilaterally**. Propose → review → merge.
