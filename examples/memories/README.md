# Sample Agent Memories

These are reference patterns distilled by Claude Opus during a pilot run of
the MDK Orchestra system. They demonstrate the format and quality of
agent-curated memories and can be used to seed the system so new runs
benefit from a prior pilot's insights from turn one.

## Usage

To seed your system with these patterns, copy them into the respective
agent memory files:

```bash
cp examples/memories/sample_maestro_memory.md agents/maestro_memory.md
cp examples/memories/sample_hashrate_memory.md agents/hashrate_memory.md
```

Alternatively, start from empty memories and let the system curate its
own patterns during runs with memory curation enabled (default ON for
real-API runs).

## Pattern format

Each pattern follows this canonical structure:

```markdown
## Pattern: <snake_case_name>
- First seen: <ISO timestamp>
- Last seen: <ISO timestamp>
- Occurrences: <int>
- Signature: <concrete telemetry fingerprint>
- Learned verdict / action: <what to do>
- Confidence: <0-1>
- Reasoning: <why this action makes sense>
- Example reference: <decision_id>
```

Patterns are written in natural language by the Maestro during
curation cycles. The curator analyzes the decision log at regular
simulated-time intervals (default 30 min sim) and distills recurring
decision shapes into patterns that specialists can read on subsequent
calls.

## How patterns become useful

- **Specialists** read their domain memory file as part of the system
  prompt on every call. When a pattern applies to an incoming flag, the
  specialist cites the pattern in its reasoning and can make a faster,
  more consistent call.
- **Maestro** reads its own memory file too. Cross-domain patterns
  (e.g. "XGBoost hashrate warnings on healthy-telemetry miners are
  unanimously noise") let Maestro downweight weak signals without
  needing to re-derive the conclusion every time.

## Sample inventory

| File | Patterns | Domain |
|---|---|---|
| `sample_maestro_memory.md` | 1 | cross-domain synthesis |
| `sample_hashrate_memory.md` | 2 | hashrate specialist |

Other agents (voltage, environment, power) had no patterns worth
distilling during the pilot run — empty memory is the honest default.
