#!/usr/bin/env bash
# Three-profile smoke — 2 minutes each of full_api / hybrid_economic / full_local.
#
# Useful for a side-by-side comparison of cost, latency, and action
# quality across profiles. Requires BOTH ANTHROPIC_API_KEY and a local
# LLM server.
#
# Expected total cost ~$0.15 (only full_api + hybrid charge anthropic).

set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Error: ANTHROPIC_API_KEY not set for full_api / hybrid_economic profiles." >&2
  exit 1
fi

HOST="${MDK_LLM_STANDARD_LOCAL_HOST:-http://127.0.0.1:11434}"
if ! curl -s --max-time 2 "${HOST}/api/tags" >/dev/null 2>&1; then
  echo "Error: no local LLM server at ${HOST} for hybrid_economic / full_local." >&2
  exit 1
fi

for profile in full_api hybrid_economic full_local; do
  echo ""
  echo "=========================================="
  echo "  profile: $profile"
  echo "=========================================="
  mdk-orchestra demo --duration 2 --profile "$profile" --dashboard-port 8000
  echo ""
done

echo "Done. Check runs/demo_stream/decisions.jsonl for the combined decision log."
