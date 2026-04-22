#!/usr/bin/env bash
# Quick smoke — 2 minutes of real-API end-to-end.
#
# Requires ANTHROPIC_API_KEY set. Expected cost ~$0.10.
# Opens the dashboard; watch http://127.0.0.1:8000 for live events.

set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Error: ANTHROPIC_API_KEY not set." >&2
  echo "Run: export ANTHROPIC_API_KEY=sk-ant-..." >&2
  exit 1
fi

mdk-orchestra demo --duration 2 --profile full_api
