#!/usr/bin/env bash
# Local-only smoke — 3 minutes with everything routed through Ollama.
#
# Zero API cost. Requires a compatible local LLM server running on
# 127.0.0.1:11434 (or whatever `standard_local.host` in the YAML points to).
# Expect ~2-3× the latency of a full_api run on the same hardware.

set -euo pipefail

HOST="${MDK_LLM_STANDARD_LOCAL_HOST:-http://127.0.0.1:11434}"
if ! curl -s --max-time 2 "${HOST}/api/tags" >/dev/null 2>&1; then
  echo "Error: no local LLM server responding at ${HOST}/api/tags" >&2
  echo "Start Ollama (or compatible) and try again:" >&2
  echo "  ollama serve &" >&2
  echo "  ollama pull qwen2.5:7b-instruct-q4_K_M" >&2
  exit 1
fi

mdk-orchestra demo --duration 3 --profile full_local
