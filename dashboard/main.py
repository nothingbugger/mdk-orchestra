"""Dashboard entry point.

    python -m dashboard.main            # default 127.0.0.1:8000
    python -m dashboard.main --port 9000
    MDK_STREAM_DIR=/tmp/mdk python -m dashboard.main
"""

from __future__ import annotations

import argparse
import os

from dashboard.app import run_dashboard


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDK Fleet live dashboard")
    parser.add_argument("--host", default=os.getenv("MDK_DASH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MDK_DASH_PORT", "8000")))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dashboard(host=args.host, port=args.port)
