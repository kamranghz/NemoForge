#!/usr/bin/env bash
# Run the NemoClaw HTTP API inside WSL (Isaac Sim on Windows → http://127.0.0.1:8010).
#
# Prerequisites:
#   - Sandbox reachable: openshell sandbox ssh-config "$NEMOCLAW_SANDBOX" works
#   - Python: pip install -r requirements.txt (same as Windows bridge)
#   - Typical: export NEMOCLAW_SANDBOX=physiclaw
#
set -euo pipefail
cd "$(dirname "$0")"
export NEMOCLAW_SANDBOX="${NEMOCLAW_SANDBOX:-physiclaw}"
exec python3 -m uvicorn wsl_nemoclaw_api:app --host 0.0.0.0 --port "${PORT:-8010}"
