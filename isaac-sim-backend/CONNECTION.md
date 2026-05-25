# Isaac Sim ↔ NemoClaw (PhysiClaw)

**Pipeline:** Isaac extension → **HTTP** `POST /generate` → **WSL** `wsl_nemoclaw_api.py` → `openshell sandbox ssh-config` + **SSH** → **OpenClaw agent** (inside sandbox) → **NVIDIA inference** (via OpenShell) → text → **`asset_resolver`** → JSON → Isaac spawns prims.

**Default URL:** `http://127.0.0.1:8010` — set `NEMOCLAW_BACKEND_URL` to change.

**Full reproducible steps:** see **`SETUP_GUIDE.md`** in this directory.

**Windows-only bridge:** `start.bat` → `http://127.0.0.1:8002` (uses `wsl_helper.py` via `nemo_client.py`).
