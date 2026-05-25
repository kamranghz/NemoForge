# PhysiClaw: Isaac Sim + NemoClaw — Setup and test (report checklist)

This is the **single supported pipeline**:

**Isaac Sim (Windows)** → `POST http://127.0.0.1:8010/generate` → **WSL API** → **OpenShell SSH** → **OpenClaw agent in sandbox** → **NVIDIA model (e.g. Nemotron)** → **JSON** → **Isaac extension** spawns USD / procedural asset.

---

## 1. Start NemoClaw in WSL (one-time + per machine)

1. Install prerequisites per [NemoClaw README](https://github.com/NVIDIA/NemoClaw) (Node 22+, Docker/OpenShell as documented).
2. Clone or use your repo at `/mnt/c/PhysiClaw/NemoClaw`.
3. From repo root:
   ```bash
   cd /mnt/c/PhysiClaw/NemoClaw
   npm install
   cd nemoclaw && npm install && npm run build && cd ..
   ```
4. Run onboarding once (creates gateway, sandbox, provider):
   ```bash
   nemoclaw onboard
   ```
   Follow prompts; note your **sandbox name** (this guide uses **`physiclaw`** as an example).

---

## 2. Ensure the sandbox is running

The API must be able to run:

`openshell sandbox ssh-config <sandbox_name>`

1. In WSL, start or confirm the sandbox (example name **`physiclaw`**):
   ```bash
   nemoclaw physiclaw connect
   ```
   If port **18789** is already forwarded, you may see a message that the forward exists — that usually means the sandbox is already reachable.
2. Optional check:
   ```bash
   openshell sandbox ssh-config physiclaw | head
   ```
   If this prints SSH config text, the bridge can reach the sandbox.

Set the same name for the Windows bridge (already default in `start.bat`):

`NEMOCLAW_SANDBOX=physiclaw`

---

## 3. Start the HTTP API (required)

**Recommended:** run the API **inside WSL** (Isaac talks to `127.0.0.1:8010`; WSL2 forwards to Linux).

```bash
cd /mnt/c/PhysiClaw/isaac-sim-backend
export NEMOCLAW_SANDBOX=physiclaw    # must match your sandbox
pip install -r requirements.txt      # once
python3 wsl_nemoclaw_api.py
```

You should see: `Uvicorn running on http://0.0.0.0:8010`.

**Verify:**

```bash
curl -s http://127.0.0.1:8010/health
```

**Alternative (Windows bridge on port 8002):** double-click or run `start.bat` from `C:\PhysiClaw\isaac-sim-backend`. Then set Isaac env `NEMOCLAW_BACKEND_URL=http://127.0.0.1:8002` before launch. The WSL API path above is simpler for demos.

---

## 4. Run Isaac Sim

1. Start **NVIDIA Isaac Sim** on Windows as you normally do.
2. Open or create a stage with a ground/grid so placed assets are visible.

---

## 5. How the extension connects

1. Enable extension **`isaacsim.nemoclaw.connector`** (Extension Manager).
2. Default backend: **`http://127.0.0.1:8010`** (WSL API).  
   To override, set Windows user env **`NEMOCLAW_BACKEND_URL`** before starting Isaac (e.g. `http://127.0.0.1:8002` for `start.bat`).
3. The panel **Send to NIM** sends:
   - `POST {BACKEND_URL}/generate` with body `{"prompt":"<your text>"}`
   - if that fails, `POST .../generate-human` (same body).

---

## 6. Test the system

1. Confirm **§3** API is running and **§2** sandbox is up.
2. In Isaac, open **NemoClaw Connector**.
3. Enter a short prompt, e.g. **`add a human`** or **`change environment hospital`**.
4. Click **Send to NIM**.
5. Expect: status success and a new prim under `/World/NemoClaw/...` or environment swap.

**CLI check from Windows (PowerShell):**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/generate" -Method Post -ContentType "application/json" -Body '{"prompt":"add a human"}'
```

---

## 7. Logging (minimal)

| Where | What |
|--------|------|
| **WSL API / Windows `main.py` terminal** | `Request received: prompt=...` then `Response sent: command=...` |
| **Isaac console / log** | `[nemoclaw] → POST ...` then `[nemoclaw] ← 200 OK keys=...` |

---

## 8. Clean file layout (`isaac-sim-backend`)

| File | Role |
|------|------|
| `wsl_nemoclaw_api.py` | WSL FastAPI :8010 |
| `main.py` | Windows FastAPI :8002 (optional) |
| `wsl_helper.py` | stdin/stdout helper for `main.py` |
| `nemo_client.py` | Spawns WSL helper from Windows |
| `prompt_builder.py` | User text → agent instructions |
| `asset_resolver.py` | Agent text → USD path / procedural spec |
| `requirements.txt` | Python deps |
| `start.bat` | Start Windows bridge |
| `run_wsl_api.sh` | Optional wrapper for WSL uvicorn |
| `SETUP_GUIDE.md` | This document |
| `CONNECTION.md` | Short architecture notes |

---

## Troubleshooting

- **`No response` in Isaac:** API not running, wrong `NEMOCLAW_BACKEND_URL`, or firewall blocking `localhost`.
- **500 / SSH errors:** Sandbox name mismatch — set `NEMOCLAW_SANDBOX` to match `openshell sandbox list`.
- **Empty or wrong asset:** Agent returned non-JSON; check WSL API logs for raw agent output snippet.
