"""
WSL-side helper — runs inside WSL via:
    wsl python3 /mnt/c/PhysiClaw/isaac-sim-backend/wsl_helper.py

Protocol (JSON lines over stdin/stdout):
  Windows → WSL  :  {"prompt": "<structured prompt>"}
  WSL → Windows  :  {"response": "<agent output>"}
               or  {"error": "<message>"}
  Windows → WSL  :  {"exit": true}   (clean shutdown)

How it connects to the agent
-----------------------------
Uses the same mechanism as the NemoClaw Telegram bridge
(scripts/telegram-bridge.js): ask openshell for an SSH config for
the sandbox, then run the agent non-interactively via SSH:

    openshell sandbox ssh-config human   → writes SSH host/port/key config
    ssh -T -F <config> openshell-human \
        'nemoclaw-start openclaw agent --agent main --local -m "..."'

No pexpect, no TTY manipulation — just a subprocess + stdout capture.
The sandbox must be running, e.g.:
    nemoclaw physiclaw connect
  (use your sandbox name; set NEMOCLAW_SANDBOX for the API/helper.)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

def _sandbox_name() -> str:
    """Sandbox name from env (e.g. physiclaw). Default matches legacy docs."""
    return os.environ.get("NEMOCLAW_SANDBOX", "human")


AGENT_TIMEOUT = 120     # seconds — model inference can be slow

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Lines that the NemoClaw startup emits before the real agent response.
# These are filtered out to obtain clean output (mirrors telegram-bridge.js).
_NOISE_FRAGMENTS = [
    "Setting up NemoClaw",
    "[plugins]",
    "(node:",
    "[UNDICI",
    "Use `node --trace",
    "NemoClaw ready",
    "NemoClaw registered",
    "openclaw agent",
    "┌─", "│ ", "└─",
    "Endpoint:", "Provider:", "Model:", "Slash:",
    "[gateway]",
    "auto-pair",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def send(obj: dict) -> None:
    """Write one JSON line to stdout (the Windows FastAPI process reads this)."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def filter_noise(raw: str) -> str:
    """Remove NemoClaw startup banner lines from agent output."""
    lines = [
        line for line in raw.splitlines()
        if line.strip() and not any(frag in line for frag in _NOISE_FRAGMENTS)
    ]
    return "\n".join(lines).strip()


# ── SSH config management ──────────────────────────────────────────────────────

def get_ssh_conf() -> str:
    """
    Ask openshell for an SSH config block that lets us connect to the sandbox.
    Returns the raw config text.
    """
    name = _sandbox_name()
    result = subprocess.run(
        ["openshell", "sandbox", "ssh-config", name],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Cannot get SSH config for sandbox '{name}'.\n"
            f"Is the sandbox running?  In a separate WSL terminal run:\n"
            f"    nemoclaw {name} connect\n"
            f"openshell stderr: {result.stderr.strip()[:300]}"
        )
    return result.stdout


def write_ssh_conf(conf_text: str, conf_dir: str) -> str:
    """Write SSH config to a temp file inside conf_dir and return its path."""
    conf_path = os.path.join(conf_dir, "ssh.conf")
    with open(conf_path, "w") as fh:
        fh.write(conf_text)
    os.chmod(conf_path, 0o600)
    return conf_path


# ── Agent query ────────────────────────────────────────────────────────────────

def query_agent(conf_path: str, prompt: str) -> str:
    """
    Run openclaw agent non-interactively inside the sandbox via SSH.

    The inner command mirrors scripts/telegram-bridge.js:
        nemoclaw-start openclaw agent --agent main --local -m '<prompt>'
    """
    session_id = f"nc-{uuid.uuid4().hex[:8]}"

    # Single-quote-safe escaping: replace ' with '\''
    safe_prompt     = prompt.replace("'", "'\\''")
    safe_session_id = session_id.replace("'", "")   # already hex, but be safe

    inner_cmd = (
        f"nemoclaw-start openclaw agent --agent main --local "
        f"-m '{safe_prompt}' --session-id '{safe_session_id}'"
    )

    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout=30",
            "-F", conf_path,
            f"openshell-{_sandbox_name()}",
            inner_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=AGENT_TIMEOUT + 10,
    )

    clean = filter_noise(strip_ansi(result.stdout))

    if not clean:
        if result.returncode != 0:
            raise RuntimeError(
                f"Agent SSH command failed (exit {result.returncode}).\n"
                f"stderr: {result.stderr.strip()[:400]}"
            )
        return "(no response)"

    return clean


def run_single_query(prompt: str) -> str:
    """
    One-shot agent query for HTTP APIs (no stdin/stdout protocol).
    Creates a temporary SSH config each call — fine for low QPS prototypes.
    """
    if not (prompt or "").strip():
        raise ValueError("empty prompt")
    ssh_conf_text = get_ssh_conf()
    conf_dir = tempfile.mkdtemp(prefix="nc-q-")
    try:
        conf_path = write_ssh_conf(ssh_conf_text, conf_dir)
        return query_agent(conf_path, prompt)
    finally:
        shutil.rmtree(conf_dir, ignore_errors=True)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Obtain SSH config once ────────────────────────────────────────────────
    try:
        ssh_conf_text = get_ssh_conf()
    except Exception as exc:
        send({"error": f"Sandbox not reachable: {exc}"})
        sys.exit(1)

    conf_dir  = tempfile.mkdtemp(prefix="nc-bridge-")
    conf_path = write_ssh_conf(ssh_conf_text, conf_dir)

    try:
        # Signal the Windows side that we are ready
        send({"status": "READY"})

        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                send({"error": f"Bad JSON from Windows: {exc}"})
                continue

            if req.get("exit"):
                break

            prompt = req.get("prompt", "")
            if not prompt:
                send({"error": "Empty prompt."})
                continue

            try:
                # Refresh SSH config if the sandbox was restarted
                if not os.path.exists(conf_path):
                    ssh_conf_text = get_ssh_conf()
                    conf_path     = write_ssh_conf(ssh_conf_text, conf_dir)

                response = query_agent(conf_path, prompt)
                send({"response": response})

            except RuntimeError as exc:
                send({"error": str(exc)})

                # Attempt config refresh for next request in case sandbox restarted
                try:
                    ssh_conf_text = get_ssh_conf()
                    conf_path     = write_ssh_conf(ssh_conf_text, conf_dir)
                except Exception:
                    pass

    finally:
        shutil.rmtree(conf_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
