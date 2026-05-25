"""
Windows-side NemoClaw client.

Starts wsl_helper.py as a persistent WSL subprocess (via interactive bash so
openshell is on PATH) and communicates with it via JSON lines over stdin/stdout.

ROOT CAUSE OF PREVIOUS 500 ERRORS:
    openshell is installed at ~/.local/bin/openshell and only appears on PATH
    in *interactive* bash sessions (.bashrc sources NVM / local bin dirs).
    Spawning with `wsl python3 ...` is non-interactive → openshell not found
    → ssh-config fails → WSL helper crashes immediately.

FIX:
    Spawn with `wsl bash -i -c "python3 ..."` so .bashrc runs first.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

WSL_HELPER_PATH = "/mnt/c/PhysiClaw/isaac-sim-backend/wsl_helper.py"
STARTUP_TIMEOUT = 60  # seconds to wait for {"status": "READY"}


def _sandbox_name() -> str:
    """Read each time — matches openshell / onboarded sandbox (default physiclaw)."""
    return os.environ.get("NEMOCLAW_SANDBOX", "physiclaw")


class NemoClawClient:
    """
    Manages a persistent WSL subprocess running wsl_helper.py.

    The helper uses  openshell sandbox ssh-config <name>  +  ssh  to run the
    OpenClaw agent non-interactively inside the sandbox (see NEMOCLAW_SANDBOX).

    Pre-requisite: sandbox reachable (e.g. nemoclaw <name> connect) so
    ssh-config succeeds.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock     = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="nemoclaw")

    # ── Private / blocking ─────────────────────────────────────────────────────

    def _spawn(self) -> subprocess.Popen:
        """
        Start wsl_helper.py inside WSL and wait for {"status": "READY"}.

        KEY: we use  bash -i  so that ~/.bashrc is sourced and openshell /
        NVM / local bin directories land on PATH before python3 runs.
        """
        name = _sandbox_name()
        logger.info("Spawning WSL helper (bash -i), sandbox=%s …", name)
        proc = subprocess.Popen(
            # bash -i  →  interactive  →  .bashrc sourced  →  openshell on PATH
            [
                "wsl",
                "bash",
                "-i",
                "-c",
                f'export NEMOCLAW_SANDBOX="{name}"; python3 {WSL_HELPER_PATH}',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                stderr_dump = proc.stderr.read()
                raise RuntimeError(
                    "WSL helper exited before sending READY.\n"
                    f"Stderr:\n{stderr_dump}\n\n"
                    "Make sure the sandbox is running:\n"
                    "  WSL> nemoclaw human connect"
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # bash -i may emit PS1 / motd lines before python3 starts
                logger.debug("Non-JSON startup line (ignored): %s", line)
                continue

            if msg.get("status") == "READY":
                logger.info("WSL helper is ready.")
                return proc
            if "error" in msg:
                proc.terminate()
                raise RuntimeError(f"WSL helper startup error: {msg['error']}")

        proc.terminate()
        raise RuntimeError(
            f"WSL helper did not send READY within {STARTUP_TIMEOUT}s.\n"
            "Sandbox may not be running — run:  nemoclaw human connect"
        )

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            logger.warning("WSL helper not running — (re)starting.")
            self._proc = self._spawn()

    def _send_query(self, structured_prompt: str) -> str:
        """Send one prompt to the helper and return its response (blocking)."""
        self._ensure_alive()

        self._proc.stdin.write(json.dumps({"prompt": structured_prompt}) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            self._proc = None
            raise RuntimeError("WSL helper closed the connection unexpectedly.")

        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON from WSL helper: {exc}\nLine: {line}"
            ) from exc

        if "error" in msg:
            raise RuntimeError(f"NemoClaw error: {msg['error']}")

        return msg.get("response", "")

    # ── Public async API ───────────────────────────────────────────────────────

    async def query(self, structured_prompt: str) -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    self._executor, self._send_query, structured_prompt
                )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.exception("Unexpected NemoClaw client error")
                raise RuntimeError(f"Internal client error: {exc}") from exc

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"exit": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
        self._executor.shutdown(wait=False)
        logger.info("NemoClawClient closed.")
