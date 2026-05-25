"""
nemoclaw_client.py — Clean LLM client for correction-action generation.

Zero dependency on ovphysx, Isaac Sim, omni, or pxr.
Requires: requests (stdlib-only fallback shown in docstring).

Supported backends (tried in order)
-------------------------------------
1. NemoClaw  — OpenAI-compatible API on http://127.0.0.1:8642/v1
2. Ollama    — native API on http://127.0.0.1:11434

Public API
----------
    client = NemoClawClient()
    available, backend = client.is_available()   # -> (True, "nemoclaw") | ...
    result  = client.correct(prompt)             # -> {action, parameters, reasoning}

    # Module-level singleton (lazy init)
    from nemoclaw_client import get_client
    result = get_client().correct(prompt)

LLM output contract
--------------------
The LLM MUST return JSON with exactly these keys::

    {
      "action":     "<action_name>",
      "parameters": {"object_id": "...", ...},
      "reasoning":  "<one sentence>"
    }

Valid actions: snap_to_surface, resolve_penetration, recompute_hull,
               repair_manifold, stack_on, remove_object.

Raises ValueError on malformed / invalid-action responses.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# ── Valid correction actions (single source of truth) ─────────────────────────
_VALID_ACTIONS = frozenset({
    "snap_to_surface",
    "resolve_penetration",
    "recompute_hull",
    "repair_manifold",
    "stack_on",
    "remove_object",
})

# ── Required top-level keys in LLM JSON response ──────────────────────────────
_REQUIRED_KEYS = {"action", "parameters", "reasoning"}


def _requests_post(url: str, payload: dict, timeout: int) -> dict:
    """
    Thin wrapper around requests.post that returns the parsed JSON body.
    Raises RuntimeError on any network or HTTP error.
    """
    try:
        import requests                 # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required: pip install requests"
        ) from exc

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Connection refused: {url}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Request timed out after {timeout}s: {url}") from exc
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed ({url}): {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} from {url}: {resp.text[:300]}"
        )
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Response from {url} is not valid JSON: {resp.text[:300]}"
        ) from exc


def _requests_get(url: str, timeout: int) -> dict:
    """GET *url* and return parsed JSON; raises RuntimeError on failure."""
    try:
        import requests                 # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("pip install requests") from exc

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}")
    return resp.json()


class NemoClawClient:
    """
    Sends physics-correction prompts to NemoClaw or Ollama and parses
    the structured action JSON they return.

    Parameters
    ----------
    nemoclaw_url : Base URL of NemoClaw's OpenAI-compatible API.
    ollama_url   : Base URL of the local Ollama server.
    model        : Model name / tag used for both backends.
    timeout      : Per-request timeout in seconds.
    """

    def __init__(
        self,
        nemoclaw_url: str = "http://127.0.0.1:8642/v1",
        ollama_url:   str = "http://127.0.0.1:11434",
        model:        str = "qwen3:14b",
        timeout:      int = 60,
    ) -> None:
        self._nemoclaw_url = nemoclaw_url.rstrip("/")
        self._ollama_url   = ollama_url.rstrip("/")
        self._model        = model
        self._timeout      = timeout
        # Cache the discovered backend between calls (reset on failure)
        self._active_backend: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_available(self) -> tuple[bool, str]:
        """
        Probe backends in priority order.

        Returns
        -------
        (True,  "nemoclaw") if NemoClaw /v1/models responds.
        (True,  "ollama")   if Ollama /api/tags responds.
        (False, "none")     if neither is reachable.
        """
        # Try NemoClaw first
        try:
            _requests_get(f"{self._nemoclaw_url}/models", timeout=5)
            log.info("[nemoclaw_client] NemoClaw reachable at %s", self._nemoclaw_url)
            self._active_backend = "nemoclaw"
            return True, "nemoclaw"
        except RuntimeError as exc:
            log.debug("[nemoclaw_client] NemoClaw unavailable: %s", exc)

        # Try Ollama fallback
        try:
            _requests_get(f"{self._ollama_url}/api/tags", timeout=5)
            log.info("[nemoclaw_client] Ollama reachable at %s", self._ollama_url)
            self._active_backend = "ollama"
            return True, "ollama"
        except RuntimeError as exc:
            log.debug("[nemoclaw_client] Ollama unavailable: %s", exc)

        log.warning("[nemoclaw_client] No LLM backend reachable.")
        self._active_backend = None
        return False, "none"

    def correct(self, prompt: str) -> dict:
        """
        Send *prompt* to the best available backend and return a parsed
        correction-action dict.

        Parameters
        ----------
        prompt : Complete correction prompt string from prompt_builder.build_prompt().

        Returns
        -------
        dict with keys: action (str), parameters (dict), reasoning (str).

        Raises
        ------
        RuntimeError  : No backend is reachable.
        ValueError    : LLM returned JSON that fails schema validation.
        """
        # Discover backend if not already cached
        if self._active_backend is None:
            available, backend = self.is_available()
            if not available:
                raise RuntimeError(
                    "No LLM backend reachable. "
                    f"Start NemoClaw at {self._nemoclaw_url} "
                    f"or Ollama at {self._ollama_url}."
                )

        # Try the active backend; on failure re-probe and try the other.
        try:
            if self._active_backend == "nemoclaw":
                raw = self._call_nemoclaw(prompt)
            else:
                raw = self._call_ollama(prompt)
        except RuntimeError as primary_exc:
            log.warning(
                "[nemoclaw_client] Primary backend '%s' failed (%s); re-probing …",
                self._active_backend,
                primary_exc,
            )
            self._active_backend = None
            available, backend = self.is_available()
            if not available:
                raise RuntimeError(
                    f"Primary backend failed ({primary_exc}) and no fallback reachable."
                ) from primary_exc
            raw = (
                self._call_nemoclaw(prompt)
                if backend == "nemoclaw"
                else self._call_ollama(prompt)
            )

        return self._parse_action(raw)

    # ── Backend callers ────────────────────────────────────────────────────────

    def _call_nemoclaw(self, prompt: str) -> str:
        """
        POST to NemoClaw /v1/chat/completions (OpenAI-compatible).

        Uses response_format=json_object to instruct the model to return
        valid JSON directly without markdown or prose wrapping.
        """
        url = f"{self._nemoclaw_url}/chat/completions"
        payload = {
            "model":           self._model,
            "messages":        [{"role": "user", "content": prompt}],
            "temperature":     0.2,
            "response_format": {"type": "json_object"},
        }
        log.info("[nemoclaw_client] → NemoClaw %s  model=%s", url, self._model)
        body = _requests_post(url, payload, self._timeout)

        # OpenAI-compatible shape: choices[0].message.content
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected NemoClaw response shape: {exc}\nBody: {str(body)[:300]}"
            ) from exc

        log.debug("[nemoclaw_client] ← NemoClaw raw: %s", content[:200])
        return content

    def _call_ollama(self, prompt: str) -> str:
        """
        POST to Ollama /api/generate (native Ollama API).

        format="json" tells Ollama to constrain output to valid JSON.
        """
        url = f"{self._ollama_url}/api/generate"
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        log.info("[nemoclaw_client] → Ollama %s  model=%s", url, self._model)
        body = _requests_post(url, payload, self._timeout)

        try:
            content = body["response"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response shape: {exc}\nBody: {str(body)[:300]}"
            ) from exc

        log.debug("[nemoclaw_client] ← Ollama raw: %s", content[:200])
        return content

    # ── Response parser / validator ────────────────────────────────────────────

    def _parse_action(self, raw: str) -> dict:
        """
        Parse *raw* as JSON and validate the correction-action schema.

        Accepts:
        - Plain JSON string: ``{"action": "...", ...}``
        - JSON inside a markdown fence: ````json ... ````
        - JSON embedded anywhere in prose (last resort regex search)

        Returns
        -------
        dict with validated keys: action, parameters, reasoning.

        Raises
        ------
        ValueError : raw cannot be parsed as JSON, or the parsed dict
                     is missing required keys, or action is not in _VALID_ACTIONS.
        """
        # Step 1: strip markdown fences if present
        stripped = re.sub(r"```(?:json)?", "", raw).strip()

        # Step 2: attempt full-string parse
        parsed: Optional[dict] = None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Step 3: fallback — find first {...} block in the text
        if parsed is None:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            raise ValueError(
                f"LLM response could not be parsed as JSON.\n"
                f"Raw (first 300 chars): {raw[:300]}"
            )

        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM response parsed as {type(parsed).__name__}, expected dict."
            )

        # Step 4: required keys
        missing = _REQUIRED_KEYS - parsed.keys()
        if missing:
            raise ValueError(
                f"LLM response missing required keys: {missing}\nParsed: {parsed}"
            )

        # Step 5: valid action value
        action = str(parsed.get("action", "")).strip()
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of: {sorted(_VALID_ACTIONS)}"
            )

        # Normalise: ensure parameters is always a dict
        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        return {
            "action":     action,
            "parameters": parameters,
            "reasoning":  str(parsed.get("reasoning", "")),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_client: Optional[NemoClawClient] = None


def get_client() -> NemoClawClient:
    """
    Return the module-level NemoClawClient singleton, creating it on first call.

    Safe to call from multiple modules; uses default constructor settings.
    Override per-instance by constructing NemoClawClient() directly.
    """
    global _client
    if _client is None:
        _client = NemoClawClient()
    return _client


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    c = NemoClawClient()
    available, backend = c.is_available()
    print(f"Backend : {backend}")
    print(f"Available: {available}")

    if len(sys.argv) > 1:
        # If a prompt string is passed as the first argument, run a quick test.
        test_prompt = sys.argv[1]
        print(f"\nSending test prompt ({len(test_prompt)} chars) …")
        try:
            result = c.correct(test_prompt)
            import json as _json
            print(_json.dumps(result, indent=2))
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
