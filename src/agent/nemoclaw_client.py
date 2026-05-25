"""
NemoForge LLM Agent Client
Supports NemoClaw (NVIDIA cloud, port 8642, OpenAI-compatible)
and Ollama (local, port 11434) as fallback.
"""
import json
import re
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# NemoClaw uses NVIDIA_API_KEY for cloud inference
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')

NEMOCLAW_URL   = "http://127.0.0.1:8642/v1"
NVIDIA_NIM_URL = "https://integrate.api.nvidia.com/v1"
OLLAMA_URL     = "http://127.0.0.1:11434"
DEFAULT_MODEL  = "meta/llama-3.1-8b-instruct"
TIMEOUT        = 120

VALID_ACTIONS = frozenset({
    'snap_to_surface',
    'resolve_penetration',
    'recompute_hull',
    'repair_manifold',
    'stack_on',
    'remove_object'
})


class NemoClawClient:
    """
    Unified LLM client for NemoForge correction agent.

    Priority:
    1. NemoClaw local (port 8642) — if running
    2. NVIDIA NIM cloud — if NVIDIA_API_KEY is set
    3. Ollama local (port 11434) — fallback
    """

    def __init__(self,
                 model:   str = DEFAULT_MODEL,
                 timeout: int = TIMEOUT):
        self.model        = model
        self.timeout      = timeout
        self._backend     = None
        self._api_key     = NVIDIA_API_KEY

    def is_available(self) -> tuple[bool, str]:
        """
        Probe backends in priority order.
        Returns (available, backend_name).
        """
        import requests

        # 1. NemoClaw local
        try:
            r = requests.get(
                f"{NEMOCLAW_URL}/models", timeout=3)
            if r.status_code == 200:
                self._backend = 'nemoclaw_local'
                log.info("Backend: NemoClaw local")
                return True, 'nemoclaw_local'
        except Exception:
            pass

        # 2. NVIDIA NIM cloud
        if self._api_key:
            try:
                r = requests.get(
                    f"{NVIDIA_NIM_URL}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=5)
                if r.status_code == 200:
                    self._backend = 'nvidia_nim'
                    log.info("Backend: NVIDIA NIM cloud")
                    return True, 'nvidia_nim'
            except Exception:
                pass

        # 3. Ollama local
        try:
            r = requests.get(
                f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                self._backend = 'ollama'
                log.info("Backend: Ollama local")
                return True, 'ollama'
        except Exception:
            pass

        self._backend = None
        log.warning("No LLM backend reachable")
        return False, 'none'

    def correct(self, prompt: str) -> dict:
        """
        Send correction prompt, return parsed action dict.

        Returns:
            {action, parameters, reasoning}

        Raises:
            RuntimeError: if no backend available
            ValueError: if response is not valid action JSON
        """
        if self._backend is None:
            self.is_available()

        if self._backend is None:
            raise RuntimeError(
                "No LLM backend available. "
                "Start Ollama: ollama serve "
                "or set NVIDIA_API_KEY for cloud inference.")

        if self._backend == 'nemoclaw_local':
            raw = self._call_openai_compatible(
                NEMOCLAW_URL, prompt, api_key=None)
        elif self._backend == 'nvidia_nim':
            raw = self._call_openai_compatible(
                NVIDIA_NIM_URL, prompt, api_key=self._api_key)
        elif self._backend == 'ollama':
            raw = self._call_ollama(prompt)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        return self._parse_action(raw)

    def _call_openai_compatible(self,
                                base_url: str,
                                prompt:   str,
                                api_key:  Optional[str]) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model":           self.model,
                "messages":        [{"role": "user",
                                     "content": prompt}],
                "temperature":     0.2,
                "response_format": {"type": "json_object"}
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _parse_action(self, raw: str) -> dict:
        """Parse and validate correction action from LLM response."""
        parsed = None

        # Attempt 1: direct parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip markdown
        if parsed is None:
            cleaned = re.sub(r'```(?:json)?', '', raw).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Attempt 3: extract first JSON block
        if parsed is None:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            raise ValueError(
                f"LLM response is not valid JSON.\n"
                f"Raw response: {raw[:400]}")

        action = parsed.get('action', '')
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Valid: {sorted(VALID_ACTIONS)}")

        return {
            'action':     action,
            'parameters': parsed.get('parameters', {}),
            'reasoning':  parsed.get('reasoning', '')
        }


# Singleton
_client: Optional[NemoClawClient] = None


def get_client(model: str = DEFAULT_MODEL) -> NemoClawClient:
    """Return singleton NemoClawClient."""
    global _client
    if _client is None:
        _client = NemoClawClient(model=model)
    return _client


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    c = NemoClawClient()
    ok, backend = c.is_available()
    print(f"Backend  : {backend}")
    print(f"Available: {ok}")
    if ok:
        print(f"Model    : {c.model}")
        print(f"API key  : {'set' if c._api_key else 'not set'}")
