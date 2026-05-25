"""
Isaac Sim bridge (Windows): FastAPI → WSL helper → OpenClaw in sandbox → JSON for Isaac.

  GET  /health
  POST /generate-human | /generate   body: {"prompt": "..."}
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from asset_resolver import resolve
from nemo_client import NemoClawClient
from prompt_builder import build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_client: NemoClawClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    logger.info("Bridge starting (Windows)")
    sb = os.environ.setdefault("NEMOCLAW_SANDBOX", "physiclaw")
    logger.info("NEMOCLAW_SANDBOX=%s", sb)
    _client = NemoClawClient()
    yield
    logger.info("Bridge shutting down")
    if _client:
        _client.close()


app = FastAPI(title="Isaac Sim NemoClaw Bridge", version="3.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v.strip()


class GenerateResponse(BaseModel):
    command: str | None = None
    asset_path: str | None = None
    procedural: dict | None = None
    position: list[float]
    rotation: list[float]
    scale: list[float]


@app.get("/health")
async def health():
    alive = _client.is_alive() if _client else False
    return {"status": "ok", "session_alive": alive}


async def _generate(req: GenerateRequest) -> GenerateResponse:
    if _client is None:
        raise HTTPException(503, "Client not initialised.")

    logger.info("Request received: prompt=%r", req.prompt)
    structured = build_prompt(req.prompt)
    try:
        agent_output = await _client.query(structured)
    except RuntimeError as exc:
        logger.error("NemoClaw query failed: %s", exc)
        raise HTTPException(500, str(exc)) from exc

    result = resolve(agent_output, req.prompt)
    out = GenerateResponse(**result)
    logger.info(
        "Response sent: command=%s asset_path=%s procedural=%s",
        out.command,
        out.asset_path,
        bool(out.procedural),
    )
    return out


@app.post("/generate-human", response_model=GenerateResponse)
async def generate_human(req: GenerateRequest) -> GenerateResponse:
    return await _generate(req)


@app.post("/generate", response_model=GenerateResponse)
async def generate_alias(req: GenerateRequest) -> GenerateResponse:
    return await _generate(req)


# =============================================================================
# ── Correction endpoint ───────────────────────────────────────────────────────
# Receives a structured physics report + iteration context and returns the
# LLM's chosen correction action {action, parameters, reasoning}.
# Does NOT touch /generate or /health.
# =============================================================================

class CorrectionRequest(BaseModel):
    scene_id:       str
    physics_report: dict
    iteration:      int                    = 0
    action_history: Optional[List[dict]]  = None


class CorrectionResponse(BaseModel):
    action:     str
    parameters: dict
    reasoning:  str
    raw:        str = ""


@app.post("/correct", response_model=CorrectionResponse)
async def correct(req: CorrectionRequest) -> CorrectionResponse:
    """
    Build a physics-correction prompt from the supplied physics_report and
    send it to the NemoClaw agent.  Returns the parsed correction action.

    Request body
    ------------
    scene_id       : Scene identifier (used as user_input stub in prompt).
    physics_report : Dict with penetration_pairs, floating_objects, etc.
    iteration      : Current RAR cycle (0-based).
    action_history : List of prior accepted action records (optional).

    Response
    --------
    action     : One of the six valid correction actions.
    parameters : Action parameters dict.
    reasoning  : One-sentence LLM justification.
    raw        : Raw LLM text (for debugging).
    """
    if _client is None:
        raise HTTPException(503, "Client not initialised.")

    # Import here to avoid circular imports at module load time.
    from prompt_builder import build_prompt  # noqa: PLC0415

    # Build the full correction prompt with all available context.
    prompt = build_prompt(
        user_input     = req.scene_id,
        physics_report = req.physics_report,
        iteration      = req.iteration,
        action_history = req.action_history or [],
    )
    logger.info(
        "[/correct] scene_id=%s iteration=%d prompt_len=%d",
        req.scene_id,
        req.iteration,
        len(prompt),
    )

    # Query the agent.
    try:
        raw_response = await _client.query(prompt)
    except RuntimeError as exc:
        logger.error("[/correct] NemoClaw query failed: %s", exc)
        raise HTTPException(500, str(exc)) from exc

    logger.debug("[/correct] raw_response: %s", raw_response[:300])

    # Parse the agent's JSON response.
    parsed: Optional[dict] = None

    # Attempt 1: direct JSON parse (strips markdown fences first).
    stripped = re.sub(r"```(?:json)?", "", raw_response).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first {...} block from prose.
    if parsed is None:
        match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        logger.error(
            "[/correct] LLM did not return valid JSON. Raw: %s", raw_response[:300]
        )
        raise HTTPException(
            status_code=422,
            detail=f"LLM did not return valid JSON: {raw_response[:200]}",
        )

    return CorrectionResponse(
        action     = str(parsed.get("action",     "")),
        parameters = parsed.get("parameters", {}) if isinstance(parsed.get("parameters"), dict) else {},
        reasoning  = str(parsed.get("reasoning",  "")),
        raw        = raw_response,
    )
