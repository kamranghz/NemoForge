"""
NemoForge API Bridge
FastAPI server connecting correction engine to LLM agent.
Run: python -m api.bridge  (from repo root with src/ on PYTHONPATH)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
import uvicorn

log = logging.getLogger(__name__)

app = FastAPI(
    title="NemoForge Bridge",
    version="2.0.0",
    description="Physics correction agent API"
)


class HealthResponse(BaseModel):
    status:  str
    version: str
    backend: str = "unknown"


class CorrectionRequest(BaseModel):
    scene_id:       str
    physics_report: dict
    iteration:      int                  = 0
    action_history: Optional[List[dict]] = None


class CorrectionResponse(BaseModel):
    action:     str
    parameters: dict
    reasoning:  str


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    raw: str


_llm_client = None

# Imported lazily to avoid import-time side effects; used by legacy /generate
from agent.nemoclaw_client import (  # noqa: E402
    NVIDIA_NIM_URL,
    NEMOCLAW_URL,
)


def _get_client():
    global _llm_client
    if _llm_client is None:
        from agent.nemoclaw_client import NemoClawClient
        _llm_client = NemoClawClient()
    return _llm_client


@app.get("/health", response_model=HealthResponse)
async def health():
    client = _get_client()
    ok, backend = client.is_available()
    return HealthResponse(
        status="ok",
        version="2.0.0",
        backend=backend
    )


@app.post("/correct", response_model=CorrectionResponse)
async def correct(req: CorrectionRequest):
    """
    Main endpoint: receive physics report, return correction action.
    """
    from correction.prompt_builder import build_prompt

    prompt = build_prompt(
        user_input=req.scene_id,
        physics_report=req.physics_report,
        iteration=req.iteration,
        action_history=req.action_history or []
    )

    client = _get_client()
    try:
        action = client.correct(prompt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return CorrectionResponse(**action)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Legacy endpoint for backward compatibility."""
    client = _get_client()
    ok, backend = client.is_available()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="No LLM backend available")
    try:
        if backend == "ollama":
            raw = client._call_ollama(req.prompt)
        elif backend == "nvidia_nim":
            raw = client._call_openai_compatible(
                NVIDIA_NIM_URL, req.prompt, api_key=client._api_key)
        else:
            raw = client._call_openai_compatible(
                NEMOCLAW_URL, req.prompt, api_key=None)
        return GenerateResponse(raw=raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "nemoforge.api.bridge:app",
        host="127.0.0.1",
        port=8010,
        reload=True
    )
