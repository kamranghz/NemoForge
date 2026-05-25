"""
WSL API: FastAPI on 0.0.0.0:8010 → openshell ssh-config + SSH → OpenClaw agent in sandbox.

  python3 wsl_nemoclaw_api.py
  # or: python3 -m uvicorn wsl_nemoclaw_api:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("NEMOCLAW_SANDBOX", "physiclaw")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool

from asset_resolver import resolve
from prompt_builder import build_prompt
from wsl_helper import run_single_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NemoClaw WSL API", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {
        "status": "ok",
        "wsl_api": True,
        "sandbox": os.environ.get("NEMOCLAW_SANDBOX", "physiclaw"),
    }


async def _generate_impl(req: GenerateRequest) -> GenerateResponse:
    logger.info("Request received: prompt=%r", req.prompt)
    structured = build_prompt(req.prompt)
    try:
        agent_output = await run_in_threadpool(run_single_query, structured)
    except (RuntimeError, ValueError) as exc:
        logger.error("Sandbox agent failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = resolve(agent_output, req.prompt)
    out = GenerateResponse(**result)
    logger.info(
        "Response sent: command=%s asset_path=%s procedural=%s",
        out.command,
        out.asset_path,
        bool(out.procedural),
    )
    return out


@app.post("/generate", response_model=GenerateResponse)
@app.post("/generate-human", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    return await _generate_impl(req)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
