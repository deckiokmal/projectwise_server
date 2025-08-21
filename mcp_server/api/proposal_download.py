# mcp_server/api/proposal_download.py
from __future__ import annotations

from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import APIRouter
from mcp_server.settings import Settings


router = APIRouter()
settings = Settings()

@router.get("/download/{filename}")
async def download(filename: str):
    path = Path(settings.proposal_generated_base_path) / filename
    return FileResponse(path, filename=filename)
