from fastapi import APIRouter
from mcp_server.utils.status_tracker import get_status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/check-status/")
async def check_status(job_id: str):
    status = get_status(job_id)
    return JSONResponse(status)
