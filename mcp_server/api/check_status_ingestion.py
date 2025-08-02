from fastapi import APIRouter, Query, HTTPException
from mcp_server.utils.status_tracker import get_status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/check-status/")
async def check_status(job_id: str = Query(..., description="Job ID dari proses upload")):
    status = get_status(job_id)
    if not status:
        raise HTTPException(404, "Status tidak ditemukan")
    
    return JSONResponse(status)
