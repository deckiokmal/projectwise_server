from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field

class ToolStatus(str):
    SUCCESS = "success"
    FAILURE = "failure"
    EMPTY   = "empty"   # ← khusus hasil kosong

class BaseToolResponse(BaseModel):
    status: ToolStatus = Field(..., description="success | failure | empty")
    error: Optional[str] = None   # wajib None kalau status ≠ failure

class RAGResult(BaseModel):
    passages: List[str] = []
    citations: List[str] = []

class RetrievalResponse(BaseToolResponse):
    data: Optional[RAGResult] = None   # None jika status != success

class GenericDataResponse(BaseToolResponse):
    data: Optional[Dict[str, Any]] = None
