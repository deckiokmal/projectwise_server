from pydantic import BaseModel
from lancedb.pydantic import LanceModel
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Skema untuk menyimpan metadata tiap chunk dokumen
# ──────────────────────────────────────────────────────────────
class ChunkMetadata(LanceModel):
    filename: str
    source: str
    chunk_index: int
    pelanggan: Optional[str]
    category: Optional[str]
    product: Optional[str]
    tahun: Optional[str]
    project: Optional[str]


class RagQuery(BaseModel):
    question: str


class RagResponse(BaseModel):
    answer: str
