# mcp_server/tools/retrieval_tools.py
from __future__ import annotations

from typing import Optional, Dict, Any

from mcp_server.utils.logger import get_logger
from mcp_server.utils.rag_pipeline import RAGPipeline
from mcp_server.settings import Settings


logger = get_logger(__name__)
settings = Settings()
rag = RAGPipeline(settings)


async def retrieval(
    query: str, k: Optional[int] = 10, metadata_filter: Optional[Dict[str, Any]] = None
):
    """Retrieval ke vector database.

    Args:
        query (str): user message
        k (Optional[int], optional): top k result. Defaults to 10.
        metadata_filter (Optional[Dict[str, Any]], optional): project, pelanggan, product, category, tahun. Defaults to None.

    Returns:
        result (Dict[str, Any]): {"status": "success", "message": message, **data}
    """
    result: Dict[str, Any] = await rag.retrieval(
        query=query, k=k, metadata_filter=metadata_filter
    )
    logger.info(f"Hasil retrieval: {result['message'][:50]}")
    return result
