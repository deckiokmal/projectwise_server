# mcp_server/utils/cpu_workers.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

# HF
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

# OpenAI (opsional)
try:
    import tiktoken
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
except Exception:
    tiktoken = None
    OpenAITokenizer = None


def _build_tokenizer(
    *,
    kind: str = "openai",
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    if kind == "openai":
        assert tiktoken and OpenAITokenizer, "tiktoken/OpenAITokenizer belum terpasang."
        enc = tiktoken.encoding_for_model(model_id or "gpt-4o")
        # OpenAI tokenizer butuh max_tokens = context window tokenizer
        return OpenAITokenizer(tokenizer=enc, max_tokens=max_tokens or 128 * 1024)

    # default: Hugging Face
    # Contoh: "nomic-ai/nomic-embed-text-v1.5"
    tok = AutoTokenizer.from_pretrained(
        model_id or "nomic-ai/nomic-embed-text-v1.5", use_fast=True
    )
    # max_tokens bisa dibiarkan None agar diambil dari tokenizer (jika tersedia)
    return HuggingFaceTokenizer(tokenizer=tok, max_tokens=max_tokens) # type: ignore


def convert_and_chunk_pdf_worker(
    pdf_path: str,
    *,
    tokenizer_kind: str = "openai",  # "hf" | "openai"
    tokenizer_model: Optional[
        str
    ] = None,  # contoh HF: "nomic-ai/nomic-embed-text-v1.5"
    tokenizer_max_tokens: Optional[int] = None,
    merge_peers: bool = True,
    export_markdown: bool = True,
) -> Dict[str, Any]:
    """
    Worker top-level (ProcessPool):
      - convert PDF -> dl_doc
      - chunk via HybridChunker (tokenizer-aware)
      - return payload ringan (chunk text + meta) + markdown opsional
    """
    conv = DocumentConverter()
    conv_result = conv.convert(str(pdf_path))
    dl_doc = conv_result.document

    tokenizer = _build_tokenizer(
        kind=tokenizer_kind,
        model_id=tokenizer_model,
        max_tokens=tokenizer_max_tokens,
    )

    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=merge_peers)
    chunk_iter = chunker.chunk(dl_doc)

    chunks_payload: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(
            chunk=ch
        )  # gunakan konteks heading/section
        meta = getattr(ch, "metadata", {}) or {}
        chunks_payload.append(
            {
                "index": i,
                "text": enriched_text,
                "meta": {
                    "pages": (meta or {}).get("pages"),
                    "headings": (meta or {}).get("headings"),
                    "chunk_type": (meta or {}).get("type"),
                },
            }
        )

    markdown = dl_doc.export_to_markdown() if export_markdown else None
    return {"status": "success", "chunks": chunks_payload, "markdown": markdown}
