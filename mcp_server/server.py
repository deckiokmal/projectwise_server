from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP  # type: ignore
from mcp_server.tools.rag_tools import RAGTools
from mcp_server.tools.docx_tools import DocGeneratorTools
from mcp_server.settings import Settings

settings = Settings()  # type: ignore

# ---------------------------------------------------------------------------
# MCP Server Instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="projectwise",
    host="localhost",
    port=5000,
)

rag_tools = RAGTools()
doc_tools = DocGeneratorTools()

# ---------------------------------------------------------------------------
# 1. Ingest product-standard PDFs into vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="add_product_knowledge",
    title="Ingest Product Knowledge",
    description=(
        "Tambahkan seluruh file PDF di direktori `product_standard` ke vectorstore RAG "
        "sebagai product knowledge dengan metadata `project` dan `tahun`."
    ),
)
def add_product_knowledge_tool(
    base_dir: Optional[str] = None,
    project_name: str = "product_standard",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """Index semua PDF product standard.

    Args:
        base_dir: Path direktori PDF.
        project_name: Label project.
        tahun: Label tahun.
    """
    rag_tools.add_product_knowledge(base_dir, project_name, tahun)
    return {"status": "Product knowledge ingestion selesai."}


# ---------------------------------------------------------------------------
# 2. Ingest KAK/TOR PDFs → Markdown → vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="add_kak_tor_knowledge",
    title="Ingest KAK/TOR Documents",
    description=(
        "Konversi seluruh PDF KAK/TOR di direktori ke Markdown, simpan file, lalu indeks "
        "ke vectorstore RAG dengan metadata custom."
    ),
)
def add_kak_tor_knowledge_tool(
    project: Optional[str] = None,
    tahun: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest KAK/TOR (PDF → MD → RAG)."""
    rag_tools.add_kak_tor_knowledge(project=project, tahun=tahun)
    return {"status": "KAK/TOR ingestion selesai."}


# ---------------------------------------------------------------------------
# 3. Ingest KAK/TOR Markdown → vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="add_kak_tor_summaries_knowledge",
    title="Ingest summaries KAK/TOR Markdown",
    description="Indeks file/direktori Markdown KAK/TOR langsung ke vectorstore RAG.",
)
def add_kak_tor_md_knowledge_tool(
    markdown_path: Optional[str] = None,
    project: str = "default",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """Index KAK/TOR markdown."""
    results = rag_tools.add_kak_tor_summaries_knowledge(markdown_path, project, tahun)
    return {"results": results}


# ---------------------------------------------------------------------------
# 4. Build Summary Tender Payload
# ---------------------------------------------------------------------------


@mcp.tool(
    name="build_summary_tender_payload",
    title="Build Summary Tender Payload",
    description=(
        "Gabungkan prompt instruction (.txt) dengan file Markdown KAK/TOR, "
        "kembali dict {instruction, context}."
    ),
)
def build_summary_tender_payload_tool(
    prompt_instruction_name: str,
    kak_tor_name: Optional[str] = None,
) -> Dict[str, str]:
    """Buat payload summary tender."""
    return rag_tools.build_summary_tender_payload(prompt_instruction_name, kak_tor_name)


# ---------------------------------------------------------------------------
# 5. Build instruction & context for LLM
# ---------------------------------------------------------------------------


@mcp.tool(
    name="build_instruction_context",
    title="Build Instruction Context",
    description="Gabungkan template prompt dengan markdown KAK/TOR terpilih.",
)
def build_instruction_context_tool(
    template_name: str,
    kak_md_dir: Optional[str] = None,
    selected_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Kembalikan instruksi & konteks LLM."""
    instruksi, context = rag_tools.build_instruction_context(
        template_name, kak_md_dir, selected_files
    )
    return {"instruksi": instruksi, "context": context}


# ---------------------------------------------------------------------------
# 6. Retrieval with similarity + metadata filter
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag_retrieval",
    title="RAG Retrieval",
    description=(
        "Similarity search dengan filter metadata, hasilkan potongan teks relevan + citation."
    ),
)
def rag_retrieval_tool(
    query: str,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """RAG retrieval."""
    result = rag_tools.retrieval_with_filter(query, k, metadata_filter)
    return {"result": result}


# ---------------------------------------------------------------------------
# 7. Reset / rebuild vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="reset_vectordb",
    title="Reset Vector Store",
    description="Hapus seluruh data dan rebuild tabel vectorstore kosong.",
)
def reset_vectorstore_tool() -> Dict[str, Any]:
    """Reset vectorstore."""
    rag_tools.reset_knowledge_base()
    return {"status": "Vectorstore berhasil di-reset."}


# ---------------------------------------------------------------------------
# 8. Update metadata massal
# ---------------------------------------------------------------------------


@mcp.tool(
    name="update_chunk_metadata",
    title="Update Chunk Metadata",
    description="Perbarui metadata untuk semua chunk sesuai filter.",
)
def update_chunk_metadata_tool(
    metadata_filter: Dict[str, Any],
    new_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Batch update metadata chunk."""
    updated = rag_tools.update_chunk_metadata(metadata_filter, new_metadata)
    return {"updated": updated}


# ---------------------------------------------------------------------------
# 9. Get vectorstore statistics
# ---------------------------------------------------------------------------


@mcp.tool(
    name="get_vectorstore_stats",
    title="Get Vectorstore Statistics",
    description="Ambil statistik vectorstore (rows, size, project unik, distribusi tahun).",
)
def get_vectorstore_stats_tool() -> Dict[str, Any]:
    """Statistik vectorstore."""
    return rag_tools.get_vectorstore_stats()


# ---------------------------------------------------------------------------
# 10. Rebuild all embeddings
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rebuild_all_embeddings",
    title="Rebuild All Embeddings",
    description="Hitung ulang embedding semua chunk (gunakan model baru).",
)
def rebuild_all_embeddings_tool(batch_size: int = 100) -> Dict[str, Any]:
    """Rebuild embeddings."""
    rag_tools.rebuild_all_embeddings(batch_size)
    return {"status": "Rebuild embeddings selesai."}


# ---------------------------------------------------------------------------
# 11. List unique metadata values
# ---------------------------------------------------------------------------


@mcp.tool(
    name="list_metadata_values",
    title="List Metadata Values",
    description="Kembalikan daftar unik nilai untuk field metadata tertentu.",
)
def list_metadata_values_tool(field: str) -> Dict[str, Any]:
    """Daftar nilai unik metadata."""
    values = rag_tools.list_metadata_values(field)
    return {"values": values}


# ---------------------------------------------------------------------------
# 12. Retrieve Product Context via RAG + Prompt Template
# ---------------------------------------------------------------------------


@mcp.tool(
    name="retrieve_product_context",
    title="Retrieve Product Context",
    description="Ambil konteks produk via RAG + template prompt.",
)
def retrieve_product_context_tool(
    product: str,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrapper DocGeneratorTools.retrieve_product_context."""
    return doc_tools.retrieve_product_context(
        product, k, metadata_filter, prompt_template
    )


# ---------------------------------------------------------------------------
# 13. Extract Text from Document → Markdown
# ---------------------------------------------------------------------------


@mcp.tool(
    name="extract_document_text",
    title="Extract Document Text",
    description="Ekstrak teks dari dokumen (.pdf/.docx/.md) → markdown.",
)
def extract_document_text_tool(file_path: str) -> Dict[str, Any]:
    """Wrapper DocGeneratorTools.extract_document_text."""
    return doc_tools.extract_document_text(file_path)


# ---------------------------------------------------------------------------
# 14. Generate Proposal Document (.docx)
# ---------------------------------------------------------------------------


@mcp.tool(
    name="generate_proposal_docx",
    title="Generate Proposal Document",
    description="Render dan simpan dokumen proposal Word (.docx).",
)
def generate_proposal_docx_tool(
    context: Dict[str, Any],
    override_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrapper DocGeneratorTools.generate_proposal."""
    return doc_tools.generate_proposal(context, override_template)


# ---------------------------------------------------------------------------
# Run the server if executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse")
