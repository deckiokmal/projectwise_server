from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from mcp_server.tools.rag_tools import RAGTools
from mcp_server.tools.docx_tools import DocGeneratorTools
from mcp_server.settings import Settings

settings = Settings()  # type: ignore

# create an MCP server for SSE Transport
mcp = FastMCP(
    name="projectwise",
    host="localhost",
    port=5000,
)

# instantiate Tools module once
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
    structured_output=True,
)
def add_product_knowledge_tool(
    base_dir: Optional[str] = None,
    project_name: str = "product_standard",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """
    Args:
        base_dir: path ke folder PDF product_standard (default dari Settings).
        project_name: metadata project label.
        tahun: metadata tahun label.
    Returns:
        status: jumlah chunk yang berhasil diindeks atau peringatan.
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
    structured_output=True,
)
def add_kak_tor_knowledge_tool(
    base_dir: Optional[str] = None,
    md_dir: Optional[str] = None,
    project: Optional[str] = None,
    tahun: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Args:
        base_dir: path ke folder PDF KAK/TOR.
        md_dir: path ke folder output Markdown.
        project: metadata project label.
        tahun: metadata tahun label.
    Returns:
        status: ringkasan proses indexing PDF KAK/TOR.
    """
    rag_tools.add_kak_tor_knowledge(base_dir, md_dir, project, tahun)
    return {"status": "KAK/TOR ingestion selesai."}


# ---------------------------------------------------------------------------
# 3. Ingest KAK/TOR Markdown → vectorstore
# ---------------------------------------------------------------------------
@mcp.tool(
    name="add_kak_tor_md_knowledge",
    title="Ingest KAK/TOR Markdown",
    description=(
        "Indeks file atau direktori Markdown KAK/TOR langsung ke vectorstore RAG tanpa "
        "menyimpan ulang ke disk."
    ),
    structured_output=True,
)
def add_kak_tor_md_knowledge_tool(
    markdown_path: Optional[str] = None,
    project: str = "default",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """
    Args:
        markdown_path: path ke file atau direktori .md KAK/TOR.
        project: metadata project label.
        tahun: metadata tahun label.
    Returns:
        results: list dict per-file hasil indexing.
    """
    results = rag_tools.add_kak_tor_md_knowledge(markdown_path, project, tahun)
    return {"results": results}


# ---------------------------------------------------------------------------
# Build Summary Tender Payload
# ---------------------------------------------------------------------------
@mcp.tool(
    name="build_summary_tender_payload",
    title="Build Summary Tender Payload",
    description=(
        "Menggabungkan prompt instruction (template .txt) dengan satu file "
        "Markdown KAK/TOR, lalu mengembalikan dict "
        '{"instruction":…, "context":…}.'
    ),
    structured_output=True,
)
def build_summary_tender_payload_tool(
    prompt_instruction_name: str,
    kak_tor_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Args:
        prompt_instruction_name : nama file .txt di folder templates (tanpa ekstensi)
        kak_tor_name            : nama file .md KAK/TOR

    Returns:
        instruction : teks prompt instruction (.txt)
        context     : konten markdown KAK/TOR (.md)
    """
    return rag_tools.build_summary_tender_payload(prompt_instruction_name, kak_tor_name)


# ---------------------------------------------------------------------------
# 4. Build instruction & context for LLM
# ---------------------------------------------------------------------------
@mcp.tool(
    name="build_instruction_context",
    title="Build Instruction Context",
    description=(
        "Gabungkan prompt template dengan konten Markdown KAK/TOR terpilih dan kembalikan "
        "tuple (instruksi, context) untuk LLM."
    ),
    structured_output=True,
)
def build_instruction_context_tool(
    template_name: str,
    kak_md_dir: Optional[str] = None,
    selected_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Args:
        template_name: nama file template (.txt) tanpa ekstensi.
        kak_md_dir: path ke folder Markdown (default dari Settings).
        selected_files: list nama file .md yang ingin digabung.
    Returns:
        instruksi: template prompt LLM.
        context: gabungan konten markdown.
    """
    instruksi, context = rag_tools.build_instruction_context(
        template_name, kak_md_dir, selected_files
    )
    return {"instruksi": instruksi, "context": context}


# ---------------------------------------------------------------------------
# 5. Retrieval with similarity + metadata filter
# ---------------------------------------------------------------------------
@mcp.tool(
    name="rag_retrieval",
    title="RAG Retrieval",
    description=(
        "Lakukan similarity search dan filter metadata di vectorstore, kembalikan potongan teks "
        "relevan beserta citation."
    ),
    structured_output=True,
)
def rag_retrieval_tool(
    query: str,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Args:
        query: pertanyaan atau kata kunci.
        k: jumlah top-k chunk.
        metadata_filter: dict filter seperti {"project":"A","tahun":"2025"}.
    Returns:
        result: teks jawaban berisi potongan dengan citation.
    """
    result = rag_tools.retrieval_with_filter(query, k, metadata_filter)
    return {"result": result}


# ---------------------------------------------------------------------------
# 6. Reset / rebuild empty vectorstore
# ---------------------------------------------------------------------------
@mcp.tool(
    name="reset_vectordb",
    title="Reset Vector Store",
    description="Hapus semua data lama dan buat ulang tabel vectorstore kosong.",
    structured_output=True,
)
def reset_vectorstore_tool() -> Dict[str, Any]:
    rag_tools.reset_knowledge_base()
    return {"status": "Vectorstore berhasil di-reset."}


# ---------------------------------------------------------------------------
# 7. Update metadata massal
# ---------------------------------------------------------------------------
@mcp.tool(
    name="update_chunk_metadata",
    title="Update Chunk Metadata",
    description=(
        "Perbarui metadata untuk semua chunk sesuai filter, misalnya mengganti project "
        "atau tahun."
    ),
    structured_output=True,
)
def update_chunk_metadata_tool(
    metadata_filter: Dict[str, Any],
    new_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Args:
        metadata_filter: dict filter chunk, e.g. {"project":"OldProj"}.
        new_metadata: dict metadata baru, e.g. {"project":"NewProj"}.
    Returns:
        updated: jumlah chunk yang di-update.
    """
    updated = rag_tools.update_chunk_metadata(metadata_filter, new_metadata)
    return {"updated": updated}


# ---------------------------------------------------------------------------
# 8. Get vectorstore statistics
# ---------------------------------------------------------------------------
@mcp.tool(
    name="get_vectorstore_stats",
    title="Get Vectorstore Statistics",
    description=(
        "Ambil statistik vectorstore: total rows, ukuran (MB), daftar project unik, "
        "distribusi tahun."
    ),
    structured_output=True,
)
def get_vectorstore_stats_tool() -> Dict[str, Any]:
    stats = rag_tools.get_vectorstore_stats()
    return stats


# ---------------------------------------------------------------------------
# 9. Rebuild all embeddings
# ---------------------------------------------------------------------------
@mcp.tool(
    name="rebuild_all_embeddings",
    title="Rebuild All Embeddings",
    description="Re-calc ulang embedding untuk semua chunk dan index ulang setelah pergantian model.",
    structured_output=True,
)
def rebuild_all_embeddings_tool(
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Args:
        batch_size: ignored (semua chunks diproses sekaligus).
    Returns:
        status: ringkasan jumlah chunk yang diembed ulang.
    """
    rag_tools.rebuild_all_embeddings(batch_size)
    return {"status": "Rebuild embeddings selesai."}


# ---------------------------------------------------------------------------
# 10. List unique metadata values
# ---------------------------------------------------------------------------
@mcp.tool(
    name="list_metadata_values",
    title="List Metadata Values",
    description="Kembalikan daftar unik nilai untuk field metadata tertentu (mis. project, tahun).",
    structured_output=True,
)
def list_metadata_values_tool(field: str) -> Dict[str, Any]:
    """
    Args:
        field: nama metadata key, misal 'project' atau 'tahun'.
    Returns:
        values: list nilai unik.
    """
    values = rag_tools.list_metadata_values(field)
    return {"values": values}


# ---------------------------------------------------------------------------
# 11. Retrieve Product Context via RAG + Prompt Template
# ---------------------------------------------------------------------------
@mcp.tool(
    name="retrieve_product_context",
    title="Retrieve Product Context",
    description=(
        "Ambil konteks produk untuk proposal melalui RAG similarity search, "
        "dengan opsi filter metadata dan prompt template."
    ),
    structured_output=True,
)
def retrieve_product_context_tool(
    product: str,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper untuk DocGeneratorTools.retrieve_product_context.

    Args:
        product: Nama atau deskripsi produk yang dicari.
        k: Jumlah top-k chunks yang diambil (default pipeline).
        metadata_filter: Filter metadata seperti {"project":"X","tahun":"2025"}.
        prompt_template: Nama file template prompt tanpa ekstensi (.txt).

    Returns:
        Dict dengan keys:
          - status: 'success' atau 'failure'
          - context: List teks chunk hasil retrieval
          - instruction: Isi prompt template (jika ada)
          - error: Pesan error (jika failure)
    """
    return doc_tools.retrieve_product_context(
        product, k, metadata_filter, prompt_template
    )


# ---------------------------------------------------------------------------
# 12. Extract Text from Document → Markdown
# ---------------------------------------------------------------------------
@mcp.tool(
    name="extract_document_text",
    title="Extract Document Text",
    description=(
        "Ekstrak teks dari file dokumen (.pdf, .docx, .md) "
        "dan kembalikan dalam format markdown untuk reasoning LLM."
    ),
    structured_output=True,
)
def extract_document_text_tool(
    file_path: str,
) -> Dict[str, Any]:
    """
    Wrapper untuk DocGeneratorTools.extract_document_text.

    Args:
        file_path: Path lokal ke file .pdf, .docx, atau .md.

    Returns:
        Dict dengan keys:
          - status: 'success' atau 'failure'
          - text: Markdown hasil ekstraksi (jika success)
          - error: Pesan error (jika failure)
    """
    return doc_tools.extract_document_text(file_path)


# ---------------------------------------------------------------------------
# 13. Generate Proposal Document (.docx)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="generate_proposal_docx",
    title="Generate Proposal Document",
    description=(
        "Render dan simpan dokumen proposal Word (.docx) berdasarkan context "
        "data dan (opsional) override template."
    ),
    structured_output=True,
)
def generate_proposal_docx_tool(
    context: Dict[str, Any],
    override_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper untuk DocGeneratorTools.generate_proposal.

    Args:
        context: Dictionary variabel template (judul, nama_pelanggan, tabel, list, dll).
        override_template: Path ke file .docx template baru (opsional).

    Returns:
        Dict dengan keys:
          - status: 'success' atau 'failure'
          - product: Identifier proposal (mis. judul_proposal)
          - path: Path file .docx yang dihasilkan (jika success)
          - error: Pesan error (jika failure)
    """
    return doc_tools.generate_proposal(context, override_template)


# run the server
if __name__ == "__main__":
    mcp.run(transport="sse")
