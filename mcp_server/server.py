from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os

from mcp_server.tools.rag_tools import RAGTools
from mcp_server.tools.docgen_product_tools import DocGeneratorTools
from mcp_server.settings import Settings
from dotenv import load_dotenv


load_dotenv()
settings = Settings()  # type: ignore


# Check if Tavily API key is set
if "TAVILY_API_KEY" not in os.environ:
    raise Exception("TAVILY_API_KEY environment variable is not set")

# Tavily API key
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]


# Create an MCP Server
mcp = FastMCP(
    name="projectwise",
    stateless_http=True,
)

# Initialize tools
tavily_client = TavilyClient(TAVILY_API_KEY)
rag_tools = RAGTools()
doc_tools = DocGeneratorTools()


# ──────────────────────────────────────────────────────────────
# Heartbeat message tool
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="heartbeat",
    title="Heartbeat message",
    description="Heartbeat message digunakan oleh client untuk keep alive koneksi",
    structured_output=True,
)
def heartbeat_tool() -> str:
    return "ok"


# ──────────────────────────────────────────────────────────────
# Utility untuk listing files dalam KAK/TOR dan Product dir
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="list_kak_files",
    title="Daftar File KAK/TOR",
    description="Menampilkan semua file KAK/TOR yang tersedia berdasarkan folder metadata pelanggan/tahun.",
    structured_output=True,
)
def list_kak_files_tool() -> Dict[str, Any]:
    base = Path(settings.kak_tor_base_path).expanduser().resolve()
    files = [str(p.relative_to(base)) for p in base.rglob("*.pdf")]
    return {"files": files}


@mcp.tool(
    name="list_product_files",
    title="Daftar File Produk",
    description="Menampilkan semua file produk PDF yang tersedia berdasarkan folder metadata kategori/tahun/produk.",
    structured_output=True,
)
def list_product_files_tool() -> Dict[str, Any]:
    base = Path(settings.product_base_path).expanduser().resolve()
    files = [str(p.relative_to(base)) for p in base.rglob("*.pdf")]
    return {"files": files}


# ──────────────────────────────────────────────────────────────
# Definisi tools untuk AI Tender Analyzer dengan RAGPipeline
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="add_product_knowledge",
    title="Ingest Produk ke Knowledge Base",
    description=(
        "Tambahkan PDF produk ke knowledge base untuk kebutuhan proposal. "
        "Simpan markdown hasilnya dan ringkasan Sizing dengan metadata `product`, `category`, `tahun`."
    ),
    structured_output=True,
)
async def add_product_knowledge_tool(
    category: str,
    product_name: str,
    tahun: str,
    filename: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return await rag_tools.ingest_product_knowledge_chunks(
        filename, product_name, category, tahun, overwrite=overwrite
    )


@mcp.tool(
    name="add_kak_tor_knowledge",
    title="Ingest KAK/TOR PDF ke Knowledge Base",
    description=(
        "Tambahkan dokumen KAK/TOR ke vectorstore dan ekspor ke markdown. "
        "Metadata mencakup nama pelanggan, proyek, dan tahun."
    ),
    structured_output=True,
)
async def add_kak_tor_knowledge_tool(
    filename: str,
    pelanggan: str,
    project: str,
    tahun: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return await rag_tools.ingest_kak_tor_chunks(
        filename, pelanggan, project, tahun, overwrite=overwrite
    )


@mcp.tool(
    name="summarize_kak_with_llm",
    title="Ringkasan KAK/TOR via LLM",
    description="Gabungkan prompt + markdown proyek dan hasilkan ringkasan melalui LLM lalu index ke vectorstore.",
    structured_output=True,
)
async def summarize_kak_with_llm_tool(
    kak_md_name: str, pelanggan: str, project: str, tahun: str
) -> Dict[str, Any]:
    if not kak_md_name.lower().endswith(".md"):
        kak_md_name += ".md"

    result = await rag_tools.build_summary_kak_payload_and_summarize(
        kak_md_name, pelanggan, project, tahun
    )
    return {"result": result}


# ──────────────────────────────────────────────────────────────
# Utility untuk Retrieval Augmented Generation
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="rag_retrieval",
    title="RAG Retrieval",
    description=(
        "1️. Metadata optional: Periksa apakah query pengguna menyertakan filter metadata "
        "(project, pelanggan, tahun). Jika tidak ada, lanjutkan retrieval tanpa filter.  \n"
        "2️. Jika metadata disertakan, pastikan nilainya spesifik (salah satu dari opsi yang tersedia).  \n"
        "3️. Panggil retrieval dengan (query, k, metadata_filter).  \n"
        "4️. Jika hasil kosong, backend akan mengembalikan daftar metadata yang valid.  \n"
        "   – Setelah menerima daftar ini, LLM harus memilih nilai yang benar dan memanggil "
        "`rag_retrieval` ulang dengan filter yang diperbaiki.  \n"
        "5️. Kembalikan array JSON berisi objek `{ text, metadata, citation, score, query_time }`."
    ),
    structured_output=True,
)
async def rag_retrieval_tool(
    query: str,
    k: Optional[int] = 10,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    MCP Tool untuk retrieval konteks (RAG). Metadata bersifat optional namun harus valid jika diberikan.
    """
    result = await rag_tools.pipeline.retrieval(
        query=query,
        k=k,
        metadata_filter=metadata_filter,
    )
    return result


@mcp.tool(
    name="list_metadata_entries",
    title="Daftar Metadata Dokumen",
    description="Menampilkan seluruh metadata dokumen (project, pelanggan, tahun, dll) yang telah masuk ke vectorstore.",
    structured_output=True,
)
async def list_metadata_entries_tool(limit: int = 20) -> List[Dict[str, Any]]:
    return await rag_tools.pipeline.list_available_metadata(limit)


@mcp.tool(
    name="reset_vector_database",
    title="Reset Ulang Vector Database",
    description=(
        "Menghapus semua data vector store, membersihkan file manifest/status, "
        "dan menginisialisasi ulang LanceDB agar siap digunakan kembali."
    ),
    structured_output=True,
)
async def reset_vector_database_tool() -> Dict[str, Any]:
    result = await rag_tools.pipeline.reset_vector_database()

    # Tambahan informasi jika berhasil
    if result.get("status") == "success":
        metadata = await rag_tools.pipeline.list_available_metadata()
        return {
            **result,
            "total_metadata_after_reset": len(metadata),
            "example_entry": metadata[0] if metadata else None,
        }

    return result


@mcp.tool(
    name="reset_and_reingest_all",
    title="Reset & Re-Ingest Semua Dokumen",
    description=(
        "Reset vectorstore dan secara otomatis melakukan ingest ulang semua file PDF dari folder "
        "kak_tor dan product_knowledge. Gunakan jika ingin memulai ulang proses index dokumen dari nol."
    ),
    structured_output=True,
)
async def reset_and_reingest_all_tool() -> Dict[str, Any]:
    base_kak_path = Path(settings.kak_tor_base_path).expanduser().resolve()
    base_prod_path = Path(settings.product_base_path).expanduser().resolve()

    # 1. Reset vector DB
    reset_result = await rag_tools.pipeline.reset_vector_database()

    if reset_result.get("status") != "success":
        return {
            "status": "error",
            "message": f"Gagal reset database: {reset_result.get('message')}",
        }

    log = {"reset": reset_result, "ingested": {"kak": [], "product": []}}

    # 2. Re-ingest semua KAK/TOR PDF
    for pdf_path in base_kak_path.rglob("*.pdf"):
        relative = pdf_path.relative_to(base_kak_path)
        parts = relative.parts
        if len(parts) < 3:
            continue  # skip invalid paths

        pelanggan = parts[0].replace("_", " ")
        tahun = parts[1]
        filename = parts[-1]
        project = Path(filename).stem.replace("_", " ")

        result = await rag_tools.ingest_kak_tor_chunks(
            filename=filename,
            pelanggan=pelanggan,
            project=project,
            tahun=tahun,
            overwrite=True,
        )
        log["ingested"]["kak"].append({"filename": str(relative), "result": result})

    # 3. Re-ingest semua Produk PDF
    for pdf_path in base_prod_path.rglob("*.pdf"):
        relative = pdf_path.relative_to(base_prod_path)
        parts = relative.parts
        if len(parts) < 3:
            continue

        category = parts[0].replace("_", " ")
        tahun = parts[1]
        product_name = parts[2].replace("_", " ")
        filename = pdf_path.name

        result = await rag_tools.ingest_product_knowledge_chunks(
            filename=str(relative),
            product_name=product_name,
            category=category,
            tahun=tahun,
            overwrite=True,
        )
        log["ingested"]["product"].append({"filename": str(relative), "result": result})

    return {
        "status": "success",
        "summary": {
            "reset_message": reset_result["message"],
            "kak_files": len(log["ingested"]["kak"]),
            "product_files": len(log["ingested"]["product"]),
        },
        "details": log,
    }


# ──────────────────────────────────────────────────────────────
# Definisi tools untuk Document Generation
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="read_project_markdown",
    title="Baca Markdown Proyek",
    description="Baca isi markdown KAK/TOR sebagai context untuk proposal pipeline.",
    structured_output=True,
)
def read_project_markdown_tool(filename: str) -> Dict[str, Any]:
    return doc_tools.read_project_markdown(filename)


@mcp.tool(
    name="get_template_placeholders",
    title="Ambil Placeholder Template Proposal",
    description="Kembalikan array dari seluruh placeholder yang harus diisi dalam template .docx proposal.",
    structured_output=True,
)
def get_template_placeholders_tool() -> Dict[str, Any]:
    return doc_tools.get_template_placeholders()


@mcp.tool(
    name="generate_proposal_docx",
    title="Generate Proposal Word Document",
    description="Generate proposal .docx dari template dan context JSON berisi field proposal.",
    structured_output=True,
)
def generate_proposal_docx_tool(
    context: Dict[str, Any],
    override_template: Optional[str] = None,
) -> Dict[str, Any]:
    return doc_tools.generate_proposal(context, override_template)


# ──────────────────────────────────────────────────────────────
# Websearch capability using Tavily API (free 1000 Credits/month)
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="websearch",
    title="Web Search",
    description="Search the web using Tavily API.",
    structured_output=True,
)
def websearch_tool(query: str) -> List[Dict]:
    try:
        response = tavily_client.search(query, max_results=10)
        return response["results"]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]


# ──────────────────────────────────────────────────────────────
# Elicitation capabilities tools
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Run FastMCP server
# ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     mcp.run(transport="sse")
