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
    title="Keep MCP Connection Alive",
    description="Client uses this tool to keep the MCP server connection alive. "
    "No arguments needed. Returns 'ok' if server is alive.",
    structured_output=True,
)
def heartbeat_tool() -> str:
    return "ok"


# ──────────────────────────────────────────────────────────────
# Utility untuk listing files dalam KAK/TOR dan Product dir
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="list_kak_tor_files",
    title="List KAK/TOR PDF Files for Project Context",
    description=(
        "Use this tool when you need to find available KAK/TOR Project PDF files before adding them to the knowledge base. "
        "Returns relative file paths from the KAK/TOR base folder. "
        "Typical usage: Step 1 before calling 'add_kak_tor_knowledge'. "
        "Example output: { 'files': ['CustomerA/2025/ProjectX.pdf'] }"
    ),
    structured_output=True,
)
def list_kak_files_tool() -> Dict[str, Any]:
    base = Path(settings.kak_tor_base_path).expanduser().resolve()
    files = [str(p.relative_to(base)) for p in base.rglob("*.pdf")]
    return {"files": files}


@mcp.tool(
    name="list_product_files_for_proposal",
    title="List Product PDF Files for Proposal Preparation",
    description=(
        "Use this tool to see available standard product-related PDF files in the knowledge base folder "
        "before adding product knowledge. "
        "Typical usage: Step 1 before calling 'add_product_knowledge'. "
        "Example output: { 'files': ['Internet/2025/Dedicated_100Mbps.pdf'] }"
    ),
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
    title="Add Product PDF to Proposal Knowledge Base",
    description=(
        "Use this tool to ingest a product PDF into the proposal knowledge base only if user explisit ask for it. "
        "Requires: category, product_name, year, filename (from list_product_files_for_proposal). "
        "Example: category='Internet Services', product_name='Internet_Dedicated', "
        "tahun='2025', filename='Internet_Dedicated.pdf'. "
        "Run this step after listing available product files."
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
    title="Add KAK/TOR PDF to Project Knowledge Base",
    description=(
        "Use this tool to ingest a KAK/TOR PDF into the knowledge base only if user explisit ask for it. "
        "Requires: filename (from list_kak_tor_files), pelanggan (customer name), project (project name), and tahun. "
        "Example: filename='CustomerA/2025/ProjectX.pdf', pelanggan='CustomerA', project='ProjectX', tahun='2025'. "
        "Run this step after listing KAK/TOR files."
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
    name="summarize_kak_tor_with_llm",
    title="Summarize KAK/TOR Markdown with LLM",
    description=(
        "Use this tool to generate a summary of the project KAK/TOR. "
        "Requires: kak_md_name (markdown filename), pelanggan, project, tahun. "
        "Example: kak_md_name='CustomerA_ProjectX.md', pelanggan='CustomerA', project='ProjectX', tahun='2025'."
    ),
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
    title="Retrieve Relevant Context from Knowledge Base",
    description=(
        "Retrieve text and metadata from the vectorstore relevant to a query. "
        "Knowledge base for project information and standard product information. "
        "Metadata filter is optional. "
        "If metadata is missing, run `list_all_metadata_entries` to see available metadata instead of guessing. "
        "Example input: query='technical specs for Dedicated 100 Mbps', k=5, metadata_filter={'project':'ProjectX'}"
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
    name="list_all_metadata_entries",
    title="List All Available Document Metadata",
    description=(
        "List available metadata entries from the vectorstore, "
        "such as project, product, customer, and year. "
        "Use this before retrieval if you need to know possible metadata values."
    ),
    structured_output=True,
)
async def list_metadata_entries_tool(limit: int = 20) -> List[Dict[str, Any]]:
    return await rag_tools.pipeline.list_available_metadata(limit)


@mcp.tool(
    name="reset_vector_database",
    title="Reset Vectorstore",
    description="Reset the entire vectorstore database. This will delete all stored documents and metadata.",
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
    name="reset_and_reingest_all_documents",
    title="Reset and Re-ingest All Documents",
    description="Reset the vectorstore and re-ingest all PDF files from KAK/TOR and product knowledge folders.",
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
    title="Read Project Markdown Content",
    description="Read a project markdown file as context for proposal generation.",
    structured_output=True,
)
def read_project_markdown_tool(filename: str) -> Dict[str, Any]:
    return doc_tools.read_project_markdown(filename)


@mcp.tool(
    name="get_template_placeholders",
    title="Get Proposal Template Placeholders",
    description="Return all placeholders in the .docx proposal template that need to be filled.",
    structured_output=True,
)
def get_template_placeholders_tool() -> Dict[str, Any]:
    return doc_tools.get_template_placeholders()


@mcp.tool(
    name="generate_proposal_docx",
    title="Generate Proposal Word Document",
    description="Generate a proposal .docx file from a template and provided JSON context.\n"
    "Run this tool only after provide template placeholder as a context.",
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
    description="Search the web for information about a query only if the information did not provider in retrieval.",
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
