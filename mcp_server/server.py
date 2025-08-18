from __future__ import annotations
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
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
# Definisi tools untuk AI Tender Analyzer dengan RAGPipeline
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="heartbeat",
    title="Heartbeat message for keeping session with client.",
    description="Session alive tool.",
    structured_output=False,
)
def heartbeat():
    return "connection alive"


# ──────────────────────────────────────────────────────────────
# Definisi tools untuk AI Tender Analyzer dengan RAGPipeline
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="ingest_product_knowledge",
    title="Ingest Product PDF into vector database for Product Knowledge Base",
    description=(
        "Use this tool to ingest a product PDF into vector database for product knowledge base only if user explisit ask for it. "
        "Requires: category, product_name, year, filename. "
        "Example: category='Internet Services', product_name='Internet_Dedicated', "
        "tahun='2025', filename='Internet_Dedicated.pdf'. "
    ),
    structured_output=True,
)
async def ingest_product_knowledge_tool(
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
    name="ingest_kak_tor_knowledge",
    title="Ingest KAK/TOR PDF into vector database for Project Knowledge Base",
    description=(
        "Use this tool to ingest a KAK/TOR PDF into vector database for project knowledge base only if user explisit ask for it. "
        "Requires: filename, pelanggan (customer name), project (project name), and tahun. "
        "Example: filename='ProjectX.pdf', pelanggan='CustomerA', project='ProjectX', tahun='2025'. "
    ),
    structured_output=True,
)
async def ingest_kak_tor_knowledge_tool(
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
    name="project_summaries_analysis",
    title="Analyze Project KAK/TOR",
    description=(
        "Use this tool to generate an analysis of the project KAK/TOR. "
        "Requires: project (project name), pelanggan, tahun. "
        "Example: project='ProjectX', pelanggan='CustomerA', tahun='2025'."
    ),
    structured_output=True,
)
async def project_summaries_analysis_tool(
    project: str, pelanggan: str, tahun: str
) -> Dict[str, Any]:
    result = await rag_tools.read_kak_summaries(
        project, pelanggan, tahun
    )
    return result


# ──────────────────────────────────────────────────────────────
# Utility untuk Retrieval Augmented Generation
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="project_product_retrieval_informations",
    title="Retrieve Relevant Context from vector database for Project and Product Knowledge Base",
    description=(
        "Retrieve relevant context from vector database for project and product knowledge base. "
        "Requires: query, k (number of results), metadata_filter (optional). "
        "Example input: query='technical specs for Dedicated 100 Mbps', k=5, metadata_filter={'project':'ProjectX', 'pelanggan':'CustomerA', 'tahun':'2025'}"
        "For product metadata input:"
        "metadata_filter={'category':'Internet Services', 'product_name':'Internet_Dedicated', 'tahun':'2025'}"
    ),
    structured_output=True,
)
async def project_product_retrieval_tool(
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


# ──────────────────────────────────────────────────────────────
# Definisi tools untuk Document Generation
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="list_all_metadata_rag",
    title="List All Available Document Metadata in vector database",
    description=(
        "List available metadata entries from the vectorstore, "
        "such as project, product, customer, and year. "
        "Use this before retrieval if you need to know possible metadata values."
    ),
    structured_output=True,
)
async def list_metadata_entries_tool(limit: int = 20) -> List[Dict[str, Any]]:
    return await rag_tools.pipeline.list_available_metadata(limit)


# ──────────────────────────────────────────────────────────────
# Definisi tools untuk Document Generation
# ──────────────────────────────────────────────────────────────
@mcp.tool(
    name="project_context_for_proposal",
    title="Read Project Content for anlysis and Proposal Generation",
    description="Read a project content file as context for analysis and proposal generation.",
    structured_output=True,
)
def project_context_for_proposal_tool(filename: str) -> Dict[str, Any]:
    return doc_tools.read_project_markdown(filename)


@mcp.tool(
    name="get_template_placeholders",
    title="Get Proposal Template Placeholders",
    description="Return all placeholders in the .docx proposal template that need to be filled for Proposal Generation."
    "Use this tools before calling 'generate_proposal_docx' tool and after 'project_context_for_proposal' tool.",
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
    title="Web Search Tool",
    description="Search the web for information about a query only if the information did not provider in retrieval or user explisit ask for it.",
    structured_output=True,
)
def websearch_tool(query: str) -> List[Dict]:
    try:
        response = tavily_client.search(query, max_results=10)
        return response["results"]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]


# ──────────────────────────────────────────────────────────────
# Websearch capability using Tavily API (free 1000 Credits/month)
# ──────────────────────────────────────────────────────────────
# TODO: next step prepare buat mcp elicitation. Human in a loop!
@mcp.tool()
async def long_running_task(task_name: str, ctx: Context[ServerSession, None], steps: int = 5) -> str:
    """Execute a task with progress updates."""
    await ctx.info(f"Starting: {task_name}")

    for i in range(steps):
        progress = (i + 1) / steps
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"Step {i + 1}/{steps}",
        )
        await ctx.debug(f"Completed step {i + 1}")

    return f"Task '{task_name}' completed"