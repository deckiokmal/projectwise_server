# mcp_server/server.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp_server.settings import Settings

from mcp_server.tools.kak_analyzer_tool import KAKTools
from mcp_server.tools.product_sizing_tool import ProductTools
from mcp_server.tools.retrieval_tools import retrieval
from mcp_server.tools.docgen_product_tools import DocGenProductTools
from tavily import TavilyClient as WEBSearch


load_dotenv()
settings = Settings()


# Check if Tavily API key is set
if "TAVILY_API_KEY" not in os.environ:
    raise Exception("TAVILY_API_KEY environment variable is not set")

# Tavily API key
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]


# Create an MCP Server
mcp = FastMCP(
    name="projectwise-mcp",
    stateless_http=True,
)

# Initialize tools
kak = KAKTools()
prod = ProductTools()
doc = DocGenProductTools()
tavily_client = WEBSearch(TAVILY_API_KEY)


# ====================================================================
# Hearbeat Message
# ====================================================================
@mcp.tool(name="heartbeat")
def heartbeat_tool():
    return True


# ====================================================================
# KAK Analyer
# ====================================================================
@mcp.tool(name="re_analyze_kak")
async def re_analyze_kak_tool(
    filename: str,
    pelanggan: str,
    project: str,
    tahun: str,
    prompt_instruction: Optional[str] = None,
):
    """
    Analysis ulang kak/tor proyek, hasil analysis proyek kak/tor.
    Jika file belum pernah di ingest ke dalam database vector (retrieval).
    lihat informasi --> filename, pelanggan, project, tahun: menggunakan retrieval.
    """
    result = await kak.generate_kak_summarize(
        filename, pelanggan, project, tahun, prompt_instruction
    )
    return result


@mcp.tool(name="read_kak_analysis")
async def read_kak_analysis_tool(
    filename: str, pelanggan: str, project: str, tahun: str
):
    """
    Baca kak summary analysis, untuk mengetahui lebih banyak tentang suatu proyek/kak.
    gunakan tool ini setelah melakukan retrieval agar mendapatkan citation terkait proyek/kak.
    """
    result = await kak.read_kak_summaries(filename, pelanggan, project, tahun)
    return result


# ====================================================================
# Product Sizing
# ====================================================================
@mcp.tool(name="re_generate_product_sizing")
async def re_generate_product_sizing_tool(
    category: str,
    product: str,
    tahun: str,
    filename: str,
    prompt_instruction: Optional[str] = None,
):
    """
    Generate ulang ringkasan context sizing product, hasil analysis sizing product.
    Jika file belum pernah di ingest ke dalam database vector (retrieval).
    Hasil dari tools ini dapat digunakan sebagai acuan menghitung harga layanan/product tertentu.
    """
    result = await prod.generate_product_summarize(
        category, product, tahun, filename, prompt_instruction
    )
    return result


@mcp.tool(name="read_product_sizing")
async def read_product_sizing_tool(
    filename: str, pelanggan: str, project: str, tahun: str
):
    """
    Baca product sizing, untuk mengetahui cara menghitung harga suatu product.
    gunakan tool ini setelah melakukan retrieval agar mendapatkan citation terkait product.
    """
    result = await kak.read_kak_summaries(filename, pelanggan, project, tahun)
    return result


# ====================================================================
# Retrieval (RAG)
# ====================================================================
@mcp.tool(name="retrieval")
async def retrieval_tool(
    query: str, k: int | None = 10, metadata_filter: Dict[str, Any] | None = None
):
    """
    retrieval berisi data proyek/kak dan product. ini adalah sumber informasi utama.
    untuk mendapatkan informasi selengkapnya tentang context suatu proyek/kak dan product.
    gunakanlah retrieval dengan optional argument: metadata_filter: Dict[str, Any].
    metadata context proyek/kak: filename, project, pelanggan, tahun
    metadata context product: filename, category, product, tahun
    UTAMAKAN QUERY TANPA METADATA.
    """
    result = await retrieval(query, k, metadata_filter)
    return result


# ====================================================================
# Document Generation Tools
# ====================================================================
@mcp.tool(name="product_template_placeholder")
def product_template_placeholders_tool(
    product_category: str, template_name: str | None = None
) -> dict:
    """
    Tools ini berguna untuk pembuatan proposal teknis atau penawaran product.
    Lihat placeholder untuk template kategori tertentu.
    product_category: internet | project-nonmigas
    """
    result = doc.get_template_placeholders(
        product_category=product_category, template_name=template_name
    )
    return result


@mcp.tool(name="product_proposal_generation")
def product_generate_proposal_tool(payload: dict) -> dict:
    """Generate proposal DOCX dengan opsi integrasi product summary.
    Contoh payload minimal:
    {
      "product_category": "internet",
      "context": {"judul_proposal": "...", "nama_pelanggan": "..."},
      "product": "internet",
      "category": "datacom",
      "tahun": "2025",
      "summary_filename": "internet_dedicated"
    }
    """
    result = doc.generate_proposal(**payload)
    return result


# ──────────────────────────────────────────────────────────────
# Websearch capability using Tavily API (free 1000 Credits/month)
# ──────────────────────────────────────────────────────────────
@mcp.tool(name="websearch")
def websearch_tool(query: str, max_results: int = 5) -> List[Dict]:
    """
    Pencarian informasi external dari web/internet.
    gunakan tool ini jika informasi yang dicari tidak ditemukan di retrieval.
    atau jika user meminta pencarian dari internet.
    """
    try:
        response = tavily_client.search(query, max_results=max_results)
        return response["results"]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]


# ──────────────────────────────────────────────────────────────
# Elicitation Human in the loop
# ──────────────────────────────────────────────────────────────
# TODO: next step prepare buat mcp elicitation. Human in a loop!
@mcp.tool(name="elicitation_test")
async def long_running_task_tool(
    task_name: str, ctx: Context[ServerSession, None], steps: int = 5
) -> str:
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
