# mcp_server/server.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any, Union, Iterable
from pathlib import Path

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
@mcp.tool()
def heartbeat_tool():
    """Cek kesehatan server MCP.

    Kapan digunakan:
    - Untuk memastikan server dan koneksi siap sebelum memanggil tool lain.

    Input:
    - (tidak ada)

    Output:
    - bool: True jika sehat.
    """
    return True


# ====================================================================
# KAK Analyer
# ====================================================================
@mcp.tool()
async def re_analyze_kak_tool(
    filename: str,
    pelanggan: str,
    project: str,
    tahun: str,
    prompt_instruction: Optional[str] = None,
):
    """Bangun ulang ringkasan/analisis KAK/TOR proyek.

    Kapan digunakan:
    - Dokumen/summary belum tersedia atau ingin di-refresh dengan instruksi baru.

    Input:
    - filename (str): Nama dokumen KAK/TOR.
    - pelanggan (str): Nama instansi/klien.
    - project (str): Nama proyek.
    - tahun (str, YYYY): Tahun proyek.
    - prompt_instruction (str, opsional): Arah analisis tambahan (gaya/penekanan).

    Output:
    - dict: Hasil ringkasan/analisis (format ditentukan modul KAKTools).
    """
    result = await kak.generate_kak_summarize(
        filename, pelanggan, project, tahun, prompt_instruction
    )
    return result


@mcp.tool()
async def read_kak_analysis_tool(
    filename: str, pelanggan: str, project: str, tahun: str
):
    """Ambil ringkasan analisis KAK/TOR yang sudah ada (beserta citation bila ada).

    Kapan digunakan:
    - Ingin membaca ulang ringkasan tanpa melakukan re-analisis.

    Input:
    - filename (str), pelanggan (str), project (str), tahun (str, YYYY)

    Output:
    - dict: Ringkasan analisis yang tersimpan.
    """
    result = await kak.read_kak_summaries(filename, pelanggan, project, tahun)
    return result


@mcp.tool()
def list_kak_file_tool(
    folder: Union[str, Path] = settings.kak_tor_summaries_base_path,
    pattern: str = "*",  # filter nama: "*.pdf", "*.py", dsb.
    recursive: bool = True,  # True untuk scan subfolder
    absolute: bool = True,  # True -> path absolut
    follow_symlinks: bool = False,  # ikut link simbolik?
) -> List[Path]:
    """
    Kembalikan daftar path file kak dalam folder.
    Hanya file (bukan folder). Ditangani aman untuk permission/symlink.

    Raises:
        FileNotFoundError: jika folder tidak ada
        NotADirectoryError: jika path bukan folder
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Bukan folder: {p}")

    it: Iterable[Path] = p.rglob(pattern) if recursive else p.glob(pattern)
    out: List[Path] = []
    for path in it:
        try:
            if not follow_symlinks and path.is_symlink():
                continue
            if path.is_file():
                out.append(path.resolve(strict=False) if absolute else path)
        except PermissionError:
            # skip item yang tak bisa diakses
            continue
    return out


# ====================================================================
# Product Sizing
# ====================================================================
@mcp.tool()
async def re_generate_product_sizing_tool(
    category: str,
    product: str,
    tahun: str,
    filename: str,
    prompt_instruction: Optional[str] = None,
):
    """Bangun ulang ringkasan sizing produk (acuan perhitungan biaya/kapasitas).

    Kapan digunakan:
    - Ringkasan sizing belum ada atau perlu diperbarui.

    Input:
    - category (str), product (str), tahun (str, YYYY), filename (str)
    - prompt_instruction (str, opsional): Arah/penekanan khusus.

    Output:
    - dict: Ringkasan sizing (format ditentukan ProductTools).
    """
    result = await prod.generate_product_summarize(
        category, product, tahun, filename, prompt_instruction
    )
    return result


@mcp.tool()
async def read_product_sizing_tool(
    filename: str, category: str, product: str, tahun: str
):
    """Ambil ringkasan sizing produk yang sudah ada.

    Kapan digunakan:
    - Ingin membaca ulang ringkasan sizing tanpa regenerasi.

    Input:
    - filename (str), pelanggan (str), project (str), tahun (str, YYYY)

    Output:
    - dict: Ringkasan sizing produk.
    """
    result = await prod.read_product_summaries(filename, category, product, tahun)
    return result


# ====================================================================
# Retrieval (RAG)
# ====================================================================
@mcp.tool()
async def retrieval_tool(
    query: str, k: int | None = 10, metadata_filter: Dict[str, Any] | None = None
):
    """Retrieval pengetahuan proyek/KAK & produk dari basis vektor.

    Kapan digunakan:
    - Membutuhkan konteks faktual dari dokumen internal (proyek/KAK/produk).

    Input:
    - query (str): Pertanyaan/kata kunci natural language.
    - k (int, opsional): Jumlah item teratas (default 10).
    - metadata_filter (dict, opsional): Penyaring metadata, mis. {filename, project, pelanggan, tahun, category, product}.

    Output:
    - dict/list: Hasil retrieval (dokumen/segmen beserta metadata & skor).
    """
    result = await retrieval(query, k, metadata_filter)
    return result


# ====================================================================
# Document Generation Tools
# ====================================================================
@mcp.tool()
def product_template_placeholders_tool(
    product_category: str, template_name: str | None = None
) -> dict:
    """Lihat placeholder template dokumen penawaran/proposal.

    Kapan digunakan:
    - Mengetahui variabel yang harus diisi pada template tertentu.

    Input:
    - product_category (str): Mis. "internet" atau "project-nonmigas".
    - template_name (str, opsional): Nama template spesifik (jika ada).

    Output:
    - dict: Daftar placeholder/variabel dan keterangan singkatnya.
    """
    result = doc.get_template_placeholders(
        product_category=product_category, template_name=template_name
    )
    return result


@mcp.tool()
def product_generate_proposal_tool(payload: dict) -> dict:
    """Hasilkan file proposal DOCX dengan opsi integrasi ringkasan produk.

    Kapan digunakan:
    - Membuat proposal teknis/komersial berdasarkan template & ringkasan produk.

    Input (payload: dict minimal):
    - product_category (str), context (dict), product (str), category (str), tahun (str), summary_filename (str, opsional)

    Output:
    - dict: Informasi hasil pembuatan (mis. path/file output, metadata).
    """
    result = doc.generate_proposal(**payload)
    return result


# ──────────────────────────────────────────────────────────────
# Websearch capability using Tavily API (free 1000 Credits/month)
# ──────────────────────────────────────────────────────────────
@mcp.tool()
def websearch_tool(query: str, max_results: int = 5) -> List[Dict]:
    """Pencarian informasi eksternal (web) via Tavily.

    Kapan digunakan:
    - Informasi tidak ada di retrieval/memori, atau pengguna minta sumber web.

    Input:
    - query (str): Kalimat pencarian.
    - max_results (int, opsional): Batas hasil (default 5).

    Output:
    - list[dict]: Daftar hasil (title/url/summary/metadata) atau {"error": "..."} saat gagal.
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
