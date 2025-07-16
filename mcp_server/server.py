from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json

from mcp.server.fastmcp import FastMCP  # type: ignore
from mcp_server.tools.rag_tools import RAGTools
from mcp_server.tools.docx_tools import DocGeneratorTools
from mcp_server.settings import Settings
from mcp_server.utils.helper import (
    _slugify,
    _to_markdown,
    _normalize_md_name,
    _list_files,
)


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
    name="ingest_product_knowledge",
    title="Ingest Product Knowledge",
    description=(
        "Tambahkan seluruh file PDF di direktori `product_standard` ke vectorstore RAG "
        "sebagai product knowledge dengan metadata `project` dan `tahun`."
    ),
)
def ingest_product_knowledge_tool(
    base_dir: Optional[str] = None,
    project_name: str = "product_standard",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """Mengindeks dokumen-dokumen product knowledge ke dalam vector store.

    Fungsi ini memindai direktori yang ditentukan untuk mencari dokumen-dokumen
    standar produk (misalnya, PDF), memprosesnya, dan menambahkannya ke
    vector store RAG. Setiap potongan dokumen (chunk) akan ditandai dengan
    nama proyek dan tahun yang diberikan untuk mempermudah pencarian di kemudian hari.

    Args:
        base_dir (Optional[str], optional): Path ke direktori yang berisi
            dokumen-dokumen product knowledge. Jika tidak disediakan, akan
            menggunakan path default dari `settings.knowledge_base_path`.
            Defaultnya adalah None.
        project_name (str, optional): Label metadata untuk proyek. Ini membantu
            dalam memfilter basis pengetahuan saat pencarian.
            Defaultnya adalah "product_standard".
        tahun (str, optional): Label metadata untuk tahun.
            Defaultnya adalah "2025".

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi status dari proses
            ingestion. Contoh: {"status": "Product knowledge ingestion selesai."}
    """
    result = rag_tools.add_product_knowledge(base_dir, project_name, tahun)
    return {"status": "Product knowledge ingestion selesai.", "result": result}


# ---------------------------------------------------------------------------
# 2. Ingest KAK/TOR PDFs → Markdown → vectorstore
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ingest_kak_tor_knowledge",
    title="Ingest KAK/TOR Documents",
    description=(
        "Konversi seluruh PDF KAK/TOR di direktori kak_tor ke Markdown, simpan file, lalu indeks "
        "ke vectorstore RAG dengan metadata custom."
    ),
)
def ingest_kak_tor_knowledge_tool(
    project: Optional[str] = None,
    tahun: Optional[str] = None,
) -> Dict[str, Any]:
    """Mengonversi dan mengindeks dokumen KAK/TOR dari PDF ke vector store.

    Fungsi ini menjalankan alur kerja lengkap untuk dokumen Kerangka Acuan
    Kerja (KAK) atau Terms of Reference (TOR). Prosesnya meliputi:
    1. Memindai direktori `settings.kak_tor_base_path` untuk file PDF.
    2. Mengonversi setiap PDF menjadi format Markdown.
    3. Menyimpan file Markdown yang dihasilkan ke `settings.kak_tor_md_base_path`.
    4. Mengindeks konten Markdown ke dalam vector store RAG.

    Setiap dokumen yang diindeks akan diberi tag metadata dengan nama proyek dan
    tahun yang diberikan untuk memfasilitasi pencarian dan pemfilteran.

    Args:
        project (Optional[str], optional): Label metadata untuk nama proyek.
            Jika None, mungkin tidak akan ada label proyek spesifik. Defaultnya None.
        tahun (Optional[str], optional): Label metadata untuk tahun dokumen.
            Jika None, mungkin tidak akan ada label tahun spesifik. Defaultnya None.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang mengonfirmasi penyelesaian proses.
            Contoh: {"status": "KAK/TOR ingestion selesai."}
    """
    result = rag_tools.add_kak_tor_knowledge(project=project, tahun=tahun)
    return {"status": "KAK/TOR ingestion selesai.", "result": result}


# ---------------------------------------------------------------------------
# 3. Ingest KAK/TOR Summaries Markdown → vectorstore
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ingest_kak_tor_summaries_knowledge",
    title="Ingest summaries KAK/TOR Markdown",
    description=(
        "Indeks file atau direktori Markdown langsung ke dalam vector store RAG.\n"
        "Args:\n"
        "  markdown_name: str file dari save_summary_markdown_tool.\n"
        "  project: str nama proyek.\n"
        "  tahun: str tahun dokumen.\n"
    ),
)
def ingest_kak_tor_summaries_knowledge_tool(
    markdown_name: Optional[str] = None,
    project: str = "default",
    tahun: str = "2025",
) -> Dict[str, Any]:
    """Mengindeks file atau direktori Markdown langsung ke dalam vector store.

    Fungsi ini dirancang untuk mengindeks konten dari file-file Markdown
    ringkasan (summary) dari dokumen KAK/TOR ke dalam vector store RAG.
    Ini berguna ketika dokumen sudah dalam format Markdown dan tidak
    memerlukan konversi dari PDF.

    Setiap dokumen yang diindeks akan ditandai dengan metadata proyek dan tahun
    untuk memfasilitasi pemfilteran dan pencarian yang terarah di masa depan.

    Args:
        markdown_name (Optional[str], optional): Nama file Markdown atau
            direktori yang berisi file-file Markdown. Jika tidak disediakan,
            fungsi akan memproses direktori default yang dikonfigurasi di
            `settings.summaries_md_base_path`. Defaultnya adalah None.
        project (str, optional): Label metadata untuk proyek. Berguna untuk
            memfilter basis pengetahuan. Defaultnya adalah "default".
        tahun (str, optional): Label metadata untuk tahun dokumen.
            Defaultnya adalah "2025".

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi hasil dari proses
            ingestion, seperti jumlah file yang berhasil diindeks.
    """
    result = rag_tools.add_kak_tor_summaries_knowledge(markdown_name, project, tahun)
    return {"status": "Ingest Markdown Summaries selesai.", "result": result}


# ---------------------------------------------------------------------------
# 4. List KAK/TOR Files di directory data/kak_tor_md
# ---------------------------------------------------------------------------
@mcp.tool(
    name="list_kak_files",
    title="List KAK/TOR Files",
    description=(
        "Tampilkan seluruh list nama files KAK/TOR di direktori `kak_tor_md_base_path`"
    ),
)
def list_kak_files() -> list[str]:
    files = _list_files(settings.kak_tor_md_base_path)
    return files


# ---------------------------------------------------------------------------
# 5. Build Summary Tender Payload
# ---------------------------------------------------------------------------
@mcp.tool(
    name="build_summary_tender_payload",
    title="Build Summary Tender Payload",
    description=(
        "Gabungkan prompt instruction (.txt) dengan file Markdown KAK/TOR di direktori `kak_tor_md_base_path`.\n"
        "Return: dict {instruction, context}"
    ),
)
def build_summary_tender_payload_tool(
    prompt_instruction_name: str = "kak_analyzer",
    kak_tor_md_name: Optional[str] = None,
) -> Dict[str, str]:
    """Mempersiapkan payload untuk LLM dengan menggabungkan instruksi dan konteks.

    Fungsi ini mengambil nama file instruksi (prompt) dan nama file KAK/TOR
    (Kerangka Acuan Kerja/Terms of Reference) dalam format Markdown. Fungsi ini
    akan membaca konten dari kedua file tersebut dan menggabungkannya menjadi
    sebuah dictionary.

    Payload yang dihasilkan memiliki dua kunci: 'instruction' dan 'context',
    yang siap untuk diteruskan ke model bahasa (LLM) untuk tugas-tugas
    seperti pembuatan ringkasan.

    Args:
        prompt_instruction_name (str): Nama file instruksi (tanpa ekstensi .txt)
            yang terletak di direktori `settings.templates_base_path`.
        kak_tor_md_name (Optional[str], optional): Nama file Markdown KAK/TOR
            (tanpa ekstensi .md) yang terletak di `settings.kak_tor_md_base_path`.
            Jika tidak disediakan, file pertama di direktori tersebut akan
            digunakan. Defaultnya adalah None.

    Returns:
        Dict[str, str]: Sebuah dictionary dengan kunci 'instruction' dan 'context'
            yang berisi konten dari masing-masing file.
    """
    kak_tor_md_name = _normalize_md_name(kak_tor_md_name)
    base = Path(settings.kak_tor_md_base_path)

    candidates = [
        base / f"{kak_tor_md_name}.md",
        base / f"{kak_tor_md_name.lower().replace(' ', '_')}.md",  # type: ignore
    ]

    candidates += list(base.glob(f"*{kak_tor_md_name}*.md"))

    for p in candidates:
        if p.is_file():
            kak_tor_md_name = p.name
            break
    files = [f.name for f in base.glob("*.md")]

    try:
        result = rag_tools.build_summary_tender_payload(
            prompt_instruction_name, kak_tor_md_name
        )

        return {"instruction": result["instruction"], "context": result["context"]}
    except Exception as e:
        return {
            "status": f"failure: {e}",
            "error": f"Markdown '{kak_tor_md_name}' tak ditemukan. File tersedia: {files}",
        }


# ---------------------------------------------------------------------------
# 6. Retrieval with similarity + metadata filter
# ---------------------------------------------------------------------------
@mcp.tool(
    name="rag_retrieval",
    title="RAG Retrieval",
    description=(
        "Gunakan tool ini jika pengguna menanyakan tentang detail produk, rincian proyek, "
        "atau isi dokumen KAK/TOR tender. "
        "Tool melakukan similarity-search (RAG) dengan filter metadata pada knowledge-base, "
        "lalu mengembalikan potongan teks paling relevan beserta citation."
        "metadata_filter (Optional[Dict[str, Any]], optional): Dictionary untuk "
        "memfilter hasil berdasarkan metadata. Contoh: `{'project': 'proyek_a'}`."
        "Defaultnya adalah None."
    ),
)
def rag_retrieval_tool(
    query: str,
    k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Melakukan pencarian di vector store menggunakan RAG dengan filter.

    Fungsi ini menjalankan pencarian kemiripan (similarity search) terhadap
    basis pengetahuan (vector store) berdasarkan query yang diberikan. Pencarian
    dapat disaring lebih lanjut menggunakan filter metadata untuk mendapatkan
    hasil yang lebih spesifik dan relevan.

    Hasilnya adalah daftar potongan teks (chunks) yang paling relevan
    disertai dengan metadata sumbernya (citation).

    Args:
        query (str): Teks atau pertanyaan yang akan digunakan untuk pencarian.
        k (Optional[int], optional): Jumlah dokumen teratas yang ingin diambil.
            Jika tidak disediakan, akan menggunakan nilai default dari
            `settings.retriever_search_k`. Defaultnya adalah None.
        metadata_filter (Optional[Dict[str, Any]], optional): Dictionary untuk
            memfilter hasil berdasarkan metadata. Contoh: `{"project": "proyek_a"}`.
            Defaultnya adalah None.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi hasil pencarian.
            Kunci 'result' akan berisi daftar dokumen yang relevan.
    """
    result = rag_tools.retrieval_with_filter(query, k, metadata_filter)
    return {"result": result}


# ---------------------------------------------------------------------------
# 7. Generate Proposal Document (.docx)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="generate_proposal_docx",
    title="Generate Proposal .docx",
    description=(
        "Generate docx proposal dari template default dengan data konteks.\n"
        "Args:\n"
        "  context: dict mapping setiap placeholder (list atau string) ke isinya.\n"
        "Contoh:\n"
        "```json\n"
        '{"judul_proposal":"…","nama_pelanggan":"…",…}\n'
        "```\n"
        "Jika ada key yang hilang, model akan diminta ulang oleh client."
    ),
)
def generate_proposal_docx_tool(
    context: Dict[str, Any],
    override_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Merender dan menyimpan dokumen proposal Word (.docx) dari data konteks.

    Fungsi ini mengambil sebuah dictionary `context` yang berisi variabel-variabel
    untuk template, dan secara opsional path ke template `.docx` lain.
    Fungsi ini akan merender template dengan data tersebut dan menyimpan
    dokumen yang dihasilkan ke direktori output yang telah dikonfigurasi.

    Args:
        context (Dict[str, Any]): Sebuah dictionary yang berisi data untuk
            dirender ke dalam template proposal. Kunci dalam dictionary ini
            harus sesuai dengan placeholder di dalam file template `.docx`.
        override_template (Optional[str], optional): Path ke file template
            `.docx` alternatif untuk menggantikan template default.
            Defaultnya adalah None.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi hasil dari proses.
            - Jika berhasil: `{"status": "success", "product": "...", "path": "..."}`
            - Jika gagal: `{"status": "failure", "error": "..."}`
    """
    return doc_tools.generate_proposal(context, override_template)


# ---------------------------------------------------------------------------
# 8. Get Template Placeholders
# ---------------------------------------------------------------------------
@mcp.tool(
    name="get_template_placeholders",
    title="Kembalikan array placeholder; simpan ke memori untuk mencocokkan context.",
    description=(
        "Dapatkan daftar semua placeholder (variabel) yang valid dari template proposal .docx default. "
        "Gunakan ini untuk mengetahui key apa saja yang harus ada di dalam parameter `context` "
        "saat memanggil `generate_proposal_docx`."
    ),
)
def get_template_placeholders_tool() -> Dict[str, Any]:
    """Mengambil daftar placeholder yang valid dari template .docx default.

    Fungsi ini memeriksa file template proposal .docx default dan mengekstrak
    semua variabel Jinja2 (placeholder) yang ada di dalamnya. Ini sangat berguna
    untuk memastikan bahwa `context` yang dikirim ke `generate_proposal_docx_tool`
    berisi semua kunci yang diperlukan.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi status dan daftar placeholder.
            - Jika berhasil: `{"status": "success", "placeholders": ["key1", "key2"]}`
            - Jika gagal: `{"status": "failure", "error": "<pesan_error>"}`
    """
    return doc_tools.get_template_placeholders()


# ---------------------------------------------------------------------------
# 9. Read Markdown KAK/TOR
# ---------------------------------------------------------------------------
@mcp.tool(
    name="read_project_markdown",
    title="Baca markdown text akan dipakai untuk menyusun context dalam pembuatan proposal.",
    description="Baca markdown KAK/TOR proyek dan kembalikan sebagai string.",
)
def read_project_markdown(project_name: str) -> Dict[str, Any]:
    base = Path(settings.kak_tor_md_base_path)
    # 1. coba persis
    candidates = [
        base / f"{project_name}.md",
        base / f"{project_name.lower().replace(' ', '_')}.md",
    ]
    # 2. fallback glob fuzzy
    candidates += list(base.glob(f"*{project_name}*.md"))

    for p in candidates:
        if p.is_file():
            return {
                "status": "success",
                "file": str(p),
                "text": p.read_text(encoding="utf-8"),
            }

    files = [f.name for f in base.glob("*.md")]
    return {
        "status": "failure",
        "error": f"Markdown '{project_name}' tak ditemukan. File tersedia: {files}",
    }


# ---------------------------------------------------------------------------
# 10. Save summaries markdown files to filesystem
# ---------------------------------------------------------------------------
@mcp.tool(
    name="save_summary_markdown",
    title="Save Summary as Markdown (.md)",
    description=(
        "Simpan string *summary* ke file .md di direktori yang sudah ditentukan.\n"
        "Anda WAJIB membuat nama proyek secara otomatis dengan format nama_pelanggan_nama_proyek.\n"
        "Argumen:\n  • summary (str): Teks ringkasan yang akan disimpan.\n  • project (str, optional): Nama proyek untuk penamaan file.\n\n"
        'Return: {"status": "success"|"failure", "file": <path>, "error": <msg>}'
    ),
)
def save_summary_markdown_tool(
    summary: Any,  # string JSON atau dict
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simpan *summary* (JSON) menjadi file Markdown di
    ``settings.summaries_md_base_path`` dengan nama file
    <slug_project>.md (tanpa timestamp).
    """
    # ----------------- Dir target -----------------
    base_dir = Path(settings.summaries_md_base_path).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Buat slug dari project (nama file asal tanpa ekstensi)
    slug = _slugify(project) if project else "summary"
    file_path = base_dir / f"{slug}.md"

    # ----------------- Normalisasi summary --------
    if (
        isinstance(summary, dict)
        and "summary" in summary
        and isinstance(summary["summary"], str)
    ):
        summary_json_str = summary["summary"]
    elif isinstance(summary, dict):
        summary_json_str = json.dumps(summary, ensure_ascii=False, indent=2)
    elif isinstance(summary, str):
        summary_json_str = summary
    else:
        return {"status": "failure", "error": "Tipe summary tidak dikenali"}

    # ----------------- Tulis file -----------------
    try:
        file_path.write_text(_to_markdown(summary_json_str), encoding="utf-8")
        return {"status": "success", "file": str(file_path)}
    except Exception as exc:
        return {"status": "failure", "error": str(exc)}


# ---------------------------------------------------------------------------
# Run the server if executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse")
