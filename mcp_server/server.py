from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from mcp.server.fastmcp import FastMCP  # type: ignore
from mcp_server.tools.rag_tools import RAGTools
from mcp_server.tools.docx_tools import DocGeneratorTools
from mcp_server.settings import Settings
from mcp_server.utils.helper import _slugify, _to_markdown, _normalize_md_name


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
# 3. Ingest KAK/TOR Markdown → vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="add_kak_tor_summaries_knowledge",
    title="Ingest summaries KAK/TOR Markdown",
    description=(
        "STEP 3 – FINAL. Panggil hanya setelah Anda MEMBANGUN sendiri JSON `markdown_name`, `project`, dan `tahun` "
        "yang MENGISI SEMUA arguments.\n"
        "Args:\n"
        "  markdown_name: str file dari save_summary_markdown_tool.\n"
        "  project: str nama proyek.\n"
        "  tahun: str tahun dokumen.\n"
    ),
)
def add_kak_tor_summaries_knowledge_tool(
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
# 4. Build Summary Tender Payload
# ---------------------------------------------------------------------------
@mcp.tool()
def list_kak_files() -> list[str]:
    base = Path(settings.kak_tor_md_base_path)
    files = [f.name for f in base.glob("*.md")]
    return files


@mcp.tool(
    name="build_summary_tender_payload",
    title="Build Summary Tender Payload",
    description=(
        "Gabungkan prompt instruction (.txt) dengan file Markdown KAK/TOR, "
        "kembali dict {instruction, context}."
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
    """Membangun payload untuk LLM dari template instruksi dan file konteks.

    Fungsi ini menggabungkan sebuah template instruksi (prompt) dengan konten
    dari satu atau lebih file Markdown (misalnya, KAK/TOR). Hasilnya adalah
    sebuah dictionary yang siap digunakan sebagai input untuk model bahasa (LLM),
    memisahkan dengan jelas antara instruksi tugas dan data konteks.

    Args:
        template_name (str): Nama file template prompt (tanpa ekstensi .txt)
            yang terletak di direktori `settings.templates_base_path`.
        kak_md_dir (Optional[str], optional): Path ke direktori yang berisi
            file-file Markdown untuk konteks. Jika tidak disediakan, akan
            menggunakan path default dari `settings.kak_tor_md_base_path`.
            Defaultnya adalah None.
        selected_files (Optional[List[str]], optional): Daftar nama file
            Markdown spesifik (termasuk ekstensi .md) yang akan digabungkan
            sebagai konteks. Jika None, semua file di `kak_md_dir` akan
            digunakan. Defaultnya adalah None.

    Returns:
        Dict[str, Any]: Sebuah dictionary dengan kunci 'instruksi' dan 'context'.
    """
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
# 7. Reset / rebuild vectorstore
# ---------------------------------------------------------------------------


@mcp.tool(
    name="reset_vectordb",
    title="Reset Vector Store",
    description="Hapus seluruh data dan rebuild tabel vectorstore kosong.",
)
def reset_vectorstore_tool() -> Dict[str, Any]:
    """Menghapus seluruh data dan membuat ulang vector store dari awal.

    Fungsi ini melakukan operasi yang bersifat destruktif dengan menghapus
    seluruh koleksi data (chunks dan embeddings) dari vector store. Setelah
    penghapusan, sebuah tabel kosong baru akan dibuat.

    Peringatan: Gunakan dengan hati-hati karena tindakan ini tidak dapat
    diurungkan dan akan menghapus semua pengetahuan yang telah diindeks.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang mengonfirmasi bahwa proses
            reset telah selesai. Contoh: {"status": "Vectorstore berhasil di-reset."}
    """
    result = rag_tools.reset_knowledge_base()
    return {"status": "Vectorstore berhasil di-reset.", "result": result}


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
    """Memperbarui metadata dari sekumpulan chunk di vector store secara massal.

    Fungsi ini memungkinkan pembaruan metadata untuk beberapa chunk sekaligus.
    Ia akan mencari semua chunk yang cocok dengan kriteria pada `metadata_filter`,
    kemudian memperbarui atau menambahkan field metadata mereka dengan data dari
    `new_metadata`.

    Ini sangat berguna untuk mengoreksi atau memperkaya informasi secara massal,
    misalnya mengubah label tahun atau menambahkan tag status pada sekelompok
    dokumen.

    Args:
        metadata_filter (Dict[str, Any]): Dictionary yang berfungsi sebagai filter
            untuk memilih chunk mana yang akan diperbarui. Kunci adalah nama field
            metadata dan nilai adalah nilai yang harus cocok.
            Contoh: `{"project": "proyek_a"}`.
        new_metadata (Dict[str, Any]): Dictionary yang berisi field dan nilai
            metadata baru yang akan diterapkan pada chunk yang cocok. Field yang
            sudah ada akan ditimpa.
            Contoh: `{"tahun": "2024", "status": "final"}`.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang melaporkan jumlah chunk yang
            berhasil diperbarui. Contoh: `{"updated": 50}`.
    """
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
    """Mengambil statistik dari vector store.

    Fungsi ini mengumpulkan dan mengembalikan data statistik mengenai
    basis pengetahuan (vector store), memberikan gambaran umum tentang
    isinya. Statistik yang dikembalikan meliputi:
    - Jumlah total chunk (baris) yang tersimpan.
    - Ukuran total vector store di disk (dalam MB).
    - Daftar unik dari semua nama proyek yang ada di metadata.
    - Distribusi jumlah chunk berdasarkan tahun.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi statistik, contoh:
            {
                "total_rows": 1234,
                "size_mb": 56.78,
                "projects": ["proyek_a", "proyek_b"],
                "tahun_distribution": {"2024": 800, "2025": 434}
            }
    """
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
    """Menghitung ulang dan memperbarui embeddings untuk semua chunk di vector store.

    Fungsi ini melakukan proses pemeliharaan dengan membaca semua teks dari
    chunk yang ada, menghitung ulang vector embedding-nya menggunakan model
    embedding yang saat ini dikonfigurasi, dan kemudian membangun ulang
    vector store dengan data yang baru.

    Proses ini sangat berguna ketika model embedding diperbarui atau diganti,
    untuk memastikan konsistensi data.

    Peringatan: Operasi ini bisa memakan waktu lama dan sumber daya yang
    intensif, tergantung pada jumlah data di vector store.

    Args:
        batch_size (int, optional): Ukuran batch untuk memproses chunk saat
            membangun ulang. Defaultnya adalah 100.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang mengonfirmasi penyelesaian proses.
    """
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
    """Mengambil daftar nilai unik untuk field metadata tertentu dari vector store.

    Fungsi ini sangat berguna untuk menemukan nilai-nilai yang tersedia
    untuk pemfilteran dalam pencarian. Misalnya, untuk mendapatkan semua
    nama proyek atau tahun yang telah diindeks.

    Args:
        field (str): Nama field metadata yang ingin diperiksa.
            Contoh: 'project', 'tahun', 'filename'.

    Returns:
        Dict[str, Any]: Sebuah dictionary dengan kunci 'values' yang berisi
            daftar nilai unik yang ditemukan untuk field tersebut.
            Contoh: {"values": ["proyek_a", "proyek_b", "proyek_c"]}
    """
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
    """Mengambil konteks produk menggunakan RAG dan menggabungkannya dengan template prompt.

    Fungsi ini berfungsi sebagai jembatan ke sistem RAG untuk mencari informasi
    relevan tentang suatu produk. Hasil pencarian (konteks) kemudian dapat
    digabungkan dengan sebuah template instruksi (prompt) untuk mempersiapkan
    input yang siap digunakan oleh model bahasa (LLM) dalam tugas seperti
    pembuatan proposal.

    Args:
        product (str): Nama atau deskripsi produk yang akan dicari konteksnya
            di dalam basis pengetahuan.
        k (Optional[int], optional): Jumlah potongan (chunk) teratas yang paling
            relevan untuk diambil. Jika tidak disediakan, akan menggunakan nilai
            default yang dikonfigurasi dalam sistem RAG. Defaultnya adalah None.
        metadata_filter (Optional[Dict[str, Any]], optional): Filter untuk
            mempersempit pencarian berdasarkan metadata yang ada.
            Contoh: `{"tahun": "2024"}`. Defaultnya adalah None.
        prompt_template (Optional[str], optional): Nama file template prompt
            (tanpa ekstensi .txt) yang akan digabungkan dengan konteks.
            File ini harus berada di direktori prompt yang telah dikonfigurasi.
            Defaultnya adalah None.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi hasil retrieval.
            Contoh sukses: `{"status": "success", "context": [...], "instruction": "..."}`.
            Contoh gagal: `{"status": "failure", "error": "pesan error"}`.
    """
    return doc_tools.retrieve_product_context(
        product, k, metadata_filter, prompt_template
    )


# ---------------------------------------------------------------------------
# 13. Extract Text from Document → Markdown
# ---------------------------------------------------------------------------


@mcp.tool(
    name="extract_document_text",
    title="Extract Document Text",
    description="Ekstrak teks dari dokumen (.pdf/.docx) → markdown.",
)
def extract_document_text_tool(file_path: str) -> Dict[str, Any]:
    """Mengekstrak teks dari file dokumen dan mengonversinya ke format Markdown.

    Fungsi ini mengambil path ke sebuah file dokumen (seperti PDF, DOCX, atau
    Markdown) dan mengekstrak konten teksnya. Hasilnya dikembalikan dalam
    format Markdown, yang ideal untuk diproses lebih lanjut oleh model bahasa
    (LLM).

    Args:
        file_path (str): Path lokal ke file dokumen yang akan diekstrak.
            Format yang didukung termasuk .pdf, .docx.

    Returns:
        Dict[str, Any]: Sebuah dictionary yang berisi status dan hasil ekstraksi.
            - Jika berhasil: `{"status": "success", "text": "<teks_markdown>"}`
            - Jika gagal: `{"status": "failure", "error": "<pesan_error>"}`
    """
    return doc_tools.extract_document_text(file_path)


# ---------------------------------------------------------------------------
# 14. Generate Proposal Document (.docx)
# ---------------------------------------------------------------------------


@mcp.tool(
    name="generate_proposal_docx",
    title="(STEP-3) Generate Proposal .docx",
    description=(
        "STEP 3 – FINAL. Panggil hanya setelah Anda MEMBANGUN sendiri JSON `context` "
        "yang MENGISI SEMUA placeholder.\n"
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
# 15. Get Template Placeholders
# ---------------------------------------------------------------------------


@mcp.tool(
    name="get_template_placeholders",
    title="STEP 2. Kembalikan array placeholder; simpan ke memori untuk mencocokkan context.",
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
# 16. Read Markdown KAK/TOR
# ---------------------------------------------------------------------------
@mcp.tool(
    name="read_project_markdown",
    title="STEP 1 dari pembuatan proposal. Hasil text akan dipakai untuk menyusun context.",
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
# MCP Tool Definition
# ---------------------------------------------------------------------------


@mcp.tool(
    name="save_summary_markdown",
    title="Save Summary as Markdown (.md)",
    description=(
        "Simpan string *summary* ke file .md di direktori yang sudah ditentukan.\n\n"
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
    ``settings.summaries_md_base_path``.
    """
    # ----------------- Dir target -----------------
    base_dir = Path(settings.summaries_md_base_path).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(project) if project else "summary"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = base_dir / f"{slug}_{timestamp}.md"

    # ----------------- Normalisasi summary --------
    # 1) Jika bentuk {'summary': '{...json...'} → ambil field dalam
    if (
        isinstance(summary, dict)
        and "summary" in summary
        and isinstance(summary["summary"], str)
    ):
        summary_json_str = summary["summary"]

    # 2) Jika sudah dict Python → dump ke JSON
    elif isinstance(summary, dict):
        summary_json_str = json.dumps(summary, ensure_ascii=False, indent=2)

    # 3) Jika string → anggap string JSON (atau teks apapun)
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
