from __future__ import annotations
import re
import tiktoken
import unicodedata
from pathlib import Path
from typing import Optional, List
from mcp_server.utils.logger import logger


# ---------------------------------------------------------------------------
# Utility helpers Safe Filename
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Ubah text menjadi slug yang aman untuk nama folder.

    Args:
        text (str): text yang akan diubah menjadi slug.

    Returns:
        str: slug aman. contoh: "Nama Folder" -> "nama_folder"
    """
    slug = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")


# helper build_summary_tender_payload_tool - kak_tor_md_name
_MD_RE = re.compile(r"\.md$", re.I)


def normalize_md_name(name: Optional[str]) -> Optional[str]:
    """
    • Hilangkan ekstensi .md (huruf besar/kecil)
    • Trim spasi di kiri-kanan
    • Kembalikan None bila arg None/blank
    """
    if not name:
        return None
    return _MD_RE.sub("", name.strip())


# ============================================================
# Helper untuk membungkus JSON ke Markdown Ringkasan
# ============================================================
def to_kak_markdown(content: str) -> str:
    """Bungkus JSON ke blok code Markdown."""
    return f"## Ringkasan Tender\n\n```json\n{content}\n```\n"


def to_product_markdown(content: str) -> str:
    """Bungkus JSON ke blok code Markdown."""
    return f"## Ringkasan sizing produk\n\n```json\n{content}\n```\n"


# ============================================================
# Helper untuk menampilkan daftar file di direktori
# ============================================================
def list_files(base_path: str) -> list[str]:
    """
    Tampilkan seluruh list nama files di direktori `base_path`
    dengan ekstensi .md dan .pdf, terurut secara alfabet.
    """
    base = Path(base_path).expanduser().resolve()

    # 1) Pastikan path ada dan adalah direktori
    if not base.exists() or not base.is_dir():
        logger.warning(f"Path tidak ditemukan atau bukan direktori: {base}")
        return []

    # 2) Iterasi dan filter
    files = [
        f.name
        for f in base.iterdir()
        if f.is_file() and f.suffix.lower() in (".md", ".pdf")
    ]

    return sorted(files)


# ============================================================
# Helper untuk membersihkan teks UTF-8
# Mengganti karakter yang tidak valid dengan 'replace'
# ============================================================
def clean_utf8(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


# ============================================================
# Helper untuk memilih tokenizer yang sesuai dengan model
# ============================================================
def get_tokenizer(model_name: str):
    """
    Kembalikan tokenizer sesuai model.
    Jika model menggunakan OpenAI, gunakan tokenizer tiktoken.
    Jika model Ollama, fallback ke tokenizer 'cl100k_base'.
    """
    try:
        if model_name.startswith("gpt") or "openai" in model_name:
            return tiktoken.encoding_for_model(model_name)
        else:
            # fallback tokenizer universal
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(
            f"Tokenizer untuk model '{model_name}' tidak ditemukan, menggunakan default.\n{e}"
        )
        return tiktoken.get_encoding("cl100k_base")


# ============================================================
# Fungsi untuk membagi teks panjang menjadi potongan sesuai token limit
# ============================================================
def split_by_token_limit(text: str, tokenizer, max_tokens: int) -> List[str]:
    """
    Membagi teks panjang menjadi potongan kecil sesuai token limit.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


# ============================================================
# Helper ringkasan dokumen panjang dengan chunking
# ============================================================
async def summarize_long_product_text(
    llm, full_text: str, instruction: str, model_name: str, max_tokens: int = 3000
) -> str:
    """
    Ringkas teks panjang dengan memecahnya menjadi beberapa bagian
    jika jumlah token melebihi context window model.
    """
    try:
        tokenizer = get_tokenizer(model_name)
        total_tokens = len(tokenizer.encode(full_text))

        logger.info(
            f"Token total input: {total_tokens} | Limit per chunk: {max_tokens}"
        )

        if total_tokens <= max_tokens:
            # Jika tidak melebihi limit → langsung ringkas
            summary = await llm.generate_text(input=full_text, instructions=instruction)
            return summary

        # Jika panjang → bagi menjadi bagian kecil
        chunks = split_by_token_limit(full_text, tokenizer, max_tokens)
        summaries = []

        for idx, part in enumerate(chunks):
            logger.info(f"Ringkas bagian {idx + 1}/{len(chunks)}")
            try:
                part_summary = await llm.generate_text(
                    input=part, instructions=instruction
                )
                summaries.append(part_summary.strip())
            except Exception as e:
                logger.warning(f"Gagal ringkas bagian ke-{idx + 1}: {e}")

        # Gabungkan semua ringkasan per bagian menjadi satu teks utuh
        full_summary = "\n\n".join(summaries).strip()

        if not full_summary:
            raise ValueError("Ringkasan akhir kosong setelah seluruh bagian diringkas.")

        return full_summary

    except Exception as e:
        logger.error(f"Gagal melakukan ringkasan multi-bagian: {e}")
        return "[Ringkasan gagal dibuat karena input terlalu panjang atau kesalahan internal]"
