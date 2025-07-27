from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Optional
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


# Helper save_summary_markdown_tool
def to_markdown(content: str) -> str:
    """Bungkus JSON ke blok code Markdown."""
    return f"## Ringkasan Tender\n\n```json\n{content}\n```\n"


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
