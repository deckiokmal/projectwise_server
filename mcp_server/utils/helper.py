from __future__ import annotations
import re
import unicodedata
from typing import Optional
from pathlib import Path
from mcp_server.utils.logger import logger


# ---------------------------------------------------------------------------
# Utility helpers Safe Filename
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """
    Ubah text menjadi slug a-z0-9 dengan underscore.
    Contoh: "Proyek Core TTD" → "proyek_core_ttd"
    """
    # Normalisasi Unicode → ASCII
    slug = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    # Ganti semua karakter selain a-z0-9 menjadi underscore
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    return slug or "summary"


# Helper save_summary_markdown_tool
def _to_markdown(content: str) -> str:
    """Bungkus JSON ke blok code Markdown."""
    return f"## Ringkasan Tender\n\n```json\n{content}\n```\n"


# helper build_summary_tender_payload_tool - kak_tor_md_name
_MD_RE = re.compile(r"\.md$", re.I)


def _normalize_md_name(name: Optional[str]) -> Optional[str]:
    """
    • Hilangkan ekstensi .md (huruf besar/kecil)
    • Trim spasi di kiri-kanan
    • Kembalikan None bila arg None/blank
    """
    if not name:
        return None
    return _MD_RE.sub("", name.strip())


def _list_files(base_path: str) -> list[str]:
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
