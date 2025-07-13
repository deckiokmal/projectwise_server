from __future__ import annotations
import textwrap
import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Utility helpers Safe Filename
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    slug = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    return slug or "summary"


# Helper save_summary_markdown_tool
def _to_markdown(content: str) -> str:
    """Bungkus JSON ke blok code Markdown."""
    return textwrap.dedent(f"""\
    ## Ringkasan Tender

    ```json
    {content}
    ```
    """)


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
