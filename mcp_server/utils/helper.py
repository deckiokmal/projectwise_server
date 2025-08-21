# mcp_server/utils/helper.py
from __future__ import annotations

import tiktoken
from pathlib import Path
from typing import List
from mcp_server.utils.logger import get_logger
from mcp_server.utils.llm_chains import LLMChains
import json


logger = get_logger(__name__)


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
def _split_by_token_limit(text: str, tokenizer, max_tokens: int) -> List[str]:
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
    llmchains: LLMChains,
    full_text: str,
    instruction: str,
    model_name: str,
    prefer: str = "chat",
    max_tokens: int = 3000,
) -> str:
    """
    Ringkas teks panjang dengan memecahnya menjadi beberapa bagian
    jika jumlah token melebihi context window model.
    """
    try:
        # Inisiasi tokenizer
        tokenizer = get_tokenizer(model_name)
        total_tokens = len(tokenizer.encode(full_text))

        logger.info(
            f"Token total input: {total_tokens} | Limit per chunk: {max_tokens}"
        )

        # Pesan awal - 1
        msg = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": full_text},
        ]

        # Check token limit
        # Eksekusi summary jika tidak melebihi limit
        if total_tokens <= max_tokens:
            # Jika tidak melebihi limit → langsung ringkas
            summary = await llmchains.generate_text(
                messages_or_input=msg, prefer=prefer
            )
            return summary.get("data")  # type: ignore

        # Jika panjang → bagi menjadi bagian kecil
        chunks = _split_by_token_limit(full_text, tokenizer, max_tokens)

        # Pesan yang telah di chunks
        msg_part = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": chunks},
        ]

        summaries = []

        # Jalankan summary tiap chunk
        for idx, part in enumerate(chunks):
            logger.info(f"Ringkas bagian {idx + 1}/{len(chunks)}")
            try:
                part_summary = await llmchains.generate_text(
                    messages_or_input=msg_part, prefer=prefer
                )
                summaries.append(part_summary.get("data").strip())  # type: ignore
            except Exception as e:
                logger.warning(f"Gagal ringkas bagian ke-{idx + 1}: {e}")

        # Gabungkan semua ringkasan per chunk menjadi satu teks utuh
        full_summary = "\n\n".join(summaries).strip()
        if not full_summary:
            raise ValueError("Ringkasan akhir kosong setelah seluruh bagian diringkas.")

        # Analysis kembali hasil summary tiap chunks
        msg_full = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": full_summary},
        ]

        summary_sum = await llmchains.generate_text(
            messages_or_input=msg_full, prefer=prefer
        )

        # Kembalikan summary utuh
        return summary_sum.get("data")  # type: ignore

    except Exception as e:
        logger.error(f"Gagal melakukan ringkasan multi-bagian: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": "[Ringkasan gagal dibuat karena input terlalu panjang atau kesalahan internal]",
                "error": e,
            }
        )
