# mcp_server/tools/kak_analyzer_tool.py
from __future__ import annotations

import asyncio
from pathlib import Path
from functools import partial
from typing import Dict, Optional, Any

from mcp_server.settings import Settings
from mcp_server.utils.cpu_workers import convert_and_chunk_pdf_worker
from mcp_server.utils.multiprocessing_utils import get_cpu_pool

from mcp_server.utils.logger import get_logger
from mcp_server.utils.rag_pipeline import RAGPipeline
from mcp_server.utils.llm_chains import LLMChains
from mcp_server.utils.ingestion_manifest_utils import (
    load_manifest,
    save_manifest,
    build_kak_unique_key,
)
from mcp_server.utils.helper import summarize_long_product_text
from mcp_server.utils.kak_path_utils import (
    resolve_kak_pdf,
    save_kak_md,
    open_kak_md,
    save_kak_summary,
    open_kak_summary,
)


logger = get_logger(__name__)


class KAKTools:
    def __init__(self):
        """
        Inisialisasi KAKTools dengan RAGPipeline async-ready.
        """
        self.settings = Settings()
        self.pipeline = RAGPipeline()
        self.llmchains = LLMChains()
        self.manifest_path = Path(self.settings.json_ingestion_manifest_file)
        self._manifest = load_manifest(self.manifest_path)

    # ===============
    # KAK/TOR tools
    # ===============
    async def ingest_kak_file(
        self,
        filename: str,
        pelanggan: str,
        project: str,
        tahun: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        1. Extract file PDF index ke vectorstore jika belum pernah diingest.
        2. Export ke Markdown.
        3. Update manifest.
        4. metadata: project, pelanggan, tahun
        5. Struktur path berbasis <pelanggan>/<tahun>
        """
        logger.info("KnowledgeBase: ingest_kak_file")

        # 1) cek input argument
        if not filename:
            return {"status": "error", "message": "Parameter 'filename' wajib diisi."}
        if not pelanggan:
            return {"status": "error", "message": "Parameter 'pelanggan' wajib diisi."}
        if not project:
            return {"status": "error", "message": "Parameter 'project' wajib diisi."}
        if not tahun:
            return {"status": "error", "message": "Parameter 'tahun' wajib diisi."}

        # 2) resolve full path kak pdf
        pdf_info = resolve_kak_pdf(
            self.settings,
            pelanggan,
            project,
            tahun,
            filename,
            create_dirs=False,
            unique=overwrite,
        )
        loop = asyncio.get_running_loop()
        pool = get_cpu_pool()

        try:
            # === 1) CPU-bound di worker (dl_doc dipakai di sana) ===
            func = partial(
                convert_and_chunk_pdf_worker,
                pdf_path=str(pdf_info.full_path),
                # kirim kwargs melalui partial
                tokenizer_kind="openai",
                tokenizer_model=self.settings.llm_model,
                tokenizer_max_tokens=128 * 1024,
                merge_peers=True,
                export_markdown=True,
            )
            worker_out = await loop.run_in_executor(pool, func)

            if worker_out.get("status") != "success":
                raise RuntimeError(
                    worker_out.get("message") or "Gagal convert/chunk PDF"
                )

            chunks_payload = worker_out["chunks"]  # list[dict]
            markdown: str | None = worker_out.get("markdown")

            # 3. Hapus entri lama di vectorDB jika overwrite
            if overwrite:
                await self.pipeline.delete_by_filename(pdf_info.filename_final)
                logger.info(
                    f"Overwrite aktif. Data lama dihapus: {pdf_info.filename_final}."
                )

            # === 3) Embedding + upsert dari payload ===
            ingest_result = await self.pipeline.ingest_kak_chunks_from_payload(
                chunks_payload=chunks_payload,
                filename=pdf_info.filename_final,
                pelanggan=pdf_info.pelanggan_final,
                project=pdf_info.project_final,
                tahun=tahun,
            )
            added = (ingest_result.get("data") or {}).get("added", 0)
            logger.info(f"Berhasil menambahkan {added} chunk ke vectorstore.")

        except Exception as e:
            logger.exception(f"Gagal ingest file '{filename}': {e}")
            return {"status": "error", "message": str(e)}

        # === 4) Simpan markdown bila tersedia ===
        md_info = None
        if markdown:
            md_info = save_kak_md(
                self.settings,
                pelanggan,
                project,
                tahun,
                filename,
                markdown,
                unique=overwrite,
            )

        # === 5) Update manifest ===
        unique_key = build_kak_unique_key(pelanggan, tahun, project)
        self._manifest[unique_key] = True
        save_manifest(self.manifest_path, self._manifest)

        return {
            "status": "success",
            "message": f"jumlah chunk ditambahkan: {added}",
            "markdown_file": (str(md_info.full_path) if md_info else None),
        }

    async def generate_kak_summarize(
        self,
        filename: str,
        pelanggan: str,
        project: str,
        tahun: str,
        prompt_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gabungkan prompt instruction + markdown kak dan hasilkan ringkasan via LLMChain,
        lalu simpan dalam file Markdown <pelanggan>/<tahun>/*_summary.md
        """
        logger.info("KnowledgeBase: generate_kak_summarize")

        # 1) cek input argument
        if not filename:
            return {"status": "error", "message": "Parameter 'filename' wajib diisi."}
        if not pelanggan:
            return {"status": "error", "message": "Parameter 'pelanggan' wajib diisi."}
        if not project:
            return {"status": "error", "message": "Parameter 'project' wajib diisi."}
        if not tahun:
            return {"status": "error", "message": "Parameter 'tahun' wajib diisi."}

        # 2) Baca Markdown KAK/TOR menggunakan util
        md_text = open_kak_md(
            self.settings,
            pelanggan=pelanggan,
            tahun=tahun,
            project=project,
            filename=filename,
        )
        full_input = f"<kak_tor>\n{md_text.strip()}\n</kak_tor>"

        # 3) Baca Prompt Instruction template
        prompt: str = prompt_instruction or "project_analysis.txt"
        prompt_path = Path(self.settings.prompts_base_path) / prompt
        if not prompt_path.is_file():
            logger.error(f"Template statis tidak ditemukan: {prompt_path}")

        instruction = prompt_path.read_text(encoding="utf-8", errors="replace")

        # 4) Proses summarize kak menggunakan utils LLMChain
        try:
            # 1. Panggil LLMChain untuk ringkasan
            summary_llm = await summarize_long_product_text(
                llmchains=self.llmchains,
                full_text=full_input,
                instruction=instruction.strip(),
                model_name=self.settings.llm_model,
                max_tokens=self.settings.max_token,
            )

            if not summary_llm.strip():
                raise ValueError("Ringkasan kosong, tidak bisa diproses.")

            if summary_llm.strip().startswith("[Ringkasan gagal"):
                logger.error("LLM gagal merangkum. Ringkasan tidak akan disimpan.")
                return {
                    "status": "error",
                    "summary": summary_llm,
                    "message": "LLM gagal membuat ringkasan yang valid.",
                }

            # 2. Simpan ringkasan sebagai file Markdown (nama otomatis *_summary.md)
            summary_info = save_kak_summary(
                self.settings,
                pelanggan=pelanggan,
                project=project,
                tahun=tahun,
                filename=project,
                markdown=summary_llm,
                unique=True,  # jika mau overwrite aman
            )

            return {
                "status": "success",
                "summary": summary_llm,
                "message": "KAK berhasil di analysis.",
                "summary_file": str(summary_info.full_path),
            }

        except Exception as e:
            logger.error(f"Gagal merangkum atau menyimpan KAK/TOR: {e}")
            return {"status": "error", "message": f"[Gagal menjalankan LLM]: {e}"}

    async def read_kak_summaries(
        self,
        filename: str,
        pelanggan: str,
        project: str,
        tahun: str,
    ) -> dict[str, str]:
        """
        Baca KAK summary menggunakan Path KAK Utils.
        Mengembalikan string summary.
        """
        logger.info(
            f"Membaca kak summaries pelanggan {pelanggan} tahun {tahun} project {project}"
        )

        # 1) cek input argument
        if not filename or filename.endswith("_summary.md"):
            return {"status": "error", "message": "Parameter 'filename' wajib diisi."}
        if not pelanggan:
            return {"status": "error", "message": "Parameter 'pelanggan' wajib diisi."}
        if not project:
            return {"status": "error", "message": "Parameter 'project' wajib diisi."}
        if not tahun:
            return {"status": "error", "message": "Parameter 'tahun' wajib diisi."}

        # 2) Baca markdown summary
        try:
            summaries = open_kak_summary(
                self.settings,
                pelanggan=pelanggan,
                project=project,
                tahun=tahun,
                filename=filename,
            )
            return {"status": "success", "summary": summaries}

        except Exception as e:
            logger.error(f"Gagal membaca KAK summary: {e}")
            return {
                "status": "error",
                "message": f"Gagal membaca KAK summary: {e}",
            }
