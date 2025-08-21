# file: mcp_server/tools/docgen_product_tools.py
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Coroutine

from mcp_server.settings import Settings
from mcp_server.utils.document_pipeline import DocumentPipeline, DocumentRequest
from mcp_server.utils.product_path_utils import open_product_summary
from mcp_server.utils.logger import get_logger


logger = get_logger(__name__)


class DocGenProductTools:
    """Facade utilitas untuk pembuatan dokumen proposal produk.

    - Method sinkron agar mudah dipanggil dari @mcp.tool() yang tidak async.
    - Memastikan semua konfigurasi mengacu ke Settings().
    - Logging & error handling di titik penting.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        pipeline: Optional[DocumentPipeline] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.dp = pipeline or DocumentPipeline()

    # ──────────────────────────────────────────────────────────────
    # Util internal: jalankan coroutine pada konteks sync
    # ──────────────────────────────────────────────────────────────
    def _run_async(self, coro: Coroutine[Any, Any, Dict[str, Any]]) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            result: Dict[str, Any] = {}
            exc: list[BaseException] = []

            def worker():
                nonlocal result
                try:
                    result = asyncio.run(coro)
                except BaseException as e:
                    exc.append(e)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join()
            if exc:
                raise exc[0]
            return result

    # ──────────────────────────────────────────────────────────────
    # 1) Ambil placeholder template (zero-arg friendly)
    # ──────────────────────────────────────────────────────────────
    def get_template_placeholders(
        self,
        *,
        product_category: str,
        template_name: Optional[str] = None,
        override_template: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Kembalikan daftar placeholder dari template.

        Prioritas: override > template_name > default_map(category).
        """
        t0 = time.perf_counter()
        try:
            logger.info(
                "[DocGen] Preview placeholders | cat=%s | tmpl=%s | override=%s",
                product_category,
                template_name,
                override_template,
            )

            async def _do():
                return await self.dp.preview_placeholders(
                    category=product_category,
                    template_name=template_name,
                    override=Path(override_template) if override_template else None,
                )

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            logger.exception("[DocGen] Gagal preview placeholders")
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    # ──────────────────────────────────────────────────────────────
    # 2) Baca product summary (MD) untuk dipakai ke context
    # ──────────────────────────────────────────────────────────────
    def read_product_summary(
        self, product: str, category: str, tahun: str, filename: str
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            logger.info(
                "[DocGen] Read product summary | %s / %s / %s / %s",
                product,
                category,
                tahun,
                filename,
            )
            content = open_product_summary(
                self.settings, product, category, tahun, filename
            )
            return {
                "status": "success",
                "message": "Summary loaded",
                "data": {"content": content},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }
        except Exception as e:
            logger.warning("[DocGen] Gagal read product summary: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    # ──────────────────────────────────────────────────────────────
    # 3) Generate proposal DOCX dengan integrasi product summary
    # ──────────────────────────────────────────────────────────────
    def generate_proposal(
        self,
        *,
        product_category: str,
        context: Dict[str, Any],
        template_name: Optional[str] = None,
        override_template: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        use_llm_fill: Optional[bool] = None,
        # opsi integrasi summary produk
        product: Optional[str] = None,
        category: Optional[str] = None,
        tahun: Optional[str] = None,
        summary_filename: Optional[str] = None,
        summary_target_field: Optional[str] = "executive_summary",
        summary_mode: Optional[str] = "append",
    ) -> Dict[str, Any]:
        """Bangun dokumen proposal .docx siap unduh.

        - Bila argumen `product/category/tahun/summary_filename` diberikan, pipeline akan otomatis merge
          konten MD ke field `summary_target_field` (default: executive_summary) dengan mode append/prepend/override.
        """
        t0 = time.perf_counter()
        try:
            if context is None:
                return {
                    "status": "error",
                    "message": "'context' wajib diisi.",
                    "data": {},
                    "timing": {},
                }

            logger.info(
                "[DocGen] Generate | cat=%s | tmpl=%s | override=%s | out=%s | use_llm=%s",
                product_category,
                template_name,
                override_template,
                output_dir,
                use_llm_fill,
            )
            req = DocumentRequest(
                product_category=product_category,
                context=context,
                template_name=template_name,
                override_template_path=Path(override_template)
                if override_template
                else None,
                output_dir=Path(output_dir) if output_dir else None,
                use_llm_fill=use_llm_fill,
                # pass-through opsi product summary
                product=product,
                category=category,
                tahun=tahun,
                summary_filename=summary_filename,
                summary_target_field=summary_target_field,
                summary_mode=summary_mode
                if summary_mode in ("append", "override", "prepend")
                else "append",
            )

            async def _do():
                return await self.dp.generate(req)

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            logger.exception("[DocGen] Gagal generate proposal")
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    # ──────────────────────────────────────────────────────────────
    # 4) Admin helper – delegasi ke pipeline
    # ──────────────────────────────────────────────────────────────
    def list_templates(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.list_templates()

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    def add_template(
        self, template_name: str, source_path: str | Path
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.add_template(template_name, Path(source_path))

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    def update_template(
        self, template_name: str, source_path: str | Path
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.update_template(template_name, Path(source_path))

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    def delete_template(self, template_name: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.delete_template(template_name)

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    def list_documents(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.list_documents()

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }

    def delete_document(self, name: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:

            async def _do():
                return await self.dp.delete_document(name)

            r = self._run_async(_do())
            r.setdefault("timing", {})
            r["timing"].setdefault("elapsed_s", round(time.perf_counter() - t0, 6))
            return r
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "data": {},
                "timing": {"elapsed_s": round(time.perf_counter() - t0, 6)},
            }
