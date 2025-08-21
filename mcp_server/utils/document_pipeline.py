# file: mcp_server/utils/document_pipeline.py
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator
from docxtpl import DocxTemplate

from mcp_server.utils.llm_chains import LLMChains
from mcp_server.utils.logger import get_logger
from mcp_server.utils.product_path_utils import open_product_summary


logger = get_logger(__name__)

try:
    from mcp_server.settings import Settings as AppSettings  # type: ignore
except Exception:  # pragma: no cover
    AppSettings = None  # type: ignore


# =============================
#             Errors
# =============================
class DocumentPipelineError(Exception):
    """Kesalahan umum pada pipeline dokumen."""


class TemplateNotFoundError(DocumentPipelineError):
    """Template tidak ditemukan."""


class ContextValidationError(DocumentPipelineError):
    """Context invalid / tak lengkap."""


class LLMFillError(DocumentPipelineError):
    """Gagal mengisi placeholder via LLM."""


class SaveError(DocumentPipelineError):
    """Gagal menyimpan dokumen."""


# =============================
#            Schemas
# =============================
class DocumentSettings(BaseModel):
    """Setting sentral pipeline (diambil dari Settings()).

    - Jangan membuat settings lain. Semua mengacu ke `mcp_server.settings.Settings`.
    - Jika `Settings` tidak tersedia, gunakan default/env var (fallback).
    """

    proposal_template_base_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "PROPOSAL_TEMPLATE_BASE_PATH", "mcp_server/data/document_templates"
            )
        )
    )
    proposal_generated_base_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("PROPOSAL_GENERATED_BASE_PATH", "data/proposal_generated")
        )
    )
    public_download_base_url: str = Field(
        default_factory=lambda: str(
            os.getenv("PUBLIC_DOWNLOAD_BASE_URL", "http://localhost:5000/api")
        )
    )

    max_cpu_workers: int = Field(default=max(1, (os.cpu_count() or 2) - 1))
    max_concurrent_proccess: int = Field(default=64)

    default_template_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "internet": "proposal_product_internet.docx",
            "project-nonmigas": "proposal_project_nonmigas.docx",
            "project-migas": "proposal_project_migas.docx",
            "managed-service": "proposal_managed_service.docx",
            "one-time-charge": "proposal_one-time-charge.docx",
            "security": "proposal_security.docx",
            "cloud": "proposal_cloud.docx",
        }
    )

    max_llm_fill: int = 8
    use_llm_default: bool = True

    list_keys: List[str] = Field(
        default_factory=lambda: [
            "list_tujuan",
            "list_hardware",
            "list_software",
            "list_lisensi",
            "list_jasa",
            "scope_of_work",
            "out_of_scope",
            "deliverables",
            "project_assumption",
            "detail_tahapan_metodologi_pelaksanaa_pekerjaan",
            "timeframe_sesuai_metodologi_pelaksanaan_pekerjaan",
            "term_and_condition_penawaran_harga",
        ]
    )
    optional_keys: List[str] = Field(
        default_factory=lambda: [
            "judul_proposal",
            "nama_pelanggan",
            "tanggal_hari_ini",
            "ringkasan_kebutuhan",
            "ringkasan_manfaat",
            "executive_summary",
            "response_time",
            "response_detail",
            "response_decscription",
            "resolution_time",
            "resolution_detail",
            "resolution_decscription",
        ]
    )

    @classmethod
    def from_app_settings(cls) -> "DocumentSettings":
        if AppSettings is None:
            return cls()
        app = AppSettings()  # type: ignore
        return cls(
            proposal_template_base_path=Path(
                getattr(
                    app,
                    "proposal_template_base_path",
                    "mcp_server/data/document_templates",
                )
            ),
            proposal_generated_base_path=Path(
                getattr(app, "proposal_generated_base_path", "data/proposal_generated")
            ),
            public_download_base_url=str(
                getattr(app, "public_download_base_url", "http://localhost:5000/api")
            ),
            max_cpu_workers=getattr(
                app, "max_cpu_workers", max(1, (os.cpu_count() or 2) - 1)
            ),
            max_concurrent_proccess=getattr(app, "max_concurrent_proccess", 64),
            default_template_map=getattr(
                app, "proposal_default_template_map", cls().default_template_map
            ),
            max_llm_fill=getattr(app, "proposal_max_llm_fill", 8),
            use_llm_default=getattr(app, "proposal_use_llm_default", True),
            list_keys=getattr(app, "proposal_list_keys", cls().list_keys),
            optional_keys=getattr(app, "proposal_optional_keys", cls().optional_keys),
        )


class ProposalContext(BaseModel):
    """Skema konteks proposal (fleksibel + extra_fields)."""

    judul_proposal: Optional[str] = None
    nama_pelanggan: Optional[str] = None
    tanggal_hari_ini: Optional[str] = None
    ringkasan_kebutuhan: Optional[str] = None
    ringkasan_manfaat: Optional[str] = None
    executive_summary: Optional[str] = None

    list_tujuan: Optional[List[str]] = None
    list_hardware: Optional[List[str]] = None
    list_software: Optional[List[str]] = None
    list_lisensi: Optional[List[str]] = None
    list_jasa: Optional[List[str]] = None
    scope_of_work: Optional[List[str]] = None
    out_of_scope: Optional[List[str]] = None
    deliverables: Optional[List[str]] = None
    project_assumption: Optional[List[str]] = None
    detail_tahapan_metodologi_pelaksanaa_pekerjaan: Optional[List[str]] = None
    timeframe_sesuai_metodologi_pelaksanaan_pekerjaan: Optional[List[str]] = None
    term_and_condition_penawaran_harga: Optional[List[str]] = None

    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_fields", mode="before")
    @classmethod
    def ensure_extra_dict(cls, v):
        return v or {}

    def to_render_context(self, settings: DocumentSettings) -> Dict[str, Any]:
        base = self.model_dump(exclude_none=True)
        extra = base.pop("extra_fields", {})
        ctx = {**base, **extra}
        # normalisasi: jaga konsistensi keys list & optional
        for k in settings.list_keys:
            ctx.setdefault(k, [])
        for k in settings.optional_keys:
            ctx.setdefault(k, "")
        return ctx


class DocumentRequest(BaseModel):
    """Permintaan generate dokumen (dinamis untuk product proposal)."""

    product_category: str = Field(
        description="Kategori produk (internet/security/...)."
    )
    context: Dict[str, Any]

    template_name: Optional[str] = Field(default=None)
    override_template_path: Optional[Path] = Field(default=None)
    output_dir: Optional[Path] = Field(default=None)
    use_llm_fill: Optional[bool] = None

    # ==== Integrasi product summary (memanfaatkan product_path_utils) ====
    product: Optional[str] = Field(
        default=None, description="Nama produk standar (untuk summary)"
    )
    category: Optional[str] = Field(
        default=None, description="Kategori produk standar (untuk summary)"
    )
    tahun: Optional[str] = Field(default=None, description="Tahun (untuk summary)")
    summary_filename: Optional[str] = Field(
        default=None, description="Nama file summary MD (tanpa _summary.md)"
    )
    summary_target_field: Optional[str] = Field(default="executive_summary")
    summary_mode: Optional[Literal["append", "override", "prepend"]] = Field(
        default="append"
    )


class PipelineResult(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, Any] = Field(default_factory=dict)


# =============================
#        Helper Functions
# =============================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)


def _safe_filename(base: str) -> str:
    base = (base or "proposal_tanpa_nama").strip()
    safe = "".join(c for c in base if c.isalnum() or c in (" ", ".", "_", "-"))
    return (safe.rstrip() + ".docx").replace("  ", " ")


def _parse_json_loose(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _render_docx_job(tmpl: str, ctx: Dict[str, Any], dst: str) -> str:
    doc = DocxTemplate(tmpl)
    doc.render(ctx)
    doc.save(dst)
    return dst


# =============================
#        Document Pipeline
# =============================
class DocumentPipeline:
    """Pipeline produksi dokumen proposal berbasis template .docx.

    - Semua konfigurasi via `Settings()` → diproyeksikan ke `DocumentSettings`
    - CPU-bound rendering di ProcessPool, throttled oleh semaphore global
    - LLMChain opsional untuk mengisi placeholder yang kosong
    - **Baru**: Otomatis merge *product standard summary* (MD) ke context
    """

    def __init__(
        self,
        settings: Optional[DocumentSettings] = None,
        llm_chain: Optional[LLMChains] = None,
    ) -> None:  # type: ignore[name-defined]
        self.settings = settings or DocumentSettings.from_app_settings()
        self._process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.settings.max_cpu_workers)
        )
        self._sema = asyncio.Semaphore(self.settings.max_concurrent_proccess)
        self._llm_chain = llm_chain
        if self._llm_chain is None and LLMChains is not None:  # type: ignore[name-defined]
            try:
                self._llm_chain = LLMChains()  # type: ignore[call-arg]
            except Exception as e:
                logger.warning(f"LLMChain tidak aktif: {e}")
                self._llm_chain = None
        self.settings.proposal_generated_base_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DocumentPipeline siap: base=%s | out=%s | workers=%s | limit=%s",
            str(self.settings.proposal_template_base_path),
            str(self.settings.proposal_generated_base_path),
            self.settings.max_cpu_workers,
            self.settings.max_concurrent_proccess,
        )

    # -------------------------------------------------------------
    #                     PUBLIC API (Core)
    # -------------------------------------------------------------
    async def generate(self, req: DocumentRequest) -> Dict[str, Any]:
        """Workflow end-to-end membuat dokumen proposal.

        Langkah:
        1) Resolve template → 2) Normalisasi context → 3) (opsional) Merge product summary
        4) (opsional) LLM fill → 5) Render & Save → 6) Return dict konsisten
        """
        t0 = time.perf_counter_ns()
        steps: Dict[str, float] = {}
        started_at = _now_iso()

        async with self._sema:  # throttle global
            try:
                # 1) Tentukan template
                s = time.perf_counter_ns()
                template_path = await self._resolve_template(
                    category=req.product_category,
                    template_name=req.template_name,
                    override=req.override_template_path,
                )
                steps["resolve_template_ms"] = _duration_ms(s)

                # 2) Placeholder + context
                s = time.perf_counter_ns()
                placeholders = await self._get_placeholders(template_path)
                proposal_ctx = ProposalContext(**req.context)
                render_ctx = proposal_ctx.to_render_context(self.settings)
                steps["prepare_context_ms"] = _duration_ms(s)

                # 3) Merge product summary jika diminta
                if req.product and req.category and req.tahun and req.summary_filename:
                    s = time.perf_counter_ns()
                    render_ctx = await self._merge_product_summary(
                        render_ctx,
                        product=req.product,
                        category=req.category,
                        tahun=req.tahun,
                        filename=req.summary_filename,
                        target_field=req.summary_target_field or "executive_summary",
                        mode=req.summary_mode or "append",
                    )
                    steps["merge_summary_ms"] = _duration_ms(s)

                # 4) Auto-fill via LLM (opsional)
                use_llm = (
                    req.use_llm_fill
                    if req.use_llm_fill is not None
                    else self.settings.use_llm_default
                )
                if use_llm and self._llm_chain is not None:
                    s = time.perf_counter_ns()
                    render_ctx = await self._fill_missing_with_llm(
                        render_ctx, placeholders, req.product_category
                    )
                    steps["llm_fill_ms"] = _duration_ms(s)

                # 5) Render & Save
                s = time.perf_counter_ns()
                output_dir = (
                    req.output_dir or self.settings.proposal_generated_base_path
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                filename_base = (
                    render_ctx.get("nama_pelanggan")
                    or render_ctx.get("judul_proposal")
                    or "proposal_tanpa_nama"
                )
                filename = _safe_filename(str(filename_base))
                out_path = output_dir / filename
                await self._render_and_save(template_path, render_ctx, out_path)
                steps["render_save_ms"] = _duration_ms(s)

                ended_at = _now_iso()
                result = PipelineResult(
                    status="success",
                    message="Dokumen berhasil dibuat",
                    data={
                        "output_path": str(out_path),
                        "filename": out_path.name,
                        "template_used": str(template_path),
                        "placeholders": placeholders,
                        "download_url": self._build_download_url(out_path.name),
                    },
                    timing={
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "duration_ms": _duration_ms(t0),
                        "steps_ms": steps,
                    },
                )
                logger.info(
                    "Generate OK | file=%s | dur=%.2fms",
                    out_path.name,
                    result.timing["duration_ms"],
                )
                return result.model_dump()

            except DocumentPipelineError as e:
                logger.warning("Generate gagal (pipeline): %s", e)
                return PipelineResult(
                    status="error",
                    message=str(e),
                    data={},
                    timing={
                        "started_at": started_at,
                        "ended_at": _now_iso(),
                        "duration_ms": _duration_ms(t0),
                        "steps_ms": steps,
                    },
                ).model_dump()

            except Exception as e:  # jaga-jaga
                logger.exception("Generate gagal (unhandled)")
                return PipelineResult(
                    status="error",
                    message=f"Terjadi kesalahan tak terduga: {e}",
                    data={},
                    timing={
                        "started_at": started_at,
                        "ended_at": _now_iso(),
                        "duration_ms": _duration_ms(t0),
                        "steps_ms": steps,
                    },
                ).model_dump()

    # -------------------------------------------------------------
    #                 PUBLIC API (Administration)
    # -------------------------------------------------------------
    async def list_templates(self) -> Dict[str, Any]:
        try:
            files = await asyncio.to_thread(
                lambda: sorted(
                    [
                        p.name
                        for p in self.settings.proposal_template_base_path.glob(
                            "*.docx"
                        )
                    ]
                )
            )
            return {
                "status": "success",
                "message": f"{len(files)} template ditemukan",
                "data": {"templates": files},
                "timing": {},
            }
        except Exception as e:
            logger.exception("List templates gagal")
            return {
                "status": "error",
                "message": f"List templates gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def add_template(self, name: str, source_path: Path) -> Dict[str, Any]:
        try:
            dest = self.settings.proposal_template_base_path / name

            def _copy():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(source_path, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())

            await asyncio.to_thread(_copy)
            return {
                "status": "success",
                "message": "Template ditambahkan",
                "data": {"path": str(dest)},
                "timing": {},
            }
        except Exception as e:
            logger.exception("Add template gagal")
            return {
                "status": "error",
                "message": f"Add template gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def update_template(self, name: str, source_path: Path) -> Dict[str, Any]:
        try:
            dest = self.settings.proposal_template_base_path / name
            if not dest.exists():
                raise TemplateNotFoundError(f"Template {name} tidak ditemukan")

            def _overwrite():
                with open(source_path, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())

            await asyncio.to_thread(_overwrite)
            return {
                "status": "success",
                "message": "Template diperbarui",
                "data": {"path": str(dest)},
                "timing": {},
            }
        except DocumentPipelineError as e:
            return {"status": "error", "message": str(e), "data": {}, "timing": {}}
        except Exception as e:
            logger.exception("Update template gagal")
            return {
                "status": "error",
                "message": f"Update template gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def delete_template(self, name: str) -> Dict[str, Any]:
        try:
            target = self.settings.proposal_template_base_path / name
            if not target.exists():
                raise TemplateNotFoundError(f"Template {name} tidak ditemukan")
            await asyncio.to_thread(target.unlink)
            return {
                "status": "success",
                "message": "Template dihapus",
                "data": {"name": name},
                "timing": {},
            }
        except DocumentPipelineError as e:
            return {"status": "error", "message": str(e), "data": {}, "timing": {}}
        except Exception as e:
            logger.exception("Delete template gagal")
            return {
                "status": "error",
                "message": f"Delete template gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def list_documents(self) -> Dict[str, Any]:
        try:
            files = await asyncio.to_thread(
                lambda: sorted(
                    [
                        p.name
                        for p in self.settings.proposal_generated_base_path.glob(
                            "*.docx"
                        )
                    ]
                )
            )
            return {
                "status": "success",
                "message": f"{len(files)} dokumen ditemukan",
                "data": {"documents": files},
                "timing": {},
            }
        except Exception as e:
            logger.exception("List documents gagal")
            return {
                "status": "error",
                "message": f"List documents gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def delete_document(self, name: str) -> Dict[str, Any]:
        try:
            target = self.settings.proposal_generated_base_path / name
            if not target.exists():
                raise SaveError(f"Dokumen {name} tidak ditemukan")
            await asyncio.to_thread(target.unlink)
            return {
                "status": "success",
                "message": "Dokumen dihapus",
                "data": {"name": name},
                "timing": {},
            }
        except DocumentPipelineError as e:
            return {"status": "error", "message": str(e), "data": {}, "timing": {}}
        except Exception as e:
            logger.exception("Delete document gagal")
            return {
                "status": "error",
                "message": f"Delete document gagal: {e}",
                "data": {},
                "timing": {},
            }

    async def preview_placeholders(
        self,
        *,
        category: str,
        template_name: Optional[str] = None,
        override: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Lihat daftar placeholder yang tersedia pada template yang akan dipakai.
        Prioritas pemilihan template: override > template_name > default map (by category).
        Tidak melakukan render dokumen.
        """
        t0 = time.perf_counter_ns()
        steps: Dict[str, float] = {}
        started_at = _now_iso()

        try:
            # 1) Resolve template
            s = time.perf_counter_ns()
            template_path = await self._resolve_template(
                category=category,
                template_name=template_name,
                override=override,
            )
            steps["resolve_template_ms"] = _duration_ms(s)

            # 2) Ambil placeholders
            s = time.perf_counter_ns()
            placeholders = await self._get_placeholders(template_path)
            steps["get_placeholders_ms"] = _duration_ms(s)

            ended_at = _now_iso()
            result = PipelineResult(
                status="success",
                message=f"{len(placeholders)} placeholder ditemukan",
                data={
                    "template_used": str(template_path),
                    "placeholders": placeholders,
                },
                timing={
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_ms": _duration_ms(t0),
                    "steps_ms": steps,
                },
            )
            # Log informatif (untuk debugging)
            logger.info(
                "Preview placeholders OK | template=%s | count=%d | dur=%.2fms",
                template_path.name,
                len(placeholders),
                result.timing["duration_ms"],
            )
            return result.model_dump()

        except DocumentPipelineError as e:
            logger.warning("Preview placeholders gagal (pipeline): %s", e)
            return PipelineResult(
                status="error",
                message=str(e),
                data={},
                timing={
                    "started_at": started_at,
                    "ended_at": _now_iso(),
                    "duration_ms": _duration_ms(t0),
                    "steps_ms": steps,
                },
            ).model_dump()

        except Exception as e:
            # Jaga-jaga agar error tak mematikan server
            logger.exception("Preview placeholders gagal (unhandled)")
            return PipelineResult(
                status="error",
                message=f"Terjadi kesalahan tak terduga: {e}",
                data={},
                timing={
                    "started_at": started_at,
                    "ended_at": _now_iso(),
                    "duration_ms": _duration_ms(t0),
                    "steps_ms": steps,
                },
            ).model_dump()

    # -------------------------------------------------------------
    #                 INTERNALS (Core Helpers)
    # -------------------------------------------------------------
    async def _resolve_template(
        self, category: str, template_name: Optional[str], override: Optional[Path]
    ) -> Path:
        # Prioritas: override > template_name > default map
        if override:
            p = Path(override)
            if not p.is_file():
                raise TemplateNotFoundError(f"Template override tidak ditemukan: {p}")
            logger.debug("Template pakai override: %s", p)
            return p
        if template_name:
            p = self.settings.proposal_template_base_path / template_name
            if not p.is_file():
                raise TemplateNotFoundError(
                    f"Template {template_name} tidak ditemukan di {self.settings.proposal_template_base_path}"
                )
            logger.debug("Template pakai nama spesifik: %s", p)
            return p
        tmpl = self.settings.default_template_map.get(category)
        if not tmpl:
            raise TemplateNotFoundError(
                f"Tidak ada template default untuk kategori '{category}'. Sediakan template_name atau override_template_path."
            )
        p = self.settings.proposal_template_base_path / tmpl
        if not p.is_file():
            raise TemplateNotFoundError(f"Template default tidak ditemukan: {p}")
        logger.debug("Template pakai default map: %s", p)
        return p

    async def _get_placeholders(self, template_path: Path) -> List[str]:
        def _load():
            tpl = DocxTemplate(str(template_path))
            return sorted(list(tpl.get_undeclared_template_variables()))

        try:
            placeholders: List[str] = await asyncio.to_thread(_load)
            logger.debug("Placeholders: %s", placeholders)
            return placeholders
        except Exception as e:
            logger.exception("Gagal membaca placeholder template")
            raise DocumentPipelineError(
                f"Gagal membaca placeholder dari {template_path}: {e}"
            )

    async def _merge_product_summary(
        self,
        render_ctx: Dict[str, Any],
        *,
        product: str,
        category: str,
        tahun: str,
        filename: str,
        target_field: str = "executive_summary",
        mode: Literal["append", "override", "prepend"] = "append",
    ) -> Dict[str, Any]:
        """Ambil konten summary produk (MD) dan gabungkan ke `render_ctx[target_field]`.

        - **append**   : tambahkan di akhir konten lama
        - **prepend**  : tambahkan di awal
        - **override** : ganti konten lama
        """
        try:
            # NB: open_product_summary membaca dari Settings.product_summaries_base_path
            app = AppSettings() if AppSettings else None  # type: ignore
            md_text = open_product_summary(
                app or self.settings, product, category, tahun, filename
            )
            old = str(render_ctx.get(target_field, "") or "")
            if mode == "override":
                new_val = md_text
            elif mode == "prepend":
                new_val = f"{md_text}\n\n{old}" if old else md_text
            else:
                new_val = f"{old}\n\n{md_text}" if old else md_text
            render_ctx[target_field] = new_val
            logger.debug("Merge product summary → field '%s' (%s)", target_field, mode)
            return render_ctx
        except Exception as e:
            # Jangan gagal total; cukup logging peringatan
            logger.warning(
                "Gagal merge product summary (%s/%s/%s/%s): %s",
                product,
                category,
                tahun,
                filename,
                e,
            )
            return render_ctx

    async def _fill_missing_with_llm(
        self, render_ctx: Dict[str, Any], placeholders: List[str], category: str
    ) -> Dict[str, Any]:
        if self._llm_chain is None:
            logger.info("Lewati LLM fill (LLMChain nonaktif)")
            return render_ctx
        string_candidates = [
            p
            for p in placeholders
            if not isinstance(render_ctx.get(p, ""), list)
            and (render_ctx.get(p) in (None, ""))
        ]
        if not string_candidates:
            return render_ctx
        to_fill = string_candidates[: self.settings.max_llm_fill]
        ctx_summary = {
            "judul_proposal": render_ctx.get("judul_proposal", ""),
            "nama_pelanggan": render_ctx.get("nama_pelanggan", ""),
            "ringkasan_kebutuhan": render_ctx.get("ringkasan_kebutuhan", ""),
            "ringkasan_manfaat": render_ctx.get("ringkasan_manfaat", ""),
            "executive_summary": render_ctx.get("executive_summary", ""),
        }
        instructions = (
            "Anda adalah asisten teknis presales. Lengkapi placeholder dokumen proposal "
            "secara ringkas, konkret, dan profesional dalam Bahasa Indonesia. "
            "Kembalikan hanya JSON valid (tanpa penjelasan)."
        )
        user_prompt = (
            "Isi narasi untuk placeholder berikut berdasarkan kategori produk dan konteks.\n"
            f"Kategori: {category}\n"
            f"Konteks ringkas: {json.dumps(ctx_summary, ensure_ascii=False)}\n"
            f"Placeholder: {json.dumps(to_fill, ensure_ascii=False)}\n\n"
            'Balas dengan JSON seperti: {"judul_proposal": "...", "executive_summary": "..."}'
        )
        try:
            msg = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_prompt},
            ]
            text = await self._llm_chain.generate_text(
                prefer="responses", messages_or_input=msg
            )  # type: ignore[union-attr]
            text_rslt = text.get("data")
            llm_json = _parse_json_loose(text_rslt) # type: ignore
            if not isinstance(llm_json, dict):
                raise LLMFillError("Respon LLM tidak berupa JSON object")
            for k, v in llm_json.items():
                if k in render_ctx and isinstance(v, str):
                    render_ctx[k] = v
            logger.debug("LLM filled keys: %s", list(llm_json.keys()))
            return render_ctx
        except Exception as e:
            logger.warning("LLM fill gagal: %s", e)
            return render_ctx

    async def _render_and_save(
        self, template_path: Path, context: Dict[str, Any], out_path: Path
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._process_pool,
                _render_docx_job,
                str(template_path),
                context,
                str(out_path),
            )
        except (AttributeError, RuntimeError) as e:
            logger.warning("ProcessPool gagal (%s). Fallback ke ThreadPool.", e)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 2) * 4)
            ) as tp:
                await loop.run_in_executor(
                    tp, _render_docx_job, str(template_path), context, str(out_path)
                )
        except Exception as e:
            logger.exception("Render/Save gagal")
            raise SaveError(f"Gagal render/simpan dokumen: {e}")

    def _build_download_url(self, filename: str) -> str:
        base = self.settings.public_download_base_url
        return (
            f"{base.rstrip('/')}/download/{filename}"
            if base
            else str((self.settings.proposal_generated_base_path / filename).absolute())
        )
