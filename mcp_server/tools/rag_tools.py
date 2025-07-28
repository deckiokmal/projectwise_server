# tools/rag_tools.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

from docling.document_converter import DocumentConverter

from mcp_server.utils.rag_pipeline import RAGPipeline
from mcp_server.utils.llm_chains import LLMChain
from mcp_server.utils.logger import logger
from mcp_server.utils.helper import (
    slugify,
    to_kak_markdown,
    to_product_markdown,
    clean_utf8,
    summarize_long_product_text,
)
from mcp_server.settings import Settings


class RAGTools:
    def __init__(self):
        """
        Inisialisasi RAGTools dengan RAGPipeline async-ready.
        """
        self.settings = Settings()  # type: ignore
        self.pipeline = RAGPipeline()
        self.llm = LLMChain()
        self.manifest_path = Path(self.settings.manifest_file_path)
        self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._manifest = {}
        else:
            self._manifest = {}

    def _save_manifest(self):
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(
                json.dumps(self._manifest, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Gagal menyimpan manifest: {e}")

    async def delete_vector_by_filename(self, filename: str) -> int:
        """
        Hapus entri vector berdasarkan nama file yang digunakan sebagai metadata.filename
        """
        try:
            deleted_rows = await self.pipeline.table.delete(
                f"metadata.filename = '{filename}'"
            )
            logger.info(f"{deleted_rows} entri vectorstore dihapus untuk: {filename}")
            return deleted_rows  # type: ignore
        except Exception as e:
            logger.error(
                f"Gagal menghapus vector metadata.filename = '{filename}': {e}"
            )
            return 0

    async def ingest_kak_tor_chunks(
        self,
        filename: str,
        pelanggan: str,
        project: str,
        tahun: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Proses satu file PDF menjadi Markdown dan index ke vectorstore jika belum pernah diingest.
        Struktur markdown dan metadata berbasis pelanggan/proyek/tahun.
        """
        pelanggan_slug = slugify(pelanggan)
        tahun_str = tahun or str(datetime.now().year)
        path = (
            Path(self.settings.kak_tor_base_path).expanduser().resolve()
            / pelanggan_slug
            / tahun_str
            / filename
        )
        if not path.exists():
            return {"status": "failure", "error": f"File tidak ditemukan: {filename}"}

        unique_key = f"{pelanggan_slug}_{tahun_str}_{path.name}"

        if unique_key in self._manifest and not overwrite:
            logger.info(f"File '{unique_key}' sudah pernah diingest. Skip.")
            return {"status": "skipped", "reason": "sudah pernah diingest"}

        try:
            # 1. Konversi PDF ke document
            result = DocumentConverter().convert(source=str(path))

            # 2. Chunk dan embed
            chunks = await self.pipeline._chunk_document_meta_kak(
                dl_doc=result.document,
                filename=path.name,
                project=project,
                pelanggan=pelanggan,
                tahun=tahun_str,
            )

            # Hapus entri lama jika overwrite
            if overwrite:
                total_before = await self.pipeline.table.count_rows()
                await self.delete_vector_by_filename(filename)
                total_after = await self.pipeline.table.count_rows()
                logger.info(
                    f"Overwrite aktif. Data lama dihapus: {total_before - total_after} row."
                )

            # Tambahkan ke vectorstore
            try:
                logger.debug(
                    f"Contoh metadata sebelum add:\n{json.dumps(chunks[0]['metadata'], indent=2)}"
                )
                await self.pipeline.safe_add(chunks, context_name=filename)
                logger.info(f"Berhasil menambahkan {len(chunks)} chunk ke vectorstore.")
            except Exception as e:
                logger.error(f"Gagal menambahkan chunk ke vectorstore: {e}")

            # 3. Export ke Markdown
            md_base = (
                Path(self.settings.kak_tor_md_base_path) / pelanggan_slug / tahun_str
            )
            md_base.mkdir(parents=True, exist_ok=True)
            md_file = md_base / f"{path.stem}.md"
            md_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
            logger.info(f"Markdown disimpan ke: {md_file}")

            # 4. Update manifest
            self._manifest[unique_key] = True
            self._save_manifest()

            return {
                "status": "success",
                "chunks": len(chunks),
                "markdown_file": str(md_file),
            }
        except Exception as e:
            logger.error(f"Gagal ingest file '{filename}': {e}")
            return {"status": "failure", "error": str(e)}

    async def ingest_product_knowledge_chunks(
        self,
        filename: str,
        product_name: str,
        category: str,
        tahun: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Proses satu file PDF produk berdasarkan filename dan index ke vectorstore jika belum pernah diingest.
        buat dan simpan markdown.
        """
        # Struktur folder: category/<tahun>/<product_name>
        category_slug = slugify(category)
        tahun_str = str(tahun or datetime.now().year)
        product_slug = slugify(product_name)
        path = (
            Path(self.settings.product_base_path)
            / category_slug
            / tahun_str
            / product_slug
            / filename
        )
        if not path.exists():
            return {
                "status": "failure",
                "error": f"File produk tidak ditemukan: {path.name}",
            }

        unique_key = f"{category_slug}_{tahun_str}__{product_slug}_{filename}"

        if unique_key in self._manifest and not overwrite:
            logger.info(f"File '{unique_key}' sudah pernah diingest. Skip.")
            return {"status": "skipped", "reason": "sudah pernah diingest"}

        try:
            # 1. Konversi PDF ke document
            result = DocumentConverter().convert(source=str(path))

            # 2. Chunk dan embed
            chunks = await self.pipeline._chunk_document_meta_prod(
                dl_doc=result.document,
                filename=path.name,
                product=product_name,
                category=category,
                tahun=tahun_str,
            )

            # Hapus entri lama jika overwrite
            if overwrite:
                total_before = await self.pipeline.table.count_rows()
                await self.delete_vector_by_filename(filename)
                total_after = await self.pipeline.table.count_rows()
                logger.info(
                    f"Overwrite aktif. Data lama dihapus: {total_before - total_after} row."
                )

            # Tambahkan ke vectorstore
            try:
                await self.pipeline.safe_add(chunks, context_name=filename)
                logger.info(f"Berhasil menambahkan {len(chunks)} chunk ke vectorstore.")
            except Exception as e:
                logger.error(f"Gagal menambahkan chunk ke vectorstore: {e}")

            # 3. Export ke Markdown
            md_base = (
                Path(self.settings.product_base_path)
                / category_slug
                / tahun_str
                / product_slug
            )
            md_file = md_base / f"{path.stem}.md"
            md_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
            logger.info(f"Markdown product disimpan ke: {md_file}")

            # 4. Update manifest
            self._manifest[unique_key] = True
            self._save_manifest()

            return {
                "status": "success",
                "chunks": len(chunks),
                "markdown_file": str(md_file),
            }
        except Exception as e:
            logger.error(f"Gagal ingest file produk '{filename}': {e}")
            return {"status": "failure", "error": str(e)}

    async def build_summary_kak_payload_and_summarize(
        self,
        kak_tor_name: str,
        pelanggan: str,
        project: str,
        tahun: str | None = None,
    ) -> Dict[str, str]:
        """
        Gabungkan template + isi markdown dan hasilkan ringkasan via LLMChain,
        lalu simpan dalam file Markdown dan index ke vectorstore sebagai potongan (chunks) yang valid.
        """
        # Instruction template
        tmpl_path = Path(self.settings.summaries_instruction_path)
        logger.debug(f"Template path: {tmpl_path}")
        if not tmpl_path.is_file():
            logger.error(f"Template statis tidak ditemukan: {tmpl_path}")
            raise FileNotFoundError(f"Template statis tidak ditemukan: {tmpl_path}")
        instruction = tmpl_path.read_text(encoding="utf-8", errors="replace")

        # Check parameter
        if not pelanggan:
            raise ValueError("Parameter 'pelanggan' wajib diisi.")
        if not kak_tor_name:
            raise ValueError("Parameter 'kak_tor_name' wajib diisi.")
        if not project:
            raise ValueError("Parameter 'project' wajib diisi.")

        # Baca file Markdown KAK/TOR
        # Struktur folder: pelanggan/<tahun>/<kak_tor_name>
        pelanggan_slug = slugify(pelanggan)
        project_slug = slugify(project)
        tahun_str = tahun or str(datetime.now().year)

        md_base = (
            Path(self.settings.kak_tor_md_base_path).expanduser().resolve()
            / pelanggan_slug
            / tahun_str
        )

        if not md_base.is_dir():
            raise FileNotFoundError(f"Folder Markdown tidak ditemukan: {md_base}")

        md_path = md_base / kak_tor_name
        if not md_path.is_file():
            available = [p.name for p in md_base.glob("*.md")]
            raise FileNotFoundError(
                f"File '{kak_tor_name}' tidak ditemukan di {md_base}.\n"
                f"Pilihan yang tersedia: {available}"
            )

        md_text = md_path.read_text(encoding="utf-8")
        full_input = f"<kak_tor>\n{md_text.strip()}\n</kak_tor>"

        # Proses ringkasan
        try:
            # # Panggil LLMChain untuk ringkasan
            summary = await summarize_long_product_text(
                llm=self.llm,
                full_text=full_input,
                instruction=instruction.strip(),
                model_name=self.llm.model,
                max_tokens=self.settings.max_token,
            )

            if not summary.strip():
                raise ValueError("Ringkasan kosong, tidak bisa diproses.")

            if summary.strip().startswith("[Ringkasan gagal"):
                logger.error("LLM gagal merangkum. Ringkasan tidak akan disimpan.")
                return {
                    "summary": summary,
                    "message": "LLM gagal membuat ringkasan yang valid.",
                }

            # Simpan sebagai file Markdown
            summaries_base = (
                Path(self.settings.kak_tor_summaries_base_path)
                / pelanggan_slug
                / tahun_str
            )

            summaries_base.mkdir(parents=True, exist_ok=True)

            filename = f"{project_slug}_summary.md"
            md_file = summaries_base / filename
            markdown_text = to_kak_markdown(summary)

            # Check apakah file sudah ada
            overwritten = md_file.exists()
            try:
                md_file.write_text(markdown_text, encoding="utf-8")
            except Exception as e:
                logger.error(f"Gagal menulis file Markdown: {e}")
                return {"summary": f"[Gagal menyimpan ringkasan ke file]: {e}"}

            if overwritten:
                deleted_rows = await self.delete_vector_by_filename(filename)
                logger.warning(
                    f"Ringkasan lama ditimpa: {md_file}\nDeleted {deleted_rows} baris dengan metadata.filename = '{filename}'"
                )
                overwrite_message = f"File ringkasan telah ditimpa: {md_file}"
            else:
                logger.info(f"Summary disimpan ke file: {md_file}")
                overwrite_message = "Ringkasan baru berhasil disimpan."

            # Konversi file markdown ke DoclingDocument
            result = DocumentConverter().convert(source=str(md_file))

            # Chunk dan embed menggunakan pipeline yang validasi vektor
            chunks = await self.pipeline._chunk_document_meta_kak(
                dl_doc=result.document,
                filename=filename,
                project=project,
                pelanggan=pelanggan,
                tahun=tahun_str,
            )

            # Tambahkan ke vectorstore
            try:
                await self.pipeline.table.add(chunks)
                logger.info(
                    f"Ringkasan dimasukkan ke vectorstore sebagai {len(chunks)} chunk."
                )
            except Exception as e:
                logger.error(f"Gagal menambahkan ringkasan ke vectorstore: {e}")
                return {"summary": f"[Gagal menyimpan ringkasan ke vectorstore]: {e}"}

            return {
                "summary": summary,
                "summary_file": str(md_file),
                "message": overwrite_message,
            }

        except Exception as e:
            logger.error(f"Gagal merangkum atau menyimpan KAK/TOR: {e}")
            return {"summary": f"[Gagal menjalankan LLM]: {e}"}

    async def build_summary_product_payload_and_summarize(
        self,
        filename: str,
        product_name: str,
        category: str,
        tahun: str,
    ) -> Dict[str, str]:
        """
        Gabungkan template + isi markdown dan hasilkan ringkasan via LLMChain,
        lalu simpan dalam file Markdown dan index ke vectorstore sebagai potongan (chunks) yang valid.
        """
        # Instruction template
        tmpl_path = Path(self.settings.product_summaries_instruction_path)
        if not tmpl_path.is_file():
            raise FileNotFoundError(f"Template statis tidak ditemukan: {tmpl_path}")
        logger.info(
            f"Info dari build_summary_product_payload_and_summarize : Template path: {tmpl_path}"
        )
        instruction = tmpl_path.read_text(encoding="utf-8")

        # Check parameter
        if not filename:
            raise ValueError("Parameter 'filename' wajib diisi.")
        if not product_name:
            raise ValueError("Parameter 'product_name' wajib diisi.")
        if not category:
            raise ValueError("Parameter 'category' wajib diisi.")

        # Baca file Markdown produk
        # Struktur folder: category/<tahun>/<product_name>
        md_base = (
            Path(self.settings.product_base_path) / category / tahun / product_name
        )

        if not md_base.is_dir():
            raise FileNotFoundError(f"Folder Markdown tidak ditemukan: {md_base}")

        md_path = md_base / filename
        if not md_path.is_file():
            available = [p.name for p in md_base.glob("*.md")]
            logger.info(f"Available files: {available}")
            raise FileNotFoundError(
                f"File '{product_name}.md' tidak ditemukan di {md_base}.\n"
                f"Pilihan yang tersedia: {available}"
            )

        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        full_input = f"<product>\n{md_text.strip()}\n</product>"

        # Untuk debugging, tampilkan input awal 200 karakter awal
        logger.info(f"Input untuk LLM:\n{full_input}...")
        logger.info(f"Input untuk LLM:\n{instruction.strip()[:200]}...")

        # Proses ringkasan
        try:
            # Panggil LLMChain untuk ringkasan
            summary = await summarize_long_product_text(
                llm=self.llm,
                full_text=full_input,
                instruction=instruction.strip(),
                model_name=self.llm.model,
                max_tokens=self.settings.max_token,
            )

            # Untuk debugging, tampilkan ringkasan awal
            logger.info(f"Ringkasan LLM: {repr(summary[:200])}")

            if not summary.strip():
                raise ValueError("Ringkasan kosong, tidak bisa diproses.")

            if summary.strip().startswith("[Ringkasan gagal"):
                logger.error("LLM gagal merangkum. Ringkasan tidak akan disimpan.")
                return {
                    "summary": summary,
                    "message": "LLM gagal membuat ringkasan yang valid.",
                }

            # Simpan sebagai file Markdown
            product_slug = slugify(product_name)
            summaries_base = (
                Path(self.settings.product_base_path) / category / tahun / product_slug
            )

            summaries_base.mkdir(parents=True, exist_ok=True)

            # overwrite nama file agar ringkasan disimpan sebagai '<product_name>_summary.md'
            filename = f"{product_name}_summary.md"
            md_file = summaries_base / filename
            markdown_text = to_product_markdown(summary)

            # Check apakah file sudah ada
            overwritten = md_file.exists()

            # markdown_text = clean_utf8(markdown_text)
            # md_file.write_text(markdown_text, encoding="utf-8")

            try:
                md_file.write_text(markdown_text, encoding="utf-8")
            except UnicodeEncodeError:
                logger.warning("Gagal simpan markdown asli, fallback ke cleaned utf-8.")
                cleaned = clean_utf8(markdown_text)

                md_file.write_text(cleaned, encoding="utf-8")

            if overwritten:
                deleted_rows = await self.delete_vector_by_filename(filename)
                logger.warning(
                    f"Ringkasan lama ditimpa: {md_file}\nDeleted {deleted_rows} baris dengan metadata.filename = '{filename}'"
                )
                overwrite_message = f"File ringkasan telah ditimpa: {md_file}"
            else:
                logger.info(f"Summary disimpan ke file: {md_file}")
                overwrite_message = "Ringkasan baru berhasil disimpan."

            # Konversi file markdown ke DoclingDocument
            result = DocumentConverter().convert(source=str(md_file))
            if not result or not result.document:
                raise ValueError(f"Gagal parsing Markdown dari {md_file}")

            # Chunk dan embed menggunakan pipeline yang validasi vektor
            chunks = await self.pipeline._chunk_document_meta_prod(
                dl_doc=result.document,
                filename=filename,
                product=product_name,
                category=category,
                tahun=tahun,
            )

            # Tambahkan ke vectorstore
            try:
                await self.pipeline.table.add(chunks)
                logger.info(
                    f"Ringkasan dimasukkan ke vectorstore sebagai {len(chunks)} chunk."
                )
            except Exception as e:
                logger.error(f"Gagal menambahkan ringkasan ke vectorstore: {e}")
                return {"summary": f"[Gagal menyimpan ringkasan ke vectorstore]: {e}"}

            logger.debug(f"Summary Preview: {repr(summary[:200])}")
            return {
                "summary": summary,
                "summary_file": str(md_file),
                "message": overwrite_message,
            }

        except Exception as e:
            logger.error(f"Gagal merangkum atau menyimpan produk: {e}")
            return {"summary": f"[Gagal menjalankan LLM]: {e}"}
