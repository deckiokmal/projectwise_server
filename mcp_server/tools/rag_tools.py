# tools/rag_tools.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

from docling.document_converter import DocumentConverter

from mcp_server.utils.rag_pipeline import RAGPipeline
from mcp_server.utils.rag_pipeline import ChunkMetadata
from mcp_server.utils.llm_chains import LLMChain
from mcp_server.utils.logger import logger
from mcp_server.utils.helper import slugify, to_markdown
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
        buat summaries sizing products calculated dan simpan markdown.
        ingest markdown ke vectorstore sebagai 1 chunk utuh.
        """
        try:
            # Struktur folder: category/<tahun>/<product_name>
            category_slug = slugify(category)
            product_slug = slugify(product_name)
            tahun_str = str(tahun or datetime.now().year)
            md_base = (
                Path(self.settings.product_base_path)
                / category_slug
                / tahun_str
                / product_slug
            )
            md_base.mkdir(parents=True, exist_ok=True)

            # Lokasi file PDF diasumsikan sudah ada di folder tujuan
            pdf_path = md_base / f"{product_slug}.pdf"
            if not pdf_path.exists():
                return {
                    "status": "failure",
                    "error": f"File PDF tidak ditemukan: {pdf_path}",
                }

            unique_key = f"{category_slug}_{tahun_str}__{product_slug}"
            if unique_key in self._manifest and not overwrite:
                logger.info(
                    f"File '{unique_key}' sudah pernah diingest. Skip. Lokasi: {pdf_path}"
                )
                return {"status": "skipped", "reason": "sudah pernah diingest"}

            result = DocumentConverter().convert(source=str(pdf_path))
            chunks = await self.pipeline._chunk_document_meta_prod(
                result.document,
                filename=pdf_path.name,
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
                await self.pipeline.table.add(chunks)
                logger.info(f"Berhasil menambahkan {len(chunks)} chunk ke vectorstore.")
            except Exception as e:
                logger.error(f"Gagal menambahkan chunk ke vectorstore: {e}")

            # Simpan markdown
            md_file = md_base / f"{pdf_path.stem}.md"
            if md_file.exists():
                if overwrite:
                    md_file.unlink()
                    logger.warning(f"Markdown lama dihapus dan ditimpa: {md_file}")
                else:
                    logger.warning(f"Markdown sudah ada dan akan ditimpa: {md_file}")
            md_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
            logger.info(f"Markdown disimpan ke: {md_file}")

            self._manifest[unique_key] = True
            self._save_manifest()

            try:
                # Jalankan LLM untuk ringkasan sizing products calculated
                summary = await self.llm.generate_text(
                    input=result.document.export_to_markdown(),
                    instructions="Buatkan ringkasan sizing product berdasarkan dokumen berikut.",
                )
                summary_path = md_base / f"{pdf_path.stem}_summary.md"

                if summary_path.exists():
                    if overwrite:
                        summary_path.unlink()
                        logger.warning(
                            f"Ringkasan lama dihapus dan ditimpa: {summary_path}"
                        )
                    else:
                        logger.warning(
                            f"Ringkasan sudah ada dan akan ditimpa: {summary_path}"
                        )

                summary_path.write_text(summary, encoding="utf-8")

                # Ingest sebagai 1 chunk utuh
                entry = {
                    "text": summary,
                    "vector": list(await self.pipeline.embed.aembed_query(summary)),
                    "metadata": {
                        "filename": summary_path.name,
                        "source": summary_path.name,
                        "chunk_index": 0,
                        "product": product_name,
                        "category": category,
                        "tahun": tahun_str,
                    },
                }

                # Validasi (opsional, untuk debug atau hard check)
                for i, entry in enumerate(chunks):
                    if not isinstance(entry["metadata"], ChunkMetadata):
                        logger.error(
                            f"Chunk ke-{i} memiliki metadata invalid: {entry['metadata']}"
                        )
                        raise ValueError(
                            f"Metadata bukan ChunkMetadata di chunk ke-{i}"
                        )

                # Tambahkan ringkasan ke vectorstore
                try:
                    await self.pipeline.table.add([entry])
                    logger.info(
                        f"Berhasil menambahkan ringkasan sizing ke vectorstore: {summary_path}"
                    )
                except Exception as e:
                    logger.error(f"Gagal menambahkan ringkasan ke vectorstore: {e}")
                    return {"status": "failure", "error": str(e)}

            except Exception as e:
                logger.error(f"Gagal merangkum dokumen produk: {e}")

            return {
                "status": "success",
                "chunks": len(chunks),
                "pdf_file": str(pdf_path),
                "markdown_file": str(md_file),
            }
        except Exception as e:
            logger.error(f"Gagal proses PDF produk ({filename}): {e}")
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
        tmpl_path = Path(self.settings.summaries_instruction_path)
        if not tmpl_path.is_file():
            raise FileNotFoundError(f"Template statis tidak ditemukan: {tmpl_path}")
        instruction = tmpl_path.read_text(encoding="utf-8")

        if not pelanggan:
            raise ValueError("Parameter 'pelanggan' wajib diisi.")
        if not kak_tor_name:
            raise ValueError("Parameter 'kak_tor_name' wajib diisi.")
        if not project:
            raise ValueError("Parameter 'project' wajib diisi.")

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

        try:
            # Panggil LLMChain untuk ringkasan
            summary = await self.llm.generate_text(
                input=full_input, instructions=instruction.strip()
            )

            if not summary.strip():
                raise ValueError("Ringkasan kosong, tidak bisa diproses.")

            # Simpan sebagai file Markdown
            summaries_base = (
                Path(self.settings.kak_tor_summaries_base_path)
                / pelanggan_slug
                / tahun_str
            )
            summaries_base.mkdir(parents=True, exist_ok=True)

            filename = f"{project_slug}_summary.md"
            md_file = summaries_base / filename
            markdown_text = to_markdown(summary)

            overwritten = md_file.exists()
            md_file.write_text(markdown_text, encoding="utf-8")

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
