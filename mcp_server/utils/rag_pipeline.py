# utils/rag_pipeline.py
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb
from lancedb.pydantic import LanceModel, Vector
from mcp_server.utils.schemas import ChunkMetadata

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from mcp_server.settings import Settings
from mcp_server.utils.logger import logger


# ──────────────────────────────────────────────────────────────
# Skema untuk menyimpan metadata tiap chunk dokumen
# ──────────────────────────────────────────────────────────────
def build_chunks_schema(vector_dim: int):
    class Chunks(LanceModel):
        text: str
        vector: Vector(vector_dim)  # type: ignore
        metadata: ChunkMetadata

    return Chunks


# ──────────────────────────────────────────────────────────────
# RAGPipeline Async — mendukung chunking, indexing, dan retrieval berbasis LanceDB
# ──────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        """
        Inisialisasi pipeline RAG:
        - Membaca konfigurasi dari Settings.
        - Mengatur embedding model.
        - Membuat koneksi ke LanceDB.
        """
        self.settings = Settings()  # type: ignore
        self.product_base_path = Path(self.settings.product_base_path)
        self.collection_name = self.settings.collection_name

        # Pilih jenis embedding berdasarkan konfigurasi
        if self.settings.embedding_model.startswith("text-embedding"):
            self.embed = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,  # type: ignore
            )
            logger.info("Menggunakan OpenAI Async Embeddings.")
        else:
            self.embed = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_host,
            )
            logger.info("Menggunakan Ollama Embeddings.")

        self.vector_dim: int = self.settings.vector_dim
        self._run_sync(self.db_connect())

    # LanceDB utility methods
    async def db_connect(self):
        # Hitung dimensi vektor dari contoh input
        example_vector = await self.embed.aembed_query("dimension_check")
        self.vector_dim = len(example_vector)
        Chunks = build_chunks_schema(self.vector_dim)

        # Koneksi ke LanceDB
        if self.settings.db_connection == "s3":
            self.db = await lancedb.connect_async(
                self.settings.vector_store_path,
                storage_options={
                    "aws_access_key_id": self.settings.aws_access_key_id,
                },
            )
        elif self.settings.db_connection == "cloud":
            self.db = await lancedb.connect_async(
                self.settings.vector_store_path,
                api_key=self.settings.cloud_api_key,
                client_config={"retry_config": {"retries": 5}},
            )
        else:
            # Default ke local
            self.db = await lancedb.connect_async(self.settings.vector_store_path)
            logger.info(
                f"Menggunakan local vector store di {self.settings.vector_store_path}"
            )

        # Buat tabel jika belum ada
        try:
            self.table = await self.db.open_table(self.collection_name)
            _ = await self.table.count_rows()  # validasi tabel ada
        except Exception:
            logger.info("Tabel tidak ditemukan, membuat baru...")
            self.table = await self.db.create_table(
                self.collection_name, schema=Chunks, mode="overwrite"
            )
        logger.info(
            f"RAGPipeline siap (dim={self.vector_dim}, koleksi='{await self.db.table_names()}')."
        )

    def _run_sync(self, coro):
        """Utility internal untuk menjalankan async function secara sinkron di init."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(coro)

    def _validate_vector_dim(self, vector: List[float]):
        """
        Validasi panjang vektor embedding.
        """
        if len(vector) != self.vector_dim:
            logger.debug(f"Ukuran vector: {len(vector)} | target: {self.vector_dim}")
            raise ValueError(
                f"Dimensi vektor salah: {len(vector)} != {self.vector_dim}"
            )

    def _sanitize(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitasi metadata:
        - Hilangkan spasi
        - Ubah semua string ke lowercase
        - Angka tetap dipertahankan (chunk_index)
        - Kosong atau None → None
        """
        clean = {}
        for k in [
            "filename",
            "source",
            "chunk_index",
            "pelanggan",
            "category",
            "product",
            "tahun",
            "project",
        ]:
            val = meta.get(k, None)

            if k == "chunk_index":
                try:
                    clean[k] = int(val) if val not in (None, "", "null") else None
                except Exception:
                    clean[k] = None
            else:
                if isinstance(val, str) and val.strip():
                    clean[k] = val.strip().lower()
                else:
                    clean[k] = None

        return clean

    def to_vector_entry(
        self,
        text: str,
        vector: List[float],
        metadata_raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Bangun 1 vector entry LanceDB dengan metadata yang benar dan tidak kosong seluruhnya.
        """

        # Konversi ke LanceModel → lalu ke dict
        meta_obj = ChunkMetadata(**self._sanitize(metadata_raw))

        return {
            "text": text,
            "vector": list(vector),
            "metadata": meta_obj.model_dump(),  # HARUS dict, bukan LanceModel
        }

    async def safe_add(
        self, chunks: List[Dict[str, Any]], context_name: str = "unknown"
    ) -> bool:
        """
        Tambahkan chunks ke vectorstore dengan validasi jumlah sebelum & sesudah.

        Args:
            chunks: List[Dict] dengan field text, vector, metadata
            context_name: Nama konteks (filename/proyek) untuk log

        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            before = await self.table.count_rows()
            await self.table.add(chunks)
            after = await self.table.count_rows()
            added = after - before

            if added != len(chunks):
                logger.warning(
                    f"Jumlah chunk ditambahkan tidak sesuai: "
                    f"expected={len(chunks)}, actual={added}"
                )
            else:
                logger.info(
                    f"{added} chunk berhasil ditambahkan untuk '{context_name}'."
                )

            return added > 0

        except Exception as e:
            logger.error(f"Gagal menambahkan chunks untuk '{context_name}': {e}")
            return False

    async def _chunk_document_meta_kak(
        self, dl_doc, filename: str, project: str, pelanggan: str, tahun: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk dokumen KAK/TOR, hitung embedding setiap chunk, dan siapkan metadata.
        """
        chunker = HybridChunker(merge_peers=True)
        chunks = chunker.chunk(dl_doc=dl_doc)

        entries = []
        for idx, chunk in enumerate(chunks):
            text = chunk.text
            vec = await self.embed.aembed_query(text)
            self._validate_vector_dim(vec)

            # Build metadata sebagai LanceModel
            raw_meta = {
                "filename": filename,
                "source": filename,
                "chunk_index": idx,
                "pelanggan": pelanggan,
                "project": project,
                "tahun": tahun,
            }

            entries.append(
                self.to_vector_entry(
                    text=chunk.text,
                    vector=vec,
                    metadata_raw=raw_meta,
                )
            )

        return entries

    async def _chunk_document_meta_prod(
        self, dl_doc, filename: str, product: str, category: str, tahun: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk dokumen product, hitung embedding setiap chunk, dan siapkan metadata.
        """
        chunker = HybridChunker(merge_peers=True)
        chunks = chunker.chunk(dl_doc=dl_doc)

        entries = []
        for idx, chunk in enumerate(chunks):
            text = chunk.text
            vec = await self.embed.aembed_query(text)
            self._validate_vector_dim(vec)

            # Build metadata sebagai LanceModel
            raw_meta = {
                "filename": filename,
                "source": filename,
                "chunk_index": idx,
                "product": product,
                "category": category,
                "tahun": tahun,
            }

            entries.append(
                self.to_vector_entry(
                    text=chunk.text,
                    vector=vec,
                    metadata_raw=raw_meta,
                )
            )

        return entries

    async def retrieval(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Melakukan retrieval dokumen relevan berdasarkan query dan filter metadata.
        Mengembalikan hasil dalam bentuk list of dict {text, metadata, citation}

        Args:
            query: pertanyaan user.
            k: jumlah top-k yang ingin diambil.
            metadata_filter: dictionary untuk memfilter metadata (project, tahun, dst).

        Returns:
            List[Dict]: hasil retrieval
        """
        try:
            if not query or not query.strip():
                logger.warning("Query kosong.")
                return [{"error": "Query kosong. Masukkan pertanyaan yang jelas."}]

            top_k = k or self.settings.retriever_search_k

            start = time.perf_counter()  # Start timing
            q_vec = await self.embed.aembed_query(query)
            self._validate_vector_dim(q_vec)

            builder = await self.table.search(list(q_vec))

            # ─ Filter metadata jika diberikan ─
            ALLOWED_FIELDS = {
                "filename",
                "source",
                "chunk_index",
                "pelanggan",
                "category",
                "product",
                "tahun",
                "project",
            }

            # Validasi filter metadata
            if metadata_filter:
                if not isinstance(metadata_filter, dict):
                    return [{"error": "Metadata filter harus dictionary."}]
                clauses = []
                for field, val in metadata_filter.items():
                    if field not in ALLOWED_FIELDS:
                        logger.warning(f"Metadata field tidak valid: {field}")
                        continue
                    key = f"metadata.{field}"
                    if val is None:
                        continue
                    elif isinstance(val, (list, tuple)):
                        items = ", ".join(f"'{v}'" for v in val)
                        clauses.append(f"{key} IN ({items})")
                    else:
                        val_str = str(val).strip()
                        if val_str:
                            clauses.append(f"{key} = '{val_str}'")
                if clauses:
                    builder = builder.where(" AND ".join(clauses))
                    logger.info(f"Filter metadata diterapkan: {' AND '.join(clauses)}")
                else:
                    logger.warning("Filter metadata kosong atau tidak valid, dilewati.")

            # ─ Ambil hasil ─
            df = await builder.limit(top_k).to_pandas()
            if df.empty:
                total = await self.table.count_rows()
                logger.info(
                    f"Tidak ditemukan hasil untuk '{query}'. Total chunk: {total}"
                )
                return []

            query_time = time.perf_counter() - start  # Selesai timing

            results = []
            for _, row in df.iterrows():
                meta = row.get("metadata")
                if not meta:
                    logger.warning("Chunk dengan metadata NULL/kosong dilewati.")
                    continue

                citation = (
                    f"[{meta.get('filename', '')} - {meta.get('pelanggan', '')} - "
                    f"{meta.get('category', '')} - {meta.get('project', '')} - "
                    f"{meta.get('product', '')} - {meta.get('tahun', '')}]"
                )

                results.append(
                    {
                        "text": row["text"],
                        "metadata": meta,
                        "citation": citation,
                        "score": float(
                            row.get("score", 0.0)
                        ),  # LanceDB menyediakan score
                        "query_time": query_time,  # Waktu query
                    }
                )

            logger.info(
                f"Ditemukan {len(results)} hasil retrieval untuk query: '{query}' "
                f"(waktu: {query_time:.2f}s)"
            )
            for i, r in enumerate(results[:k]):  # Log top-k results
                logger.debug(f"[{i + 1}] Score={r['score']:.4f} | {r['citation']}")

            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [{"error": f"Retrieval gagal: {str(e)}"}]

    async def reset_vector_database(self) -> Dict[str, Any]:
        """
        Reset ulang LanceDB vectorstore, menghapus data dan reinitialize ulang koneksi DB dan tabel.
        """
        try:
            vector_path = Path(self.settings.vector_store_path)

            # 1. Hapus folder vector store
            if vector_path.exists() and vector_path.is_dir():
                shutil.rmtree(vector_path)
                logger.info(f"Direktori vector store '{vector_path}' telah dihapus.")
            else:
                logger.warning(f"Direktori tidak ditemukan: {vector_path}")

            # 2. Hapus file manifest dan status
            manifest_file = Path("mcp_server/data/ingested_manifest.json")
            status_file = Path("mcp_server/data/ingestion_status.json")
            for f in [manifest_file, status_file]:
                if f.exists():
                    f.unlink()
                    logger.info(f"File '{f}' telah dihapus.")

            # 3. Inisialisasi ulang koneksi LanceDB
            logger.info("Menginisialisasi ulang koneksi LanceDB...")
            await self.db_connect()

            # 4. Validasi koneksi baru
            try:
                row_count = await self.table.count_rows()
                logger.info(
                    f"Vectorstore berhasil di-reset dan siap. Baris: {row_count}"
                )
            except Exception as inner_e:
                logger.warning(
                    f"Tabel berhasil dibuat tapi count_rows gagal: {inner_e}"
                )

            return {
                "status": "success",
                "message": "Vector DB berhasil di-reset dan diinisialisasi ulang.",
            }

        except Exception as e:
            logger.error(f"Reset vector database gagal: {e}")
            return {"status": "error", "message": str(e)}

    async def list_available_metadata(self) -> List[Dict[str, Any]]:
        """List metadata entries from the vectorstore.

        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries.
        """
        try:
            builder = await self.table.search([0.0] * self.vector_dim)
            builder = builder.where("metadata IS NOT NULL")
            df = await builder.limit(5000).to_pandas()
            result = [row["metadata"] for _, row in df.iterrows()]
            return result
        except Exception as e:
            logger.error(f"Gagal mengambil metadata dari vectorstore: {e}")
            return []
