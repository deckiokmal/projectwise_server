# mcp_server/utils/rag_pipeline_util.py
from __future__ import annotations

import uuid
import asyncio
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import lancedb
from lancedb.pydantic import LanceModel, Vector
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from mcp_server.settings import Settings
from mcp_server.utils.logger import get_logger


try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PointStruct,
        VectorParams,
        Condition,
    )

    _HAS_QDRANT = True
except Exception:  # pragma: no cover
    _HAS_QDRANT = False


logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# util timing
# ────────────────────────────────────────────────────────────────────────────────
def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TimeTrack:
    started_ms: int

    def done(self) -> Dict[str, Any]:
        ended_ms = _now_ms()
        return {
            "ts_start": self.started_ms,
            "ts_end": ended_ms,
            "duration_ms": max(0, ended_ms - self.started_ms),
        }


class _Timer:
    """Context manager untuk timing sederhana."""

    def __enter__(self):
        self._t = TimeTrack(_now_ms())
        return self._t

    def __exit__(self, exc_type, exc, tb):
        self.info = self._t.done()  # type: ignore


# ────────────────────────────────────────────────────────────────────────────────
# Skema metadata (KAK & Product) — single file
# ────────────────────────────────────────────────────────────────────────────────
class ChunkMetaKAK(LanceModel):
    filename: str
    source: str
    chunk_index: int
    # meta spesifik KAK
    pelanggan: Optional[str] = None
    project: Optional[str] = None
    tahun: Optional[str] = None


class ChunkMetaProduct(LanceModel):
    filename: str
    source: str
    chunk_index: int
    # meta spesifik Product
    product: Optional[str] = None
    category: Optional[str] = None
    tahun: Optional[str] = None


# Skema LanceDB untuk table — dibangun dinamis sesuai vector dim


def build_schema_kak(vector_dim: int):
    class ChunksKAK(LanceModel):
        text: str
        vector: Vector(vector_dim)  # type: ignore
        metadata: ChunkMetaKAK

    return ChunksKAK


def build_schema_product(vector_dim: int):
    class ChunksProduct(LanceModel):
        text: str
        vector: Vector(vector_dim)  # type: ignore
        metadata: ChunkMetaProduct

    return ChunksProduct


# ────────────────────────────────────────────────────────────────────────────────
# Return helpers (konsisten): status, message, time, data
# ────────────────────────────────────────────────────────────────────────────────


def _ok(message: str = "ok", **data: Any) -> Dict[str, Any]:
    return {"status": "success", "message": message, **data}


def _err(message: str, **data: Any) -> Dict[str, Any]:
    return {"status": "error", "message": message, **data}


# ────────────────────────────────────────────────────────────────────────────────
# Embedding wrapper (async)
# ────────────────────────────────────────────────────────────────────────────────
class AsyncEmbedder:
    """Abstraksi embedder dengan antarmuka async.

    - OpenAIEmbeddings & OllamaEmbeddings dari LangChain sudah mendukung `aembed_query`.
    - Batasi concurrency dengan semaphore agar stabil di beban tinggi.
    """

    def __init__(
        self, settings: Settings, semaphore: Optional[asyncio.Semaphore] = None
    ):
        self.settings = settings
        self.sema = semaphore or asyncio.Semaphore(settings.max_concurrent_proccess)

        self._embed_primary = None
        self._embed_fallback = None
        try:
            # Default: OpenAI
            self._embed_primary = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.embed_llm_api_key, # type: ignore
            )
            logger.info("Embedder primary: OpenAIEmbeddings")
            # Siapkan fallback Ollama
            self._embed_fallback = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_host,
            )
            logger.info("Embedder fallback: OllamaEmbeddings")
        except Exception as e:
            logger.warning(
                f"OpenAIEmbeddings gagal inisialisasi: {e}. Pakai Ollama saja."
            )
            self._embed_primary = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_host,
            )
            self._embed_fallback = None

    async def aembed_query(self, text: str) -> List[float]:
        async with self.sema:
            try:
                return await self._embed_primary.aembed_query(text)  # type: ignore
            except Exception as e:
                if self._embed_fallback is None:
                    raise
                logger.warning(f"aembed_query primary gagal: {e}. Fallback → Ollama.")
                return await self._embed_fallback.aembed_query(text)  # type: ignore

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        async with self.sema:
            try:
                if hasattr(self._embed_primary, "aembed_documents"):
                    return await self._embed_primary.aembed_documents(texts)  # type: ignore
                # Jika primary tidak punya aembed_documents, fallback ke per-item
            except Exception as e:
                if self._embed_fallback is None:
                    raise
                logger.warning(
                    f"aembed_documents primary gagal: {e}. Fallback → Ollama."
                )
                if hasattr(self._embed_fallback, "aembed_documents"):
                    return await self._embed_fallback.aembed_documents(texts)  # type: ignore
                # Fallback terakhir: per item di luar semaphore
        return [await self.aembed_query(t) for t in texts]


# ────────────────────────────────────────────────────────────────────────────────
# VectorStore backend protocol
# ────────────────────────────────────────────────────────────────────────────────
class VectorStore(Protocol):
    async def connect(self) -> None: ...
    async def add(self, rows: List[Dict[str, Any]]) -> int: ...
    async def search(
        self, query_vec: List[float], k: int, where: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...
    async def delete(self, where: str) -> int: ...
    async def count(self) -> int: ...
    async def wipe(self) -> None: ...
    async def list_metadata(self, limit: int = 20) -> List[Dict[str, Any]]: ...


# ────────────────────────────────────────────────────────────────────────────────
# LanceDB Backend (async)
# ────────────────────────────────────────────────────────────────────────────────
class LanceDBBackend(VectorStore):
    def __init__(self, settings: Settings, vector_dim: int, collection_name: str):
        self.settings = settings
        self.vector_dim = int(vector_dim)
        self.collection_name = collection_name
        self.db = None
        self.table = None

    async def connect(self) -> None:
        # Bangun schema gabungan yang fleksibel (text, vector, metadata:any)
        class _MetaAny(LanceModel):  # LanceDB schema metadata
            filename: Optional[str] = None
            source: Optional[str] = None
            chunk_index: Optional[int] = None
            pelanggan: Optional[str] = None
            project: Optional[str] = None
            tahun: Optional[str] = None
            product: Optional[str] = None
            category: Optional[str] = None

        class _Generic(LanceModel):
            text: str
            vector: Vector(self.vector_dim)  # type: ignore
            metadata: _MetaAny

        # Koneksi
        self.db = await lancedb.connect_async(self.settings.vector_store_path)
        try:
            self.table = await self.db.open_table(self.collection_name)
            await self.table.count_rows()
        except Exception:
            logger.info("LanceDB: koleksi belum ada, membuat baru…")
            self.table = await self.db.create_table(
                self.collection_name, schema=_Generic, mode="create"
            )
        logger.info(
            f"LanceDB terhubung (path={self.settings.vector_store_path}, table={self.collection_name})"
        )

    async def add(self, rows: List[Dict[str, Any]]) -> int:
        """Lancedb vector store"""
        if not rows:
            return 0

        clean_rows: List[Dict[str, Any]] = []
        for i, r in enumerate(rows):
            vec = r.get("vector")
            if vec is None:
                raise ValueError(f"baris[{i}] tidak memiliki 'vector'")
            if np is not None and hasattr(vec, "shape"):  # numpy array -> list
                vec = vec.tolist()
            if len(vec) != self.vector_dim:
                raise ValueError(
                    f"vector_dim mismatch di baris[{i}]: got {len(vec)}, expected {self.vector_dim}"
                )

            md = r.get("metadata") or {}
            text = r.get("text", "")
            rid = r.get("id")  # opsional, kalau ada ikutkan

            clean_rows.append(
                {
                    "id": rid,  # biarkan None kalau tidak dipakai; LanceDB akan tambah kolom 'id'
                    "vector": vec,
                    "text": text,
                    "metadata": md,  # simpan sebagai satu kolom dict (schema-less)
                }
            )

        before = await self.table.count_rows()  # type: ignore
        await self.table.add(clean_rows)  # type: ignore
        after = await self.table.count_rows()  # type: ignore
        return int(max(0, after - before))

    async def search(
        self,
        query_vec: List[float],
        k: int,
        where: Optional[Union[Dict[str, Any], str]] = None,
    ) -> List[Dict[str, Any]]:
        # Validasi dimensi query vector
        if len(query_vec) != self.vector_dim:
            raise ValueError(
                f"vector_dim mismatch pada query: got {len(query_vec)}, expected {self.vector_dim}"
            )

        builder = await self.table.search(query_vec)  # type: ignore

        # Bangun ekspresi WHERE dari dict (schema-less): {"field": value | [v1,v2,...]}
        where_expr: Optional[str] = None
        if isinstance(where, dict) and where:

            def to_sql_lit(v: Any) -> str:
                if isinstance(v, str):
                    return "'" + v.replace("'", "''") + "'"
                if isinstance(v, bool):
                    return "TRUE" if v else "FALSE"
                return str(v)

            must_parts: List[str] = []
            should_parts: List[str] = []
            for key, val in where.items():
                if val is None:
                    continue
                if isinstance(val, (list, tuple, set)):
                    lits = ", ".join(to_sql_lit(x) for x in val)
                    should_parts.append(f"{key} IN ({lits})")
                else:
                    must_parts.append(f"{key} = {to_sql_lit(val)}")

            expr_bits: List[str] = []
            if must_parts:
                expr_bits.append(" AND ".join(must_parts))
            if should_parts:
                expr_bits.append("(" + " OR ".join(should_parts) + ")")
            where_expr = " AND ".join(expr_bits) if expr_bits else None

        elif isinstance(where, str) and where.strip():
            where_expr = where.strip()

        if where_expr:
            builder = builder.where(where_expr)

        df = await builder.limit(int(k)).to_pandas()
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            score_val = row.get("score")
            if score_val is None:
                score_val = row.get("vector_score", row.get("distance", 0.0))
            out.append(
                {
                    "text": row.get("text"),
                    "metadata": row.get("metadata") or {},
                    "score": float(score_val),
                }
            )
        return out

    async def delete(self, where: str) -> int:
        return int(await self.table.delete(where))  # type: ignore

    async def count(self) -> int:
        return int(await self.table.count_rows())  # type: ignore

    async def wipe(self) -> None:
        # Hapus table & buat ulang
        try:
            await self.db.drop_table(self.collection_name)  # type: ignore
        except Exception:
            pass
        await self.connect()

    async def list_metadata(self, limit: int = 20) -> List[Dict[str, Any]]:
        # Trik: pencarian dummy lalu ambil payload metadata
        builder = await self.table.search([0.0] * self.vector_dim)  # type: ignore
        df = await builder.limit(limit).to_pandas()
        return [r.get("metadata") or {} for _, r in df.iterrows()]


# ────────────────────────────────────────────────────────────────────────────────
# Qdrant Backend (wrapped sync → async)
# ────────────────────────────────────────────────────────────────────────────────
class QdrantBackend(VectorStore):
    def __init__(self, settings: Settings, vector_dim: int, collection_name: str):
        if not _HAS_QDRANT:
            raise RuntimeError("qdrant-client tidak terpasang")
        self.settings = settings
        self.vector_dim = int(vector_dim)
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        # mapping distance
        dist_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "euclidean": Distance.EUCLID,
        }
        self.distance = dist_map.get(settings.qdrant_distance.lower(), Distance.COSINE)

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def connect(self) -> None:
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30,
        )

        # Buat hanya jika belum ada; hindari recreate (data hilang)
        try:
            await self._run(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim, distance=self.distance
                ),
            )
        except Exception as e:
            if "exist" not in str(e).lower():
                raise

        logger.info(
            f"Qdrant terhubung (url={self.settings.qdrant_url}, collection={self.collection_name})"
        )

    async def add(self, rows: List[Dict[str, Any]]) -> int:
        """qdrant vector store"""
        if not rows:
            return 0

        points = []
        for i, r in enumerate(rows):
            vec = r.get("vector")
            if vec is None:
                raise ValueError(f"baris[{i}] tidak memiliki 'vector'")

            # Pastikan tipe list (Qdrant butuh list[float])
            if np is not None and hasattr(vec, "shape"):  # numpy array
                vec = vec.tolist()

            # Validasi dimensi per-row
            if len(vec) != self.vector_dim:
                raise ValueError(
                    f"vector_dim mismatch di baris[{i}]: got {len(vec)}, expected {self.vector_dim}"
                )

            md = r.get("metadata") or {}  # boleh berbeda antar dokumen (schema-less)
            text = r.get("text", "")

            # Buat id unik per-row (deterministik)
            pid = r.get("id")
            if not pid:
                base = f"{md.get('document_type', 'doc')}::{md.get('filename', '')}::{md.get('chunk_index', i)}"
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, base))

            payload = {**md, "text": text}

            pt = PointStruct(
                id=pid,  # str/int wajib; jangan None
                vector=vec,
                payload=payload,
            )
            points.append(pt)

        # Gunakan keyword args agar kompatibel lintas versi client
        await self._run(
            self.client.upsert, collection_name=self.collection_name, points=points # type: ignore
        )  # type: ignore
        return len(points)

    def _to_filter(
        self, where: Optional[str], metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Filter]:
        # where string dari LanceDB tidak dipakai di Qdrant; gunakan metadata_filter (lebih aman)
        if not metadata_filter:
            return None
        must: List[FieldCondition] = []
        for k, v in metadata_filter.items():
            key = f"metadata.{k}"
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                must.append(FieldCondition(key=key, match=MatchAny(any=list(v))))
            else:
                must.append(
                    FieldCondition(key=key, match=MatchValue(value=str(v).lower()))
                )
        return Filter(must=must) if must else None  # type: ignore

    async def search(
        self,
        query_vec: List[float],
        k: int,
        where: Optional[Union[Dict[str, Any], str]] = None,
    ) -> List[Dict[str, Any]]:
        # Validasi dimensi query vector
        if len(query_vec) != self.vector_dim:
            raise ValueError(
                f"vector_dim mismatch pada query: got {len(query_vec)}, expected {self.vector_dim}"
            )

        # Bangun Filter Qdrant dari dict sederhana: {"field": value | [v1, v2, ...]}
        qfilter: Optional[Filter] = None
        if isinstance(where, dict) and where:
            must: List[Condition] = []
            should: List[Condition] = []
            for key, val in where.items():
                if val is None:
                    continue
                if isinstance(val, (list, tuple, set)):
                    # OR antar nilai untuk field yang sama → pakai SHOULD
                    for v in val:
                        should.append(
                            FieldCondition(key=key, match=MatchValue(value=v))
                        )
                else:
                    # AND antar field berbeda → pakai MUST
                    must.append(FieldCondition(key=key, match=MatchValue(value=val)))
            if must or should:
                qfilter = Filter(must=must or None, should=should or None)

        elif isinstance(where, str) and where.strip():
            # Qdrant tidak menerima ekspresi string seperti LanceDB.
            # Abaikan atau log-kan sesuai kebutuhan Anda.
            logger.warning(
                "QdrantBackend.search: parameter 'where' bertipe string diabaikan."
            )

        # Eksekusi pencarian Qdrant
        res = await self._run(
            self.client.search, # type: ignore
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=int(k),
            query_filter=qfilter,  # beberapa versi menggunakan 'filter', keyword ini kompatibel
            with_payload=True,
            with_vectors=False,
        )

        out: List[Dict[str, Any]] = []
        for p in res:
            payload = p.payload or {}
            out.append(
                {
                    "text": payload.get("text"),
                    "metadata": payload,
                    "score": float(p.score),
                }
            )
        return out

    async def delete(self, where: str) -> int:
        # where sederhana: metadata.filename = 'xxx'
        if "metadata.filename" not in where:
            return 0
        filename = where.split("=")[-1].strip().strip("'\"")
        flt = Filter(
            must=[
                FieldCondition(
                    key="metadata.filename", match=MatchValue(value=filename)
                )
            ]  # type: ignore
        )
        r = await self._run(self.client.delete, self.collection_name, flt)  # type: ignore
        return int(getattr(r, "deleted", 0))

    async def count(self) -> int:
        info = await self._run(self.client.count, self.collection_name, exact=True)  # type: ignore
        return int(getattr(info, "count", 0))

    async def wipe(self) -> None:
        await self._run(self.client.delete_collection, self.collection_name)  # type: ignore
        await self.connect()

    async def list_metadata(self, limit: int = 20) -> List[Dict[str, Any]]:
        # Cari tanpa filter, ambil payload metadata
        res = await self._run(
            self.client.scroll,  # type: ignore
            self.collection_name,
            limit=limit,
            with_payload=True,
        )
        points = res[0] if isinstance(res, (tuple, list)) else []
        return [getattr(p, "payload", {}).get("metadata", {}) for p in points]


# ────────────────────────────────────────────────────────────────────────────────
# RAGPipeline Pro
# ────────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    """Pipeline RAG siap produksi.

    Komponen:
    - Embedder async (OpenAI/Ollama via LangChain) dengan semaphore.
    - Backend vector store ganda (LanceDB, Qdrant) dengan fallback otomatis.
    - API ingest, retrieval, dan administrasi.
    - Return dict konsisten + timing untuk observabilitas.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()  # type: ignore
        self.logger = logger

        # Semaphore untuk batasi kemacetan embed API
        self.embedder = AsyncEmbedder(self.settings)

        # Hitung dimensi vektor (sekali saat start)
        # --- Vector dimension ---
        # 1) Jika diset via Settings, pakai itu; jika tidak, ukur dari embedder.
        cfg_dim = int(self.settings.vector_dim or 0)
        if cfg_dim > 0:
            self.vector_dim = cfg_dim
        else:
            self.vector_dim = len(
                asyncio.get_event_loop().run_until_complete(
                    self.embedder.aembed_query("__dim__")
                )
            )

        # 2) Validasi: jika dimensi embed aktual beda dengan self.vector_dim → koreksi & log.
        try:
            _actual_dim = len(
                asyncio.get_event_loop().run_until_complete(
                    self.embedder.aembed_query("__probe__")
                )
            )
            if _actual_dim != self.vector_dim:
                logger.warning(
                    f"vector_dim mismatch: settings/terpakai={self.vector_dim}, embed_actual={_actual_dim}. "
                    f"Menggunakan embed_actual."
                )
                self.vector_dim = _actual_dim
        except Exception:
            # Jika probing gagal, biarkan self.vector_dim sesuai pengaturan/estimasi.
            pass

        # Backend utama & fallback
        self.collection = self.settings.collection_name
        if _HAS_QDRANT:
            self.backend_name: Literal["lancedb", "qdrant"] = "qdrant"
            self.backend: VectorStore = QdrantBackend(
                self.settings, self.vector_dim, self.collection
            )
            self.fallback_backend: Optional[VectorStore] = LanceDBBackend(
                self.settings, self.vector_dim, self.collection
            )
        else:
            self.backend_name = "lancedb"
            self.backend = LanceDBBackend(
                self.settings, self.vector_dim, self.collection
            )
            self.fallback_backend = None

        # Inisialisasi koneksi (sinkron saat init supaya cepat terdeteksi error)
        asyncio.get_event_loop().run_until_complete(self._connect_bootstrap())

    async def _connect_bootstrap(self) -> None:
        """Sambungkan backend utama; jika gagal dan ada fallback, gunakan fallback."""
        try:
            await self.backend.connect()
            self.backend_name = (
                "qdrant" if isinstance(self.backend, QdrantBackend) else "lancedb"
            )
            self.logger.info(f"RAGPipeline siap. Backend utama: {self.backend_name}")
        except Exception as e:
            self.logger.error(f"Backend utama gagal: {e}")
            if self.fallback_backend is None:
                raise
            self.logger.warning("Mengaktifkan fallback backend: LanceDB")
            self.backend = self.fallback_backend
            try:
                await self.backend.connect()
            except Exception as fe:
                self.logger.error(f"Fallback LanceDB juga gagal: {fe}")
                raise
            self.backend_name = "lancedb"
            self.logger.info("RAGPipeline siap. Backend fallback: lancedb")

    # ──────────────────────────────────────────────────────────────────────
    # Util
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisasi metadata agar seragam untuk filter & pencarian.
        - Semua string → lowercase + strip
        - `chunk_index` dipaksa int
        - Field yang dikenal tetap: filename, source, chunk_index, pelanggan, project, tahun, product, category
        """
        allowed = {
            "filename",
            "source",
            "chunk_index",
            "pelanggan",
            "project",
            "tahun",
            "product",
            "category",
        }
        out: Dict[str, Any] = {}
        for k in allowed:
            v = meta.get(k)
            if k == "chunk_index":
                try:
                    out[k] = int(v) if v not in (None, "", "null") else None
                except Exception:
                    out[k] = None
            else:
                out[k] = v.strip().lower() if isinstance(v, str) and v.strip() else None
        return out

    @staticmethod
    def _build_where_from_filter(
        metadata_filter: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Bangun klausa WHERE (LanceDB) & dict filter (Qdrant).
        - LanceDB: pakai 'metadata.<field>' dan normalisasi string → lower + escape.
        - Qdrant: kembalikan dict 'cleaned' tanpa prefix 'metadata.' (payload top-level)."""
        if not metadata_filter:
            return None, {}

        clauses: List[str] = []
        cleaned: Dict[str, Any] = {}

        def _to_lower(v: Any) -> Any:
            return v.lower() if isinstance(v, str) else v

        def _sql_lit(v: Any) -> str:
            if isinstance(v, str):
                return "'" + v.replace("'", "''").lower() + "'"
            if isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            return str(v)

        for field, val in metadata_filter.items():
            if val is None:
                continue
            key = f"metadata.{field}"

            if isinstance(val, (list, tuple, set)):
                items = [x for x in val if x is not None]
                if not items:
                    continue
                clauses.append(
                    f"{key} IN ({', '.join(_sql_lit(_to_lower(x)) for x in items)})"
                )
                cleaned[field] = [_to_lower(x) for x in items]
            else:
                sval = _to_lower(val)
                clauses.append(f"{key} = {_sql_lit(sval)}")
                cleaned[field] = sval

        return (" AND ".join(clauses) if clauses else None), cleaned

    def _make_citation(self, meta: Dict[str, Any]) -> str:
        parts = [
            meta.get("filename", ""),
            meta.get("pelanggan", ""),
            meta.get("category", ""),
            meta.get("project", ""),
            meta.get("product", ""),
            meta.get("tahun", ""),
        ]
        return "[" + " - ".join([p for p in parts if p]) + "]"

    # ──────────────────────────────────────────────────────────────────────
    # Admin & health
    # ──────────────────────────────────────────────────────────────────────
    async def health(self) -> Dict[str, Any]:
        """Cek kesehatan backend & embedder."""
        with _Timer() as t:
            try:
                # embed 1 kata
                vec = await self.embedder.aembed_query("health")
                c = await self.backend.count()
                return _ok(
                    "healthy",
                    time=t.done(),
                    data={
                        "backend": self.backend_name,
                        "vector_dim": len(vec),
                        "count": c,
                    },
                )
            except Exception as e:
                return _err("unhealthy", time=t.done(), error=str(e))

    async def stats(self) -> Dict[str, Any]:
        """Statistik singkat koleksi."""
        with _Timer() as t:
            try:
                c = await self.backend.count()
                return _ok(
                    "ok", time=t.done(), data={"backend": self.backend_name, "count": c}
                )
            except Exception as e:
                return _err(str(e), time=t.done())

    async def reset(self) -> Dict[str, Any]:
        """Reset koleksi (hapus & buat ulang). HATI‑HATI!"""
        with _Timer() as t:
            try:
                await self.backend.wipe()
                return _ok(
                    "vector store direset",
                    time=t.done(),
                    data={"backend": self.backend_name},
                )
            except Exception as e:
                return _err(f"gagal reset: {e}", time=t.done())

    async def switch_backend(
        self, name: Literal["lancedb", "qdrant"]
    ) -> Dict[str, Any]:
        """Ganti backend aktif secara dinamis."""
        with _Timer() as t:
            try:
                if name == "lancedb":
                    self.backend = LanceDBBackend(
                        self.settings, self.vector_dim, self.collection
                    )
                elif name == "qdrant":
                    if not _HAS_QDRANT:
                        return _err("qdrant-client tidak tersedia", time=t.done())
                    self.backend = QdrantBackend(
                        self.settings, self.vector_dim, self.collection
                    )
                else:
                    return _err("backend tidak dikenal", time=t.done())
                await self.backend.connect()
                self.backend_name = name
                return _ok("backend diganti", time=t.done(), data={"backend": name})
            except Exception as e:
                return _err(f"gagal switch backend: {e}", time=t.done())

    # ──────────────────────────────────────────────────────────────────────
    # Legacy Pipeline Chunk and Ingest
    # ──────────────────────────────────────────────────────────────────────
    async def _chunk_doc(self, dl_doc) -> List[Any]:
        """Jalankan HybridChunker di process pool Thread agar event loop tidak ter-block."""
        loop = asyncio.get_running_loop()
        chunker = HybridChunker(merge_peers=True)

        return await loop.run_in_executor(None, lambda: chunker.chunk(dl_doc=dl_doc))  # type: ignore

    async def ingest_kak_chunks(
        self,
        dl_doc: Any,
        *,
        filename: str,
        pelanggan: str,
        project: str,
        tahun: str,
    ) -> Dict[str, Any]:
        """
        Legacy Pipeline. Gunakan ingest_kak_chunks_from_payload()
        Ingest dokumen **KAK/TOR** (chunk → embed → add).

        Returns: dict {status, message, time, data:{added, meta}}
        """
        with _Timer() as t:
            try:
                # 1) Chunking (CPU heavy) → jalankan di thread executor
                chunks = await self._chunk_doc(dl_doc)
                texts = [c.text for c in chunks]

                # 2) Embedding (async + semaphore)
                vectors = await self.embedder.aembed_documents(texts)

                # 3) Bangun rows
                rows: List[Dict[str, Any]] = []
                for idx, (text, vec) in enumerate(zip(texts, vectors)):
                    meta = self._sanitize_meta(
                        {
                            "filename": filename,
                            "source": filename,
                            "chunk_index": idx,
                            "pelanggan": pelanggan,
                            "project": project,
                            "tahun": tahun,
                        }
                    )
                    rows.append({"text": text, "vector": vec, "metadata": meta})

                # 4) Tambahkan ke vector store
                added = await self.backend.add(rows)
                msg = f"{added} chunk KAK ditambahkan"
                return _ok(
                    msg,
                    time=t.done(),
                    data={
                        "added": added,
                        "meta": {
                            "filename": filename,
                            "pelanggan": pelanggan,
                            "project": project,
                            "tahun": tahun,
                        },
                    },
                )
            except Exception as e:
                self.logger.exception("Gagal ingest KAK")
                return _err(f"ingest KAK gagal: {e}", time=t.done())

    async def ingest_product_chunks(
        self,
        dl_doc: Any,
        *,
        filename: str,
        product: str,
        category: str,
        tahun: str,
    ) -> Dict[str, Any]:
        """
        Legacy Pipeline. Gunakan ingest_product_chunks_from_payload()
        Ingest dokumen **Product** (chunk → embed → add).
        """
        with _Timer() as t:
            try:
                chunks = await self._chunk_doc(dl_doc)
                texts = [c.text for c in chunks]
                vectors = await self.embedder.aembed_documents(texts)
                rows: List[Dict[str, Any]] = []
                for idx, (text, vec) in enumerate(zip(texts, vectors)):
                    meta = self._sanitize_meta(
                        {
                            "filename": filename,
                            "source": filename,
                            "chunk_index": idx,
                            "product": product,
                            "category": category,
                            "tahun": tahun,
                        }
                    )
                    rows.append({"text": text, "vector": vec, "metadata": meta})
                added = await self.backend.add(rows)
                msg = f"{added} chunk Product ditambahkan"
                return _ok(
                    msg,
                    time=t.done(),
                    data={
                        "added": added,
                        "meta": {
                            "filename": filename,
                            "product": product,
                            "category": category,
                            "tahun": tahun,
                        },
                    },
                )
            except Exception as e:
                self.logger.exception("Gagal ingest Product")
                return _err(f"ingest Product gagal: {e}", time=t.done())

    # ──────────────────────────────────────────────────────────────────────
    # Ingest (KAK & Product)
    # ──────────────────────────────────────────────────────────────────────
    async def ingest_kak_chunks_from_payload(
        self,
        *,
        chunks_payload: List[Dict[str, Any]],
        filename: str,
        pelanggan: str,
        project: str,
        tahun: str,
    ) -> Dict[str, Any]:
        """
        Payload -> embed -> upsert (return _ok / _err style).
        """
        t0 = time.perf_counter()
        try:
            # early return kalau payload kosong
            if not chunks_payload:
                return _ok(
                    "0 chunk kak ditambahkan",
                    time=(time.perf_counter() - t0),
                    data={
                        "added": 0,
                        "meta": {
                            "filename": filename,
                            "pelanggan": pelanggan,
                            "project": project,
                            "tahun": tahun,
                        },
                    },
                )

            texts: List[str] = []
            metas: List[Dict[str, Any]] = []

            for c in chunks_payload:
                t = (c.get("text") or "").strip()
                if not t:
                    continue  # skip chunk kosong

                cm = c.get("meta") or {}
                meta = {
                    "filename": filename,
                    "source": filename,
                    "chunk_index": int(
                        c.get("index") if c.get("index") is not None else len(metas) # type: ignore
                    ),
                    "pelanggan": pelanggan,
                    "project": project,
                    "tahun": str(tahun),  # pastikan string
                    "document_type": "kak",  # konsisten lowercase
                    "pages": list(cm.get("pages") or []),  # JSON-able
                    "headings": list(cm.get("headings") or []),  # JSON-able
                    "chunk_type": cm.get("chunk_type"),
                }
                metas.append(self._sanitize_meta(meta))
                texts.append(t)

            # jika semua kosong setelah filter
            if not texts:
                return _ok(
                    "0 chunk kak ditambahkan (semua kosong setelah filter)",
                    time=(time.perf_counter() - t0),
                    data={
                        "added": 0,
                        "meta": {
                            "filename": filename,
                            "pelanggan": pelanggan,
                            "project": project,
                            "tahun": tahun,
                        },
                    },
                )

            vectors = await self.embedder.aembed_documents(texts)

            # validasi panjang agar tidak ter-truncate diam2 oleh zip()
            if len(vectors) != len(texts) or len(metas) != len(texts):
                raise RuntimeError(
                    f"embedding mismatch: texts={len(texts)} vectors={len(vectors)} metas={len(metas)}"
                )

            rows = [
                {"text": t, "vector": v, "metadata": m}
                for t, v, m in zip(texts, vectors, metas)
            ]
            added = await self.backend.add(rows)

            return _ok(
                f"{added} chunk kak ditambahkan",
                time=(time.perf_counter() - t0),
                data={
                    "added": added,
                    "meta": {
                        "filename": filename,
                        "pelanggan": pelanggan,
                        "project": project,
                        "tahun": tahun,
                    },
                },
            )
        except Exception as e:
            self.logger.exception("Gagal ingest payload kak")
            return _err(
                f"ingest payload kak gagal: {e}", time=(time.perf_counter() - t0)
            )

    async def ingest_product_chunks_from_payload(
        self,
        *,
        chunks_payload: List[Dict[str, Any]],
        filename: str,
        category: str,
        product: str,
        tahun: str,
    ) -> Dict[str, Any]:
        """Payload -> embed -> upsert (return _ok / _err style)."""
        t0 = time.perf_counter()
        try:
            # early return kalau payload kosong
            if not chunks_payload:
                return _ok(
                    "0 chunk product ditambahkan",
                    time=(time.perf_counter() - t0),
                    data={
                        "added": 0,
                        "meta": {
                            "filename": filename,
                            "category": category,
                            "product": product,
                            "tahun": tahun,
                        },
                    },
                )

            texts: List[str] = []
            metas: List[Dict[str, Any]] = []

            for c in chunks_payload:
                t = (c.get("text") or "").strip()
                if not t:
                    continue  # skip chunk kosong

                cm = c.get("meta") or {}
                meta = {
                    "filename": filename,
                    "source": filename,
                    "chunk_index": int(
                        c.get("index") if c.get("index") is not None else len(metas) # type: ignore
                    ),
                    "category": category,
                    "product": product,
                    "tahun": str(tahun),  # pastikan string
                    "document_type": "product",  # konsisten lowercase
                    "pages": list(cm.get("pages") or []),
                    "headings": list(cm.get("headings") or []),
                    "chunk_type": cm.get("chunk_type"),
                }
                metas.append(self._sanitize_meta(meta))
                texts.append(t)

            if not texts:
                return _ok(
                    "0 chunk product ditambahkan (semua kosong setelah filter)",
                    time=(time.perf_counter() - t0),
                    data={
                        "added": 0,
                        "meta": {
                            "filename": filename,
                            "category": category,
                            "product": product,
                            "tahun": tahun,
                        },
                    },
                )

            vectors = await self.embedder.aembed_documents(texts)

            if len(vectors) != len(texts) or len(metas) != len(texts):
                raise RuntimeError(
                    f"embedding mismatch: texts={len(texts)} vectors={len(vectors)} metas={len(metas)}"
                )

            rows = [
                {"text": t, "vector": v, "metadata": m}
                for t, v, m in zip(texts, vectors, metas)
            ]
            added = await self.backend.add(rows)

            return _ok(
                f"{added} chunk product ditambahkan",
                time=(time.perf_counter() - t0),
                data={
                    "added": added,
                    "meta": {
                        "filename": filename,
                        "category": category,
                        "product": product,
                        "tahun": tahun,
                    },
                },
            )
        except Exception as e:
            self.logger.exception("Gagal ingest payload product")
            return _err(
                f"ingest payload product gagal: {e}", time=(time.perf_counter() - t0)
            )

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────
    async def retrieval(
        self,
        *,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieval dengan filter metadata.

        Catatan:
        - Untuk LanceDB, gunakan klausa `where` berbasis string.
        - Untuk Qdrant, gunakan `metadata_filter` dict.
        """
        with _Timer() as t:
            try:
                if not query or not query.strip():
                    return _err("query kosong", time=t.done())

                top_k = int(k or self.settings.retriever_search_k)
                if top_k <= 0:
                    top_k = 5  # default aman

                q_vec = await self.embedder.aembed_query(query)
                where_expr, cleaned_filter = self._build_where_from_filter(
                    metadata_filter
                )

                # Panggilan backend sesuai tipe:
                if isinstance(self.backend, QdrantBackend):
                    # Qdrant menerima dict filter (tanpa 'metadata.' prefix)
                    rows = await self.backend.search(q_vec, top_k, cleaned_filter)
                else:
                    # LanceDB menerima where string (pakai 'metadata.<field>')
                    rows = await self.backend.search(q_vec, top_k, where_expr)

                if not rows:
                    return _ok(
                        "no_results",
                        time=t.done(),
                        data={"results": [], "total": await self.backend.count()},
                    )

                results: List[Dict[str, Any]] = []
                for r in rows:
                    meta = r.get("metadata") or {}
                    results.append(
                        {
                            "text": r.get("text"),
                            "metadata": meta,
                            "citation": self._make_citation(meta),
                            "score": float(r.get("score", 0.0)),
                        }
                    )

                return _ok(
                    "ok",
                    time=t.done(),
                    data={"results": results, "count": len(results)},
                )

            except Exception as e:
                self.logger.exception("Retrieval error")
                return _err(f"retrieval gagal: {e}", time=t.done())

    # ──────────────────────────────────────────────────────────────────────
    # Administrasi: hapus, list meta, reindex ringan
    # ──────────────────────────────────────────────────────────────────────
    async def list_available_metadata(self, limit: int = 20) -> Dict[str, Any]:
        with _Timer() as t:
            try:
                metas = await self.backend.list_metadata(limit)
                return _ok("ok", time=t.done(), data={"metadata": metas})
            except Exception as e:
                return _err(str(e), time=t.done())

    async def delete_by_filename(self, filename: str) -> Dict[str, Any]:
        with _Timer() as t:
            try:
                if not filename:
                    return _err("filename kosong", time=t.done())
                n = await self.backend.delete(f"metadata.filename = '{filename}'")
                return _ok("ok", time=t.done(), data={"deleted": int(n)})
            except Exception as e:
                return _err(f"gagal menghapus: {e}", time=t.done())

    async def delete_by_filter(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        with _Timer() as t:
            try:
                where, cleaned = self._build_where_from_filter(metadata_filter)
                if isinstance(self.backend, QdrantBackend):
                    # Qdrant tidak menerima where string bebas di delete(); lakukan per filename jika ada
                    if "filename" in cleaned and isinstance(cleaned["filename"], str):
                        n = await self.backend.delete(
                            f"metadata.filename = '{cleaned['filename']}'"
                        )
                        return _ok("ok", time=t.done(), data={"deleted": int(n)})
                    return _err(
                        "delete by filter terbatas di Qdrant kecuali berdasarkan filename",
                        time=t.done(),
                    )
                else:
                    if not where:
                        return _err("filter kosong", time=t.done())
                    n = await self.backend.delete(where)
                    return _ok("ok", time=t.done(), data={"deleted": int(n)})
            except Exception as e:
                return _err(f"gagal delete: {e}", time=t.done())

    # ──────────────────────────────────────────────────────────────────────
    # Util tambahan (opsional untuk produksi)
    # ──────────────────────────────────────────────────────────────────────
    async def warmup(self) -> Dict[str, Any]:
        """Panggil di startup untuk memanaskan koneksi & cache."""
        with _Timer() as t:
            try:
                _ = await self.embedder.aembed_query("warmup")
                _ = await self.backend.count()
                return _ok("warmed", time=t.done(), data={"backend": self.backend_name})
            except Exception as e:
                return _err(str(e), time=t.done())
