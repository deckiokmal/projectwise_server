from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from mcp_server.settings import Settings
from mcp_server.utils.logger import logger


# —————————————————————————————————————————
#  Skema untuk metadata & chunk
# —————————————————————————————————————————


class ChunkMetadata(LanceModel):
    filename: Optional[str]
    source: Optional[str]
    chunk_index: Optional[int]
    project: Optional[str]
    tahun: Optional[str]


def build_chunks_schema(vector_dim: int):
    class Chunks(LanceModel):
        text: str
        vector: Vector(vector_dim)  # type: ignore
        metadata: ChunkMetadata

    return Chunks


class RagQuery(BaseModel):
    question: str


class RagResponse(BaseModel):
    answer: str


# —————————————————————————————————————————
#  RAGPipeline: modular & DRY
# —————————————————————————————————————————


class RAGPipeline:
    def __init__(self):
        """
        Inisialisasi RAGPipeline:
        - Membaca konfigurasi dari Settings.
        - Menentukan path dan nama koleksi.
        - Memilih dan menginisialisasi model embedding (OpenAI atau Ollama).
        - Menghitung dimensi vektor dan membangun schema chunk.
        - Menghubungkan ke LanceDB, membuka atau membuat tabel vectorstore.
        """
        settings = Settings()  # type: ignore

        # Inisialisasi paths & nama koleksi
        self.base_path = Path(settings.knowledge_base_path)
        self.collection_name = settings.collection_name
        self.persist_dir = settings.vector_store_path

        # Pilih embedding
        if settings.embedding_model.startswith("text-embedding"):
            self.embed = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,  # type: ignore
            )
            logger.info("Menggunakan OpenAI Embeddings.")
        else:
            self.embed = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_host,
            )
            logger.info("Menggunakan Ollama Embeddings.")

        # Hitung dimensi vektor & buat schema
        example_vector = self.embed.embed_query("dimension_check")
        self.vector_dim = len(example_vector)
        Chunks = build_chunks_schema(self.vector_dim)

        # Koneksi ke LanceDB
        self.db = lancedb.connect(self.persist_dir)
        try:
            self.table = self.db.open_table(self.collection_name)
            # Cek integritas
            _ = self.table.count_rows()
        except Exception:
            logger.info("Tabel tidak ada, membuat tabel baru")
            self.table = self.db.create_table(
                self.collection_name, schema=Chunks, mode="create"
            )

        logger.info(
            f"RAGPipeline siap (dim={self.vector_dim}, koleksi='{self.collection_name}')."
        )

    # —————————————————————————————————————————
    #  Utility: validasi & chunking
    # —————————————————————————————————————————

    def _validate_vector_dim(self, vector: List[float]):
        """
        Memastikan vektor embedding memiliki panjang yang sesuai dengan dimensi yang disimpan.

        Args:
            vector (List[float]): Vektor embedding yang dihasilkan model.

        Raises:
            ValueError: Jika panjang `vector` tidak sama dengan `self.vector_dim`.
        """
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Dimensi vektor salah: embedding={len(vector)}, store={self.vector_dim}"
            )

    def _chunk_document(
        self, dl_doc, filename: str, project: str, tahun: str
    ) -> List[Dict[str, Any]]:
        """
        Memecah dokumen menjadi potongan (chunk), menghitung embedding untuk tiap chunk,
        dan menyiapkan struktur entri untuk diindeks.

        Args:
            dl_doc: Objek dokumen dari DocumentConverter.
            filename (str): Nama file sumber.
            project (str): Nama proyek untuk metadata.
            tahun (str): Tahun untuk metadata.

        Returns:
            List[Dict[str, Any]]: Daftar entri dengan keys `text`, `vector`, dan `metadata`.
        """
        chunker = HybridChunker(merge_peers=True)
        chunks = chunker.chunk(dl_doc=dl_doc)
        entries = []
        for idx, chunk in enumerate(chunks):
            text = chunk.text
            vec = self.embed.embed_query(text)
            self._validate_vector_dim(vec)
            entries.append(
                {
                    "text": text,
                    "vector": list(vec),
                    "metadata": {
                        "filename": filename,
                        "source": filename,
                        "chunk_index": idx,
                        "project": project,
                        "tahun": tahun,
                    },
                }
            )
        return entries

    # —————————————————————————————————————————
    #  Ingest semua PDF di folder
    # —————————————————————————————————————————

    def setup_vector_store(
        self,
        force_recreate: bool = False,
        project: str = "default",
        tahun: str = "2025",
    ):
        """
        Menyiapkan dan mengindeks semua file PDF dalam folder knowledge_base_path.

        Args:
            force_recreate (bool): Jika True, reset terlebih dahulu vectorstore.
            project (str): Label proyek untuk metadata setiap chunk.
            tahun (str): Label tahun untuk metadata setiap chunk.
        """
        if force_recreate:
            logger.info("Rebuild vectorstore diminta.")
            self.reset_vectorstore()

        pdf_files = list(self.base_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning("Tidak ada PDF di knowledge_base_path.")
            return

        to_add = []
        for path in pdf_files:
            filename = path.name
            # skip jika sudah ada
            exists = (
                self.table.count_rows(filter=f"metadata.filename = '{filename}'") > 0
            )
            if exists:
                logger.debug(f"Skip '{filename}', sudah terindeks.")
                continue

            try:
                result = DocumentConverter().convert(source=str(path))
                to_add.extend(
                    self._chunk_document(result.document, filename, project, tahun)
                )
                logger.info(f"'{filename}' dipecah dan siap diindeks.")
            except Exception as e:
                logger.warning(f"Gagal proses '{filename}': {e}")

        if to_add:
            self.table.add(to_add)
            logger.info(f"{len(to_add)} chunk ditambahkan ke vectorstore.")
        else:
            logger.info("Tidak ada chunk baru untuk diindeks.")

    # —————————————————————————————————————————
    #  Ingest file MD ringkasan
    # —————————————————————————————————————————

    def ingest_markdown(
        self, markdown_path: str, project: str = "default", tahun: str = "2025"
    ) -> Dict[str, Any]:
        """
        Mengonversi dan mengindeks konten dari file Markdown ke dalam vectorstore.

        Args:
            markdown_path (str): Path ke file .md yang akan di-ingest.
            project (str): Label proyek untuk metadata.
            tahun (str): Label tahun untuk metadata.

        Returns:
            Dict[str, Any]: Pesan sukses atau error dalam bentuk dict.
        """
        path = Path(markdown_path)
        if not path.exists():
            msg = f"File tidak ditemukan: {markdown_path}"
            logger.warning(msg)
            return {"error": msg}

        try:
            result = DocumentConverter().convert(source=path)
            entries = self._chunk_document(result.document, path.name, project, tahun)
            self.table.add(entries)
            msg = f"{len(entries)} chunk dari '{path.name}' diindeks."
            logger.info(msg)
            return {"message": msg}
        except Exception as e:
            logger.exception("Gagal ingest markdown.")
            return {"error": str(e)}

    # —————————————————————————————————————————
    #  Reset vectorstore
    # —————————————————————————————————————————

    def reset_vectorstore(self):
        """
        Mereset vectorstore dengan cara menghapus dan membuat ulang tabel LanceDB
        sesuai schema chunk yang telah ditentukan.
        """
        self.db.drop_table(self.collection_name)
        Chunks = build_chunks_schema(self.vector_dim)
        self.table = self.db.create_table(
            self.collection_name, schema=Chunks, mode="create"
        )
        logger.info("Vectorstore di-reset dan tabel baru dibuat.")

    # —————————————————————————————————————————
    #  Retrieval
    # —————————————————————————————————————————

    def retrieval(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Melakukan similarity search dan filter metadata pada vectorstore.

        Args:
            query (str): Teks pertanyaan atau query.
            k (Optional[int]): Jumlah hasil teratas yang diinginkan.
            metadata_filter (Optional[Dict[str, Any]]): Filter metadata sebagai dict,
                misal {"project": "Alpha", "tahun": "2025"} atau nilai list untuk IN-clause.

        Returns:
            str: Gabungan teks chunk hasil pencarian beserta sumbernya, atau string kosong jika tidak ada hasil.
        """
        settings = Settings()  # type: ignore
        top_k = k or settings.retriever_search_k

        # 1. embed query
        q_vec = self.embed.embed_query(query)
        self._validate_vector_dim(q_vec)

        # 2. mulai build pencarian
        builder = self.table.search(list(q_vec))

        # 3. kalau ada metadata_filter, rakit string .where(...)
        if metadata_filter:
            clauses: List[str] = []
            for field, val in metadata_filter.items():
                key = f"metadata.{field}"
                # kalau list/tuple → IN‐clause
                if isinstance(val, (list, tuple)):
                    items = ", ".join(f"'{v}'" for v in val)
                    clauses.append(f"{key} IN ({items})")
                else:
                    # asumsi string atau number
                    if isinstance(val, str):
                        clauses.append(f"{key} = '{val}'")
                    else:
                        clauses.append(f"{key} = {val}")
            filter_expr = " AND ".join(clauses)
            builder = builder.where(filter_expr)

        # 4. limit dan ambil pandas DataFrame
        df = builder.limit(top_k).to_pandas()

        if df.empty:
            logger.info(
                f"Tidak ada hasil untuk '{query}' dengan filter {metadata_filter}."
            )
            return ""

        # 5. format jawaban dengan citation metadata
        contexts = []
        for _, row in df.iterrows():
            meta = row["metadata"] or {}
            citation = (
                f"[{meta.get('filename', '')}"
                f" - {meta.get('project', '')}"
                f" - {meta.get('tahun', '')}]"
            )
            contexts.append(f"{row['text']}\n\nSumber: {citation}")

        logger.info(
            f"{len(contexts)} hasil retrieval untuk '{query}' dengan filter {metadata_filter}."
        )
        return "\n\n".join(contexts)
