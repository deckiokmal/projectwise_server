### 1. Extraction
Method ada di mcp_server/utils/cpu_workers.py

1. support PDF file
2. menggunakan Docling DocumentConverter() untuk convert PDF -> Docling Object
3. Processing menggunakan Pool Executor -> Multi processing dengan cpu-workers
4. chunk berbasis tokenizer. support Hugging Face dan Openai Tokenizer
5. Menggunakan contextual HybridChunker untuk menghasilkan chunk yang baik
6. Output juga default export to markdown document yang bisa disimpan ke file system


cara pakai:
```python
loop = asyncio.get_running_loop()
pool = get_cpu_pool()

# === 1) CPU-bound di worker (dl_doc dipakai di sana) ===
worker_out = await loop.run_in_executor(
    pool,
    convert_and_chunk_pdf_worker,
    str(pdf_info.full_path),
    tokenizer_kind=self.settings.tokenizer_kind,
    tokenizer_model=self.settings.embedding_model,
    tokenizer_max_tokens=self.settings.tokenizer_max_token,     
)

if worker_out.get("status") != "success":
    raise RuntimeError(worker_out.get("message") or "Gagal convert/chunk PDF")

chunks_payload = worker_out["chunks"]        # list[dict]
markdown: str | None = worker_out.get("markdown")
```


# ProjectWise RAG System — README

> **Tujuan:** dokumentasi arsitektur, alur kerja, konfigurasi, dan cara pakai sistem RAG (Retrieval‑Augmented Generation) berbasis FastAPI + Docling + LangChain dengan eksekusi CPU‑bound via multiprocessing, embedding OpenAI ↔ Ollama, serta vektor store **Qdrant (default)** dengan fallback **LanceDB**.

---

## 1) Gambaran Umum

Sistem ini mengotomatisasi *ingestion* dokumen PDF (KAK/TOR atau Product/Teknis) menjadi potongan (chunk) yang *tokenizer‑aware*, melakukan embedding vektor, lalu menyimpannya ke basis vektor. Proses berat (konversi PDF → `dl_doc` Docling → HybridChunker) dipindahkan ke **process pool** agar event loop tetap responsif.

**Komponen utama:**

- **Orkestrator**: `KAKTools` untuk ingest dan ringkasan KAK/TOR.
- **Worker CPU**: convert PDF → Docling `dl_doc` → HybridChunker (tokenizer OpenAI/HF) → hasilkan payload chunk + markdown.
- **RAGPipeline**: embedding (OpenAI default; fallback Ollama), vector store (Qdrant default; fallback LanceDB), retrieval, administrasi.
- **Multiprocessing utils**: singleton `ProcessPoolExecutor` (dibuat di lifespan FastAPI).
- **Helper & Tokenizer**: util ringkasan teks panjang berbasis LangChain; util wrapper tokenizer bila diperlukan.

---

## 2) Alur Data End‑to‑End

1. **Resolve path PDF** → `resolve_kak_pdf(...)` (util KAK path).
2. **CPU worker (multiprocessing)**  
   a. `DocumentConverter().convert(path)` → `dl_doc`  
   b. `HybridChunker(tokenizer=..., merge_peers=True).chunk(dl_doc)`  
   c. `contextualize(...)` setiap chunk → **`chunks_payload`** (text+meta)  
   d. `dl_doc.export_to_markdown()` → **markdown** opsional
3. **Back to main process (async)**  
   a. **Overwrite** (opsional) → hapus dokumen lama di vektor store  
   b. **Embedding** `aembed_documents(texts)`  
   c. **Upsert** rows `{text, vector, metadata}` ke vector store  
   d. **Simpan** markdown + **update manifest**
4. **Retrieval**: embed query → cari ke backend aktif (Qdrant/LanceDB) + filter metadata.
5. **(Opsional) Summarization**: LLMChains merangkum markdown KAK → simpan `*_summary.md`.

---

## 3) Pemetaan Berkas (Modul)

- **Ingestion & Summarization** — `mcp_server/tools/kak_analyzer_tool.py`  
  - Method `ingest_kak_file(...)`: mengatur seluruh alur ingestion.  
  - Memanggil worker proses via `run_in_executor(pool, convert_and_chunk_pdf_worker, ...)`.  
  - Menulis markdown dan update manifest.  
  - Tersedia juga `generate_kak_summarize(...)` untuk merangkum KAK.  

- **Worker CPU** — `mcp_server/utils/cpu_workers.py`  
  - Fungsi top‑level `convert_and_chunk_pdf_worker(...)` (picklable).  
  - Mendukung **OpenAI** (tiktoken) dan **HuggingFace** (AutoTokenizer) untuk tokenisasi chunking.  
  - Menggunakan Docling `HybridChunker` + `contextualize`.  

- **RAG Pipeline** — `mcp_server/utils/rag_pipeline.py` (atau `rag_pipeline_util.py`)  
  - Kelas `RAGPipeline`:  
    - **Embedding**: `OpenAIEmbeddings` default, fallback `OllamaEmbeddings` (LangChain).  
    - **Vector Store**: **Qdrant** default (jarak pilih via `qdrant_distance`), fallback **LanceDB**.  
    - **Vector Dim**: pakai `settings.vector_dim` jika di-set; kalau tidak, **probe** dimensi dari embedder di startup.  
    - API: `ingest_*_from_payload`, `retrieval`, `delete_*`, `list_metadata`, `switch_backend`, `warmup`.  

- **Multiprocessing** — `mcp_server/utils/multiprocessing_utils.py`  
  - `create_cpu_pool()` membuat singleton `ProcessPoolExecutor`.  
  - `get_cpu_pool()` mengambil instance (digunakan di runtime).  
  - `shutdown_cpu_pool()` menutup pool saat shutdown.

- **Helper** — `mcp_server/utils/helper.py`  
  - Ringkasan teks panjang (`summarize_long_product_text`) berbasis LangChain + chunking token.  
  - `list_files`, `clean_utf8`, `get_tokenizer`.

- **Tokenizer (opsional)** — `mcp_server/utils/tokenizer.py`  
  - `OpenAITokenizerWrapper` (HF‑style) untuk kasus tertentu yang memerlukan kompatibilitas antarmuka.  

> **Catatan**: Struktur direktori/penamaan file dapat sedikit berbeda sesuai repositori Anda; sesuaikan dengan path aktual.

---

## 4) Konfigurasi (Settings & ENV)

Pastikan variabel berikut (nama bisa bervariasi sesuai `Settings` Anda):

- **LLM & Embedding**
  - `LLM_MODEL`: nama model chat/LLM (mis. `gpt-4o` atau model Ollama).
  - `LLM_API_KEY`: API key untuk OpenAI jika menggunakan OpenAI.  
  - `EMBEDDING_MODEL`: nama model embedding (mis. OpenAI `text-embedding-3-*` atau HF `nomic-ai/nomic-embed-text-v1.5`).

- **Tokenizer untuk Chunking**
  - `TOKENIZER_KIND`: `openai` | `hf` (menentukan jalur tokenisasi di worker).  
  - `TOKENIZER_MODEL`: (opsional) model tokenizer HF/OpenAI spesifik.  
  - `TOKENIZER_MAX_TOKEN`: (opsional) batas token saat chunker menghitung ukuran potongan.

- **Vector Store**
  - **Qdrant (default)**: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_DISTANCE` (`cosine`/`dot`/`euclid`).  
  - **LanceDB (fallback)**: `VECTOR_STORE_PATH` (path direktori database).  
  - `COLLECTION_NAME`: nama koleksi/tabel.

- **Dimensi Vektor**
  - `VECTOR_DIM`: default `1536`. Bila tidak diset, pipeline akan **probe** dari embedder saat startup.  
    - Disarankan **samakan** dengan dimensi model embedding yang dipakai.

- **Concurrency**
  - `MAX_CONCURRENT_PROCCESS`: batas semafor untuk embedding async.  
  - `CPU_POOL_WORKERS`: jumlah proses di process pool.

> Semua nilai di atas dibaca melalui `Settings()` Anda. Sesuaikan nama field ENV dengan yang dipakai di kode.

---

## 5) Menjalankan Aplikasi

1. **Install dependensi** (contoh):
   ```bash
   pip install -r requirements.txt
   # Tambahan (opsional):
   pip install transformers tiktoken qdrant-client lancedb langchain langchain-openai
   ```

2. **Set ENV minimal** (contoh):
   ```bash
   set LLM_MODEL=gpt-4o
   set LLM_API_KEY=sk-...
   set EMBEDDING_MODEL=text-embedding-3-small
   set VECTOR_DIM=1536
   set QDRANT_URL=http://localhost:6333
   set QDRANT_API_KEY=
   set VECTOR_STORE_PATH=./.vectordb
   set MAX_CONCURRENT_PROCCESS=4
   set TOKENIZER_KIND=openai
   set TOKENIZER_MODEL=gpt-4o
   set TOKENIZER_MAX_TOKEN=8192
   ```

3. **Jalankan FastAPI** (mode dev):
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload
   ```
   > Pastikan **lifespan** Anda membuat **process pool** di startup dan mematikannya saat shutdown.

---

## 6) Cara Ingest KAK/TOR (Programatik)

Contoh pemanggilan langsung kelas (tanpa endpoint HTTP):

```python
from mcp_server.tools.kak_analyzer_tool import KAKTools
import asyncio

async def main():
    tool = KAKTools()
    res = await tool.ingest_kak_file(
        filename="Draft TOR Pekerjaan Jasa Perawatan Data Center.pdf",
        pelanggan="AcmeCorp",
        project="DC-Upgrade",
        tahun="2025",
        overwrite=True,
    )
    print(res)

asyncio.run(main())
```

**Catatan penting:**
- Jalur **konversi+chunk** berjalan di **process pool** → responsif untuk API.  
- Jika `overwrite=True`, entri lama untuk `filename` akan dihapus sebelum upsert ulang.

---

## 7) Summarization (Ringkasan KAK)

Gunakan `generate_kak_summarize(...)` untuk menggabungkan **template prompt** + **markdown KAK** dan menghasilkan ringkasan.

```python
res = await tool.generate_kak_summarize(
    filename="Draft TOR ... .pdf",
    pelanggan="AcmeCorp",
    project="DC-Upgrade",
    tahun="2025",
    prompt_instruction="project_analysis.txt",  # path template prompt
)
```

Ringkasan disimpan sebagai `*_summary.md` pada struktur direktori KAK Anda.

---

## 8) Retrieval

Panggil `RAGPipeline.retrieval(query=..., k=..., metadata_filter=...)`.  
- Untuk **Qdrant**, gunakan `metadata_filter` dict.  
- Untuk **LanceDB**, builder akan menggunakan klausa `where` string.

Contoh filter:
```python
res = await pipeline.retrieval(
    query="SLA untuk pemeliharaan UPS?",
    k=5,
    metadata_filter={"pelanggan": "acmecorp", "tahun": "2025"},
)
```

---

## 9) Administrasi Vector Store

- Hapus per filename: `pipeline.delete_by_filename("nama_file.pdf")`  
- Daftar metadata: `pipeline.list_available_metadata(limit=20)`  
- Reset koleksi: `pipeline.reset()` *(hati‑hati: menghapus data)*  
- Ganti backend runtime: `pipeline.switch_backend("qdrant" | "lancedb")`

---

## 10) Praktik Terbaik & Tuning

- **Multiprocessing**:  
  - Gunakan **fungsi top‑level** untuk worker agar picklable (hindari *bound method*).  
  - Batasi jumlah proses `CPU_POOL_WORKERS` = `min(cpu-1, 4)` lalu skala bertahap.
- **Tokenizer‑aware chunking**:  
  - *Match* tokenizer chunker dengan model embedding (OpenAI ↔ tiktoken, HF/Nomic ↔ AutoTokenizer).  
  - Gunakan `contextualize` di HybridChunker untuk memperkaya konten chunk.
- **Vector dim**:  
  - Set `VECTOR_DIM` sesuai model embedding **atau** biarkan sistem **probe** dimensi.  
  - Jika mismatch dengan koleksi eksisting, pertimbangkan buat koleksi baru.
- **Overwrite**:  
  - Hapus dulu by filename sebelum upsert ulang untuk mencegah duplikasi.
- **Monitoring**:  
  - Catat durasi convert/chunk/embed/upsert per file dan jumlah chunk terbuat.

---

## 11) Troubleshooting

- **`cannot pickle '_thread.RLock' object'`**  
  - Penyebab: mengirim **bound method/objek non‑picklable** ke ProcessPool.  
  - Solusi: gunakan **fungsi top‑level** (`convert_and_chunk_pdf_worker`) dengan argumen primitif.

- **Perbedaan dimensi vektor**  
  - Pastikan `VECTOR_DIM` sesuai dimensi embedding. Atau kosongkan agar **probe** otomatis.

- **Qdrant “collection exists”**  
  - Kode sudah menangani idempotensi pembuatan koleksi. Jika ubah dimensi, buat koleksi baru.

- **Token limit saat summarization**  
  - Atur `max_tokens` input ringkasan (`helper.summarize_long_product_text`) agar tidak melebihi context window model.

---

## 12) Roadmap Ide Peningkatan

- Batch embedding/upsert (mini‑batch) untuk throughput lebih tinggi.
- Streaming ingestion & progress callback per halaman.
- Auto‑OCR per halaman (opsional) dengan deteksi adaptif.
- Deduplication / re‑ranking saat retrieval.
- UI monitoring (Grafana/Prometheus) untuk metrik ingestion/retrieval.

---

## 13) Lisensi & Kredit

- **Docling** untuk konversi & chunking.  
- **LangChain** untuk LLM & embeddings.  
- **Qdrant/LanceDB** untuk penyimpanan vektor.

