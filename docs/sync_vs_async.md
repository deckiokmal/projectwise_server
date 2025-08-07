## 1. Definisi Singkat

* **Sync**: Setiap permintaan (request) diproses satu per satu; thread/process akan menunggu hingga operasi I/O selesai sebelum lanjut.
* **Async**: Operasi I/O (misalnya HTTP, database, disk, GPU inference) dijalankan secara non-blocking; satu thread dapat menangani banyak permintaan bersamaan dengan `await`.

---

## 2. Kapan Menggunakan Sync

| Kasus                                                  | Alasan                                                                                         |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Kode sederhana, throughput rendah (<10 RPS)            | Overhead pengelolaan event loop tidak sebanding; sync lebih mudah dibaca dan di-debug.         |
| Operasi CPU-bound murni (komputasi internal tanpa I/O) | Async tidak membantu bila tidak ada I/O; cukup gunakan Python sync biasa atau multiprocessing. |
| Library/dependency tidak mendukung async               | Misalnya beberapa ORM atau SDK lawas tanpa async driver—forcing async malah mempersulit.       |

---

## 3. Kapan Menggunakan Async

| Kasus                                          | Alasan                                                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Web Framework                                  | **FastAPI** native async ⇒ mampu menangani ratusan RPS dengan I/O-bound tanpa menambah thread.                             |
| Inference Engine Calls ke vLLM (HTTP/gRPC)     | Pemanggilan ke service eksternal bersifat I/O-bound ⇒ async meningkatkan throughput pada single process.                   |
| Akses Vector DB & Redis                        | Driver async (e.g. `chromadb[async]`, `aioredis`) mengizinkan overlap I/O ⇒ latency p99 turun, resource CPU lebih efisien. |
| Multimodal Pre-processing (disk I/O / network) | Async file read (aiofiles), async HTTP (httpx-async) mempercepat pipeline.                                                 |
| Function Calling / Orchestration Workflow      | Task dispatching, batching, paralelisasi step via `asyncio.gather()` atau job queue (e.g. Celery with async support).      |

---

## 4. Rekomendasi Per Komponen Tech Stack

| Komponen                           | Mode         | Rekomendasi Implementasi                                                                                                                         |
| ---------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **FastAPI**                        | Async        | Gunakan `async def` di endpoint, `await llm.chat.completions.create()`. Otomatis benefit Uvicorn worker tanpa thread blocking.                   |
| **vLLM OpenAI-mode server**        | N/A          | Layanan terpisah—Flask/FASTAPI Anda cukup I/O call ke sana; penggunaan async di client meningkatkan concurrency.                                 |
| **ChromaDB / LanceDB**             | Async        | Pilih driver async (`chromadb[async]`) agar query embedding & similarity tidak block.                                                            |
| **Redis (Short-Term Memory)**      | Async        | `aioredis` atau `redis.asyncio` untuk cache session; menghindari blocking event loop.                                                            |
| **Database SQL (PostgreSQL)**      | Sync / Async | **Opsional**: untuk metadata & logs ringan, sync (`psycopg2`) cukup. Jika ingin full async pipeline, gunakan `asyncpg` + `SQLAlchemy Async ORM`. |
| **LangChain**                      | Async        | Banyak komponen support async (chains, tools). Gunakan `await chain.arun()` agar step chaining I/O-bound dapat overlap.                          |
| **Multimodal (OpenCLIP, Whisper)** | Sync         | Pre-load dan batch processing bisa di thread pool (`concurrent.futures.ThreadPoolExecutor`) — komputasi GPU-bound, async tidak signifikan.       |
| **File I/O**                       | Async        | `aiofiles` untuk load dokumen sebelum embedding.                                                                                                 |

---

## 5. Kesimpulan & Next Steps

1. **Utamakan Async** untuk semua **I/O‐bound** (HTTP, database, disk, GPU inference calls) demi **throughput tinggi**.
2. **Gunakan Sync** hanya pada bagian **CPU‐bound murni** atau library yang tidak mendukung async.
3. **Hibrid**: Anda bisa mix—misal `async def endpoint` yang di dalamnya memanggil fungsi CPU‐bound via `run_in_executor`.

### Contoh Skeleton FastAPI Async

```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
llm = OpenAI(api_key="xxx", base_url="http://vllm:8000/v1")

@app.post("/chat")
async def chat(request: dict):
    # I/O-bound ke vLLM
    resp = await llm.chat.completions.create(**request)
    return resp
```

---
Berikut panduan **step by step** untuk mengenali operasi I/O di dalam kode Python secara umum:

---

## 1. Pahami Perbedaan I/O-bound dan CPU-bound

1. **I/O-bound**: Blok kode yang waktunya didominasi oleh operasi di luar CPU—misalnya akses file, jaringan, database, atau komunikasi ke GPU/Disk.
2. **CPU-bound**: Blok kode yang waktunya didominasi oleh perhitungan murni—misalnya loop matematis, transformasi data in-memory, enkripsi, atau kompresi.

---

## 2. Identifikasi Berdasarkan Modul/Kata Kunci

### 2.1. File System

* Fungsi built-in & modul:

  * `open()`, `os.open()`, `os.listdir()`, `pathlib.Path.read_text()`
  * Library tingkat tinggi: `shutil.copy()`, `tarfile`, `zipfile`

### 2.2. Jaringan / HTTP

* Pemanggilan HTTP/socket:

  * `requests.get()`, `httpx.request()`, `urllib.request`, `socket.connect()`

### 2.3. Database

* Driver & ORM:

  * Sync: `psycopg2.connect()`, `cursor.execute()`, `session.query()`
  * Async: `asyncpg.fetch()`, `sqlalchemy.ext.asyncio`

### 2.4. Messaging & Cache

* Redis, RabbitMQ, Kafka:

  * `redis.Redis()`, `pika.BlockingConnection()`, `kafka-python`

### 2.5. GPU/Accelerator Calls

* Library: `torch.load()`, `tensorflow.keras.models.load_model()`, inferensi via HTTP/gRPC

> **Tip:** Buat daftar “hotspot” I/O di proyek Anda dengan mencari modul/modul di atas.

---

## 3. Profiling & Pengukuran Waktu

1. **cProfile**:

   ```bash
   python -m cProfile -o prof.out your_script.py
   ```

   Lalu buka dengan `snakeviz prof.out` untuk melihat fungsi yang paling lama.
2. **time.perf\_counter**:

   ```python
   import time
   start = time.perf_counter()
   data = requests.get(url)
   print("I/O took", time.perf_counter() - start, "detik")
   ```
3. **pyinstrument**:

   ```bash
   pip install pyinstrument
   pyinstrument your_script.py
   ```

   Guna menampilkan flame graph, memudahkan melihat call stack I/O-bound.

---

## 4. Monitoring `iowait` di Sistem

* Jalankan `vmstat 1` atau `top`/`htop`:

  * Kolom **wa** (iowait) tinggi → banyak operasi I/O yang membuat CPU menunggu.
  * CPU usage rendah namun program lambat → indikasi I/O-bound.

---

## 5. Linter & Static Analysis

* Beberapa IDE/Plugin (PyCharm, VSCode) memberi peringatan “blocking I/O in async function”.
* Tools seperti **Barrier** atau **flake8-asyncio** dapat menandai pemanggilan blocking di dalam `async def`.

---

## 6. Logging Berbasis Decorator

Buat decorator sederhana untuk menandai fungsi I/O:

```python
import time, functools

def io_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 0.01:  # threshold dalam detik
            print(f"[I/O] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# Contoh penggunaan
@io_profiler
def load_file(path):
    with open(path) as f:
        return f.read()
```

Decorator ini membantu Anda secara otomatis mencatat fungsi mana yang I/O-bound.

---

## 7. Kesimpulan

1. **Scan kode** untuk modul/file/network/db → itu I/O-bound.
2. **Profiling** (cProfile, pyinstrument) → lihat durasi fungsi.
3. **Monitoring sistem** (`iowait`) → konfirmasi di level OS.
4. **Static analysis** & **IDE hints** → identifikasi blocking I/O dalam async context.
5. **Automasi** dengan decorator → catat runtime I/O tanpa perlu profiling manual.

Dengan langkah-langkah di atas, Anda dapat **konsisten** mengenali dan memisahkan operasi I/O di kode Python, sehingga memudahkan keputusan **async vs sync** di arsitektur Anda.
