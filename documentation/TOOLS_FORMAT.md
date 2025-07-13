## Panduan Format Input & Output MCP Server **ProjectWise**

_(berlaku untuk endpoint SSE `http://<host>:5000/sse`)_

Seluruh _tool_ di-bawah dipanggil sebagai **function call** (OpenAI tools) dan **mengembalikan JSON** karena setiap handler didekorasi dengan `structured_output=True` .
Kecuali disebutkan berbeda, **semua argumen bersifat opsional** dan nilai default-nya ditunjukkan pada kolom _Default_.

---

### 1. `add_product_knowledge`

| Parameter      | Tipe          | Default                        | Keterangan               |
| -------------- | ------------- | ------------------------------ | ------------------------ |
| `base_dir`     | `str \| null` | _Settings.knowledge_base_path_ | Folder PDF produk        |
| `project_name` | `str`         | `"product_standard"`           | Label metadata `project` |
| `tahun`        | `str`         | `"2025"`                       | Label metadata `tahun`   |

**Output**

```json
{ "status": "Product knowledge ingestion selesai." }
```

---

### 2. `add_kak_tor_knowledge`

| Parameter  | Tipe          | Default                         | Keterangan             |
| ---------- | ------------- | ------------------------------- | ---------------------- |
| `base_dir` | `str \| null` | _Settings.kak_tor_base_path_    | Folder PDF KAK/TOR     |
| `md_dir`   | `str \| null` | _Settings.kak_tor_md_base_path_ | Folder output Markdown |
| `project`  | `str \| null` | `null`                          | Label metadata         |
| `tahun`    | `str \| null` | `null`                          | Label metadata         |

**Output**

```json
{ "status": "KAK/TOR ingestion selesai." }
```

---

### 3. `add_kak_tor_md_knowledge`

| Parameter       | Tipe          | Default                         |
| --------------- | ------------- | ------------------------------- |
| `markdown_path` | `str \| null` | _Settings.kak_tor_md_base_path_ |
| `project`       | `str`         | `"default"`                     |
| `tahun`         | `str`         | `"2025"`                        |

**Output**

```json
{ "results": [ ... ] }   // list ringkasan tiap file
```

---

### 4. `build_instruction_context`

| Parameter        | Tipe                | Keterangan                                              |
| ---------------- | ------------------- | ------------------------------------------------------- |
| `template_name`  | `str`               | Nama file `.txt` di folder _templates_ (tanpa ekstensi) |
| `kak_md_dir`     | `str \| null`       | Folder Markdown sumber                                  |
| `selected_files` | `List[str] \| null` | Daftar nama `.md` (jika hanya sebagian)                 |

**Output**

```json
{
  "instruksi": "<prompt LLM>",
  "context": "<gabungan markdown>"
}
```

---

### 5. `rag_retrieval`

| Parameter         | Tipe           | Default                       | Keterangan                               |
| ----------------- | -------------- | ----------------------------- | ---------------------------------------- |
| `query`           | `str`          | —                             | Kalimat tanya/kata kunci                 |
| `k`               | `int \| null`  | _Settings.retriever_search_k_ |                                          |
| `metadata_filter` | `dict \| null` | `null`                        | Contoh: `{"project":"A","tahun":"2025"}` |

**Output**

```json
{ "result": "<teks+citation>" }
```

---

### 6. `reset_vectordb`

**Input** : —
**Output**

```json
{ "status": "Vectorstore berhasil di-reset." }
```

---

### 7. `update_chunk_metadata`

| Parameter         | Tipe   |
| ----------------- | ------ |
| `metadata_filter` | `dict` |
| `new_metadata`    | `dict` |

**Output**

```json
{ "updated": <int> }   // jumlah chunk ter-update
```

---

### 8. `get_vectorstore_stats`

**Input** : —
**Output**

```json
{
  "total_rows": <int>,
  "size_mb": <float>,
  "projects": [ "<project1>", ... ],
  "tahun_distribution": { "2024": 120, "2025": 340, ... }
}
```

---

### 9. `rebuild_all_embeddings`

| Parameter    | Tipe  | Default                                        |
| ------------ | ----- | ---------------------------------------------- |
| `batch_size` | `int` | `100` _(tidak dipakai, semua chunk sekaligus)_ |

**Output**

```json
{ "status": "Rebuild embeddings selesai." }
```

---

### 10. `list_metadata_values`

| Parameter | Tipe  | Contoh                  |
| --------- | ----- | ----------------------- |
| `field`   | `str` | `"project"` / `"tahun"` |

**Output**

```json
{ "values": [ "Alpha", "Beta", ... ] }
```

---

### 11. `retrieve_product_context`

| Parameter         | Tipe           | Default            | Keterangan                      |
| ----------------- | -------------- | ------------------ | ------------------------------- |
| `product`         | `str`          | —                  | Nama/deskripsi produk           |
| `k`               | `int \| null`  | _pipeline default_ |                                 |
| `metadata_filter` | `dict \| null` | `null`             |                                 |
| `prompt_template` | `str \| null`  | `null`             | Nama file `.txt` tanpa ekstensi |

**Output – Sukses**

```json
{
  "status": "success",
  "context": "<teks chunk>",
  "instruction": "<prompt template atau empty>"
}
```

**Output – Gagal**

```json
{ "status": "failure", "error": "<pesan>" }
```

---

### 12. `extract_document_text`

| Parameter   | Tipe  |
| ----------- | ----- |
| `file_path` | `str` |

**Output – Sukses**

```json
{ "status": "success", "text": "<markdown>" }
```

**Output – Gagal**

```json
{ "status": "failure", "error": "<pesan>" }
```

---

### 13. `generate_proposal_docx`

| Parameter           | Tipe          | Keterangan                                   |
| ------------------- | ------------- | -------------------------------------------- |
| `context`           | `dict`        | Variabel-variabel _docxtpl_ (lihat template) |
| `override_template` | `str \| null` | Path `.docx` pengganti                       |

**Output – Sukses**

```json
{
  "status": "success",
  "product": "<judul_proposal | nama_pelanggan>",
  "path": "<lokasi file .docx>"
}
```

**Output – Gagal**

```json
{ "status": "failure", "error": "<pesan>" }
```

---

### Cara Umum Memanggil Tool

```python
response = await session.call_tool("<tool_name>", {<args>})
result_dict = json.loads(response.content[0].text)
```

Pastikan tool name cocok dengan daftar di atas dan argumen dikirim sebagai **JSON serialisable**.
