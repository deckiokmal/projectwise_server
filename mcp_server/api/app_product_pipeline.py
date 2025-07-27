# File: mcp_server/app_product_pipeline.py

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from mcp_server.settings import Settings
from mcp_server.utils.helper import slugify
from mcp_server.utils.status_tracker import save_status
from mcp_server.tools.rag_tools import RAGTools

router = APIRouter()
settings = Settings()
rag_tools = RAGTools()


@router.post("/upload-product/")
async def upload_product(
    background_tasks: BackgroundTasks,
    category: str = Form(...),
    product_name: str = Form(...),
    tahun: int = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):  # type: ignore
        raise HTTPException(400, "Hanya menerima file PDF")

    category_slug = slugify(category)
    product_slug = slugify(product_name)
    tahun_str = str(tahun)

    out_dir = (
        Path(settings.product_base_path) / category_slug / tahun_str / product_slug
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"{product_slug}.pdf"
    saved_path = out_dir / saved_filename
    saved_path.write_bytes(await file.read())

    job_id = str(uuid.uuid4())

    unique_key = f"{category_slug}_{tahun_str}__{product_slug}"
    if unique_key in rag_tools._manifest:
        save_status(job_id, "skipped", "File sudah pernah diingest sebelumnya")
        return JSONResponse(
            {
                "status": "skipped",
                "file": str(saved_path),
                "job_id": job_id,
                "message": "File sudah pernah diingest sebelumnya dan tidak diproses ulang.",
            }
        )

    save_status(job_id, "pending", "File produk tersimpan. Menunggu ingestion")

    response_data = {
        "status": "tersimpan",
        "file": str(saved_path),
        "job_id": job_id,
        "message": "File berhasil disimpan, ingestion sedang dijalankan sebagai background task.",
    }

    background_tasks.add_task(
        process_product_pipeline,
        saved_filename,
        category,
        product_name,
        tahun_str,
        job_id,
    )

    return JSONResponse(response_data)


async def process_product_pipeline(filename, category, product_name, tahun, job_id):
    try:
        save_status(job_id, "running", "Sedang ingestion dan ringkasan produk")

        result = await rag_tools.ingest_product_knowledge_chunks(
            filename=filename,
            product_name=product_name,
            category=category,
            tahun=tahun,
            overwrite=False,
        )

        if result.get("status") != "success":
            raise Exception(result.get("error", "Gagal ingest produk."))

        save_status(
            job_id, "success", "Produk berhasil diingest dan diringkas.", result
        )

    except Exception as e:
        save_status(job_id, "failure", f"Gagal proses produk: {e}")
