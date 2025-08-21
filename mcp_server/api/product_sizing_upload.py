# mcp_server/api/product_sizing_upload.py
from __future__ import annotations

import uuid
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from mcp_server.tools.product_sizing_tool import ProductTools
from mcp_server.settings import Settings
from mcp_server.utils.ingestion_manifest_utils import (
    save_status,
    build_product_unique_key,
)
from mcp_server.utils.product_path_utils import save_product_pdf

router = APIRouter()
settings = Settings()
product_tools = ProductTools()

PDF_MAX_MB = 100  # batas ukuran file (opsional)


# ====================
# Internal Utility
# ====================
async def _read_pdf_bytes(file: UploadFile, max_mb: int = PDF_MAX_MB) -> bytes:
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            HTTP_400_BAD_REQUEST, detail=f"Ukuran file melebihi {max_mb} MB"
        )
    return data


def _background_ingest_sync(
    filename: str, category: str, product: str, tahun: str, job_id: str
):
    """
    Wrapper sinkron agar cocok dengan Starlette BackgroundTasks.
    Menjalankan coroutine async `process_ingest_product_file` di event loop terpisah.
    """
    import anyio

    anyio.run(process_ingest_product_file, filename, category, product, tahun, job_id)


@router.post("/upload-product/")
async def upload_product(
    background_tasks: BackgroundTasks,
    product: str = Form(...),
    category: str = Form(...),
    tahun: str = Form(...),
    file: UploadFile = File(...),
):
    """
    status: skipped, pending, running, success, error
    """
    # 1) Baca bytes file (perbaikan tipe untuk save_kak_pdf)
    data = await _read_pdf_bytes(file)

    # 2) Simpan file PDF
    try:
        pdf_info = save_product_pdf(
            settings,
            category=category,
            product=product,
            tahun=tahun,
            filename=str(file.filename),
            data=data,  # <-- bytes sesuai tipe fungsi
            unique=True,
        )
    finally:
        # Tutup UploadFile agar tidak bocor descriptor
        await file.close()

    # 3) Generate job_id
    job_id = str(uuid.uuid4())
    unique_key = build_product_unique_key(
        category_final=pdf_info.category_final,
        tahun=tahun,
        product_final=pdf_info.product_final,
    )

    # 4) Cek status ingestion dari kak_tools (berdasar key konten, bukan nama unik)
    if (
        getattr(product_tools, "_manifest", None)
        and unique_key in product_tools._manifest
    ):
        save_status(
            job_id,
            "skipped",
            "File sudah pernah diingest sebelumnya",
            result={"unique_key": unique_key},
        )
        return JSONResponse(
            {
                "status": "skipped",
                "file": str(file.filename),
                "job_id": job_id,
                "message": "File sudah pernah diingest sebelumnya dan tidak diproses ulang.",
            }
        )

    # 5) Update status pending
    save_status(
        job_id,
        "pending",
        "File tersimpan. Menunggu ingestion Product",
        result={"unique_key": unique_key},
    )

    # 6) Kirim feedback awal ke user
    response_data = {
        "status": "tersimpan",
        "file": pdf_info.filename_final,
        "job_id": job_id,
        "message": "File berhasil disimpan, ingestion berjalan di background task.",
    }

    # 7) Jalankan ingestion & summary di background (pakai wrapper sinkron)
    background_tasks.add_task(
        _background_ingest_sync,
        pdf_info.filename_final,
        pdf_info.product_final,
        pdf_info.category_final,
        tahun,
        job_id,
    )

    return JSONResponse(response_data)


async def process_ingest_product_file(
    filename: str, category: str, product: str, tahun: str, job_id: str
):
    # 1) update status running manifest
    save_status(
        job_id,
        "running",
        "Ingestion dan ringkasan sedang diproses",
        result={"filename": filename},
    )

    try:
        # 2) ingestion proses
        ingest_result = await product_tools.ingest_product_file(
            filename=filename,
            category=category,
            product=product,
            tahun=tahun,
            overwrite=False,
        )

        if ingest_result.get("status") != "success":
            raise RuntimeError(
                ingest_result.get("message", "Product tersimpan tapi gagal ingest.")
            )

        # update status ingestion
        save_status(
            job_id,
            "running",
            "Product berhasil diingest ke knowledge base.",
            result=ingest_result,
        )

        # 3) summary analysis
        summary = await product_tools.generate_product_summarize(
            filename=filename,
            category=category,
            product=product,
            tahun=tahun,
        )

        # update status summary analysis
        save_status(
            job_id,
            "success",
            "Summary analysis berhasil dibuat.",
            result=summary,
        )

    except Exception as e:
        # potong pesan jika terlalu panjang
        msg = f"Gagal proses ingestion Product: {e}"
        save_status(job_id, "error", msg[:2000])
