# File: mcp_server/app_kak_pipeline.py

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


@router.post("/upload-kak-tor/")
async def upload_kak_tor(
    background_tasks: BackgroundTasks,
    project_name: str = Form(...),
    pelanggan: str = Form(...),
    tahun: int = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):  # type: ignore
        raise HTTPException(400, "Hanya file PDF yang diperbolehkan.")

    pelanggan_slug = slugify(pelanggan)
    project_slug = slugify(project_name)
    tahun_str = str(tahun)

    out_dir = Path(settings.kak_tor_base_path) / pelanggan_slug / tahun_str
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"{project_slug}.pdf"
    saved_path = out_dir / saved_filename
    content = await file.read()
    saved_path.write_bytes(content)

    job_id = str(uuid.uuid4())

    # cek status ingestion dari rag_tools
    unique_key = f"{pelanggan_slug}_{tahun_str}_{saved_filename}"
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

    save_status(job_id, "pending", "File tersimpan. Menunggu ingestion KAK/TOR")

    # Kirim feedback awal ke user
    response_data = {
        "status": "tersimpan",
        "file": str(saved_path),
        "job_id": job_id,
        "message": "File berhasil disimpan, ingestion sedang dijalankan sebagai background task.",
    }

    background_tasks.add_task(
        process_kak_pipeline,
        saved_filename,
        pelanggan,
        project_name,
        tahun_str,
        job_id,
    )

    return JSONResponse(response_data)


async def process_kak_pipeline(filename, pelanggan, project, tahun, job_id):
    try:
        save_status(job_id, "running", "Ingestion dan ringkasan sedang diproses")

        ingest_result = await rag_tools.ingest_kak_tor_chunks(
            filename=filename,
            pelanggan=pelanggan,
            project=project,
            tahun=tahun,
            overwrite=False,
        )

        if ingest_result.get("status") != "success":
            raise Exception(ingest_result.get("error", "Gagal ingest KAK/TOR."))

        kak_md_name = Path(ingest_result["markdown_file"]).name

        summary = await rag_tools.build_summary_kak_payload_and_summarize(
            kak_tor_name=kak_md_name,
            pelanggan=pelanggan,
            project=project,
            tahun=tahun,
        )

        save_status(job_id, "success", "KAK berhasil diingest dan diringkas.", summary)

    except Exception as e:
        save_status(job_id, "failure", f"Gagal proses ingestion KAK: {e}")
