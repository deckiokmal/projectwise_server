# mcp_server/utils/ingestion_manifest_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping, Dict

from mcp_server.utils.logger import get_logger
from mcp_server.settings import Settings

# kita manfaatkan util atomic write dari path utils biar konsisten & aman
from mcp_server.utils.kak_path_utils import write_text_atomic


logger = get_logger(__name__)
settings = Settings()


def load_manifest(manifest_path: str | Path) -> dict:
    """
    Baca manifest JSON. Kalau file belum ada / rusak → kembalikan {}.
    Tidak melempar exception demi kestabilan pipeline.
    """
    p = Path(manifest_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Manifest rusak di %s: %s — memakai {}.", p, e)
        return {}


def save_manifest(
    manifest_path: str | Path, manifest: Mapping | MutableMapping
) -> None:
    """
    Simpan manifest JSON secara atomic. Jika gagal, log peringatan & lanjutkan.
    """
    p = Path(manifest_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(manifest), ensure_ascii=False, indent=2)
        write_text_atomic(p, payload, encoding="utf-8")
    except Exception as e:
        logger.warning("Gagal menyimpan manifest %s: %s", p, e)


# Helper untuk menyimpan status job - Ingestion KAK/TOR dan Product
# Status disimpan dalam file JSON di direktori data
STATUS_FILE = Path(settings.json_ingestion_status_file).expanduser().resolve()
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)


def save_status(job_id: str, status: str, message: str = "", result: Dict = {}):
    try:
        if STATUS_FILE.exists():
            all_status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        else:
            all_status = {}
        all_status[job_id] = {
            "status": status,
            "message": message,
            "result": result,
        }
        STATUS_FILE.write_text(json.dumps(all_status, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Gagal simpan status: {e}")


def get_status(job_id: str) -> Dict:
    try:
        if not STATUS_FILE.exists():
            return {"status": "not_found"}
        all_status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return all_status.get(job_id, {"status": "not_found"})
    except Exception:
        return {"status": "error"}


# utility build data manifest konsisten kak dan product
def build_kak_unique_key(pelanggan_final: str, tahun: str, project_final: str) -> str:
    return f"{pelanggan_final}_{tahun}_{project_final}"


def build_product_unique_key(
    product_final: str, tahun: str, category_final: str
) -> str:
    return f"{category_final}_{tahun}_{product_final}"
