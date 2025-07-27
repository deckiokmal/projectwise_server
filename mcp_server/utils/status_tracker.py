import json
from typing import Dict
from pathlib import Path
from mcp_server.settings import Settings

settings = Settings()


# Helper untuk menyimpan status job - Ingestion KAK/TOR dan Product
# Status disimpan dalam file JSON di direktori data
STATUS_FILE = Path(settings.status_file_path).expanduser().resolve()
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
