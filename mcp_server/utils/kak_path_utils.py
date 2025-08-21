# mcp_server/utils/kak_path_utils.py
from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ============================================================
# Sanitasi & Regex
# ============================================================
RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")  # non-alfanumerik -> underscore
RE_MULTI_UNDER = re.compile(r"_+")
RE_TR_SUMMARY = re.compile(r"(?i)(?:[_\-\s])?summary$")  # akhiran "summary"
ILLEGAL_WIN_CHARS = r'<>:"/\\|?*'  # karakter ilegal utk komponen nama file (Windows)
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _sanitize_component(name: str) -> str:
    """
    Sanitasi SEDERHANA untuk komponen path/filename:
    - lowercase + hapus aksen
    - ganti karakter ilegal & non-alfanumerik -> underscore
    - rapikan underscore (tak boleh di awal/akhir)
    """
    name = (_strip_accents(name) if name else "").lower().strip().rstrip(". ")
    if not name:
        return "untitled"
    name = "".join("_" if c in ILLEGAL_WIN_CHARS else c for c in name)
    name = RE_NON_ALNUM.sub("_", name)
    name = RE_MULTI_UNDER.sub("_", name).strip("_")
    return name or "untitled"


def _avoid_windows_reserved(stem: str) -> str:
    return f"{stem}_" if stem in _WINDOWS_RESERVED else stem


# ============================================================
# Struktur hasil resolusi
# ============================================================
@dataclass(frozen=True)
class PathInfo:
    project_root: Path
    base_dir_abs: Path
    filename_final: str
    pelanggan_final: str
    project_final: str
    full_path: Path
    existed_before: bool
    created_dirs: bool
    message: str


# ============================================================
# Project root helper
# ============================================================
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Cari root proyek via marker umum. Fallback ke CWD bila tak ada.
    """
    markers = {".git", "pyproject.toml", "requirements.txt", ".projectroot", ".env"}
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    return Path.cwd().resolve()


def _ensure_rel(s: str) -> str:
    """
    Paksa path jadi RELATIF (hapus drive & leading slash).
    """
    s = (s or "").strip()
    s = re.sub(r"^[a-zA-Z]:[\\/]*", "", s)  # buang drive Windows
    return s.lstrip("/\\") or "."


def _join_and_resolve(project_root: Path, base_rel: str, filename: str) -> Path:
    """
    Gabung & resolve path, pastikan tetap di bawah project_root (anti traversal).
    """
    base_rel = _ensure_rel(base_rel)
    filename = os.path.basename(filename)  # cegah sisipan path
    candidate = (project_root / base_rel / filename).resolve()
    common = os.path.commonpath([str(candidate), str(project_root)])
    if Path(common) != project_root:
        raise ValueError(f"Path keluar dari project root: {candidate}")
    return candidate


# ============================================================
# Normalisasi nama file per jenis
# ============================================================
def _norm_pdf_filename(src: str) -> str:
    # Deprecated: gunakan _norm_pdf_from_project(project)
    base = _sanitize_component(src)
    return f"{_avoid_windows_reserved(base)}.pdf"


def _norm_md_filename(src: str) -> str:
    # Deprecated: gunakan _norm_md_from_project(project)
    base = _sanitize_component(src)
    return f"{_avoid_windows_reserved(base)}.md"


def _norm_summary_md_filename(src: str) -> str:
    # Deprecated: summary kini tanpa suffix. Delegasi ke _norm_md_from_project.
    base = _sanitize_component(src)
    return f"{_avoid_windows_reserved(base)}.md"


def _project_to_stem(project: str) -> str:
    """
    Ambil nama dasar (stem) dari 'project' yang sudah disanitasi & aman.
    """
    stem = _avoid_windows_reserved(_sanitize_component(project or ""))
    return stem or "untitled"


def _norm_pdf_from_project(project: str) -> str:
    return f"{_project_to_stem(project)}.pdf"


def _norm_md_from_project(project: str) -> str:
    return f"{_project_to_stem(project)}.md"


# ============================================================
# Resolver generik per base + (pelanggan/tahun)
# ============================================================
def _resolve_under_base(
    base_dir_rel: str,
    pelanggan: str,
    project: str,
    tahun: str,
    filename_final: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """_summary_

    Args:
        base_dir_rel (str): _description_
        pelanggan (str): _description_
        project (str): _description_
        tahun (str): _description_
        filename_final (str): _description_
        project_root (Optional[Path], optional): _description_. Defaults to None.
        create_dirs (bool, optional): _description_. Defaults to True.
        unique (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        FileNotFoundError: _description_

    Returns:
        PathInfo: _description_
    """
    pr = (project_root or find_project_root()).resolve()

    # susun <base>/<pelanggan>/<tahun>
    base_rel = _ensure_rel(base_dir_rel)
    pelanggan_dir = _avoid_windows_reserved(_sanitize_component(pelanggan))
    project_sanitize = _avoid_windows_reserved(_sanitize_component(project))
    tahun_dir = _avoid_windows_reserved(_sanitize_component(str(tahun)))
    tier_rel = f"{base_rel}/{pelanggan_dir}/{tahun_dir}"

    base_abs = (pr / _ensure_rel(tier_rel)).resolve()
    common = os.path.commonpath([str(base_abs), str(pr)])
    if Path(common) != pr:
        raise ValueError(f"Base dir keluar dari project root: {base_abs}")

    created_dirs = False
    if create_dirs and not base_abs.exists():
        base_abs.mkdir(parents=True, exist_ok=True)
        created_dirs = True

    if not base_abs.exists() or not base_abs.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {base_abs}")

    full = _join_and_resolve(pr, tier_rel, filename_final)
    existed = full.exists()

    if existed and unique:
        stem = full.stem
        parent = full.parent
        i = 1
        while full.exists():
            full = parent / f"{stem} ({i}){full.suffix}"
            i += 1

    if existed and not unique:
        msg = "File sudah ada dan akan ditimpa."
    elif existed and unique:
        msg = "Nama unik dipakai karena file sudah ada."
    elif created_dirs:
        msg = "Folder dibuat dan path siap."
    else:
        msg = "OK"

    return PathInfo(
        project_root=pr,
        base_dir_abs=base_abs,
        filename_final=full.name,
        pelanggan_final=pelanggan_dir,
        project_final=project_sanitize,
        full_path=full,
        existed_before=existed,
        created_dirs=created_dirs,
        message=msg,
    )


# ============================================================
# Resolver spesifik (sesuai Settings.*)
# ============================================================
def resolve_kak_pdf(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: str,  # dipertahankan demi kompatibilitas, TIDAK dipakai
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.kak_tor_base_path>/<pelanggan>/<tahun>/<project>.pdf"""
    fname = _norm_pdf_from_project(project)  # <-- KUNCI: dari project
    return _resolve_under_base(
        settings.kak_tor_base_path,
        pelanggan,
        project,
        tahun,
        fname,
        project_root=project_root,
        create_dirs=create_dirs,
        unique=unique,
    )


def resolve_kak_md(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: str,  # dipertahankan demi kompatibilitas, TIDAK dipakai
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.kak_tor_md_base_path>/<pelanggan>/<tahun>/<project>.md"""
    fname = _norm_md_from_project(project)  # <-- KUNCI: dari project
    return _resolve_under_base(
        settings.kak_tor_md_base_path,
        pelanggan,
        project,
        tahun,
        fname,
        project_root=project_root,
        create_dirs=create_dirs,
        unique=unique,
    )


def resolve_kak_summary_md(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: str,  # dipertahankan demi kompatibilitas, TIDAK dipakai
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.kak_tor_summaries_base_path>/<pelanggan>/<tahun>/<project>.md"""
    # Tidak ada suffix '_summary' lagi
    fname = _norm_md_from_project(project)  # <-- KUNCI: dari project
    return _resolve_under_base(
        base_dir_rel=settings.kak_tor_summaries_base_path,
        pelanggan=pelanggan,
        project=project,
        tahun=tahun,
        filename_final=fname,
        project_root=project_root,
        create_dirs=create_dirs,
        unique=unique,
    )


# ============================================================
# I/O util – atomic write, read, delete
# ============================================================
def write_bytes_atomic(target: Path, data: bytes) -> None:
    """
    Tulis bytes secara atomic (untuk PDF):
    - tulis ke file sementara di folder yang sama
    - fsync
    - replace -> hindari partial write
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if hasattr(os, "replace"):
            os.replace(tmp_path, target)
        else:
            shutil.move(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def write_text_atomic(target: Path, content: str, *, encoding: str = "utf-8") -> None:
    """
    Tulis text secara atomic (untuk MD):
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if hasattr(os, "replace"):
            os.replace(tmp_path, target)
        else:
            shutil.move(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_text(path: Path, *, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def delete_file(path: Path) -> bool:
    """
    Hapus file; return True jika terhapus, False jika tidak ada.
    Raise exception bila path adalah folder.
    """
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        raise IsADirectoryError(f"Target adalah folder: {p}")
    p.unlink()
    return True


# ============================================================
# Helper exist dan list
# ============================================================
def exists_kak_pdf(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_pdf(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return info.full_path.exists()


def exists_kak_md(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return info.full_path.exists()


def exists_kak_summary(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_summary_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return info.full_path.exists()


def list_kak_files(
    settings, base_attr: str, pelanggan: str, tahun: str, pattern: str = "*"
):
    root = find_project_root()
    base_rel = getattr(settings, base_attr)
    from_path = (
        root
        / base_rel
        / _sanitize_component(pelanggan)
        / _sanitize_component(str(tahun))
    ).resolve()
    return list(from_path.glob(pattern)) if from_path.exists() else []


# ============================================================
# API siap pakai (ringkas)
# ============================================================
def save_kak_pdf(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: Optional[str],  # diabaikan
    data: bytes,
    *,
    unique=False,
) -> PathInfo:
    info = resolve_kak_pdf(
        settings,
        pelanggan,
        project,
        tahun,
        project,  # pakai project untuk konsistensi
        create_dirs=True,
        unique=unique,
    )
    write_bytes_atomic(info.full_path, data)
    return info


def open_kak_pdf(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bytes:
    info = resolve_kak_pdf(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return read_bytes(info.full_path)


def remove_kak_pdf(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_pdf(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return delete_file(info.full_path)


def save_kak_md(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: Optional[str],  # diabaikan
    markdown: str,
    *,
    unique=False,
) -> PathInfo:
    info = resolve_kak_md(
        settings, pelanggan, project, tahun, project, create_dirs=True, unique=unique
    )
    write_text_atomic(info.full_path, markdown)
    return info


def open_kak_md(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> str:
    info = resolve_kak_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return read_text(info.full_path)


def remove_kak_md(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return delete_file(info.full_path)


def save_kak_summary(
    settings,
    pelanggan: str,
    project: str,
    tahun: str,
    filename: Optional[str],  # diabaikan
    markdown: str,
    *,
    unique=False,
) -> PathInfo:
    info = resolve_kak_summary_md(
        settings, pelanggan, project, tahun, project, create_dirs=True, unique=unique
    )
    write_text_atomic(info.full_path, markdown)
    return info


def open_kak_summary(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> str:
    info = resolve_kak_summary_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return read_text(info.full_path)


def remove_kak_summary(
    settings, pelanggan: str, project: str, tahun: str, filename: Optional[str] = None
) -> bool:
    info = resolve_kak_summary_md(
        settings,
        pelanggan,
        project,
        tahun,
        filename or project,
        create_dirs=False,
        unique=False,
    )
    return delete_file(info.full_path)
