from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List

# ============================================================
# Sanitasi & Regex
# ============================================================
RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")  # non-alfanumerik -> underscore
RE_MULTI_UNDER = re.compile(r"_+")
ILLEGAL_WIN_CHARS = r'<>:"/\\|?*'  # karakter ilegal utk komponen nama file (Windows)


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _sanitize_component(name: str) -> str:
    """
    Sanitasi SEDERHANA untuk komponen path/filename:
    - lowercase + hapus aksen
    - ganti karakter ilegal & non-alfanumerik -> underscore
    - rapikan underscore
    """
    name = (_strip_accents(name) if name else "").lower().strip().rstrip(". ")
    if not name:
        return "untitled"
    name = "".join("_" if c in ILLEGAL_WIN_CHARS else c for c in name)
    name = RE_NON_ALNUM.sub("_", name)
    name = RE_MULTI_UNDER.sub("_", name).strip("_")
    return name or "untitled"


# ============================================================
# Struktur hasil resolusi
# ============================================================
@dataclass(frozen=True)
class PathInfo:
    project_root: Path
    base_dir_abs: Path
    filename_final: str
    full_path: Path
    existed_before: bool
    created_dirs: bool
    message: str


# ============================================================
# Project root & path guard
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
# Normalisasi nama file .docx
# ============================================================
def normalize_docx_filename(src: str) -> str:
    """
    Nama file DOCX -> <basename>.docx (lowercase, spasi -> underscore).
    """
    base = os.path.basename((src or "").strip()).rstrip(" .")
    base, _ext = os.path.splitext(base)
    base = _sanitize_component(base)
    return f"{base}.docx"


# ============================================================
# Resolver TEMPLATE (global library)
# ============================================================
def resolve_proposal_template_docx(
    settings,
    template_name: str,
    *,
    category: Optional[str] = None,  # opsional: subfolder kategori
    subdirs: Optional[Iterable[str]] = None,  # opsional: subfolder bebas (urut)
    project_root: Optional[Path] = None,
    create_dirs: bool = False,  # default False: template biasanya read-only
) -> PathInfo:
    """
    <base = settings.proposal_template_base_path> / [category|subdirs...] / <template_name>.docx
    """
    pr = (project_root or find_project_root()).resolve()

    # susun relative path: base / (opsional kategori/subdirs)
    base_rel = _ensure_rel(settings.proposal_template_base_path)

    parts: List[str] = [base_rel]
    if category:
        parts.append(_sanitize_component(category))
    if subdirs:
        parts.extend(_sanitize_component(s) for s in subdirs)
    tier_rel = "/".join(parts)

    base_abs = (pr / _ensure_rel(tier_rel)).resolve()
    common = os.path.commonpath([str(base_abs), str(pr)])
    if Path(common) != pr:
        raise ValueError(f"Base dir keluar dari project root: {base_abs}")

    created_dirs = False
    if create_dirs and not base_abs.exists():
        base_abs.mkdir(parents=True, exist_ok=True)
        created_dirs = True

    if not base_abs.exists() or not base_abs.is_dir():
        # Untuk template, biasanya kita ingin tahu jika folder belum ada.
        raise FileNotFoundError(f"Folder template tidak ditemukan: {base_abs}")

    fname = normalize_docx_filename(template_name)
    full = _join_and_resolve(pr, tier_rel, fname)

    existed = full.exists()
    msg = "OK" if existed else "Template belum ada."
    return PathInfo(
        project_root=pr,
        base_dir_abs=base_abs,
        filename_final=fname,
        full_path=full,
        existed_before=existed,
        created_dirs=created_dirs,
        message=msg,
    )


def list_proposal_templates(
    settings,
    *,
    category: Optional[str] = None,
    subdirs: Optional[Iterable[str]] = None,
    project_root: Optional[Path] = None,
    pattern: str = "*.docx",
) -> list[Path]:
    """
    Daftar file template (*.docx) di bawah base template (opsional kategori/subdirs).
    """
    pr = (project_root or find_project_root()).resolve()
    base_rel = _ensure_rel(settings.proposal_template_base_path)
    parts: List[str] = [base_rel]
    if category:
        parts.append(_sanitize_component(category))
    if subdirs:
        parts.extend(_sanitize_component(s) for s in subdirs)
    tier_rel = "/".join(parts)

    base_abs = (pr / _ensure_rel(tier_rel)).resolve()
    if not base_abs.exists():
        return []
    return list(base_abs.glob(pattern))


# ============================================================
# Resolver GENERATED (per pelanggan/tahun)
# ============================================================
def resolve_proposal_generated_docx(
    settings,
    pelanggan: str,
    tahun: str,
    filename: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """
    <base = settings.proposal_generated_base_path>/<pelanggan>/<tahun>/<filename>.docx
    """
    pr = (project_root or find_project_root()).resolve()

    base_rel = _ensure_rel(settings.proposal_generated_base_path)
    pelanggan_dir = _sanitize_component(pelanggan)
    tahun_dir = _sanitize_component(str(tahun))
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
        raise FileNotFoundError(f"Folder generated tidak ditemukan: {base_abs}")

    fname = normalize_docx_filename(filename)
    full = _join_and_resolve(pr, tier_rel, fname)

    existed = full.exists()
    if existed and unique:
        stem = full.stem
        parent = full.parent
        i = 1
        while full.exists():
            full = parent / f"{stem} ({i}).docx"
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
        full_path=full,
        existed_before=existed,
        created_dirs=created_dirs,
        message=msg,
    )


# ============================================================
# I/O util – atomic write, read, delete
# ============================================================
def write_bytes_atomic(target: Path, data: bytes) -> None:
    """
    Tulis bytes secara atomic:
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


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
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
# API siap pakai
# ============================================================
def open_template_docx(
    settings,
    template_name: str,
    *,
    category: Optional[str] = None,
    subdirs: Optional[Iterable[str]] = None,
) -> bytes:
    info = resolve_proposal_template_docx(
        settings, template_name, category=category, subdirs=subdirs, create_dirs=False
    )
    if not info.full_path.exists():
        raise FileNotFoundError(f"Template tidak ditemukan: {info.full_path}")
    return read_bytes(info.full_path)


def save_generated_docx(
    settings,
    pelanggan: str,
    tahun: str,
    filename: str,
    data: bytes,
    *,
    unique: bool = False,
) -> PathInfo:
    info = resolve_proposal_generated_docx(
        settings, pelanggan, tahun, filename, create_dirs=True, unique=unique
    )
    write_bytes_atomic(info.full_path, data)
    return info


def open_generated_docx(settings, pelanggan: str, tahun: str, filename: str) -> bytes:
    info = resolve_proposal_generated_docx(
        settings, pelanggan, tahun, filename, create_dirs=False, unique=False
    )
    return read_bytes(info.full_path)


def remove_generated_docx(settings, pelanggan: str, tahun: str, filename: str) -> bool:
    info = resolve_proposal_generated_docx(
        settings, pelanggan, tahun, filename, create_dirs=False, unique=False
    )
    return delete_file(info.full_path)
