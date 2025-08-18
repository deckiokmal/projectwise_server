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
    """
    Nama file PDF -> <basename>.pdf
    - dari argumen filename (tanpa tambahan '_summary')
    - lowercase, spasi → underscore, sanitasi karakter
    """
    base = os.path.basename((src or "").strip()).rstrip(" .")
    base, _ext = os.path.splitext(base)
    base = _sanitize_component(base)
    return f"{base}.pdf"


def _norm_md_filename(src: str) -> str:
    """
    Nama file MD -> <basename>.md (untuk direktori product_tor_md)
    - dari argumen filename (tanpa tambahan '_summary')
    - lowercase, spasi → underscore, sanitasi karakter
    """
    base = os.path.basename((src or "").strip()).rstrip(" .")
    base, _ext = os.path.splitext(base)
    base = _sanitize_component(base)
    return f"{base}.md"


def _norm_summary_md_filename(src: str) -> str:
    """
    Nama file SUMMARY MD -> <basename>_summary.md
    - pastikan suffix '_summary' meskipun user sudah tulis 'summary' dgn variasi
    """
    base = os.path.basename((src or "").strip()).rstrip(" .")
    base, _ext = os.path.splitext(base)
    base = _sanitize_component(base)
    if RE_TR_SUMMARY.search(base):
        base = RE_TR_SUMMARY.sub("", base).rstrip("_- ")
    base = f"{base}_summary" if base else "summary"
    return f"{base}.md"


# ============================================================
# Resolver generik per base + (product/tahun)
# ============================================================
def _resolve_under_base(
    base_dir_rel: str,
    product: str,
    tahun: str,
    filename_final: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    pr = (project_root or find_project_root()).resolve()

    # susun <base>/<product>/<tahun>
    base_rel = _ensure_rel(base_dir_rel)
    product_dir = _sanitize_component(product)
    tahun_dir = _sanitize_component(str(tahun))
    tier_rel = f"{base_rel}/{product_dir}/{tahun_dir}"

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
        full_path=full,
        existed_before=existed,
        created_dirs=created_dirs,
        message=msg,
    )


# ============================================================
# Resolver spesifik (sesuai Settings.*)
# ============================================================
def resolve_product_pdf(
    settings,
    product: str,
    tahun: str,
    filename: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.product_base_path>/<product>/<tahun>/<filename>.pdf"""
    fname = _norm_pdf_filename(filename)
    return _resolve_under_base(
        settings.product_base_path,
        product,
        tahun,
        fname,
        project_root=project_root,
        create_dirs=create_dirs,
        unique=unique,
    )


def resolve_product_md(
    settings,
    product: str,
    tahun: str,
    filename: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.product_md_base_path>/<product>/<tahun>/<filename>.md"""
    fname = _norm_md_filename(filename)
    return _resolve_under_base(
        settings.product_md_base_path,
        product,
        tahun,
        fname,
        project_root=project_root,
        create_dirs=create_dirs,
        unique=unique,
    )


def resolve_product_summary_md(
    settings,
    product: str,
    tahun: str,
    filename: str,
    *,
    project_root: Optional[Path] = None,
    create_dirs: bool = True,
    unique: bool = False,
) -> PathInfo:
    """<base= settings.product_summaries_base_path>/<product>/<tahun>/<filename>_summary.md"""
    fname = _norm_summary_md_filename(filename)
    return _resolve_under_base(
        settings.product_summaries_base_path,
        product,
        tahun,
        fname,
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
# API siap pakai (ringkas)
# ============================================================
def save_product_pdf(
    settings, product: str, tahun: str, filename: str, data: bytes, *, unique=False
) -> PathInfo:
    info = resolve_product_pdf(
        settings, product, tahun, filename, create_dirs=True, unique=unique
    )
    write_bytes_atomic(info.full_path, data)
    return info


def open_product_pdf(settings, product: str, tahun: str, filename: str) -> bytes:
    info = resolve_product_pdf(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return read_bytes(info.full_path)


def remove_product_pdf(settings, product: str, tahun: str, filename: str) -> bool:
    info = resolve_product_pdf(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return delete_file(info.full_path)


def save_product_md(
    settings, product: str, tahun: str, filename: str, markdown: str, *, unique=False
) -> PathInfo:
    info = resolve_product_md(
        settings, product, tahun, filename, create_dirs=True, unique=unique
    )
    write_text_atomic(info.full_path, markdown)
    return info


def open_product_md(settings, product: str, tahun: str, filename: str) -> str:
    info = resolve_product_md(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return read_text(info.full_path)


def remove_product_md(settings, product: str, tahun: str, filename: str) -> bool:
    info = resolve_product_md(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return delete_file(info.full_path)


def save_product_summary(
    settings, product: str, tahun: str, filename: str, markdown: str, *, unique=False
) -> PathInfo:
    info = resolve_product_summary_md(
        settings, product, tahun, filename, create_dirs=True, unique=unique
    )
    write_text_atomic(info.full_path, markdown)
    return info


def open_product_summary(settings, product: str, tahun: str, filename: str) -> str:
    info = resolve_product_summary_md(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return read_text(info.full_path)


def remove_product_summary(settings, product: str, tahun: str, filename: str) -> bool:
    info = resolve_product_summary_md(
        settings, product, tahun, filename, create_dirs=False, unique=False
    )
    return delete_file(info.full_path)
