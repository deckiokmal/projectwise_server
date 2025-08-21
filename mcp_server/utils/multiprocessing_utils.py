# mcp_server/utils/multiprocessing_utils.py
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

_EXECUTOR: Optional[ProcessPoolExecutor] = None


def create_cpu_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """Buat ProcessPoolExecutor (singleton). Panggil di lifespan startup."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR
    if max_workers is None:
        cpu = os.cpu_count() or 4
        # sisakan 1 core utk event loop/IO/DB/embedding
        max_workers = max(1, min(cpu - 1, 4))
    _EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
    return _EXECUTOR


def get_cpu_pool() -> ProcessPoolExecutor:
    """Ambil instance pool; auto-create jika belum ada (untuk test/backdoor)."""
    global _EXECUTOR
    if _EXECUTOR is None:
        return create_cpu_pool()
    return _EXECUTOR


def shutdown_cpu_pool(*, wait: bool = False, cancel_futures: bool = True) -> None:
    """Matikan pool pada app shutdown."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=wait, cancel_futures=cancel_futures)
        _EXECUTOR = None
