# main.py
from __future__ import annotations

import asyncio
import contextlib
from fastapi import FastAPI
import multiprocessing as mp
from mcp_server.utils.logger import get_logger
from mcp_server.settings import Settings

from mcp_server.utils.multiprocessing_utils import create_cpu_pool, shutdown_cpu_pool
from mcp_server.server import mcp as projectwise_mcp
from mcp_server.api.kak_analyzer_upload import router as kak_router
from mcp_server.api.product_sizing_upload import router as product_router
from mcp_server.api.check_status_ingestion import router as status_router
from mcp_server.api.proposal_download import router as download_router
import uvicorn

logger = get_logger(__name__)
settings = Settings()


# Lifespan untuk manage MCP + ProcessPool
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Pastikan "spawn" untuk Windows; abaikan jika sudah diset.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    async with contextlib.AsyncExitStack() as stack:
        # 1) Start ProcessPoolExecutor (multiprocessing) sekali
        pool = create_cpu_pool(
            settings.max_cpu_workers
        )  # atau create_cpu_pool(settings.CPU_POOL_WORKERS)
        app.state.cpu_pool = pool
        logger.info("CPU ProcessPoolExecutor initialized.")

        # 2) Pastikan pool dimatikan saat shutdown
        GRACEFUL_WAIT = getattr(settings, "cpu_pool_graceful_wait", True)
        CANCEL_FUTURES = getattr(settings, "cpu_pool_cancel_futures", False)

        stack.callback(
            lambda: shutdown_cpu_pool(wait=GRACEFUL_WAIT, cancel_futures=CANCEL_FUTURES)
        )

        # 3) Jalankan MCP session manager
        await stack.enter_async_context(projectwise_mcp.session_manager.run())

        # 4) Yield ke FastAPI (app berjalan)
        yield

        # (teardown otomatis oleh AsyncExitStack → shutdown_cpu_pool dipanggil)
        logger.info("CPU ProcessPoolExecutor shutdown complete.")


# Gunakan lifespan untuk mengelola session managers + pool
app = FastAPI(lifespan=lifespan)

# Endpoint API untuk Ingest pdf
app.include_router(kak_router, prefix="/api")
app.include_router(product_router, prefix="/api")
app.include_router(status_router, prefix="/api")
app.include_router(download_router, prefix="/api")

# Inisialisasi MCP Server
app.mount("/projectwise", projectwise_mcp.streamable_http_app())


async def main():
    """Main async entrypoint untuk menjalankan Uvicorn."""
    config = uvicorn.Config(app, host="0.0.0.0", port=5000, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Serve dibatalkan saat shutdown yang diminta.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        logger.info("Shutdown: tugas async dibatalkan (normal saat SIGINT/SIGTERM).")
    except KeyboardInterrupt:
        logger.info("Shutdown diminta oleh user (Ctrl+C).")
    except Exception as e:
        logger.exception("Kesalahan tak terduga saat shutdown: %s", e)
    finally:
        logger.info("Server benar-benar berhenti dengan mulus.")
    """
    Cara menjalankan:
    uv run main.py -> untuk menjalankan FastMCP server
    npx @modelcontextprotocol/inspector -> debugging MCP server

    MCP URI untuk client: http://localhost:5000/projectwise/mcp/
    """
