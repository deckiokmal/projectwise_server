import contextlib
from fastapi import FastAPI
from mcp_server.server import mcp as projectwise_mcp
from mcp_server.api.app_kak_pipeline import router as kak_router
from mcp_server.api.app_product_pipeline import router as product_router
from mcp_server.api.check_status_ingestion import router as status_router


# Buat kombinasi lifespan untuk manage session managers mcp
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(projectwise_mcp.session_manager.run())
        yield


# Gunakan lifespan untuk mengelola session managers
app = FastAPI(lifespan=lifespan)

# Endpoint API untuk Ingest pdf
app.include_router(kak_router, prefix="/api")
app.include_router(product_router, prefix="/api")
app.include_router(status_router, prefix="/api")

# Inisialisasi MCP Server
app.mount("/projectwise", projectwise_mcp.streamable_http_app())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)

    # run the FastMCP server
    """
    uv run main.py -> to run the FastMCP server
    npx @modelcontextprotocol/inspector -> for debugging MCP server
    
    mcp URI for client : http://localhost:5000/projectwise/mcp
    """
