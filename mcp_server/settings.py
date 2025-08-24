# mcp_server/settings.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# .env absolute path
env_path = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # ====================================
    # Model dan parameter LLM
    # ====================================
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11443")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    max_token: int = 28000  # 30.000 token untuk ringkasan - qwen25-72b

    # RAG Pipeline config Chunk dengan token-aware
    embed_llm_api_key: str = os.getenv("EMBEDDING_LLM_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    tokenizer_kind: str = "openai"  # hf | openai
    tokenizer_max_token: int = 128 * 1024  # OpenAI context windows

    # ====================================
    # Direktori penyimpanan dan dokumen
    # ====================================
    # 1) Document Knowledge Base (LanceDB)
    kak_tor_base_path: str = "data/kak_tor_pdf"
    kak_tor_md_base_path: str = "data/kak_tor_md"
    kak_tor_summaries_base_path: str = "data/kak_tor_summaries"

    product_base_path: str = "data/product_standard_pdf"
    product_md_base_path: str = "data/product_standard_md"
    product_summaries_base_path: str = "data/product_standard_summaries"

    # 2) Document Prompting
    prompts_base_path: str = "mcp_server/prompt"

    # 3) Document Proposal Proyek
    proposal_template_base_path: str = "mcp_server/data/document_templates"
    proposal_generated_base_path: str = "data/proposal_generated"
    public_download_base_url: str = "http://localhost:5000/api"

    # 4) JSON File
    json_ingestion_status_file: str = "mcp_server/data/ingestion_status.json"
    json_ingestion_manifest_file: str = "mcp_server/data/ingested_manifest.json"

    # ====================================
    # Database Vector LanceDB
    # ====================================
    # Pengaturan chunking & retrieval
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    cloud_api_key: str = os.getenv("LANCEDB_CLOUD_API_KEY", "")

    # DB Local Config
    db_connection: str = "local"  # local, s3, cloud
    chunk_size: int = 200
    chunk_overlap: int = 20
    # retriever_search_k: int = 10

    # Vector store & backend
    vector_backend: str = os.getenv("VECTOR_BACKEND", "lancedb")  # lancedb|qdrant
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vectorstore")
    collection_name: str = os.getenv("VECTOR_COLLECTION", "projectwise_mcp")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "Cosine")  # Cosine|Dot|Euclid

    # Retriever
    retriever_search_k: int = int(os.getenv("RETRIEVER_K", "8"))

    # Concurrency & performance
    max_concurrent_proccess: int = int(os.getenv("MAX_CONCURRENT_PROCCESS", "8"))
    max_cpu_workers: int = max(1, int(os.getenv("MAX_CPU_WORKERS", "4")))

    # Vector dim (akan dihitung dari model jika 0)
    vector_dim: int = int(os.getenv("VECTOR_DIM", "1536"))

    # ====================================
    # Base config ENV
    # ====================================
    model_config = SettingsConfigDict(env_file=str(env_path), env_file_encoding="utf-8")
