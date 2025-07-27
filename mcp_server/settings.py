from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# .env absolute path
env_path = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Konfigurasi Model
    model_config = SettingsConfigDict(env_file=str(env_path), env_file_encoding="utf-8")

    # Model dan parameter LLM
    ollama_host: str = "http://localhost:11434"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # type: ignore
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0

    # Direktori penyimpanan dan dokumen
    kak_tor_base_path: str = "mcp_server/data/kak_tor"
    kak_tor_md_base_path: str = "mcp_server/data/kak_tor_md"
    kak_tor_summaries_base_path: str = "mcp_server/data/kak_tor_summaries"
    product_base_path: str = "mcp_server/data/product_standard"
    templates_base_path: str = "mcp_server/data/templates/prompts"
    summaries_instruction_path: str = (
        "mcp_server/data/templates/prompts/kak_analyzer.txt"
    )
    proposal_template_path: str = (
        "mcp_server/data/templates/proposals/proposal_template.docx"
    )
    proposal_generate_path: str = "mcp_server/data/proposal_generated"
    knowledge_source_extensions: List[str] = [".pdf", ".md"]

    status_file_path: str = "mcp_server/data/ingestion_status.json"
    manifest_file_path: str = "mcp_server/data/ingested_manifest.json"

    # Pengaturan chunking & retrieval (LanceDB))
    db_connection: str = "local"  # local, s3, cloud
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    cloud_api_key: str = os.getenv("LANCEDB_CLOUD_API_KEY", "")
    vector_store_path: str = (
        "lancedb_storage"  # "s3://my-bucket/lancedb" "db://my_database"
    )
    collection_name: str = "projectwise_knowledge"
    vector_dim: int = 1536
    chunk_size: int = 200
    chunk_overlap: int = 20
    retriever_search_k: int = 10
