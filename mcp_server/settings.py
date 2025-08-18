from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# .env absolute path
env_path = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # ====================================
    # Konfigurasi Model
    # ====================================
    model_config = SettingsConfigDict(env_file=str(env_path), env_file_encoding="utf-8")

    # ====================================
    # Model dan parameter LLM
    # ====================================
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # type: ignore
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_token: int = 16000  # 100.000 token untuk ringkasan

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
    vector_store_path: str = (
        "lancedb_storage"  # "s3://my-bucket/lancedb" "db://my_database"
    )
    collection_name: str = "projectwise_knowledge"
    vector_dim: int = 1536
    chunk_size: int = 200
    chunk_overlap: int = 20
    retriever_search_k: int = 10
