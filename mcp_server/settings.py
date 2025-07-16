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

    # Kunci API dan host model
    openai_api_key: str = os.getenv("OPENAI_API_KEY") # type: ignore
    ollama_host: str = "http://localhost:11434"

    # Model dan parameter LLM
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0

    # Direktori penyimpanan dan dokumen
    vector_store_path: str = "lancedb_storage"
    prompt_directory: str = "mcp_server/prompts"
    kak_tor_base_path: str = "mcp_server/data/kak_tor"
    kak_tor_md_base_path: str = "mcp_server/data/kak_tor_md"
    summaries_md_base_path: str = "mcp_server/data/summaries"
    knowledge_base_path: str = "mcp_server/data/product_standard"
    templates_base_path: str = "mcp_server/data/templates/prompts"
    summary_output_directory: str = "mcp_server/data/summaries"
    proposal_template_path: str = "mcp_server/data/templates/proposals/proposal_template.docx"
    proposal_generate_path: str = "mcp_server/data/proposal_generated"
    knowledge_source_extensions: List[str] = [".pdf", ".md"]

    # Pengaturan chunking & retrieval
    collection_name: str = "projectwise_knowledge"
    chunk_size: int = 200
    chunk_overlap: int = 20
    retriever_search_k: int = 10
