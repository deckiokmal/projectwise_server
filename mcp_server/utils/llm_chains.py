# utils/llm_chains.py
from openai import OpenAI
from typing import Optional, Dict, Any, List, Union
from mcp_server.settings import Settings
from mcp_server.utils.logger import logger
import traceback

settings = Settings()  # type: ignore


class LLMChain:
    """
    Kelas utilitas untuk menjalankan single-shot LLM call menggunakan OpenAI Responses API.
    """

    def __init__(
        self,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
        api_key: Optional[str] = settings.openai_api_key,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    async def generate_text(
        self,
        input: Union[str, List[Dict[str, str]]],
        instructions: Optional[str] = None,
    ) -> str:
        """
        Kirim permintaan LLM dengan mode sederhana atau message-role.

        Args:
            input: string prompt biasa, atau list of message dicts (role: user/developer).
            instructions: sistem-level instructions (opsional, untuk prompt injection).

        Returns:
            Output string hasil dari model.
        """
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "input": input,
                "temperature": self.temperature,
            }
            if instructions:
                payload["instructions"] = instructions

            logger.debug("Calling OpenAI Responses API: Instruction Payload.")
            response = self.client.responses.create(**payload)
            return response.output_text

        except Exception as e:
            logger.error(f"LLMChain generate_text error: {e}")
            traceback.print_exc()
            return "[Gagal memanggil LLMChain]"

    async def generate_with_prompt_template(
        self,
        prompt_id: str,
        variables: Dict[str, Union[str, Dict]],
        version: Optional[str] = None,
    ) -> str:
        """
        Menggunakan reusable prompt template dari OpenAI dashboard.

        Args:
            prompt_id: ID template yang sudah didaftarkan di dashboard.
            variables: mapping key-value untuk di-substitute dalam template.
            version: versi khusus jika ingin spesifik (opsional).

        Returns:
            String hasil output dari model.
        """
        try:
            prompt_obj = {
                "id": prompt_id,
                "variables": variables,
            }
            if version:
                prompt_obj["version"] = version

            logger.debug("Calling with prompt template: KAK_Analyzer.")
            response = self.client.responses.create(
                model=self.model,
                prompt=prompt_obj,  # type: ignore
                temperature=self.temperature,
            )  # type: ignore
            return response.output_text

        except Exception as e:
            logger.error(f"LLMChain generate_with_prompt_template error: {e}")
            traceback.print_exc()
            return "[Gagal memanggil prompt template]"
