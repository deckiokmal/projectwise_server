from pathlib import Path
from typing import Any, Dict, Optional

from docling.document_converter import DocumentConverter

from mcp_server.settings import Settings
from mcp_server.utils.logger import logger
from mcp_server.utils.docx_generator import DocumentGenerator
from mcp_server.utils.rag_pipeline import RAGPipeline


class DocGeneratorTools:
    """
    Collection of MCP server tools for:
      * Retrieving product context via RAG
      * Extracting text from uploaded documents
      * Generating .docx proposal documents

    Intended to be registered with @mcp.tool() in server.py.
    """

    def __init__(self) -> None:
        """
        Initialize DocGeneratorTools with settings and utilities.

        Attributes:
            generator: DocumentGenerator instance for .docx templating
            rag_pipeline: RAGPipeline instance for retrieval
            output_dir: Directory to save generated proposals
            prompt_dir: Directory containing .txt prompt templates
        """
        cfg = Settings()  # type: ignore
        # Document generator using default proposal template
        self.generator = DocumentGenerator(template_path=cfg.proposal_template_path)
        # RAG pipeline for similarity search
        self.rag_pipeline = RAGPipeline()
        # Output directory for generated proposal documents
        self.output_dir: Path = Path(cfg.proposal_generate_path)
        # Directory for prompt templates (for LLM instructions)
        self.prompt_dir: Path = Path(__file__).resolve().parent.parent / "prompts"

        logger.info(
            f"DocGeneratorTools initialized: template={cfg.proposal_template_path}, "
            f"output_dir={self.output_dir}, prompt_dir={self.prompt_dir}"
        )

    def retrieve_product_context(
        self,
        product: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve and return product context for proposal via RAG similarity search.

        Steps:
          1. Perform retrieval of top-k chunks for the product query.
          2. Optionally load a prompt template to guide LLM client.

        Args:
            product: Product name or description to search for.
            k: Number of top chunks to return (defaults to pipeline's default).
            metadata_filter: Optional dict to filter retrieval by metadata.
            prompt_template: Filename (without .txt) of prompt template.

        Returns:
            Dict with:
              - status: 'success' or 'failure'
              - context: List of retrieved text chunks (on success)
              - instruction: Prompt template text (empty if not provided)
              - error: Error message (on failure)
        """
        try:
            logger.info(
                f"retrieve_product_context: product={product}, k={k}, filter={metadata_filter}"
            )
            chunks = self.rag_pipeline.retrieval(
                query=product,
                k=k,
                metadata_filter=metadata_filter,
            )
            instruction = ""
            if prompt_template:
                tpl_file = self.prompt_dir / f"{prompt_template}.txt"
                if tpl_file.is_file():
                    instruction = tpl_file.read_text(encoding="utf-8")
                    logger.debug(f"Loaded prompt template: {tpl_file}")
                else:
                    logger.warning(f"Prompt template not found: {tpl_file}")

            return {"status": "success", "context": chunks, "instruction": instruction}

        except Exception as e:
            logger.exception("retrieve_product_context failed")
            return {"status": "failure", "error": str(e)}

    def extract_document_text(self, file_path: str) -> Dict[str, Any]:
        """
        Convert an uploaded document to plain markdown text for LLM reasoning.

        Args:
            file_path: Local filesystem path to .pdf, .docx, or .md file.

        Returns:
            Dict with:
              - status: 'success' or 'failure'
              - text: Extracted markdown text (on success)
              - error: Error message (on failure)
        """
        path = Path(file_path)
        if not path.is_file():
            msg = f"File not found: {file_path}"
            logger.error(msg)
            return {"status": "failure", "error": msg}

        try:
            logger.info(f"Extracting text from document: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(source=str(path))
            markdown_text = result.document.export_to_markdown()
            logger.info(f"Extraction succeeded ({len(markdown_text)} chars)")
            return {"status": "success", "text": markdown_text}

        except Exception as e:
            logger.exception("extract_document_text failed")
            return {"status": "failure", "error": str(e)}

    def generate_proposal(
        self,
        context: Dict[str, Any],
        override_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render and save a proposal .docx using the provided context data.

        Args:
            context: Dict of template variables for docx rendering.
            override_template: Optional path to a different .docx template.

        Returns:
            Dict with:
              - status: 'success' or 'failure'
              - product: Identifier (e.g. 'judul_proposal')
              - path: Filesystem path of saved .docx (on success)
              - error: Error message (on failure)
        """
        try:
            logger.info("generate_proposal: starting document generation")
            if override_template:
                self.generator.load_template(override_template)

            output_path = self.generator.generate(
                context=context,
                output_dir=self.output_dir,
            )

            product_id = context.get("judul_proposal") or context.get(
                "nama_pelanggan", ""
            )
            logger.info(f"Document generated: {output_path}")

            return {
                "status": "success",
                "product": product_id,
                "path": str(output_path),
            }

        except Exception as e:
            logger.exception("generate_proposal failed")
            return {"status": "failure", "error": str(e)}
