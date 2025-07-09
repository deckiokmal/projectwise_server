from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from docxtpl import DocxTemplate
from mcp_server.settings import Settings
from mcp_server.utils.logger import logger


class DocumentGenerationError(Exception):
    """Raised when an error occurs during document generation pipeline."""

    pass


class TemplateNotFoundError(DocumentGenerationError):
    """Raised when the specified template file is not found."""

    pass


class ContextValidationError(DocumentGenerationError):
    """Raised when the provided context is invalid or incomplete."""

    pass


class DocumentGenerator:
    """Document utility class for generating Word (.docx) documents using docxtpl SDK.

    Features:
    - Load and override template
    - Validate and normalize context
    - Render and save document
    """

    DEFAULT_LIST_KEYS: List[str] = [
        "daftar_manfaat",
        "deliverables",
        "project_assumption",
    ]
    DEFAULT_OPTIONAL_KEYS: List[str] = [
        "executive_summary",
        "daftar_hardware",
        "daftar_software",
        "daftar_lisensi",
        "daftar_jasa",
        "scope_of_work",
        "out_of_scope",
        "response_time",
        "response_detail",
        "response_description",
        "resolution_time",
        "resolution_detail",
        "resolution_description",
    ]
    DEFAULT_FILENAME_KEYS: List[str] = [
        "nama_pelanggan",
        "judul_proposal",
    ]

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        list_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        filename_keys: Optional[List[str]] = None,
    ) -> None:
        """Initialize DocumentGenerator with optional custom keys and template path.

        Args:
            template_path: Path to .docx template; if None, from
                           Settings.proposal_template_path
            list_keys: Override keys for list normalization
            optional_keys: Override keys for optional string normalization
            filename_keys: Override keys for filename generation

        Raises:
        TemplateNotFoundError: When template file is not found
        DocumentGenerationError: When template loading fails
        """
        cfg = Settings()  # type: ignore
        path = (
            Path(template_path) if template_path else Path(cfg.proposal_template_path)
        )
        if not path.is_file():
            logger.error(f"Template not found: {path}")
            raise TemplateNotFoundError(f"Template .docx not found: {path}")
        try:
            self._doc = DocxTemplate(str(path))
            self.template_path = path
            self.list_keys = list_keys or list(self.DEFAULT_LIST_KEYS)
            self.optional_keys = optional_keys or list(self.DEFAULT_OPTIONAL_KEYS)
            self.filename_keys = filename_keys or list(self.DEFAULT_FILENAME_KEYS)
            logger.info(f"Initialized DocumentGenerator with template: {path}")
        except Exception as e:
            logger.exception(f"Error loading template: {path}")
            raise DocumentGenerationError(f"Error loading template: {e}")

    def load_template(self, template_path: Union[str, Path]) -> None:
        """Override the current template with a new .docx file.

        Args:
            template_path: Path to the new template file

        Raises:
            TemplateNotFoundError: When the new template is not found
            DocumentGenerationError: When loading the new template fails
        """
        path = Path(template_path)
        if not path.is_file():
            logger.error(f"Template not found: {path}")
            raise TemplateNotFoundError(f"Template .docx not found: {path}")
        try:
            self._doc = DocxTemplate(str(path))
            self.template_path = path
            logger.info(f"Template overridden: {path}")
        except Exception as e:
            logger.exception(f"Error overriding template: {path}")
            raise DocumentGenerationError(f"Error loading template: {e}")

    def validate_context(self, context: Dict[str, Any]) -> None:
        """Validate that context is a dict.

        Args:
            context: Data context for rendering

        Raises:
            ContextValidationError: If context is not a dict
        """
        if not isinstance(context, dict):
            logger.error(f"Invalid context type: {type(context)}")
            raise ContextValidationError("Context must be a dict.")
        logger.debug(f"Context validated: {list(context.keys())}")

    def normalize_context(self, context: Dict[str, Any]) -> None:
        """Ensure list and optional keys exist in context with default values.

        Args:
            context: Data context for rendering
        """
        for key in self.list_keys:
            context.setdefault(key, [])
        for key in self.optional_keys:
            context.setdefault(key, "")
        logger.debug("Context normalized")

    def generate_filename(self, context: Dict[str, Any]) -> str:
        """Generate a safe filename based on priority fields in context.

        Args:
            context: Data context after normalization

        Returns:
            A sanitized filename with .docx extension
        """
        base = next(
            (str(context[k]) for k in self.filename_keys if context.get(k)),
            "proposal_tanpa_nama",
        )
        safe = "".join(c for c in base if c.isalnum() or c in (" ", ".", "_")).rstrip()
        filename = f"{safe}.docx"
        logger.debug(f"Generated filename: {filename}")
        return filename

    def generate(
        self,
        context: Dict[str, Any],
        output_dir: Union[str, Path],
        override_template: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate document: override template, render with context, and save.

        Args:
            context: Dictionary data for rendering
            output_dir: Directory to save the generated document
            override_template: Optional path to override the template

        Returns:
            Path to the saved document

        Raises:
            DocumentGenerationError: If any step fails
        """
        logger.info("Starting document generation pipeline")
        if override_template:
            self.load_template(override_template)
        self.validate_context(context)
        self.normalize_context(context)
        try:
            self._doc.render(context)
            logger.info("Template rendered successfully")
        except Exception as e:
            logger.exception("Rendering failed")
            raise DocumentGenerationError(f"Rendering failed: {e}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.generate_filename(context)
        file_path = out_path / filename
        try:
            self._doc.save(str(file_path))
            logger.info(f"Document saved at: {file_path}")
            return file_path
        except Exception as e:
            logger.exception("Saving document failed")
            raise DocumentGenerationError(f"Saving document failed: {e}")


class TemplateManager:
    """CRUD service for managing .docx templates.

    Methods:
        list_templates: List available templates
        get_template: Retrieve a template path
        add_template: Copy a new template into the directory
        update_template: Overwrite an existing template
        delete_template: Remove a template file
    """

    def __init__(self, template_dir: Union[str, Path]) -> None:
        """Initialize with directory for templates."""
        self.template_dir = Path(template_dir)
        logger.info(f"TemplateManager initialized with dir: {self.template_dir}")

    def list_templates(self) -> List[str]:
        """Return list of available template filenames."""
        try:
            templates = [p.name for p in self.template_dir.glob("*.docx")]
            logger.info(f"Templates found: {templates}")
            return templates
        except Exception as e:
            logger.exception("Listing templates failed")
            raise DocumentGenerationError(f"Listing templates failed: {e}")

    def get_template(self, name: str) -> Path:
        """Retrieve a template path by filename."""
        path = self.template_dir / name
        if not path.is_file():
            logger.error(f"Template not found: {name}")
            raise TemplateNotFoundError(f"Template {name} not found.")
        logger.info(f"Template retrieved: {path}")
        return path

    def add_template(self, name: str, source_path: Union[str, Path]) -> Path:
        """Add a new template by copying from source_path."""
        src = Path(source_path)
        dest = self.template_dir / name
        if not src.is_file():
            logger.error(f"Source template not found: {src}")
            raise TemplateNotFoundError(f"Source template not found: {src}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as f_src, dest.open("wb") as f_dest:
                f_dest.write(f_src.read())
            logger.info(f"Template added: {dest}")
            return dest
        except Exception as e:
            logger.exception("Adding template failed")
            raise DocumentGenerationError(f"Adding template failed: {e}")

    def update_template(self, name: str, source_path: Union[str, Path]) -> Path:
        """Overwrite an existing template with a new file."""
        existing = self.get_template(name)
        src = Path(source_path)
        if not src.is_file():
            logger.error(f"Source template not found: {src}")
            raise TemplateNotFoundError(f"Source template not found: {src}")
        try:
            with src.open("rb") as f_src, existing.open("wb") as f_dest:
                f_dest.write(f_src.read())
            logger.info(f"Template updated: {existing}")
            return existing
        except Exception as e:
            logger.exception("Updating template failed")
            raise DocumentGenerationError(f"Updating template failed: {e}")

    def delete_template(self, name: str) -> None:
        """Delete a template by filename."""
        path = self.get_template(name)
        try:
            path.unlink()
            logger.info(f"Template deleted: {path}")
        except Exception as e:
            logger.exception("Deleting template failed")
            raise DocumentGenerationError(f"Deleting template failed: {e}")


class DocumentRepository:
    """CRUD service for managing generated documents.

    Methods:
        list_documents: List saved documents
        get_document: Retrieve a document path
        delete_document: Remove a generated document
    """

    def __init__(self, docs_dir: Union[str, Path]) -> None:
        """Initialize with directory for generated documents."""
        self.docs_dir = Path(docs_dir)
        logger.info(f"DocumentRepository initialized with dir: {self.docs_dir}")

    def list_documents(self) -> List[str]:
        """Return list of generated document filenames."""
        try:
            docs = [p.name for p in self.docs_dir.glob("*.docx")]
            logger.info(f"Documents found: {docs}")
            return docs
        except Exception as e:
            logger.exception("Listing documents failed")
            raise DocumentGenerationError(f"Listing documents failed: {e}")

    def get_document(self, name: str) -> Path:
        """Retrieve a document path by filename."""
        path = self.docs_dir / name
        if not path.is_file():
            logger.error(f"Document not found: {name}")
            raise FileNotFoundError(f"Document {name} not found.")
        logger.info(f"Document retrieved: {path}")
        return path

    def delete_document(self, name: str) -> None:
        """Delete a generated document by filename."""
        path = self.get_document(name)
        try:
            path.unlink()
            logger.info(f"Document deleted: {path}")
        except Exception as e:
            logger.exception("Deleting document failed")
            raise DocumentGenerationError(f"Deleting document failed: {e}")
