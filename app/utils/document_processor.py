import hashlib
from typing import Tuple, Dict, Any, BinaryIO
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument


class DocumentProcessor:
    """Utility class for document text extraction and processing.

    Design decisions:
    - Support for common document formats (PDF, DOCX, TXT)
    - Consistent error handling across format types
    - File hash generation for deduplication
    - Metadata extraction where possible
    - Memory-efficient processing for large files
    """

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}
    MAX_TEXT_LENGTH = 1_000_000  # 1MB of text

    @classmethod
    def get_file_hash(cls, content: bytes) -> str:
        """Generate SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def extract_text_and_metadata(
        cls, content: bytes, filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from document content.

        Returns:
            Tuple of (extracted_text, metadata_dict)

        Raises:
            ValueError: If file format is not supported
            RuntimeError: If text extraction fails
        """
        file_extension = Path(filename).suffix.lower()

        if file_extension not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_extension}")

        try:
            if file_extension == ".pdf":
                return cls._extract_from_pdf(content)
            elif file_extension == ".docx":
                return cls._extract_from_docx(content)
            elif file_extension == ".txt":
                return cls._extract_from_txt(content)
            else:
                raise ValueError(f"No extractor for format: {file_extension}")

        except Exception as e:
            raise RuntimeError(
                f"Text extraction failed for {filename}: {str(e)}"
            ) from e

    @classmethod
    def _extract_from_pdf(cls, content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF content."""
        from io import BytesIO

        pdf_buffer = BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_buffer)

        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
            except Exception as e:
                # Log warning but continue processing other pages
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to extract text from PDF page {page_num}: {e}"
                )

        full_text = "\n\n".join(text_parts)

        # Extract metadata
        metadata = {
            "page_count": len(reader.pages),
            "extraction_method": "PyPDF2",
            "extracted_pages": len(text_parts),
        }

        # Add PDF metadata if available
        if reader.metadata:
            pdf_meta = reader.metadata
            metadata.update(
                {
                    "title": pdf_meta.get("/Title", ""),
                    "author": pdf_meta.get("/Author", ""),
                    "subject": pdf_meta.get("/Subject", ""),
                    "creator": pdf_meta.get("/Creator", ""),
                    "creation_date": str(pdf_meta.get("/CreationDate", "")),
                }
            )

        # Truncate if too long
        if len(full_text) > cls.MAX_TEXT_LENGTH:
            full_text = full_text[: cls.MAX_TEXT_LENGTH]
            metadata["truncated"] = True
            metadata["original_length"] = len(full_text)

        return full_text, metadata

    @classmethod
    def _extract_from_docx(cls, content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from DOCX content."""
        from io import BytesIO

        docx_buffer = BytesIO(content)
        doc = DocxDocument(docx_buffer)

        # Extract text from paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():  # Only add non-empty paragraphs
                paragraphs.append(para.text)

        full_text = "\n\n".join(paragraphs)

        # Extract metadata
        metadata = {
            "paragraph_count": len(paragraphs),
            "extraction_method": "python-docx",
        }

        # Add document properties if available
        if hasattr(doc, "core_properties"):
            props = doc.core_properties
            metadata.update(
                {
                    "title": props.title or "",
                    "author": props.author or "",
                    "subject": props.subject or "",
                    "created": str(props.created) if props.created else "",
                    "modified": str(props.modified) if props.modified else "",
                }
            )

        # Truncate if too long
        if len(full_text) > cls.MAX_TEXT_LENGTH:
            full_text = full_text[: cls.MAX_TEXT_LENGTH]
            metadata["truncated"] = True
            metadata["original_length"] = len(full_text)

        return full_text, metadata

    @classmethod
    def _extract_from_txt(cls, content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text from plain text content."""
        # Try different encodings
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                text = content.decode(encoding)

                metadata = {
                    "encoding": encoding,
                    "extraction_method": "direct_decode",
                    "line_count": text.count("\n") + 1,
                }

                # Truncate if too long
                if len(text) > cls.MAX_TEXT_LENGTH:
                    text = text[: cls.MAX_TEXT_LENGTH]
                    metadata["truncated"] = True
                    metadata["original_length"] = len(text)

                return text, metadata

            except UnicodeDecodeError:
                continue

        raise RuntimeError("Could not decode text file with any supported encoding")

    @classmethod
    def validate_file_size(cls, content: bytes, max_size: int) -> bool:
        """Validate file size is within limits."""
        return len(content) <= max_size

    @classmethod
    def get_file_info(cls, content: bytes, filename: str) -> Dict[str, Any]:
        """Get basic file information without full text extraction."""
        file_extension = Path(filename).suffix.lower()

        info = {
            "filename": filename,
            "file_type": file_extension.lstrip("."),
            "file_size": len(content),
            "file_hash": cls.get_file_hash(content),
            "is_supported": file_extension in cls.SUPPORTED_FORMATS,
        }

        return info
