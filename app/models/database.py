from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    JSON,
    LargeBinary,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Document(Base):
    """Document model for storing uploaded documents and metadata.

    Design decisions:
    - UUID primary key for better distribution and security
    - Separate content storage for large files (could move to S3 later)
    - Metadata stored as JSON for flexibility
    - File hash for deduplication and integrity checks
    """

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256
    content = Column(LargeBinary)  # Raw file content
    extracted_text = Column(Text)  # Extracted text from document
    doc_metadata = Column(JSON, default=dict)  # Flexible metadata storage

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', type='{self.file_type}')>"


class DocumentChunk(Base):
    """Document chunks for processing large documents.

    Design decisions:
    - Separate table for chunks to optimize queries and storage
    - Chunk index for maintaining order
    - Token count for LLM processing optimization
    """

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    chunk_metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return (
            f"<DocumentChunk(document_id={self.document_id}, index={self.chunk_index})>"
        )


class Summary(Base):
    """Summary model for storing generated summaries.

    Design decisions:
    - Separate confidence score for AI model reliability tracking
    - Processing time for performance monitoring
    - Version tracking for summary improvements
    - Structured metadata for extensibility
    """

    __tablename__ = "summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    summary_text = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    confidence_score = Column(Float)  # 0.0 to 1.0
    processing_time_seconds = Column(Float)
    model_version = Column(String(100))
    summary_metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return (
            f"<Summary(document_id={self.document_id}, word_count={self.word_count})>"
        )
