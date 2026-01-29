from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class DocumentBase(BaseModel):
    """Base document schema with common fields."""

    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, docx, txt)")
    doc_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    """Schema for document creation requests."""

    pass


class DocumentResponse(DocumentBase):
    """Schema for document API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    file_size: int
    file_hash: str
    created_at: datetime
    updated_at: Optional[datetime] = None


class DocumentIngestRequest(BaseModel):
    """Request schema for document ingestion endpoint."""

    extract_immediately: bool = Field(
        default=True,
        description="Whether to extract text immediately or queue for async processing",
    )
    doc_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata to store with document"
    )


class DocumentIngestResponse(BaseModel):
    """Response schema for document ingestion."""

    document_id: UUID
    message: str
    processing_status: str = Field(..., description="immediate, queued, or failed")


class SummarizeRequest(BaseModel):
    """Request schema for document summarization."""

    document_id: UUID
    force_regenerate: bool = Field(
        default=False, description="Force regeneration even if cached summary exists"
    )
    summary_length: Optional[str] = Field(
        default="medium", description="Summary length: short, medium, long"
    )
    focus_areas: Optional[list[str]] = Field(
        default=None, description="Specific areas to focus on in summary"
    )


class SummaryResponse(BaseModel):
    """Response schema for document summaries."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    document_id: UUID
    summary_text: str
    word_count: int
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="AI confidence score between 0 and 1"
    )
    processing_time_seconds: Optional[float]
    model_version: Optional[str]
    summary_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
