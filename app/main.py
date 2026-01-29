from datetime import datetime
import logging
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import get_settings
from .schemas.document import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    SummarizeRequest,
    SummaryResponse,
    ErrorResponse,
)
from .services.database import DatabaseService

from .services.vector import VectorService
from .services.orchestrator import DocumentSummarizerOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready document summarization API with multi-agent architecture",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global services (will be properly initialized in lifespan)
db_service = DatabaseService()
vector_service = VectorService()
orchestrator = DocumentSummarizerOrchestrator(db_service, vector_service)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Starting Document Summarizer API with Vector Database...")
        await db_service.connect()
        await vector_service.initialize_collection()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown."""
    try:
        logger.info("Shutting down Document Summarizer API...")
        await db_service.disconnect()
        await vector_service.close()
        logger.info("All services shut down successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error", detail=str(exc), error_code="VALIDATION_ERROR"
        ).dict(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Bad Request", detail=str(exc), error_code="BAD_REQUEST"
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
        ).dict(),
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    try:
        stats = await orchestrator.get_orchestrator_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.version,
            "stats": stats,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# API Endpoints
@app.post(
    "/documents/ingest",
    response_model=DocumentIngestResponse,
    status_code=201,
    tags=["Documents"],
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    extract_immediately: bool = True,
    doc_metadata: Optional[str] = None,  # JSON string for additional metadata
):
    """Ingest a document for processing.
    
    **Supported formats:** PDF, DOCX, TXT
    
    **Process:**
    1. Validates file format and size
    2. Checks for duplicates (by content hash)
    3. Optionally extracts text immediately
    4. Stores document and metadata in database
    
    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/documents/ingest" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@document.pdf" \
         -F "extract_immediately=true"
    ```
    """
    try:
        # Validate file size
        if file.size and file.size > settings.max_document_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_document_size} bytes",
            )

        # Read file content
        content = await file.read()

        # Parse metadata if provided
        parsed_doc_metadata = None
        if doc_metadata:
            import json

            try:
                parsed_doc_metadata = json.loads(doc_metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid JSON in metadata parameter"
                )

        # Process document
        result = await orchestrator.ingest_document(
            file_content=content,
            filename=file.filename,
            extract_immediately=extract_immediately,
            metadata=parsed_doc_metadata,
        )

        return DocumentIngestResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document")


@app.post("/documents/summarize", response_model=SummaryResponse, tags=["Documents"])
async def summarize_document(
    request: SummarizeRequest, background_tasks: BackgroundTasks
):
    """Generate a summary for an ingested document.

    **Multi-Agent Process:**
    1. **ChunkingAgent**: Splits document into optimal chunks
    2. **SummarizationAgent**: Generates summary using LLM
    3. **ValidatorAgent**: Validates summary quality

    **Example Request:**
    ```json
    {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "force_regenerate": false,
      "summary_length": "medium",
      "focus_areas": ["key findings", "recommendations"]
    }
    ```

    **Example Response:**
    ```json
    {
      "id": "123e4567-e89b-12d3-a456-426614174001",
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "summary_text": "This document presents...",
      "word_count": 247,
      "confidence_score": 0.89,
      "processing_time_seconds": 12.34,
      "model_version": "gpt-4-turbo-preview",
      "summary_metadata": {
        "length_type": "medium",
        "focus_areas": ["key findings"],
        "validation_score": 0.92
      }
    }
    ```
    """
    try:
        result = await orchestrator.summarize_document(
            document_id=request.document_id,
            force_regenerate=request.force_regenerate,
            summary_length=request.summary_length,
            focus_areas=request.focus_areas,
        )

        # Convert to response model (filter out internal fields)
        response_data = {
            "id": UUID(
                str(result.get("id", result["document_id"]))
            ),  # Fallback to doc_id if no summary id
            "document_id": result["document_id"],
            "summary_text": result["summary_text"],
            "word_count": result["word_count"],
            "confidence_score": result.get("confidence_score"),
            "processing_time_seconds": result.get("processing_time_seconds"),
            "model_version": result.get("model_version"),
            "summary_metadata": result.get("summary_metadata", {}),
            "created_at": result.get("created_at", datetime.utcnow()),
        }

        return SummaryResponse(**response_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate summary")


@app.get(
    "/documents/{document_id}/summary",
    response_model=SummaryResponse,
    tags=["Documents"],
)
async def get_document_summary(document_id: UUID):
    """Retrieve an existing summary for a document.

    **Example Request:**
    ```bash
    curl "http://localhost:8000/documents/123e4567-e89b-12d3-a456-426614174000/summary"
    ```

    **Database Retrieval:**
    - Retrieves summary from database storage
    """
    try:
        result = await orchestrator.get_summary(document_id)

        if not result:
            raise HTTPException(
                status_code=404, detail="Summary not found for this document"
            )

        return SummaryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve summary")


@app.get("/documents/{document_id}", tags=["Documents"])
async def get_document(document_id: UUID):
    """Retrieve a specific document by ID.

    **Parameters:**
    - `document_id`: UUID of the document

    **Example Request:**
    ```bash
    curl "http://localhost:8000/documents/{document_id}"
    ```

    **Response:**
    ```json
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "filename": "document.pdf",
      "file_type": "pdf",
      "file_size": 1024,
      "created_at": "2026-01-28T10:00:00Z",
      "metadata": {...}
    }
    ```
    """
    try:
        document = await db_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Return document info without the binary content
        return {
            "id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "file_hash": document.file_hash,
            "created_at": document.created_at.isoformat(),
            "updated_at": (
                document.updated_at.isoformat() if document.updated_at else None
            ),
            "doc_metadata": document.doc_metadata,
            "extracted_text_preview": (
                document.extracted_text[:500] + "..."
                if document.extracted_text and len(document.extracted_text) > 500
                else document.extracted_text
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@app.get("/documents", tags=["Documents"])
async def list_documents(page: int = 1, page_size: int = 20):
    """List all documents with pagination.

    **Parameters:**
    - `page`: Page number (default: 1)
    - `page_size`: Number of documents per page (default: 20, max: 100)

    **Example Request:**
    ```bash
    curl "http://localhost:8000/documents?page=1&page_size=10"
    ```

    **Response:**
    ```json
    {
      "documents": [...],
      "total": 150,
      "page": 1,
      "page_size": 10,
      "total_pages": 15
    }
    ```
    """
    try:
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=400, detail="Page size must be between 1 and 100"
            )

        # Get total count
        stats = await db_service.get_document_stats()
        total_documents = stats.get("total_documents", 0)
        total_pages = (total_documents + page_size - 1) // page_size

        # For now, return empty list since we need to implement pagination in database service
        documents = []

        return {
            "documents": documents,
            "total": total_documents,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents list")


# Additional utility endpoints
@app.post("/search/semantic", tags=["Search"])
async def semantic_search(request_body: dict):
    """Perform semantic search across documents.

    **Request Body:**
    ```json
    {
      "query": "machine learning algorithms",
      "limit": 5,
      "min_similarity": 0.7
    }
    ```
    """
    try:
        query = request_body.get("query")
        limit = request_body.get("limit", 10)
        min_similarity = request_body.get("min_similarity", 0.5)

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        if limit > 50:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 50")
        if min_similarity < 0 or min_similarity > 1:
            raise HTTPException(
                status_code=400, detail="min_similarity must be between 0 and 1"
            )

        # For now, return empty results since we need to implement search
        # This would use the vector service to search similar documents
        return {
            "results": [],
            "query": query,
            "limit": limit,
            "min_similarity": min_similarity,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/stats", tags=["System"])
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        return await orchestrator.get_orchestrator_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: UUID):
    """Delete a document and all associated data.

    **WARNING:** This action is irreversible and will delete:
    - Original document content
    - Extracted text
    - Generated chunks
    - Summary data
    - Vector embeddings (if memory is enabled)
    """
    try:
        # Delete from database
        deleted = await db_service.delete_document(document_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from vector database if memory is enabled
        if settings.enable_memory:
            await vector_service.delete_document_embeddings(document_id)

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "vector_embeddings_deleted": settings.enable_memory,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
