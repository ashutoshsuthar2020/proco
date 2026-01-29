import asyncio
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
import logging
from datetime import datetime

from ..agents.chunking import ChunkingAgent
from ..agents.summarization import SummarizationAgent
from ..agents.validator import ValidatorAgent
from ..agents.memory import MemoryAgent
from ..services.database import DatabaseService
from ..services.vector import VectorService
from ..utils.document_processor import DocumentProcessor
from ..config import get_settings


class DocumentSummarizerOrchestrator:
    """Orchestrator for coordinating the multi-agent document summarization workflow with long-term memory.

    Design decisions:
    - Centralized workflow coordination for consistency
    - Async processing with proper error handling
    - Integrated vector database for long-term memory
    - Memory-enhanced agent pipeline
    - Configurable agent pipeline
    - Comprehensive logging for observability
    - Transaction-like behavior with rollback capabilities
    """

    def __init__(
        self,
        db_service: DatabaseService,
        vector_service: VectorService,
    ):
        self.db_service = db_service
        self.vector_service = vector_service
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize agents
        self.chunking_agent = ChunkingAgent()
        self.summarization_agent = SummarizationAgent()
        self.validator_agent = ValidatorAgent()

        # Initialize memory agent if enabled
        self.memory_agent = (
            MemoryAgent(self.vector_service) if self.settings.enable_memory else None
        )

        self.doc_processor = DocumentProcessor()

    async def ingest_document(
        self,
        file_content: bytes,
        filename: str,
        extract_immediately: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a document with optional immediate text extraction."""

        start_time = datetime.utcnow()
        document_id = uuid4()

        try:
            # Check for existing document by content hash
            file_hash = await self.db_service.calculate_content_hash(file_content)
            existing_doc = await self.db_service.get_document_by_hash(file_hash)

            if existing_doc:
                self.logger.info(f"Document already exists with hash {file_hash}")
                return {
                    "document_id": existing_doc.id,
                    "processing_status": "immediate",
                    "message": "Document already exists in database",
                }

            # Extract text immediately if requested
            extracted_text = None
            extraction_metadata = None

            if extract_immediately:
                try:
                    extracted_text, extraction_metadata = (
                        DocumentProcessor.extract_text_and_metadata(
                            file_content, filename
                        )
                    )
                    self.logger.info(
                        f"Extracted {len(extracted_text)} characters from {filename}"
                    )
                except Exception as e:
                    self.logger.warning(f"Text extraction failed for {filename}: {e}")

            # Store document in database
            document_id = await self.db_service.store_document(
                document_id=document_id,
                filename=filename,
                content=file_content,
                extracted_text=extracted_text,
                file_hash=file_hash,
                metadata={
                    **(metadata or {}),
                    "extraction_metadata": extraction_metadata,
                    "ingestion_timestamp": start_time.isoformat(),
                },
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "document_id": document_id,
                "processing_status": "immediate",
                "message": f"Document ingested successfully"
                + (
                    " with text extraction"
                    if extracted_text
                    else " (text extraction pending)"
                ),
            }

        except Exception as e:
            self.logger.error(f"Document ingestion failed: {e}")
            raise

    async def summarize_document(
        self,
        document_id: UUID,
        summary_length: str = "medium",
        focus_areas: Optional[List[str]] = None,
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Generate summary for a document using the multi-agent pipeline with memory enhancement.

        Returns:
            Summary dictionary with text, metadata, validation results, and memory context
        """
        start_time = datetime.utcnow()
        focus_areas = focus_areas or []

        try:
            # Get document from database
            document = await self.db_service.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")

            # Check if summary already exists
            if not force_regenerate:
                existing_summary = await self.db_service.get_summary(document_id)
                if existing_summary:
                    self.logger.info(
                        f"Summary already exists for document {document_id}"
                    )
                    # Return existing summary in expected format
                    return {
                        "id": existing_summary.id,
                        "document_id": existing_summary.document_id,
                        "summary_text": existing_summary.summary_text,
                        "word_count": existing_summary.word_count,
                        "confidence_score": existing_summary.confidence_score,
                        "processing_time_seconds": existing_summary.processing_time_seconds,
                        "model_version": existing_summary.model_version,
                        "summary_metadata": existing_summary.summary_metadata,
                        "created_at": existing_summary.created_at,
                    }

            # Ensure we have extracted text
            if not document.extracted_text:
                # Try to extract text now
                try:
                    extracted_text, extraction_metadata = (
                        DocumentProcessor.extract_text_and_metadata(
                            document.content, document.filename
                        )
                    )
                    # Update document with extracted text
                    document.extracted_text = extracted_text
                    await self.db_service.update_document_text(
                        document_id, extracted_text
                    )
                except Exception as e:
                    raise ValueError(f"Cannot extract text from document: {e}")

            # Step 1: Chunk the document
            chunk_input = {"text": document.extracted_text, "document_id": document_id}
            chunks = await self.chunking_agent.process(chunk_input)

            # Store chunks in database
            chunk_objects = await self.db_service.create_chunks(chunks)

            # Step 2: Get memory context if memory is enabled
            memory_context = None
            similar_docs = []

            if self.memory_agent:
                try:
                    memory_input = {
                        "text": document.extracted_text,
                        "document_id": document_id,
                        "operation": "retrieve_memory",
                    }
                    memory_result = await self.memory_agent.process(memory_input)
                    memory_context = memory_result.get("memory_context")
                    similar_docs = memory_result.get("similar_documents", [])
                    if memory_context:
                        self.logger.info(
                            f"Retrieved memory context for document {document_id}"
                        )
                except Exception as e:
                    self.logger.warning(f"Memory retrieval failed: {e}")

            # Step 3: Generate summary using summarization agent
            summarization_input = {
                "document_id": document_id,
                "chunks": chunks,
                "summary_length": summary_length,
                "focus_areas": focus_areas or [],
                "memory_context": memory_context,
                "similar_docs": similar_docs,
            }
            summary_result = await self.summarization_agent.process(summarization_input)

            # Step 4: Validate the summary
            validation_input = {
                "summary": summary_result["summary_text"],
                "original_chunks": chunks,
                "summary_metadata": summary_result.get("summary_metadata", {}),
            }
            validation_result = await self.validator_agent.process(validation_input)

            # Step 5: Store in vector database for future memory if memory is enabled
            if self.memory_agent:
                try:
                    storage_input = {
                        "document_id": document_id,
                        "text": document.extracted_text,
                        "summary_text": summary_result["summary_text"],
                        "operation": "store_memory",
                    }
                    await self.memory_agent.process(storage_input)
                except Exception as e:
                    self.logger.warning(f"Memory storage failed: {e}")

            # Combine all results
            result = {
                **summary_result,
                "validation": validation_result,
                "document_metadata": {
                    "document_id": str(document_id),
                    "filename": document.filename,
                    "chunk_count": len(chunks),
                },
                "memory_context": bool(memory_context),
                "similar_documents_count": len(similar_docs),
                "processing_time_seconds": (
                    datetime.utcnow() - start_time
                ).total_seconds(),
            }

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Store summary in database (create new or update existing)
            existing_summary = await self.db_service.get_summary(document_id)

            if existing_summary and force_regenerate:
                # Update existing summary
                summary_id = await self.db_service.update_summary(
                    document_id=document_id,
                    summary_data={
                        "summary_text": result["summary_text"],
                        "word_count": len(result["summary_text"].split()),
                        "confidence_score": result["confidence_score"],
                        "processing_time_seconds": processing_time,
                        "model_version": result.get("model_version"),
                        "summary_metadata": {
                            "summary_length": summary_length,
                            "focus_areas": focus_areas,
                            "memory_enhanced": bool(memory_context),
                            "validation_score": validation_result.get(
                                "overall_score", 0
                            ),
                            "processing_metadata": result.get("summary_metadata", {}),
                        },
                    },
                )
            elif not existing_summary:
                # Create new summary only if none exists
                summary_id = await self.db_service.store_summary(
                    document_id=document_id,
                    summary_text=result["summary_text"],
                    confidence_score=result["confidence_score"],
                    processing_time_seconds=processing_time,
                    model_version=result.get("model_version"),
                    metadata={
                        "summary_length": summary_length,
                        "focus_areas": focus_areas,
                        "memory_enhanced": bool(memory_context),
                        "validation_score": validation_result.get("overall_score", 0),
                        "processing_metadata": result.get("summary_metadata", {}),
                    },
                )
            else:
                # Summary exists but force_regenerate is False, this shouldn't happen
                # because we already returned early in that case
                self.logger.warning(
                    f"Summary exists for {document_id} but not regenerating"
                )
                summary_id = existing_summary.id

            result["summary_id"] = summary_id

            self.logger.info(
                f"Successfully generated summary for document {document_id} "
                f"in {result['processing_time_seconds']:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Summarization failed for document {document_id}: {e}")
            raise

    async def get_summary(self, document_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve existing summary for a document."""
        try:
            # Try to get from database
            db_summary = await self.db_service.get_document_summary(document_id)
            if db_summary:
                return {
                    "document_id": str(document_id),
                    "summary_text": db_summary.summary_text,
                    "confidence_score": db_summary.confidence_score,
                    "created_at": db_summary.created_at.isoformat(),
                    "metadata": db_summary.metadata or {},
                    "source": "database",
                }

            return None

        except Exception as e:
            self.logger.error(
                f"Failed to retrieve summary for document {document_id}: {e}"
            )
            raise

    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator and system statistics including vector database."""
        try:
            db_stats = await self.db_service.get_document_stats()
            vector_stats = await self.vector_service.get_collection_stats()

            agent_stats = {
                "chunking": self.chunking_agent.get_metrics(),
                "summarization": self.summarization_agent.get_metrics(),
                "validator": self.validator_agent.get_metrics(),
            }

            if self.memory_agent:
                agent_stats["memory"] = self.memory_agent.get_metrics()

            return {
                "database": db_stats,
                "vector_db": vector_stats,
                "agents": agent_stats,
                "memory_enabled": self.settings.enable_memory,
                "system_health": "healthy",
            }
        except Exception as e:
            self.logger.error(f"Failed to get orchestrator stats: {e}")
            return {"system_health": "error", "error": str(e)}

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and all related data."""
        try:
            # Delete from vector database if memory is enabled
            if self.memory_agent:
                try:
                    await self.memory_agent.delete_document_memory(document_id)
                except Exception as e:
                    self.logger.warning(f"Vector database cleanup failed: {e}")

            # Delete from database (cascades to summaries and chunks)
            success = await self.db_service.delete_document(document_id)

            if success:
                self.logger.info(f"Successfully deleted document {document_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
