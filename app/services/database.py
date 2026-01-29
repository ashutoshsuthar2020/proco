from typing import Optional, List, Dict, Any
from uuid import UUID
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, delete, text

from ..models.database import Document, DocumentChunk, Summary
from ..config import get_settings


class DatabaseService:
    """Service for database operations with async SQLAlchemy 2.0.

    Design decisions:
    - Uses SQLAlchemy 2.0 async features directly
    - Connection pooling for scalability
    - Centralized database logic for maintainability
    - Transaction support for data consistency
    """

    def __init__(self):
        self.settings = get_settings()

        # Create async engine
        self.engine = create_async_engine(
            self.settings.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            ),
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            echo=self.settings.debug,
        )

        # Create async session factory
        self.async_session = async_sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def connect(self):
        """Test database connection."""
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

    async def disconnect(self):
        """Close database connections."""
        await self.engine.dispose()

    async def calculate_content_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    # Document operations
    async def create_document(self, document_data: Dict[str, Any]) -> Document:
        """Create a new document record."""
        async with self.async_session() as session:
            document = Document(**document_data)
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document

    async def store_document(
        self,
        document_id: UUID,
        filename: str,
        content: bytes,
        extracted_text: str,
        file_hash: str,
        metadata: Dict[str, Any] = None,
    ) -> UUID:
        """Store a document with all its content and metadata."""
        import os

        document_data = {
            "id": document_id,
            "filename": filename,
            "file_type": os.path.splitext(filename)[1].lower().lstrip(".") or "unknown",
            "file_size": len(content),
            "file_hash": file_hash,
            "content": content,
            "extracted_text": extracted_text,
            "doc_metadata": metadata or {},
        }

        document = await self.create_document(document_data)
        return document.id

    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()

    async def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash (for deduplication)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            return result.scalar_one_or_none()

    # Document chunk operations
    async def create_chunks(self, chunks: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create multiple document chunks."""
        if not chunks:
            return []

        # Add IDs to chunks if not present
        for chunk in chunks:
            if "id" not in chunk:
                from uuid import uuid4

                chunk["id"] = uuid4()

        async with self.async_session() as session:
            chunk_objects = [DocumentChunk(**chunk) for chunk in chunks]
            session.add_all(chunk_objects)
            await session.commit()
            return chunk_objects

    async def get_chunks_by_document_id(self, document_id: UUID) -> List[DocumentChunk]:
        """Get all chunks for a document, ordered by index."""
        async with self.async_session() as session:
            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            return list(result.scalars().all())

    # Summary operations
    async def create_summary(self, summary_data: Dict[str, Any]) -> Summary:
        """Create a new summary record."""
        async with self.async_session() as session:
            summary = Summary(**summary_data)
            session.add(summary)
            await session.commit()
            await session.refresh(summary)
            return summary

    async def store_summary(
        self,
        document_id: UUID,
        summary_text: str,
        confidence_score: float = None,
        processing_time_seconds: float = None,
        model_version: str = None,
        metadata: Dict[str, Any] = None,
    ) -> UUID:
        """Store a summary with all its metadata."""
        summary_data = {
            "document_id": document_id,
            "summary_text": summary_text,
            "word_count": len(summary_text.split()),
            "confidence_score": confidence_score,
            "processing_time_seconds": processing_time_seconds,
            "model_version": model_version,
            "summary_metadata": metadata or {},
        }

        summary = await self.create_summary(summary_data)
        return summary.id

    async def get_summary(self, document_id: UUID) -> Optional[Summary]:
        """Get summary by document ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Summary).where(Summary.document_id == document_id)
            )
            return result.scalar_one_or_none()

    async def update_summary(
        self, document_id: UUID, summary_data: Dict[str, Any]
    ) -> Optional[Summary]:
        """Update existing summary."""
        async with self.async_session() as session:
            # Remove document_id from update data if present
            update_data = {k: v for k, v in summary_data.items() if k != "document_id"}

            await session.execute(
                update(Summary)
                .where(Summary.document_id == document_id)
                .values(**update_data)
            )
            await session.commit()

            # Return updated summary
            result = await session.execute(
                select(Summary).where(Summary.document_id == document_id)
            )
            return result.scalar_one_or_none()

    # Utility operations
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and all related records."""
        async with self.async_session() as session:
            async with session.begin():
                # Delete in order: summaries, chunks, document
                await session.execute(
                    delete(Summary).where(Summary.document_id == document_id)
                )
                await session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == document_id
                    )
                )
                result = await session.execute(
                    delete(Document).where(Document.id == document_id)
                )
                return result.rowcount > 0

    async def get_document_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.async_session() as session:
            # Get counts
            doc_result = await session.execute(
                select(text("COUNT(*)")).select_from(Document)
            )
            summary_result = await session.execute(
                select(text("COUNT(*)")).select_from(Summary)
            )
            chunk_result = await session.execute(
                select(text("COUNT(*)")).select_from(DocumentChunk)
            )

            # Get average confidence score
            avg_confidence_result = await session.execute(
                select(text("AVG(confidence_score)")).select_from(Summary)
            )

            return {
                "total_documents": doc_result.scalar() or 0,
                "total_summaries": summary_result.scalar() or 0,
                "total_chunks": chunk_result.scalar() or 0,
                "avg_summary_confidence": float(avg_confidence_result.scalar() or 0),
            }
