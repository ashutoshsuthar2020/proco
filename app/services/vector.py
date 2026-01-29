from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import logging
import asyncio
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CreateCollection,
    PointStruct,
    Filter,
    FieldCondition,
    SearchRequest,
    ScoredPoint,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import get_settings


@dataclass
class DocumentEmbedding:
    """Document embedding with metadata."""

    document_id: UUID
    chunk_id: Optional[UUID]
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class SimilarDocument:
    """Similar document result."""

    document_id: UUID
    chunk_id: Optional[UUID]
    similarity_score: float
    text_preview: str
    metadata: Dict[str, Any]


class VectorService:
    """Vector database service for long-term memory and semantic search.

    Design decisions:
    - Uses Qdrant for production-ready vector storage
    - OpenAI embeddings for semantic similarity
    - Async operations for performance
    - Collection management with automatic initialization
    - Metadata filtering for contextual search
    - Batch processing for efficiency
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize clients
        self.qdrant_client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30,
        )

        self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

        self._collection_initialized = False

    async def initialize_collection(self) -> bool:
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = await self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.settings.qdrant_collection_name not in collection_names:
                # Create collection
                await self.qdrant_client.create_collection(
                    collection_name=self.settings.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.vector_dimension, distance=Distance.COSINE
                    ),
                )
                self.logger.info(
                    f"Created Qdrant collection: {self.settings.qdrant_collection_name}"
                )

            self._collection_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant collection: {e}")
            return False

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                input=text, model=self.settings.openai_embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    async def store_document_embedding(
        self,
        document_id: UUID,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: Optional[UUID] = None,
    ) -> bool:
        """Store document or chunk embedding in vector database."""
        if not self._collection_initialized:
            await self.initialize_collection()

        try:
            # Generate embedding
            embedding = await self.generate_embedding(text)

            # Prepare point data
            point_id = str(chunk_id if chunk_id else document_id)

            # Enhanced metadata
            enhanced_metadata = {
                **metadata,
                "document_id": str(document_id),
                "chunk_id": str(chunk_id) if chunk_id else None,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "text_length": len(text),
                "created_at": metadata.get("created_at"),
                "file_type": metadata.get("file_type", "unknown"),
            }

            # Create point
            point = PointStruct(
                id=point_id, vector=embedding, payload=enhanced_metadata
            )

            # Store in Qdrant
            await self.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name, points=[point]
            )

            self.logger.debug(f"Stored embedding for document {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store embedding: {e}")
            return False

    async def store_multiple_embeddings(
        self, embeddings: List[DocumentEmbedding]
    ) -> int:
        """Store multiple embeddings in batch for efficiency."""
        if not self._collection_initialized:
            await self.initialize_collection()

        if not embeddings:
            return 0

        try:
            points = []
            for emb in embeddings:
                point_id = str(emb.chunk_id if emb.chunk_id else emb.document_id)

                point = PointStruct(
                    id=point_id,
                    vector=emb.embedding,
                    payload={
                        **emb.metadata,
                        "document_id": str(emb.document_id),
                        "chunk_id": str(emb.chunk_id) if emb.chunk_id else None,
                        "text_preview": (
                            emb.text[:200] + "..." if len(emb.text) > 200 else emb.text
                        ),
                        "text_length": len(emb.text),
                    },
                )
                points.append(point)

            # Batch upsert
            await self.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name, points=points
            )

            self.logger.info(f"Stored {len(points)} embeddings in batch")
            return len(points)

        except Exception as e:
            self.logger.error(f"Failed to store batch embeddings: {e}")
            return 0

    async def find_similar_documents(
        self,
        query_text: str,
        limit: Optional[int] = None,
        exclude_document_id: Optional[UUID] = None,
        file_type_filter: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[SimilarDocument]:
        """Find similar documents using semantic search."""
        if not self._collection_initialized:
            await self.initialize_collection()

        limit = limit or self.settings.max_similar_documents
        min_score = min_score or self.settings.similarity_threshold

        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query_text)

            # Build filter conditions
            filter_conditions = []

            if exclude_document_id:
                filter_conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=FieldCondition.MatchValue(value=str(exclude_document_id)),
                        range=None,
                    )
                )

            if file_type_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="file_type",
                        match=FieldCondition.MatchValue(value=file_type_filter),
                    )
                )

            # Search similar vectors
            search_result = await self.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Get more results to filter by score
                with_payload=True,
                score_threshold=min_score,
                query_filter=(
                    Filter(must_not=filter_conditions) if filter_conditions else None
                ),
            )

            # Convert to SimilarDocument objects
            similar_docs = []
            for result in search_result:
                if result.score >= min_score:
                    similar_doc = SimilarDocument(
                        document_id=UUID(result.payload["document_id"]),
                        chunk_id=(
                            UUID(result.payload["chunk_id"])
                            if result.payload.get("chunk_id")
                            else None
                        ),
                        similarity_score=result.score,
                        text_preview=result.payload.get("text_preview", ""),
                        metadata={
                            k: v
                            for k, v in result.payload.items()
                            if k not in ["document_id", "chunk_id", "text_preview"]
                        },
                    )
                    similar_docs.append(similar_doc)

            # Limit final results
            similar_docs = similar_docs[:limit]

            self.logger.debug(f"Found {len(similar_docs)} similar documents for query")
            return similar_docs

        except Exception as e:
            self.logger.error(f"Failed to find similar documents: {e}")
            return []

    async def get_document_context(
        self, document_id: UUID, query_text: Optional[str] = None
    ) -> List[SimilarDocument]:
        """Get contextual documents for memory-enhanced processing."""
        try:
            if query_text:
                # Use query text for semantic search
                return await self.find_similar_documents(
                    query_text=query_text,
                    exclude_document_id=document_id,
                    limit=self.settings.memory_context_window,
                )
            else:
                # Get documents with similar metadata/type
                # This is a simplified approach - could be enhanced with document content
                return await self.find_similar_documents(
                    query_text="document summary analysis",  # Generic query
                    exclude_document_id=document_id,
                    limit=self.settings.memory_context_window,
                )

        except Exception as e:
            self.logger.error(f"Failed to get document context: {e}")
            return []

    async def delete_document_embeddings(self, document_id: UUID) -> bool:
        """Delete all embeddings for a document."""
        if not self._collection_initialized:
            return True  # Nothing to delete

        try:
            # Delete points by document_id filter
            await self.qdrant_client.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=FieldCondition.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )

            self.logger.info(f"Deleted embeddings for document {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document embeddings: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        try:
            if not self._collection_initialized:
                return {"status": "not_initialized"}

            collection_info = await self.qdrant_client.get_collection(
                collection_name=self.settings.qdrant_collection_name
            )

            return {
                "status": "healthy",
                "collection_name": self.settings.qdrant_collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vector_dimension": self.settings.vector_dimension,
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close vector database connections."""
        try:
            await self.qdrant_client.close()
        except Exception as e:
            self.logger.error(f"Error closing Qdrant client: {e}")
