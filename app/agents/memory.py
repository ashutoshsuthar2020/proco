from typing import Dict, Any, List, Optional
from uuid import UUID
from .base import BaseAgent
from ..services.vector import VectorService, SimilarDocument


class MemoryAgent(BaseAgent):
    """Agent that provides long-term memory capabilities using vector similarity.

    Design decisions:
    - Enhances summarization with contextual information from similar documents
    - Uses semantic search to find relevant historical content
    - Provides confidence boosting when similar documents are found
    - Integrates seamlessly with existing agent pipeline
    - Configurable similarity thresholds and context window
    """

    def __init__(self, vector_service: VectorService, config: Dict[str, Any] = None):
        super().__init__("MemoryAgent", config)
        self.vector_service = vector_service

        # Memory configuration
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.context_window = self.config.get("context_window", 3)
        self.confidence_boost = self.config.get("confidence_boost", 0.1)
        self.enable_context_summary = self.config.get("enable_context_summary", True)

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Enhance document processing with memory context.

        Args:
            input_data: Dict with 'document_id', 'text' or 'summary', and operation type

        Returns:
            Enhanced data with memory context and confidence adjustments
        """
        document_id = input_data["document_id"]
        operation = input_data.get(
            "operation", "summarize"
        )  # 'summarize', 'chunk', 'validate'

        try:
            # Determine query text based on operation
            if operation == "summarize":
                query_text = input_data.get("summary_text", input_data.get("text", ""))
            elif operation == "chunk":
                query_text = input_data.get("text", "")[
                    :500
                ]  # Use first 500 chars for chunking context
            elif operation == "validate":
                query_text = input_data.get("summary", "")
            else:
                query_text = input_data.get("text", "")

            if not query_text:
                self.logger.warning(
                    f"No query text provided for memory search, document {document_id}"
                )
                return self._enhance_with_empty_context(input_data)

            # Find similar documents
            similar_docs = await self.vector_service.find_similar_documents(
                query_text=query_text,
                exclude_document_id=document_id,
                limit=self.context_window,
                min_score=self.similarity_threshold,
            )

            if not similar_docs:
                self.logger.debug(
                    f"No similar documents found for document {document_id}"
                )
                return self._enhance_with_empty_context(input_data)

            # Generate memory context
            memory_context = await self._generate_memory_context(similar_docs)

            # Enhance input data with memory
            enhanced_data = {
                **input_data,
                "memory_context": memory_context,
                "similar_documents": [
                    {
                        "document_id": str(doc.document_id),
                        "similarity_score": doc.similarity_score,
                        "text_preview": doc.text_preview,
                        "metadata": doc.metadata,
                    }
                    for doc in similar_docs
                ],
                "memory_enhanced": True,
                "confidence_boost": self._calculate_confidence_boost(similar_docs),
            }

            self.logger.info(
                f"Enhanced document {document_id} with {len(similar_docs)} similar documents"
            )

            return enhanced_data

        except Exception as e:
            self.logger.error(
                f"Memory processing failed for document {document_id}: {e}"
            )
            # Return original data if memory fails - don't break the pipeline
            return self._enhance_with_empty_context(input_data)

    def _enhance_with_empty_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add empty memory context to maintain consistent data structure."""
        return {
            **input_data,
            "memory_context": None,
            "similar_documents": [],
            "memory_enhanced": False,
            "confidence_boost": 0.0,
        }

    async def _generate_memory_context(
        self, similar_docs: List[SimilarDocument]
    ) -> str:
        """Generate contextual information from similar documents."""
        if not self.enable_context_summary or not similar_docs:
            return ""

        try:
            context_parts = ["Based on similar documents in the knowledge base:"]

            for i, doc in enumerate(similar_docs, 1):
                similarity_pct = int(doc.similarity_score * 100)
                doc_type = doc.metadata.get("file_type", "document")

                context_parts.append(
                    f"{i}. Similar {doc_type} ({similarity_pct}% match): {doc.text_preview}"
                )

            context_parts.append(
                "\nConsider these similar documents when generating the summary to maintain consistency and identify common patterns."
            )

            return "\n".join(context_parts)

        except Exception as e:
            self.logger.error(f"Failed to generate memory context: {e}")
            return ""

    def _calculate_confidence_boost(self, similar_docs: List[SimilarDocument]) -> float:
        """Calculate confidence boost based on similarity scores."""
        if not similar_docs:
            return 0.0

        # Average similarity score weighted by relevance
        avg_similarity = sum(doc.similarity_score for doc in similar_docs) / len(
            similar_docs
        )

        # Scale confidence boost based on:
        # 1. Average similarity score
        # 2. Number of similar documents found
        # 3. Configured boost factor

        boost = (
            avg_similarity
            * min(len(similar_docs) / self.context_window, 1.0)
            * self.confidence_boost
        )

        return min(boost, 0.2)  # Cap boost at 0.2 to avoid over-confidence

    async def store_document_memory(
        self,
        document_id: UUID,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
    ) -> bool:
        """Store document chunks in vector database for future memory retrieval."""
        try:
            from ..services.vector import DocumentEmbedding

            embeddings = []

            for chunk in chunks:
                # Generate embedding for chunk
                embedding_vector = await self.vector_service.generate_embedding(
                    chunk["content"]
                )

                # Create embedding object
                doc_embedding = DocumentEmbedding(
                    document_id=document_id,
                    chunk_id=chunk.get("id"),
                    text=chunk["content"],
                    embedding=embedding_vector,
                    metadata={
                        **document_metadata,
                        "chunk_index": chunk.get("chunk_index", 0),
                        "token_count": chunk.get("token_count", 0),
                        "chunk_metadata": chunk.get("chunk_metadata", {}),
                    },
                )

                embeddings.append(doc_embedding)

            # Store all embeddings in batch
            stored_count = await self.vector_service.store_multiple_embeddings(
                embeddings
            )

            success = stored_count == len(embeddings)
            if success:
                self.logger.info(
                    f"Stored {stored_count} chunk embeddings for document {document_id}"
                )
            else:
                self.logger.warning(
                    f"Only stored {stored_count}/{len(embeddings)} embeddings"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to store document memory: {e}")
            return False

    async def get_memory_insights(self, document_id: UUID) -> Dict[str, Any]:
        """Get insights about document's relationship to existing knowledge base."""
        try:
            # This would require getting the document content first
            # For now, return basic insights structure
            return {
                "document_id": str(document_id),
                "memory_status": "stored",
                "insights": {
                    "similar_documents_count": 0,
                    "knowledge_cluster": "general",
                    "uniqueness_score": 1.0,  # High = very unique, Low = similar to existing
                    "recommended_focus_areas": [],
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get memory insights: {e}")
            return {"error": str(e)}
