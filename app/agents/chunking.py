from typing import List, Dict, Any
import tiktoken
from .base import BaseAgent
from ..config import get_settings


class ChunkingAgent(BaseAgent):
    """Agent responsible for splitting documents into manageable chunks.

    Design decisions:
    - Uses tiktoken for accurate token counting (OpenAI-compatible)
    - Implements sliding window with overlap to preserve context
    - Handles different content types with appropriate strategies
    - Optimizes chunk boundaries to avoid splitting sentences
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ChunkingAgent", config)
        self.settings = get_settings()
        self.encoding = tiktoken.encoding_for_model(self.settings.openai_model)

        # Chunking parameters from config or defaults
        self.chunk_size = self.config.get("chunk_size", self.settings.chunk_size)
        self.chunk_overlap = self.config.get(
            "chunk_overlap", self.settings.chunk_overlap
        )

    async def process(
        self, input_data: Dict[str, Any], **kwargs
    ) -> List[Dict[str, Any]]:
        """Split document text into optimized chunks.

        Args:
            input_data: Dict with 'text', 'document_id', and optional 'metadata'

        Returns:
            List of chunk dictionaries with content, tokens, and metadata
        """
        text = input_data["text"]
        document_id = input_data["document_id"]

        if not text or not text.strip():
            self.logger.warning(f"Empty text for document {document_id}")
            return []

        # Split into sentences first for better chunk boundaries
        sentences = self._split_into_sentences(text)
        chunks = self._create_chunks_from_sentences(sentences)

        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            token_count = len(self.encoding.encode(chunk_text))

            chunk_objects.append(
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk_text,
                    "token_count": token_count,
                    "chunk_metadata": {
                        "chunk_method": "sentence_boundary",
                        "overlap_tokens": self.chunk_overlap if i > 0 else 0,
                    },
                }
            )

        self.logger.info(
            f"Created {len(chunk_objects)} chunks for document {document_id}"
        )
        return chunk_objects

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics.

        Production note: Consider using spaCy or NLTK for better sentence detection.
        """
        import re

        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r"[.!?]+\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """Create chunks by combining sentences up to token limit."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))

            # If adding this sentence would exceed limit, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(" ".join(current_chunk))
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = (
                    len(self.encoding.encode(overlap_text)) if overlap_text else 0
                )

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_text(self, chunk_text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if not chunk_text:
            return ""

        words = chunk_text.split()
        overlap_words = []
        overlap_tokens = 0

        # Add words from the end until we reach overlap limit
        for word in reversed(words):
            word_tokens = len(self.encoding.encode(word))
            if overlap_tokens + word_tokens > self.chunk_overlap:
                break
            overlap_words.insert(0, word)
            overlap_tokens += word_tokens

        return " ".join(overlap_words)
