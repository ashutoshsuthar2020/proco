from typing import Dict, Any, Optional, List
import asyncio
import time
from openai import AsyncOpenAI
from .base import BaseAgent
from ..config import get_settings


class SummarizationAgent(BaseAgent):
    """Agent responsible for generating summaries using LLM.

    Design decisions:
    - Async OpenAI client for better performance
    - Configurable prompts for different summary types
    - Token usage tracking for cost optimization
    - Confidence scoring based on model response patterns
    - Handles both single chunks and multi-chunk summarization
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SummarizationAgent", config)
        self.settings = get_settings()

        # Initialize OpenAI client only if API key is available
        self.ai_enabled = bool(
            self.settings.openai_api_key and self.settings.enable_ai_summarization
        )
        if self.ai_enabled:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            self.logger.warning(
                "OpenAI API key not available - using fallback summarization"
            )

        # Summary configuration
        self.model = self.config.get("model", self.settings.openai_model)
        self.max_tokens = self.config.get(
            "max_tokens", self.settings.max_tokens_per_request
        )

        # Length mappings for different summary types
        self.length_configs = {
            "short": {
                "max_words": 100,
                "style": "concise bullet points with key takeaways",
                "format": "bullets",
            },
            "medium": {
                "max_words": 300,
                "style": "structured sections with main points and details",
                "format": "structured",
            },
            "long": {
                "max_words": 600,
                "style": "comprehensive analysis with detailed sections",
                "format": "comprehensive",
            },
        }

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate summary for document content with optional memory enhancement.

        Args:
            input_data: Dict with 'chunks', 'document_id', and optional parameters

        Returns:
            Summary dictionary with text, metadata, and confidence score
        """
        chunks = input_data["chunks"]
        document_id = input_data["document_id"]
        summary_length = input_data.get("summary_length", "medium")
        focus_areas = input_data.get("focus_areas", [])
        memory_context = input_data.get("memory_context")  # Memory enhancement
        similar_docs = input_data.get("similar_documents", [])

        start_time = time.time()

        try:
            # Use fallback summarization if AI is not available
            if not self.ai_enabled:
                summary_text = self._generate_fallback_summary(chunks, summary_length)
                processing_time = time.time() - start_time
                word_count = len(summary_text.split())
                confidence_score = 0.7  # Default confidence for fallback
            else:
                # AI-powered summarization
                if len(chunks) == 1:
                    # Single chunk - direct summarization with memory context
                    summary_text = await self._summarize_single_chunk(
                        chunks[0]["content"],
                        summary_length,
                        focus_areas,
                        memory_context,
                    )
                else:
                    # Multiple chunks - hierarchical summarization with memory context
                    summary_text = await self._summarize_multiple_chunks(
                        chunks, summary_length, focus_areas, memory_context
                    )

                processing_time = time.time() - start_time
                word_count = len(summary_text.split())
                confidence_score = self._calculate_confidence_score(
                    summary_text, chunks, similar_docs
                )

            return {
                "document_id": document_id,
                "summary_text": summary_text,
                "word_count": word_count,
                "confidence_score": confidence_score,
                "processing_time_seconds": processing_time,
                "model_version": self.model,
                "summary_metadata": {
                    "length_type": summary_length,
                    "focus_areas": focus_areas,
                    "chunk_count": len(chunks),
                    "total_tokens": sum(chunk["token_count"] for chunk in chunks),
                    "memory_enhanced": bool(memory_context),
                    "similar_docs_referenced": len(similar_docs),
                },
            }

        except Exception as e:
            self.logger.error(
                f"Summarization failed for document {document_id}: {str(e)}"
            )
            raise

    async def _summarize_single_chunk(
        self,
        content: str,
        length: str,
        focus_areas: List[str],
        memory_context: Optional[str] = None,
    ) -> str:
        """Summarize a single chunk of content with optional memory context."""

        # Use fallback if AI is disabled or API call fails
        if not self.ai_enabled:
            return self._generate_simple_chunk_summary(content, length)

        try:
            prompt = self._build_prompt(
                content, length, focus_areas, memory_context=memory_context
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more consistent summaries
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"OpenAI API failed, using fallback: {str(e)}")
            return self._generate_simple_chunk_summary(content, length)

    async def _summarize_multiple_chunks(
        self,
        chunks: List[Dict[str, Any]],
        length: str,
        focus_areas: List[str],
        memory_context: Optional[str] = None,
    ) -> str:
        """Hierarchical summarization for multiple chunks with memory context.

        Strategy: Summarize chunks in parallel, then create final summary with memory context
        """

        # Use fallback if AI is disabled
        if not self.ai_enabled:
            return self._generate_fallback_summary(chunks, length)

        try:
            # Step 1: Summarize each chunk in parallel (without memory context to avoid repetition)
            chunk_summaries = await asyncio.gather(
                *[
                    self._summarize_single_chunk(chunk["content"], "short", focus_areas)
                    for chunk in chunks
                ]
            )

            # Step 2: Combine chunk summaries into final summary with memory context
            combined_text = " ".join(chunk_summaries)
            final_prompt = self._build_prompt(
                combined_text,
                length,
                focus_areas,
                is_final=True,
                memory_context=memory_context,
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": final_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(
                f"OpenAI API failed for multi-chunk, using fallback: {str(e)}"
            )
            return self._generate_fallback_summary(chunks, length)

    def _build_prompt(
        self,
        content: str,
        length: str,
        focus_areas: List[str],
        is_final: bool = False,
        memory_context: Optional[str] = None,
    ) -> str:
        """Build summarization prompt based on parameters and memory context."""
        length_config = self.length_configs[length]
        format_type = length_config.get("format", "structured")

        prompt_parts = [
            f"Please provide a {length} summary of the following content.",
            f"Target length: approximately {length_config['max_words']} words.",
            f"Style: {length_config['style']}.",
        ]

        # Add specific formatting instructions based on length type
        if format_type == "bullets":
            prompt_parts.extend(
                [
                    "\nFormat your response as:",
                    "📋 **Key Points:**",
                    "• [Main point 1]",
                    "• [Main point 2]",
                    "• [Main point 3]",
                    "\n💡 **Main Takeaway:**",
                    "[One sentence summary of the most important insight]",
                ]
            )
        elif format_type == "structured":
            prompt_parts.extend(
                [
                    "\nFormat your response with clear sections:",
                    "📖 **Overview:**",
                    "[Brief 2-3 sentence overview]",
                    "\n🔍 **Key Points:**",
                    "• [Main point 1 with brief explanation]",
                    "• [Main point 2 with brief explanation]",
                    "• [Main point 3 with brief explanation]",
                    "\n💭 **Conclusion:**",
                    "[Key takeaway or implication]",
                ]
            )
        elif format_type == "comprehensive":
            prompt_parts.extend(
                [
                    "\nFormat your response with detailed sections:",
                    "📖 **Executive Summary:**",
                    "[2-3 sentence high-level overview]",
                    "\n🔍 **Main Topics:**",
                    "• [Topic 1]: [Detailed explanation]",
                    "• [Topic 2]: [Detailed explanation]",
                    "• [Topic 3]: [Detailed explanation]",
                    "\n📊 **Key Insights:**",
                    "• [Important finding or pattern 1]",
                    "• [Important finding or pattern 2]",
                    "\n💡 **Implications:**",
                    "[What this means and why it matters]",
                ]
            )

        if focus_areas:
            prompt_parts.append(f"\nFocus particularly on: {', '.join(focus_areas)}.")

        # Add memory context if available
        if memory_context:
            prompt_parts.append("\n🧠 **Additional Context from Knowledge Base:**")
            prompt_parts.append(memory_context)
            prompt_parts.append(
                "\nPlease incorporate insights from similar documents where relevant."
            )

        if is_final:
            prompt_parts.append(
                "\nNote: This content consists of summaries from different sections. Create a cohesive final summary that follows the format above."
            )

        prompt_parts.extend(["\n📄 **Content to summarize:**", content])

        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """System prompt for consistent summarization behavior."""
        return (
            "You are an expert document summarization assistant that creates structured, readable summaries. "
            "Always follow the exact format provided in the user prompt with proper sections and emoji headers. "
            "Create clear, accurate, and well-organized summaries that capture the key points and main ideas. "
            "Use bullet points, sections, and formatting to make content easy to scan and understand. "
            "Focus on factual accuracy while maintaining the original context and meaning. "
            "Ensure each section serves a purpose and provides valuable information to the reader."
        )

    def _calculate_confidence_score(
        self,
        summary: str,
        chunks: List[Dict[str, Any]],
        similar_docs: List[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence score based on summary characteristics and memory context.

        This is a heuristic approach - in production, you might use:
        - Model confidence scores (if available)
        - Semantic similarity between summary and original
        - Coverage analysis
        - Memory context relevance
        """
        if not summary or not chunks:
            return 0.0

        similar_docs = similar_docs or []

        # Simple heuristics for confidence scoring
        score = 1.0

        # Penalize very short summaries
        word_count = len(summary.split())
        if word_count < 20:
            score -= 0.3

        # Penalize generic language patterns
        generic_phrases = ["this document", "the text discusses", "in summary"]
        generic_count = sum(
            1 for phrase in generic_phrases if phrase in summary.lower()
        )
        score -= generic_count * 0.1

        # Boost score for structured content
        if any(marker in summary for marker in ["•", "-", "1.", "Key points:"]):
            score += 0.1

        # Boost confidence when similar documents provide context
        if similar_docs:
            memory_boost = min(len(similar_docs) * 0.05, 0.15)  # Max 0.15 boost
            score += memory_boost

        return max(0.0, min(1.0, score))

    def _generate_fallback_summary(
        self, chunks: List[Dict[str, Any]], summary_length: str
    ) -> str:
        """Generate a comprehensive structured extractive summary when AI is not available."""

        # Combine all chunk content
        full_text = " ".join([chunk["content"] for chunk in chunks])

        # Get target word count and extract comprehensive information
        length_config = self.length_configs.get(summary_length, {"max_words": 300})
        target_words = length_config["max_words"]
        format_type = length_config.get("format", "structured")

        # Extract detailed information
        doc_analysis = self._analyze_document_content(full_text)
        key_points = self._extract_key_points(full_text, target_words // 2)
        main_summary = self._extract_main_summary(full_text, target_words // 2)

        # Build structured summary based on format type
        if format_type == "bullets":
            structured_summary = f"📋 **Key Points:**\n"
            for i, point in enumerate(key_points[:5], 1):
                structured_summary += f"{i}. {point}\n"
            structured_summary += f"\n💡 **Document Type:** {doc_analysis['type']}"
            if doc_analysis["key_entities"]:
                structured_summary += f"\n🏷️ **Key Entities:** {', '.join(doc_analysis['key_entities'][:3])}"

        elif format_type == "comprehensive":
            structured_summary = f"📖 **Executive Summary:**\n{main_summary}\n\n"
            structured_summary += f"🔍 **Key Highlights:**\n"
            for i, point in enumerate(key_points[:4], 1):
                structured_summary += f"• {point}\n"
            structured_summary += f"\n📊 **Document Analysis:**\n"
            structured_summary += f"• Type: {doc_analysis['type']}\n"
            structured_summary += f"• Content Length: {len(full_text.split())} words\n"
            if doc_analysis["key_entities"]:
                structured_summary += (
                    f"• Key Entities: {', '.join(doc_analysis['key_entities'][:4])}\n"
                )

        else:  # structured (default)
            structured_summary = f"📄 **Document Overview:**\n{main_summary}\n\n"
            structured_summary += f"🔍 **Key Points:**\n"
            for i, point in enumerate(key_points[:4], 1):
                structured_summary += f"{i}. {point}\n"
            structured_summary += f"\n📋 **Document Details:**\n"
            structured_summary += f"• Document Type: {doc_analysis['type']}\n"
            structured_summary += f"• Content Sections: {doc_analysis['sections']}\n"
            if doc_analysis["key_entities"]:
                structured_summary += f"• Key Information: {', '.join(doc_analysis['key_entities'][:3])}\n"

        return structured_summary

    def _generate_simple_chunk_summary(self, content: str, length: str) -> str:
        """Generate a comprehensive structured extractive summary for a single chunk."""

        # Get target word count and analyze content
        length_config = self.length_configs.get(length, {"max_words": 300})
        target_words = length_config["max_words"]
        format_type = length_config.get("format", "structured")

        # Extract detailed information
        doc_analysis = self._analyze_document_content(content)
        key_points = self._extract_key_points(content, target_words // 2)
        main_summary = self._extract_main_summary(content, target_words // 2)

        # Apply enhanced structure based on format type
        if format_type == "bullets":
            structured_summary = f"📋 **Key Points:**\n"
            for i, point in enumerate(key_points[:4], 1):
                structured_summary += f"{i}. {point}\n"
            if doc_analysis["key_entities"]:
                structured_summary += f"\n🏷️ **Key Details:** {', '.join(doc_analysis['key_entities'][:3])}"

        elif format_type == "comprehensive":
            structured_summary = f"📖 **Executive Summary:**\n{main_summary}\n\n"
            structured_summary += f"🔍 **Key Highlights:**\n"
            for i, point in enumerate(key_points[:4], 1):
                structured_summary += f"• {point}\n"
            structured_summary += f"\n📊 **Document Type:** {doc_analysis['type']}"
            if doc_analysis["key_entities"]:
                structured_summary += f"\n💡 **Key Details:** {', '.join(doc_analysis['key_entities'][:3])}"

        else:  # structured (default)
            structured_summary = f"📄 **Overview:**\n{main_summary}\n\n"
            structured_summary += f"🔍 **Key Points:**\n"
            for i, point in enumerate(key_points[:3], 1):
                structured_summary += f"{i}. {point}\n"
            structured_summary += f"\n📋 **Document Info:**\n"
            structured_summary += f"• Type: {doc_analysis['type']}\n"
            if doc_analysis["key_entities"]:
                structured_summary += (
                    f"• Key Details: {', '.join(doc_analysis['key_entities'][:2])}"
                )

        return structured_summary

    def _extract_key_information(self, text: str) -> Dict[str, str]:
        """Extract key information from document content for fallback summaries."""
        text_lower = text.lower()

        # Detect document type based on keywords
        doc_type = "Document"
        key_details = []

        # Check for resume/CV indicators
        if any(
            keyword in text_lower
            for keyword in [
                "experience",
                "education",
                "skills",
                "university",
                "bachelor",
                "software engineer",
            ]
        ):
            doc_type = "Resume/CV"
            # Extract name if present (usually at the beginning)
            first_lines = text.split("\\n")[:3]
            for line in first_lines:
                if (
                    len(line.strip()) > 0
                    and len(line.split()) <= 5
                    and not any(char in line for char in "@.com")
                ):
                    # Likely a name
                    key_details.append(f"Candidate: {line.strip()}")
                    break
            # Extract current role if mentioned
            if "software engineer" in text_lower:
                key_details.append("Role: Software Engineer")
            # Extract company if mentioned
            for company in ["cisco", "google", "microsoft", "amazon", "apple"]:
                if company in text_lower:
                    key_details.append(f"Company: {company.title()}")
                    break

        # Check for business/technical document indicators
        elif any(
            keyword in text_lower
            for keyword in ["project", "system", "architecture", "api", "database"]
        ):
            doc_type = "Technical Document"
            if "api" in text_lower:
                key_details.append("Contains API information")
            if "system" in text_lower:
                key_details.append("System-related content")

        # Check for academic document indicators
        elif any(
            keyword in text_lower
            for keyword in ["research", "abstract", "methodology", "conclusion"]
        ):
            doc_type = "Academic Paper"

        # Check for business document indicators
        elif any(
            keyword in text_lower
            for keyword in ["revenue", "profit", "market", "business", "strategy"]
        ):
            doc_type = "Business Document"

        return {
            "doc_type": doc_type,
            "key_details": " | ".join(key_details) if key_details else None,
        }

    def _analyze_document_content(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of document content."""
        text_lower = text.lower()
        lines = [line.strip() for line in text.split("\\n") if line.strip()]

        # Document type detection with more categories
        doc_type = "General Document"
        key_entities = []
        sections = 0

        # Resume/CV Detection
        if any(
            keyword in text_lower
            for keyword in [
                "experience",
                "education",
                "skills",
                "university",
                "bachelor",
                "resume",
                "cv",
            ]
        ):
            doc_type = "Professional Resume/CV"

            # Extract name (usually in first few lines)
            for line in lines[:5]:
                if (
                    len(line.split()) <= 4
                    and len(line) > 2
                    and not any(char in line for char in ["@", ".com", "http"])
                ):
                    if not any(
                        word in line.lower()
                        for word in ["phone", "email", "address", "linkedin"]
                    ):
                        key_entities.append(f"Name: {line}")
                        break

            # Extract contact info
            for line in lines[:10]:
                if "@" in line and ".com" in line:
                    key_entities.append(
                        f"Email: {line.split()[0] if ' ' in line else line}"
                    )
                elif any(
                    keyword in line.lower() for keyword in ["phone", "mobile", "tel"]
                ):
                    numbers = "".join(filter(str.isdigit, line))
                    if len(numbers) >= 10:
                        key_entities.append(f"Phone: {numbers}")

            # Extract current role/company
            if "software engineer" in text_lower:
                key_entities.append("Role: Software Engineer")
            elif "developer" in text_lower:
                key_entities.append("Role: Developer")
            elif "manager" in text_lower:
                key_entities.append("Role: Manager")

            # Extract notable companies
            for company in [
                "cisco",
                "google",
                "microsoft",
                "amazon",
                "apple",
                "meta",
                "netflix",
                "uber",
            ]:
                if company in text_lower:
                    key_entities.append(f"Company: {company.title()}")
                    break

            # Count sections
            section_headers = [
                "experience",
                "education",
                "skills",
                "projects",
                "work",
                "employment",
            ]
            sections = sum(1 for header in section_headers if header in text_lower)

        # Technical Document Detection
        elif any(
            keyword in text_lower
            for keyword in [
                "api",
                "system",
                "architecture",
                "database",
                "server",
                "cloud",
            ]
        ):
            doc_type = "Technical Documentation"

            # Extract technical keywords
            tech_terms = []
            if "api" in text_lower:
                tech_terms.append("API")
            if "kubernetes" in text_lower or "k8s" in text_lower:
                tech_terms.append("Kubernetes")
            if "python" in text_lower:
                tech_terms.append("Python")
            if "docker" in text_lower:
                tech_terms.append("Docker")
            if "aws" in text_lower or "amazon web services" in text_lower:
                tech_terms.append("AWS")

            if tech_terms:
                key_entities.append(f"Technologies: {', '.join(tech_terms[:3])}")

            sections = len(
                [line for line in lines if line.endswith(":") or "##" in line]
            )

        # Business Document Detection
        elif any(
            keyword in text_lower
            for keyword in [
                "revenue",
                "profit",
                "business",
                "strategy",
                "market",
                "sales",
            ]
        ):
            doc_type = "Business Document"

            # Extract business metrics
            if "revenue" in text_lower:
                key_entities.append("Contains: Revenue data")
            if "profit" in text_lower:
                key_entities.append("Contains: Profit information")
            if "market" in text_lower:
                key_entities.append("Contains: Market analysis")

        # Academic Paper Detection
        elif any(
            keyword in text_lower
            for keyword in [
                "abstract",
                "methodology",
                "research",
                "conclusion",
                "bibliography",
            ]
        ):
            doc_type = "Academic Paper"

            # Extract academic elements
            if "abstract" in text_lower:
                key_entities.append("Structure: Abstract included")
            if "methodology" in text_lower:
                key_entities.append("Structure: Methodology section")
            if "conclusion" in text_lower:
                key_entities.append("Structure: Conclusion section")

        return {
            "type": doc_type,
            "key_entities": key_entities,
            "sections": (
                sections
                if sections > 0
                else len([line for line in lines if ":" in line])
            ),
        }

    def _extract_key_points(self, text: str, max_words: int) -> List[str]:
        """Extract key points from text content."""
        sentences = [
            s.strip() for s in text.split(".") if s.strip() and len(s.split()) > 3
        ]

        # Score sentences based on importance indicators
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()

            # Boost score for sentences with important keywords
            importance_keywords = [
                "experience",
                "skills",
                "responsible",
                "developed",
                "managed",
                "led",
                "created",
                "implemented",
                "achieved",
            ]
            score += sum(
                2 for keyword in importance_keywords if keyword in sentence_lower
            )

            # Boost for sentences with numbers (metrics, dates, etc.)
            score += len(
                [
                    word
                    for word in sentence.split()
                    if any(char.isdigit() for char in word)
                ]
            )

            # Boost for sentences with proper nouns (likely important entities)
            score += len(
                [
                    word
                    for word in sentence.split()
                    if word[0].isupper() and len(word) > 2
                ]
            )

            # Penalize very short or very long sentences
            word_count = len(sentence.split())
            if word_count < 5:
                score -= 2
            elif word_count > 25:
                score -= 1

            scored_sentences.append((sentence, score))

        # Sort by score and extract top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        key_points = []
        total_words = 0

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= max_words and len(key_points) < 5:
                # Clean up the sentence
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith("."):
                    clean_sentence += "."
                key_points.append(clean_sentence)
                total_words += sentence_words
            else:
                break

        return (
            key_points
            if key_points
            else [text.split(".")[0] + "." if "." in text else text[:100] + "..."]
        )

    def _extract_main_summary(self, text: str, max_words: int) -> str:
        """Extract a coherent main summary from the text."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        if not sentences:
            return text[: max_words * 5] + "..." if len(text) > max_words * 5 else text

        # Start with first few sentences for context
        summary_sentences = []
        word_count = 0

        for sentence in sentences[:10]:  # Look at first 10 sentences
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_words:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                break

        if not summary_sentences:
            # Fallback to word-based truncation
            words = text.split()[:max_words]
            return " ".join(words) + ("..." if len(text.split()) > max_words else "")

        summary = ". ".join(summary_sentences)
        if not summary.endswith("."):
            summary += "."

        return summary
