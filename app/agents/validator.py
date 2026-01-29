from typing import Dict, Any, List, Tuple
import re
from .base import BaseAgent


class ValidationResult:
    """Result object for validation checks."""

    def __init__(
        self, is_valid: bool, issues: List[str] = None, suggestions: List[str] = None
    ):
        self.is_valid = is_valid
        self.issues = issues or []
        self.suggestions = suggestions or []
        self.score = 1.0 - (len(self.issues) * 0.2)  # Simple scoring


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating summary quality and completeness.

    Design decisions:
    - Multiple validation checks for comprehensive quality assessment
    - Configurable thresholds for different quality metrics
    - Actionable feedback for summary improvement
    - Integration with confidence scoring from SummarizationAgent
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ValidatorAgent", config)

        # Validation thresholds (configurable)
        self.min_word_count = self.config.get("min_word_count", 20)
        self.max_word_count = self.config.get("max_word_count", 1000)
        self.min_sentence_count = self.config.get("min_sentence_count", 2)
        self.max_repetition_ratio = self.config.get("max_repetition_ratio", 0.3)
        self.min_coverage_score = self.config.get("min_coverage_score", 0.3)

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate summary quality and completeness.

        Args:
            input_data: Dict with 'summary', 'original_chunks', and metadata

        Returns:
            Validation result with score, issues, and suggestions
        """
        summary = input_data["summary"]
        original_chunks = input_data.get("original_chunks", [])
        summary_metadata = input_data.get("summary_metadata", {})

        # Run all validation checks
        checks = [
            self._check_length_requirements(summary),
            self._check_content_quality(summary),
            self._check_structure(summary),
            self._check_repetition(summary),
            self._check_coverage(summary, original_chunks),
        ]

        # Combine results
        all_issues = []
        all_suggestions = []
        validity_scores = []

        for result in checks:
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)
            validity_scores.append(result.score)

        # Calculate overall validation score
        overall_score = sum(validity_scores) / len(validity_scores)
        is_valid = overall_score >= 0.7  # 70% threshold for validity

        validation_result = {
            "is_valid": is_valid,
            "validation_score": overall_score,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "checks_performed": len(checks),
            "metadata": {
                "individual_scores": dict(
                    zip(
                        ["length", "quality", "structure", "repetition", "coverage"],
                        validity_scores,
                    )
                )
            },
        }

        if not is_valid:
            self.logger.warning(
                f"Summary validation failed with score {overall_score:.2f}"
            )
            self.logger.debug(f"Issues: {all_issues}")

        return validation_result

    def _check_length_requirements(self, summary: str) -> ValidationResult:
        """Check if summary meets length requirements."""
        word_count = len(summary.split())
        sentence_count = len([s for s in re.split(r"[.!?]+", summary) if s.strip()])

        issues = []
        suggestions = []

        if word_count < self.min_word_count:
            issues.append(
                f"Summary too short ({word_count} words, minimum {self.min_word_count})"
            )
            suggestions.append("Expand key points with more detail")
        elif word_count > self.max_word_count:
            issues.append(
                f"Summary too long ({word_count} words, maximum {self.max_word_count})"
            )
            suggestions.append("Condense content while maintaining key information")

        if sentence_count < self.min_sentence_count:
            issues.append(
                f"Too few sentences ({sentence_count}, minimum {self.min_sentence_count})"
            )
            suggestions.append("Break down ideas into multiple sentences")

        return ValidationResult(len(issues) == 0, issues, suggestions)

    def _check_content_quality(self, summary: str) -> ValidationResult:
        """Check content quality indicators."""
        issues = []
        suggestions = []

        # Check for placeholder text
        placeholder_patterns = [
            r"\[.*\]",  # [placeholder text]
            r"TODO",
            r"FIXME",
            r"XXX",
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, summary, re.IGNORECASE):
                issues.append("Contains placeholder or incomplete text")
                suggestions.append("Replace placeholder text with actual content")
                break

        # Check for overly generic language
        generic_patterns = [
            r"this document discusses",
            r"the text is about",
            r"in conclusion",
            r"to summarize",
        ]

        generic_count = sum(
            1
            for pattern in generic_patterns
            if re.search(pattern, summary, re.IGNORECASE)
        )

        if generic_count > 2:
            issues.append("Contains too much generic language")
            suggestions.append("Use more specific and informative language")

        return ValidationResult(len(issues) == 0, issues, suggestions)

    def _check_structure(self, summary: str) -> ValidationResult:
        """Check summary structure and formatting."""
        issues = []
        suggestions = []

        # Check for proper sentence structure
        sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]

        for i, sentence in enumerate(sentences):
            # Check sentence length
            if len(sentence.split()) > 50:
                issues.append(
                    f"Sentence {i+1} is too long ({len(sentence.split())} words)"
                )
                suggestions.append("Break long sentences into shorter, clearer ones")

        # Check for consistent formatting
        if summary.count("•") > 0 and summary.count("-") > 0:
            issues.append("Inconsistent bullet point formatting")
            suggestions.append("Use consistent bullet point style throughout")

        return ValidationResult(len(issues) == 0, issues, suggestions)

    def _check_repetition(self, summary: str) -> ValidationResult:
        """Check for excessive repetition in the summary."""
        words = summary.lower().split()
        if not words:
            return ValidationResult(
                False, ["Empty summary"], ["Generate non-empty summary"]
            )

        # Calculate word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1

        total_meaningful_words = sum(word_freq.values())
        repetition_ratio = (
            max(word_freq.values()) / total_meaningful_words
            if total_meaningful_words > 0
            else 0
        )

        issues = []
        suggestions = []

        if repetition_ratio > self.max_repetition_ratio:
            most_repeated = max(word_freq, key=word_freq.get)
            issues.append(
                f"Excessive repetition of '{most_repeated}' ({repetition_ratio:.1%} of content)"
            )
            suggestions.append("Use synonyms and varied language to avoid repetition")

        return ValidationResult(len(issues) == 0, issues, suggestions)

    def _check_coverage(
        self, summary: str, original_chunks: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Check if summary covers the main content adequately.

        Note: This is a simplified coverage check. Production systems might use:
        - Semantic similarity models
        - Topic modeling
        - Named entity overlap analysis
        """
        if not original_chunks:
            return ValidationResult(True, [], [])

        issues = []
        suggestions = []

        # Simple keyword-based coverage check
        original_text = " ".join(chunk["content"] for chunk in original_chunks).lower()
        summary_text = summary.lower()

        # Extract important words (simplified approach)
        import re

        original_words = set(
            re.findall(r"\b\w{4,}\b", original_text)
        )  # Words with 4+ chars
        summary_words = set(re.findall(r"\b\w{4,}\b", summary_text))

        if not original_words:
            return ValidationResult(True, [], [])

        # Calculate coverage ratio
        coverage_ratio = len(original_words.intersection(summary_words)) / len(
            original_words
        )

        if coverage_ratio < self.min_coverage_score:
            issues.append(
                f"Low content coverage ({coverage_ratio:.1%}, minimum {self.min_coverage_score:.1%})"
            )
            suggestions.append(
                "Include more key concepts and terms from the original content"
            )

        return ValidationResult(len(issues) == 0, issues, suggestions)
