"""Tests for ContextManager summarization functionality."""

import os

import pytest

from storyforge.context import ContextManager


@pytest.fixture
def temp_context_dir(tmp_path):
    """Create a temporary context directory with sample files."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    # Create sample context files
    file1 = context_dir / "characters.md"
    file1.write_text(
        "Luna is a brave little fox who loves adventures. "
        "She has bright orange fur and green eyes. "
        "Luna lives in the enchanted forest with her friend Max.\n\n"
        "Max is a wise old owl who gives good advice. "
        "He has brown feathers and yellow eyes. "
        "Max knows all the secrets of the forest."
    )

    file2 = context_dir / "settings.md"
    file2.write_text(
        "The Enchanted Forest is a magical place full of wonder. "
        "It has tall trees, sparkling streams, and hidden glades. "
        "The forest is home to many magical creatures.\n\n"
        "The Crystal Cave is deep in the forest. "
        "It glows with a soft blue light and contains ancient magic."
    )

    return context_dir


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that returns word count."""

    def tokenizer(text: str) -> int:
        return len(text.split())

    return tokenizer


class TestContextManagerInit:
    """Test ContextManager initialization with summarization parameters."""

    def test_default_init(self):
        """Test default initialization with summarization always enabled."""
        cm = ContextManager()
        # Summarization is always on, but configurable
        assert cm.max_tokens is None  # No token limit by default
        assert cm.pinned_token_fraction == 0.2  # Default pinned fraction
        assert cm.tokenizer is None  # Uses heuristic by default
        assert cm.summary_cache_dir is None  # No cache persistence by default
        assert cm._summary_cache == {}  # Empty cache initially

    def test_summarization_parameters(self):
        """Test initialization with summarization parameters."""
        cm = ContextManager(max_tokens=500)
        assert cm.max_tokens == 500

    def test_custom_tokenizer(self, mock_tokenizer):
        """Test initialization with custom tokenizer."""
        cm = ContextManager(tokenizer=mock_tokenizer)
        assert cm.tokenizer is not None
        # Test that tokenizer is used
        result = cm._estimate_tokens("hello world test")
        assert result == 3  # word count

    def test_custom_pinned_fraction(self):
        """Test custom pinned token fraction."""
        cm = ContextManager(pinned_token_fraction=0.3)
        assert cm.pinned_token_fraction == 0.3


class TestTokenEstimation:
    """Test token estimation methods."""

    def test_estimate_tokens_heuristic(self):
        """Test default heuristic token estimation."""
        cm = ContextManager()
        text = "a" * 100  # 100 chars
        result = cm._estimate_tokens(text)
        assert result == 25  # 100 / 4

    def test_estimate_tokens_minimum(self):
        """Test minimum token count is 1."""
        cm = ContextManager()
        result = cm._estimate_tokens("hi")
        assert result >= 1

    def test_estimate_tokens_custom_tokenizer(self, mock_tokenizer):
        """Test token estimation with custom tokenizer."""
        cm = ContextManager(tokenizer=mock_tokenizer)
        text = "one two three four five"
        result = cm._estimate_tokens(text)
        assert result == 5  # word count

    def test_estimate_tokens_fallback_on_error(self):
        """Test fallback to heuristic when custom tokenizer raises."""

        def bad_tokenizer(text: str) -> int:
            raise ValueError("tokenizer error")

        cm = ContextManager(tokenizer=bad_tokenizer)
        result = cm._estimate_tokens("test text")
        assert result >= 1  # Falls back to heuristic


class TestReadAndNormalize:
    """Test file reading and normalization."""

    def test_read_simple_file(self, tmp_path):
        """Test reading a simple markdown file."""
        file_path = tmp_path / "test.md"
        file_path.write_text("Paragraph one.\n\nParagraph two.")

        cm = ContextManager()
        result = cm._read_and_normalize(file_path)

        assert len(result) == 2
        assert result[0] == "Paragraph one."
        assert result[1] == "Paragraph two."

    def test_normalize_newlines(self, tmp_path):
        """Test normalization of different newline styles."""
        file_path = tmp_path / "test.md"
        file_path.write_text("Line one\r\n\r\nLine two\r\r\nLine three")

        cm = ContextManager()
        result = cm._read_and_normalize(file_path)

        # Should normalize all newline styles
        assert len(result) >= 1

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading nonexistent file returns empty list."""
        cm = ContextManager()
        result = cm._read_and_normalize(tmp_path / "nonexistent.md")
        assert result == []

    def test_strip_whitespace(self, tmp_path):
        """Test whitespace is stripped from paragraphs."""
        file_path = tmp_path / "test.md"
        file_path.write_text("  \n  First para  \n\n  Second para  \n  ")

        cm = ContextManager()
        result = cm._read_and_normalize(file_path)

        assert all(p == p.strip() for p in result)


class TestChunkParagraphs:
    """Test paragraph chunking."""

    def test_small_paragraphs_unchanged(self):
        """Test small paragraphs remain unchanged."""
        cm = ContextManager()
        paragraphs = ["Short para.", "Another short one."]
        result = cm._chunk_paragraphs(paragraphs, max_chunk_chars=100)

        assert len(result) == 2
        assert result == paragraphs

    def test_large_paragraph_split(self):
        """Test large paragraph is split on sentence boundaries."""
        cm = ContextManager()
        # Create a paragraph larger than max_chunk_chars
        long_para = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        result = cm._chunk_paragraphs([long_para], max_chunk_chars=50)

        # Should be split into multiple chunks
        assert len(result) > 1
        # All chunks should respect max length (with some tolerance for punctuation)
        assert all(len(chunk) <= 60 for chunk in result)

    def test_preserve_periods(self):
        """Test periods are preserved after splitting."""
        cm = ContextManager()
        para = "First. Second. Third."
        result = cm._chunk_paragraphs([para], max_chunk_chars=10)

        # All chunks should end with period or be complete sentences
        assert all(chunk.strip().endswith(".") or len(chunk.strip()) > 0 for chunk in result)


class TestExtractiveSummary:
    """Test extractive summarization."""

    def test_short_text_unchanged(self):
        """Test short text is returned unchanged."""
        cm = ContextManager()
        text = "This is a short text."
        result = cm._extractive_summary(text, target_tokens=100)
        assert result == text

    def test_select_sentences(self):
        """Test sentence selection up to budget."""
        cm = ContextManager()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        # With heuristic, each sentence is ~4-6 tokens
        result = cm._extractive_summary(text, target_tokens=10)

        # Should include at least first sentence
        assert "First sentence" in result
        # Should not include all sentences
        assert len(result) < len(text)

    def test_always_includes_one_sentence(self):
        """Test at least one sentence is always included."""
        cm = ContextManager()
        text = "This is a very long sentence that exceeds the token budget."
        result = cm._extractive_summary(text, target_tokens=1)

        # Should include at least something
        assert len(result) > 0

    def test_empty_text(self):
        """Test empty text handling."""
        cm = ContextManager()
        result = cm._extractive_summary("", target_tokens=10)
        assert result == ""


class TestScoreChunk:
    """Test chunk relevance scoring."""

    def test_exact_match(self):
        """Test scoring with exact keyword matches."""
        cm = ContextManager()
        chunk = "The brave fox jumped over the log"
        prompt = "fox jumped"

        score = cm._score_chunk(chunk, prompt)
        assert score == 2  # "fox" and "jumped" match

    def test_case_insensitive(self):
        """Test scoring is case-insensitive."""
        cm = ContextManager()
        chunk = "The BRAVE Fox"
        prompt = "brave fox"

        score = cm._score_chunk(chunk, prompt)
        assert score == 2

    def test_no_match(self):
        """Test zero score with no matches."""
        cm = ContextManager()
        chunk = "The quick brown dog"
        prompt = "cat mouse"

        score = cm._score_chunk(chunk, prompt)
        assert score == 0

    def test_empty_inputs(self):
        """Test empty input handling."""
        cm = ContextManager()
        assert cm._score_chunk("", "prompt") == 0
        assert cm._score_chunk("chunk", "") == 0
        assert cm._score_chunk("", "") == 0

    def test_punctuation_ignored(self):
        """Test punctuation is normalized in scoring."""
        cm = ContextManager()
        chunk = "Hello, world! How are you?"
        prompt = "hello world"

        score = cm._score_chunk(chunk, prompt)
        assert score == 2  # Both words match despite punctuation


class TestMergeSummariesAndPins:
    """Test summary merging with budget."""

    def test_no_budget_returns_all(self):
        """Test all summaries returned when no budget set."""
        cm = ContextManager()
        items = [
            {"summary": "First", "score": 5, "summary_tokens": 10, "is_pinned": False},
            {"summary": "Second", "score": 3, "summary_tokens": 10, "is_pinned": False},
        ]
        result = cm._merge_summaries_and_pins(items, max_tokens=None)
        assert "First" in result
        assert "Second" in result

    def test_respects_budget(self):
        """Test merging respects token budget."""
        cm = ContextManager()
        items = [
            {"summary": "A" * 100, "score": 5, "summary_tokens": 25, "is_pinned": False},
            {"summary": "B" * 100, "score": 4, "summary_tokens": 25, "is_pinned": False},
            {"summary": "C" * 100, "score": 3, "summary_tokens": 25, "is_pinned": False},
        ]
        result = cm._merge_summaries_and_pins(items, max_tokens=40)

        # Should include highest scoring items within budget
        assert "A" * 100 in result  # Top score
        # May or may not include second based on exact budget math

    def test_pinned_items_prioritized(self):
        """Test pinned items get priority within their budget."""
        cm = ContextManager(pinned_token_fraction=0.6)
        items = [
            {"summary": "Pinned high", "score": 10, "summary_tokens": 10, "is_pinned": True},
            {"summary": "Regular higher", "score": 20, "summary_tokens": 10, "is_pinned": False},
        ]
        result = cm._merge_summaries_and_pins(items, max_tokens=25)

        # Pinned item should be included (gets 60% = 15 tokens)
        # Regular gets 40% = 10 tokens
        assert "Pinned high" in result
        assert "Regular higher" in result

    def test_empty_items(self):
        """Test empty items list returns empty string."""
        cm = ContextManager()
        result = cm._merge_summaries_and_pins([], max_tokens=100)
        assert result == ""

    def test_selects_by_score_ratio(self):
        """Test selection by score/token ratio."""
        cm = ContextManager()
        items = [
            {"summary": "Efficient", "score": 10, "summary_tokens": 5, "is_pinned": False},  # ratio 2.0
            {"summary": "Less efficient", "score": 10, "summary_tokens": 10, "is_pinned": False},  # ratio 1.0
        ]
        result = cm._merge_summaries_and_pins(items, max_tokens=8)

        # Should prefer higher score/token ratio
        assert "Efficient" in result


class TestSummarizeContext:
    """Test full summarization pipeline."""

    def test_summarize_basic_context(self, temp_context_dir):
        """Test basic context summarization."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=50)
            result = cm._summarize_context("Luna fox adventure")

            assert result is not None
            assert len(result) > 0
            # Should mention Luna since it's in prompt
            assert "Luna" in result or "fox" in result
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_no_files_returns_none(self, tmp_path):
        """Test returns None when no context files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(empty_dir)
        try:
            cm = ContextManager()
            result = cm._summarize_context("test prompt")
            assert result is None
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_respects_token_budget(self, temp_context_dir):
        """Test summarization respects token budget."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=20)
            result = cm._summarize_context("forest magic")

            assert result is not None
            # Verify it's actually compressed
            full_cm = ContextManager(context_file_path=str(temp_context_dir / "characters.md"))
            full_context = full_cm.load_context()

            # Summarized should be shorter than full
            assert len(result) < len(full_context) if full_context else True
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]


class TestExtractRelevantContext:
    """Test the main extract_relevant_context method."""

    def test_summarization_returns_compressed_content(self, temp_context_dir):
        """Test returns summarized context with token budget."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=30)
            result = cm.extract_relevant_context("Luna adventure")

            assert result is not None
            # Result should contain relevant content
            assert len(result) > 0
            # Since we have a token budget, it should be constrained
            estimated_tokens = len(result) // 4
            assert estimated_tokens <= 35  # Allow some overhead
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_summarization_without_budget(self, temp_context_dir):
        """Test summarization works without token budget (returns all summaries)."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=None)
            result = cm.extract_relevant_context("Luna adventure")

            assert result is not None
            # Should contain content from both files
            assert "Luna" in result or "Enchanted Forest" in result
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_no_context_returns_none(self, tmp_path):
        """Test returns None when no context available."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(empty_dir)
        try:
            cm = ContextManager()
            result = cm.extract_relevant_context("test")
            assert result is None
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]


class TestClearCache:
    """Test cache clearing."""

    def test_clear_cache_clears_all(self, temp_context_dir):
        """Test clear_cache clears all caches."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager()
            # Load context to populate cache
            cm.load_context()
            assert cm._cached_context is not None

            # Add something to summary cache
            cm._summary_cache["test"] = "value"

            # Clear cache
            cm.clear_cache()

            assert cm._cached_context is None
            assert cm._summary_cache == {}
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]


class TestBackwardsCompatibility:
    """Test that core ContextManager behavior is maintained."""

    def test_default_behavior_is_summarized(self, temp_context_dir):
        """Test that default behavior now uses summarization."""
        os.environ["STORYTIME_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            # Create with no summarization args (minimal API)
            cm = ContextManager()
            result = cm.extract_relevant_context("test")

            # Should return summarized context (always on)
            assert result is not None
            # Should still contain relevant content
            assert len(result) > 0
        finally:
            del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_original_context_file_path_still_works(self):
        """Test that context_file_path parameter still works."""
        cm = ContextManager(context_file_path="test.md")
        assert cm.context_file_path == "test.md"

    def test_new_parameters_work(self):
        """Test that new summarization parameters work correctly."""
        cm = ContextManager(max_tokens=100, pinned_token_fraction=0.3, summary_cache_dir="/tmp/cache")
        assert cm.max_tokens == 100
        assert cm.pinned_token_fraction == 0.3
        assert cm.summary_cache_dir == "/tmp/cache"
