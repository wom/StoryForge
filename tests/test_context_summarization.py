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
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=50)
            result = cm._summarize_context("Luna fox adventure")

            assert result is not None
            assert len(result) > 0
            # Should mention Luna since it's in prompt
            assert "Luna" in result or "fox" in result
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_no_files_returns_none(self, tmp_path):
        """Test returns None when no context files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(empty_dir)
        try:
            cm = ContextManager()
            result = cm._summarize_context("test prompt")
            assert result is None
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_respects_token_budget(self, temp_context_dir):
        """Test summarization respects token budget."""
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
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
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]


class TestExtractRelevantContext:
    """Test the main extract_relevant_context method."""

    def test_summarization_returns_compressed_content(self, temp_context_dir):
        """Test returns summarized context with token budget."""
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
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
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_summarization_without_budget(self, temp_context_dir):
        """Test summarization works without token budget (returns all summaries)."""
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            cm = ContextManager(max_tokens=None)
            result = cm.extract_relevant_context("Luna adventure")

            assert result is not None
            # Should contain content from both files
            assert "Luna" in result or "Enchanted Forest" in result
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_no_context_returns_none(self, tmp_path):
        """Test returns None when no context available."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(empty_dir)
        try:
            cm = ContextManager()
            result = cm.extract_relevant_context("test")
            assert result is None
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]


class TestClearCache:
    """Test cache clearing."""

    def test_clear_cache_clears_all(self, temp_context_dir):
        """Test clear_cache clears all caches."""
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
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
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]


class TestSelectContextFiles:
    """Test temporal stratified sampling for file selection."""

    def _make_files(self, tmp_path, count):
        """Create numbered files with ascending mtimes."""
        files = []
        for i in range(count):
            f = tmp_path / f"story_{i:03d}.md"
            f.write_text(f"Story {i}")
            # Set distinct mtimes
            os.utime(f, (1000000 + i, 1000000 + i))
            files.append(f)
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def test_small_pool_returns_all(self, tmp_path):
        """Test small file pools (<= threshold) return all files."""
        cm = ContextManager()
        files = self._make_files(tmp_path, cm.STRATIFIED_THRESHOLD)
        result = cm._select_context_files(files)
        assert result == files
        assert not cm._has_old_context

    def test_large_pool_stratified(self, tmp_path):
        """Test large pool uses stratified sampling."""
        cm = ContextManager()
        files = self._make_files(tmp_path, 30)
        result = cm._select_context_files(files)

        # Should include recent files
        recent = files[-cm.RECENT_FILE_COUNT :]
        for f in recent:
            assert f in result

        # Should have sampled from old eras
        assert cm._has_old_context

        # Total should be <= RECENT_FILE_COUNT + NUM_ERAS * SAMPLES_PER_ERA
        max_expected = cm.RECENT_FILE_COUNT + cm.NUM_ERAS * cm.SAMPLES_PER_ERA
        assert len(result) <= max_expected

    def test_result_sorted_by_mtime(self, tmp_path):
        """Test result is sorted oldest-first."""
        cm = ContextManager()
        files = self._make_files(tmp_path, 25)
        result = cm._select_context_files(files)
        mtimes = [f.stat().st_mtime for f in result]
        assert mtimes == sorted(mtimes)


class TestDivideIntoEras:
    """Test _divide_into_eras static method."""

    def test_empty_files(self):
        """Test empty list returns empty."""
        assert ContextManager._divide_into_eras([], 5) == []

    def test_zero_eras(self):
        """Test zero eras returns empty."""
        from pathlib import Path

        assert ContextManager._divide_into_eras([Path("a")], 0) == []

    def test_even_division(self, tmp_path):
        """Test even division into eras."""
        files = [tmp_path / f"{i}.md" for i in range(10)]
        eras = ContextManager._divide_into_eras(files, 5)
        assert len(eras) == 5
        assert all(len(era) == 2 for era in eras)

    def test_uneven_division(self, tmp_path):
        """Test uneven files are distributed across eras."""
        files = [tmp_path / f"{i}.md" for i in range(7)]
        eras = ContextManager._divide_into_eras(files, 3)
        # ceil(7/3) = 3 per chunk -> 3, 3, 1
        total = sum(len(era) for era in eras)
        assert total == 7

    def test_more_eras_than_files(self, tmp_path):
        """Test more eras than files produces one per era."""
        files = [tmp_path / f"{i}.md" for i in range(3)]
        eras = ContextManager._divide_into_eras(files, 10)
        total = sum(len(era) for era in eras)
        assert total == 3


class TestDeduplicateSentences:
    """Test sentence deduplication."""

    def test_no_duplicates(self):
        """Test text without duplicates is unchanged."""
        cm = ContextManager()
        text = "First sentence. Second sentence. Third sentence."
        result = cm._deduplicate_sentences(text)
        assert result == text

    def test_removes_exact_duplicates(self):
        """Test exact duplicate sentences are removed."""
        cm = ContextManager()
        text = "Luna is brave. Max is wise. Luna is brave. The end."
        result = cm._deduplicate_sentences(text)
        assert result.count("Luna is brave") == 1
        assert "Max is wise" in result
        assert "The end" in result

    def test_case_insensitive_dedup(self):
        """Test deduplication is case-insensitive."""
        cm = ContextManager()
        text = "Hello world. HELLO WORLD. Something else."
        result = cm._deduplicate_sentences(text)
        # Should keep only the first occurrence
        assert "Hello world" in result
        assert "HELLO WORLD" not in result
        assert "Something else" in result

    def test_empty_text(self):
        """Test empty text returns empty."""
        cm = ContextManager()
        assert cm._deduplicate_sentences("") == ""

    def test_single_sentence(self):
        """Test single sentence is unchanged."""
        cm = ContextManager()
        text = "Just one sentence."
        result = cm._deduplicate_sentences(text)
        assert result == text

    def test_preserves_trailing_period(self):
        """Test trailing period is preserved."""
        cm = ContextManager()
        text = "First. Second. First."
        result = cm._deduplicate_sentences(text)
        assert result.rstrip().endswith(".")


class TestScoreChunkCharacterBonus:
    """Test character bonus scoring in _score_chunk."""

    def test_character_bonus_applied(self):
        """Test character bonus is added when known character appears in both."""
        cm = ContextManager()
        cm._known_characters = {"Luna", "Max"}
        chunk = "Luna explored the forest near the river"
        prompt = "Luna goes on an adventure"
        score = cm._score_chunk(chunk, prompt)
        # Base score + CHARACTER_SCORE_BONUS for Luna
        assert score >= cm.CHARACTER_SCORE_BONUS

    def test_no_bonus_without_known_characters(self):
        """Test no bonus when _known_characters is empty."""
        cm = ContextManager()
        cm._known_characters = set()
        chunk = "Luna explored"
        prompt = "Luna adventure"
        score = cm._score_chunk(chunk, prompt)
        # Only base keyword scoring
        assert score == 1  # Only "Luna" matches (as keyword)

    def test_multiple_character_bonuses(self):
        """Test multiple characters each add a bonus."""
        cm = ContextManager()
        cm._known_characters = {"Luna", "Max"}
        chunk = "Luna and Max ran through the meadow"
        prompt = "Luna and Max go on a trip"
        score = cm._score_chunk(chunk, prompt)
        # Should include bonuses for both Luna and Max
        assert score >= 2 * cm.CHARACTER_SCORE_BONUS

    def test_no_bonus_character_only_in_chunk(self):
        """Test no bonus when character only in chunk, not prompt."""
        cm = ContextManager()
        cm._known_characters = {"Luna"}
        chunk = "Luna ran fast"
        prompt = "the forest was dark"
        score = cm._score_chunk(chunk, prompt)
        assert score < cm.CHARACTER_SCORE_BONUS


class TestMergeSummariesDeduplicates:
    """Test that _merge_summaries_and_pins applies deduplication."""

    def test_dedup_applied_to_merged_output(self):
        """Test duplicate sentences are removed from merged output."""
        cm = ContextManager()
        items = [
            {"summary": "Luna is brave. Max is wise.", "score": 5, "summary_tokens": 10, "is_pinned": False},
            {"summary": "Luna is brave. The forest glows.", "score": 3, "summary_tokens": 10, "is_pinned": False},
        ]
        result = cm._merge_summaries_and_pins(items, max_tokens=None)
        # "Luna is brave" should appear only once
        assert result.count("Luna is brave") == 1
        assert "Max is wise" in result
        assert "The forest glows" in result


class TestHasOldContext:
    """Test has_old_context property."""

    def test_default_false(self):
        """Test default is False."""
        cm = ContextManager()
        assert cm.has_old_context is False

    def test_set_by_select(self, tmp_path):
        """Test set to True when old files are sampled."""
        cm = ContextManager()
        files = []
        for i in range(25):
            f = tmp_path / f"story_{i:03d}.md"
            f.write_text(f"Story {i}")
            os.utime(f, (1000000 + i, 1000000 + i))
            files.append(f)
        files = sorted(files, key=lambda p: p.stat().st_mtime)
        cm._select_context_files(files)
        assert cm.has_old_context is True


class TestBackwardsCompatibility:
    """Test that core ContextManager behavior is maintained."""

    def test_default_behavior_is_summarized(self, temp_context_dir):
        """Test that default behavior now uses summarization."""
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(temp_context_dir)
        try:
            # Create with no summarization args (minimal API)
            cm = ContextManager()
            result = cm.extract_relevant_context("test")

            # Should return summarized context (always on)
            assert result is not None
            # Should still contain relevant content
            assert len(result) > 0
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

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
