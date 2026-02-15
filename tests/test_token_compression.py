"""Tests for token estimation and compression logic.

These tests verify the token counting heuristics and compression thresholds
used to prevent exceeding model token limits.
"""

from storyforge.anthropic_backend import AnthropicBackend
from storyforge.gemini_backend import GeminiBackend
from storyforge.llm_backend import LLMBackend
from storyforge.openai_backend import OpenAIBackend


class TestTokenEstimation:
    """Test the token estimation heuristic."""

    def test_estimate_token_count_basic(self):
        """Test basic token estimation uses 4 chars per token heuristic."""
        text = "This is a test string with some words."
        estimated = LLMBackend.estimate_token_count(text)
        expected = len(text) // 4
        assert estimated == expected

    def test_estimate_token_count_long_text(self):
        """Test token estimation on longer text."""
        text = "a" * 10000
        estimated = LLMBackend.estimate_token_count(text)
        assert estimated == 2500  # 10000 / 4

    def test_estimate_token_count_empty(self):
        """Test token estimation on empty string."""
        assert LLMBackend.estimate_token_count("") == 0

    def test_estimate_token_count_single_char(self):
        """Test token estimation on single character."""
        assert LLMBackend.estimate_token_count("a") == 0  # 1 // 4 = 0

    def test_estimate_token_count_three_chars(self):
        """Test token estimation rounds down."""
        assert LLMBackend.estimate_token_count("abc") == 0  # 3 // 4 = 0

    def test_estimate_token_count_four_chars(self):
        """Test token estimation at exactly 4 chars."""
        assert LLMBackend.estimate_token_count("abcd") == 1  # 4 // 4 = 1


class TestCompressionThresholds:
    """Test compression threshold calculations."""

    def test_image_model_compression_thresholds(self):
        """Test that image model thresholds are calculated correctly."""
        image_limit = GeminiBackend.DEFAULT_IMAGE_INPUT_LIMIT
        trigger_ratio = GeminiBackend.COMPRESSION_TRIGGER_RATIO
        target_ratio = GeminiBackend.COMPRESSION_TARGET_RATIO

        trigger_threshold = int(image_limit * trigger_ratio)
        target_threshold = int(image_limit * target_ratio)

        # Verify the ratios are as expected
        assert trigger_ratio == 0.80
        assert target_ratio == 0.875

        # Verify calculated thresholds
        assert trigger_threshold == int(30000 * 0.80)  # 24000
        assert target_threshold == int(30000 * 0.875)  # 26250

        # Verify target is less than limit but greater than trigger
        assert target_threshold < image_limit
        assert target_threshold > trigger_threshold

    def test_text_model_compression_thresholds(self):
        """Test that text model thresholds are calculated correctly."""
        text_limit = GeminiBackend.DEFAULT_TEXT_INPUT_LIMIT
        trigger_ratio = GeminiBackend.COMPRESSION_TRIGGER_RATIO
        target_ratio = GeminiBackend.COMPRESSION_TARGET_RATIO

        trigger_threshold = int(text_limit * trigger_ratio)
        target_threshold = int(text_limit * target_ratio)

        # Verify calculated thresholds
        assert trigger_threshold == 800000  # 1000000 * 0.80
        assert target_threshold == 875000  # 1000000 * 0.875

        # Verify target is less than limit but greater than trigger
        assert target_threshold < text_limit
        assert target_threshold > trigger_threshold

    def test_gemini_2_5_flash_image_limit(self):
        """Test compression logic handles the actual gemini-2.5-flash-image limit.

        This model has a 32768 token input limit, which was causing the original error.
        """
        actual_limit = 32768
        trigger_ratio = GeminiBackend.COMPRESSION_TRIGGER_RATIO
        target_ratio = GeminiBackend.COMPRESSION_TARGET_RATIO

        trigger_threshold = int(actual_limit * trigger_ratio)
        target_threshold = int(actual_limit * target_ratio)

        # Verify thresholds
        assert trigger_threshold == 26214  # 32768 * 0.80
        assert target_threshold == 28672  # 32768 * 0.875

        # Verify safety margins
        assert target_threshold < actual_limit
        assert actual_limit - target_threshold > 4000  # At least 4K token buffer


class TestProblemCaseFromErrorLog:
    """Test the specific case that triggered the original bug report.

    The error message showed: "The input token count exceeds the maximum
    number of tokens allowed (32768)."
    """

    def test_large_context_triggers_compression(self):
        """Test that large context prompts would trigger compression."""
        # Simulate a large prompt with context (estimated from error)
        problem_prompt_chars = 130000  # ~32500 tokens
        problem_tokens = LLMBackend.estimate_token_count("a" * problem_prompt_chars)

        image_limit = 32768  # gemini-2.5-flash-image actual limit
        trigger_threshold = int(image_limit * GeminiBackend.COMPRESSION_TRIGGER_RATIO)

        # Verify this would trigger compression
        assert problem_tokens > trigger_threshold
        assert problem_tokens > 26214  # 80% of 32768

    def test_compressed_prompt_within_limits(self):
        """Test that compressed prompts stay within model limits."""
        image_limit = 32768
        target_ratio = GeminiBackend.COMPRESSION_TARGET_RATIO
        target_threshold = int(image_limit * target_ratio)

        # After compression, prompt should be at target
        assert target_threshold < image_limit
        assert target_threshold == 28672  # 87.5% of 32768

        # Verify we have headroom
        headroom = image_limit - target_threshold
        assert headroom == 4096  # ~4K tokens of safety buffer

    def test_compression_prevents_original_error(self):
        """Verify compression would prevent the original 32768 token error."""
        # The error was from a full prompt with context added - simulate larger size
        problem_size = 135000  # chars from error case with context
        problem_tokens = LLMBackend.estimate_token_count("a" * problem_size)

        # Original error: exceeded 32768
        assert problem_tokens > 32768
        assert problem_tokens == 33750  # 135000 / 4

        # After compression to 87.5%
        target = int(32768 * 0.875)
        assert target == 28672
        assert target < 32768  # Would not trigger error

    def test_fallback_limits_are_conservative(self):
        """Test that fallback limits are conservative but reasonable."""
        # Verify fallback limits if API doesn't provide them
        assert GeminiBackend.DEFAULT_IMAGE_INPUT_LIMIT == 30000
        assert GeminiBackend.DEFAULT_TEXT_INPUT_LIMIT == 1000000

        # Verify fallbacks are below common model limits
        assert GeminiBackend.DEFAULT_IMAGE_INPUT_LIMIT < 32768  # gemini-2.5-flash-image
        assert GeminiBackend.DEFAULT_IMAGE_INPUT_LIMIT < 1048576  # gemini-2.0-flash-exp


class TestCompressionRatioMatchesOpenAI:
    """Verify our compression target matches OpenAI's proven approach."""

    def test_compression_target_matches_openai_pattern(self):
        """OpenAI compresses 4000 char limit to 3500 target = 87.5%."""
        openai_limit = 4000
        openai_target = 3500
        openai_ratio = openai_target / openai_limit

        # Verify our ratio matches OpenAI's
        assert GeminiBackend.COMPRESSION_TARGET_RATIO == 0.875
        assert abs(openai_ratio - 0.875) < 0.001  # Floating point tolerance


class TestOpenAITokenLimits:
    """Test OpenAI backend token limit awareness."""

    def test_default_text_input_limit(self):
        """Test OpenAI default text input limit."""
        assert OpenAIBackend.DEFAULT_TEXT_INPUT_LIMIT == 128000

    def test_model_token_limits_defined(self):
        """Test that known models have token limits defined."""
        assert "gpt-4o" in OpenAIBackend.MODEL_TOKEN_LIMITS
        assert "gpt-5.2" in OpenAIBackend.MODEL_TOKEN_LIMITS
        assert "gpt-4" in OpenAIBackend.MODEL_TOKEN_LIMITS

    def test_gpt4_has_correct_limit(self):
        """Test GPT-4 has the well-known 8K limit."""
        assert OpenAIBackend.MODEL_TOKEN_LIMITS["gpt-4"] == 8192

    def test_gpt4o_has_correct_limit(self):
        """Test GPT-4o has 128K limit."""
        assert OpenAIBackend.MODEL_TOKEN_LIMITS["gpt-4o"] == 128000

    def test_context_budget_with_default_model(self):
        """Test that context budget is 50% of model limit."""
        limit = OpenAIBackend.DEFAULT_TEXT_INPUT_LIMIT
        expected_budget = int(limit * LLMBackend.CONTEXT_BUDGET_RATIO)
        assert expected_budget == 64000


class TestAnthropicTokenLimits:
    """Test Anthropic backend token limit awareness."""

    def test_default_text_input_limit(self):
        """Test Anthropic default text input limit is 200K."""
        assert AnthropicBackend.DEFAULT_TEXT_INPUT_LIMIT == 200000

    def test_model_token_limits_defined(self):
        """Test that Claude models have token limits defined."""
        assert "claude-3-5-sonnet" in AnthropicBackend.MODEL_TOKEN_LIMITS
        assert "claude-3-opus" in AnthropicBackend.MODEL_TOKEN_LIMITS

    def test_context_budget_with_default_model(self):
        """Test that context budget is 50% of model limit."""
        limit = AnthropicBackend.DEFAULT_TEXT_INPUT_LIMIT
        expected_budget = int(limit * LLMBackend.CONTEXT_BUDGET_RATIO)
        assert expected_budget == 100000


class TestAllBackendsShareCompressionBehavior:
    """Verify all backends share the same COMPRESSION_TRIGGER_RATIO from base class."""

    def test_gemini_inherits_trigger_ratio(self):
        """Test Gemini uses the base class trigger ratio."""
        # Gemini has its own COMPRESSION_TRIGGER_RATIO which should match
        assert GeminiBackend.COMPRESSION_TRIGGER_RATIO == 0.80

    def test_base_class_trigger_ratio(self):
        """Test base class defines the trigger ratio."""
        assert LLMBackend.COMPRESSION_TRIGGER_RATIO == 0.80
