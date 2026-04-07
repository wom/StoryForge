import os
from unittest.mock import MagicMock, patch

import pytest

from storyforge.llm_backend import ERROR_STORY_SENTINEL, LLMBackend, get_backend
from storyforge.prompt import Prompt


class DummyBackend(LLMBackend):
    name = "dummy"

    def generate_story(self, prompt: Prompt) -> str:
        raise NotImplementedError("Test implementation")

    def generate_image(self, prompt: Prompt):
        raise NotImplementedError("Test implementation")

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        raise NotImplementedError("Test implementation")

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        raise NotImplementedError("Test implementation")


# Original abstract base class tests
def test_generate_story_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_story(prompt)


def test_generate_image_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_image(prompt)


def test_generate_image_name_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_image_name(prompt, "story")


class TestTextInputLimit:
    """Test text_input_limit property and context budget."""

    def test_default_text_input_limit(self):
        """Test default text_input_limit when _text_input_limit is not set."""
        backend = DummyBackend()
        assert backend.text_input_limit == 8192

    def test_custom_text_input_limit(self):
        """Test text_input_limit reads from _text_input_limit."""
        backend = DummyBackend()
        backend._text_input_limit = 200000
        assert backend.text_input_limit == 200000

    def test_get_context_token_budget(self):
        """Test context budget is 50% of text_input_limit."""
        backend = DummyBackend()
        backend._text_input_limit = 128000
        assert backend.get_context_token_budget() == 64000

    def test_get_context_token_budget_default(self):
        """Test context budget with default limit."""
        backend = DummyBackend()
        # Default is 8192, budget is 50% = 4096
        assert backend.get_context_token_budget() == 4096

    def test_context_budget_ratio(self):
        """Test CONTEXT_BUDGET_RATIO is 0.50."""
        assert LLMBackend.CONTEXT_BUDGET_RATIO == 0.50

    def test_compression_trigger_ratio(self):
        """Test COMPRESSION_TRIGGER_RATIO is 0.80."""
        assert LLMBackend.COMPRESSION_TRIGGER_RATIO == 0.80

    def test_check_and_truncate_prompt_under_limit(self):
        """Test that prompts under limit are returned unchanged."""
        backend = DummyBackend()
        backend._text_input_limit = 100000
        short_prompt = "A short prompt"
        result = backend._check_and_truncate_prompt(short_prompt)
        assert result == short_prompt

    def test_check_and_truncate_prompt_over_limit(self):
        """Test that prompts over 80% of limit are truncated."""
        backend = DummyBackend()
        backend._text_input_limit = 100  # 100 tokens = 400 chars
        # 80% trigger = 80 tokens = 320 chars
        long_prompt = "x" * 400  # 100 tokens, over 80% trigger
        result = backend._check_and_truncate_prompt(long_prompt)
        assert len(result) == 320  # 80 tokens * 4 chars


# Factory function tests
class TestGetBackend:
    """Test the get_backend factory function."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.gemini_backend.GeminiBackend")
    def test_get_backend_auto_detect_gemini(self, mock_gemini):
        """Test auto-detection of Gemini backend via API key."""
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        backend = get_backend()

        mock_gemini.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_auto_detect_openai(self, mock_openai):
        """Test auto-detection of OpenAI backend via API key."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend()

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "gemini", "GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.gemini_backend.GeminiBackend")
    def test_get_backend_explicit_gemini(self, mock_gemini):
        """Test explicit Gemini backend selection."""
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        backend = get_backend()

        mock_gemini.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_get_backend_no_api_keys(self):
        """Test error when no API keys are available."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        assert "No LLM backend available" in str(exc_info.value)
        assert "GEMINI_API_KEY" in str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.anthropic_backend.AnthropicBackend")
    def test_get_backend_auto_detect_anthropic(self, mock_anthropic):
        """Test auto-detection of Anthropic backend via API key."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        backend = get_backend()

        mock_anthropic.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.anthropic_backend.AnthropicBackend")
    def test_get_backend_explicit_anthropic(self, mock_anthropic):
        """Test explicit Anthropic backend selection."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        backend = get_backend()

        mock_anthropic.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "openai", "OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_explicit_openai(self, mock_openai):
        """Test explicit OpenAI backend selection."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend()

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "unknown"}, clear=True)
    def test_get_backend_unknown_backend(self):
        """Test error for unknown backends."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        error_msg = str(exc_info.value)
        assert "Unknown backend 'unknown'" in error_msg
        assert "Supported backends: gemini, openai, anthropic" in error_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_explicit_parameter(self, mock_openai):
        """Test explicit backend parameter overrides environment."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend("openai")

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    def test_get_backend_import_error_handling(self):
        """Test handling of import errors."""
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError) as exc_info:
                get_backend("gemini")

            assert "Failed to import gemini backend" in str(exc_info.value)


class TestBuildImagePromptRequest:
    """Test the shared _build_image_prompt_request method."""

    def test_basic_prompt_request(self):
        """Test basic image prompt request construction."""
        backend = DummyBackend()
        result = backend._build_image_prompt_request(
            story="A mouse found cheese.",
            context="",
            num_prompts=2,
        )
        assert "exactly 2" in result
        assert "A mouse found cheese." in result
        assert "Opening scene" in result
        assert "Climactic or closing scene" in result
        assert "CHARACTER VISUAL DESCRIPTIONS" not in result

    def test_prompt_request_with_context(self):
        """Test image prompt request includes context."""
        backend = DummyBackend()
        result = backend._build_image_prompt_request(
            story="A mouse found cheese.",
            context="Forest setting",
            num_prompts=1,
        )
        assert "Forest setting" in result
        assert "CONTEXT" in result

    def test_prompt_request_with_character_descriptions(self):
        """Test image prompt request includes character descriptions."""
        backend = DummyBackend()
        result = backend._build_image_prompt_request(
            story="A mouse found cheese.",
            context="",
            num_prompts=1,
            character_descriptions="- Mouse: small, brown fur, round ears",
        )
        assert "CHARACTER VISUAL DESCRIPTIONS" in result
        assert "small, brown fur, round ears" in result
        assert "consistent" in result.lower()


class TestParseNumberedPrompts:
    """Test the shared _parse_numbered_prompts method."""

    def test_parse_standard_format(self):
        """Test parsing '1. prompt' format."""
        text = "1. First scene description\n2. Second scene description"
        result = LLMBackend._parse_numbered_prompts(text, 2)
        assert result == ["First scene description", "Second scene description"]

    def test_parse_parenthesis_format(self):
        """Test parsing '1) prompt' format."""
        text = "1) First scene\n2) Second scene\n3) Third scene"
        result = LLMBackend._parse_numbered_prompts(text, 3)
        assert result == ["First scene", "Second scene", "Third scene"]

    def test_parse_colon_format(self):
        """Test parsing '1: prompt' format."""
        text = "1: Scene one\n2: Scene two"
        result = LLMBackend._parse_numbered_prompts(text, 2)
        assert result == ["Scene one", "Scene two"]

    def test_parse_bold_format(self):
        """Test parsing '**1.** prompt' format."""
        text = "**1.** Bold scene one\n**2.** Bold scene two"
        result = LLMBackend._parse_numbered_prompts(text, 2)
        assert result == ["Bold scene one", "Bold scene two"]

    def test_parse_multiline_prompts(self):
        """Test that non-numbered continuation lines are joined."""
        text = "1. First scene starts here\nand continues on this line\n2. Second scene"
        result = LLMBackend._parse_numbered_prompts(text, 2)
        assert result is not None
        assert "First scene starts here and continues on this line" in result[0]
        assert result[1] == "Second scene"

    def test_parse_too_few_prompts_returns_none(self):
        """Test that fewer prompts than expected returns None."""
        text = "1. Only one prompt"
        result = LLMBackend._parse_numbered_prompts(text, 3)
        assert result is None

    def test_parse_extra_prompts_truncated(self):
        """Test that extra prompts are truncated to requested number."""
        text = "1. One\n2. Two\n3. Three\n4. Four"
        result = LLMBackend._parse_numbered_prompts(text, 2)
        assert result == ["One", "Two"]

    def test_parse_empty_text_returns_none(self):
        """Test that empty text returns None."""
        result = LLMBackend._parse_numbered_prompts("", 2)
        assert result is None


class TestSegmentStory:
    """Test the _segment_story helper."""

    def test_segment_even_split(self):
        """Test even paragraph distribution."""
        story = "Para one.\nPara two.\nPara three.\nPara four."
        result = LLMBackend._segment_story(story, 2)
        assert len(result) == 2
        assert "Para one." in result[0]
        assert "Para two." in result[0]
        assert "Para three." in result[1]
        assert "Para four." in result[1]

    def test_segment_fewer_paragraphs(self):
        """Test fewer paragraphs than segments pads with last."""
        story = "Only one."
        result = LLMBackend._segment_story(story, 3)
        assert len(result) == 3
        assert result[0] == "Only one."

    def test_segment_empty_story(self):
        """Test empty story returns repeated full story."""
        result = LLMBackend._segment_story("", 2)
        assert len(result) == 2

    def test_segment_single(self):
        """Test single segment returns the whole story content."""
        story = "Para one.\nPara two.\nPara three."
        result = LLMBackend._segment_story(story, 1)
        assert len(result) == 1
        assert "Para one." in result[0]
        assert "Para three." in result[0]


class TestGetSceneLabels:
    """Test the _get_scene_labels helper."""

    def test_single_image(self):
        """Test single image gets a key-moment label."""
        labels = LLMBackend._get_scene_labels(1)
        assert len(labels) == 1
        assert "key moment" in labels[0].lower()

    def test_two_images(self):
        """Test two images get opening and closing labels."""
        labels = LLMBackend._get_scene_labels(2)
        assert len(labels) == 2
        assert "opening" in labels[0].lower()

    def test_five_images(self):
        """Test five images get five distinct labels."""
        labels = LLMBackend._get_scene_labels(5)
        assert len(labels) == 5
        # All labels should be unique
        assert len(set(labels)) == 5

    def test_six_images_extends(self):
        """Test more than 5 images extends with additional labels."""
        labels = LLMBackend._get_scene_labels(6)
        assert len(labels) == 6


class TestFallbackImagePrompts:
    """Test the updated _generate_fallback_image_prompts."""

    def test_fallback_includes_scene_labels(self):
        """Test fallback prompts include scene labels."""
        backend = DummyBackend()
        story = "First paragraph.\nSecond paragraph.\nThird paragraph."
        result = backend._generate_fallback_image_prompts(story, "", 2)
        assert len(result) == 2
        assert "opening" in result[0].lower()

    def test_fallback_includes_context(self):
        """Test fallback prompts include context when provided."""
        backend = DummyBackend()
        story = "A story."
        result = backend._generate_fallback_image_prompts(story, "Forest setting", 1)
        assert len(result) == 1
        assert "Forest setting" in result[0]

    def test_fallback_includes_story_segments(self):
        """Test fallback prompts include distinct story segments."""
        backend = DummyBackend()
        story = "Start of story.\nMiddle part.\nEnd of story."
        result = backend._generate_fallback_image_prompts(story, "", 2)
        assert len(result) == 2
        # First segment should have start, second should have end
        assert "Start of story" in result[0]
        assert "End of story" in result[1]


class TestSanitizeImageName:
    """Test the _sanitize_image_name static method."""

    def test_normal_filename_with_extension(self):
        """Test extension is stripped from a normal filename."""
        assert LLMBackend._sanitize_image_name("brave_mouse.png") == "brave_mouse"

    def test_multiple_dots(self):
        """Test only text before the first dot is kept."""
        assert LLMBackend._sanitize_image_name("image.tar.gz") == "image"

    def test_no_extension(self):
        """Test filename without extension is returned as-is."""
        assert LLMBackend._sanitize_image_name("brave_mouse") == "brave_mouse"

    def test_special_characters_stripped(self):
        """Test non-alphanumeric, non-underscore characters are removed."""
        assert LLMBackend._sanitize_image_name("image-@#$%.png") == "image"

    def test_spaces_stripped(self):
        """Test spaces are removed from the filename."""
        assert LLMBackend._sanitize_image_name("my image.png") == "myimage"

    def test_empty_string_fallback(self):
        """Test empty string returns the fallback name."""
        assert LLMBackend._sanitize_image_name("") == "story_image"

    def test_only_dots_fallback(self):
        """Test string of only dots returns the fallback name."""
        assert LLMBackend._sanitize_image_name("...") == "story_image"

    def test_only_special_chars_fallback(self):
        """Test string of only special chars returns the fallback name."""
        assert LLMBackend._sanitize_image_name("!!!.png") == "story_image"

    def test_underscores_preserved(self):
        """Test underscores and digits are preserved."""
        assert LLMBackend._sanitize_image_name("my_image_123") == "my_image_123"

    def test_mixed_valid_invalid_chars(self):
        """Test mix of valid and invalid characters."""
        assert LLMBackend._sanitize_image_name("hello-world_2.jpg") == "helloworld_2"


class TestIsTransientError:
    """Test transient error detection logic."""

    def test_503_in_message(self):
        """503 in error string should be transient."""
        error = Exception("503 UNAVAILABLE. This model is experiencing high demand.")
        assert LLMBackend._is_transient_error(error) is True

    def test_429_in_message(self):
        """429 rate limit in error string should be transient."""
        error = Exception("429 Too Many Requests")
        assert LLMBackend._is_transient_error(error) is True

    def test_500_in_message(self):
        """500 internal server error should be transient."""
        error = Exception("500 Internal Server Error")
        assert LLMBackend._is_transient_error(error) is True

    def test_502_in_message(self):
        """502 bad gateway should be transient."""
        error = Exception("502 Bad Gateway")
        assert LLMBackend._is_transient_error(error) is True

    def test_status_code_attribute(self):
        """Error with status_code attribute should be detected."""
        error = Exception("Service unavailable")
        error.status_code = 503
        assert LLMBackend._is_transient_error(error) is True

    def test_overloaded_keyword(self):
        """'overloaded' keyword should be detected as transient."""
        error = Exception("The server is currently overloaded")
        assert LLMBackend._is_transient_error(error) is True

    def test_high_demand_keyword(self):
        """'high demand' keyword should be detected as transient."""
        error = Exception("This model is currently experiencing high demand")
        assert LLMBackend._is_transient_error(error) is True

    def test_rate_limit_keyword(self):
        """'rate limit' keyword should be detected as transient."""
        error = Exception("rate limit exceeded, please slow down")
        assert LLMBackend._is_transient_error(error) is True

    def test_401_not_transient(self):
        """401 authentication error should NOT be transient."""
        error = Exception("401 Unauthorized: Invalid API key")
        assert LLMBackend._is_transient_error(error) is False

    def test_400_not_transient(self):
        """400 bad request should NOT be transient."""
        error = Exception("400 Bad Request: Invalid parameter")
        assert LLMBackend._is_transient_error(error) is False

    def test_generic_error_not_transient(self):
        """Generic errors without status codes should NOT be transient."""
        error = Exception("Something went wrong with parsing")
        assert LLMBackend._is_transient_error(error) is False

    def test_response_attribute_status(self):
        """Error with response.status_code attribute should be detected."""
        error = Exception("API error")
        error.response = MagicMock(status_code=503)
        assert LLMBackend._is_transient_error(error) is True

    def test_resource_exhausted_status_code_string(self):
        """Gemini RESOURCE_EXHAUSTED status_code string should be transient."""
        error = Exception("quota exceeded")
        error.status_code = "RESOURCE_EXHAUSTED"
        assert LLMBackend._is_transient_error(error) is True

    def test_unavailable_status_code_string(self):
        """gRPC UNAVAILABLE status_code string should be transient."""
        error = Exception("service unavailable")
        error.status_code = "UNAVAILABLE"
        assert LLMBackend._is_transient_error(error) is True

    def test_resource_exhausted_in_message(self):
        """RESOURCE_EXHAUSTED in error message should be transient."""
        error = Exception("RESOURCE_EXHAUSTED: quota exceeded for model")
        assert LLMBackend._is_transient_error(error) is True

    def test_non_numeric_non_grpc_status_not_transient(self):
        """Non-numeric, non-gRPC status_code should not crash or be transient."""
        error = Exception("some error")
        error.status_code = "INVALID_ARGUMENT"
        assert LLMBackend._is_transient_error(error) is False

    def test_response_with_grpc_status_string(self):
        """Error with response.status_code as gRPC string should be detected."""
        error = Exception("API error")
        error.response = MagicMock(status_code="RESOURCE_EXHAUSTED")
        assert LLMBackend._is_transient_error(error) is True


class TestRetryTransient:
    """Test retry logic with exponential backoff."""

    def setup_method(self):
        self.backend = DummyBackend()

    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    def test_succeeds_after_transient_retry(self, mock_random, mock_sleep):
        """Should succeed after a transient error followed by success."""
        call_count = 0

        def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("503 UNAVAILABLE")
            return "success"

        with patch("storyforge.console.console"):
            result = self.backend._retry_transient(flaky_call, operation="test")

        assert result == "success"
        assert call_count == 2
        mock_sleep.assert_called_once_with(10.0)  # BASE_RETRY_DELAY * 2^0 + 0 jitter

    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    def test_exponential_backoff_delays(self, mock_random, mock_sleep):
        """Should use exponential backoff: 10s, 20s, 40s."""
        call_count = 0

        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("503 UNAVAILABLE")

        with patch("storyforge.console.console"):
            with pytest.raises(Exception, match="503 UNAVAILABLE"):
                self.backend._retry_transient(always_fails, operation="test")

        assert call_count == 4  # 1 initial + 3 retries
        delays = [c[0][0] for c in mock_sleep.call_args_list]
        assert delays == [10.0, 20.0, 40.0]

    @patch("storyforge.llm_backend.time.sleep")
    def test_no_retry_on_permanent_error(self, mock_sleep):
        """Should NOT retry on non-transient errors (e.g., 401)."""
        call_count = 0

        def auth_error():
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized: Invalid API key")

        with pytest.raises(Exception, match="401 Unauthorized"):
            self.backend._retry_transient(auth_error, operation="test")

        assert call_count == 1
        mock_sleep.assert_not_called()

    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    def test_retry_exhaustion_raises_last_error(self, mock_random, mock_sleep):
        """After max retries, should raise the last exception."""

        def always_503():
            raise Exception("503 UNAVAILABLE")

        with patch("storyforge.console.console"):
            with pytest.raises(Exception, match="503 UNAVAILABLE"):
                self.backend._retry_transient(always_503, operation="test")

        assert mock_sleep.call_count == 3  # 3 retries

    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    def test_succeeds_on_third_attempt(self, mock_random, mock_sleep):
        """Should succeed on the third attempt after two transient failures."""
        call_count = 0

        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("429 Too Many Requests")
            return "finally worked"

        with patch("storyforge.console.console"):
            result = self.backend._retry_transient(fails_twice, operation="test")

        assert result == "finally worked"
        assert call_count == 3
        assert mock_sleep.call_count == 2


class TestClassifyStoryError:
    """Test error classification for user-facing messages."""

    def test_503_classified_as_transient(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(f"{ERROR_STORY_SENTINEL}: 503 UNAVAILABLE")
        assert is_transient is True
        assert "temporarily unavailable" in msg.lower()
        assert "503" in msg

    def test_429_classified_as_transient(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(f"{ERROR_STORY_SENTINEL}: 429 Too Many Requests")
        assert is_transient is True
        assert "temporarily unavailable" in msg.lower()

    def test_high_demand_classified_as_transient(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(
            f"{ERROR_STORY_SENTINEL}: This model is currently experiencing high demand"
        )
        assert is_transient is True

    def test_401_classified_as_auth(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(f"{ERROR_STORY_SENTINEL}: 401 UNAUTHENTICATED")
        assert is_transient is False
        assert "api key" in msg.lower()

    def test_generic_error_classified(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(f"{ERROR_STORY_SENTINEL}: Connection timeout")
        assert is_transient is False
        assert "Connection timeout" in msg

    def test_bare_sentinel_classified(self):
        from storyforge.llm_backend import classify_story_error

        msg, is_transient = classify_story_error(ERROR_STORY_SENTINEL)
        assert is_transient is False
        assert "Story generation failed" in msg
