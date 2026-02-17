import os
from unittest.mock import MagicMock, patch

import pytest

from storyforge.llm_backend import LLMBackend, get_backend
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
