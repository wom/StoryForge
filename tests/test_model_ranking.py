"""Tests for version-aware model ranking."""

from storyforge.model_ranking import (
    extract_anthropic_info,
    extract_gemini_info,
    extract_openai_info,
    rank_models,
    score_model,
)


class TestGeminiExtraction:
    """Test version/tier extraction for Gemini models."""

    def test_versioned_pro(self):
        assert extract_gemini_info("gemini-3.1-pro") == (3.1, "pro")

    def test_versioned_flash(self):
        assert extract_gemini_info("gemini-2.5-flash") == (2.5, "flash")

    def test_versioned_flash_image(self):
        assert extract_gemini_info("gemini-2.5-flash-image") == (2.5, "flash")

    def test_versioned_flash_lite(self):
        assert extract_gemini_info("gemini-3.0-flash-lite") == (3.0, "flash-lite")

    def test_alias_pro_latest(self):
        # Cross-gen aliases get high synthetic version
        assert extract_gemini_info("gemini-pro-latest") == (99.0, "pro")

    def test_alias_flash_latest(self):
        assert extract_gemini_info("gemini-flash-latest") == (99.0, "flash")

    def test_with_models_prefix(self):
        assert extract_gemini_info("models/gemini-3.0-pro") == (3.0, "pro")

    def test_unknown_format(self):
        assert extract_gemini_info("some-random-model") is None

    def test_nano(self):
        assert extract_gemini_info("gemini-2.0-nano") == (2.0, "nano")


class TestOpenAIExtraction:
    """Test version/tier extraction for OpenAI models."""

    def test_gpt_versioned(self):
        assert extract_openai_info("gpt-5.4") == (5.4, "")

    def test_gpt_versioned_mini(self):
        assert extract_openai_info("gpt-5.4-mini") == (5.4, "mini")

    def test_gpt_versioned_nano(self):
        assert extract_openai_info("gpt-4.1-nano") == (4.1, "nano")

    def test_gpt_image(self):
        # gpt-image gets +10 version offset
        result = extract_openai_info("gpt-image-1.5")
        assert result is not None
        assert result[0] == 11.5  # 1.5 + 10.0

    def test_gpt_image_major_only(self):
        result = extract_openai_info("gpt-image-1")
        assert result is not None
        assert result[0] == 11.0  # 1.0 + 10.0

    def test_o_series(self):
        assert extract_openai_info("o3") == (3.0, "")

    def test_o_series_mini(self):
        assert extract_openai_info("o4-mini") == (4.0, "mini")

    def test_dalle(self):
        assert extract_openai_info("dall-e-3") == (3.0, "")

    def test_legacy_gpt(self):
        assert extract_openai_info("gpt-4") == (4.0, "")

    def test_unknown(self):
        assert extract_openai_info("whisper-1") is None


class TestAnthropicExtraction:
    """Test version/tier extraction for Anthropic models."""

    def test_new_format_opus(self):
        assert extract_anthropic_info("claude-opus-4-7") == (4.7, "opus")

    def test_new_format_sonnet(self):
        assert extract_anthropic_info("claude-sonnet-4-6") == (4.6, "sonnet")

    def test_new_format_haiku(self):
        assert extract_anthropic_info("claude-haiku-4-5") == (4.5, "haiku")

    def test_legacy_format(self):
        assert extract_anthropic_info("claude-3-5-sonnet-20241022") == (3.5, "sonnet")

    def test_legacy_opus(self):
        assert extract_anthropic_info("claude-3-0-opus") == (3.0, "opus")

    def test_unknown(self):
        assert extract_anthropic_info("not-a-claude-model") is None


class TestScoring:
    """Test composite scoring."""

    def test_version_dominates(self):
        # Higher version always wins regardless of tier
        assert score_model("gemini-3.0-flash", "gemini") > score_model("gemini-2.5-pro", "gemini")

    def test_tier_breaks_ties(self):
        # Same version: pro > flash
        assert score_model("gemini-3.0-pro", "gemini") > score_model("gemini-3.0-flash", "gemini")

    def test_openai_version_order(self):
        assert score_model("gpt-5.4", "openai") > score_model("gpt-5.2", "openai")

    def test_anthropic_tier_order(self):
        # Same gen: opus > sonnet > haiku
        opus = score_model("claude-opus-4-7", "anthropic")
        sonnet = score_model("claude-sonnet-4-6", "anthropic")
        haiku = score_model("claude-haiku-4-5", "anthropic")
        assert opus > sonnet > haiku

    def test_unknown_model_scores_zero(self):
        assert score_model("unknown-model", "gemini") == 0.0

    def test_cross_gen_alias_scores_highest(self):
        # Aliases auto-update, so they should score very high
        assert score_model("gemini-pro-latest", "gemini") > score_model("gemini-3.1-pro", "gemini")


class TestRanking:
    """Test the full rank_models function."""

    def test_gemini_text_ranking(self):
        models = [
            {"name": "models/gemini-2.5-pro", "supported_generation_methods": ["generateContent"]},
            {"name": "models/gemini-3.0-pro", "supported_generation_methods": ["generateContent"]},
            {"name": "models/gemini-3.0-flash", "supported_generation_methods": ["generateContent"]},
        ]
        best = rank_models(models, "gemini", "text")
        assert best == "gemini-3.0-pro"

    def test_gemini_skips_preview(self):
        models = [
            {"name": "models/gemini-3.1-pro-preview", "supported_generation_methods": ["generateContent"]},
            {"name": "models/gemini-3.0-pro", "supported_generation_methods": ["generateContent"]},
        ]
        best = rank_models(models, "gemini", "text")
        assert best == "gemini-3.0-pro"

    def test_gemini_skips_nano_for_text(self):
        models = [
            {"name": "gemini-3.0-nano"},
            {"name": "gemini-2.5-flash"},
        ]
        best = rank_models(models, "gemini", "text")
        assert best == "gemini-2.5-flash"

    def test_openai_text_ranking(self):
        models = [
            {"name": "gpt-4o"},
            {"name": "gpt-5.2"},
            {"name": "gpt-5.4"},
            {"name": "gpt-5.4-mini"},
        ]
        best = rank_models(models, "openai", "text")
        assert best == "gpt-5.4"

    def test_openai_image_ranking(self):
        models = [
            {"name": "dall-e-3"},
            {"name": "gpt-image-1"},
            {"name": "gpt-image-1.5"},
        ]
        best = rank_models(models, "openai", "image")
        assert best == "gpt-image-1.5"

    def test_anthropic_text_ranking(self):
        models = [
            {"name": "claude-sonnet-4-6"},
            {"name": "claude-opus-4-7"},
            {"name": "claude-haiku-4-5"},
        ]
        best = rank_models(models, "anthropic", "text")
        assert best == "claude-opus-4-7"

    def test_anthropic_skips_haiku_for_text(self):
        models = [
            {"name": "claude-haiku-4-5"},
        ]
        best = rank_models(models, "anthropic", "text")
        assert best is None

    def test_empty_list_returns_none(self):
        assert rank_models([], "openai", "text") is None

    def test_blocklist(self):
        models = [
            {"name": "gpt-5.4"},
            {"name": "gpt-5.2"},
        ]
        best = rank_models(models, "openai", "text", blocklist=["gpt-5.4"])
        assert best == "gpt-5.2"

    def test_all_filtered_returns_none(self):
        models = [
            {"name": "gemini-3.0-nano"},  # skipped for text
        ]
        best = rank_models(models, "gemini", "text")
        assert best is None

    def test_models_prefix_stripped(self):
        """Models with 'models/' prefix still rank correctly."""
        models = [
            {"name": "models/gemini-3.0-pro"},
        ]
        best = rank_models(models, "gemini", "text")
        assert best == "gemini-3.0-pro"

    def test_future_models_ranked_higher(self):
        """Demonstrates auto-preference for newer versions."""
        models = [
            {"name": "gpt-5.4"},
            {"name": "gpt-6.0"},  # hypothetical future model
        ]
        best = rank_models(models, "openai", "text")
        assert best == "gpt-6.0"
