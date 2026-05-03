"""
AnthropicBackend: Implementation of LLMBackend using Anthropic's Claude API.
Provides methods to generate stories and text-based content using Claude models.
Note: Claude does not support image generation, so image-related methods return None.
"""

import logging
import os
from typing import Any

import anthropic

from .llm_backend import ERROR_STORY_SENTINEL, LLMBackend
from .model_cache import ModelCache
from .model_discovery import find_anthropic_text_model, list_anthropic_models
from .prompt import Prompt

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """
    LLM backend implementation using Anthropic's Claude API.
    Requires ANTHROPIC_API_KEY environment variable to be set.

    Note: This backend only supports text generation. Image generation
    methods will return None since Claude cannot generate images.
    """

    name = "anthropic"

    # Claude model context window sizes
    MODEL_TOKEN_LIMITS: dict[str, int] = {
        "claude-opus-4": 200000,
        "claude-sonnet-4": 200000,
        "claude-haiku-4": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }
    DEFAULT_TEXT_INPUT_LIMIT = 200000

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the Anthropic client using the API key from environment variables.

        Args:
            config: Optional Config object for model configuration.

        Raises:
            RuntimeError: If ANTHROPIC_API_KEY is not set.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self._cache = ModelCache()

        # Get model from config or use discovery
        configured_model = ""
        if config is not None:
            configured_model = config.get_field_value("system", "anthropic_story_model") or ""

        if configured_model:
            self._story_model = configured_model
        else:
            self._story_model = self._discover_model()

        # Set text input limit using dynamic lookup with static fallback
        self._text_input_limit = self._get_model_limit(self._story_model)

    def _get_model_limit(self, model_name: str) -> int:
        """Get the token limit for a given model name.

        Checks cached model metadata for dynamic token limits first, then
        falls back to the static MODEL_TOKEN_LIMITS dict with prefix matching,
        and finally to DEFAULT_TEXT_INPUT_LIMIT.

        Args:
            model_name: The model name to look up.

        Returns:
            int: The token limit for the model.
        """
        # Check cached models for token limit metadata from the Anthropic API
        cached_models = self._cache.get("anthropic")
        if cached_models:
            for model in cached_models:
                model_id = model.get("id", "") or model.get("name", "")
                if model_id == model_name:
                    token_limit = model.get("input_token_limit") or model.get("context_window")
                    if token_limit and isinstance(token_limit, int):
                        return int(token_limit)
                    break

        # Exact match in static dict
        if model_name in self.MODEL_TOKEN_LIMITS:
            return self.MODEL_TOKEN_LIMITS[model_name]

        # Prefix match for versioned models (e.g., "claude-3-5-sonnet-20241022")
        for prefix, limit in self.MODEL_TOKEN_LIMITS.items():
            if model_name.startswith(prefix):
                return limit

        logger.warning(
            "Unknown Anthropic model '%s', using default limit of %d tokens",
            model_name,
            self.DEFAULT_TEXT_INPUT_LIMIT,
        )
        return self.DEFAULT_TEXT_INPUT_LIMIT

    def _discover_model(self) -> str:
        """Discover the best available Anthropic text model, using cache when possible."""
        cached = self._cache.get("anthropic")
        if cached is not None:
            return find_anthropic_text_model(cached)

        models = list_anthropic_models()
        if models:
            self._cache.set("anthropic", models)
        return find_anthropic_text_model(models)

    def _is_model_not_found_error(self, error: Exception) -> bool:
        """Check if an error indicates the model is no longer available."""
        if isinstance(error, anthropic.NotFoundError):
            return True
        error_str = str(error).lower()
        return any(
            phrase in error_str
            for phrase in (
                "model_not_found",
                "not_found_error",
                "does not exist",
                "decommissioned",
                "deprecated",
                "invalid_model",
            )
        )

    def _handle_model_not_found(self, error: Exception) -> str:
        """Handle a model-not-found error by invalidating cache and re-discovering."""
        old_model = self._story_model
        self._cache.invalidate("anthropic")
        new_model = self._discover_model()
        self._story_model = new_model
        logger.warning("Model '%s' is no longer available, falling back to '%s'", old_model, new_model)
        return new_model

    def get_model_info(self) -> dict[str, str | None]:
        """Return model information for the Anthropic backend.

        Returns:
            Dict with story_model set to the Claude model in use,
            image_model is None since Anthropic cannot generate images.
        """
        return {
            "story_model": self._story_model,
            "image_model": None,
        }

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        """Extract text content from an Anthropic API response."""
        if response.content:
            for content_block in response.content:
                if content_block.type == "text" and hasattr(content_block, "text"):
                    return str(content_block.text).strip()
        return None

    def generate_story(self, prompt: Prompt) -> str:
        """
        Generate a story based on the given Prompt object using Claude.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Use the Prompt's comprehensive prompt building
            story_prompt = prompt.story

            # Safety check: truncate if prompt exceeds model limits
            story_prompt = self._check_and_truncate_prompt(story_prompt)

            def _call() -> str:
                response = self.client.messages.create(
                    model=self._story_model,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": story_prompt}],
                )
                text = self._extract_text(response)
                if text:
                    return text
                return "[Error: No valid response from Claude]"

            return self._retry_transient(_call, operation="story generation")
        except Exception as e:
            if self._is_model_not_found_error(e):
                self._handle_model_not_found(e)
                try:
                    return self._retry_transient(_call, operation="story generation")
                except Exception as retry_err:
                    logger.warning("Story generation failed after model fallback: %s", retry_err)
                    return f"{ERROR_STORY_SENTINEL}: {retry_err}"
            logger.warning("Story generation failed: %s", e)
            return f"{ERROR_STORY_SENTINEL}: {e}"

    def generate_image(
        self,
        prompt: Prompt,
        reference_image_bytes: bytes | None = None,
        override_prompt: str | None = None,
    ) -> tuple[object | None, bytes | None]:
        """
        Claude does not support image generation, so this method returns None.

        Args:
            prompt (Prompt): A Prompt object containing image generation parameters.
            reference_image_bytes (Optional[bytes]): Reference image bytes (unused).
            override_prompt (Optional[str]): Override prompt text (unused).

        Returns:
            Tuple[None, None]: Always returns None since Claude cannot generate images.
        """
        logger.warning("Claude (Anthropic) does not support image generation. Use Gemini or OpenAI for images.")
        return None, None

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story using Claude's text generation capabilities.

        Args:
            prompt (Prompt): A Prompt object containing the original parameters.
            story (str): The generated story.

        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            # Use the Prompt's comprehensive image name prompt building
            name_prompt = prompt.image_name(story)

            response = self.client.messages.create(
                model=self._story_model,
                max_tokens=100,
                temperature=0.3,  # Lower temperature for more consistent naming
                messages=[{"role": "user", "content": name_prompt}],
            )

            # Extract name with proper null checking
            text = self._extract_text(response)
            if text:
                return self._sanitize_image_name(text)

            return "story_image"
        except Exception:
            logger.debug("Image name generation failed, using fallback", exc_info=True)
            return "story_image"

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """Break the story into detailed image prompts using Claude's text model.

        Uses the shared LLM-based approach: builds a standardized instruction,
        sends it to Claude, and parses numbered prompts from the response.
        Falls back to mechanical paragraph splitting on failure.

        Args:
            story: The generated story text.
            context: Additional context (may include character descriptions).
            num_prompts: Number of image prompts to generate.

        Returns:
            List of detailed image prompts, one per requested image.
        """
        try:
            image_prompt_request = self._build_image_prompt_request(story, context, num_prompts)

            response = self.client.messages.create(
                model=self._story_model,
                max_tokens=2000,
                temperature=0.5,
                messages=[{"role": "user", "content": image_prompt_request}],
            )

            text = self._extract_text(response)
            if text:
                parsed = self._parse_numbered_prompts(text, num_prompts)
                if parsed:
                    return parsed

            return self._generate_fallback_image_prompts(story, context, num_prompts)

        except Exception:
            logger.debug("Image prompt generation failed, using fallback", exc_info=True)
            return self._generate_fallback_image_prompts(story, context, num_prompts)
