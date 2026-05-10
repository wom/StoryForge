"""Gemini backend for StoryForge.

This implementation uses dynamic model discovery to automatically select
the best available models for text and image generation. It caches the
model list on first use. Image model can be overridden via `GEMINI_IMAGE_MODEL`
env var, otherwise it auto-discovers the best available image generation model
(typically `gemini-flash-latest`).
"""

import logging
import os
from io import BytesIO
from typing import Any, ClassVar

from google import genai
from google.genai import types
from PIL import Image

from .llm_backend import ERROR_STORY_SENTINEL, LLMBackend
from .model_cache import ModelCache
from .model_discovery import find_image_generation_model, find_text_generation_model, list_gemini_models
from .prompt import Prompt

logger = logging.getLogger(__name__)


class GeminiBackend(LLMBackend):
    name = "gemini"
    _image_model: ClassVar[str | None] = None
    _text_model: ClassVar[str | None] = None

    # Token limit configuration
    DEFAULT_IMAGE_INPUT_LIMIT = 30000  # Conservative fallback for image models
    DEFAULT_TEXT_INPUT_LIMIT = 1000000  # Conservative fallback for text models
    COMPRESSION_TRIGGER_RATIO = 0.80  # Compress when prompt exceeds 80% of limit
    COMPRESSION_TARGET_RATIO = 0.875  # Compress to 87.5% of limit (matches OpenAI)

    def __init__(self, config: Any = None) -> None:
        """Initialize Gemini backend.

        Args:
            config: Optional Config object for model configuration.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)
        self._cache = ModelCache()

        # Read configured models from config
        configured_text = ""
        configured_image = ""
        if config is not None:
            configured_text = config.get_field_value("system", "gemini_story_model") or ""
            configured_image = config.get_field_value("system", "gemini_image_model") or ""

        # Env var override for image model (highest priority)
        env_image = os.environ.get("GEMINI_IMAGE_MODEL")

        if GeminiBackend._text_model is None or GeminiBackend._image_model is None:
            if configured_text and (configured_image or env_image):
                # Both models explicitly configured, no need for discovery
                GeminiBackend._text_model = GeminiBackend._text_model or configured_text
                GeminiBackend._image_model = GeminiBackend._image_model or (env_image or configured_image)
            else:
                # Need discovery for at least one model
                models = self._get_models(api_key)
                if GeminiBackend._text_model is None:
                    GeminiBackend._text_model = configured_text or find_text_generation_model(models)
                if GeminiBackend._image_model is None:
                    GeminiBackend._image_model = env_image or configured_image or find_image_generation_model(models)

        # Extract and cache token limits for the selected models
        self._image_input_limit = self._get_model_input_limit(
            GeminiBackend._image_model, self.DEFAULT_IMAGE_INPUT_LIMIT, "image"
        )
        self._text_input_limit = self._get_model_input_limit(
            GeminiBackend._text_model, self.DEFAULT_TEXT_INPUT_LIMIT, "text"
        )

    def _get_models(self, api_key: str) -> list[dict[str, Any]]:
        """Get model list from disk cache or API discovery."""
        cached = self._cache.get("gemini")
        if cached is not None:
            return cached

        try:
            models = list_gemini_models(api_key)
            self._cache.set("gemini", models)
            return models
        except Exception:
            logger.warning("Model discovery failed, using defaults", exc_info=True)
            return []

    def _is_model_not_found_error(self, error: Exception) -> bool:
        """Check if an exception indicates a model-not-found error."""
        error_str = str(error).lower()
        return "not found" in error_str or "404" in error_str

    def _handle_model_not_found(self, model_type: str) -> str:
        """Invalidate cache, re-discover models, and return updated model name.

        Args:
            model_type: Either "text" or "image".

        Returns:
            The newly discovered model name.
        """
        logger.warning("Model not found for %s, invalidating cache and re-discovering", model_type)
        self._cache.invalidate("gemini")

        api_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            models = list_gemini_models(api_key)
            self._cache.set("gemini", models)
        except Exception:
            logger.warning("Re-discovery failed, using defaults", exc_info=True)
            models = []

        if model_type == "text":
            GeminiBackend._text_model = find_text_generation_model(models)
            return GeminiBackend._text_model or "gemini-flash-latest"
        else:
            GeminiBackend._image_model = find_image_generation_model(models)
            return GeminiBackend._image_model or "gemini-flash-latest"

    def _get_model_input_limit(self, model_name: str | None, default_limit: int, model_type: str) -> int:
        """Get input token limit for a model with fallback.

        Args:
            model_name: Name of the model to look up.
            default_limit: Fallback limit if not found.
            model_type: Type of model ('image' or 'text') for logging.

        Returns:
            int: Input token limit for the model.
        """
        if not model_name:
            logger.warning(
                f"⚠️  No model information available for {model_type} model, "
                f"using fallback limit of {default_limit} tokens"
            )
            return default_limit

        models = self._cache.get("gemini")
        if not models:
            logger.warning(
                f"⚠️  No model information available for {model_type} model, "
                f"using fallback limit of {default_limit} tokens"
            )
            return default_limit

        # Search for the model in cached models
        for model_info in models:
            model_full_name = model_info.get("name", "")
            if model_name in model_full_name:
                limit = model_info.get("input_token_limit")
                if limit is not None:
                    logger.debug(f"Found input limit for {model_name}: {limit} tokens")
                    return int(limit)
                else:
                    logger.warning(
                        f"⚠️  No input_token_limit found for {model_name}, "
                        f"using fallback limit of {default_limit} tokens"
                    )
                    return default_limit

        # Model not found in cache
        logger.warning(f"⚠️  Model {model_name} not found in cache, using fallback limit of {default_limit} tokens")
        return default_limit

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        """Extract text content from a Gemini API response."""
        candidates = getattr(response, "candidates", None)
        if candidates:
            content = getattr(candidates[0], "content", None)
            if content:
                parts = getattr(content, "parts", None) or []
                if parts and getattr(parts[0], "text", None):
                    return str(parts[0].text).strip()
        return None

    def _compress_prompt(self, prompt_text: str, target_tokens: int) -> str:
        """Compress a prompt to fit within token limits.

        Args:
            prompt_text: The original prompt text.
            target_tokens: Target token count after compression.

        Returns:
            str: Compressed prompt text.
        """
        original_tokens = self.estimate_token_count(prompt_text)
        logger.info(f"Compressing prompt: ~{original_tokens} tokens → target ~{target_tokens} tokens")

        compression_prompt = (
            f"Please create a concise, detailed prompt (maximum {target_tokens * 4} characters) "
            f"from this longer description. Preserve all key visual details, character descriptions, "
            f"and important story elements. Focus on concrete details like character appearance, "
            f"setting, and actions.\n\nOriginal prompt:\n{prompt_text}"
        )

        try:
            model = self._text_model or "gemini-flash-latest"
            response = self.client.models.generate_content(model=model, contents=compression_prompt)
            compressed = self._extract_text(response)
            if compressed:
                compressed_tokens = self.estimate_token_count(compressed)
                logger.info(f"Compression successful: ~{compressed_tokens} tokens")
                return compressed
        except Exception as e:
            logger.error(f"Prompt compression failed: {e}. Using truncation fallback.")

        # Fallback: simple truncation
        target_chars = target_tokens * 4
        truncated = prompt_text[:target_chars]
        logger.warning(f"Using truncated prompt ({len(truncated)} chars)")
        return truncated

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """Break the story into detailed image prompts using Gemini's text model.

        Uses the shared LLM-based approach: builds a standardized instruction,
        sends it to the Gemini text model, and parses numbered prompts from the
        response. Falls back to mechanical paragraph splitting on failure.

        Args:
            story: The generated story text.
            context: Additional context (may include character descriptions).
            num_prompts: Number of image prompts to generate.

        Returns:
            List of detailed image prompts, one per requested image.
        """
        try:
            image_prompt_request = self._build_image_prompt_request(story, context, num_prompts)
            model = self._text_model or "gemini-flash-latest"
            response = self.client.models.generate_content(model=model, contents=image_prompt_request)
            text = self._extract_text(response)
            if text:
                parsed = self._parse_numbered_prompts(text, num_prompts)
                if parsed:
                    return parsed
            return self._generate_fallback_image_prompts(story, context, num_prompts)
        except Exception:
            logger.debug("Image prompt generation failed, using fallback", exc_info=True)
            return self._generate_fallback_image_prompts(story, context, num_prompts)

    def generate_video_prompt(self, story: str, context: str, num_scenes: int) -> list[str]:
        """Break the story into scene-by-scene video prompts using Gemini's text model.

        Args:
            story: The generated story text.
            context: Additional context (may include character descriptions).
            num_scenes: Number of scene prompts to generate.

        Returns:
            List of detailed video scene prompts.
        """
        try:
            video_prompt_request = self._build_video_prompt_request(story, context, num_scenes)
            model = self._text_model or "gemini-flash-latest"
            response = self.client.models.generate_content(model=model, contents=video_prompt_request)
            text = self._extract_text(response)
            if text:
                parsed = self._parse_numbered_prompts(text, num_scenes)
                if parsed:
                    return parsed
            return self._generate_fallback_video_prompts(story, context, num_scenes)
        except Exception:
            logger.debug("Video prompt generation failed, using fallback", exc_info=True)
            return self._generate_fallback_video_prompts(story, context, num_scenes)

    def generate_story(self, prompt: Prompt) -> str:
        try:
            contents = prompt.story

            # Check if prompt needs compression
            prompt_tokens = self.estimate_token_count(contents)
            trigger_limit = int(self._text_input_limit * self.COMPRESSION_TRIGGER_RATIO)

            if prompt_tokens > trigger_limit:
                target_tokens = int(self._text_input_limit * self.COMPRESSION_TARGET_RATIO)
                contents = self._compress_prompt(contents, target_tokens)

            model = self._text_model or "gemini-pro-latest"

            def _call() -> str:
                response = self.client.models.generate_content(model=model, contents=contents)
                text = self._extract_text(response)
                if text:
                    return text
                return "[Error: No valid response from Gemini]"

            return self._retry_transient(_call, operation="story generation")
        except Exception as e:
            logger.warning("Story generation failed: %s", e)
            return f"{ERROR_STORY_SENTINEL}: {e}"

    def _extract_image_from_response(self, resp: Any) -> tuple[Image.Image | None, bytes | None]:
        if not resp:
            return None, None

        # Common newer shape: resp.generated_images -> items with .image.image_bytes
        try:
            gen_images = getattr(resp, "generated_images", None)
            if gen_images:
                for gi in gen_images:
                    imgobj = getattr(gi, "image", None)
                    if imgobj and getattr(imgobj, "image_bytes", None):
                        data = imgobj.image_bytes
                        return Image.open(BytesIO(data)), data
        except Exception:
            logger.debug("Could not extract image from generated_images", exc_info=True)

        # Fallback: candidates[].content.parts[].inline_data.data (used in tests)
        try:
            candidates = getattr(resp, "candidates", None)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            data = inline.data
                            try:
                                return Image.open(BytesIO(data)), data
                            except Exception:
                                logger.debug("Could not open image from inline_data", exc_info=True)
                                continue
        except Exception:
            logger.debug("Could not extract image from candidates", exc_info=True)

        # Log diagnostic info when no image was found
        logger.warning(
            "No image data found in API response. generated_images=%s, candidates=%s",
            bool(getattr(resp, "generated_images", None)),
            bool(getattr(resp, "candidates", None)),
        )
        # Log any text content returned instead of an image
        try:
            candidates = getattr(resp, "candidates", None)
            if candidates and len(candidates) > 0:
                parts = getattr(candidates[0].content, "parts", None) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        logger.warning("API returned text instead of image: %s", text[:200])
        except Exception:
            pass

        return None, None

    def generate_image(
        self,
        prompt: Prompt,
        reference_image_bytes: bytes | None = None,
        override_prompt: str | None = None,
    ) -> tuple[Image.Image | None, bytes | None]:
        text_prompt = override_prompt if override_prompt else prompt.image(1)[0]

        # Check if prompt needs compression
        prompt_tokens = self.estimate_token_count(text_prompt)
        trigger_limit = int(self._image_input_limit * self.COMPRESSION_TRIGGER_RATIO)

        if prompt_tokens > trigger_limit:
            target_tokens = int(self._image_input_limit * self.COMPRESSION_TARGET_RATIO)
            text_prompt = self._compress_prompt(text_prompt, target_tokens)

        contents: str | list[Any]
        if reference_image_bytes:
            contents = [
                types.Part(inline_data=types.Blob(mime_type="image/png", data=reference_image_bytes)),
                (
                    "Create the next illustration in the same visual style, "
                    "with consistent characters and art style as the reference image. "
                    f"{text_prompt}"
                ),
            ]
        else:
            contents = text_prompt

        model = self._image_model or "gemini-flash-latest"
        try:

            def _call() -> Any:
                return self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

            response = self._retry_transient(_call, operation="image generation")
        except Exception as e:
            logger.error(f"Failed to generate image with model {model}: {e}")
            return None, None

        return self._extract_image_from_response(response)

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        try:
            contents = prompt.image_name(story)
            model = self._text_model or "gemini-flash-latest"
            response = self.client.models.generate_content(model=model, contents=contents)
            text = self._extract_text(response)
            if text:
                return self._sanitize_image_name(text)
            return "story_image"
        except Exception:
            logger.debug("Image name generation failed, using fallback", exc_info=True)
            return "story_image"
