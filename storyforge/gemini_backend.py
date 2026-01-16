"""Gemini backend for StoryForge.

This implementation uses dynamic model discovery to automatically select
the best available models for text and image generation. It caches the
model list on first use. Image model can be overridden via `GEMINI_IMAGE_MODEL`
env var, otherwise it auto-discovers the best available image generation model
(typically `gemini-2.5-flash-image`).
"""

import os
from io import BytesIO
from typing import Any, ClassVar

from google import genai
from google.genai import types
from PIL import Image

from .llm_backend import LLMBackend
from .model_discovery import find_image_generation_model, find_text_generation_model, list_gemini_models
from .prompt import Prompt


class GeminiBackend(LLMBackend):
    name = "gemini"
    _cached_models: ClassVar[list[dict[str, Any]] | None] = None
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
            config: Optional Config object (currently unused, for API consistency).
        """
        import logging

        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

        # Discover and cache models on initialization
        if GeminiBackend._cached_models is None:
            try:
                GeminiBackend._cached_models = list_gemini_models(api_key)
            except Exception:
                # If discovery fails, use None and fallback to defaults
                GeminiBackend._cached_models = []

        # Determine models to use
        if GeminiBackend._image_model is None:
            # Check environment variable override first
            env_model = os.environ.get("GEMINI_IMAGE_MODEL")
            if env_model:
                GeminiBackend._image_model = env_model
            else:
                GeminiBackend._image_model = find_image_generation_model(GeminiBackend._cached_models)

        if GeminiBackend._text_model is None:
            GeminiBackend._text_model = find_text_generation_model(GeminiBackend._cached_models)

        # Extract and cache token limits for the selected models
        self._image_input_limit = self._get_model_input_limit(
            GeminiBackend._image_model, self.DEFAULT_IMAGE_INPUT_LIMIT, "image"
        )
        self._text_input_limit = self._get_model_input_limit(
            GeminiBackend._text_model, self.DEFAULT_TEXT_INPUT_LIMIT, "text"
        )

    def _get_model_input_limit(self, model_name: str | None, default_limit: int, model_type: str) -> int:
        """Get input token limit for a model with fallback.

        Args:
            model_name: Name of the model to look up.
            default_limit: Fallback limit if not found.
            model_type: Type of model ('image' or 'text') for logging.

        Returns:
            int: Input token limit for the model.
        """
        if not model_name or not GeminiBackend._cached_models:
            self.logger.warning(
                f"⚠️  No model information available for {model_type} model, "
                f"using fallback limit of {default_limit} tokens"
            )
            return default_limit

        # Search for the model in cached models
        for model_info in GeminiBackend._cached_models:
            model_full_name = model_info.get("name", "")
            if model_name in model_full_name:
                limit = model_info.get("input_token_limit")
                if limit is not None:
                    self.logger.debug(f"Found input limit for {model_name}: {limit} tokens")
                    return int(limit)
                else:
                    self.logger.warning(
                        f"⚠️  No input_token_limit found for {model_name}, "
                        f"using fallback limit of {default_limit} tokens"
                    )
                    return default_limit

        # Model not found in cache
        self.logger.warning(
            f"⚠️  Model {model_name} not found in cache, using fallback limit of {default_limit} tokens"
        )
        return default_limit

    def _compress_prompt(self, prompt_text: str, target_tokens: int) -> str:
        """Compress a prompt to fit within token limits.

        Args:
            prompt_text: The original prompt text.
            target_tokens: Target token count after compression.

        Returns:
            str: Compressed prompt text.
        """
        original_tokens = self.estimate_token_count(prompt_text)
        self.logger.info(f"Compressing prompt: ~{original_tokens} tokens → target ~{target_tokens} tokens")

        compression_prompt = (
            f"Please create a concise, detailed prompt (maximum {target_tokens * 4} characters) "
            f"from this longer description. Preserve all key visual details, character descriptions, "
            f"and important story elements. Focus on concrete details like character appearance, "
            f"setting, and actions.\n\nOriginal prompt:\n{prompt_text}"
        )

        try:
            model = self._text_model or "gemini-2.5-flash"
            response = self.client.models.generate_content(model=model, contents=compression_prompt)
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", None) or []
                    if parts and getattr(parts[0], "text", None):
                        compressed = str(parts[0].text).strip()
                        compressed_tokens = self.estimate_token_count(compressed)
                        self.logger.info(f"Compression successful: ~{compressed_tokens} tokens")
                        return compressed
        except Exception as e:
            self.logger.error(f"Prompt compression failed: {e}. Using truncation fallback.")

        # Fallback: simple truncation
        target_chars = target_tokens * 4
        truncated = prompt_text[:target_chars]
        self.logger.warning(f"Using truncated prompt ({len(truncated)} chars)")
        return truncated

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        prompts: list[str] = []
        for i in range(num_prompts):
            base = paragraphs[i] if i < len(paragraphs) else story
            prompt = f"Create a detailed, child-friendly illustration for this part of the story: {base}"
            if context:
                prompt += f"\nContext: {context}"
            prompts.append(prompt)
        return prompts

    def generate_story(self, prompt: Prompt) -> str:
        try:
            contents = prompt.story

            # Check if prompt needs compression
            prompt_tokens = self.estimate_token_count(contents)
            trigger_limit = int(self._text_input_limit * self.COMPRESSION_TRIGGER_RATIO)

            if prompt_tokens > trigger_limit:
                target_tokens = int(self._text_input_limit * self.COMPRESSION_TARGET_RATIO)
                contents = self._compress_prompt(contents, target_tokens)

            model = self._text_model or "gemini-2.5-pro"
            response = self.client.models.generate_content(model=model, contents=contents)
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", None) or []
                    if parts and getattr(parts[0], "text", None):
                        text: str = parts[0].text
                        return text.strip()
            return "[Error: No valid response from Gemini]"
        except Exception:
            return "[Error generating story]"

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
            pass

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
                                continue
        except Exception:
            pass

        return None, None

    def generate_image(
        self, prompt: Prompt, reference_image_bytes: bytes | None = None
    ) -> tuple[Image.Image | None, bytes | None]:
        text_prompt = prompt.image(1)[0]

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

        model = self._image_model or "gemini-2.5-flash-image"
        try:
            response = self.client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            self.logger.error(f"Failed to generate image with model {model}: {e}")
            return None, None

        return self._extract_image_from_response(response)

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        try:
            contents = prompt.image_name(story)
            model = self._text_model or "gemini-2.5-flash"
            response = self.client.models.generate_content(model=model, contents=contents)
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", None) or []
                    if parts and getattr(parts[0], "text", None):
                        text: str = parts[0].text
                        name = text.strip()
                        return name.split(".")[0]
            return "story_image"
        except Exception:
            return "story_image"
