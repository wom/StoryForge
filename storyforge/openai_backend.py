"""
OpenAIBackend: Implementation of LLMBackend using OpenAI APIs.
Provides methods to generate stories, images, and image filenames using OpenAI models.
"""

import logging
import os
from io import BytesIO
from typing import Any, Literal

import openai
from PIL import Image

from .llm_backend import ERROR_STORY_SENTINEL, LLMBackend
from .model_cache import ModelCache
from .model_discovery import find_openai_image_model, find_openai_text_model, list_openai_models
from .prompt import Prompt

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """
    LLM backend implementation using OpenAI APIs.
    Requires OPENAI_API_KEY environment variable to be set.

    Uses configurable models for story and image generation:
    - Story model: Defaults to gpt-5.4 (configurable via system.openai_story_model)
    - Image model: Defaults to gpt-image-1.5 (configurable via system.openai_image_model)
    """

    name = "openai"

    # Known context window sizes for OpenAI models
    MODEL_TOKEN_LIMITS: dict[str, int] = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4.1": 1047576,
        "gpt-4.1-mini": 1047576,
        "gpt-4.1-nano": 1047576,
        "gpt-5": 128000,
        "gpt-5.2": 128000,
        "gpt-5.3": 128000,
        "gpt-5.4": 128000,
        "gpt-5.5": 128000,
        "gpt-image-1": 128000,
        "gpt-image-1.5": 128000,
        "o1": 200000,
        "o1-mini": 128000,
        "o3": 200000,
        "o3-mini": 200000,
        "o4-mini": 200000,
    }
    DEFAULT_TEXT_INPUT_LIMIT = 128000

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the OpenAI client using the API key from environment variables.

        Model selection uses lazy discovery: if a model is explicitly configured,
        it is used directly (no API hit). If no model is configured, discovery
        checks the cache first, then calls the OpenAI models API if needed.

        Args:
            config: Optional Config object for retrieving model settings.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)
        self._cache = ModelCache()

        # Get model settings from config (empty/None means "use discovery")
        configured_story: str | None = None
        configured_image: str | None = None
        if config is not None:
            configured_story = config.get_field_value("system", "openai_story_model") or None
            configured_image = config.get_field_value("system", "openai_image_model") or None

        if configured_story and configured_image:
            # Both explicitly configured — trust user, no API hit
            self.story_model = configured_story
            self.image_model = configured_image
        elif configured_story:
            # Only story model configured; discover image model
            self.story_model = configured_story
            _, self.image_model = self._discover_and_select_models()
        elif configured_image:
            # Only image model configured; discover story model
            self.image_model = configured_image
            self.story_model, _ = self._discover_and_select_models()
        else:
            # Nothing configured — discover both
            self.story_model, self.image_model = self._discover_and_select_models()

        # Set text input limit based on model
        self._text_input_limit = self._get_model_limit(self.story_model)

    def _get_model_limit(self, model_name: str) -> int:
        """Get the token limit for a given model name.

        Checks the cached model list for token metadata first, then falls back
        to the static MODEL_TOKEN_LIMITS dict with exact and prefix matching.
        Falls back to DEFAULT_TEXT_INPUT_LIMIT if no match is found.

        Note: The OpenAI models API (model.list()) does not currently return
        context window sizes in model metadata. Dynamic limits would require a
        separate endpoint or external data source. The static dict remains the
        primary source for now.

        Args:
            model_name: The model name to look up.

        Returns:
            int: The token limit for the model.
        """
        # Check cached models for token limit metadata (future-proofing)
        cached_models = self._cache.get("openai")
        if cached_models:
            for model in cached_models:
                model_id = model.get("id", "") or model.get("name", "")
                if model_id == model_name:
                    # Check for context_window or similar token limit fields
                    token_limit = model.get("context_window") or model.get("input_token_limit")
                    if token_limit and isinstance(token_limit, int):
                        return int(token_limit)
                    break

        # Exact match in static dict
        if model_name in self.MODEL_TOKEN_LIMITS:
            return self.MODEL_TOKEN_LIMITS[model_name]

        # Prefix match for versioned models (e.g., "gpt-4o-2024-05-13")
        for prefix, limit in self.MODEL_TOKEN_LIMITS.items():
            if model_name.startswith(prefix):
                return limit

        logger.warning(
            "Unknown OpenAI model '%s', using default limit of %d tokens",
            model_name,
            self.DEFAULT_TEXT_INPUT_LIMIT,
        )
        return self.DEFAULT_TEXT_INPUT_LIMIT

    def _discover_and_select_models(self) -> tuple[str, str]:
        """Discover available models via cache or API and select best text/image models.

        Returns:
            Tuple of (text_model, image_model) names.
        """
        models = self._cache.get("openai")
        if models is None:
            models = list_openai_models()
            if models:
                self._cache.set("openai", models)
        text_model = find_openai_text_model(models) if models else "gpt-5.2"
        image_model = find_openai_image_model(models) if models else "gpt-image-1.5"
        return text_model, image_model

    def _is_model_not_found_error(self, error: Exception) -> bool:
        """Check if an error indicates the model is not found or deprecated."""
        # Check for openai NotFoundError (HTTP 404)
        if hasattr(error, "status_code") and getattr(error, "status_code", None) == 404:
            return True
        error_str = str(error).lower()
        indicators = ("model_not_found", "does not exist", "decommissioned", "deprecated", "invalid_model")
        return any(ind in error_str for ind in indicators)

    def _handle_model_not_found(self, error: Exception, model_type: str) -> str:
        """Handle a model-not-found error by invalidating cache and re-discovering.

        Args:
            error: The exception that was raised.
            model_type: Either "text" or "image".

        Returns:
            The new model name to use.
        """
        old_model = self.story_model if model_type == "text" else self.image_model
        self._cache.invalidate("openai")
        text_model, image_model = self._discover_and_select_models()
        new_model = text_model if model_type == "text" else image_model
        logger.warning("Model '%s' is no longer available, falling back to '%s'", old_model, new_model)
        return new_model

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        """Extract text content from an OpenAI API response."""
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return str(response.choices[0].message.content).strip()
        return None

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """Break the story into detailed image prompts using OpenAI's text model.

        Uses the shared LLM-based approach: builds a standardized instruction,
        sends it to the configured story model, and parses numbered prompts from
        the response. Falls back to mechanical paragraph splitting on failure.

        Args:
            story: The generated story text.
            context: Additional context (may include character descriptions).
            num_prompts: Number of image prompts to generate.

        Returns:
            List of detailed image prompts, one per requested image.
        """
        try:
            image_prompt_request = self._build_image_prompt_request(story, context, num_prompts)

            response = self.client.chat.completions.create(
                model=self.story_model,
                messages=[{"role": "user", "content": image_prompt_request}],
                temperature=0.5,
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

    def generate_story(self, prompt: Prompt) -> str:
        """
        Generate a story based on the given Prompt object using OpenAI LLM.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Use the Prompt's comprehensive prompt building
            contents = prompt.story

            # Safety check: truncate if prompt exceeds model limits
            contents = self._check_and_truncate_prompt(contents)

            def _call() -> str:
                response = self.client.chat.completions.create(
                    model=self.story_model, messages=[{"role": "user", "content": contents}], temperature=1
                )
                text = self._extract_text(response)
                if text:
                    return text
                return "[Error: No valid response from OpenAI]"

            return self._retry_transient(_call, operation="story generation")
        except Exception as e:
            if self._is_model_not_found_error(e):
                self.story_model = self._handle_model_not_found(e, "text")
                self._text_input_limit = self._get_model_limit(self.story_model)
                try:
                    contents = self._check_and_truncate_prompt(prompt.story)

                    def _retry_call() -> str:
                        response = self.client.chat.completions.create(
                            model=self.story_model,
                            messages=[{"role": "user", "content": contents}],
                            temperature=1,
                        )
                        text = self._extract_text(response)
                        if text:
                            return text
                        return "[Error: No valid response from OpenAI]"

                    return self._retry_transient(_retry_call, operation="story generation")
                except Exception as retry_e:
                    logger.warning("Story generation failed after fallback: %s", retry_e)
                    return f"{ERROR_STORY_SENTINEL}: {retry_e}"
            logger.warning("Story generation failed: %s", e)
            return f"{ERROR_STORY_SENTINEL}: {e}"

    def generate_image(
        self,
        prompt: Prompt,
        reference_image_bytes: bytes | None = None,
        override_prompt: str | None = None,
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given Prompt object using OpenAI
        DALL-E model, optionally using a reference image for consistency.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive image
                generation parameters including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images (currently not used by DALL-E).
            override_prompt (Optional[str]): If provided, use this text prompt
                instead of building one from the Prompt object.

        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        try:
            # Use override prompt if provided, otherwise build from Prompt object
            text_prompt = override_prompt if override_prompt else prompt.image(1)[0]

            # Note: DALL-E doesn't support reference images like Gemini does
            # We'll include a style consistency note if reference_image_bytes is provided
            if reference_image_bytes:
                text_prompt = (
                    "Create the next illustration in a consistent visual style "
                    "with the same characters and art style as previous images. "
                    f"{text_prompt}"
                )

            # DALL-E 3 has a 4000 character limit for prompts
            # If the prompt is too long, use OpenAI to create a concise version
            if len(text_prompt) > 4000:
                compression_prompt = (
                    "Please create a concise, detailed image generation prompt (maximum 3500 characters) "
                    "from this longer description. Keep all the important visual details, characters, "
                    "setting, mood, and style information:\n\n"
                    f"{text_prompt}"
                )

                compression_response = self.client.chat.completions.create(
                    model=self.story_model,
                    messages=[{"role": "user", "content": compression_prompt}],
                )

                compressed = self._extract_text(compression_response)
                if compressed:
                    text_prompt = compressed
                else:
                    # Fallback to simple truncation if compression fails
                    text_prompt = text_prompt[:3900] + "..."

            # Generate image using configured model (gpt-image-1.5 or dall-e-3)
            # Use quality="auto" for gpt-image models, "standard" for dall-e models
            quality: Literal["auto", "standard"] = "auto" if "gpt-image" in self.image_model else "standard"
            image_model = self.image_model

            def _call() -> tuple[object | None, bytes | None]:
                response = self.client.images.generate(
                    prompt=text_prompt,
                    model=image_model,
                    size="1024x1024",
                    quality=quality,
                    n=1,
                )

                if response.data and len(response.data) > 0:
                    item = response.data[0]
                    if item.url:
                        import requests

                        img_response = requests.get(item.url, timeout=30)
                        if img_response.status_code == 200:
                            image_bytes = img_response.content
                            image = Image.open(BytesIO(image_bytes))
                            return image, image_bytes
                    elif getattr(item, "b64_json", None):
                        import base64

                        b64_data: str = item.b64_json  # type: ignore[assignment]
                        image_bytes = base64.b64decode(b64_data)
                        image = Image.open(BytesIO(image_bytes))
                        return image, image_bytes

                return None, None

            return self._retry_transient(_call, operation="image generation")

        except Exception as e:
            if self._is_model_not_found_error(e):
                self.image_model = self._handle_model_not_found(e, "image")
                try:
                    quality_retry: Literal["auto", "standard"] = (
                        "auto" if "gpt-image" in self.image_model else "standard"
                    )
                    retry_model = self.image_model

                    def _retry_call() -> tuple[object | None, bytes | None]:
                        response = self.client.images.generate(
                            prompt=text_prompt,
                            model=retry_model,
                            size="1024x1024",
                            quality=quality_retry,
                            n=1,
                        )
                        if response.data and len(response.data) > 0:
                            item = response.data[0]
                            if item.url:
                                import requests

                                img_response = requests.get(item.url, timeout=30)
                                if img_response.status_code == 200:
                                    image_bytes = img_response.content
                                    image = Image.open(BytesIO(image_bytes))
                                    return image, image_bytes
                            elif getattr(item, "b64_json", None):
                                import base64

                                b64_data: str = item.b64_json  # type: ignore[assignment]
                                image_bytes = base64.b64decode(b64_data)
                                image = Image.open(BytesIO(image_bytes))
                                return image, image_bytes
                        return None, None

                    return self._retry_transient(_retry_call, operation="image generation")
                except Exception as retry_e:
                    logger.error("Failed to generate image after fallback: %s", retry_e)
                    return None, None
            logger.error("Failed to generate image with DALL-E: %s", e)
            return None, None

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story.

        Args:
            prompt (Prompt): A Prompt object containing the original parameters.
            story (str): The generated story.

        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            # Use the Prompt's comprehensive image name prompt building
            contents = prompt.image_name(story)

            response = self.client.chat.completions.create(
                model=self.story_model, messages=[{"role": "user", "content": contents}], temperature=1
            )

            # Extract name with proper null checking
            text = self._extract_text(response)
            if text:
                return self._sanitize_image_name(text)
            return "story_image"
        except Exception as e:
            if self._is_model_not_found_error(e):
                self.story_model = self._handle_model_not_found(e, "text")
                self._text_input_limit = self._get_model_limit(self.story_model)
                try:
                    contents = prompt.image_name(story)
                    response = self.client.chat.completions.create(
                        model=self.story_model,
                        messages=[{"role": "user", "content": contents}],
                        temperature=1,
                    )
                    text = self._extract_text(response)
                    if text:
                        return self._sanitize_image_name(text)
                    return "story_image"
                except Exception:
                    logger.debug("Image name generation failed after fallback", exc_info=True)
                    return "story_image"
            logger.debug("Image name generation failed, using fallback", exc_info=True)
            return "story_image"
