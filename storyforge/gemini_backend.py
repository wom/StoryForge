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

    def __init__(self) -> None:
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
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate image with model {model}: {e}")
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
