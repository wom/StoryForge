"""
OpenAIBackend: Implementation of LLMBackend using OpenAI APIs.
Provides methods to generate stories, images, and image filenames using OpenAI models.
"""

# TODO: Implement dynamic model discovery for OpenAI (similar to Gemini backend)
# TODO: Add support for transparent backgrounds (background="transparent" parameter)
# TODO: Add support for high input fidelity (input_fidelity="high" parameter)
# TODO: Add support for multi-turn image editing via Responses API

import os
from io import BytesIO
from typing import Any, Literal

import openai
from PIL import Image

from .llm_backend import LLMBackend
from .prompt import Prompt


class OpenAIBackend(LLMBackend):
    """
    LLM backend implementation using OpenAI APIs.
    Requires OPENAI_API_KEY environment variable to be set.

    Uses configurable models for story and image generation:
    - Story model: Defaults to gpt-5.2 (configurable via system.openai_story_model)
    - Image model: Defaults to gpt-image-1.5 (configurable via system.openai_image_model)
    """

    name = "openai"

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the OpenAI client using the API key from environment variables.

        Args:
            config: Optional Config object for retrieving model settings.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            pass

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)

        # Get model settings from config or use defaults
        if config is not None:
            self.story_model = config.get_field_value("system", "openai_story_model")
            self.image_model = config.get_field_value("system", "openai_image_model")
        else:
            # Fallback to latest models if no config provided
            self.story_model = "gpt-5.2"
            self.image_model = "gpt-image-1.5"

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Break the given story into detailed image prompts using OpenAI's understanding.
        Each prompt will be incredibly detailed for use with DALL-E image generation.

        Args:
            story (str): The generated story to break into image prompts.
            context (str): Additional context for the story.
            num_prompts (int): The number of image prompts to return.

        Returns:
            list[str]: A list of detailed image prompts describing scenes from the story.
        """
        try:
            # Create a comprehensive prompt for GPT to generate image descriptions
            image_prompt_request = (
                f"Please break this story into {num_prompts} detailed, progressive image prompts "
                f"that would be perfect for generating illustrations with DALL-E. Each prompt should be "
                f"incredibly detailed, focusing on visual elements like character appearance, "
                f"setting details, colors, lighting, and mood.\n\n"
                f"Story: {story}\n\n"
            )

            if context:
                image_prompt_request += f"Context: {context}\n\n"

            image_prompt_request += (
                f"Return exactly {num_prompts} image prompts, each on a new line, numbered 1-{num_prompts}. "
                f"Each should be child-friendly, detailed, and suitable for AI image generation."
            )

            response = self.client.chat.completions.create(
                model=self.story_model,
                messages=[{"role": "user", "content": image_prompt_request}],
            )

            # Extract and parse the prompts
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                text = response.choices[0].message.content.strip()
                # Parse numbered prompts from GPT's response
                lines = text.split("\n")
                prompts: list[str] = []

                for line in lines:
                    line = line.strip()
                    # Look for numbered lines like "1. " or "1) "
                    if line and (line[0].isdigit() or line.startswith(str(len(prompts) + 1))):
                        # Remove the number and clean up
                        cleaned = line
                        for prefix in [
                            f"{len(prompts) + 1}. ",
                            f"{len(prompts) + 1}) ",
                            f"{len(prompts) + 1}: ",
                        ]:
                            if cleaned.startswith(prefix):
                                cleaned = cleaned[len(prefix) :]
                                break
                        if cleaned:
                            prompts.append(cleaned)

                # If we got the right number of prompts, return them
                if len(prompts) == num_prompts:
                    return prompts

            # Fallback: Simple story-based prompts
            return self._generate_fallback_image_prompts(story, context, num_prompts)

        except Exception:
            # Fallback to simple story-based prompts
            return self._generate_fallback_image_prompts(story, context, num_prompts)

    def _generate_fallback_image_prompts(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Generate fallback image prompts when OpenAI API fails.

        Args:
            story (str): The story text.
            context (str): Additional context.
            num_prompts (int): Number of prompts needed.

        Returns:
            list[str]: Simple fallback image prompts.
        """
        # Simple fallback: split story into paragraphs, or repeat the story if not enough
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        prompts = []

        for i in range(num_prompts):
            if i < len(paragraphs):
                base = paragraphs[i]
            else:
                base = story
            prompt = f"Create a detailed, child-friendly illustration for this part of the story: {base}"
            if context:
                prompt += f"\nContext: {context}"
            prompts.append(prompt)

        return prompts

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

            response = self.client.chat.completions.create(
                model=self.story_model, messages=[{"role": "user", "content": contents}], temperature=1
            )

            # Extract the story text from the response with proper null checking
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "[Error: No valid response from OpenAI]"
        except Exception as e:
            # Return a generic error message if generation fails
            return f"[Error generating story: {str(e)}]"

    def generate_image(
        self, prompt: Prompt, reference_image_bytes: bytes | None = None
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given Prompt object using OpenAI
        DALL-E model, optionally using a reference image for consistency.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive image
                generation parameters including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images (currently not used by DALL-E).

        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        try:
            # Use the Prompt's comprehensive image prompt building
            text_prompt = prompt.image(1)[0]

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

                if (
                    compression_response.choices
                    and compression_response.choices[0].message
                    and compression_response.choices[0].message.content
                ):
                    text_prompt = compression_response.choices[0].message.content.strip()
                else:
                    # Fallback to simple truncation if compression fails
                    text_prompt = text_prompt[:3900] + "..."

            # Generate image using configured model (gpt-image-1.5 or dall-e-3)
            # Use quality="auto" for gpt-image models, "standard" for dall-e models
            quality: Literal["auto", "standard"] = "auto" if "gpt-image" in self.image_model else "standard"
            response = self.client.images.generate(
                prompt=text_prompt,
                model=self.image_model,
                size="1024x1024",
                quality=quality,
                n=1,
            )

            if response.data and len(response.data) > 0 and response.data[0].url:
                # Download the image from the URL
                import requests

                img_response = requests.get(response.data[0].url, timeout=30)
                if img_response.status_code == 200:
                    image_bytes = img_response.content
                    image = Image.open(BytesIO(image_bytes))
                    return image, image_bytes

            return None, None

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error generating image with DALL-E: {e}")
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
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                name = response.choices[0].message.content.strip()
                # Remove file extension if present
                name = name.split(".")[0]
                # Clean up any unwanted characters to match Anthropic pattern
                name = "".join(c for c in name if c.isalnum() or c == "_")
                return name if name else "story_image"
            else:
                return "story_image"
        except Exception:
            # Fallback filename
            return "story_image"
