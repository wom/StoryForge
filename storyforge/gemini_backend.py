"""
GeminiBackend: Implementation of LLMBackend using Google Gemini APIs.
Provides methods to generate stories, images, and image filenames using Gemini models.
"""

import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from .llm_backend import LLMBackend
from .prompt import Prompt


class GeminiBackend(LLMBackend):
    """
    LLM backend implementation using Google Gemini APIs.
    Requires GEMINI_API_KEY environment variable to be set.
    """

    name = "gemini"

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Return a list of image prompts by splitting the story into num_prompts chunks.
        This is a stub implementation to satisfy the abstract base class.
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

    def __init__(self) -> None:
        """
        Initialize the Gemini client using the API key from environment variables.
        Raises:
            RuntimeError: If GEMINI_API_KEY is not set.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

    def generate_story(self, prompt: Prompt) -> str:
        """
        Generate a story based on the given Prompt object using Gemini LLM.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Use the Prompt's comprehensive prompt building
            contents = prompt.story

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                # model="gemini-2.5-pro",
                contents=contents,
            )
            # Extract the story text from the response with proper null checking
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
                and response.candidates[0].content.parts[0].text
            ):
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "[Error: No valid response from Gemini]"
        except Exception:
            # Return a generic error message if generation fails
            return "[Error generating story]"

    def generate_image(
        self, prompt: Prompt, reference_image_bytes: bytes | None = None
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given Prompt object using Gemini
        image model, optionally using a reference image for consistency.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive image
                generation parameters including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images.

        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        # Use the Prompt's comprehensive image prompt building
        text_prompt = prompt.image(1)[0]

        # Build contents with optional reference image
        if reference_image_bytes:
            # For subsequent images, include reference image for consistency
            contents: list | str = [
                types.Part(inline_data=types.Blob(mime_type="image/png", data=reference_image_bytes)),
                (
                    "Create the next illustration in the same visual style, "
                    "with consistent characters and art style as the reference image. "
                    f"{text_prompt}"
                ),
            ]
        else:
            # First image - no reference needed
            contents = text_prompt

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        # Iterate through response parts to find image data with proper null checking
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None and part.inline_data.data is not None:
                    try:
                        # Load image from bytes
                        image = Image.open(BytesIO(part.inline_data.data))
                        image_bytes = part.inline_data.data
                        return image, image_bytes
                    except Exception:
                        # Skip parts that fail to decode as images
                        continue
        # Return (None, None) if no image found
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

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            # Extract name with proper null checking
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
                and response.candidates[0].content.parts[0].text
            ):
                name = response.candidates[0].content.parts[0].text.strip()
                # Remove file extension if present
                name = name.split(".")[0]
                return name
            else:
                return "story_image"
        except Exception:
            # Fallback filename
            return "story_image"
