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


class GeminiBackend(LLMBackend):
    """
    LLM backend implementation using Google Gemini APIs.
    Requires GEMINI_API_KEY environment variable to be set.
    """

    def __init__(self):
        """
        Initialize the Gemini client using the API key from environment variables.
        Raises:
            RuntimeError: If GEMINI_API_KEY is not set.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

    def generate_story(self, prompt: str) -> str:
        """
        Generate a short story based on the given prompt using Gemini LLM.
        Args:
            prompt (str): The prompt to base the story on.
        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Write a short story based on this prompt: {prompt}",
            )
            # Extract the story text from the response
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            # Return a generic error message if generation fails
            return "[Error generating story]"

    def generate_image(self, prompt: str) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given story prompt using Gemini
        image model.
        Args:
            prompt (str): The story prompt to illustrate.
        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        contents = f"Create a detailed, beautiful illustration for this story: {prompt}"
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        # Iterate through response parts to find image data
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
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

    def generate_image_name(self, prompt: str, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story.
        Args:
            prompt (str): The original prompt.
            story (str): The generated story.
        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    f"Given this story: {story}\n"
                    "Suggest a short, creative, and descriptive filename for an "
                    "image illustrating it (no spaces, no special characters, "
                    "just letters, numbers, and underscores)."
                ),
            )
            name = response.candidates[0].content.parts[0].text.strip()
            # Remove file extension if present
            name = name.split(".")[0]
            return name
        except Exception:
            # Fallback filename
            return "story_image"
