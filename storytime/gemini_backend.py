import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional

from .llm_backend import LLMBackend

class GeminiBackend(LLMBackend):
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

    def generate_story(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Write a short story based on this prompt: {prompt}"
            )
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return "[Error generating story]"

    def generate_image(self, prompt: str) -> Tuple[Optional[object], Optional[bytes]]:
        contents = f"Create a detailed, beautiful illustration for this story: {prompt}"
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image_bytes = part.inline_data.data
                    return image, image_bytes
                except Exception:
                    continue
        return None, None

    def generate_image_name(self, prompt: str, story: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Given this story: {story}\nSuggest a short, creative, and descriptive filename for an image illustrating it (no spaces, no special characters, just letters, numbers, and underscores)."
            )
            name = response.candidates[0].content.parts[0].text.strip()
            name = name.split(".")[0]
            return name
        except Exception:
            return "story_image"
