from typing import Tuple, Optional

class LLMBackend:
    def generate_story(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_image(self, prompt: str) -> Tuple[Optional[object], Optional[bytes]]:
        raise NotImplementedError

    def generate_image_name(self, prompt: str, story: str) -> str:
        raise NotImplementedError
