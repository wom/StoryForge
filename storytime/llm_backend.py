"""
Abstract base class for Large Language Model (LLM) backends.
Defines the required interface for any backend that can generate stories and images.
"""
from typing import Tuple, Optional
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    """
    Abstract base class for Large Language Model backends.
    
    This class defines the interface that all LLM backends must implement
    to provide story generation, image generation, and image naming capabilities.
    """
    
    @abstractmethod
    def generate_story(self, prompt: str) -> str:
        """
        Generate a story based on the given prompt.
        
        Args:
            prompt (str): The user's story prompt or description.
            
        Returns:
            str: The generated story text.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_story method")

    @abstractmethod
    def generate_image(self, prompt: str) -> Tuple[Optional[object], Optional[bytes]]:
        """
        Generate an image based on the given prompt.
        
        Args:
            prompt (str): The prompt or description for image generation.
            
        Returns:
            Tuple[Optional[object], Optional[bytes]]: A tuple containing:
                - The image object (PIL.Image or similar), or None if generation failed
                - The raw image bytes, or None if generation failed
                
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_image method")

    @abstractmethod
    def generate_image_name(self, prompt: str, story: str) -> str:
        """
        Generate a descriptive filename for an image based on the prompt and story.
        
        Args:
            prompt (str): The original user prompt.
            story (str): The generated story text.
            
        Returns:
            str: A suggested filename (without extension) that is descriptive
                 and filesystem-safe (no spaces or special characters).
                 
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_image_name method")
