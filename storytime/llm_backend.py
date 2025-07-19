"""
LLM Backend Factory and Abstract Base Class.

This module provides:
1. LLMBackend abstract base class defining the interface for all backends
2. get_backend() factory function for automatic backend selection
3. Backend detection logic based on environment variables

Supported backends:
- Gemini (Google AI): requires GEMINI_API_KEY
- Future: OpenAI, Anthropic, etc.
"""
import os
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



def get_backend(backend_name: Optional[str] = None) -> LLMBackend:
    """
    Factory function to get an appropriate LLM backend instance.
    
    Args:
        backend_name (Optional[str]): Specific backend to use ("gemini", "openai", etc.).
                                    If None, auto-detect based on available API keys.
    
    Returns:
        LLMBackend: An instance of the selected backend.
        
    Raises:
        RuntimeError: If no suitable backend is found or API keys are missing.
        ImportError: If required dependencies for the backend are not installed.
    
    Environment Variables:
        LLM_BACKEND: Override automatic detection (e.g., "gemini", "openai")
        GEMINI_API_KEY: Required for Gemini backend
        OPENAI_API_KEY: Required for OpenAI backend (future)
        ANTHROPIC_API_KEY: Required for Anthropic backend (future)
    """
    
    # Override backend selection if explicitly requested
    if backend_name is None:
        backend_name = os.environ.get("LLM_BACKEND", "").lower()
    
    # If still no explicit backend, auto-detect based on available API keys
    if not backend_name:
        if os.environ.get("GEMINI_API_KEY"):
            backend_name = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            backend_name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend_name = "anthropic"
        else:
            raise RuntimeError(
                "No LLM backend available. Please set one of the following environment variables:\n"
                "- GEMINI_API_KEY (for Google Gemini)\n"
                "- OPENAI_API_KEY (for OpenAI - future)\n"
                "- ANTHROPIC_API_KEY (for Anthropic - future)\n"
                "Or set LLM_BACKEND to specify which backend to use."
            )
    
    # Import and instantiate the requested backend
    try:
        if backend_name == "gemini":
            from .gemini_backend import GeminiBackend
            return GeminiBackend()
        
        elif backend_name == "openai":
            # Future implementation
            raise RuntimeError(
                "OpenAI backend not yet implemented. "
                "Please use Gemini backend by setting GEMINI_API_KEY."
            )
        
        elif backend_name == "anthropic":
            # Future implementation
            raise RuntimeError(
                "Anthropic backend not yet implemented. "
                "Please use Gemini backend by setting GEMINI_API_KEY."
            )
        
        else:
            raise RuntimeError(
                f"Unknown backend '{backend_name}'. "
                f"Supported backends: gemini (more coming soon)"
            )
    
    except ImportError as e:
        raise ImportError(
            f"Failed to import {backend_name} backend. "
            f"Please ensure required dependencies are installed: {e}"
        )


def list_available_backends() -> dict:
    """
    List all available backends and their status.
    
    Returns:
        dict: Backend name -> {"available": bool, "reason": str}
    """
    backends = {}
    
    # Check Gemini
    try:
        import google.genai
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        backends["gemini"] = {
            "available": has_key,
            "reason": "Ready" if has_key else "GEMINI_API_KEY not set"
        }
    except ImportError:
        backends["gemini"] = {
            "available": False,
            "reason": "google-genai package not installed"
        }
    
    # Future backends would be checked here
    backends["openai"] = {
        "available": False,
        "reason": "Not yet implemented"
    }
    
    backends["anthropic"] = {
        "available": False,
        "reason": "Not yet implemented"
    }
    
    return backends
