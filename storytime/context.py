"""
Context Management for StoryTime.

This module provides utilities for loading and managing story context from files
like character descriptions, background information, and story examples.

Future enhancements:
- Intelligent character detection from prompts
- Context relevance scoring and filtering
- Token usage optimization
- Multiple context profile support
"""

from pathlib import Path


class ContextManager:
    """
    Manages story context loading and processing.

    Currently supports basic file loading. Future versions will include:
    - Smart context extraction based on prompt analysis
    - Character-specific context filtering
    - Token budget management
    - Context caching and optimization
    """

    def __init__(self, context_file_path: str | None = None):
        """
        Initialize the ContextManager.

        Args:
            context_file_path: Path to the context file (e.g., family.md).
                             If None, will look for data/family.md in project root.
        """
        self.context_file_path = context_file_path
        self._cached_context: str | None = None

    def load_context(self) -> str | None:
        """
        Load context from the configured file.

        Returns:
            str: The loaded context content, or None if file not found/readable.

        Future enhancements:
        - Parse and structure context data (characters, relationships, etc.)
        - Validate context format
        - Support multiple context file formats (JSON, YAML, etc.)
        """
        if self._cached_context is not None:
            return self._cached_context

        context_path = self._resolve_context_path()
        if not context_path or not context_path.exists():
            return None

        try:
            with open(context_path, encoding="utf-8") as f:
                self._cached_context = f.read().strip()
            return self._cached_context
        except OSError:
            return None

    def _resolve_context_path(self) -> Path | None:
        """
        Resolve the context file path.

        Returns:
            Path: The resolved path to the context file.

        Future enhancements:
        - Support environment variable overrides
        - Search multiple default locations
        - Support URL-based context loading
        """
        if self.context_file_path:
            return Path(self.context_file_path)

        # Default: look for data/family.md relative to project root
        # Find project root by looking for pyproject.toml
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:  # Not at filesystem root
            if (current_dir / "pyproject.toml").exists():
                default_path = current_dir / "data" / "family.md"
                if default_path.exists():
                    return default_path
                break
            current_dir = current_dir.parent

        return None

    def extract_relevant_context(self, prompt: str) -> str | None:
        """
        Extract context relevant to the given prompt.

        Currently returns all context. Future smart extraction will:
        - Detect character names mentioned in the prompt
        - Score context sections by relevance
        - Filter to only include relevant characters and relationships
        - Respect token budget limits
        - Include relevant story examples for consistency

        Args:
            prompt: The user's story prompt

        Returns:
            str: Relevant context for the prompt, or None if no context available
        """
        # TODO: Implement smart context extraction
        # For now, return all context (basic implementation)
        full_context = self.load_context()

        if not full_context:
            return None

        # Future implementation points:
        # 1. Parse prompt for character names (Ethan, Isaac, etc.)
        # 2. Extract character descriptions for mentioned characters
        # 3. Add related characters (family members, pets)
        # 4. Include relevant story examples
        # 5. Apply token budget limits

        return full_context

    def clear_cache(self):
        """Clear cached context data."""
        self._cached_context = None


def get_default_context_manager() -> ContextManager:
    """
    Get a default context manager instance.

    Returns:
        ContextManager: Configured with default settings
    """
    return ContextManager()
