"""
Context Management for StoryForge.

This module provides utilities for loading and managing story context from files
like character descriptions, background information, and story examples.

Future enhancements:
- Intelligent character detection from prompts
- Context relevance scoring and filtering
- Token usage optimization
- Multiple context profile support
"""

from pathlib import Path

from platformdirs import user_data_dir

# Use "StoryForge" as appauthor for user_data_dir to ensure user-agnostic,
# organization-consistent data storage


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
        Load context from all markdown files in the context directory.

        Returns:
            str: The concatenated context content from all .md files,
            or None if none found.

        New behavior:
        - Finds all .md files in the context directory (default or specified)
        - Sorts them by last modified date (oldest to newest)
        - Concatenates their contents

        """
        if self._cached_context is not None:
            return self._cached_context

        # If a specific file is set, use only that file
        if self.context_file_path:
            context_files = [Path(self.context_file_path)]
        else:
            import os

            # Allow tests to override the context directory via env var
            test_context_dir = os.environ.get("STORYTIME_TEST_CONTEXT_DIR")
            if test_context_dir:
                context_dir = Path(test_context_dir)
                if context_dir.exists() and context_dir.is_dir():
                    context_files = sorted(context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    context_files = []
            else:
                # Prefer ./context/ in the current working directory if it exists
                local_context_dir = Path("context")
                if local_context_dir.exists() and local_context_dir.is_dir():
                    context_files = sorted(local_context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    user_dir = Path(user_data_dir("StoryForge", "StoryForge")) / "context"
                    if user_dir.exists() and user_dir.is_dir():
                        context_files = sorted(user_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                    else:
                        context_files = []

        if not context_files:
            return None

        contents = []
        for file_path in context_files:
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        contents.append(f.read().strip())
                except OSError:
                    continue

        if not contents:
            return None

        self._cached_context = "\n\n".join(contents)
        return self._cached_context

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

        # Use cross-platform user data directory for context files
        context_dir = Path(user_data_dir("StoryForge", "StoryForge")) / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        default_path = context_dir / "family.md"
        if default_path.exists():
            return default_path

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
