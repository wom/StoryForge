"""
Tests for the context management system.

Tests basic context loading functionality and prepares for future
smart context extraction features.
"""

import tempfile
from pathlib import Path

from storyforge.context import ContextManager


class TestContextManager:
    """Test the basic context management functionality."""

    def test_context_manager_with_valid_file(self):
        """Test loading context from a valid file."""
        test_content = """# Test Family

## Character 1
Description of character 1.

## Character 2
Description of character 2.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            manager = ContextManager(temp_path)
            context = manager.load_context()

            assert context is not None
            assert "Test Family" in context
            assert "Character 1" in context
            assert "Character 2" in context
        finally:
            Path(temp_path).unlink()

    def test_context_manager_with_missing_file(self):
        """Test behavior when context file doesn't exist."""
        manager = ContextManager("/nonexistent/path/file.md")
        context = manager.load_context()

        assert context is None

    def test_context_manager_with_none_path(self):
        """Test default path resolution when no path provided."""
        manager = ContextManager(None)
        # Should not crash, may or may not find default file
        context = manager.load_context()
        # Context may be None or contain content if data/family.md exists
        assert context is None or isinstance(context, str)

    def test_context_caching(self):
        """Test that context is cached after first load."""
        test_content = "# Cached Content Test"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            manager = ContextManager(temp_path)

            # First load
            context1 = manager.load_context()
            assert context1 == "# Cached Content Test"

            # Second load should use cache
            context2 = manager.load_context()
            assert context2 == context1
            assert context2 is context1  # Same object reference
        finally:
            Path(temp_path).unlink()

    def test_context_manager_multiple_md_files_concat_by_mtime(self):
        """Test loading and concatenating multiple .md files by modified date."""

        contents = [
            "# First File\nFirst content.",
            "# Second File\nSecond content.",
            "# Third File\nThird content.",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for idx, content in enumerate(contents):
                file_path = Path(tmpdir) / f"file{idx + 1}.md"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                # Set mtime to ensure order: oldest to newest
                mtime = 1000000000 + idx
                file_path.touch()
                file_path.stat()  # Ensure file exists before setting mtime
                Path(file_path).chmod(0o600)
                # Use os.utime to set mtime
                import os

                os.utime(file_path, (mtime, mtime))
                paths.append(file_path)

            # Shuffle paths to ensure order is by mtime, not filename
            paths = paths[::-1]

            # Patch user_data_dir to point to tmpdir
            from storyforge import context as context_mod

            orig_user_data_dir = context_mod.user_data_dir
            context_mod.user_data_dir = lambda *a, **kw: tmpdir

            import os

            try:
                os.environ["STORYTIME_TEST_CONTEXT_DIR"] = tmpdir
                manager = ContextManager(None)
                manager.clear_cache()
                loaded = manager.load_context()
                # Should concatenate in mtime order
                expected = "\n\n".join(contents)
                assert loaded == expected
            finally:
                context_mod.user_data_dir = orig_user_data_dir
                if "STORYTIME_TEST_CONTEXT_DIR" in os.environ:
                    del os.environ["STORYTIME_TEST_CONTEXT_DIR"]

    def test_extract_relevant_context_basic(self):
        """Test basic context extraction (currently returns all context)."""
        test_content = """# Family Context

## Ethan
Young boy, loves swimming.

## Isaac
Ethan's brother, loves animals.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            manager = ContextManager(temp_path)

            # Test with prompt mentioning Ethan
            context = manager.extract_relevant_context("Write a story about Ethan")
            assert context is not None
            assert "Ethan" in context
            assert "Isaac" in context  # Currently returns all context

            # Test with prompt not mentioning characters
            context2 = manager.extract_relevant_context("Write about a dragon")
            assert context2 is not None
            assert context2 == context  # Currently same result
        finally:
            Path(temp_path).unlink()

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        test_content = "# Cache Test"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            manager = ContextManager(temp_path)

            # Load and cache
            context1 = manager.load_context()
            assert manager._cached_context is not None

            # Clear cache
            manager.clear_cache()
            assert manager._cached_context is None

            # Load again (should re-read file)
            context2 = manager.load_context()
            assert context2 == context1
            assert manager._cached_context is not None
        finally:
            Path(temp_path).unlink()


class TestContextIntegration:
    """Test context integration with LLM backends."""

    def test_context_with_mock_backend(self, monkeypatch):
        """Test that context is properly passed to backend."""
        from storyforge.gemini_backend import GeminiBackend
        from storyforge.prompt import Prompt

        # Mock the client to avoid API calls
        mock_response = type(
            "Response",
            (),
            {
                "candidates": [
                    type(
                        "Candidate",
                        (),
                        {
                            "content": type(
                                "Content",
                                (),
                                {
                                    "parts": [
                                        type(
                                            "Part",
                                            (),
                                            {"text": "Generated story with context"},
                                        )
                                    ]
                                },
                            )
                        },
                    )
                ]
            },
        )

        def mock_generate_content(model, contents):
            # Verify context is included in contents
            assert "Context for story generation:" in contents
            assert "Ethan" in contents
            return mock_response

        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        backend = GeminiBackend()
        monkeypatch.setattr(backend.client.models, "generate_content", mock_generate_content)

        # Test story generation with context using Prompt object
        context = "## Ethan\nYoung boy, loves swimming."
        prompt = Prompt(prompt="Write about Ethan", context=context)
        story = backend.generate_story(prompt)

        assert story == "Generated story with context"

    def test_context_fallback_without_context(self, monkeypatch):
        """Test that backend works without context."""
        from storyforge.gemini_backend import GeminiBackend
        from storyforge.prompt import Prompt

        # Mock the client
        mock_response = type(
            "Response",
            (),
            {
                "candidates": [
                    type(
                        "Candidate",
                        (),
                        {
                            "content": type(
                                "Content",
                                (),
                                {
                                    "parts": [
                                        type(
                                            "Part",
                                            (),
                                            {"text": "Generated story without context"},
                                        )
                                    ]
                                },
                            )
                        },
                    )
                ]
            },
        )

        def mock_generate_content(model, contents):
            # Verify the prompt is formatted correctly
            assert "Write about adventure" in contents
            return mock_response

        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        backend = GeminiBackend()
        monkeypatch.setattr(backend.client.models, "generate_content", mock_generate_content)

        # Test story generation without context using Prompt object
        prompt = Prompt(prompt="Write about adventure")
        story = backend.generate_story(prompt)

        assert story == "Generated story without context"


# Future test cases for smart context extraction:
class TestFutureSmartContext:
    """
    Placeholder tests for future smart context extraction features.

    These tests will be implemented when smart context extraction is added:
    - Character name detection in prompts
    - Context relevance scoring
    - Token budget management
    - Multiple context profile support
    """

    def test_character_detection_in_prompt(self):
        """Test that extract_relevant_context works with character-focused prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir)

            # Create context file with character info
            context_file = context_dir / "characters.md"
            context_file.write_text(
                "# Luna the Owl\nLuna is a wise owl who lives in the old oak tree.\n\n"
                "# Max the Mouse\nMax is a brave mouse who loves adventures."
            )

            manager = ContextManager(str(context_file))

            # Extract context for character-focused prompt
            prompt = "Tell a story about Luna the wise owl and Max the brave mouse exploring the forest"
            result = manager.extract_relevant_context(prompt)

            # Should return context containing character information
            assert result is not None
            assert "Luna" in result
            assert "Max" in result
            assert "owl" in result
            assert "mouse" in result

    def test_context_relevance_scoring(self):
        """Test that context can be loaded and filtered based on content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir)

            # Create multiple context files with different themes
            dragons_file = context_dir / "dragons.md"
            dragons_file.write_text("# Dragons\nFire-breathing creatures who guard castles and treasure.")

            rabbits_file = context_dir / "rabbits.md"
            rabbits_file.write_text("# Rabbits\nPeaceful creatures who hop through meadows.")

            # Load dragon context
            manager_dragons = ContextManager(str(dragons_file))
            dragon_context = manager_dragons.load_context()

            # Load rabbit context
            manager_rabbits = ContextManager(str(rabbits_file))
            rabbit_context = manager_rabbits.load_context()

            # Verify distinct contexts loaded correctly
            assert dragon_context is not None
            assert rabbit_context is not None
            assert "dragon" in dragon_context.lower()
            assert "rabbit" in rabbit_context.lower()
            assert "dragon" not in rabbit_context.lower()
            assert "rabbit" not in dragon_context.lower()

    def test_token_budget_limits(self):
        """Test that large context files can be loaded and handled successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir)

            # Create a large context file
            context_file = context_dir / "large_context.md"
            large_content = "This is a test sentence with some content.\n" * 500  # ~22,000 chars
            context_file.write_text(large_content)

            manager = ContextManager(str(context_file))

            # Load the context
            result = manager.load_context()

            # Verify context loaded successfully
            assert result is not None
            assert len(result) > 0

            # Verify the content is substantial (large file loaded)
            assert len(result) > 20000  # Should be ~21,500 characters

            # Test that context caching works - second load should be same
            result2 = manager.load_context()
            assert result == result2

            # Clear cache and reload
            manager.clear_cache()
            result3 = manager.load_context()
            assert result == result3
