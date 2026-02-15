"""Tests for ContextManager character registry functionality."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from storyforge.context import ContextManager


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with context and registry support."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    return context_dir


@pytest.fixture
def sample_story_content() -> str:
    """Create sample story content for extraction."""
    return (
        "# Story Context: A woodland adventure\n\n"
        "**Generated on:** 2025-01-01 12:00:00\n"
        "**Original Prompt:** A woodland adventure\n"
        "**Characters:** Luna, Max\n\n"
        "## Story\n\n"
        "Luna walked through the enchanted forest with her friend Max. "
        "Luna was brave and curious, always looking for new adventures. "
        "Max was a wise owl who knew many secrets of the forest. "
        "Together Luna and Max discovered a hidden cave. "
        "Luna jumped with joy while Max observed carefully. "
        "The cave glowed with magical crystals that Luna had never seen before."
    )


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Create sample metadata dict."""
    return {"characters": "Luna, Max", "theme": "adventure"}


class TestGetRegistryPath:
    """Test _get_registry_path method."""

    def test_returns_path_object(self):
        """Test that registry path is a Path."""
        cm = ContextManager()
        path = cm._get_registry_path()
        assert isinstance(path, Path)

    def test_ends_with_expected_filename(self):
        """Test registry filename is character_registry.json."""
        cm = ContextManager()
        path = cm._get_registry_path()
        assert path.name == "character_registry.json"


class TestLoadRegistry:
    """Test _load_registry method."""

    def test_missing_file_returns_empty(self, tmp_path: Path):
        """Test missing registry file returns empty structure."""
        cm = ContextManager()
        # Point to a nonexistent directory
        cm.get_context_directory = lambda: tmp_path / "nonexistent"  # type: ignore[assignment]
        registry = cm._load_registry()
        assert registry == {"characters": {}, "last_updated": None}

    def test_corrupt_json_returns_empty(self, tmp_path: Path):
        """Test corrupt JSON returns empty structure."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry_path = tmp_path / "character_registry.json"
        registry_path.write_text("not valid json{{{")
        registry = cm._load_registry()
        assert registry == {"characters": {}, "last_updated": None}

    def test_missing_characters_key_returns_empty(self, tmp_path: Path):
        """Test JSON without 'characters' key returns empty."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry_path = tmp_path / "character_registry.json"
        registry_path.write_text(json.dumps({"foo": "bar"}))
        registry = cm._load_registry()
        assert registry == {"characters": {}, "last_updated": None}

    def test_valid_registry_loaded(self, tmp_path: Path):
        """Test valid registry file is loaded correctly."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry_data = {
            "characters": {
                "Luna": {
                    "first_appeared": "story_001",
                    "appearances": ["story_001"],
                    "traits": ["Luna is brave."],
                }
            },
            "last_updated": "2025-01-01T00:00:00",
        }
        registry_path = tmp_path / "character_registry.json"
        registry_path.write_text(json.dumps(registry_data))
        registry = cm._load_registry()
        assert "Luna" in registry["characters"]
        assert registry["characters"]["Luna"]["traits"] == ["Luna is brave."]


class TestSaveRegistry:
    """Test _save_registry method."""

    def test_save_creates_file(self, tmp_path: Path):
        """Test save creates the registry file."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry = {"characters": {"Luna": {"traits": []}}, "last_updated": None}
        cm._save_registry(registry)
        path = tmp_path / "character_registry.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "Luna" in data["characters"]
        assert data["last_updated"] is not None  # Should be set by save

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """Test save creates parent directories if needed."""
        cm = ContextManager()
        nested = tmp_path / "deep" / "nested"
        cm.get_context_directory = lambda: nested  # type: ignore[assignment]
        registry = {"characters": {}, "last_updated": None}
        cm._save_registry(registry)
        assert (nested / "character_registry.json").exists()

    def test_save_sets_last_updated(self, tmp_path: Path):
        """Test save sets last_updated timestamp."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry = {"characters": {}, "last_updated": None}
        cm._save_registry(registry)
        path = tmp_path / "character_registry.json"
        data = json.loads(path.read_text())
        assert data["last_updated"] is not None


class TestExtractCharactersFromStory:
    """Test _extract_characters_from_story method."""

    def test_extract_from_metadata_field(self):
        """Test characters extracted from metadata 'characters' field."""
        cm = ContextManager()
        content = "Some story about Luna and Max."
        metadata: dict[str, Any] = {"characters": "Luna, Max"}
        result = cm._extract_characters_from_story(content, metadata)
        assert "Luna" in result
        assert "Max" in result

    def test_extract_heuristic_ner(self):
        """Test heuristic extraction from capitalized word frequency."""
        cm = ContextManager()
        content = (
            "## Story\n\n"
            "Milo ran through the forest. Milo was very brave. "
            "Milo found a treasure. The treasure was shiny."
        )
        metadata: dict[str, Any] = {}
        result = cm._extract_characters_from_story(content, metadata)
        assert "Milo" in result  # Appears 3+ times

    def test_stopwords_not_extracted(self):
        """Test common stopwords are not treated as characters."""
        cm = ContextManager()
        # "The" appears many times but should be filtered
        content = "## Story\n\nThe cat sat. The cat ran. The cat jumped. The cat played."
        metadata: dict[str, Any] = {}
        result = cm._extract_characters_from_story(content, metadata)
        assert "The" not in result

    def test_trait_sentences_extracted(self, sample_story_content: str, sample_metadata: dict[str, Any]):
        """Test trait sentences are extracted for characters."""
        cm = ContextManager()
        result = cm._extract_characters_from_story(sample_story_content, sample_metadata)
        assert "Luna" in result
        # Luna should have traits extracted
        assert len(result["Luna"]) > 0
        # Trait sentences should mention Luna
        assert any("Luna" in trait for trait in result["Luna"])

    def test_trait_cap(self):
        """Test that traits are capped at 3 per character in extraction."""
        cm = ContextManager()
        # Create content with many sentences about the same character
        sentences = [f"Luna did thing {i}." for i in range(10)]
        content = "## Story\n\n" + " ".join(sentences)
        metadata: dict[str, Any] = {"characters": "Luna"}
        result = cm._extract_characters_from_story(content, metadata)
        # Extraction caps at 3 traits per character
        assert len(result["Luna"]) <= 3

    def test_empty_content(self):
        """Test empty content returns empty dict."""
        cm = ContextManager()
        result = cm._extract_characters_from_story("", {})
        assert result == {}

    def test_short_names_ignored(self):
        """Test single-character names from metadata are ignored."""
        cm = ContextManager()
        result = cm._extract_characters_from_story("A story.", {"characters": "A, B"})
        assert "A" not in result
        assert "B" not in result


class TestUpdateCharacterRegistry:
    """Test update_character_registry method."""

    def test_add_new_character(self, tmp_path: Path, sample_story_content: str, sample_metadata: dict[str, Any]):
        """Test adding a new character to empty registry."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        cm.update_character_registry(sample_story_content, sample_metadata, "story_001")
        registry = cm._load_registry()
        assert "Luna" in registry["characters"]
        assert registry["characters"]["Luna"]["first_appeared"] == "story_001"
        assert "story_001" in registry["characters"]["Luna"]["appearances"]

    def test_update_existing_character(self, tmp_path: Path):
        """Test updating an existing character adds appearance and traits."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        # First story
        content1 = "## Story\n\nLuna ran fast. Luna jumped high. Luna was brave."
        cm.update_character_registry(content1, {"characters": "Luna"}, "story_001")

        # Second story with same character
        content2 = "## Story\n\nLuna found a key. Luna opened the door. Luna smiled."
        cm.update_character_registry(content2, {"characters": "Luna"}, "story_002")

        registry = cm._load_registry()
        luna = registry["characters"]["Luna"]
        assert luna["first_appeared"] == "story_001"
        assert "story_001" in luna["appearances"]
        assert "story_002" in luna["appearances"]

    def test_trait_deduplication(self, tmp_path: Path):
        """Test that duplicate traits are not added."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        content = "## Story\n\nLuna was brave. Luna jumped high. Luna was brave."
        cm.update_character_registry(content, {"characters": "Luna"}, "story_001")
        # Same traits again
        cm.update_character_registry(content, {"characters": "Luna"}, "story_002")

        registry = cm._load_registry()
        luna = registry["characters"]["Luna"]
        # Traits should not have duplicates
        trait_lower = [t.lower() for t in luna["traits"]]
        assert len(trait_lower) == len(set(trait_lower))

    def test_trait_cap_enforced(self, tmp_path: Path):
        """Test MAX_TRAITS_PER_CHARACTER is respected during update."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        for i in range(10):
            content = f"## Story\n\nLuna action_{i}_a. Luna action_{i}_b. Luna action_{i}_c."
            cm.update_character_registry(content, {"characters": "Luna"}, f"story_{i:03d}")

        registry = cm._load_registry()
        assert len(registry["characters"]["Luna"]["traits"]) <= cm.MAX_TRAITS_PER_CHARACTER

    def test_no_duplicate_appearances(self, tmp_path: Path):
        """Test same filename doesn't add duplicate appearance."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        content = "## Story\n\nLuna ran. Luna jumped. Luna played."
        cm.update_character_registry(content, {"characters": "Luna"}, "story_001")
        cm.update_character_registry(content, {"characters": "Luna"}, "story_001")

        registry = cm._load_registry()
        assert registry["characters"]["Luna"]["appearances"].count("story_001") == 1


class TestBuildCharacterRegistry:
    """Test build_character_registry full scan."""

    def test_build_from_context_files(self, tmp_path: Path):
        """Test full registry build from context files."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(context_dir)

        try:
            # Create context files
            (context_dir / "story1.md").write_text(
                "# Story Context: Adventure\n"
                "**Characters:** Luna, Max\n\n"
                "## Story\n\n"
                "Luna explored the cave. Luna found gems. Luna was happy. "
                "Max flew overhead. Max was wise. Max guided Luna."
            )
            (context_dir / "story2.md").write_text(
                "# Story Context: Journey\n"
                "**Characters:** Luna, Pip\n\n"
                "## Story\n\n"
                "Luna sailed the sea. Luna was adventurous. Luna loved the ocean. "
                "Pip was a small mouse. Pip was curious. Pip followed Luna."
            )

            cm = ContextManager()
            cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
            result = cm.build_character_registry()

            assert result  # Non-empty string
            assert "Luna" in result

            # Check registry was saved
            registry = cm._load_registry()
            assert "Luna" in registry["characters"]
            assert len(registry["characters"]["Luna"]["appearances"]) >= 1
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_build_with_no_files(self, tmp_path: Path):
        """Test build with no context files returns empty string."""
        context_dir = tmp_path / "empty_context"
        context_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(context_dir)

        try:
            cm = ContextManager()
            result = cm.build_character_registry()
            assert result == ""
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]


class TestFormatRegistryForPrompt:
    """Test format_registry_for_prompt method."""

    def test_empty_registry(self, tmp_path: Path):
        """Test empty registry returns empty string."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        result = cm.format_registry_for_prompt()
        assert result == ""

    def test_formatted_output(self, tmp_path: Path):
        """Test formatted output contains expected elements."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry = {
            "characters": {
                "Luna": {
                    "first_appeared": "story_001",
                    "appearances": ["story_001", "story_002"],
                    "traits": ["Luna is brave.", "Luna loves adventure."],
                },
                "Max": {
                    "first_appeared": "story_001",
                    "appearances": ["story_001"],
                    "traits": ["Max is wise."],
                },
            },
            "last_updated": "2025-01-01T00:00:00",
        }
        cm._save_registry(registry)

        result = cm.format_registry_for_prompt()
        assert "## Known Characters" in result
        assert "**Luna**" in result
        assert "2 stories" in result
        assert "**Max**" in result
        assert "1 story)" in result

    def test_sorted_by_appearances(self, tmp_path: Path):
        """Test characters sorted by appearance count (most first)."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry = {
            "characters": {
                "Rare": {
                    "first_appeared": "s1",
                    "appearances": ["s1"],
                    "traits": [],
                },
                "Frequent": {
                    "first_appeared": "s1",
                    "appearances": ["s1", "s2", "s3"],
                    "traits": [],
                },
            },
            "last_updated": None,
        }
        cm._save_registry(registry)

        result = cm.format_registry_for_prompt()
        # Frequent should appear before Rare
        freq_pos = result.index("Frequent")
        rare_pos = result.index("Rare")
        assert freq_pos < rare_pos

    def test_budget_trimming(self, tmp_path: Path):
        """Test registry is trimmed to MAX_REGISTRY_TOKENS."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        # Create many characters to exceed budget
        characters: dict[str, Any] = {}
        for i in range(100):
            characters[f"Character_{i:03d}"] = {
                "first_appeared": "s1",
                "appearances": ["s1"],
                "traits": [f"Trait for character {i}." * 5],
            }
        registry = {"characters": characters, "last_updated": None}
        cm._save_registry(registry)

        result = cm.format_registry_for_prompt()
        tokens = cm._estimate_tokens(result)
        assert tokens <= cm.MAX_REGISTRY_TOKENS


class TestPopulateKnownCharacters:
    """Test _populate_known_characters method."""

    def test_populates_from_registry(self, tmp_path: Path):
        """Test characters populated from registry."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]
        registry = {
            "characters": {
                "Luna": {"first_appeared": "s1", "appearances": ["s1"], "traits": []},
                "Max": {"first_appeared": "s1", "appearances": ["s1"], "traits": []},
            },
            "last_updated": None,
        }
        cm._save_registry(registry)
        cm._populate_known_characters()
        assert cm._known_characters == {"Luna", "Max"}

    def test_fallback_to_metadata_scan(self, tmp_path: Path):
        """Test fallback to scanning metadata when no registry."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(context_dir)

        try:
            (context_dir / "story.md").write_text(
                "# Story Context\n**Characters:** Pip, Cleo\n\n## Story\n\nA short story."
            )
            cm = ContextManager()
            cm.get_context_directory = lambda: tmp_path / "nonexistent"  # type: ignore[assignment]  # No registry
            cm._populate_known_characters()
            assert "Pip" in cm._known_characters
            assert "Cleo" in cm._known_characters
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]

    def test_empty_when_no_sources(self, tmp_path: Path):
        """Test empty set when no registry and no files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.environ["STORYFORGE_TEST_CONTEXT_DIR"] = str(empty_dir)

        try:
            cm = ContextManager()
            cm.get_context_directory = lambda: tmp_path / "nonexistent"  # type: ignore[assignment]
            cm._populate_known_characters()
            assert cm._known_characters == set()
        finally:
            del os.environ["STORYFORGE_TEST_CONTEXT_DIR"]


class TestRegistryRoundtrip:
    """Integration tests for registry save/load cycle."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Test registry survives save/load cycle."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        registry = {
            "characters": {
                "Luna": {
                    "first_appeared": "story_001",
                    "appearances": ["story_001", "story_002"],
                    "traits": ["Luna is brave.", "Luna loves exploring."],
                }
            },
            "last_updated": None,
        }
        cm._save_registry(registry)
        loaded = cm._load_registry()

        assert loaded["characters"]["Luna"]["first_appeared"] == "story_001"
        assert loaded["characters"]["Luna"]["appearances"] == ["story_001", "story_002"]
        assert loaded["characters"]["Luna"]["traits"] == ["Luna is brave.", "Luna loves exploring."]

    def test_update_then_format(self, tmp_path: Path):
        """Test update followed by format produces expected output."""
        cm = ContextManager()
        cm.get_context_directory = lambda: tmp_path  # type: ignore[assignment]

        content = "## Story\n\nLuna explored the cave. Luna found gems. Luna was happy."
        cm.update_character_registry(content, {"characters": "Luna"}, "story_001")

        result = cm.format_registry_for_prompt()
        assert "## Known Characters" in result
        assert "**Luna**" in result
        assert "1 story)" in result
