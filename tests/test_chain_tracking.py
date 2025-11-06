"""Tests for story chain tracking functionality."""

from pathlib import Path

import pytest

from storyforge.context import ContextManager


@pytest.fixture
def context_dir(tmp_path):
    """Create a temporary context directory."""
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir(parents=True)
    return ctx_dir


@pytest.fixture
def context_manager(context_dir, monkeypatch):
    """Create a ContextManager with a temporary context directory."""

    # Mock the context directory
    def mock_get_context_directory(self):
        return context_dir

    monkeypatch.setattr(ContextManager, "get_context_directory", mock_get_context_directory)
    return ContextManager()


def create_context_file(context_dir: Path, filename: str, prompt: str, extended_from: str | None = None) -> Path:
    """Helper to create a context file with metadata."""
    filepath = context_dir / filename
    content = f"""# Story Context: {prompt}

**Generated on:** 2025-11-05 12:00:00

**Original Prompt:** {prompt}

"""
    if extended_from:
        content += f"**Extended From:** {extended_from}\n\n"

    content += """**Characters:** TestChar

**Setting:** TestWorld

## Story

This is a test story about {prompt}.

## Refinements Applied

None
"""
    filepath.write_text(content)
    return filepath


def test_get_story_chain_single_story(context_manager, context_dir):
    """Test chain reconstruction for non-extended story."""
    # Create single story context file
    filepath = create_context_file(context_dir, "original_story.md", "A test story")

    # Get chain
    chain = context_manager.get_story_chain(filepath)

    # Verify chain contains only one story
    assert len(chain) == 1
    assert chain[0]["filename"] == "original_story"  # stem, not .md
    assert chain[0]["prompt"] == "A test story"
    assert "extended_from" not in chain[0]


def test_get_story_chain_two_parts(context_manager, context_dir):
    """Test chain reconstruction for simple extension (2 parts)."""
    # Create original story
    create_context_file(context_dir, "wizard_20251105_120000.md", "A wizard discovers an artifact")

    # Create extended story
    extended = create_context_file(
        context_dir,
        "wizard_20251105_120000_20251105_130000.md",
        "A wizard discovers an artifact",
        "wizard_20251105_120000",
    )

    # Get chain from extended story
    chain = context_manager.get_story_chain(extended)

    # Verify chain contains both stories in correct order
    assert len(chain) == 2
    assert chain[0]["filename"] == "wizard_20251105_120000"
    assert chain[1]["filename"] == "wizard_20251105_120000_20251105_130000"
    assert "extended_from" not in chain[0]
    assert chain[1]["extended_from"] == "wizard_20251105_120000"


def test_get_story_chain_multiple_extensions(context_manager, context_dir):
    """Test chain reconstruction for multi-level extensions (3+ parts)."""
    # Create chain: original -> extended -> extended again
    create_context_file(context_dir, "story_001.md", "Original story")
    create_context_file(context_dir, "story_001_002.md", "Original story continued", "story_001")
    extended2 = create_context_file(context_dir, "story_001_002_003.md", "Original story finale", "story_001_002")

    # Get chain from final story
    chain = context_manager.get_story_chain(extended2)

    # Verify all three stories in correct order
    assert len(chain) == 3
    assert chain[0]["filename"] == "story_001"
    assert chain[1]["filename"] == "story_001_002"
    assert chain[2]["filename"] == "story_001_002_003"
    assert "extended_from" not in chain[0]
    assert chain[1]["extended_from"] == "story_001"
    assert chain[2]["extended_from"] == "story_001_002"


def test_get_story_chain_missing_parent(context_manager, context_dir):
    """Test chain reconstruction when parent file is deleted."""
    # Create extended story with parent reference but don't create the parent
    extended = create_context_file(context_dir, "story_extended.md", "Extended story", "story_original")

    # Get chain
    chain = context_manager.get_story_chain(extended)

    # Verify chain contains only the child story (breaks at missing parent)
    assert len(chain) == 1
    assert chain[0]["filename"] == "story_extended"
    assert chain[0]["extended_from"] == "story_original"


def test_get_story_chain_circular_reference(context_manager, context_dir):
    """Test chain reconstruction handles circular references."""
    # This is a bit contrived but tests the safety mechanism
    # Create story A that references B
    story_a = create_context_file(context_dir, "story_a.md", "Story A", "story_b")
    # Create story B that references A (circular!)
    create_context_file(context_dir, "story_b.md", "Story B", "story_a")

    # Get chain from story A
    chain = context_manager.get_story_chain(story_a)

    # Verify chain terminates without infinite loop
    # Should get A -> B -> (detects A is already seen, stops)
    assert len(chain) <= 2
    assert chain[0]["filename"] == "story_b"  # B is the parent, so it comes first
    assert chain[1]["filename"] == "story_a"


def test_write_chain_to_file_single_story(context_manager, context_dir, tmp_path):
    """Test exporting a single story (no extensions)."""
    # Create single story
    filepath = create_context_file(context_dir, "single_story.md", "Standalone tale")

    # Export to file
    output_path = tmp_path / "exported.txt"
    result = context_manager.write_chain_to_file(filepath, output_path)

    # Verify file was created
    assert result == output_path
    assert output_path.exists()

    # Verify content
    content = output_path.read_text()
    assert "COMPLETE STORY CHAIN" in content
    assert "Total stories in chain: 1" in content
    assert "PART 1 of 1" in content
    assert "Original prompt: Standalone tale" in content
    assert "END OF STORY CHAIN" in content


def test_write_chain_to_file_multiple_stories(context_manager, context_dir, tmp_path):
    """Test exporting a multi-part chain."""
    # Create chain with 3 parts
    create_context_file(context_dir, "part1.md", "Beginning")
    create_context_file(context_dir, "part2.md", "Middle", "part1")
    part3 = create_context_file(context_dir, "part3.md", "End", "part2")

    # Export
    output_path = tmp_path / "complete.txt"
    result = context_manager.write_chain_to_file(part3, output_path)

    # Verify file
    assert result.exists()
    content = result.read_text()

    # Verify all parts are included in chronological order
    assert "Total stories in chain: 3" in content
    assert "PART 1 of 3" in content
    assert "PART 2 of 3" in content
    assert "PART 3 of 3" in content

    # Verify separators and metadata
    assert content.count("=" * 80) >= 8  # Header + footer + 3 part headers
    assert "Source: part1" in content  # filename is stem, no .md
    assert "Source: part2" in content
    assert "Source: part3" in content
    assert "Extended from: part1" in content
    assert "Extended from: part2" in content


def test_write_chain_to_file_creates_directory(context_manager, context_dir, tmp_path):
    """Test that export creates output directory if needed."""
    # Create story
    filepath = create_context_file(context_dir, "story.md", "Test")

    # Export to non-existent directory
    output_path = tmp_path / "subdir" / "nested" / "output.txt"
    result = context_manager.write_chain_to_file(filepath, output_path)

    # Verify directory was created
    assert result.parent.exists()
    assert result.exists()


def test_write_chain_to_file_empty_chain(context_manager, tmp_path):
    """Test export with invalid/missing context."""
    # Attempt to export non-existent file
    fake_path = tmp_path / "nonexistent.md"
    output_path = tmp_path / "output.txt"

    # Should raise ValueError when no stories found
    with pytest.raises(ValueError, match="No stories found in chain"):
        context_manager.write_chain_to_file(fake_path, output_path)


def test_parse_context_metadata_with_extended_from(context_manager, context_dir):
    """Test that parse_context_metadata extracts extended_from field."""
    # Create extended story
    filepath = create_context_file(context_dir, "extended.md", "Extended story", "parent_story")

    # Parse metadata
    metadata = context_manager.parse_context_metadata(filepath)

    # Verify extended_from is extracted
    assert "extended_from" in metadata
    assert metadata["extended_from"] == "parent_story"


def test_parse_context_metadata_without_extended_from(context_manager, context_dir):
    """Test that parse_context_metadata handles missing extended_from."""
    # Create original story (no parent)
    filepath = create_context_file(context_dir, "original.md", "Original story")

    # Parse metadata
    metadata = context_manager.parse_context_metadata(filepath)

    # Verify extended_from is not present
    assert "extended_from" not in metadata


def test_chain_with_long_names(context_manager, context_dir):
    """Test chain tracking with very long filenames."""
    # Create chain with long names
    name1 = "wizard_discovers_magical_artifact_20251105_120000"
    name2 = f"{name1}_20251105_130000"
    name3 = f"{name2}_20251105_140000"

    create_context_file(context_dir, f"{name1}.md", "Wizard story")
    create_context_file(context_dir, f"{name2}.md", "Wizard story", name1)
    final = create_context_file(context_dir, f"{name3}.md", "Wizard story", name2)

    # Get chain
    chain = context_manager.get_story_chain(final)

    # Verify all parts found despite long names
    assert len(chain) == 3
    assert chain[0]["filename"] == name1
    assert chain[1]["filename"] == name2
    assert chain[2]["filename"] == name3
