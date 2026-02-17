"""Tests for the interactive TUI story picker."""

from __future__ import annotations

import pytest

from storyforge.story_picker import StoryPicker, pick_story  # noqa: F401

SAMPLE_STORIES = [
    {
        "filepath": "/tmp/story1.md",
        "filename": "adventure_tale_20251022_123456",
        "timestamp": "2025-10-22 12:34:56",
        "characters": "Alice, Bob",
        "theme": "adventure",
        "prompt": "A story about exploring caves",
        "preview": "Once upon a time in a deep, dark cave, Alice and Bob discovered a glowing crystal...",
    },
    {
        "filepath": "/tmp/story2.md",
        "filename": "friendship_story_20251023_234567",
        "timestamp": "2025-10-23 23:45:67",
        "characters": "Charlie",
        "theme": "friendship",
        "preview": "In a magical forest, Charlie met a talking squirrel...",
    },
    {
        "filepath": "/tmp/story3.md",
        "filename": "courage_quest_20251024_111111",
        "timestamp": "2025-10-24 11:11:11",
        "theme": "courage",
        "preview": "The brave knight set out on a dangerous journey...",
    },
]


class TestStoryPickerApp:
    """Test the Textual StoryPicker app."""

    @pytest.mark.asyncio
    async def test_select_first_story_with_enter(self):
        """Test selecting the first (default) story by pressing Enter.

        Enter opens the full-story view; a second Enter confirms the selection.
        """
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("enter")   # open full story view
            await pilot.press("enter")   # confirm selection
        assert app.return_value == 0

    @pytest.mark.asyncio
    async def test_cancel_with_escape(self):
        """Test cancelling the picker with Escape."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("escape")
        assert app.return_value is None

    @pytest.mark.asyncio
    async def test_cancel_with_q(self):
        """Test cancelling the picker with q."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("q")
        assert app.return_value is None

    @pytest.mark.asyncio
    async def test_navigate_down_and_select(self):
        """Test navigating down one item and selecting."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("enter")  # full story view
            await pilot.press("enter")  # confirm
        assert app.return_value == 1

    @pytest.mark.asyncio
    async def test_navigate_to_last_and_select(self):
        """Test navigating to the last item and selecting."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")  # full story view
            await pilot.press("enter")  # confirm
        assert app.return_value == 2

    @pytest.mark.asyncio
    async def test_preview_updates_on_highlight(self):
        """Test that the preview panel updates when a new story is highlighted."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            from textual.widgets import Static

            # Initial preview should show first story
            preview = app.query_one("#preview-panel", Static)
            # Use .content to access the Static widget's text
            content = str(preview.content)
            assert "adventure_tale" in content

            # Navigate down - preview should show second story
            await pilot.press("down")
            content = str(preview.content)
            assert "friendship_story" in content

    @pytest.mark.asyncio
    async def test_preview_shows_metadata(self):
        """Test that the preview panel shows story metadata."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test():
            from textual.widgets import Static

            preview = app.query_one("#preview-panel", Static)
            content = str(preview.content)

            # Should show timestamp, characters, theme, and preview
            assert "2025-10-22" in content
            assert "Alice, Bob" in content
            assert "adventure" in content

    @pytest.mark.asyncio
    async def test_single_story_list(self):
        """Test picker with only one story."""
        single = [SAMPLE_STORIES[0]]
        app = StoryPicker(single)
        async with app.run_test() as pilot:
            await pilot.press("enter")  # full story view
            await pilot.press("enter")  # confirm
        assert app.return_value == 0

    @pytest.mark.asyncio
    async def test_option_list_has_all_stories(self):
        """Test that the option list contains all stories."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            from textual.widgets import OptionList

            option_list = app.query_one("#story-list", OptionList)
            assert option_list.option_count == 3
            await pilot.press("escape")


    @pytest.mark.asyncio
    async def test_full_story_view_escape_returns_to_picker(self):
        """Test that Escape from full-story view returns to the picker."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("enter")   # open full story view
            await pilot.press("escape")  # back to picker
            # Now press escape again to cancel entirely
            await pilot.press("escape")
        assert app.return_value is None

    @pytest.mark.asyncio
    async def test_full_story_view_escape_then_pick_different(self):
        """Test going back from full-story view and selecting a different story."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("enter")   # full story of item 0
            await pilot.press("escape")  # back to picker
            await pilot.press("down")    # move to item 1
            await pilot.press("enter")   # full story of item 1
            await pilot.press("enter")   # confirm item 1
        assert app.return_value == 1

    @pytest.mark.asyncio
    async def test_full_story_view_shows_content(self):
        """Test that the full-story view displays story text."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("enter")  # open full story
            from textual.widgets import Static

            content = str(app.query_one("#full-story-content", Static).content)
            # Should show the filename in the header
            assert "adventure_tale" in content
            # Since the file doesn't exist on disk, falls back to preview text
            assert "Once upon a time" in content or "no content" in content.lower()
            await pilot.press("escape")
            await pilot.press("escape")

    @pytest.mark.asyncio
    async def test_quit_from_full_story_view(self):
        """Test that 'q' exits the entire picker even from full-story view."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            await pilot.press("enter")  # full story view
            await pilot.press("q")      # quit entirely
        assert app.return_value is None

    @pytest.mark.asyncio
    async def test_preview_hint_text(self):
        """Test that the preview panel shows the 'Press Enter' hint."""
        app = StoryPicker(SAMPLE_STORIES)
        async with app.run_test() as pilot:
            from textual.widgets import Static

            preview = app.query_one("#preview-panel", Static)
            content = str(preview.content)
            assert "Enter" in content  # hint to press Enter
            await pilot.press("escape")


class TestPickStoryFunction:
    """Test the pick_story convenience function."""

    def test_pick_story_empty_list(self):
        """Test pick_story with an empty story list — should return None."""
        # With no stories, the app exits immediately (nothing to select)
        result = pick_story([])
        assert result is None
