"""Pilot tests for the standardized StoryForge Textual shell."""

from __future__ import annotations

import pytest
from textual.widgets import Button, OptionList, TextArea

from storyforge.mcp_models import DraftResult, GenerationRequest, SessionSummary, StorySummary
from storyforge.tui import HomeScreen, PickerScreen, ReviewScreen, StoryForgeApp


class FakeClient:
    """Minimal async MCP facade for deterministic TUI tests."""

    def __init__(self) -> None:
        self.generation_request = None

    async def create_draft(self, request):
        self.generation_request = request
        return DraftResult(
            session_id="session-test",
            status="active",
            story="A generated test story.",
            output_directory="output",
            checkpoint_phase="story_save",
        )

    async def list_stories(self, chain_only=False):
        if chain_only:
            return []
        return [
            StorySummary(
                id="story-one",
                filename="story-one",
                filepath="/tmp/story-one.md",
                preview="A brave mouse found a map.",
            )
        ]

    async def list_sessions(self, limit=15):
        return [
            SessionSummary(
                session_id="session-test",
                status="active",
                current_phase="story_save",
                prompt_preview="A brave mouse",
            )
        ]


@pytest.mark.asyncio
async def test_home_screen_exposes_all_primary_workflows():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        ids = {button.id for button in app.screen.query(Button)}
        assert ids == {"new", "continue", "extend", "export", "world", "config", "models"}


@pytest.mark.asyncio
async def test_generate_route_prefills_prompt_and_opens_review():
    fake = FakeClient()
    app = StoryForgeApp(
        route="generate",
        initial_request=GenerationRequest(prompt="A prefilled brave mouse"),
        client=fake,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#prompt", TextArea).text == "A prefilled brave mouse"
        app.screen.query_one("#generate", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        assert fake.generation_request.prompt == "A prefilled brave mouse"


@pytest.mark.asyncio
async def test_extend_route_uses_split_pane_picker():
    app = StoryForgeApp(route="extend", client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, PickerScreen)
        assert app.screen.query_one("#item-list", OptionList).option_count == 1
        assert "brave mouse" in str(app.screen.query_one("#preview-panel").content)
