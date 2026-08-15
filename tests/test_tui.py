"""Pilot tests for the standardized StoryForge Textual shell."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Checkbox, Input, OptionList, Select, TextArea

from storyforge.mcp_models import DraftResult, GenerationRequest, SessionSummary, StorySummary, WorkflowResult
from storyforge.tui import (
    HomeScreen,
    MediaScreen,
    NewStoryScreen,
    PickerScreen,
    ProgressScreen,
    ReviewScreen,
    StoryForgeApp,
)


class FakeClient:
    """Minimal async MCP facade for deterministic TUI tests."""

    def __init__(self) -> None:
        self.generation_request = None
        self.finalize_request = None
        self.config_data = {
            "values": {
                "story": {},
                "images": {},
                "output": {},
                "system": {},
            }
        }

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

    async def get_config(self):
        return self.config_data

    async def finalize_story(self, request):
        self.finalize_request = request
        return WorkflowResult(message="Complete")


@pytest.mark.asyncio
async def test_home_screen_exposes_all_primary_workflows():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        ids = {button.id for button in app.screen.query(Button)}
        assert ids == {"new", "continue", "extend", "export", "world", "config", "models"}


@pytest.mark.asyncio
async def test_home_panel_is_centered_in_wide_terminal():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        home = app.screen.query_one("#home")
        left_space = home.region.x
        right_space = app.screen.size.width - home.region.right

        assert abs(left_space - right_space) <= 1


@pytest.mark.asyncio
async def test_generate_route_prefills_prompt_and_opens_review():
    fake = FakeClient()
    fake.config_data["values"]["story"]["age_range"] = "preschool"
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
        assert fake.generation_request.age_range == "preschool"


@pytest.mark.asyncio
async def test_extend_route_uses_split_pane_picker():
    app = StoryForgeApp(route="extend", client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, PickerScreen)
        assert app.screen.query_one("#item-list", OptionList).option_count == 1
        assert "brave mouse" in str(app.screen.query_one("#preview-panel").content)


@pytest.mark.asyncio
async def test_new_story_form_loads_configured_defaults():
    fake = FakeClient()
    fake.config_data = {
        "values": {
            "story": {
                "age_range": "preschool",
                "length": "short",
                "style": "fantasy",
                "tone": "gentle",
                "theme": "kindness",
            },
            "images": {"image_style": "watercolor", "image_count": "4"},
            "output": {"output_dir": "configured-output", "use_context": "false"},
            "system": {},
        }
    }
    app = StoryForgeApp(client=fake)

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.query_one("#new", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, NewStoryScreen)
        assert app.screen.query_one("#age_range", Select).value == "preschool"
        assert app.screen.query_one("#length", Select).value == "short"
        assert app.screen.query_one("#style", Select).value == "fantasy"
        assert app.screen.query_one("#tone", Select).value == "gentle"
        assert app.screen.query_one("#theme", Select).value == "kindness"
        assert app.screen.query_one("#image_style", Select).value == "watercolor"
        assert app.screen.query_one("#output_dir", Input).value == "configured-output"
        assert app.screen.query_one("#use_context", Checkbox).value is False


@pytest.mark.asyncio
async def test_cancelling_workflow_leaves_progress_screen():
    app = StoryForgeApp(client=FakeClient())
    worker = MagicMock()

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(ProgressScreen("Working"))
        app._active_worker = worker
        app.cancel_active_workflow()
        await pilot.pause()

        worker.cancel.assert_called_once_with()
        assert isinstance(app.screen, HomeScreen)


@pytest.mark.asyncio
async def test_media_screen_rejects_out_of_range_video_count():
    fake = FakeClient()
    app = StoryForgeApp(client=fake)
    draft = DraftResult(
        session_id="session-test",
        status="active",
        story="Draft",
        output_directory="output",
        checkpoint_phase="story_save",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(MediaScreen(draft))
        app.screen.query_one("#video_count", Input).value = "21"
        app.screen.query_one("#finish", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, MediaScreen)
        assert fake.finalize_request is None
