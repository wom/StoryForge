"""Pilot tests for the standardized StoryForge Textual shell."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from textual.widgets import Button, Checkbox, Input, OptionList, Select, TextArea

from storyforge.mcp_models import (
    DraftResult,
    GeneratedStory,
    GeneratedStorySummary,
    GenerationRequest,
    SessionSummary,
    StorySummary,
    WorkflowResult,
)
from storyforge.tui import (
    HomeScreen,
    ImageViewerScreen,
    MediaScreen,
    NewStoryScreen,
    PickerScreen,
    ProgressScreen,
    ResultScreen,
    ReviewScreen,
    StoryBrowserScreen,
    StoryForgeApp,
    StoryReaderScreen,
    TerminalImage,
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
        self.generated_story = GeneratedStory(
            id="/tmp/storyforge_output_test/story.txt",
            title="The Lantern Fox",
            story_path="/tmp/storyforge_output_test/story.txt",
            generated_at="2026-08-14 12:00:00",
            preview="A fox carried a lantern home.",
            image_count=0,
            content="Story: The Lantern Fox\n\nA fox carried a lantern home.",
            output_directory="/tmp/storyforge_output_test",
        )

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

    async def list_generated_stories(self):
        return [
            GeneratedStorySummary(
                id=self.generated_story.id,
                title=self.generated_story.title,
                story_path=self.generated_story.story_path,
                generated_at=self.generated_story.generated_at,
                preview=self.generated_story.preview,
                image_count=self.generated_story.image_count,
            )
        ]

    async def get_generated_story(self, story_id):
        assert story_id == self.generated_story.id
        return self.generated_story

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
        assert ids == {"new", "stories", "continue", "extend", "export", "world", "config", "models"}


@pytest.mark.asyncio
async def test_home_panel_is_centered_in_wide_terminal():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        home = app.screen.query_one("#home-panel")
        left_space = home.region.x
        right_space = app.screen.size.width - home.region.right

        assert abs(left_space - right_space) <= 1


@pytest.mark.asyncio
async def test_story_browser_opens_generated_story_for_reading():
    fake = FakeClient()
    app = StoryForgeApp(client=fake)
    async with app.run_test(size=(150, 50)) as pilot:
        await pilot.pause()
        app.screen.query_one("#stories", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, StoryBrowserScreen)
        assert app.screen.query_one("#story-list", OptionList).option_count == 1
        app.screen.query_one("#read", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, StoryReaderScreen)
        assert app.screen.query_one("#library-story-text").content == fake.generated_story.content


@pytest.mark.asyncio
async def test_story_reader_opens_and_navigates_generated_images(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)
    story = FakeClient().generated_story.model_copy(
        update={"image_count": 2, "image_paths": [str(first), str(second)]}
    )
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))
        app.screen.query_one("#images", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, ImageViewerScreen)
        canvas = app.screen.query_one("#image-canvas", TerminalImage)
        assert canvas.image_path == str(first)
        app.screen.query_one("#next", Button).press()
        await pilot.pause()
        assert canvas.image_path == str(second)
        assert str(canvas.render())
        with patch("storyforge.tui.open_path_externally", return_value="Windows Explorer") as opener:
            app.screen.query_one("#open-external", Button).press()
            await pilot.pause()

        opener.assert_called_once_with(str(second))


@pytest.mark.asyncio
async def test_progress_panel_is_compact_and_centered_in_wide_terminal():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        await app.push_screen(ProgressScreen("Generating Draft"))
        await pilot.pause()
        panel = app.screen.query_one("#progress-panel")
        left_space = panel.region.x
        right_space = app.screen.size.width - panel.region.right

        assert abs(left_space - right_space) <= 1
        assert panel.region.width <= 72
        assert panel.region.height == 16


@pytest.mark.asyncio
async def test_review_screen_keeps_story_visible_and_actions_compact():
    story = "Once upon a test.\n\n[Square brackets are story text.]"
    draft = DraftResult(
        session_id="session-test",
        status="active",
        story=story,
        output_directory="output",
        checkpoint_phase="story_save",
    )
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test(size=(150, 50)) as pilot:
        await pilot.pause()
        await app.push_screen(ReviewScreen(draft))
        await pilot.pause()

        reader = app.screen.query_one("#story-reader")
        story_text = app.screen.query_one("#story-text")
        home_button = app.screen.query_one("#home", Button)

        assert reader.size.height > 0
        assert story_text.content == story
        assert home_button.region.height == 3
        assert home_button.region.width < reader.region.width


@pytest.mark.asyncio
async def test_owned_mcp_client_suppresses_subprocess_output():
    client = AsyncMock()
    client.__aenter__.return_value = client
    with patch("storyforge.tui.StoryForgeMCPClient", return_value=client) as client_type:
        app = StoryForgeApp()
        async with app.run_test() as pilot:
            await pilot.pause()

    client_type.assert_called_once_with(
        progress_callback=app._on_progress,
        suppress_server_output=True,
    )
    client.__aenter__.assert_awaited_once_with()
    client.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_owned_mcp_client_opens_and_closes_in_the_same_task():
    class TaskBoundClient:
        def __init__(self) -> None:
            self.enter_task = None
            self.exit_task = None

        async def __aenter__(self):
            self.enter_task = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            self.exit_task = asyncio.current_task()

    client = TaskBoundClient()
    with patch("storyforge.tui.StoryForgeMCPClient", return_value=client):
        app = StoryForgeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+q")
            await pilot.pause()

    assert client.enter_task is not None
    assert client.exit_task is client.enter_task


@pytest.mark.asyncio
async def test_owned_mcp_client_startup_failure_does_not_break_shutdown():
    client = AsyncMock()
    client.__aenter__.side_effect = RuntimeError("server did not start")
    with patch("storyforge.tui.StoryForgeMCPClient", return_value=client):
        app = StoryForgeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ResultScreen)
            assert "server did not start" in app.screen.message


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
