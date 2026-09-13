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
    ConfigEditorScreen,
    ConfigScreen,
    ExtensionOptionsScreen,
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
            },
            "content": "[story]\nlength = short\n",
        }
        self.saved_config = None
        self.generated_story = GeneratedStory(
            id="/tmp/storyforge_output_test/story.txt",
            title="The Lantern Fox",
            story_path="/tmp/storyforge_output_test/story.txt",
            generated_at="2026-08-14 12:00:00",
            preview="A fox carried a lantern home.",
            image_count=0,
            context_id="story-one",
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

    async def write_config(self, content):
        self.saved_config = content
        return {"path": self.config_data.get("path")}

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
        assert ids == {"new", "stories", "continue", "export", "world", "config", "models"}


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
async def test_escape_from_home_exits_without_exposing_textual_root_screen():
    app = StoryForgeApp(client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert [type(screen).__name__ for screen in app.screen_stack] == ["HomeScreen"]

        await pilot.press("escape")
        await pilot.pause()

        assert app.is_running is False


@pytest.mark.asyncio
async def test_escape_from_direct_route_returns_home_then_exits():
    app = StoryForgeApp(route="stories", client=FakeClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, StoryBrowserScreen)
        assert [type(screen).__name__ for screen in app.screen_stack] == ["HomeScreen", "StoryBrowserScreen"]

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert app.is_running is False


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
        assert app.screen.query_one("#copy-story", Button).label == "Copy Story"
        assert app.screen.query_one("#extend", Button).label == "Extend"
        assert not app.screen.query("#copy-video-prompt")


@pytest.mark.asyncio
async def test_story_reader_extends_the_open_story_without_a_picker():
    story = FakeClient().generated_story
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))
        app.screen.query_one("#extend", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, ExtensionOptionsScreen)
        assert app.screen.story.id == story.context_id
        assert app.screen.story.filename == story.title


@pytest.mark.asyncio
async def test_story_reader_hides_extend_when_story_has_no_saved_context():
    story = FakeClient().generated_story.model_copy(update={"context_id": None})
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))

        assert not app.screen.query("#extend")


@pytest.mark.asyncio
async def test_story_reader_copies_portable_story_and_distinct_video_prompt():
    story = FakeClient().generated_story.model_copy(
        update={
            "content": "Léa’s story 😀",
            "video_prompt_content": "北京 — a cinematic scene",
        }
    )
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))
        with (
            patch("storyforge.tui.pyperclip.copy") as copy,
            patch(
                "storyforge.tui.pyperclip.paste",
                side_effect=["Lea's story :grinning:", "BeiJing - a cinematic scene"],
            ),
            patch("storyforge.tui.StoryReaderScreen.notify") as notify,
        ):
            app.screen.query_one("#copy-story", Button).press()
            await pilot.pause()
            app.screen.query_one("#copy-video-prompt", Button).press()
            await pilot.pause()

        assert [call.args[0] for call in copy.call_args_list] == [
            "Lea's story :grinning:",
            "BeiJing - a cinematic scene",
        ]
        assert [call.args[0] for call in notify.call_args_list] == [
            "Copied story to clipboard",
            "Copied video prompt to clipboard",
        ]


@pytest.mark.asyncio
async def test_story_reader_hides_video_prompt_when_portable_content_matches_story():
    story = FakeClient().generated_story.model_copy(update={"content": "Cafe", "video_prompt_content": "Café"})
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))

        assert not app.screen.query("#copy-video-prompt")


@pytest.mark.asyncio
async def test_story_reader_reports_actionable_clipboard_failure():
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(FakeClient().generated_story))
        with (
            patch("storyforge.tui.pyperclip.copy", side_effect=RuntimeError("clipboard service unavailable")),
            patch("storyforge.tui.StoryReaderScreen.notify") as notify,
        ):
            app.screen.query_one("#copy-story", Button).press()
            await pilot.pause()

        message = notify.call_args.args[0]
        assert "Could not copy story: clipboard service unavailable" in message
        assert "wl-clipboard" in message
        assert notify.call_args.kwargs == {"severity": "error"}


@pytest.mark.asyncio
async def test_story_reader_retries_until_clipboard_content_matches():
    app = StoryForgeApp(client=FakeClient())
    content = FakeClient().generated_story.content

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(FakeClient().generated_story))
        with (
            patch("storyforge.tui.pyperclip.copy") as copy,
            patch("storyforge.tui.pyperclip.paste", side_effect=["old", "still old", content]) as paste,
            patch("storyforge.tui.time.sleep") as sleep,
            patch("storyforge.tui.StoryReaderScreen.notify") as notify,
        ):
            app.screen.query_one("#copy-story", Button).press()
            await pilot.pause()

        copy.assert_called_once_with(content)
        assert paste.call_count == 3
        assert sleep.call_args_list == [((0.05,),), ((0.05,),)]
        notify.assert_called_once_with("Copied story to clipboard")


@pytest.mark.asyncio
async def test_story_reader_accepts_clipboard_newline_normalization():
    story = FakeClient().generated_story.model_copy(update={"content": "First line\nSecond line"})
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(story))
        with (
            patch("storyforge.tui.pyperclip.copy"),
            patch("storyforge.tui.pyperclip.paste", return_value="First line\r\nSecond line"),
            patch("storyforge.tui.StoryReaderScreen.notify") as notify,
        ):
            app.screen.query_one("#copy-story", Button).press()
            await pilot.pause()

        notify.assert_called_once_with("Copied story to clipboard")


@pytest.mark.asyncio
async def test_story_reader_reports_clipboard_verification_mismatch():
    app = StoryForgeApp(client=FakeClient())

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(StoryReaderScreen(FakeClient().generated_story))
        with (
            patch("storyforge.tui.pyperclip.copy"),
            patch("storyforge.tui.pyperclip.paste", return_value="different content") as paste,
            patch("storyforge.tui.time.sleep") as sleep,
            patch("storyforge.tui.StoryReaderScreen.notify") as notify,
        ):
            app.screen.query_one("#copy-story", Button).press()
            await pilot.pause()

        assert paste.call_count == 3
        assert sleep.call_count == 2
        assert "clipboard content did not match" in notify.call_args.args[0]
        assert notify.call_args.kwargs == {"severity": "error"}


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
async def test_new_story_form_keeps_field_rows_compact():
    app = StoryForgeApp(
        route="generate",
        initial_request=GenerationRequest(prompt="A compact form"),
        client=FakeClient(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        rows = list(app.screen.query(".field-row"))
        assert len(rows) == 6
        assert {row.region.height for row in rows} == {4}
        assert app.screen.query_one("#prompt", TextArea).region.height == 5


@pytest.mark.asyncio
async def test_configuration_screen_formats_grouped_values_and_defaults():
    fake = FakeClient()
    fake.config_data = {
        "values": {
            "story": {"length": "short", "age_range": "middle_grade", "voice": ""},
            "images": {"image_style": "watercolor", "image_count": "3"},
            "output": {"use_context": "true", "output_dir": ""},
            "system": {"backend": "", "debug": "false"},
        },
        "path": None,
    }
    app = StoryForgeApp(client=fake)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#config", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, ConfigScreen)
        summary = str(app.screen.query_one("#config-summary").content)
        assert "Story\n" in summary
        assert "Age range" in summary
        assert "Middle Grade" in summary
        assert "Voice                 Not set" in summary
        assert "Use context           Enabled" in summary
        assert "Debug                 Disabled" in summary
        assert app.screen.query_one("#config-path").content == "Using built-in defaults · no configuration file"
        assert app.screen.query_one("#create-config", Button).label == "Create Config"
        assert not app.screen.query("#edit-config")


@pytest.mark.asyncio
async def test_configuration_screen_opens_existing_config_in_app_for_editing():
    fake = FakeClient()
    fake.config_data["path"] = "/tmp/storyforge.ini"
    app = StoryForgeApp(client=fake)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(ConfigScreen(fake.config_data))
        app.screen.query_one("#edit-config", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, ConfigEditorScreen)
        assert app.screen.query_one("#config-content", TextArea).text == "[story]\nlength = short\n"

        app.screen.query_one("#config-content", TextArea).text = "[story]\nlength = long\n"
        app.screen.query_one("#save", Button).press()
        await pilot.pause()

        assert fake.saved_config == "[story]\nlength = long\n"
        assert isinstance(app.screen, ResultScreen)
        assert app.screen.result_title == "Configuration Saved"


@pytest.mark.asyncio
async def test_failed_config_save_returns_to_editor_with_unsaved_content():
    fake = FakeClient()
    fake.config_data["path"] = "/tmp/storyforge.ini"
    fake.write_config = AsyncMock(side_effect=RuntimeError("Invalid story length"))
    app = StoryForgeApp(client=fake)
    invalid_content = "[story]\nlength = invalid\n"

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(ConfigScreen(fake.config_data))
        app.screen.query_one("#edit-config", Button).press()
        await pilot.pause()
        app.screen.query_one("#config-content", TextArea).text = invalid_content

        with patch.object(app, "notify") as notify:
            app.screen.query_one("#save", Button).press()
            await pilot.pause()

        assert isinstance(app.screen, ConfigEditorScreen)
        assert app.screen.query_one("#config-content", TextArea).text == invalid_content
        notify.assert_called_once_with(
            "Could not save configuration: Invalid story length",
            severity="error",
        )


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
