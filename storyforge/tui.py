"""Unified Textual interface for StoryForge MCP workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image, UnidentifiedImageError
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.worker import get_current_worker

from .external_viewer import open_path_externally
from .mcp_client import StoryForgeMCPClient
from .mcp_models import (
    DraftResult,
    ExportRequest,
    ExtensionRequest,
    FinalizeRequest,
    GeneratedStory,
    GeneratedStorySummary,
    GenerationRequest,
    RefinementRequest,
    SessionSummary,
    StorySummary,
    WorkflowResult,
)


def _request_from_config(
    data: dict[str, Any],
    initial: GenerationRequest | None = None,
) -> GenerationRequest:
    """Build a blank form request from the MCP configuration resource."""
    values = data.get("values", data)
    story = values.get("story", {})
    images = values.get("images", {})
    output = values.get("output", {})
    system = values.get("system", {})

    characters_value = story.get("characters")
    if isinstance(characters_value, str):
        characters = [item.strip() for item in characters_value.split(",") if item.strip()] or None
    else:
        characters = list(characters_value or []) or None

    use_context_value = output.get("use_context", True)
    use_context = (
        use_context_value
        if isinstance(use_context_value, bool)
        else str(use_context_value).strip().lower() in {"1", "true", "yes", "on"}
    )
    image_count_value = images.get("image_count")

    configured = GenerationRequest(
        prompt=" ",
        length=story.get("length") or None,
        age_range=story.get("age_range") or None,
        style=story.get("style") or None,
        tone=story.get("tone") or None,
        voice=story.get("voice") or None,
        theme=story.get("theme") or None,
        learning_focus=story.get("learning_focus") or None,
        setting=story.get("setting") or None,
        characters=characters,
        image_style=images.get("image_style") or None,
        image_count=int(image_count_value) if image_count_value not in {None, ""} else None,
        output_dir=output.get("output_dir") or None,
        use_context=use_context,
        world_file=output.get("world_file") or None,
        backend=system.get("backend") or None,
        verbose=str(system.get("verbose", "false")).lower() in {"1", "true", "yes", "on"},
        debug=str(system.get("debug", "false")).lower() in {"1", "true", "yes", "on"},
    )
    if initial is None:
        return configured

    overrides = {
        field: value
        for field, value in initial.model_dump().items()
        if value is not None and (field not in {"verbose", "debug"} or value is True)
    }
    return configured.model_copy(update=overrides)


class StoryForgeScreen(Screen[None]):
    """Base screen with consistent shell and navigation."""

    BINDINGS = [Binding("escape", "back", "Back", show=True)]

    @property
    def storyforge_app(self) -> StoryForgeApp:
        """Narrow Textual's application type for screen event handlers."""
        return cast("StoryForgeApp", super().app)

    def action_back(self) -> None:
        app = self.storyforge_app
        if len(app.screen_stack) > 1:
            app.pop_screen()
        else:
            app.exit()


class HomeScreen(StoryForgeScreen):
    """StoryForge workflow launcher."""

    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="home-panel"):
            yield Static("[bold cyan]StoryForge[/bold cyan]", id="brand")
            yield Static("Create and continue illustrated stories", classes="subtitle")
            with Horizontal(classes="home-row"):
                yield Button("New Story", id="new", variant="primary")
                yield Button("Stories", id="stories")
                yield Button("Continue", id="continue")
                yield Button("Extend", id="extend")
            with Horizontal(classes="home-row"):
                yield Button("Export Chain", id="export")
                yield Button("World", id="world")
                yield Button("Configuration", id="config")
                yield Button("Models", id="models")
        yield Footer()

    def action_quit(self) -> None:
        self.storyforge_app.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.storyforge_app
        routes: dict[str, Callable[[], Any]] = {
            "new": app.show_new_story,
            "stories": app.show_stories,
            "continue": app.show_continue,
            "extend": app.show_extend,
            "export": app.show_export,
            "world": app.show_world,
            "config": app.show_config,
            "models": app.show_models,
        }
        action = routes.get(event.button.id or "")
        if action is not None:
            action()


class NewStoryScreen(StoryForgeScreen):
    """Schema-aligned form for a new story."""

    def __init__(self, initial: GenerationRequest | None = None) -> None:
        super().__init__()
        self.initial = initial or GenerationRequest(prompt=" ")

    @staticmethod
    def _options(values: list[str]) -> list[tuple[str, str]]:
        return [((value.replace("_", " ").title() if value else "None"), value) for value in values]

    def compose(self) -> ComposeResult:
        initial = self.initial
        yield Header()
        with VerticalScroll(id="form"):
            yield Label("New Story", classes="screen-title")
            yield Label("Prompt")
            yield TextArea(initial.prompt.strip(), id="prompt")
            with Horizontal(classes="field-row"):
                with Vertical(classes="field"):
                    yield Label("Age range")
                    yield Select(
                        self._options(["toddler", "preschool", "early_reader", "middle_grade"]),
                        value=initial.age_range or "early_reader",
                        id="age_range",
                    )
                with Vertical(classes="field"):
                    yield Label("Length")
                    yield Select(
                        self._options(["flash", "short", "medium", "bedtime"]),
                        value=initial.length or "bedtime",
                        id="length",
                    )
            with Horizontal(classes="field-row"):
                with Vertical(classes="field"):
                    yield Label("Style")
                    yield Select(
                        self._options(["adventure", "comedy", "fantasy", "fairy_tale", "friendship", "random"]),
                        value=initial.style or "random",
                        id="style",
                    )
                with Vertical(classes="field"):
                    yield Label("Tone")
                    yield Select(
                        self._options(["gentle", "exciting", "silly", "heartwarming", "magical", "random"]),
                        value=initial.tone or "random",
                        id="tone",
                    )
            yield Label("Characters (comma-separated)")
            yield Input(value=", ".join(initial.characters or []), id="characters")
            yield Label("Setting")
            yield Input(value=initial.setting or "", id="setting")
            with Horizontal(classes="field-row"):
                with Vertical(classes="field"):
                    yield Label("Theme")
                    yield Select(
                        self._options(["courage", "kindness", "teamwork", "problem_solving", "creativity", "random"]),
                        value=initial.theme or "random",
                        id="theme",
                    )
                with Vertical(classes="field"):
                    yield Label("Voice")
                    yield Select(
                        self._options(
                            [
                                "",
                                "anapestic",
                                "sardonic",
                                "picaresque",
                                "iambic",
                                "fable",
                                "gothic",
                                "nonsense",
                                "lyrical",
                                "epistolary",
                                "random",
                            ]
                        ),
                        value=initial.voice or "",
                        id="voice",
                    )
            with Horizontal(classes="field-row"):
                with Vertical(classes="field"):
                    yield Label("Learning focus")
                    yield Select(
                        self._options(["", "counting", "colors", "letters", "emotions", "nature"]),
                        value=initial.learning_focus or "",
                        id="learning_focus",
                    )
                with Vertical(classes="field"):
                    yield Label("Image style")
                    yield Select(
                        self._options(["chibi", "realistic", "cartoon", "watercolor", "sketch"]),
                        value=initial.image_style or "chibi",
                        id="image_style",
                    )
            yield Label("Backend (leave blank for automatic selection)")
            yield Input(value=initial.backend or "", id="backend")
            yield Label("Output directory (leave blank for automatic name)")
            yield Input(value=initial.output_dir or "", id="output_dir")
            yield Label("World file override")
            yield Input(value=initial.world_file or "", id="world_file")
            yield Checkbox("Use saved story context", value=initial.use_context is not False, id="use_context")
            with Horizontal(classes="actions"):
                yield Button("Generate Draft", id="generate", variant="success")
                yield Button("Cancel", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.action_back()
            return
        if event.button.id != "generate":
            return
        prompt = self.query_one("#prompt", TextArea).text.strip()
        if not prompt:
            self.notify("Enter a story prompt", severity="error")
            return
        characters = [value.strip() for value in self.query_one("#characters", Input).value.split(",")]
        request = GenerationRequest(
            prompt=prompt,
            age_range=str(self.query_one("#age_range", Select).value),
            length=str(self.query_one("#length", Select).value),
            style=str(self.query_one("#style", Select).value),
            tone=str(self.query_one("#tone", Select).value),
            voice=str(self.query_one("#voice", Select).value) or None,
            theme=str(self.query_one("#theme", Select).value),
            learning_focus=str(self.query_one("#learning_focus", Select).value) or None,
            characters=[value for value in characters if value] or None,
            setting=self.query_one("#setting", Input).value.strip() or None,
            backend=self.query_one("#backend", Input).value.strip() or None,
            image_style=str(self.query_one("#image_style", Select).value),
            image_count=self.initial.image_count,
            output_dir=self.query_one("#output_dir", Input).value.strip() or None,
            use_context=self.query_one("#use_context", Checkbox).value,
            world_file=self.query_one("#world_file", Input).value.strip() or None,
            verbose=self.initial.verbose,
            debug=self.initial.debug,
        )
        self.storyforge_app.create_draft(request)


class ProgressScreen(StoryForgeScreen):
    """Responsive progress view fed by MCP notifications."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, title: str) -> None:
        super().__init__()
        self.progress_title = title

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="progress-panel"):
            yield Static(f"[bold cyan]{self.progress_title}[/bold cyan]", classes="screen-title")
            yield LoadingIndicator()
            yield Static("Starting StoryForge MCP workflow…", id="progress-message")
            yield Static("", id="progress-value", classes="subtitle")
        yield Footer()

    def update_progress(self, progress: float, total: float | None, message: str | None) -> None:
        if message:
            self.query_one("#progress-message", Static).update(message.replace("_", " ").title())
        if total:
            self.query_one("#progress-value", Static).update(f"{round(progress / total * 100)}%")

    def action_cancel(self) -> None:
        self.storyforge_app.cancel_active_workflow()


class ReviewScreen(StoryForgeScreen):
    """Story reading and refinement screen."""

    def __init__(self, draft: DraftResult) -> None:
        super().__init__()
        self.draft = draft

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="review"):
            yield Static("[bold cyan]Review Story[/bold cyan]", classes="screen-title")
            yield VerticalScroll(
                Static(self.draft.story, id="story-text", markup=False),
                id="story-reader",
            )
            yield Input(placeholder="Optional refinement instructions", id="refinement")
            with Horizontal(classes="actions"):
                yield Button("Accept", id="accept", variant="success")
                yield Button("Refine", id="refine", variant="warning")
                yield Button("Home", id="home")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "accept":
            self.storyforge_app.push_screen(MediaScreen(self.draft))
        elif event.button.id == "refine":
            instructions = self.query_one("#refinement", Input).value.strip()
            if not instructions:
                self.notify("Enter refinement instructions", severity="error")
                return
            self.storyforge_app.refine_draft(self.draft.session_id, instructions)
        elif event.button.id == "home":
            self.storyforge_app.go_home()


class MediaScreen(StoryForgeScreen):
    """Final media and context decisions."""

    def __init__(self, draft: DraftResult) -> None:
        super().__init__()
        self.draft = draft

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="media-form"):
            yield Static("[bold cyan]Finish Story[/bold cyan]", classes="screen-title")
            yield Label("Video prompt scenes (0 to skip)")
            yield Input(value="0", type="integer", id="video_count")
            yield Label("Illustrations (0 to skip, maximum 5)")
            yield Input(
                value=str(self.draft.metadata.get("image_count", 0) or 0),
                type="integer",
                id="image_count",
            )
            yield Checkbox("Save as future story context", id="save_context")
            with Horizontal(classes="actions"):
                yield Button("Finish", id="finish", variant="success")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
            return
        if event.button.id != "finish":
            return
        try:
            video = int(self.query_one("#video_count", Input).value or "0")
            images = int(self.query_one("#image_count", Input).value or "0")
        except ValueError:
            self.notify("Media counts must be whole numbers", severity="error")
            return
        if not 0 <= video <= 20:
            self.notify("Video scene count must be between 0 and 20", severity="error")
            return
        if not 0 <= images <= 5:
            self.notify("Image count must be between 0 and 5", severity="error")
            return
        self.storyforge_app.finalize_story(
            FinalizeRequest(
                session_id=self.draft.session_id,
                video_scene_count=video,
                image_count=images,
                save_context=self.query_one("#save_context", Checkbox).value,
            )
        )


PickerKind = Literal["extend", "export", "continue"]


class PickerScreen(StoryForgeScreen):
    """Reusable split-pane picker based on the original sf-extend UI."""

    def __init__(
        self,
        title: str,
        kind: PickerKind,
        items: list[StorySummary] | list[SessionSummary],
    ) -> None:
        super().__init__()
        self.title = title
        self.kind = kind
        self.items = items

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="picker-layout"):
            labels = []
            for item in self.items:
                if isinstance(item, StorySummary):
                    suffix = f" · {item.chain_length} parts" if item.chain_length > 1 else ""
                    labels.append(f"{item.filename}{suffix}")
                else:
                    labels.append(f"{item.prompt_preview} · {item.status}")
            yield OptionList(*labels, id="item-list")
            yield Static("", id="preview-panel")
        yield Footer()

    def on_mount(self) -> None:
        if self.items:
            self._update_preview(0)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._update_preview(event.option_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        item = self.items[event.option_index]
        if self.kind == "extend" and isinstance(item, StorySummary):
            self.storyforge_app.push_screen(ExtensionOptionsScreen(item))
        elif self.kind == "export" and isinstance(item, StorySummary):
            self.storyforge_app.push_screen(ExportOptionsScreen(item))
        elif self.kind == "continue" and isinstance(item, SessionSummary):
            self.storyforge_app.resume_session(item.session_id)

    def _update_preview(self, index: int) -> None:
        item = self.items[index]
        if isinstance(item, StorySummary):
            text = (
                f"[bold cyan]{item.filename}[/bold cyan]\n\n"
                f"[bold]Generated:[/bold] {item.timestamp or 'Unknown'}\n"
                f"[bold]Chain:[/bold] {item.chain_length} part(s)\n"
                f"[bold]Characters:[/bold] {item.characters or '—'}\n"
                f"[bold]Theme:[/bold] {item.theme or '—'}\n\n"
                f"[bold]Preview[/bold]\n{item.preview or 'No preview available.'}\n\n"
                "[dim]Press Enter to continue[/dim]"
            )
        else:
            text = (
                f"[bold cyan]{item.prompt_preview}[/bold cyan]\n\n"
                f"[bold]Created:[/bold] {item.created_at}\n"
                f"[bold]Status:[/bold] {item.status}\n"
                f"[bold]Phase:[/bold] {item.current_phase}\n"
                f"[bold]Progress:[/bold] {item.completion_percentage}%\n\n"
                "[dim]Press Enter to resume[/dim]"
            )
        self.query_one("#preview-panel", Static).update(text)


class StoryBrowserScreen(StoryForgeScreen):
    """Browse every generated story artifact and open it for reading."""

    def __init__(self, stories: list[GeneratedStorySummary]) -> None:
        super().__init__()
        self.stories = stories

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="story-browser"):
            yield Static("Story Library", classes="screen-title literal-title", markup=False)
            with Horizontal(id="story-browser-layout"):
                labels = [
                    Text(f"{story.title} · {story.image_count} image{'s' if story.image_count != 1 else ''}")
                    for story in self.stories
                ]
                yield OptionList(*labels, id="story-list")
                yield Static("", id="story-preview", markup=False)
            with Horizontal(classes="actions"):
                yield Button("Read", id="read", variant="primary")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        if self.stories:
            self._update_preview(0)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._update_preview(event.option_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.storyforge_app.open_generated_story(self.stories[event.option_index].id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "read":
            option_list = self.query_one("#story-list", OptionList)
            if option_list.highlighted is not None:
                self.storyforge_app.open_generated_story(self.stories[option_list.highlighted].id)

    def _update_preview(self, index: int) -> None:
        story = self.stories[index]
        image_label = f"{story.image_count} image{'s' if story.image_count != 1 else ''}"
        self.query_one("#story-preview", Static).update(
            f"{story.title}\n\nGenerated: {story.generated_at or 'Unknown'}\n"
            f"Images: {image_label}\n\n{story.preview or 'No preview available.'}\n\n"
            "Press Enter or choose Read to open"
        )


class StoryReaderScreen(StoryForgeScreen):
    """Distraction-free reader for one generated story."""

    def __init__(self, story: GeneratedStory) -> None:
        super().__init__()
        self.story = story

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="story-viewer"):
            yield Static(self.story.title, classes="screen-title literal-title", markup=False)
            details = self.story.generated_at or "Generation time unknown"
            if self.story.image_paths:
                details += f" · {len(self.story.image_paths)} image{'s' if len(self.story.image_paths) != 1 else ''}"
            yield Static(details, classes="subtitle", markup=False)
            yield VerticalScroll(
                Static(self.story.content, id="library-story-text", markup=False),
                id="library-story-reader",
            )
            with Horizontal(classes="actions"):
                if self.story.image_paths:
                    yield Button(f"View Images ({len(self.story.image_paths)})", id="images", variant="primary")
                yield Button("Back", id="back")
                yield Button("Home", id="home")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "images":
            self.storyforge_app.push_screen(ImageViewerScreen(self.story.image_paths))
        elif event.button.id == "back":
            self.action_back()
        elif event.button.id == "home":
            self.storyforge_app.go_home()


class TerminalImage(Static):
    """Render a local raster image with true-color half-block terminal cells."""

    def __init__(self, image_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.image_path = image_path
        self._cache_key: tuple[str, int, int] | None = None
        self._cached_render = Text()

    def set_image(self, image_path: str) -> None:
        self.image_path = image_path
        self._cache_key = None
        self.refresh()

    def render(self) -> Text:
        width = max(1, self.size.width)
        height = max(1, self.size.height)
        cache_key = (self.image_path, width, height)
        if cache_key == self._cache_key:
            return self._cached_render

        try:
            with Image.open(self.image_path) as source:
                image = source.convert("RGB")
                image.thumbnail((width, height * 2), Image.Resampling.LANCZOS)
                rendered = self._render_half_blocks(image)
        except (OSError, UnidentifiedImageError):
            rendered = Text("This image could not be displayed.", justify="center", style="red")

        self._cache_key = cache_key
        self._cached_render = rendered
        return rendered

    @staticmethod
    def _render_half_blocks(image: Image.Image) -> Text:
        rendered = Text(justify="center", no_wrap=True, overflow="crop")
        pixels = image.load()
        assert pixels is not None
        for y in range(0, image.height, 2):
            bottom_y = min(y + 1, image.height - 1)
            for x in range(image.width):
                top = cast(tuple[int, int, int], pixels[x, y])
                bottom = cast(tuple[int, int, int], pixels[x, bottom_y])
                rendered.append(
                    "▀",
                    Style(
                        color=Color.from_rgb(*top),
                        bgcolor=Color.from_rgb(*bottom),
                    ),
                )
            if y + 2 < image.height:
                rendered.append("\n")
        return rendered


class ImageViewerScreen(StoryForgeScreen):
    """Navigate generated illustrations without leaving StoryForge."""

    BINDINGS = [
        Binding("left", "previous_image", "Previous", show=True),
        Binding("right", "next_image", "Next", show=True),
        Binding("o", "open_external", "Open External", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self, image_paths: list[str]) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.image_index = 0

    def compose(self) -> ComposeResult:
        current = Path(self.image_paths[self.image_index])
        yield Header()
        with Vertical(id="image-viewer"):
            yield Static(current.name, id="image-title", classes="screen-title literal-title", markup=False)
            yield TerminalImage(str(current), id="image-canvas")
            yield Static(self._position_text(), id="image-position", classes="subtitle")
            with Horizontal(classes="actions image-actions"):
                yield Button("Previous", id="previous")
                yield Button("Next", id="next", variant="primary")
                yield Button("Open External", id="open-external")
                yield Button("Back", id="back")
        yield Footer()

    def action_previous_image(self) -> None:
        self._show_image(self.image_index - 1)

    def action_next_image(self) -> None:
        self._show_image(self.image_index + 1)

    def action_open_external(self) -> None:
        path = self.image_paths[self.image_index]
        try:
            viewer = open_path_externally(path)
        except (OSError, RuntimeError) as error:
            self.notify(str(error), severity="error")
        else:
            self.notify(f"Opened {Path(path).name} in {viewer}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "previous":
            self.action_previous_image()
        elif event.button.id == "next":
            self.action_next_image()
        elif event.button.id == "open-external":
            self.action_open_external()
        elif event.button.id == "back":
            self.action_back()

    def _show_image(self, index: int) -> None:
        self.image_index = index % len(self.image_paths)
        path = self.image_paths[self.image_index]
        self.query_one("#image-title", Static).update(Path(path).name)
        self.query_one("#image-canvas", TerminalImage).set_image(path)
        self.query_one("#image-position", Static).update(self._position_text())

    def _position_text(self) -> str:
        return f"Image {self.image_index + 1} of {len(self.image_paths)} · use ← and → to navigate"


class ExtensionOptionsScreen(StoryForgeScreen):
    """Continuation direction form."""

    def __init__(self, story: StorySummary) -> None:
        super().__init__()
        self.story = story

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="extension-form"):
            yield Static(f"[bold cyan]Extend {self.story.filename}[/bold cyan]", classes="screen-title")
            yield Select(
                [("Leave a cliffhanger", "cliffhanger"), ("Wrap up the story", "wrap_up")],
                value="cliffhanger",
                id="ending",
            )
            yield Input(placeholder="Optional direction for the continuation", id="direction")
            with Horizontal(classes="actions"):
                yield Button("Generate Draft", id="generate", variant="success")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "generate":
            ending = cast(Literal["wrap_up", "cliffhanger"], str(self.query_one("#ending", Select).value))
            self.storyforge_app.create_extension_draft(
                ExtensionRequest(
                    story_id=self.story.id,
                    ending_type=ending,
                    direction=self.query_one("#direction", Input).value.strip() or None,
                )
            )


class ExportOptionsScreen(StoryForgeScreen):
    """Story-chain export destination form."""

    def __init__(self, story: StorySummary) -> None:
        super().__init__()
        self.story = story

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="export-form"):
            yield Static(f"[bold cyan]Export {self.story.filename}[/bold cyan]", classes="screen-title")
            yield Static(f"{self.story.chain_length} story parts will be combined.")
            yield Input(placeholder="Output path (leave blank for automatic name)", id="output")
            with Horizontal(classes="actions"):
                yield Button("Export", id="export", variant="success")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "export":
            self.storyforge_app.export_chain(
                ExportRequest(
                    story_id=self.story.id,
                    output=self.query_one("#output", Input).value.strip() or None,
                )
            )


class ResultScreen(StoryForgeScreen):
    """Consistent completion or failure result."""

    def __init__(self, title: str, message: str, artifacts: list[str] | None = None, error: bool = False) -> None:
        super().__init__()
        self.result_title = title
        self.message = message
        self.artifacts = artifacts or []
        self.error = error

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="result-panel"):
            color = "red" if self.error else "green"
            yield Static(f"[bold {color}]{self.result_title}[/bold {color}]", classes="screen-title")
            yield Static(self.message)
            if self.artifacts:
                yield Static("\n".join(f"• {path}" for path in self.artifacts), id="artifacts")
            with Horizontal(classes="actions"):
                yield Button("Home", id="home", variant="primary")
                yield Button("Quit", id="quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "home":
            self.storyforge_app.go_home()
        elif event.button.id == "quit":
            self.storyforge_app.exit()


class WorldScreen(StoryForgeScreen):
    """World-file viewer and editor."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="world-screen"):
            yield Static("[bold cyan]Story World[/bold cyan]", classes="screen-title")
            yield Static(str(self.data.get("path", "")), classes="subtitle")
            yield TextArea(str(self.data.get("content", "")), id="world-content")
            with Horizontal(classes="actions"):
                yield Button("Save", id="save", variant="success")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "save":
            self.storyforge_app.save_world(self.query_one("#world-content", TextArea).text)


class DataScreen(StoryForgeScreen):
    """Simple structured-data screen for config and model cache state."""

    def __init__(self, title: str, content: str, kind: Literal["config", "models"]) -> None:
        super().__init__()
        self.data_title = title
        self.content = content
        self.kind = kind

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="data-screen"):
            yield Static(f"[bold cyan]{self.data_title}[/bold cyan]", classes="screen-title")
            yield VerticalScroll(Static(self.content), id="data-reader")
            with Horizontal(classes="actions"):
                if self.kind == "config":
                    yield Button("Initialize Config", id="init", variant="primary")
                else:
                    yield Button("Invalidate Cache", id="invalidate", variant="primary")
                    yield Button("Clear Cache", id="clear", variant="error")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "init":
            self.storyforge_app.init_config()
        elif event.button.id == "invalidate":
            self.storyforge_app.invalidate_models()
        elif event.button.id == "clear":
            self.storyforge_app.clear_models()


class StoryForgeApp(App[None]):
    """Full-screen StoryForge MCP client."""

    TITLE = "StoryForge"
    CSS = """
    Screen { background: $surface; align: center middle; }
    Header { background: $primary-background; }
    #home-panel, #form, #review, #media-form, #extension-form, #export-form,
    #progress-panel, #result-panel, #world-screen, #data-screen, #story-browser,
    #story-viewer, #image-viewer {
        width: 90%; max-width: 110; height: auto; max-height: 1fr;
        margin: 1 2; padding: 1 2; border: solid $primary;
    }
    #home-panel { align: center middle; height: 1fr; }
    #brand { text-align: center; text-style: bold; width: 100%; }
    .subtitle { color: $text-muted; margin-bottom: 1; text-align: center; width: 100%; }
    .screen-title { margin-bottom: 1; }
    .home-row, .field-row, .actions { height: auto; margin-top: 1; }
    .home-row { align-horizontal: center; }
    .home-row Button { min-width: 18; margin: 0 1; }
    .field { width: 1fr; margin-right: 1; }
    .actions Button { margin-right: 1; }
    TextArea#prompt { height: 8; }
    #picker-layout { height: 1fr; }
    #item-list { width: 1fr; min-width: 30; border: solid $primary; }
    #item-list:focus { border: solid $accent; }
    #preview-panel { width: 2fr; border: solid $primary; padding: 1 2; overflow-y: auto; }
    #story-browser, #story-viewer, #image-viewer { height: 1fr; }
    #story-browser-layout { height: 1fr; }
    #story-list { width: 2fr; min-width: 32; border: solid $primary; }
    #story-list:focus { border: solid $accent; }
    #story-preview { width: 3fr; height: 1fr; border: solid $primary; padding: 1 2; overflow-y: auto; }
    #story-reader, #library-story-reader, #data-reader {
        height: 1fr; border: solid $primary; padding: 1 2;
    }
    #library-story-text { width: 100%; }
    .literal-title { color: $accent; text-style: bold; }
    #image-canvas { height: 1fr; width: 100%; content-align: center middle; overflow: hidden; }
    #image-title, #image-position { width: 100%; text-align: center; }
    .image-actions { align-horizontal: center; }
    #world-content { height: 1fr; }
    #progress-panel {
        align: center middle;
        width: 72;
        max-width: 90%;
        height: 16;
        max-height: 80%;
        padding: 2 4;
        text-align: center;
    }
    #result-panel { align: center middle; height: 1fr; text-align: center; }
    #progress-message, #progress-value { width: 100%; text-align: center; }
    LoadingIndicator { width: 100%; height: 5; }
    """

    def __init__(
        self,
        route: str = "home",
        initial_request: GenerationRequest | None = None,
        client: StoryForgeMCPClient | Any | None = None,
    ) -> None:
        super().__init__()
        self.route = route
        self.initial_request = initial_request
        self.client: Any = client
        self._owns_client = client is None
        self._active_worker: Any = None
        self._client_owner_task: asyncio.Task[None] | None = None
        self._client_ready = asyncio.Event()
        self._client_stop = asyncio.Event()
        self._client_owner_error: BaseException | None = None

    async def on_mount(self) -> None:
        try:
            if self.client is None:
                self._client_owner_task = asyncio.create_task(
                    self._run_owned_client(),
                    name="storyforge-mcp-client",
                )
                await self._client_ready.wait()
                if self._client_owner_error is not None:
                    task = self._client_owner_task
                    self._client_owner_task = None
                    await task
            self.call_after_refresh(self._show_initial_route)
        except Exception as error:
            self.push_screen(ResultScreen("MCP Server Error", str(error), error=True))

    async def on_unmount(self) -> None:
        task = self._client_owner_task
        if not self._owns_client or task is None:
            return

        self._client_stop.set()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            self._client_owner_task = None

    async def _run_owned_client(self) -> None:
        """Keep the AnyIO-backed MCP context in one asyncio task for its lifetime."""
        try:
            async with StoryForgeMCPClient(
                progress_callback=self._on_progress,
                suppress_server_output=True,
            ) as client:
                self.client = client
                self._client_ready.set()
                await self._client_stop.wait()
        except BaseException as error:
            self._client_owner_error = error
            self._client_ready.set()
            raise

    def get_default_screen(self) -> Screen:
        """Use StoryForge Home as the real root instead of Textual's empty screen."""
        return HomeScreen()

    def _show_initial_route(self) -> None:
        routes: dict[str, Callable[[], Any]] = {
            "generate": self.show_new_story,
            "stories": self.show_stories,
            "continue": self.show_continue,
            "extend": self.show_extend,
            "export": self.show_export,
            "world": self.show_world,
            "config": self.show_config,
            "models": self.show_models,
        }
        action = routes.get(self.route)
        if action is not None:
            action()

    def go_home(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        if not isinstance(self.screen, HomeScreen):
            self.switch_screen(HomeScreen())

    def show_new_story(self) -> None:
        self._show_new_story_with_defaults()

    def _on_progress(self, progress: float, total: float | None, message: str | None) -> None:
        if isinstance(self.screen, ProgressScreen):
            self.screen.update_progress(progress, total, message)

    async def _start_progress(self, title: str) -> ProgressScreen:
        self._active_worker = get_current_worker()
        screen = ProgressScreen(title)
        await self.push_screen(screen)
        return screen

    def cancel_active_workflow(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if isinstance(self.screen, ProgressScreen):
            if len(self.screen_stack) > 1:
                self.pop_screen()
            else:
                self.switch_screen(HomeScreen())
        self.notify("Cancellation requested; the current provider call may finish first.", severity="warning")

    @work(exclusive=True)
    async def _show_new_story_with_defaults(self) -> None:
        await self._start_progress("Loading Story Defaults")
        try:
            request = _request_from_config(await self.client.get_config(), self.initial_request)
            self.switch_screen(NewStoryScreen(request))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Story Defaults", str(error), error=True))

    @work(exclusive=True)
    async def create_draft(self, request: GenerationRequest) -> None:
        self._active_worker = get_current_worker()
        await self._start_progress("Generating Draft")
        try:
            draft = await self.client.create_draft(request)
            self.switch_screen(ReviewScreen(draft))
        except Exception as error:
            self.switch_screen(ResultScreen("Generation Failed", str(error), error=True))

    @work(exclusive=True)
    async def create_extension_draft(self, request: ExtensionRequest) -> None:
        self._active_worker = get_current_worker()
        await self._start_progress("Extending Story")
        try:
            draft = await self.client.create_extension_draft(request)
            self.switch_screen(ReviewScreen(draft))
        except Exception as error:
            self.switch_screen(ResultScreen("Extension Failed", str(error), error=True))

    @work(exclusive=True)
    async def refine_draft(self, session_id: str, instructions: str) -> None:
        self._active_worker = get_current_worker()
        await self._start_progress("Refining Story")
        try:
            draft = await self.client.refine_draft(RefinementRequest(session_id=session_id, instructions=instructions))
            self.switch_screen(ReviewScreen(draft))
        except Exception as error:
            self.switch_screen(ResultScreen("Refinement Failed", str(error), error=True))

    @work(exclusive=True)
    async def finalize_story(self, request: FinalizeRequest) -> None:
        self._active_worker = get_current_worker()
        await self._start_progress("Finishing Story")
        try:
            result: WorkflowResult = await self.client.finalize_story(request)
            self.switch_screen(ResultScreen("Story Complete", result.message, result.artifacts))
        except Exception as error:
            self.switch_screen(ResultScreen("Finalization Failed", str(error), error=True))

    @work(exclusive=True)
    async def show_extend(self) -> None:
        await self._start_progress("Loading Stories")
        try:
            stories = await self.client.list_stories()
            screen: StoryForgeScreen = (
                PickerScreen("Extend Story", "extend", stories)
                if stories
                else ResultScreen("No Saved Stories", "Generate and save a story before extending one.")
            )
            self.switch_screen(screen)
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Stories", str(error), error=True))

    @work(exclusive=True)
    async def show_stories(self) -> None:
        await self._start_progress("Loading Story Library")
        try:
            stories = await self.client.list_generated_stories()
            screen: StoryForgeScreen = (
                StoryBrowserScreen(stories)
                if stories
                else ResultScreen("No Generated Stories", "Generate a story to add it to your library.")
            )
            self.switch_screen(screen)
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Story Library", str(error), error=True))

    @work(exclusive=True)
    async def open_generated_story(self, story_id: str) -> None:
        await self._start_progress("Opening Story")
        try:
            self.switch_screen(StoryReaderScreen(await self.client.get_generated_story(story_id)))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Open Story", str(error), error=True))

    @work(exclusive=True)
    async def show_export(self) -> None:
        await self._start_progress("Loading Story Chains")
        try:
            stories = await self.client.list_stories(chain_only=True)
            screen: StoryForgeScreen = (
                PickerScreen("Export Story Chain", "export", stories)
                if stories
                else ResultScreen("No Story Chains", "Extend a story first to create a chain.")
            )
            self.switch_screen(screen)
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Chains", str(error), error=True))

    @work(exclusive=True)
    async def show_continue(self) -> None:
        await self._start_progress("Loading Sessions")
        try:
            sessions = await self.client.list_sessions()
            screen: StoryForgeScreen = (
                PickerScreen("Continue Session", "continue", sessions)
                if sessions
                else ResultScreen("No Sessions", "No StoryForge checkpoints are available.")
            )
            self.switch_screen(screen)
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Sessions", str(error), error=True))

    @work(exclusive=True)
    async def resume_session(self, session_id: str) -> None:
        await self._start_progress("Resuming Session")
        try:
            draft = await self.client.resume_session(session_id)
            self.switch_screen(ReviewScreen(draft))
        except Exception as error:
            self.switch_screen(ResultScreen("Resume Failed", str(error), error=True))

    @work(exclusive=True)
    async def export_chain(self, request: ExportRequest) -> None:
        await self._start_progress("Exporting Story Chain")
        try:
            result = await self.client.export_chain(request)
            self.switch_screen(ResultScreen("Export Complete", result.message, result.artifacts))
        except Exception as error:
            self.switch_screen(ResultScreen("Export Failed", str(error), error=True))

    @work(exclusive=True)
    async def show_world(self) -> None:
        await self._start_progress("Loading Story World")
        try:
            self.switch_screen(WorldScreen(await self.client.read_world()))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load World", str(error), error=True))

    @work(exclusive=True)
    async def save_world(self, content: str) -> None:
        await self._start_progress("Saving Story World")
        try:
            result = await self.client.write_world(content, overwrite=True)
            self.switch_screen(ResultScreen("World Saved", f"Saved {result.get('path', '')}"))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Save World", str(error), error=True))

    @work(exclusive=True)
    async def show_config(self) -> None:
        await self._start_progress("Loading Configuration")
        try:
            data = await self.client.get_config()
            values = data.get("values", data)
            content = "\n".join(f"[{section}]\n{settings}" for section, settings in values.items())
            self.switch_screen(DataScreen("Configuration", content, "config"))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Configuration", str(error), error=True))

    @work(exclusive=True)
    async def init_config(self) -> None:
        await self._start_progress("Initializing Configuration")
        try:
            result = await self.client.init_config()
            self.switch_screen(ResultScreen("Configuration Ready", result.message, result.artifacts))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Initialize Configuration", str(error), error=True))

    @work(exclusive=True)
    async def show_models(self) -> None:
        await self._start_progress("Loading Model Cache")
        try:
            models = await self.client.list_models()
            lines = []
            for backend, entries in models.items():
                lines.append(f"[bold cyan]{backend.title()}[/bold cyan] · {len(entries)} cached")
                lines.extend(f"  • {entry.get('name', entry.get('id', 'unknown'))}" for entry in entries)
                lines.append("")
            self.switch_screen(DataScreen("Model Cache", "\n".join(lines) or "No cached models.", "models"))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Load Models", str(error), error=True))

    @work(exclusive=True)
    async def invalidate_models(self) -> None:
        await self._start_progress("Invalidating Model Cache")
        try:
            result = await self.client.invalidate_models()
            self.switch_screen(ResultScreen("Cache Invalidated", result.message))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Invalidate Cache", str(error), error=True))

    @work(exclusive=True)
    async def clear_models(self) -> None:
        await self._start_progress("Clearing Model Cache")
        try:
            result = await self.client.clear_models(confirmed=True)
            self.switch_screen(ResultScreen("Cache Cleared", result.message))
        except Exception as error:
            self.switch_screen(ResultScreen("Could Not Clear Cache", str(error), error=True))


def run_tui(route: str = "home", initial_request: GenerationRequest | None = None) -> None:
    """Launch StoryForge's full-screen MCP client."""
    StoryForgeApp(route=route, initial_request=initial_request).run()
