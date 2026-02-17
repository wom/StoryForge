"""Interactive TUI story picker using Textual.

Provides a full-screen interactive picker with arrow-key navigation,
a live preview panel, and a full-story reading view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, OptionList, Static


class StoryPicker(App[int | None]):
    """Full-screen story picker with live preview and full-story view.

    Modes:
    - **Picker**: OptionList on the left, metadata + preview on the right.
      Press Enter to open the full story.  Press Escape / q to cancel.
    - **Full story**: Scrollable full story text.
      Press Enter to confirm selection.  Press Escape to go back to the picker.

    Returns the selected story index (0-based) or None if cancelled.
    """

    TITLE = "StoryForge — Select a Story"

    CSS = """
    #picker-layout {
        height: 1fr;
    }

    #story-list {
        width: 1fr;
        min-width: 30;
        border: solid $primary;
    }

    #story-list:focus {
        border: solid $accent;
    }

    #preview-panel {
        width: 2fr;
        border: solid $primary;
        padding: 1 2;
        overflow-y: auto;
    }

    #full-story-view {
        display: none;
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
    }

    #full-story-view.visible {
        display: block;
    }

    #picker-layout.hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("escape", "back_or_cancel", "Back / Cancel", show=True),
        Binding("enter", "confirm", "Select", show=True),
        Binding("q", "quit_app", "Quit", show=True, priority=True),
    ]

    def __init__(self, stories: list[dict[str, Any]]) -> None:
        """Initialize the picker with a list of story context dicts.

        Args:
            stories: List of context dicts from ContextManager.list_available_contexts().
        """
        super().__init__()
        self._stories = stories
        self._viewing_full_story = False
        self._highlighted_index: int | None = None

    def compose(self) -> ComposeResult:
        """Build the picker layout: header, list + preview, full-story container, footer."""
        yield Header()
        with Horizontal(id="picker-layout"):
            options = []
            for idx, ctx in enumerate(self._stories):
                label = f"{idx + 1}. {ctx['filename']}"
                ts = ctx.get("timestamp", "")
                if ts:
                    label += f"  ({ts})"
                options.append(label)
            yield OptionList(*options, id="story-list")
            yield Static("Select a story to see its preview.", id="preview-panel")
        yield VerticalScroll(Static("", id="full-story-content"), id="full-story-view")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the list and show the first story's preview."""
        option_list = self.query_one("#story-list", OptionList)
        option_list.focus()
        if self._stories:
            self._highlighted_index = 0
            self._update_preview(0)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update preview panel when a different story is highlighted."""
        self._highlighted_index = event.option_index
        self._update_preview(event.option_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """When Enter is pressed on a list item, show the full story view."""
        self._highlighted_index = event.option_index
        self._show_full_story()

    # -- actions ---------------------------------------------------------------

    def action_back_or_cancel(self) -> None:
        """Escape: return to picker from full-story view, or cancel entirely."""
        if self._viewing_full_story:
            self._show_picker()
        else:
            self.exit(None)

    def action_confirm(self) -> None:
        """Enter: in full-story view, confirm the selection."""
        if self._viewing_full_story and self._highlighted_index is not None:
            self.exit(self._highlighted_index)
        # In picker mode OptionList handles Enter natively (fires OptionSelected).

    def action_quit_app(self) -> None:
        """Quit the picker entirely."""
        self.exit(None)

    # -- view switching --------------------------------------------------------

    def _show_full_story(self) -> None:
        """Switch from picker to full-story reading view."""
        if self._highlighted_index is None:
            return

        ctx = self._stories[self._highlighted_index]
        story_text = self._load_full_story(ctx)

        header = f"[bold cyan]📖 {ctx['filename']}[/bold cyan]"
        hint = "[dim]Enter: extend this story  ·  Esc: back to list[/dim]"
        content = f"{header}\n{hint}\n\n{story_text}"

        self.query_one("#full-story-content", Static).update(content)
        self.query_one("#picker-layout").add_class("hidden")
        self.query_one("#full-story-view").add_class("visible")
        self.query_one("#full-story-view").focus()
        self._viewing_full_story = True

    def _show_picker(self) -> None:
        """Switch from full-story view back to picker."""
        self.query_one("#full-story-view").remove_class("visible")
        self.query_one("#picker-layout").remove_class("hidden")
        self.query_one("#story-list", OptionList).focus()
        self._viewing_full_story = False

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _load_full_story(ctx: dict[str, Any]) -> str:
        """Read the full story text from the context file on disk."""
        raw_path = ctx.get("filepath")
        if raw_path is None:
            return ctx.get("preview", "(no content available)")

        filepath = Path(raw_path) if not isinstance(raw_path, Path) else raw_path
        if not filepath.exists():
            return ctx.get("preview", "(no content available)")

        try:
            raw = filepath.read_text(encoding="utf-8")
        except OSError:
            return ctx.get("preview", "(unable to read file)")

        # Extract just the story body after the metadata header
        if "## Story" in raw:
            return raw.split("## Story", 1)[1].strip()
        return raw.strip()

    def _update_preview(self, index: int) -> None:
        """Update the preview panel with story details."""
        if index < 0 or index >= len(self._stories):
            return

        ctx = self._stories[index]
        lines: list[str] = []

        lines.append(f"[bold cyan]{ctx['filename']}[/bold cyan]")
        lines.append("")

        ts = ctx.get("timestamp", "Unknown")
        lines.append(f"[bold]Generated:[/bold] {ts}")

        if "characters" in ctx:
            lines.append(f"[bold]Characters:[/bold] {ctx['characters']}")
        if "theme" in ctx:
            lines.append(f"[bold]Theme:[/bold] {ctx['theme']}")
        if "prompt" in ctx:
            lines.append(f"[bold]Prompt:[/bold] {ctx['prompt']}")

        lines.append("")

        preview = ctx.get("preview", "")
        if preview:
            lines.append("[bold]Preview:[/bold]")
            lines.append(preview)

        lines.append("")
        lines.append("[dim]Press Enter to read full story[/dim]")

        panel = self.query_one("#preview-panel", Static)
        panel.update("\n".join(lines))


def pick_story(stories: list[dict[str, Any]]) -> int | None:
    """Run the interactive story picker and return the selected index (0-based) or None.

    Args:
        stories: List of context dicts from ContextManager.list_available_contexts().

    Returns:
        Selected index (0-based) or None if cancelled or list is empty.
    """
    if not stories:
        return None
    app = StoryPicker(stories)
    return app.run()
