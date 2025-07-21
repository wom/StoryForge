"""
StoryTime: Textual TUI app for generating illustrated stories using Gemini LLM backend.
Handles user input, confirmation dialogs, and asynchronous story/image generation.
"""

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, LoadingIndicator, Log, Static

from .context import get_default_context_manager
from .gemini_backend import GeminiBackend


class StoryApp(App):
    """
    Main Textual application for interactive story and image generation.
    Presents a prompt input, output log, and confirmation dialogs.
    """

    CSS_PATH = None  # No custom CSS
    BINDINGS = [("q", "quit", "Quit")]  # Keyboard shortcut to quit

    def __init__(self, context_file: str | None = None, *args, **kwargs):
        """
        Initialize the StoryApp with Gemini backend and optional context.

        Args:
            context_file: Path to context file (e.g., family.md). If None,
                         will use default context manager to find data/family.md
        """
        super().__init__(*args, **kwargs)
        self.backend = GeminiBackend()
        # Future enhancement: Context will be intelligently filtered per prompt
        self.context_manager = get_default_context_manager()
        if context_file:
            from .context import ContextManager

            self.context_manager = ContextManager(context_file)

    def compose(self) -> ComposeResult:
        """
        Compose the main UI layout: prompt input, generate button, output log,
        and hidden confirmation dialog.
        """
        yield Vertical(
            Static("Enter a story prompt:"),
            Input(placeholder="Type your story prompt here...", id="prompt_input"),
            Button("Generate Story & Image", id="generate_btn"),
            Log(id="output_log", highlight=True, max_lines=100),
            # Confirmation dialog, hidden by default
            Static("", id="confirm_dialog", classes="hidden"),
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button presses for generating stories/images and confirmation
        dialog actions.
        Args:
            event (Button.Pressed): The button press event.
        """
        if event.button.id == "generate_btn":
            # User clicked 'Generate Story & Image'
            prompt_input = self.query_one("#prompt_input", Input)
            output_log = self.query_one("#output_log", Log)
            prompt = prompt_input.value.strip()
            if not prompt:
                output_log.write("[red]Please enter a story prompt.[/red]")
                return
            # Show confirmation dialog
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update(
                "[bold]Are you sure?[/bold]\n\n"
                "This will generate a story and an image based on your prompt, "
                "and save the image to disk.\n\nContinue?\n\n"
            )
            confirm_dialog.remove_class("hidden")
            confirm_dialog.mount(
                Horizontal(
                    Button("Yes", id="confirm_yes"), Button("No", id="confirm_no")
                )
            )
            self.set_focus(confirm_dialog)
            self._pending_prompt = prompt  # Store prompt for later use
            return
        elif event.button.id == "confirm_yes":
            # User confirmed generation
            await self._do_generation()
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update("")
            confirm_dialog.add_class("hidden")
            # Remove all children (buttons) from dialog
            for child in list(confirm_dialog.children):
                child.remove()
        elif event.button.id == "confirm_no":
            # User cancelled generation
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update("")
            confirm_dialog.add_class("hidden")
            # Remove all children (buttons) from dialog
            for child in list(confirm_dialog.children):
                child.remove()
            # Remove spinner if present
            try:
                spinner = self.query("#working_spinner").first()
                if spinner is not None:
                    spinner.remove()
            except Exception:
                pass
            self._pending_prompt = None

    async def _do_generation(self):
        """
        Asynchronously generate the story and image, update the UI, and save
        the image to disk.
        Handles spinner display and error reporting.
        """
        prompt = getattr(self, "_pending_prompt", None)
        if not prompt:
            return
        output_log = self.query_one("#output_log", Log)
        spinner = LoadingIndicator(id="working_spinner")
        self.mount(spinner)
        try:
            output_log.clear()
            output_log.write(f"[bold]Generating story for:[/bold] {prompt}")
            await asyncio.sleep(0.1)  # Let spinner show
            # Load context for story generation
            # Future enhancement: Smart filtering based on prompt analysis
            context = self.context_manager.extract_relevant_context(prompt)

            # Generate story in a thread to avoid blocking UI
            story = await asyncio.to_thread(
                self.backend.generate_story, prompt, context
            )
            output_log.write(f"[green]Story:[/green]\n{story}")
            output_log.write("[bold]Generating image...[/bold]")
            # Generate image in a thread
            image, image_bytes = await asyncio.to_thread(
                self.backend.generate_image, prompt
            )
            if image is None:
                output_log.write("[red]Failed to generate image.[/red]")
                return
            output_log.write("[bold]Generating image name...[/bold]")
            # Generate image filename in a thread
            image_name = await asyncio.to_thread(
                self.backend.generate_image_name, prompt, story
            )
            if not image_name:
                image_name = "story_image.png"
            else:
                # Sanitize filename
                image_name = image_name.replace(" ", "_").replace("/", "-") + ".png"
            # Save image to disk
            with open(image_name, "wb") as f:
                f.write(image_bytes)
            output_log.write(f"[green]Image saved as:[/green] {image_name}")
        finally:
            spinner.remove()
            self._pending_prompt = None


def main(context_file: str | None = None):
    """
    Entry point for running the StoryApp.

    Args:
        context_file: Optional path to context file
    """
    StoryApp(context_file=context_file).run()


if __name__ == "__main__":
    main()
