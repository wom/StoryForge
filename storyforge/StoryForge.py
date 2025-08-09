"""
StoryForge: Unified CLI and TUI app for generating illustrated stories
using Gemini LLM backend.

Provides both command-line interface with 'story', 'image', and 'tui' commands,
and a Textual TUI app for interactive story generation.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from platformdirs import user_data_dir

# Use "StoryForge" as appauthor for user_data_dir to ensure user-agnostic,
# organization-consistent data storage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, LoadingIndicator, Log, Static

from .context import ContextManager, get_default_context_manager
from .llm_backend import get_backend
from .prompt import Prompt

console = Console()
app = typer.Typer(
    add_completion=True,
    help="StoryForge - Generate stories and images with AI",
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)

@app.callback()
def main(ctx: typer.Context, prompt: str = typer.Argument(None, help="Story prompt")):
    """
    If called as 'storytime <prompt>', run the story command by default.
    """
    if ctx.invoked_subcommand is None and prompt:
        # Only forward the prompt argument to the story command, use defaults for options
        story_params = {
            "prompt": prompt,
            "length": "bedtime",
            "age_range": "early_reader",
            "style": "random",
            "tone": "random",
            "theme": "random",
            "learning_focus": None,
            "setting": None,
            "characters": None,
            "output_dir": None,
            "use_context": True,
            "verbose": False,
        }
        ctx.invoke(story, **story_params)


@dataclass
class CLIArgs:
    name: str
    age: int | None
    verbose: bool
    color: str


def generate_default_output_dir() -> str:
    """Generate a timestamped output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"storyforge_output_{timestamp}"
    return output_dir


def show_prompt_summary_and_confirm(
    prompt: str,
    age_range: str,
    style: str,
    tone: str,
    theme: str | None,
    length: str,
    setting: str | None,
    characters: list[str] | None,
    learning_focus: str | None,
    generation_type: str = "story",
) -> bool:
    """Display a summary of the prompt and ask for user confirmation."""

    # If any argument is a Typer Option/Argument object, replace with its default or None
    def _extract_value(val):
        # Typer Option/Argument objects have __class__.__name__ like 'OptionInfo'
        if hasattr(val, "__class__") and "OptionInfo" in val.__class__.__name__:
            return val.default if hasattr(val, "default") else None
        return val

    prompt = _extract_value(prompt)
    age_range = _extract_value(age_range)
    style = _extract_value(style)
    tone = _extract_value(tone)
    theme = _extract_value(theme)
    length = _extract_value(length)
    setting = _extract_value(setting)
    learning_focus = _extract_value(learning_focus)
    if characters is not None and isinstance(characters, list):
        characters = [_extract_value(c) for c in characters]

    console.print(
        f"\n[bold cyan]📋 {generation_type.title()} Generation Summary:[/bold cyan]"
    )
    console.print(f"[bold]Prompt:[/bold] {prompt}")
    console.print(f"[bold]Age Range:[/bold] {age_range}")
    console.print(f"[bold]Length:[/bold] {length}")
    console.print(f"[bold]Style:[/bold] {style}")
    console.print(f"[bold]Tone:[/bold] {tone}")

    if theme and theme != "random":
        console.print(f"[bold]Theme:[/bold] {theme}")
    if learning_focus:
        console.print(f"[bold]Learning Focus:[/bold] {learning_focus}")
    if setting:
        console.print(f"[bold]Setting:[/bold] {setting}")
    if characters:
        console.print(f"[bold]Characters:[/bold] {', '.join(characters)}")

    console.print()
    return Confirm.ask(
        f"[bold green]Proceed with {generation_type} generation?[/bold green]"
    )


@app.command()
def story(
    prompt: str = typer.Argument(..., help="The story prompt to generate from"),
    length: str = typer.Option(
        "bedtime", "--length", "-l", help="Story length (flash, short, medium, bedtime)"
    ),
    age_range: str = typer.Option(
        "early_reader",
        "--age-range",
        "-a",
        help="Target age group (toddler, preschool, early_reader, middle_grade)",
    ),
    style: str = typer.Option(
        "random",
        "--style",
        "-s",
        help="Story style (adventure, comedy, fantasy, fairy_tale, friendship)",
    ),
    tone: str = typer.Option(
        "random",
        "--tone",
        "-t",
        help="Story tone (gentle, exciting, silly, heartwarming, magical)",
    ),
    theme: str | None = typer.Option(
        "random",
        "--theme",
        help="Story theme (courage, kindness, teamwork, problem_solving, creativity)",
    ),
    learning_focus: str | None = typer.Option(
        None,
        "--learning-focus",
        help="Learning focus (counting, colors, letters, emotions, nature). Default: None (no learning focus)",
    ),
    setting: str | None = typer.Option(None, "--setting", help="Story setting"),
    characters: Annotated[
        list[str] | None,
        typer.Option("--character", help="Character names/descriptions (multi-use)"),
    ] = None,
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the image (default: auto-generated)",
    ),
    use_context: bool = typer.Option(
        True,
        "--use-context/--no-use-context",
        help=(
            "By default, all .md files in the context/ directory are used as "
            "context for story generation. Use --no-use-context to disable this "
            "behavior."
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Generate a story and illustration from a prompt.

    By default, all .md files in the context/ directory are used as additional context
    for story generation.
    Use the --no-use-context flag to disable including these context files.
    """

    if not prompt.strip():
        console.print(
            "[red]Error:[/red] Please provide a non-empty story prompt.", style="bold"
        )
        raise typer.Exit(1)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_default_output_dir()
        console.print(
            f"[bold blue]📁 Generated output directory:[/bold blue] {output_dir}"
        )

    # Show prompt summary and get confirmation
    if not show_prompt_summary_and_confirm(
        prompt=prompt,
        age_range=age_range,
        style=style,
        tone=tone,
        theme=theme,
        length=length,
        setting=setting,
        characters=characters,
        learning_focus=learning_focus,
        generation_type="story",
    ):
        console.print("[yellow]Story generation cancelled.[/yellow]")
        raise typer.Exit(0)

    try:
        # Initialize backend
        if verbose:
            console.print("[dim]Initializing Gemini backend...[/dim]")

        backend = get_backend()

        # Load context files if --use-context is enabled (default).
        # By default, all .md files in the context/ directory are included as context
        # for story generation.
        # Future enhancement: Smart context extraction will filter this
        # based on characters mentioned in the prompt.
        # Create Prompt instance
        try:
            # Handle None values and convert to appropriate types
            characters_list = characters if characters else None
            theme_value = theme if theme else None
            learning_focus_value = learning_focus if learning_focus else None

            # If --use-context (default), load all .md files in context/ as context for
            # story generation.
            if use_context:
                # Use cross-platform user data directory for context files
                context_dir = Path(user_data_dir("StoryForge", "StoryForge")) / "context"
                context: str | None = ""
                if context_dir.is_dir():
                    md_files = [f for f in context_dir.iterdir() if f.suffix == ".md"]
                    contents = []
                    for fpath in md_files:
                        try:
                            with open(fpath, encoding="utf-8") as f:
                                contents.append(f.read())
                        except Exception as e:
                            if verbose:
                                console.print(
                                    f"[yellow]Warning: Could not read {fpath}: "
                                    f"{e}[/yellow]"
                                )
                    context = "\n".join(contents) if contents else None
                else:
                    context = None
            else:
                # If --no-use-context is specified, do not include any context files.
                context = None

            story_prompt = Prompt(
                prompt=prompt,
                context=context,
                length=cast("Literal['flash', 'short', 'medium', 'bedtime']", length),
                age_range=cast(
                    "Literal['toddler', 'preschool', 'early_reader', 'middle_grade']",
                    age_range,
                ),
                style=cast(
                    "Literal['adventure', 'comedy', 'fantasy', 'fairy_tale', "
                    "'friendship', 'random']",
                    style,
                ),
                tone=cast(
                    "Literal['gentle', 'exciting', 'silly', 'heartwarming', "
                    "'magical', 'random']",
                    tone,
                ),
                theme=cast(
                    "Literal['courage', 'kindness', 'teamwork', 'problem_solving', "
                    "'creativity', 'family', 'random'] | None",
                    theme_value,
                ),
                setting=setting,
                characters=characters_list,
                learning_focus=cast(
                    "Literal['counting', 'colors', 'letters', 'emotions', "
                    "'nature', 'random'] | None",
                    learning_focus_value,
                ),
            )

            if verbose:
                console.print("[dim]Created prompt with parameters:[/dim]")
                console.print(f"[dim]  Length: {story_prompt.length}[/dim]")
                console.print(f"[dim]  Age Range: {story_prompt.age_range}[/dim]")
                console.print(f"[dim]  Style: {story_prompt.style}[/dim]")
                console.print(f"[dim]  Tone: {story_prompt.tone}[/dim]")
                console.print(f"[dim]  Theme: {story_prompt.theme}[/dim]")
                console.print(
                    f"[dim]  Learning Focus: {story_prompt.learning_focus}[/dim]"
                )

        except ValueError as e:
            console.print(
                f"[red]Error:[/red] Invalid parameter value: {e}", style="bold"
            )
            raise typer.Exit(1) from e

        # Generate story
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating story..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("story", total=None)
            story = backend.generate_story(story_prompt)
            # read in a file and populate a variable `story`
            # story = load_story_from_file()

        if story == "[Error generating story]":
            console.print(
                "[red]Error:[/red] Failed to generate story. "
                "Please check your API key and try again.",
                style="bold",
            )
            raise typer.Exit(1)

        # Display the generated story
        console.print("\n[bold green]Generated Story:[/bold green]")
        console.print(f"[dim]Prompt:[/dim] {prompt}")
        console.print()
        console.print(story)
        console.print()

        # Always save the story and print the message before image generation
        story_filename = "story.txt"
        story_path = os.path.join(output_dir, story_filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(f"Story: {prompt}\n\n")
            f.write(story)
        console.print(f"[bold green]✅ Story saved as:[/bold green] {story_path}")

        # Present image generation options using Confirm.ask for test compatibility
        if Confirm.ask("Would you like to generate illustrations for the story?"):
            # make num_paragraphs be the number of paragraphs in the story
            num_paragrpahs = len([p.strip() for p in story.split("\n") if p.strip()])
            print(num_paragrpahs)
            sys.exit(1)
            
        else:
            console.print("[yellow]Image generation skipped by user.[/yellow]")
            # (Story already saved above, do not save again here)

        # Always ask if user wants to save as future context after story generation
        save_as_context = Confirm.ask(
            "[bold blue]Save this story as future context for character "
            "development?[/bold blue]"
        )
        if save_as_context:
            # Use cross-platform user data directory for context files
            context_dir = Path(user_data_dir("StoryForge", "StoryForge")) / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            context_filename = f"story_{timestamp}.md"
            # Use pathlib for cross-platform compatibility and error handling
            context_path = Path(context_dir) / context_filename
            try:
                with open(context_path, "w", encoding="utf-8") as f:
                    f.write("# Story Context\n\n")
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"**Generated:** {timestamp_str}\n\n")
                    f.write("## Story Parameters\n\n")
                    f.write(f"- **Prompt:** {prompt}\n")
                    f.write(f"- **Age Range:** {age_range}\n")
                    f.write(f"- **Length:** {length}\n")
                    f.write(f"- **Style:** {style}\n")
                    f.write(f"- **Tone:** {tone}\n")
                    if theme and theme != "random":
                        f.write(f"- **Theme:** {theme}\n")
                    if learning_focus:
                        f.write(f"- **Learning Focus:** {learning_focus}\n")
                    if setting:
                        f.write(f"- **Setting:** {setting}\n")
                    if characters:
                        f.write(f"- **Characters:** {', '.join(characters)}\n")
                    f.write("\n## Generated Story\n\n")
                    f.write(story)
                    f.write("\n\n## Usage Notes\n\n")
                    f.write("This story can be referenced for:\n")
                    f.write("- Character consistency in future stories\n")
                    f.write("- Setting and world-building continuity\n")
                    f.write("- Tone and style reference\n")
                    f.write("- Educational content alignment\n")
                console.print(
                    f"[bold green]✅ Context saved as:[/bold green] {context_path}"
                )
            except Exception as e:
                console.print(
                    f"[red]Error saving context file:[/red] {e}", style="bold"
                )

    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            console.print(
                "[red]Error:[/red] GEMINI_API_KEY environment variable not set.",
                style="bold",
            )
            console.print(
                "[dim]Please set your Gemini API key: "
                "export GEMINI_API_KEY=your_key_here[/dim]"
            )
        else:
            console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error:[/red] {e}", style="bold")
        else:
            console.print(
                "[red]Error:[/red] An unexpected error occurred. "
                "Use --verbose for details.",
                style="bold",
            )
        raise typer.Exit(1) from e


@app.command()
def image(
    prompt: str = typer.Argument(..., help="The image prompt to generate from"),
    length: str = typer.Option(
        "bedtime", "--length", "-l", help="Story length (flash, short, medium, bedtime)"
    ),
    age_range: str = typer.Option(
        "preschool",
        "--age-range",
        "-a",
        help="Target age group (toddler, preschool, early_reader, middle_grade)",
    ),
    style: str = typer.Option(
        "random",
        "--style",
        "-s",
        help="Story style (adventure, comedy, fantasy, fairy_tale, friendship)",
    ),
    tone: str = typer.Option(
        "random",
        "--tone",
        "-t",
        help="Story tone (gentle, exciting, silly, heartwarming, magical)",
    ),
    theme: str | None = typer.Option(
        "random",
        "--theme",
        help="Story theme (courage, kindness, teamwork, problem_solving, creativity)",
    ),
    learning_focus: str | None = typer.Option(
        None,
        "--learning-focus",
        help="Learning focus (counting, colors, letters, emotions, nature). Default: None (no learning focus)",
    ),
    setting: str | None = typer.Option(None, "--setting", help="Story setting"),
    characters: Annotated[
        list[str] | None,
        typer.Option("--character", help="Character names/descriptions (multi-use)"),
    ] = None,
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the image (default: auto-generated)",
    ),
    filename: str | None = typer.Option(
        None, "--filename", "-f", help="Custom filename (without extension)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Generate an image from a prompt."""

    if not prompt.strip():
        console.print(
            "[red]Error:[/red] Please provide a non-empty image prompt.", style="bold"
        )
        raise typer.Exit(1)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_default_output_dir()
        console.print(
            f"[bold blue]📁 Generated output directory:[/bold blue] {output_dir}"
        )

    # Show prompt summary and get confirmation
    if not show_prompt_summary_and_confirm(
        prompt=prompt,
        age_range=age_range,
        style=style,
        tone=tone,
        theme=theme,
        length=length,
        setting=setting,
        characters=characters,
        learning_focus=learning_focus,
        generation_type="image",
    ):
        console.print("[yellow]Image generation cancelled.[/yellow]")
        raise typer.Exit(0)

    try:
        # Initialize backend
        if verbose:
            console.print("[dim]Initializing Gemini backend...[/dim]")

        backend = get_backend()

        # Create Prompt instance
        try:
            # Handle None values and convert to appropriate types
            characters_list = characters if characters else None
            theme_value = theme if theme else None
            learning_focus_value = learning_focus if learning_focus else None

            image_prompt = Prompt(
                prompt=prompt,
                context=None,  # No context for standalone image generation
                length=cast("Literal['flash', 'short', 'medium', 'bedtime']", length),
                age_range=cast(
                    "Literal['toddler', 'preschool', 'early_reader', 'middle_grade']",
                    age_range,
                ),
                style=cast(
                    "Literal['adventure', 'comedy', 'fantasy', 'fairy_tale', "
                    "'friendship', 'random']",
                    style,
                ),
                tone=cast(
                    "Literal['gentle', 'exciting', 'silly', 'heartwarming', "
                    "'magical', 'random']",
                    tone,
                ),
                theme=cast(
                    "Literal['courage', 'kindness', 'teamwork', 'problem_solving', "
                    "'creativity', 'family', 'random'] | None",
                    theme_value,
                ),
                setting=setting,
                characters=characters_list,
                learning_focus=cast(
                    "Literal['counting', 'colors', 'letters', 'emotions', "
                    "'nature', 'random'] | None",
                    learning_focus_value,
                ),
            )

            if verbose:
                console.print("[dim]Created prompt with parameters:[/dim]")
                console.print(f"[dim]  Style: {image_prompt.style}[/dim]")
                console.print(f"[dim]  Tone: {image_prompt.tone}[/dim]")
                console.print(f"[dim]  Theme: {image_prompt.theme}[/dim]")
                console.print(
                    f"[dim]  Learning Focus: {image_prompt.learning_focus}[/dim]"
                )

        except ValueError as e:
            console.print(
                f"[red]Error:[/red] Invalid parameter value: {e}", style="bold"
            )
            raise typer.Exit(1) from e

        # Generate image
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating image..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("image", total=None)
            image, image_bytes = backend.generate_image(image_prompt)

        if image is None or image_bytes is None:
            console.print(
                "[red]Error:[/red] Failed to generate image. "
                "Please check your API key and try again.",
                style="bold",
            )
            raise typer.Exit(1)

        # Determine filename
        if filename:
            image_name = filename
        else:
            # Generate filename based on prompt
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating filename..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("filename", total=None)
                raw_name = backend.generate_image_name(
                    image_prompt, prompt
                )  # Use prompt as "story" for naming

            if not raw_name or raw_name == "story_image":
                # Fallback: create filename from prompt
                image_name = prompt.lower().replace(" ", "_")[:30]  # Limit to 30 chars
            else:
                # Clean up the generated name - take only the first line/word
                lines = raw_name.split("\n")
                first_line = lines[0].strip()
                # Take first word or phrase before common separators
                for sep in [".", ",", ":", ";", "!", "?"]:
                    first_line = first_line.split(sep)[0]
                image_name = (
                    first_line.strip() if first_line.strip() else "generated_image"
                )

        # Sanitize filename using standard library methods
        import string

        # Keep alphanumeric, underscore, hyphen, and space
        # (will convert space to underscore)
        valid_chars = string.ascii_letters + string.digits + "_- "
        # Filter to valid characters and replace spaces with underscores
        clean_chars = "".join(c if c in valid_chars else "_" for c in image_name)
        image_name = clean_chars.replace(" ", "_")

        # Ensure .png extension
        if not image_name.endswith(".png"):
            image_name += ".png"

        # Save image to specified directory
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(output_dir, image_name)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        console.print(f"[bold green]✅ Image saved as:[/bold green] {image_path}")
        console.print(f"[dim]Prompt:[/dim] {prompt}")

        if verbose:
            console.print(f"[dim]Image size: {len(image_bytes)} bytes[/dim]")

    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            console.print(
                "[red]Error:[/red] GEMINI_API_KEY environment variable not set.",
                style="bold",
            )
            console.print(
                "[dim]Please set your Gemini API key: "
                "export GEMINI_API_KEY=your_key_here[/dim]"
            )
        else:
            console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error:[/red] {e}", style="bold")
        else:
            console.print(
                "[red]Error:[/red] An unexpected error occurred. "
                "Use --verbose for details.",
                style="bold",
            )
        raise typer.Exit(1) from e


@app.command()
def tui(
    context_file: str | None = typer.Option(
        None, "--context-file", "-c", help="Path to context file (e.g., family.md)"
    ),
):
    """Launch the interactive TUI (Text User Interface) for story generation."""
    console.print("[bold blue]Launching StoryForge TUI...[/bold blue]")
    StoryApp(context_file=context_file).run()


def load_story_from_file() -> str | None:
    """
    Load story content from the default story file.
    
    Returns:
        str | None: The story content with newlines replaced by spaces, 
                   or None if the file doesn't exist.
    """
    story_file = Path("storytime/test_story.txt")
    if not story_file.exists():
        print(f"[yellow]Warning:[/yellow] Story file {story_file} does not exist.")
        return None
    
    with open(story_file, "r", encoding="utf-8") as f:
        story = f.read().strip()
    # return story.replace("\n", " ")
    return story


class StoryApp(App):
    """
    Main Textual application for interactive story and image generation.
    Presents a prompt input, output log, and confirmation dialogs.
    """

    CSS_PATH = None  # No custom CSS
    BINDINGS = [("q", "quit", "Quit")]  # Keyboard shortcut to quit

    def __init__(self, context_file: str | None = None, *args, **kwargs):
        """
        Initialize the StoryApp with backend and optional context.

        Args:
            context_file: Path to context file (e.g., family.md). If None,
                         will use default context manager to find data/family.md
        """
        super().__init__(*args, **kwargs)
        from .llm_backend import get_backend

        self.backend = get_backend()
        # Future enhancement: Context will be intelligently filtered per prompt
        self.context_manager = get_default_context_manager()
        if context_file:
            self.context_manager = ContextManager(context_file)
        self._pending_prompt: str | None = None

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

            # Create Prompt object
            prompt_obj = Prompt(prompt=prompt, context=context)

            # Generate story in a thread to avoid blocking UI
            story = await asyncio.to_thread(self.backend.generate_story, prompt_obj)

            output_log.write(f"[green]Story:[/green]\n{story}")
            output_log.write(
                "[bold green]How would you like to generate illustrations?[/bold green]"
            )
            output_log.write(
                "1) Use the story as context and describe the image yourself"
            )
            output_log.write(
                "2) Break the story into logical chunks (paragraphs) and "
                "generate an image for each"
            )
            output_log.write("3) Do not generate any images")
            # Placeholder for TUI input: in a real TUI, present buttons and dialogs
            # For now, default to option 3 (skip) for demonstration
            option = 3
            # TODO: Replace with actual TUI input handling for options and descriptions
            if option == 3:
                output_log.write("[yellow]Image generation skipped by user.[/yellow]")
                return
            elif option == 1:
                user_desc = "Illustration for the story"  # TODO: Get from TUI input
                image_prompt_text = f"{user_desc}\n\nStory context:\n{story}"
                output_log.write("[bold]Generating image...[/bold]")
                # Create Prompt object for image generation
                user_image_prompt = Prompt(
                    prompt=image_prompt_text,
                    context=prompt_obj.context,
                    length=prompt_obj.length,
                    age_range=prompt_obj.age_range,
                    style=prompt_obj.style,
                    tone=prompt_obj.tone,
                    theme=prompt_obj.theme,
                    setting=prompt_obj.setting,
                    characters=prompt_obj.characters,
                    learning_focus=prompt_obj.learning_focus,
                )
                image, image_bytes = await asyncio.to_thread(
                    self.backend.generate_image, user_image_prompt
                )
                if image is None:
                    output_log.write("[red]Failed to generate image.[/red]")
                    return
                image_name = "story_image.png"
                if image_bytes is not None:
                    with open(image_name, "wb") as f:
                        f.write(image_bytes)
                    output_log.write(f"[green]Image saved as:[/green] {image_name}")
                else:
                    output_log.write("[red]No image data to save.[/red]")
                    return
            elif option == 2:
                paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
                style_hint = ""  # TODO: Get from TUI input
                for idx, para in enumerate(paragraphs, 1):
                    para_desc = para  # TODO: Get from TUI input
                    image_prompt_text = f"{para_desc}\n\nStory context:\n{story}"
                    output_log.write(
                        f"[bold]Generating image for paragraph {idx}...[/bold]"
                    )
                    # Create Prompt object for image generation
                    final_prompt_text = (
                        image_prompt_text
                        if not style_hint
                        else f"{image_prompt_text}\nStyle: {style_hint}"
                    )
                    para_image_prompt = Prompt(
                        prompt=final_prompt_text,
                        context=prompt_obj.context,
                        length=prompt_obj.length,
                        age_range=prompt_obj.age_range,
                        style=prompt_obj.style,
                        tone=prompt_obj.tone,
                        theme=prompt_obj.theme,
                        setting=prompt_obj.setting,
                        characters=prompt_obj.characters,
                        learning_focus=prompt_obj.learning_focus,
                    )
                    image, image_bytes = await asyncio.to_thread(
                        self.backend.generate_image, para_image_prompt
                    )
                    image_name = f"story_paragraph_{idx}.png"
                    if image_bytes:
                        with open(image_name, "wb") as f:
                            f.write(image_bytes)
                        output_log.write(f"[green]Image saved as:[/green] {image_name}")
            else:
                output_log.write(
                    "[yellow]Invalid option. Skipping image generation.[/yellow]"
                )
                return
        finally:
            spinner.remove()
            self._pending_prompt = None


if __name__ == "__main__":
    app()
