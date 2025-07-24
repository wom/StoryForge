import os
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from .context import ContextManager
from .llm_backend import get_backend
from .prompt import Prompt

console = Console()
app = typer.Typer(
    add_completion=True,
    help="StoryTime CLI - Generate stories and images with AI",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataclass
class CLIArgs:
    name: str
    age: int | None
    verbose: bool
    color: str


def generate_default_output_dir() -> str:
    """Generate a timestamped output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"storytime_output_{timestamp}"
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
    if learning_focus and learning_focus != "random":
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
        "short", "--length", "-l", help="Story length (flash, short, medium, bedtime)"
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
        "random",
        "--learning-focus",
        help="Learning focus (counting, colors, letters, emotions, nature)",
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
    context_file: str | None = typer.Option(
        None, "--context-file", "-c", help="Path to context file (e.g., family.md)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Generate a story and illustration from a prompt."""

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

        # Load context if specified
        # Future enhancement: Smart context extraction will filter this
        # based on characters mentioned in the prompt
        context_manager = ContextManager(context_file)
        context = context_manager.extract_relevant_context(prompt)

        if verbose and context:
            console.print(
                f"[dim]Loaded context from: "
                f"{context_manager._resolve_context_path()}[/dim]"
            )
        elif verbose:
            console.print("[dim]No context file loaded[/dim]")

        # Create Prompt instance
        try:
            # Handle None values and convert to appropriate types
            characters_list = characters if characters else None
            theme_value = theme if theme else None
            learning_focus_value = learning_focus if learning_focus else None

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

        # Generate image
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating illustration..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("image", total=None)
            image, image_bytes = backend.generate_image(story_prompt)

        if image is None or image_bytes is None:
            console.print(
                "[yellow]Warning:[/yellow] Failed to generate image, "
                "but story was created successfully."
            )
            return

        # Generate image filename
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating filename..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("filename", total=None)
            image_name = backend.generate_image_name(story_prompt, story)

        if not image_name:
            image_name = "story_image"
        else:
            # Sanitize filename
            image_name = image_name.replace(" ", "_").replace("/", "-")

        # Ensure .png extension
        if not image_name.endswith(".png"):
            image_name += ".png"

        # Save image to specified directory
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(output_dir, image_name)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        console.print(f"[bold green]✅ Image saved as:[/bold green] {image_path}")

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
def image(
    prompt: str = typer.Argument(..., help="The image prompt to generate from"),
    length: str = typer.Option(
        "short", "--length", "-l", help="Story length (flash, short, medium, bedtime)"
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
        "random",
        "--learning-focus",
        help="Learning focus (counting, colors, letters, emotions, nature)",
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


if __name__ == "__main__":
    app()
