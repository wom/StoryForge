from dataclasses import dataclass
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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


@app.command()
def story(
    prompt: str = typer.Argument(..., help="The story prompt to generate from"),
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
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to save the image"
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
                length=length,  # type: ignore
                age_range=age_range,  # type: ignore
                style=style,  # type: ignore
                tone=tone,  # type: ignore
                theme=theme_value,  # type: ignore
                setting=setting,
                characters=characters_list,
                learning_focus=learning_focus_value,  # type: ignore
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
        import os

        if output_dir != ".":
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
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to save the image"
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
                length=length,  # type: ignore
                age_range=age_range,  # type: ignore
                style=style,  # type: ignore
                tone=tone,  # type: ignore
                theme=theme_value,  # type: ignore
                setting=setting,
                characters=characters_list,
                learning_focus=learning_focus_value,  # type: ignore
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
        import os

        if output_dir != ".":
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
