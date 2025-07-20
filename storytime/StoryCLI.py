from dataclasses import dataclass

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .llm_backend import get_backend

console = Console()
app = typer.Typer(
    add_completion=False, help="StoryTime CLI - Generate stories and images with AI"
)


@dataclass
class CLIArgs:
    name: str
    age: int | None
    verbose: bool
    color: str


@app.command()
def hello(
    name: str = typer.Argument(..., help="Your name"),
    age: int | None = typer.Option(None, "--age", "-a", help="Your age (optional)"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    color: str = typer.Option(
        "green",
        "--color",
        "-c",
        help="Text color",
        show_choices=True,
        case_sensitive=False,
    ),
):
    """Say hello to someone (original greeting command)."""
    args = CLIArgs(name, age, verbose, color)
    greeting = f"Hello, {args.name}!"
    if args.age is not None:
        greeting += f" You are {args.age} years old."
    if args.verbose:
        greeting += " (Verbose mode enabled)"
    text = Text(greeting, style=args.color)
    console.print(text)


@app.command()
def story(
    prompt: str = typer.Argument(..., help="The story prompt to generate from"),
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to save the image"
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

        # Generate story
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating story..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("story", total=None)
            story = backend.generate_story(prompt)

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
            image, image_bytes = backend.generate_image(prompt)

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
            image_name = backend.generate_image_name(prompt, story)

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

        # Generate image
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating image..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("image", total=None)
            image, image_bytes = backend.generate_image(prompt)

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
                    prompt, prompt
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
