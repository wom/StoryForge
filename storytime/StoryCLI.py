import typer
from rich.console import Console
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from dataclasses import dataclass
from typing import Optional

from .gemini_backend import GeminiBackend

console = Console()
app = typer.Typer(add_completion=False, help="StoryTime CLI - Generate stories and images with AI")

@dataclass
class CLIArgs:
    name: str
    age: Optional[int]
    verbose: bool
    color: str

@app.command()
def hello(
    name: str = typer.Argument(..., help="Your name"),
    age: Optional[int] = typer.Option(None, "--age", "-a", help="Your age (optional)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    color: str = typer.Option("green", "--color", "-c", help="Text color", show_choices=True, case_sensitive=False)
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
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Directory to save the image"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Generate a story and illustration from a prompt."""
    
    if not prompt.strip():
        console.print("[red]Error:[/red] Please provide a non-empty story prompt.", style="bold")
        raise typer.Exit(1)
    
    try:
        # Initialize backend
        if verbose:
            console.print("[dim]Initializing Gemini backend...[/dim]")
        
        backend = GeminiBackend()
        
        # Generate story
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating story..."),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("story", total=None)
            story = backend.generate_story(prompt)
        
        if story == "[Error generating story]":
            console.print("[red]Error:[/red] Failed to generate story. Please check your API key and try again.", style="bold")
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
            transient=True
        ) as progress:
            progress.add_task("image", total=None)
            image, image_bytes = backend.generate_image(prompt)
        
        if image is None or image_bytes is None:
            console.print("[yellow]Warning:[/yellow] Failed to generate image, but story was created successfully.")
            return
        
        # Generate image filename
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating filename..."),
            console=console,
            transient=True
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
            console.print("[red]Error:[/red] GEMINI_API_KEY environment variable not set.", style="bold")
            console.print("[dim]Please set your Gemini API key: export GEMINI_API_KEY=your_key_here[/dim]")
        else:
            console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error:[/red] {e}", style="bold")
        else:
            console.print("[red]Error:[/red] An unexpected error occurred. Use --verbose for details.", style="bold")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()