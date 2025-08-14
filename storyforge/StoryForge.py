"""
StoryForge: Simplified CLI for generating illustrated stories using multiple LLM backends.
Supports Google Gemini and Anthropic Claude backends.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_data_dir
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from .llm_backend import get_backend
from .prompt import Prompt

console = Console()

# Create Typer app instance for entrypoint
app = typer.Typer(help="StoryForge: Generate illustrated stories using AI language models (Gemini/Claude)")


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
    image_style: str = "chibi",
    generation_type: str = "story",
    backend_name: str | None = None,
) -> bool:
    """Display a summary of the prompt and ask for user confirmation."""
    console.print(f"\n[bold cyan]📋 {generation_type.title()} Generation Summary:[/bold cyan]")
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
    console.print(f"[bold]Image Style:[/bold] {image_style}")
    if backend_name:
        console.print(f"[bold]Backend:[/bold] {backend_name}")
    console.print()
    return Confirm.ask(f"[bold green]Proceed with {generation_type} generation?[/bold green]")


def load_story_from_file(rel_path: str) -> str | None:
    """
    Load story content from the default story file.
    Returns:
        str | None: The story content, or None if the file doesn't exist.
    """
    story_file = Path(rel_path)
    if not story_file.exists():
        print(f"[yellow]Warning:[/yellow] Story file {story_file} does not exist.")
        return None
    with open(story_file, encoding="utf-8") as f:
        story = f.read().strip()
    return story


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The story prompt to generate from (positional, required)"),
    length: str = typer.Option("bedtime", "--length", "-l", help="Story length (flash, short, medium, bedtime)"),
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
        help=("Learning focus (counting, colors, letters, emotions, nature). Default: None (no learning focus)"),
    ),
    setting: str | None = typer.Option(None, "--setting", help="Story setting"),
    characters: Annotated[
        list[str] | None,
        typer.Option("--character", help="Character names/descriptions (multi-use)"),
    ] = None,
    image_style: str = typer.Option(
        "chibi",
        "--image-style",
        help="Image art style (chibi, realistic, cartoon, watercolor, sketch). Default: chibi",
    ),
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode (use local file instead of backend for story generation)"
    ),
):
    if debug:
        verbose = True  # Ensure verbose is enabled in debug mode

    if prompt is None or not str(prompt).strip():
        console.print("[red]Error:[/red] Please provide a non-empty story prompt.", style="bold")
        raise typer.Exit(1)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_default_output_dir()
        console.print(f"[bold blue]📁 Generated output directory:[/bold blue] {output_dir}")

    try:
        # Initialize backend
        if verbose:
            console.print("[dim]Initializing AI backend...[/dim]")

        backend = get_backend()
        backend_name = backend.name

        if verbose:
            # Show which backend was selected
            console.print(f"[dim]Using {backend_name} backend[/dim]")

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
            image_style=image_style,
            generation_type="story",
            backend_name=backend_name,
        ):
            console.print("[yellow]Story generation cancelled.[/yellow]")
            raise typer.Exit(0)

        # Load context files if --use-context is enabled (default).
        try:
            characters_list = characters if characters else None
            theme_value = theme if theme else None
            learning_focus_value = learning_focus if learning_focus else None

            if verbose:
                console.print(f"[cyan][DEBUG] use_context flag is set to: {use_context}[/cyan]")
            if use_context:
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
                                console.print(f"[yellow]Warning: Could not read {fpath}: {e}[/yellow]")
                    context = "\n".join(contents) if contents else None
                else:
                    context = None
            else:
                context = None
                if verbose:
                    console.print("[cyan][DEBUG] Context loading skipped due to --no-use-context[/cyan]")

            if verbose:
                console.print(f"[cyan][DEBUG] Context value before Prompt creation: {repr(context)}[/cyan]")

            story_prompt = Prompt(
                prompt=prompt,
                context=context,
                length=length,
                age_range=age_range,
                style=style,
                tone=tone,
                theme=theme_value,
                setting=setting,
                characters=characters_list,
                learning_focus=learning_focus_value,
                image_style=image_style,
            )

            if verbose:
                console.print("[dim]Created prompt with parameters:[/dim]")
                console.print(f"[dim]  Length: {story_prompt.length}[/dim]")
                console.print(f"[dim]  Age Range: {story_prompt.age_range}[/dim]")
                console.print(f"[dim]  Style: {story_prompt.style}[/dim]")
                console.print(f"[dim]  Tone: {story_prompt.tone}[/dim]")
                console.print(f"[dim]  Theme: {story_prompt.theme}[/dim]")
                console.print(f"[dim]  Learning Focus: {story_prompt.learning_focus}[/dim]")
                console.print(f"[cyan][DEBUG] About to call generate_story (debug={debug})[/cyan]")

        except ValueError as e:
            console.print(f"[red]Error:[/red] Invalid parameter value: {e}", style="bold")
            raise typer.Exit(1) from e

        # Refinement loop
        refinements = None
        accepted = False
        while not accepted:
            # Update prompt with refinements if any
            story_prompt = Prompt(
                prompt=prompt,
                context=context,
                length=length,
                age_range=age_range,
                style=style,
                tone=tone,
                theme=theme_value,
                setting=setting,
                characters=characters_list,
                learning_focus=learning_focus_value,
                image_style=image_style,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating story..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("story", total=None)
                if debug:
                    story = load_story_from_file("storyforge/test_story.txt")
                    console.print("[cyan][DEBUG] load_story_from_file was called and returned.[/cyan]")
                else:
                    story = backend.generate_story(story_prompt)
                    if verbose:
                        console.print("[cyan][DEBUG] generate_story was called and returned.[/cyan]")

            if story is None or story == "[Error generating story]":
                console.print(
                    "[red]Error:[/red] Failed to generate story. Please check your API key and try again.",
                    style="bold",
                )
                raise typer.Exit(1)

            # Display the generated story
            console.print("\n[bold green]Generated Story:[/bold green]")
            console.print(f"[dim]Prompt:[/dim] {prompt}")
            if refinements:
                console.print(f"[dim]Refinements:[/dim] {refinements}")
            console.print()
            console.print(story)
            console.print()

            # Ask if user wants to refine
            # Custom refinement confirmation: treat empty <cr> or 'n' as "no"
            if Confirm.ask(
                "[bold yellow]Would you like to refine the story?[/bold yellow]",
                default=False,
                show_default=True,
            ):
                ref_base = "Keep the story as similar as possible, but apply the following refinements."
                ref_base += " After incorporating these changes, please generate a new version of the "
                ref_base += "story while making sure to honor all concepts of good storytelling: \n\n{}"
                refinements = typer.prompt("Refinements:")
                prompt = ref_base.format(refinements, story)
            else:
                accepted = True

        # Always save the story and print the message before image generation
        story_filename = "story.txt"
        story_path = os.path.join(output_dir, story_filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(f"Story: {prompt}\n\n")
            f.write(story if story is not None else "")
        console.print(f"[bold green]✅ Story saved as:[/bold green] {story_path}")

        # At this point, story is guaranteed to be a string (not None)
        assert isinstance(story, str), "Story must be a string at this point"

        # Present image generation options using Confirm.ask for test compatibility
        if Confirm.ask("Would you like to generate illustrations for the story?"):
            # Ask how many images to generate
            num_images = typer.prompt("How many images would you like to generate?", type=int, default=1)

            if num_images > 0:
                msg = f"Generating {num_images} image{'s' if num_images > 1 else ''}..."
                console.print(f"[bold blue]{msg}[/bold blue]")

                reference_image_bytes = None
                for i in range(num_images):
                    console.print(f"[dim]Generating image {i + 1} of {num_images}...[/dim]")

                    with Progress(
                        SpinnerColumn(),
                        TextColumn(f"[bold blue]Creating illustration {i + 1}..."),
                        console=console,
                        transient=True,
                    ) as progress:
                        progress.add_task("image", total=None)

                        # Use story as prompt for image generation
                        # Add qualifier when generating multiple images
                        story_prompt_for_image = story
                        if num_images > 1:
                            # Generate qualifier based on image position
                            ordinals = ["first", "second", "third", "fourth", "fifth"]
                            fractions = {
                                2: ["first half", "second half"],
                                3: ["first third", "second third", "third third"],
                                4: ["first quarter", "second quarter", "third quarter", "fourth quarter"],
                                5: ["first fifth", "second fifth", "third fifth", "fourth fifth", "fifth fifth"],
                            }

                            if num_images in fractions and i < len(fractions[num_images]):
                                fraction = fractions[num_images][i]
                                qualifier = f"This image should illustrate the {fraction} of the story. "
                            else:
                                # Fallback for unsupported numbers or edge cases
                                ordinal = ordinals[i] if i < len(ordinals) else f"{i + 1}th"
                                qualifier = (
                                    f"This image should illustrate the {ordinal} part of the story "
                                    f"(part {i + 1} of {num_images}). "
                                )

                            story_prompt_for_image = qualifier + story

                        image_prompt = Prompt(
                            prompt=story_prompt_for_image,
                            context=context,
                            length=length,
                            age_range=age_range,
                            style=style,
                            tone=tone,
                            theme=theme_value,
                            setting=setting,
                            characters=characters_list,
                            learning_focus=learning_focus_value,
                            image_style=image_style,
                        )

                        # Generate image with reference for consistency (if available)
                        image, image_bytes = backend.generate_image(image_prompt, reference_image_bytes)

                    if image and image_bytes:
                        # Generate filename
                        image_name = backend.generate_image_name(image_prompt, story)
                        image_filename = f"{image_name}_{i + 1:02d}.png" if num_images > 1 else f"{image_name}.png"
                        image_path = os.path.join(output_dir, image_filename)

                        # Save image
                        with open(image_path, "wb") as f:  # type: ignore[assignment]
                            f.write(image_bytes)  # type: ignore[arg-type]
                        console.print(f"[bold green]✅ Image {i + 1} saved as:[/bold green] {image_path}")

                        # Use first image as reference for subsequent ones
                        if i == 0:
                            reference_image_bytes = image_bytes
                    else:
                        console.print(f"[red]❌ Failed to generate image {i + 1}[/red]")
            else:
                console.print("[yellow]No images will be generated.[/yellow]")
        else:
            console.print("[yellow]Image generation skipped by user.[/yellow]")

        # Always ask if user wants to save as future context after story generation
        save_as_context = Confirm.ask(
            "[bold blue]Save this story as future context for character development?[/bold blue]"
        )
        if save_as_context:
            context_dir = Path(user_data_dir("StoryForge", "StoryForge")) / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            context_filename = f"story_{timestamp}.md"
            context_path = Path(context_dir) / context_filename
            print("DEBUG EXIT REACHED")
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
                    f.write(story if story is not None else "")
                    f.write("\n\n## Usage Notes\n\n")
                    f.write("This story can be referenced for:\n")
                    f.write("- Character consistency in future stories\n")
                    f.write("- Setting and world-building continuity\n")
                    f.write("- Tone and style reference\n")
                    f.write("- Educational content alignment\n")
                console.print(f"[bold green]✅ Context saved as:[/bold green] {context_path}")
            except Exception as e:
                console.print(f"[red]Error saving context file:[/red] {e}", style="bold")

    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            console.print(
                "[red]Error:[/red] GEMINI_API_KEY environment variable not set.",
                style="bold",
            )
            console.print("[dim]Please set your Gemini API key: export GEMINI_API_KEY=your_key_here[/dim]")
        elif "ANTHROPIC_API_KEY" in str(e):
            console.print(
                "[red]Error:[/red] ANTHROPIC_API_KEY environment variable not set.",
                style="bold",
            )
            console.print("[dim]Please set your Anthropic API key: export ANTHROPIC_API_KEY=your_key_here[/dim]")
        else:
            console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from e

    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error:[/red] {e}", style="bold")
        else:
            console.print(
                "[red]Error:[/red] An unexpected error occurred. Use --verbose for details.",
                style="bold",
            )
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
