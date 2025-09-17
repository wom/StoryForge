"""
StoryForge: Simplified CLI for generating illustrated stories using multiple LLM backends.
Supports Google Gemini and Anthropic Claude backends.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm

from .checkpoint import CheckpointManager
from .config import Config, ConfigError, load_config
from .phase_executor import PhaseExecutor
from .schema.cli_integration import (
    generate_boolean_cli_option,
    generate_cli_option,
    generate_multi_option,
    validate_cli_arguments,
)

console = Console()

# Create Typer app instance for entrypoint
app = typer.Typer(
    help="StoryForge: Generate illustrated stories using AI language models.\n\n"
    "Configuration: Use --init-config to create a config file with default values.\n"
    "Environment: Set STORYFORGE_CONFIG to use a custom config file location.\n\n"
    "Checkpoint System: StoryForge automatically saves progress during execution.\n"
    "Use --continue to resume from previous sessions or retry from any completed phase.\n"
    "Checkpoint files are stored in ~/.local/share/StoryForge/checkpoints/"
)


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


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def main(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(
        None,
        help="The story prompt to generate from (positional, required unless using --init-config or --continue)",
    ),
    length: str | None = generate_cli_option("length"),
    age_range: str | None = generate_cli_option("age_range"),
    style: str | None = generate_cli_option("style"),
    tone: str | None = generate_cli_option("tone"),
    theme: str | None = generate_cli_option("theme"),
    learning_focus: str | None = generate_cli_option("learning_focus"),
    setting: str | None = generate_cli_option("setting"),
    characters: Annotated[
        list[str] | None,
        generate_multi_option("characters", "--character"),
    ] = None,
    image_style: str | None = generate_cli_option("image_style"),
    output_dir: str | None = generate_cli_option("output_dir"),
    use_context: bool | None = generate_boolean_cli_option("use_context", "--use-context/--no-use-context"),
    verbose: bool | None = generate_cli_option("verbose"),
    debug: bool | None = generate_cli_option("debug"),
    backend: str | None = generate_cli_option("backend"),
    init_config: bool = typer.Option(False, "--init-config", help="Generate a default configuration file and exit"),
    continue_session: bool = typer.Option(
        False,
        "--continue",
        help="Resume execution from a previous StoryForge session. "
        "Displays the last 5 sessions and allows you to choose which phase to resume from. "
        "For completed sessions, you can generate new images, modify the story, "
        "save as context, or restart completely with the same parameters.",
    ),
):
    # Validate CLI arguments using schema before processing
    cli_args = {
        "length": length,
        "age_range": age_range,
        "style": style,
        "tone": tone,
        "theme": theme,
        "learning_focus": learning_focus,
        "setting": setting,
        "characters": characters,
        "image_style": image_style,
        "output_dir": output_dir,
        "use_context": use_context,
        "verbose": verbose,
        "debug": debug,
        "backend": backend,
    }

    # Only validate provided CLI arguments (not None values)
    provided_args = {k: v for k, v in cli_args.items() if v is not None}
    validation_errors = validate_cli_arguments(**provided_args)

    if validation_errors:
        console.print("[red]CLI Argument Validation Errors:[/red]", style="bold")
        for error in validation_errors:
            console.print(f"  - {error}", style="red")
        raise typer.Exit(1)

    # Handle --init-config option first
    if init_config:
        try:
            config = Config()
            config_path = config.get_default_config_path()

            # Check if file exists
            if config_path.exists():
                console.print(f"[yellow]Configuration file already exists:[/yellow] {config_path}")
                console.print("[dim]Use a different location by setting STORYFORGE_CONFIG environment variable[/dim]")
                raise typer.Exit(1)

            # Create the configuration file
            created_path = config.create_default_config(config_path)
            console.print(f"[bold green]✅ Configuration file created:[/bold green] {created_path}")
            console.print()
            console.print("[bold]Configuration file locations (in priority order):[/bold]")

            for i, search_path in enumerate(config.get_config_paths(), 1):
                if search_path == created_path:
                    console.print(f"  {i}. {search_path} [bold green](created here)[/bold green]")
                else:
                    console.print(f"  {i}. {search_path}")

            console.print()
            console.print("[bold]Environment variables:[/bold]")
            console.print("  STORYFORGE_CONFIG - Override config file location")
            console.print("  GEMINI_API_KEY - Google Gemini API key")
            console.print("  OPENAI_API_KEY - OpenAI API key")
            console.print("  ANTHROPIC_API_KEY - Anthropic API key")

            raise typer.Exit(0)

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error creating configuration file:[/red] {e}", style="bold")
            raise typer.Exit(1) from None

    # Handle --continue option
    if continue_session:
        try:
            checkpoint_manager = CheckpointManager()
            checkpoint_data = checkpoint_manager.prompt_checkpoint_selection()

            if checkpoint_data is None:
                console.print("[yellow]No checkpoint selected. Exiting.[/yellow]")
                raise typer.Exit(0)

            # Get the phase to resume from
            resume_phase = checkpoint_manager.prompt_phase_selection(checkpoint_data)
            if resume_phase is None:
                console.print("[yellow]No phase selected. Exiting.[/yellow]")
                raise typer.Exit(0)

            # Execute using phase executor
            phase_executor = PhaseExecutor(checkpoint_manager)
            phase_executor.execute_from_checkpoint(checkpoint_data, resume_phase)
            raise typer.Exit(0)

        except typer.Exit:
            # Let typer.Exit propagate normally
            raise
        except Exception as e:
            console.print(f"[red]Error handling checkpoint continuation:[/red] {e}", style="bold")
            raise typer.Exit(1) from e

    try:
        # Load configuration file
        config = load_config(verbose=bool(verbose))

        # Merge configuration with CLI arguments (CLI takes precedence)
        # Only use config values if CLI argument is None (not provided)
        length = length if length is not None else config.get_field_value("story", "length")
        age_range = age_range if age_range is not None else config.get_field_value("story", "age_range")
        style = style if style is not None else config.get_field_value("story", "style")
        tone = tone if tone is not None else config.get_field_value("story", "tone")
        theme = theme if theme is not None else config.get_field_value("story", "theme")
        learning_focus = (
            learning_focus
            if learning_focus is not None
            else (config.get_field_value("story", "learning_focus") or None)
        )
        setting = setting if setting is not None else (config.get_field_value("story", "setting") or None)

        # Handle characters - merge config and CLI
        if characters is None:
            config_characters = config.get_field_value("story", "characters")
            characters = config_characters

        image_style = image_style if image_style is not None else config.get_field_value("images", "image_style")
        output_dir = output_dir if output_dir is not None else (config.get_field_value("output", "output_dir") or None)
        use_context = use_context if use_context is not None else config.get_field_value("output", "use_context")
        verbose = verbose if verbose is not None else config.get_field_value("system", "verbose")
        debug = debug if debug is not None else config.get_field_value("system", "debug")

        # Get backend from CLI or configuration (CLI takes precedence)
        config_backend = backend if backend is not None else config.get_field_value("system", "backend")

    except ConfigError as e:
        console.print(f"[red]Configuration Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}", style="bold")
        raise typer.Exit(1) from None

    if debug:
        verbose = True  # Ensure verbose is enabled in debug mode

    # Check if prompt is required (not needed for --init-config or --continue)
    if not init_config and not continue_session and (prompt is None or not str(prompt).strip()):
        # Check if user provided no arguments at all - if so, show help instead of error
        import sys

        # Get command line args, excluding the script name
        args = sys.argv[1:]

        # If no arguments provided, show help
        if not args:
            # Use the context to show help
            console.print(ctx.get_help())
            raise typer.Exit(0)

        console.print("[red]Error:[/red] Please provide a non-empty story prompt.", style="bold")
        raise typer.Exit(1)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_default_output_dir()
        console.print(f"[bold blue]📁 Generated output directory:[/bold blue] {output_dir}")

    # Use phase executor for both new sessions and checkpoint resumption
    try:
        # Prepare CLI arguments for checkpoint
        cli_arguments = {
            "length": length,
            "age_range": age_range,
            "style": style,
            "tone": tone,
            "theme": theme,
            "learning_focus": learning_focus,
            "setting": setting,
            "characters": characters,
            "image_style": image_style,
            "output_dir": output_dir,
            "use_context": use_context,
            "verbose": verbose,
            "debug": debug,
            "backend": config_backend,
        }

        # Prepare resolved configuration
        resolved_config = {
            "backend": config_backend,
            "output_directory": output_dir,
            "use_context": use_context,
            "verbose": verbose,
            "debug": debug,
            "length": length,
            "age_range": age_range,
            "style": style,
            "tone": tone,
            "theme": theme,
            "image_style": image_style,
        }

        # Execute new session with checkpointing
        checkpoint_manager = CheckpointManager()
        phase_executor = PhaseExecutor(checkpoint_manager)
        phase_executor.execute_new_session(prompt, cli_arguments, resolved_config)

    except Exception as e:
        if verbose:
            # Sanitize the error message to prevent binary data corruption
            error_msg = str(e)
            # Replace any non-printable characters with a placeholder
            sanitized_error = "".join(c if c.isprintable() or c.isspace() else "?" for c in error_msg)
            console.print(f"[red]Unexpected error:[/red] {sanitized_error}", style="bold")
        else:
            console.print(
                "[red]Error:[/red] An unexpected error occurred. Use --verbose for details.",
                style="bold",
            )
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
