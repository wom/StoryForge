"""Module entry point for ``python -m storyforge``."""


def main() -> None:
    """Invoke the Typer command-line application."""
    from storyforge.StoryForge import cli_entry

    cli_entry()


if __name__ == "__main__":
    main()
