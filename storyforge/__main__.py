"""Module entry point for ``python -m storyforge``."""


def main() -> None:
    """Invoke the Typer command-line application."""
    from storyforge.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
