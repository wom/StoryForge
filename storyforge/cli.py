"""Public command-line entry point and bare-prompt compatibility adapter."""

import sys
from collections.abc import Sequence

from .StoryForge import app

COMMANDS = frozenset({"generate", "continue", "extend", "export-chain", "config", "world", "models"})
# Do not reinterpret removed command names as story prompts. This lets Typer
# return its normal "No such command" error instead of silently accepting a
# legacy invocation.
RESERVED_COMMANDS = COMMANDS | {"main"}


def normalize_argv(argv: Sequence[str]) -> list[str]:
    """Route a bare prompt to ``generate`` without modifying process arguments."""
    args = list(argv)
    if not args:
        return ["--help"]
    if args[0].startswith("-") or args[0] in RESERVED_COMMANDS:
        return args
    return ["generate", *args]


def main(argv: Sequence[str] | None = None) -> None:
    """Run StoryForge from a console script or ``python -m storyforge``."""
    app(args=normalize_argv(sys.argv[1:] if argv is None else argv))
