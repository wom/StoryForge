"""Shared Rich console instance for StoryForge.

All modules should import console from here instead of creating their own
Console() instances, ensuring consistent output behavior.
"""

from rich.console import Console

console = Console()
