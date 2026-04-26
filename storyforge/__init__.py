"""
StoryForge: AI-powered illustrated children's story generator.

A CLI/TUI tool that generates illustrated children's stories using multiple AI backends
(Gemini, OpenAI, Anthropic) with schema-driven configuration and checkpoint support.

Main Features:
- Schema-driven configuration system
- Phase-based execution with checkpoints
- Multiple AI backend support (Gemini, OpenAI, Anthropic)
- Context management for character consistency
- World definitions for persistent world-building
- Voice archetypes for narrator style control
- Resumable execution via `storyforge continue`

CLI Usage:
    $ storyforge "A brave mouse goes on an adventure"
    $ storyforge continue  # Resume from checkpoint
    $ storyforge extend    # Extend a previous story
    $ python -m storyforge "A brave mouse goes on an adventure"
"""

# Main CLI interface
from .StoryForge import app

# Version info
__version__ = "0.0.8"
__author__ = "Chris (wom)"

# Main exports
__all__ = [
    "app",  # Main CLI application
]
