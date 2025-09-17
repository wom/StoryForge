"""
StoryForge: AI-powered illustrated children's story generator.

A CLI tool that generates illustrated children's stories using multiple AI backends
(Gemini, Claude, OpenAI) with schema-driven configuration and checkpoint support.

Main Features:
- Schema-driven configuration system
- Phase-based execution with checkpoints
- Multiple AI backend support
- Context management for character consistency
- Resumable execution with --continue flag

CLI Usage:
    $ storyforge "A brave mouse goes on an adventure"
    $ storyforge --continue  # Resume from checkpoint
    $ python -m storyforge "A brave mouse goes on an adventure"
"""

# Main CLI interface
from .StoryForge import app

# Version info
__version__ = "0.0.3"
__author__ = "wom"

# Main exports
__all__ = [
    "app",  # Main CLI application
]
