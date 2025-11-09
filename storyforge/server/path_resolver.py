"""Path resolution utilities for MCP server."""

from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir


class PathResolver:
    """Resolve and manage file paths for StoryForge server."""

    def __init__(self, output_directory: str | None = None):
        """Initialize path resolver with optional output directory override."""
        # XDG-compliant directories
        self.config_dir = Path(user_config_dir("storyforge-server", ensure_exists=True))
        self.data_dir = Path(user_data_dir("storyforge-server", ensure_exists=True))
        self.cache_dir = Path(user_cache_dir("storyforge-server", ensure_exists=True))

        # Create subdirectories
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.context_dir = self.data_dir / "context"
        self.context_dir.mkdir(exist_ok=True)

        self.stories_dir = self.data_dir / "stories"
        self.stories_dir.mkdir(exist_ok=True)

        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.context_summaries_dir = self.cache_dir / "context_summaries"
        self.context_summaries_dir.mkdir(exist_ok=True)

        # Output directory (can be overridden)
        self._output_directory = Path(output_directory) if output_directory else self.data_dir

    @property
    def output_directory(self) -> Path:
        """Get the configured output directory."""
        return self._output_directory

    def resolve_path(self, path_str: str, base_dir: Path | None = None) -> Path:
        """
        Resolve a path string to an absolute Path.

        Rules:
        1. Absolute paths → use as-is
        2. Home paths (~) → expand to user home
        3. Relative paths → relative to base_dir (or output_directory if None)

        Args:
            path_str: Path string to resolve
            base_dir: Base directory for relative paths (defaults to output_directory)

        Returns:
            Absolute Path object

        Raises:
            ValueError: If path is invalid
        """
        if not path_str:
            raise ValueError("Path string cannot be empty")

        path = Path(path_str)

        # Absolute paths
        if path.is_absolute():
            return path.resolve()

        # Home paths
        if str(path).startswith("~"):
            return path.expanduser().resolve()

        # Relative paths
        base = base_dir if base_dir else self._output_directory
        return (base / path).resolve()

    def ensure_parent_dir(self, path: Path) -> None:
        """Ensure the parent directory of a path exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, session_id: str) -> Path:
        """Get the checkpoint file path for a session."""
        return self.checkpoints_dir / f"{session_id}.yaml"

    def get_context_path(self, filename: str) -> Path:
        """Get the context file path."""
        return self.context_dir / filename

    def get_story_path(self, filename: str) -> Path:
        """Get the story file path."""
        return self.stories_dir / filename

    def get_image_path(self, filename: str) -> Path:
        """Get the image file path."""
        return self.images_dir / filename

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files."""
        return sorted(self.checkpoints_dir.glob("*.yaml"))

    def list_context_files(self) -> list[Path]:
        """List all context markdown files."""
        return sorted(self.context_dir.glob("*.md"))
