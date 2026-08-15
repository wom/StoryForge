"""Path helpers shared by UI-independent StoryForge workflows."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from platformdirs import user_data_dir

from .world_template import WORLD_FILENAME


def create_output_directory_name(*, extended: bool = False) -> str:
    """Return a collision-resistant default output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_extended" if extended else ""
    return f"storyforge_output_{timestamp}_{uuid4().hex[:8]}{suffix}"


def resolve_world_file_path() -> Path:
    """Return the local or XDG location used for the StoryForge world file."""
    local_dir = Path("context")
    if local_dir.is_dir():
        return local_dir / WORLD_FILENAME

    context_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    return context_dir / WORLD_FILENAME
