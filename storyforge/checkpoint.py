"""
Checkpoint system for StoryForge execution state persistence.

This module provides checkpoint functionality allowing users to resume
StoryForge execution from any phase using the --continue CLI parameter.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_data_dir
from rich.console import Console
from rich.prompt import Confirm, IntPrompt

console = Console()


class ExecutionPhase(Enum):
    """Enumeration of StoryForge execution phases."""

    INIT = "init"
    CONFIG_LOAD = "config_load"
    BACKEND_INIT = "backend_init"
    PROMPT_CONFIRM = "prompt_confirm"
    CONTEXT_LOAD = "context_load"
    PROMPT_BUILD = "prompt_build"
    STORY_GENERATE = "story_generate"
    STORY_SAVE = "story_save"
    IMAGE_DECISION = "image_decision"
    IMAGE_GENERATE = "image_generate"
    CONTEXT_SAVE = "context_save"
    COMPLETED = "completed"


class SessionStatus(Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GeneratedImage:
    """Data structure for generated image information."""

    filename: str
    path: str
    generated_at: str | None = None
    size_bytes: int | None = None


@dataclass
class CheckpointData:
    """Checkpoint data structure for StoryForge execution state."""

    # Session metadata
    session_id: str
    created_at: str
    updated_at: str
    status: str

    # Execution state
    current_phase: str
    completed_phases: list[str]

    # Original inputs
    original_inputs: dict[str, Any]

    # Resolved configuration
    resolved_config: dict[str, Any]

    # Generated content
    generated_content: dict[str, Any]

    # User decisions
    user_decisions: dict[str, Any]

    # Context data
    context_data: dict[str, Any] | None = None

    # Progress tracking
    progress: dict[str, Any] | None = None

    # Error information
    last_error: str | None = None
    recovery_possible: bool = True

    @classmethod
    def create_new(
        cls,
        original_prompt: str,
        cli_arguments: dict[str, Any],
        resolved_config: dict[str, Any],
    ) -> "CheckpointData":
        """Create a new checkpoint data structure."""
        now = datetime.now().isoformat() + "Z"
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sf"

        return CheckpointData(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            status=SessionStatus.ACTIVE.value,
            current_phase=ExecutionPhase.INIT.value,
            completed_phases=[],
            original_inputs={
                "prompt": original_prompt,
                "cli_arguments": cli_arguments,
            },
            resolved_config=resolved_config,
            generated_content={
                "story": None,
                "refinements": None,
                "images": [],
            },
            user_decisions={
                "story_accepted": None,
                "wants_images": None,
                "num_images_requested": None,
                "save_as_context": None,
            },
            context_data=None,
            progress={
                "total_phases": len(ExecutionPhase) - 1,  # Exclude COMPLETED
                "completed_count": 0,
                "completion_percentage": 0,
            },
        )

    def update_phase(self, new_phase: ExecutionPhase) -> None:
        """Update the current phase and mark previous as completed."""
        if self.current_phase not in self.completed_phases:
            self.completed_phases.append(self.current_phase)

        self.current_phase = new_phase.value
        self.updated_at = datetime.now().isoformat() + "Z"

        # Update progress
        if self.progress:
            self.progress["completed_count"] = len(self.completed_phases)
            self.progress["completion_percentage"] = round(
                (len(self.completed_phases) / self.progress["total_phases"]) * 100
            )

    def mark_completed(self) -> None:
        """Mark the session as completed."""
        self.status = SessionStatus.COMPLETED.value
        self.current_phase = ExecutionPhase.COMPLETED.value
        if self.progress:
            self.progress["completion_percentage"] = 100
        self.updated_at = datetime.now().isoformat() + "Z"

    def mark_failed(self, error_message: str) -> None:
        """Mark the session as failed with error information."""
        self.status = SessionStatus.FAILED.value
        self.last_error = error_message
        self.updated_at = datetime.now().isoformat() + "Z"

    def get_prompt_preview(self, max_length: int = 50) -> str:
        """Get a truncated preview of the original prompt."""
        prompt_value = self.original_inputs.get("prompt", "")
        prompt = str(prompt_value) if prompt_value is not None else ""
        if len(prompt) <= max_length:
            return prompt
        return prompt[: max_length - 3] + "..."


class CheckpointManager:
    """Manages checkpoint persistence and recovery for StoryForge."""

    def __init__(self, auto_cleanup: bool = True) -> None:
        """Initialize checkpoint manager with default storage location."""
        # Use lowercase 'storyforge' for normalized cross-platform paths
        self.checkpoint_dir = Path(user_data_dir("storyforge", "storyforge")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Perform automatic cleanup on initialization
        if auto_cleanup:
            self.auto_cleanup_on_start()

    def save_checkpoint(self, checkpoint_data: CheckpointData) -> Path:
        """Save checkpoint data to YAML file."""
        filename = f"checkpoint_{checkpoint_data.session_id}.yaml"
        checkpoint_path = self.checkpoint_dir / filename

        try:
            # Convert dataclass to dict for YAML serialization
            data_dict = asdict(checkpoint_data)

            # Add header comment
            yaml_content = f"# StoryForge Checkpoint - Session {checkpoint_data.session_id}\n"
            yaml_content += f"# Generated: {checkpoint_data.updated_at}\n\n"

            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
                yaml.dump(data_dict, f, default_flow_style=False, indent=2, sort_keys=False)

            return checkpoint_path

        except Exception as e:
            console.print(f"[red]Error saving checkpoint:[/red] {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> CheckpointData:
        """Load checkpoint data from YAML file."""
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            with open(checkpoint_path, encoding="utf-8") as f:
                # Skip comment lines at the beginning
                lines = f.readlines()
                yaml_lines = [line for line in lines if not line.strip().startswith("#")]
                yaml_content = "".join(yaml_lines)

            data_dict = yaml.safe_load(yaml_content)

            if not data_dict:
                raise ValueError(f"Empty or invalid checkpoint file: {checkpoint_path}")

            # Validate required fields
            required_fields = ["session_id", "status", "current_phase", "original_inputs"]
            missing_fields = [field for field in required_fields if field not in data_dict]
            if missing_fields:
                raise ValueError(f"Checkpoint missing required fields: {missing_fields}")

            return CheckpointData(**data_dict)

        except FileNotFoundError:
            console.print(f"[red]Checkpoint file not found:[/red] {checkpoint_path}")
            raise
        except yaml.YAMLError as e:
            console.print(f"[red]Invalid YAML in checkpoint file {checkpoint_path}:[/red] {e}")
            raise
        except (ValueError, TypeError) as e:
            console.print(f"[red]Invalid checkpoint data in {checkpoint_path}:[/red] {e}")
            raise
        except Exception as e:
            console.print(f"[red]Unexpected error loading checkpoint {checkpoint_path}:[/red] {e}")
            raise

    def find_recent_checkpoints(self, limit: int = 5) -> list[Path]:
        """Find the most recent checkpoint files."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            # Sort by modification time, newest first
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return checkpoint_files[:limit]

        except Exception as e:
            console.print(f"[red]Error finding checkpoints:[/red] {e}")
            return []

    def get_checkpoint_info(self, checkpoint_path: Path) -> dict[str, Any]:
        """Get quick info about a checkpoint without full loading."""
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                content = f.read()

            data = yaml.safe_load(content)

            return {
                "path": checkpoint_path,
                "session_id": data.get("session_id", "unknown"),
                "created_at": data.get("created_at", ""),
                "status": data.get("status", "unknown"),
                "current_phase": data.get("current_phase", "unknown"),
                "prompt_preview": data.get("original_inputs", {}).get("prompt", "")[:50],
                "completion_percentage": data.get("progress", {}).get("completion_percentage", 0),
            }

        except Exception:
            return {
                "path": checkpoint_path,
                "session_id": checkpoint_path.stem,
                "created_at": "",
                "status": "unknown",
                "current_phase": "unknown",
                "prompt_preview": "Error reading checkpoint",
                "completion_percentage": 0,
            }

    def cleanup_old_checkpoints(self, keep_recent: int = 10) -> None:
        """Remove old checkpoint files, keeping only the most recent."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove files beyond the keep limit
            for old_file in checkpoint_files[keep_recent:]:
                try:
                    old_file.unlink()
                    console.print(f"[dim]Cleaned up old checkpoint: {old_file.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Could not remove {old_file.name}: {e}[/yellow]")

        except Exception as e:
            console.print(f"[yellow]Error during checkpoint cleanup: {e}[/yellow]")

    def prompt_checkpoint_selection(self) -> CheckpointData | None:
        """Prompt user to select from available checkpoints."""
        recent_checkpoints = self.find_recent_checkpoints(5)

        if not recent_checkpoints:
            console.print("[yellow]No previous StoryForge sessions found.[/yellow]")
            return None

        console.print("\n[bold cyan]Recent StoryForge sessions:[/bold cyan]")
        checkpoint_infos = []

        for i, checkpoint_path in enumerate(recent_checkpoints, 1):
            info = self.get_checkpoint_info(checkpoint_path)
            checkpoint_infos.append(info)

            status_color = {
                "active": "red",
                "failed": "red",
                "completed": "green",
            }.get(info["status"], "yellow")

            # Format datetime for display
            try:
                dt = datetime.fromisoformat(info["created_at"].replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = "Unknown time"

            console.print(
                f'  {i}. {time_str} - "{info["prompt_preview"]}" '
                f"([{status_color}]{info['status'].upper()}[/{status_color}] at {info['current_phase']})"
            )

        console.print()

        if len(checkpoint_infos) == 1:
            if Confirm.ask("Continue from this session?"):
                try:
                    return self.load_checkpoint(checkpoint_infos[0]["path"])
                except yaml.YAMLError as e:
                    console.print(f"[red]Invalid YAML in checkpoint file {checkpoint_infos[0]['path']}:[/red] {e}")
                    if Confirm.ask("This checkpoint appears corrupted. Move it to a .corrupt file and continue?"):
                        path = checkpoint_infos[0]["path"]
                        corrupt_path = path.with_suffix(path.suffix + ".corrupt")
                        try:
                            path.rename(corrupt_path)
                            console.print(f"[dim]Moved corrupted checkpoint to {corrupt_path}[/dim]")
                        except Exception as ex:
                            console.print(f"[yellow]Could not move corrupted file: {ex}[/yellow]")
                    return None
                except Exception:
                    # For any other error, do not abort the whole flow
                    console.print("[yellow]Could not load selected checkpoint. Skipping.[/yellow]")
                    return None
            return None

        try:
            selection = IntPrompt.ask(
                "Select session to continue",
                choices=[str(i) for i in range(1, len(checkpoint_infos) + 1)] + ["q"],
                default="q",
            )

            if str(selection) == "q":
                return None

            # IntPrompt.ask returns int, so we need to handle the conversion
            selection_int = int(selection)
            selected_info = checkpoint_infos[selection_int - 1]
            try:
                return self.load_checkpoint(selected_info["path"])
            except yaml.YAMLError as e:
                console.print(f"[red]Invalid YAML in checkpoint file {selected_info['path']}:[/red] {e}")
                if Confirm.ask("This checkpoint appears corrupted. Move it to a .corrupt file and continue?"):
                    path = selected_info["path"]
                    corrupt_path = path.with_suffix(path.suffix + ".corrupt")
                    try:
                        path.rename(corrupt_path)
                        console.print(f"[dim]Moved corrupted checkpoint to {corrupt_path}[/dim]")
                    except Exception as ex:
                        console.print(f"[yellow]Could not move corrupted file: {ex}[/yellow]")
                return None
            except Exception:
                console.print("[yellow]Could not load selected checkpoint. Skipping.[/yellow]")
                return None

        except (ValueError, KeyboardInterrupt):
            console.print("[yellow]Selection cancelled.[/yellow]")
            return None

    def prompt_phase_selection(self, checkpoint_data: CheckpointData) -> ExecutionPhase | None:
        """Prompt user to select which phase to resume from for completed sessions."""
        if checkpoint_data.status != SessionStatus.COMPLETED.value:
            # For active/failed sessions, resume from current phase
            try:
                return ExecutionPhase(checkpoint_data.current_phase)
            except ValueError:
                # If phase is not recognized, it's an old incompatible checkpoint
                raise ValueError("Incompatible checkpoint format - please start a new session") from None

        console.print(f'\n[bold]Selected completed session:[/bold] "{checkpoint_data.get_prompt_preview()}"')
        console.print("\n[bold cyan]Choose phase to resume from:[/bold cyan]")
        console.print("  1. Generate new images (with same story)")
        console.print("  2. Modify story and regenerate")
        console.print("  3. Save story as context")
        console.print("  4. Start completely over with same parameters")

        try:
            choice = IntPrompt.ask("Select option", choices=["1", "2", "3", "4", "q"], default="q")

            if str(choice) == "q":
                return None
            elif choice == 1:
                return ExecutionPhase.IMAGE_DECISION
            elif choice == 2:
                return ExecutionPhase.STORY_GENERATE
            elif choice == 3:
                return ExecutionPhase.CONTEXT_SAVE
            elif choice == 4:
                return ExecutionPhase.INIT
            else:
                return None

        except (ValueError, KeyboardInterrupt):
            console.print("[yellow]Selection cancelled.[/yellow]")
            return None

    def get_checkpoint_stats(self) -> dict[str, int]:
        """
        Get statistics about checkpoint files.

        Returns:
            dict: Statistics including total count, completed count, failed count
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))

            stats = {
                "total": len(checkpoint_files),
                "completed": 0,
                "failed": 0,
                "active": 0,
            }

            for file_path in checkpoint_files:
                try:
                    info = self.get_checkpoint_info(file_path)
                    status = info.get("status", "unknown")
                    if status == "completed":
                        stats["completed"] += 1
                    elif status == "failed":
                        stats["failed"] += 1
                    elif status == "active":
                        stats["active"] += 1
                except Exception:
                    # Skip files that can't be loaded
                    continue

            return stats

        except Exception:
            # Return empty stats if operation fails
            return {"total": 0, "completed": 0, "failed": 0, "active": 0}

    def cleanup_failed_sessions(self) -> int:
        """
        Remove all failed checkpoint sessions.

        Returns:
            int: Number of failed sessions cleaned up
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            deleted_count = 0

            for file_path in checkpoint_files:
                try:
                    info = self.get_checkpoint_info(file_path)
                    if info.get("status") == "failed":
                        file_path.unlink()
                        deleted_count += 1
                        console.print(f"[dim]Removed failed session: {file_path.name}[/dim]")
                except Exception:
                    # Skip files that can't be processed
                    continue

            return deleted_count

        except Exception:
            return 0

    def cleanup_stale_active_sessions(self, max_age_hours: int = 24) -> int:
        """
        Mark active sessions older than max_age_hours as FAILED (abandoned).

        Args:
            max_age_hours: Maximum age in hours for an active session to be considered valid

        Returns:
            int: Number of stale sessions cleaned up
        """
        from datetime import datetime, timedelta

        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            stale_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            for file_path in checkpoint_files:
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        checkpoint = self.load_checkpoint(file_path)
                        if checkpoint.status == "active":
                            checkpoint.mark_failed("Session abandoned (stale)")
                            self.save_checkpoint(checkpoint)
                            stale_count += 1
                            console.print(f"[dim]Marked stale session as failed: {file_path.name}[/dim]")
                except Exception:
                    continue

            return stale_count

        except Exception:
            return 0

    def auto_cleanup_on_start(self) -> None:
        """
        Perform automatic cleanup when CheckpointManager is initialized.
        Keeps the 15 most recent checkpoints, marks stale active sessions as failed,
        and removes old failed sessions.
        """
        try:
            # Mark stale active sessions as failed (older than 24 hours)
            self.cleanup_stale_active_sessions(max_age_hours=24)

            # Clean up old checkpoints (keep 15 most recent)
            self.cleanup_old_checkpoints(keep_recent=15)

            # Clean up failed sessions that are older than 1 day
            from datetime import datetime, timedelta

            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.yaml"))
            one_day_ago = datetime.now() - timedelta(days=1)

            for file_path in checkpoint_files:
                try:
                    # Check if file is older than 1 day
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < one_day_ago:
                        info = self.get_checkpoint_info(file_path)
                        if info.get("status") == "failed":
                            file_path.unlink()
                            console.print(f"[dim]Auto-cleanup: removed old failed session {file_path.name}[/dim]")
                except Exception:
                    continue

        except Exception:
            # Silent fail for cleanup operations
            pass
