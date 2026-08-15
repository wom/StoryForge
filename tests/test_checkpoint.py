"""
Tests for the checkpoint system.

Tests checkpoint data structures, persistence, recovery, and phase execution
with comprehensive coverage of all checkpoint functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from storyforge.checkpoint import CheckpointData, CheckpointManager, ExecutionPhase, SessionStatus
from storyforge.phase_executor import PhaseExecutor


class TestCheckpointData:
    """Test the CheckpointData class functionality."""

    def test_create_new_checkpoint(self):
        """Test creating a new checkpoint with all required fields."""
        prompt = "A dragon learns to fly"
        cli_args = {
            "age_range": "preschool",
            "style": "fantasy",
            "length": "short",
        }
        config = {
            "backend": "gemini",
            "verbose": True,
            "output_directory": "test_output",
        }

        checkpoint = CheckpointData.create_new(prompt, cli_args, config)

        assert checkpoint.session_id.endswith("_sf")
        assert checkpoint.status == SessionStatus.ACTIVE.value
        assert checkpoint.current_phase == ExecutionPhase.INIT.value
        assert checkpoint.completed_phases == []
        assert checkpoint.original_inputs["prompt"] == prompt
        assert checkpoint.original_inputs["cli_arguments"] == cli_args
        assert checkpoint.resolved_config == config
        assert checkpoint.generated_content["story"] is None
        assert checkpoint.user_decisions["story_accepted"] is None
        assert checkpoint.progress["total_phases"] == len(ExecutionPhase) - 1

    def test_update_phase(self):
        """Test updating checkpoint phase."""
        checkpoint = CheckpointData.create_new("test", {}, {})
        original_time = checkpoint.updated_at

        checkpoint.update_phase(ExecutionPhase.CONFIG_LOAD)

        assert checkpoint.current_phase == ExecutionPhase.CONFIG_LOAD.value
        assert ExecutionPhase.INIT.value in checkpoint.completed_phases
        assert checkpoint.updated_at != original_time
        assert checkpoint.progress["completed_count"] == 1

    def test_mark_completed(self):
        """Test marking checkpoint as completed."""
        checkpoint = CheckpointData.create_new("test", {}, {})

        checkpoint.mark_completed()

        assert checkpoint.status == SessionStatus.COMPLETED.value
        assert checkpoint.current_phase == ExecutionPhase.COMPLETED.value
        assert checkpoint.progress["completion_percentage"] == 100

    def test_mark_failed(self):
        """Test marking checkpoint as failed with error message."""
        checkpoint = CheckpointData.create_new("test", {}, {})
        error_msg = "Backend connection failed"

        checkpoint.mark_failed(error_msg)

        assert checkpoint.status == SessionStatus.FAILED.value
        assert checkpoint.last_error == error_msg


class TestCheckpointManager:
    """Test the CheckpointManager class functionality."""

    def test_init_creates_directory(self):
        """Test that CheckpointManager creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                expected_dir = Path(tmpdir) / "checkpoints"
                assert expected_dir.exists()
                assert manager.checkpoint_dir == expected_dir

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                checkpoint = CheckpointData.create_new("test story", {"style": "adventure"}, {"backend": "gemini"})

                # Save checkpoint
                saved_path = manager.save_checkpoint(checkpoint)
                assert saved_path.exists()
                assert saved_path.name.startswith("checkpoint_")
                assert saved_path.suffix == ".yaml"

                # Load checkpoint
                loaded_checkpoint = manager.load_checkpoint(saved_path)
                assert loaded_checkpoint.session_id == checkpoint.session_id
                assert loaded_checkpoint.original_inputs["prompt"] == "test story"
                assert loaded_checkpoint.status == SessionStatus.ACTIVE.value

    def test_yaml_format_with_comments(self):
        """Test that saved YAML includes header comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                checkpoint = CheckpointData.create_new("test", {}, {})

                saved_path = manager.save_checkpoint(checkpoint)

                with open(saved_path, encoding="utf-8") as f:
                    content = f.read()

                assert f"# StoryForge Checkpoint - Session {checkpoint.session_id}" in content
                assert "# Generated:" in content

    def test_find_recent_checkpoints(self):
        """Test finding recent checkpoints sorted by modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)

                # Create multiple checkpoints
                checkpoints = []
                for i in range(3):
                    checkpoint = CheckpointData.create_new(f"test {i}", {}, {})
                    checkpoint.session_id = f"test_{i:02d}_sf"  # Override for predictable names
                    saved_path = manager.save_checkpoint(checkpoint)
                    checkpoints.append(saved_path)

                # Find recent checkpoints
                recent = manager.find_recent_checkpoints(2)
                assert len(recent) == 2
                # Should be sorted by modification time (newest first)
                assert all(path.name.startswith("checkpoint_") for path in recent)

    def test_get_checkpoint_info(self):
        """Test getting checkpoint information without full loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                checkpoint = CheckpointData.create_new("test prompt for info", {"style": "comedy"}, {})
                saved_path = manager.save_checkpoint(checkpoint)

                info = manager.get_checkpoint_info(saved_path)

                assert info["session_id"] == checkpoint.session_id
                assert info["status"] == SessionStatus.ACTIVE.value
                assert info["current_phase"] == ExecutionPhase.INIT.value
                assert info["prompt_preview"] == "test prompt for info"
                assert info["completion_percentage"] == 0

    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)

                # Create multiple checkpoints
                paths = []
                for i in range(5):
                    checkpoint = CheckpointData.create_new(f"test {i}", {}, {})
                    checkpoint.session_id = f"test_{i:02d}_sf"
                    path = manager.save_checkpoint(checkpoint)
                    paths.append(path)

                # Cleanup, keeping only 2 most recent
                manager.cleanup_old_checkpoints(keep_recent=2)

                # Check that only 2 files remain
                remaining_files = list(manager.checkpoint_dir.glob("checkpoint_*.yaml"))
                assert len(remaining_files) == 2

    def test_auto_cleanup_on_init(self):
        """Test that auto cleanup runs on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                # Create manager without auto cleanup first
                manager_no_cleanup = CheckpointManager(auto_cleanup=False)

                # Create many checkpoints
                for i in range(20):
                    checkpoint = CheckpointData.create_new(f"test {i}", {}, {})
                    checkpoint.session_id = f"test_{i:02d}_sf"
                    manager_no_cleanup.save_checkpoint(checkpoint)

                # Verify all files exist
                all_files = list(manager_no_cleanup.checkpoint_dir.glob("checkpoint_*.yaml"))
                assert len(all_files) == 20

                # Create new manager with auto cleanup (default)
                CheckpointManager()  # Should trigger cleanup

                # Verify files were cleaned up (should keep 15 most recent)
                remaining_files = list(manager_no_cleanup.checkpoint_dir.glob("checkpoint_*.yaml"))
                assert len(remaining_files) <= 15


class TestPhaseExecutor:
    """Test the PhaseExecutor class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = MagicMock()
        self.phase_executor = PhaseExecutor(self.checkpoint_manager)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_session_ids_are_unique(self):
        """Rapid checkpoint creation must not overwrite a previous session."""
        session_ids = {CheckpointData.create_new("test", {}, {}).session_id for _ in range(20)}
        assert len(session_ids) == 20

    def test_should_skip_phase_skips_completed_phases(self):
        """Test that phases are skipped when marked as completed in the current session."""
        # Create a checkpoint with all phases marked as completed
        checkpoint_data = CheckpointData.create_new("test", {}, {})
        checkpoint_data.completed_phases = [
            ExecutionPhase.INIT.value,
            ExecutionPhase.CONFIG_LOAD.value,
            ExecutionPhase.BACKEND_INIT.value,
            ExecutionPhase.PROMPT_CONFIRM.value,
            ExecutionPhase.CONTEXT_LOAD.value,
            ExecutionPhase.PROMPT_BUILD.value,
            ExecutionPhase.STORY_GENERATE.value,
            ExecutionPhase.STORY_SAVE.value,
        ]
        self.phase_executor.checkpoint_data = checkpoint_data

        # Test that phases ARE skipped when completed in THIS session
        # (Critical phases are handled by _execute_phase_sequence initialization, not skip logic)
        assert self.phase_executor._should_skip_phase(ExecutionPhase.CONFIG_LOAD) is True
        assert self.phase_executor._should_skip_phase(ExecutionPhase.BACKEND_INIT) is True

        # Test that other phases ARE also skipped when completed
        assert self.phase_executor._should_skip_phase(ExecutionPhase.STORY_GENERATE) is True
        assert self.phase_executor._should_skip_phase(ExecutionPhase.STORY_SAVE) is True

    def test_should_skip_phase_without_checkpoint(self):
        """Test that no phases are skipped when there's no checkpoint data."""
        self.phase_executor.checkpoint_data = None

        # All phases should execute when there's no checkpoint
        assert self.phase_executor._should_skip_phase(ExecutionPhase.INIT) is False
        assert self.phase_executor._should_skip_phase(ExecutionPhase.CONFIG_LOAD) is False
        assert self.phase_executor._should_skip_phase(ExecutionPhase.BACKEND_INIT) is False
        assert self.phase_executor._should_skip_phase(ExecutionPhase.STORY_GENERATE) is False

    @patch("storyforge.phase_executor.console")
    def test_execute_new_session(self, mock_console):
        """Test executing a new session with checkpointing."""
        prompt = "Test story prompt"
        cli_args = {"style": "adventure"}
        config = {"backend": "gemini"}

        # Mock the phase execution
        with patch.object(self.phase_executor, "_execute_phase_sequence") as mock_execute:
            self.phase_executor.execute_new_session(prompt, cli_args, config)

            # Verify checkpoint data was created
            assert self.phase_executor.checkpoint_data is not None
            assert self.phase_executor.checkpoint_data.original_inputs["prompt"] == prompt
            assert self.phase_executor.checkpoint_data.session_id.endswith("_sf")

            # Verify checkpoint manager was called for save
            assert self.checkpoint_manager.save_checkpoint.called

            # Verify phase execution started from INIT
            mock_execute.assert_called_once_with(ExecutionPhase.INIT)


class TestCheckpointIntegration:
    """Test checkpoint persistence integration."""

    def test_load_ignores_removed_interactive_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                checkpoint = CheckpointData.create_new("A magical adventure", {}, {})
                saved_path = manager.save_checkpoint(checkpoint)
                payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
                payload["recovery_possible"] = True
                payload["generated_content"]["images"] = ["old.png"]
                payload["user_decisions"]["prompt_confirmed"] = True
                saved_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

                loaded = manager.load_checkpoint(saved_path)

                assert "images" not in loaded.generated_content
                assert loaded.generated_content["generated_images"] == []
                assert "prompt_confirmed" not in loaded.user_decisions

    def test_checkpoint_yaml_structure(self):
        """Test that checkpoint YAML has expected structure and is readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                checkpoint = CheckpointData.create_new("test", {"key": "value"}, {"setting": True})
                checkpoint.generated_content["story"] = "Test story content"

                saved_path = manager.save_checkpoint(checkpoint)

                # Read raw YAML and verify structure
                with open(saved_path, encoding="utf-8") as f:
                    content = f.read()

                # Should have comments
                assert "# StoryForge Checkpoint" in content

                # Load YAML data
                yaml_lines = [line for line in content.split("\n") if not line.strip().startswith("#")]
                yaml_content = "\n".join(yaml_lines)
                data = yaml.safe_load(yaml_content)

                # Verify structure
                required_fields = [
                    "session_id",
                    "created_at",
                    "updated_at",
                    "status",
                    "current_phase",
                    "completed_phases",
                    "original_inputs",
                    "resolved_config",
                    "generated_content",
                    "user_decisions",
                ]
                for field in required_fields:
                    assert field in data

                # Verify nested structures
                assert "prompt" in data["original_inputs"]
                assert "cli_arguments" in data["original_inputs"]
                assert data["original_inputs"]["cli_arguments"]["key"] == "value"
                assert data["generated_content"]["story"] == "Test story content"

    def test_error_handling_in_checkpoint_operations(self):
        """Test error handling in checkpoint operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)

                # Test loading non-existent checkpoint
                non_existent_path = Path(tmpdir) / "nonexistent.yaml"
                with pytest.raises((FileNotFoundError, OSError)):  # Should raise an exception
                    manager.load_checkpoint(non_existent_path)

                # Test loading corrupted checkpoint
                corrupted_path = Path(tmpdir) / "corrupted.yaml"
                with open(corrupted_path, "w") as f:
                    f.write("invalid: yaml: content: [unclosed")

                with pytest.raises((yaml.YAMLError, ValueError)):  # Should raise an exception
                    manager.load_checkpoint(corrupted_path)

                # Test get_checkpoint_info with corrupted file (should not crash)
                info = manager.get_checkpoint_info(corrupted_path)
                assert info["prompt_preview"] == "Error reading checkpoint"
                assert info["status"] == "unknown"
