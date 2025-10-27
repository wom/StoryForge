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

    def test_get_prompt_preview(self):
        """Test prompt preview truncation."""
        short_prompt = "Short prompt"
        long_prompt = "This is a very long prompt that should be truncated because it exceeds the limit"

        checkpoint_short = CheckpointData.create_new(short_prompt, {}, {})
        checkpoint_long = CheckpointData.create_new(long_prompt, {}, {})

        assert checkpoint_short.get_prompt_preview() == short_prompt
        assert len(checkpoint_long.get_prompt_preview()) <= 50
        assert checkpoint_long.get_prompt_preview().endswith("...")


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

    def test_get_checkpoint_stats(self):
        """Test getting checkpoint statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)

                # Create checkpoints with different statuses
                active_checkpoint = CheckpointData.create_new("active", {}, {})
                active_checkpoint.session_id = "active_sf"
                manager.save_checkpoint(active_checkpoint)

                completed_checkpoint = CheckpointData.create_new("completed", {}, {})
                completed_checkpoint.session_id = "completed_sf"
                completed_checkpoint.mark_completed()
                manager.save_checkpoint(completed_checkpoint)

                failed_checkpoint = CheckpointData.create_new("failed", {}, {})
                failed_checkpoint.session_id = "failed_sf"
                failed_checkpoint.mark_failed("Test error")
                manager.save_checkpoint(failed_checkpoint)

                stats = manager.get_checkpoint_stats()

                assert stats["total"] == 3
                assert stats["active"] == 1
                assert stats["completed"] == 1
                assert stats["failed"] == 1

    def test_cleanup_failed_sessions(self):
        """Test cleaning up failed checkpoint sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)

                # Create active checkpoint
                active_checkpoint = CheckpointData.create_new("active", {}, {})
                active_path = manager.save_checkpoint(active_checkpoint)

                # Create failed checkpoint with manually different session_id
                failed_checkpoint = CheckpointData.create_new("failed", {}, {})
                failed_checkpoint.session_id = failed_checkpoint.session_id + "_failed"
                failed_checkpoint.mark_failed("Test error")
                failed_path = manager.save_checkpoint(failed_checkpoint)

                # Verify both files exist before cleanup and have different names
                assert active_path.exists()
                assert failed_path.exists()
                assert active_path != failed_path

                # Clean up failed sessions
                deleted_count = manager.cleanup_failed_sessions()

                assert deleted_count == 1
                # The active checkpoint should remain
                assert active_path.exists()
                assert not failed_path.exists()
                remaining_files = list(manager.checkpoint_dir.glob("checkpoint_*.yaml"))
                assert len(remaining_files) == 1

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

    def test_create_resumed_session(self):
        """Test creating a resumed session from original checkpoint."""
        original_checkpoint = CheckpointData.create_new(
            "original story", {"style": "adventure", "age_range": "preschool"}, {"backend": "gemini", "verbose": True}
        )
        original_checkpoint.generated_content["story"] = "Generated story content"
        original_checkpoint.user_decisions["story_accepted"] = True

        new_checkpoint = self.phase_executor._create_resumed_session(
            original_checkpoint, ExecutionPhase.STORY_GENERATE
        )

        assert new_checkpoint.session_id != original_checkpoint.session_id
        assert new_checkpoint.session_id.endswith("_sf_resumed")
        assert new_checkpoint.status == "active"
        assert new_checkpoint.current_phase == ExecutionPhase.STORY_GENERATE.value
        assert new_checkpoint.original_inputs == original_checkpoint.original_inputs
        assert new_checkpoint.resolved_config == original_checkpoint.resolved_config

    def test_get_completed_phases_before(self):
        """Test getting phases that should be completed before resume phase."""
        phases_before_story = self.phase_executor._get_completed_phases_before(ExecutionPhase.STORY_GENERATE)

        expected_phases = [
            ExecutionPhase.INIT.value,
            ExecutionPhase.CONFIG_LOAD.value,
            ExecutionPhase.BACKEND_INIT.value,
            ExecutionPhase.PROMPT_CONFIRM.value,
            ExecutionPhase.CONTEXT_LOAD.value,
            ExecutionPhase.PROMPT_BUILD.value,
        ]

        assert phases_before_story == expected_phases

        # Test with first phase
        phases_before_init = self.phase_executor._get_completed_phases_before(ExecutionPhase.INIT)
        assert phases_before_init == []

    def test_get_content_up_to_phase(self):
        """Test preserving content up to specific phases."""
        original_checkpoint = CheckpointData.create_new("test", {}, {})
        original_checkpoint.generated_content = {
            "story": "Generated story",
            "refinements": "Some refinements",
            "images": ["image1.png", "image2.png"],
        }

        # Test preserving story content for phases after story generation
        content_for_image_phase = self.phase_executor._get_content_up_to_phase(
            original_checkpoint, ExecutionPhase.IMAGE_DECISION
        )

        assert content_for_image_phase["story"] == "Generated story"
        assert content_for_image_phase["refinements"] == "Some refinements"
        assert content_for_image_phase["images"] == []  # Images should be cleared

        # Test not preserving story for early phases
        content_for_early_phase = self.phase_executor._get_content_up_to_phase(
            original_checkpoint, ExecutionPhase.CONFIG_LOAD
        )

        assert content_for_early_phase["story"] is None
        assert content_for_early_phase["refinements"] is None

    def test_get_decisions_up_to_phase(self):
        """Test preserving user decisions up to specific phases."""
        original_checkpoint = CheckpointData.create_new("test", {}, {})
        original_checkpoint.user_decisions = {
            "story_accepted": True,
            "wants_images": False,
            "num_images_requested": 0,
            "save_as_context": True,
        }

        # Test preserving story decision for story save phase
        decisions_for_story_save = self.phase_executor._get_decisions_up_to_phase(
            original_checkpoint, ExecutionPhase.STORY_SAVE
        )

        assert decisions_for_story_save["story_accepted"] is True
        assert decisions_for_story_save["wants_images"] is None  # Should be cleared
        assert decisions_for_story_save["save_as_context"] is None  # Should be cleared

        # Test not preserving decisions for earlier phases
        decisions_for_early_phase = self.phase_executor._get_decisions_up_to_phase(
            original_checkpoint, ExecutionPhase.STORY_GENERATE
        )

        assert all(decision is None for decision in decisions_for_early_phase.values())

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
    def test_execute_from_checkpoint_creates_new_session(self, mock_console):
        """Test that execute_from_checkpoint creates a new resumed session."""
        resolved_config = {"backend": "test", "verbose": False}
        original_checkpoint = CheckpointData.create_new("test story", {}, resolved_config)

        # Mock the phase execution to avoid actual execution
        with patch.object(self.phase_executor, "_execute_phase_sequence") as mock_execute:
            self.phase_executor.execute_from_checkpoint(original_checkpoint, ExecutionPhase.STORY_GENERATE)

            # Verify new session was created
            assert self.phase_executor.checkpoint_data.session_id != original_checkpoint.session_id
            assert self.phase_executor.checkpoint_data.session_id.endswith("_sf_resumed")

            # Verify checkpoint manager was called
            assert self.checkpoint_manager.save_checkpoint.called

            # Verify phase execution was called
            mock_execute.assert_called_once_with(ExecutionPhase.STORY_GENERATE)

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
    """Test checkpoint system integration."""

    def test_end_to_end_checkpoint_workflow(self):
        """Test complete checkpoint workflow from creation to resumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("storyforge.checkpoint.user_data_dir", return_value=tmpdir):
                manager = CheckpointManager(auto_cleanup=False)
                executor = PhaseExecutor(manager)

                # Create initial checkpoint
                original_checkpoint = CheckpointData.create_new(
                    "A magical adventure",
                    {"style": "fantasy", "age_range": "middle_grade"},
                    {"backend": "gemini", "verbose": False},
                )
                original_checkpoint.generated_content["story"] = "Once upon a time..."
                original_checkpoint.update_phase(ExecutionPhase.STORY_SAVE)
                original_checkpoint.mark_completed()

                # Save checkpoint
                saved_path = manager.save_checkpoint(original_checkpoint)

                # Load and resume from checkpoint
                loaded_checkpoint = manager.load_checkpoint(saved_path)

                # Create resumed session
                resumed_checkpoint = executor._create_resumed_session(loaded_checkpoint, ExecutionPhase.IMAGE_DECISION)

                # Verify resumed session properties
                assert resumed_checkpoint.session_id != loaded_checkpoint.session_id
                assert resumed_checkpoint.original_inputs["prompt"] == "A magical adventure"
                assert resumed_checkpoint.generated_content["story"] == "Once upon a time..."
                assert resumed_checkpoint.current_phase == ExecutionPhase.IMAGE_DECISION.value

                # Verify resumed session starts with empty completed_phases
                # (Critical phases will be initialized by _execute_phase_sequence before resume point)
                assert resumed_checkpoint.completed_phases == []

                # Verify parent session tracking
                assert resumed_checkpoint.progress is not None
                assert resumed_checkpoint.progress.get("resumed_from_session") == loaded_checkpoint.session_id
                assert resumed_checkpoint.progress.get("resumed_at_phase") == ExecutionPhase.IMAGE_DECISION.value

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
