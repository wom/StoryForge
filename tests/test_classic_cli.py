"""Tests for the MCP-backed classic terminal workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storyforge.classic_cli import ClassicCLI
from storyforge.mcp_models import DraftResult, SessionSummary, WorkflowResult


@pytest.mark.asyncio
async def test_continue_collects_review_refinement_and_media_decisions():
    original = DraftResult(
        session_id="session-test",
        status="active",
        story="Original draft",
        output_directory="output",
        checkpoint_phase="story_save",
    )
    revised = original.model_copy(update={"story": "Revised draft"})
    client = MagicMock()
    client.list_sessions = AsyncMock(
        return_value=[SessionSummary(session_id="session-test", prompt_preview="A brave mouse")]
    )
    client.resume_session = AsyncMock(return_value=original)
    client.refine_draft = AsyncMock(return_value=revised)
    client.finalize_story = AsyncMock(return_value=WorkflowResult(message="Complete"))

    with (
        patch("storyforge.classic_cli.IntPrompt.ask", side_effect=[1, 2, 3]),
        patch("storyforge.classic_cli.Confirm.ask", side_effect=[False, True, True, True, True]),
        patch("storyforge.classic_cli.Prompt.ask", return_value="Make it funnier"),
    ):
        result = await ClassicCLI()._continue(client)

    assert result == 0
    refinement = client.refine_draft.await_args.args[0]
    assert refinement.instructions == "Make it funnier"
    finalization = client.finalize_story.await_args.args[0]
    assert finalization.session_id == "session-test"
    assert finalization.video_scene_count == 2
    assert finalization.image_count == 3
    assert finalization.save_context is True
