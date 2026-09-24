"""Tests for the MCP-backed classic terminal workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storyforge.classic_cli import ClassicCLI
from storyforge.mcp_models import DraftResult, SessionSummary, StorySummary, WorkflowResult


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


@pytest.mark.asyncio
async def test_review_defaults_illustrations_to_resolved_count():
    draft = DraftResult(
        session_id="session-test",
        status="active",
        story="A story",
        output_directory="output",
        checkpoint_phase="story_save",
        metadata={"image_count": 4},
    )
    client = MagicMock()
    client.finalize_story = AsyncMock(return_value=WorkflowResult(message="Complete"))
    with (
        patch("storyforge.classic_cli.Confirm.ask", side_effect=[True, False, True, False]),
        patch("storyforge.classic_cli.IntPrompt.ask", return_value=4) as count_prompt,
    ):
        await ClassicCLI()._review_and_finalize(client, draft, "Story")

    assert count_prompt.call_args.kwargs["default"] == 4
    assert client.finalize_story.await_args.args[0].image_count == 4


@pytest.mark.asyncio
async def test_extension_passes_options_through_review_with_context_default():
    story = StorySummary(id="old", filename="old", filepath="old.md")
    draft = DraftResult(
        session_id="new",
        status="active",
        story="Continuation",
        output_directory="output",
        checkpoint_phase="story_save",
    )
    client = MagicMock()
    client.list_stories = AsyncMock(return_value=[story])
    client.create_extension_draft = AsyncMock(return_value=draft)
    classic = ClassicCLI()
    with (
        patch("storyforge.classic_cli.IntPrompt.ask", return_value=1),
        patch("storyforge.classic_cli.Prompt.ask", side_effect=["cliffhanger", "Go north"]),
        patch.object(classic, "_review_and_finalize", new_callable=AsyncMock, return_value=0) as review,
    ):
        await classic._extend(client, {"backend": "openai", "verbose": True, "debug": None})

    request = client.create_extension_draft.await_args.args[0]
    assert request.backend == "openai" and request.verbose is True
    review.assert_awaited_once_with(client, draft, "Continuation Draft", save_context_default=True)
