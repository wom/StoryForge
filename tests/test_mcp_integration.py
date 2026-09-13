"""Contract tests for the bundled StoryForge MCP surface."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mcp import Client

from storyforge.mcp_client import StoryForgeMCPClient
from storyforge.mcp_models import DraftResult, GenerationRequest
from storyforge.mcp_server import mcp


@pytest.mark.asyncio
async def test_server_exposes_expected_tools_and_resources():
    async with Client(mcp, raise_exceptions=True) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()

    names = {tool.name for tool in tools.tools}
    assert {
        "storyforge_create_draft",
        "storyforge_refine_draft",
        "storyforge_finalize_story",
        "storyforge_resume_session",
        "storyforge_export_chain",
        "storyforge_list_generated_stories",
        "storyforge_get_generated_story",
        "storyforge_write_config",
    } <= names
    assert {str(resource.uri) for resource in resources.resources} >= {
        "storyforge://config",
        "storyforge://world",
    }


@pytest.mark.asyncio
async def test_typed_client_consumes_structured_results():
    cached = {"gemini": [{"name": "gemini-test"}], "openai": [], "anthropic": []}
    with patch("storyforge.workflow.StoryForgeWorkflow.list_models", return_value=cached):
        async with StoryForgeMCPClient(server=mcp) as client:
            result = await client.list_models()

    assert result == cached


@pytest.mark.asyncio
async def test_create_draft_runs_through_bounded_worker():
    expected = DraftResult(
        session_id="session-test",
        status="active",
        story="Once upon a test.",
        output_directory="output",
        checkpoint_phase="story_save",
    )
    with patch("storyforge.workflow.StoryForgeWorkflow.create_draft", return_value=expected):
        async with StoryForgeMCPClient(server=mcp) as client:
            result = await client.create_draft(GenerationRequest(prompt="A test story"))

    assert result == expected


@pytest.mark.asyncio
async def test_destructive_cache_clear_requires_confirmation():
    async with Client(mcp) as client:
        result = await client.call_tool("storyforge_clear_models", {"confirmed": False})

    assert result.is_error is True
    assert "confirmed=true" in result.content[0].text
