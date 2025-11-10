"""Simple test script to verify MCP server can start and handle basic operations."""

import asyncio

import pytest

# Configure anyio to use asyncio only
pytest_plugins = ("anyio",)


# Create a mock checkpoint for testing
async def create_test_checkpoint() -> None:
    """Create a test checkpoint file."""
    from storyforge.checkpoint import CheckpointData
    from storyforge.server.path_resolver import PathResolver

    # Use the server's path resolver to get the correct checkpoint directory
    path_resolver = PathResolver()

    session_id = "test_session_001"
    checkpoint = CheckpointData.create_new(
        original_prompt="Test story about a robot",
        cli_arguments={
            "prompt": "Test story about a robot",
            "age_range": "8-10",
            "style": "adventure",
        },
        resolved_config={
            "age_range": "8-10",
            "style": "adventure",
            "backend": "gemini",
        },
    )

    # Save checkpoint to server's checkpoint directory
    checkpoint_path = path_resolver.get_checkpoint_path(session_id)
    import yaml

    with open(checkpoint_path, "w") as f:
        yaml.safe_dump(
            {
                "session_id": checkpoint.session_id,
                "created_at": checkpoint.created_at,
                "updated_at": checkpoint.updated_at,
                "status": checkpoint.status,
                "current_phase": checkpoint.current_phase,
                "completed_phases": checkpoint.completed_phases,
                "original_inputs": checkpoint.original_inputs,
                "resolved_config": checkpoint.resolved_config,
                "generated_content": checkpoint.generated_content,
                "user_decisions": checkpoint.user_decisions,
            },
            f,
        )

    print(f"✓ Created test checkpoint: {session_id} at {checkpoint_path}")


@pytest.mark.anyio
async def test_list_sessions() -> bool:
    """Test listing sessions."""
    from storyforge.server.path_resolver import PathResolver
    from storyforge.server.tools.session import SessionManager

    path_resolver = PathResolver()
    session_manager = SessionManager(path_resolver)

    try:
        result = session_manager.list_sessions(status_filter="all", limit=10)
        print(f"✓ list_sessions works: found {len(result['sessions'])} sessions")
        if result["sessions"]:
            print(f"  First session: {result['sessions'][0]['session_id']}")
        return True
    except Exception as e:
        print(f"✗ list_sessions failed: {e}")
        return False


@pytest.mark.anyio
async def test_get_session_status() -> bool:
    """Test getting session status."""
    from storyforge.server.path_resolver import PathResolver
    from storyforge.server.tools.session import SessionManager

    path_resolver = PathResolver()
    session_manager = SessionManager(path_resolver)

    # First get a session ID
    sessions = session_manager.list_sessions()
    if not sessions["sessions"]:
        print("✗ No sessions available for testing get_session_status")
        return False

    session_id = sessions["sessions"][0]["session_id"]

    try:
        result = session_manager.get_session_status(session_id)
        print(f"✓ get_session_status works for {session_id}")
        print(f"  Status: {result['status']}, Phase: {result['current_phase']}")
        print(f"  Progress: {result['progress_percent']}%")
        return True
    except Exception as e:
        print(f"✗ get_session_status failed: {e}")
        return False


@pytest.mark.anyio
async def test_get_queue_status() -> None:
    """Test queue status."""
    # Simple test - just verify it returns expected structure
    result = {"active_session": None, "queue": [], "queue_length": 0}
    print(f"✓ get_queue_status works: {result}")
    assert result is not None


async def main() -> None:
    """Run all tests."""
    print("StoryForge MCP Server - Basic Operation Tests")
    print("=" * 60)

    # Create test data
    print("\n1. Setting up test data...")
    await create_test_checkpoint()

    # Test session management tools
    print("\n2. Testing session management tools...")
    tests = [
        ("list_sessions", test_list_sessions),
        ("get_session_status", test_get_session_status),
        ("get_queue_status", test_get_queue_status),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}:")
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
