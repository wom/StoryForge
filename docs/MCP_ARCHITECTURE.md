# StoryForge MCP Architecture Plan

## Overview

This document outlines the plan to refactor StoryForge from a monolithic CLI application into a client/server architecture using the Model Context Protocol (MCP). The goal is to create a thin CLI client that communicates with a rich MCP server, enabling multiple frontend implementations (CLI, Web UI, Mobile, etc.).

## Current Architecture Analysis

### Monolithic Components

**CLI Layer (`StoryForge.py` - 792 lines)**
- Typer-based command definitions
- User input/confirmation dialogs
- Story/chain selection UI
- Rich console output formatting
- Direct coupling to all backend logic

**Core Engine (`phase_executor.py` - 949 lines)**
- 11-phase story generation workflow
- Checkpoint management integration
- Backend initialization and lifecycle
- Story generation, refinement, and image generation orchestration
- File I/O for stories and images

**Backend Abstraction (`llm_backend.py` + 3 implementations)**
- Google Gemini backend
- Anthropic Claude backend  
- OpenAI GPT backend
- Abstract interface: `generate_story()`, `generate_image()`, `generate_image_prompt()`, `generate_image_name()`

**State Management (`checkpoint.py` - 319 lines)**
- Checkpoint persistence (YAML)
- Session recovery and resumption
- Phase completion tracking
- Generated content storage

**Context Management (`context.py` - 753 lines)**
- Markdown file loading and parsing
- Story chain tracking and traversal
- Context summarization (extractive)
- Metadata extraction

**Configuration (`config.py` - 118 lines)**
- YAML config file loading
- Default value resolution
- Environment variable support

**Prompt Building (`prompt.py` - 133 lines)**
- Structured prompt creation
- Parameter validation
- Backend-specific formatting

## MCP Architecture Design

### High-Level Split

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER (Thin)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ CLI Client │  │ Web Client │  │Future: TUI │             │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│        │               │               │                    │
│        └───────────────┴───────────────┘                    │
│                         │                                   │
│                    MCP Protocol                             │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                         │                                    │
│                  MCP SERVER (Rich)                           │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              StoryForge MCP Tools                        ││
│  │  • generate_story        • continue_session              ││
│  │  • extend_story          • export_chain                  ││
│  │  • list_sessions         • list_context_files            ││
│  │  • get_session_status    • get_story_chain               ││
│  │  • generate_image        • save_context                  ││
│  │  • refine_story          • list_backends                 ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │              Core Business Logic                         ││
│  │  • PhaseExecutor         • ContextManager                ││
│  │  • CheckpointManager     • Config                        ││
│  │  • LLM Backends          • Prompt Builder                ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### MCP Server Structure

#### MCP Tools (Server-side API)

**Session Management Tools:**
```python
# Tool: storyforge_generate_story
{
  "prompt": str,
  "age_range": str,
  "style": str,
  "tone": str,
  "length": str,
  "theme": str | None,
  "characters": list[str] | None,
  "setting": str | None,
  "learning_focus": str | None,
  "image_style": str | None,
  "backend": str | None,
  "output_directory": str | None
} → {
  "session_id": str,
  "status": "active|completed|failed",
  "current_phase": str,
  "story": str | None,
  "images": list[str],
  "checkpoint_path": str
}

# Tool: storyforge_continue_session
{
  "session_id": str | None,  # If None, shows list
  "resume_phase": str | None
} → {
  "session_id": str,
  "status": str,
  "available_phases": list[str],
  "current_phase": str
}

# Tool: storyforge_delete_session
{
  "session_id": str,
  "keep_outputs": bool  # Keep story/images but remove checkpoint
} → {
  "deleted": bool,
  "cleanup_path": str
}

# Tool: storyforge_list_sessions
{
  "status_filter": "active|completed|failed|all",
  "limit": int
} → {
  "sessions": [{
    "session_id": str,
    "created_at": str,
    "status": str,
    "prompt": str,
    "current_phase": str
  }]
}

# Tool: storyforge_get_session_status
{
  "session_id": str
} → {
  "session_id": str,
  "status": str,
  "current_phase": str,
  "completed_phases": list[str],
  "story": str | None,
  "images": list[str],
  "errors": list[str],
  "progress_percent": int
}

# Tool: storyforge_get_queue_status
{} → {
  "active_session": str | None,
  "queue": [{
    "session_id": str,
    "position": int
  }],
  "queue_length": int
}
```

**Story Extension Tools:**
```python
# Tool: storyforge_list_extendable_stories
{} → {
  "stories": [{
    "context_file": str,
    "title": str,
    "prompt": str,
    "generated_at": str,
    "has_chain": bool,
    "chain_length": int
  }]
}

# Tool: storyforge_extend_story
{
  "context_file": str,
  "new_prompt": str | None,
  "ending_type": "wrap_up|cliffhanger",
  "same_config": bool
} → {
  "session_id": str,
  "parent_story": str,
  "status": str
}

# Tool: storyforge_get_story_chain
{
  "context_file": str
} → {
  "chain": [{
    "part_number": int,
    "context_file": str,
    "prompt": str,
    "generated_at": str,
    "story_preview": str
  }]
}

# Tool: storyforge_export_chain
{
  "context_file": str,
  "output_path": str | None
} → {
  "export_path": str,
  "total_parts": int,
  "combined_word_count": int
}
```

**Content Management Tools:**
```python
# Tool: storyforge_list_context_files
{
  "filter": {
    "has_chain": bool | None,
    "min_date": str | None,
    "max_chain_length": int | None,
    "search": str | None  # Search in prompts/titles
  } | None
} → {
  "files": [{
    "filename": str,
    "path": str,
    "size": int,
    "modified": str,
    "metadata": {
      "prompt": str,
      "generated_at": str,
      "extended_from": str | None
    }
  }]
}

# Tool: storyforge_get_context_content
{
  "context_file": str
} → {
  "metadata": {...},
  "story": str
}

# Tool: storyforge_save_as_context
{
  "session_id": str,
  "filename": str | None
} → {
  "context_file": str,
  "path": str
}
```

**Image Generation Tools:**
```python
# Tool: storyforge_generate_images
# Note: For generating ADDITIONAL images beyond the initial set from generate_story
{
  "session_id": str,
  "num_images": int,
  "image_style": str | None
} → {
  "images": [{
    "filename": str,
    "path": str,
    "prompt": str
  }]
}
```

**Configuration Tools:**
```python
# Tool: storyforge_list_backends
{} → {
  "backends": [{
    "name": str,
    "available": bool,
    "api_key_set": bool,
    "capabilities": {
      "story_generation": bool,
      "image_generation": bool
    }
  }]
}

# Tool: storyforge_get_config
{} → {
  "config": {...},  # Full config object
  "config_path": str
}

# Tool: storyforge_update_session_backend
{
  "session_id": str,
  "new_backend": str
} → {
  "updated": bool,
  "backend": str
}
```

**Refinement Tool:**
```python
# Tool: storyforge_refine_story
{
  "session_id": str,
  "refinement_instructions": str,
  "backend": str | None  # Use session's backend if None
} → {
  "refined_story": str,
  "session_id": str
}
```

### Client Responsibilities (Thin)

**CLI Client (`client/cli.py`)**
- Parse command-line arguments
- Make MCP tool calls to server
- Display results with Rich formatting
- Handle user prompts (confirmations, selections)
- Stream progress updates (if MCP supports)
- Local file path resolution (for output)

**Responsibilities:**
1. Argument parsing (Typer)
2. User interaction (Confirm, prompts, selections)
3. Rich console formatting and display
4. MCP client connection management
5. Error display and handling
6. Progress indication (polling-based, 2-5 second intervals)

**What moves OUT of client:**
- ALL business logic
- Phase execution
- Backend management
- Checkpoint persistence
- File I/O (except display)
- Prompt building
- Context management
- Chain tracking

### Server Responsibilities (Rich)

**MCP Server (`server/mcp_server.py`)**
- Expose all StoryForge functionality as MCP tools
- Manage PhaseExecutor lifecycle
- Handle checkpoint persistence
- Coordinate LLM backend calls
- Manage file system operations
- Track sessions and state
- **Concurrency:** Single-threaded execution (1 generation at a time)
- **Queuing:** Queue additional requests when busy

**Core Modules (Mostly Unchanged):**
- `phase_executor.py` - Orchestrates generation phases
- `checkpoint.py` - State persistence
- `llm_backend.py` + implementations - API calls
- `context.py` - Context and chain management
- `config.py` - Configuration
- `prompt.py` - Prompt building

## MCP Protocol Specifications

### Transport Layer

**Transport:** stdio (standard MCP approach)

**Server Startup Options:**
- **Option A:** Client spawns server process (on-demand)
- **Option B:** Systemd/launchd daemon (persistent)
- **Option C:** Both (daemon with on-demand fallback)

**Recommendation:** Start with Option A (client-spawned) for simplicity

### Error Response Schema

All MCP tool errors return a structured error response:

```python
{
  "error": {
    "code": str,  # Error codes: see below
    "message": str,  # Human-readable message
    "details": dict | None,  # Additional context
    "recoverable": bool,  # Can operation be retried?
    "recovery_hint": str | None  # Suggestion for recovery
  }
}
```

**Error Codes:**
- `BACKEND_UNAVAILABLE` - LLM backend not configured or API down
- `CHECKPOINT_CORRUPT` - Checkpoint file is invalid
- `SESSION_NOT_FOUND` - Invalid session_id
- `CONTEXT_FILE_NOT_FOUND` - Referenced context file missing
- `INVALID_PHASE` - Cannot resume from requested phase
- `GENERATION_FAILED` - LLM generation error
- `FILE_IO_ERROR` - File system operation failed
- `CONFIG_ERROR` - Configuration invalid
- `QUEUE_FULL` - Server queue at capacity
- `CONCURRENT_LIMIT` - Only 1 generation allowed at a time

### File Path Resolution

**Path Handling Rules:**

1. **Client sends:**
   - Absolute paths: `/home/user/stories/output.txt`
   - Home paths: `~/stories/output.txt`
   - Relative paths: `output.txt` or `stories/output.txt`

2. **Server resolves:**
   - Absolute paths → use as-is
   - Home paths (`~`) → expand to user home directory
   - Relative paths → relative to server's `output_directory` config
   - Invalid paths → return `FILE_IO_ERROR`

3. **Server returns:**
   - Always return absolute paths in responses
   - Include both `filename` and `absolute_path` in file objects

### File System Layout

**XDG-Compliant Directory Structure:**

```
~/.config/storyforge-server/
  └── config.yaml           # Server configuration

~/.config/storyforge/
  └── config.yaml           # CLI client configuration (optional)

~/.local/share/storyforge-server/
  ├── checkpoints/          # Session checkpoints
  │   ├── session_20251109_123456.yaml
  │   └── session_20251109_234567.yaml
  ├── context/              # Generated story context files
  │   ├── story_part1.md
  │   └── story_part2.md
  ├── stories/              # Generated story text files
  │   └── story_20251109_123456.txt
  └── images/               # Generated images
      └── story_20251109_123456_image1.png

~/.cache/storyforge-server/
  └── context_summaries/    # Cached context summaries
```

### Progress Updates

**Polling Model (Initial Implementation):**
- Client polls `get_session_status` at 2-5 second intervals
- Response includes `progress_percent` (0-100)
- Response includes `current_phase` for display

**Future Enhancement:**
- Consider MCP streaming support for real-time progress
- Not worth significant implementation complexity initially

### Concurrency Model

**Single-Threaded Execution:**
- Server handles **1 generation at a time**
- Additional requests are queued
- Use `get_queue_status` tool to check queue

**Queue Behavior:**
- FIFO (First In, First Out)
- Queue limit: 10 pending sessions (configurable)
- Returns `QUEUE_FULL` error if limit exceeded

## Session Lifecycle

Sessions progress through the following states:

```
CREATED → ACTIVE → COMPLETED
    ↓         ↓         ↓
    ↓     PAUSED → RESUMING
    ↓         ↓         ↓
    └─────→ FAILED ←────┘
```

**State Descriptions:**

1. **CREATED** - `generate_story` called, checkpoint initialized
2. **ACTIVE** - Phases executing, checkpoint updated after each phase
3. **PAUSED** - User interrupted (Ctrl+C), checkpoint saved at last completed phase
4. **RESUMING** - `continue_session` called, loading checkpoint
5. **COMPLETED** - All phases done, final checkpoint marked complete
6. **FAILED** - Error occurred, checkpoint preserved for debugging

**State Transitions:**
- Created → Active: Execution begins
- Active → Paused: User interruption
- Paused → Resuming: Continue requested
- Resuming → Active: Checkpoint loaded, execution resumes
- Active → Completed: Final phase completes
- Active/Resuming → Failed: Unrecoverable error

## Migration Strategy

### Phase 1: Server Foundation
1. Create MCP server skeleton with tool definitions
2. Wrap PhaseExecutor in MCP tool interface
3. Implement session management tools
4. Add checkpoint-based tools (list, continue, status)
5. Test with basic story generation

### Phase 2: Core Features
1. Implement story extension tools
2. Add chain tracking and export tools
3. Implement image generation tools
4. Add refinement support
5. Implement context file tools

### Phase 3: Client Implementation
1. Create thin CLI client
2. Replace direct function calls with MCP tool calls
3. Maintain Rich UI/UX
4. Add progress indication
5. Handle streaming responses

### Phase 4: Advanced Features
1. Add backend selection/listing
2. Configuration management via MCP
3. Session cleanup tools
4. Batch operations
5. Server-side caching

## Backward Compatibility

**Checkpoint Migration:**
- Existing checkpoint files remain compatible
- Automatic format upgrade on load (if needed)
- Old ExecutionPhase enum values supported

**Context Files:**
- All existing context markdown files work unchanged
- Metadata extraction backward compatible
- Chain tracking preserved

**Configuration:**
- Minor schema version bump (1.0 → 1.1)
- Old config files auto-upgraded
- New server-specific fields added with defaults

**Migration Path:**
- Old CLI can coexist with new MCP version
- Checkpoints created by old CLI can be resumed by server
- Context files shareable between versions

## Performance Considerations

**Context Caching:**
- Cache summarized contexts on server
- Avoid re-summarization on every request
- Cache invalidation on file modification
- Cache location: `~/.cache/storyforge-server/context_summaries/`

**Async Operations:**
- Checkpoint writes: async to avoid blocking phase execution
- File I/O: use async where beneficial (large files)
- Image generation: async per image (parallel generation)

**Rate Limiting:**
- Progress updates: max 1 response per second
- Queue status: cached for 500ms
- Context file listing: cached for 5 seconds

**Resource Management:**
- LLM backend connection pooling (if supported)
- Context file handle caching
- Memory limits for large context files (>10MB warning)

## Key Design Decisions

### 1. State Management
- **Server-side checkpoints**: All session state lives on server
- **Client-side display cache**: Clients can cache results for display
- **Stateless client**: Each request is independent

### 2. File Paths
- **Server absolute paths**: Server uses absolute paths for all files
- **Client relative paths**: Client can use relative paths in commands
- **Path resolution**: Server resolves paths based on rules in Protocol Specifications
- **Return values**: Server always returns absolute paths for clarity

### 3. Progress Updates
- **Polling model**: Client polls server for progress (2-5 second intervals)
- **Phase-based updates**: Server reports current phase and progress percent
- **Future streaming**: MCP streaming support possible later, not initial priority

### 4. User Interaction
- **Client-side prompts**: All confirmations stay in client
- **Server provides options**: Server returns choices, client displays
- **Non-blocking**: Server operations don't block on user input

### 5. Error Handling
- **Server errors**: Structured error responses with codes (see Protocol Specifications)
- **Client display**: Rich error formatting in client
- **Recovery**: Server manages checkpoint recovery, provides recovery hints

### 6. Concurrency & Queuing
- **Single-threaded**: Server handles 1 generation at a time
- **Queue model**: FIFO queue for additional requests (limit: 10)
- **Queue status**: `get_queue_status` tool for visibility
- **Busy response**: Return queue position when server is busy

## Implementation Checklist

### Server (MCP Tools)
- [ ] `storyforge_generate_story` - Core generation
- [ ] `storyforge_continue_session` - Resume from checkpoint
- [ ] `storyforge_list_sessions` - Session discovery
- [ ] `storyforge_get_session_status` - Status polling
- [ ] `storyforge_delete_session` - Session cleanup
- [ ] `storyforge_get_queue_status` - Queue visibility
- [ ] `storyforge_list_extendable_stories` - Find stories to extend
- [ ] `storyforge_extend_story` - Story extension
- [ ] `storyforge_get_story_chain` - Chain traversal
- [ ] `storyforge_export_chain` - Chain export
- [ ] `storyforge_list_context_files` - Context discovery
- [ ] `storyforge_get_context_content` - Context retrieval
- [ ] `storyforge_save_as_context` - Context saving
- [ ] `storyforge_generate_images` - Image generation
- [ ] `storyforge_refine_story` - Story refinement
- [ ] `storyforge_list_backends` - Backend discovery
- [ ] `storyforge_get_config` - Configuration access
- [ ] `storyforge_update_session_backend` - Backend switching

### Client (CLI)
- [ ] MCP client connection setup
- [ ] `sf main` command → `storyforge_generate_story`
- [ ] `sf continue` command → `storyforge_continue_session`
- [ ] `sf extend` command → `storyforge_extend_story`
- [ ] `sf export-chain` command → `storyforge_export_chain`
- [ ] `sf config init` command (local only?)
- [ ] Progress polling and display
- [ ] Error handling and display
- [ ] Rich console output formatting

### Testing
- [ ] MCP tool unit tests
- [ ] Client integration tests
- [ ] End-to-end workflow tests
- [ ] Backward compatibility tests
- [ ] Performance benchmarks

## Benefits of This Architecture

### Immediate
1. **Multiple frontends**: Easy to add web, mobile, or other UIs
2. **Thin client**: Minimal client-side logic
3. **Testability**: Server logic fully testable without UI
4. **Remote execution**: Run server anywhere, client anywhere

### Future
1. **Scalability**: Server can handle multiple clients
2. **Caching**: Server-side result caching
3. **Batch operations**: Process multiple stories server-side
4. **Integration**: Other tools can use MCP interface
5. **Monitoring**: Centralized logging and metrics

## Migration Path

### Option A: Parallel Implementation
- Keep existing monolith
- Build MCP server alongside
- Create new thin client
- Users choose which to use
- Deprecate monolith after testing

### Option B: Incremental Refactor
- Start with most complex commands (main, extend)
- Add MCP tools one at a time
- Client falls back to direct calls if server unavailable
- Gradually migrate all commands
- Remove fallback logic when complete

**Recommendation**: Option A for safety, cleaner separation

## Open Questions

1. **MCP Server deployment**: Standalone daemon or on-demand spawning?
  - Answer: Standalone. But each front end can start the daemon if necessary.
2. **Authentication**: Do we need auth for local usage?
 - Answer: No auth necessary
3. **Concurrency**: Should server handle multiple generations simultaneously?
 - Answer: We only need to handle 1 generation at once; limit if it makes the implementation cleaner. If it's *not* cleaner - then allow multiple. Whatever implementation is cleaner.
 - **Decision**: Limit to 1, use queue. Simpler implementation.
4. **Streaming**: Do we need real-time progress streaming?
 - Answer: Ideally - yes; but not worth a ton of implementation complexity.
 - **Decision**: Start with polling, add streaming as future enhancement
5. **Configuration**: Server config vs client config - where does each live?
 - Answer: Client config is up to each client, CLI will be XDG. Server should also likely honor XDG, though with a different naming convention maybe.
 - **Decision**: Server uses `~/.config/storyforge-server/`, client uses `~/.config/storyforge/`

## Next Steps

1. Create proof-of-concept MCP server with 1-2 tools
2. Create minimal CLI client that calls POC server
3. Validate architecture with end-to-end test
4. Implement remaining tools incrementally
5. Add comprehensive testing
6. Document new architecture and APIs
