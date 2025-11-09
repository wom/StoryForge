# StoryForge MCP Refactor - Quick Reference

## Executive Summary

**Goal:** Refactor StoryForge from monolithic CLI to client/server architecture via MCP (Model Context Protocol)

**Approach:** Parallel implementation (keep monolith, build MCP alongside)

**Timeline:** 4 weeks

**Key Numbers:**
- 18 MCP tools to implement
- ~300 lines for thin CLI client
- ~950 lines PhaseExecutor + backends move to server
- 1 concurrent generation (queued requests)
- 12 error codes for structured error handling

**Transport:** stdio (standard MCP)

**Configuration:** XDG-compliant (`~/.config/storyforge-server/`, `~/.local/share/`, `~/.cache/`)

## Component Migration Map

### STAYS IN CLIENT (Thin Layer)
```
✓ CLI argument parsing (Typer)
✓ User interaction (Confirm.ask, prompts)
✓ Rich console formatting
✓ Display logic (panels, markdown rendering)
✓ Progress indication (polling-based)
✓ MCP client connection
```

### MOVES TO SERVER (All Business Logic)
```
→ PhaseExecutor (all 11 phases)
→ CheckpointManager (state persistence)
→ All LLM backends (Gemini, Claude, OpenAI)
→ ContextManager (markdown loading, chain tracking)
→ Config loading and resolution
→ Prompt building and validation
→ File I/O (stories, images, context)
→ Story generation, refinement, image generation
→ Chain traversal and export
→ Backend selection and initialization
```

## MCP Tools Summary

| Tool Name | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `storyforge_generate_story` | Create new story | prompt, style, age_range, etc. | session_id, story |
| `storyforge_continue_session` | Resume checkpoint | session_id, phase | session status |
| `storyforge_list_sessions` | Find checkpoints | status filter | session list |
| `storyforge_get_session_status` | Poll progress | session_id | status, progress_percent |
| `storyforge_delete_session` | Clean up session | session_id, keep_outputs | deleted, cleanup_path |
| `storyforge_get_queue_status` | Check queue | - | active_session, queue |
| `storyforge_extend_story` | Continue story | context_file, prompt | new session_id |
| `storyforge_export_chain` | Combine chain | context_file | export path |
| `storyforge_get_story_chain` | Chain info | context_file | chain parts |
| `storyforge_list_context_files` | Available stories | filter | file list with metadata |
| `storyforge_generate_images` | Create extra images | session_id, count | image paths |
| `storyforge_refine_story` | Improve story | session_id, instructions | refined story |
| `storyforge_list_backends` | Available LLMs | - | backend capabilities |
| `storyforge_update_session_backend` | Switch backend | session_id, new_backend | updated status |

## Data Flow Example

### Before (Monolithic):
```
CLI → PhaseExecutor → LLM Backend → File System
      ↓
   Checkpoint
      ↓
   Context Manager
```

### After (MCP):
```
CLI Client → MCP Tool Call → Server → PhaseExecutor → LLM Backend
   ↑            (stdio)         ↓
   |                       Checkpoint
   |                            ↓
   |                      Context Manager
   |                            ↓
   |                       File System
   |                            ↓
   └───────── Response (JSON) ───┘

Progress Updates (Polling):
CLI → get_session_status (every 2-5s) → Server → progress_percent
```

## Command Mapping

| Old CLI Command | New Flow |
|----------------|----------|
| `sf "prompt"` | CLI → `generate_story` tool → display result |
| `sf continue` | CLI → `list_sessions` → user select → `continue_session` |
| `sf extend` | CLI → `list_context_files` → user select → `extend_story` |
| `sf export-chain` | CLI → `list_context_files` → user select → `export_chain` |
| `sf config init` | Local only (no server call) |

## File Structure

```
storyforge/
├── client/
│   ├── __init__.py
│   ├── cli.py           # Thin CLI using MCP
│   ├── mcp_client.py    # MCP client wrapper (stdio transport)
│   └── formatters.py    # Rich display helpers
│
├── server/
│   ├── __init__.py
│   ├── mcp_server.py    # MCP server + tool definitions
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── session.py   # Session management tools
│   │   ├── story.py     # Story operation tools
│   │   ├── chain.py     # Chain operation tools
│   │   └── content.py   # Content management tools
│   └── core/            # Existing modules (minimal changes)
│       ├── phase_executor.py
│       ├── checkpoint.py
│       ├── llm_backend.py
│       ├── gemini_backend.py
│       ├── anthropic_backend.py
│       ├── openai_backend.py
│       ├── context.py
│       ├── config.py
│       └── prompt.py
│
└── shared/
    ├── __init__.py
    └── types.py         # Shared type definitions

# Configuration Directories (XDG-compliant)
~/.config/storyforge-server/config.yaml  # Server config
~/.config/storyforge/config.yaml         # CLI client config
~/.local/share/storyforge-server/        # Server data
  ├── checkpoints/
  ├── context/
  ├── stories/
  └── images/
~/.cache/storyforge-server/              # Server cache
  └── context_summaries/
```

## Implementation Order

1. **Week 1: Server Foundation**
   - [ ] MCP server skeleton with stdio transport
   - [ ] Error response schema implementation
   - [ ] File path resolution logic
   - [ ] Wrap PhaseExecutor in `storyforge_generate_story` tool
   - [ ] Basic checkpoint tools: list, get_status, continue
   - [ ] Session cleanup tool: delete_session
   - [ ] Queue management: get_queue_status

2. **Week 2: Core Features**
   - [ ] Extension tools: list_extendable, extend_story
   - [ ] Chain tools: get_story_chain, export_chain
   - [ ] Image generation tools: generate_images
   - [ ] Content management: list_context_files (with filters), get_context_content
   - [ ] Backend tools: list_backends, update_session_backend
   - [ ] Refinement: refine_story

3. **Week 3: Client**
   - [ ] Thin CLI client with Typer
   - [ ] MCP client connection (stdio)
   - [ ] Rich UI preservation (panels, progress bars)
   - [ ] Progress polling implementation (2-5s intervals)
   - [ ] Error display and recovery
   - [ ] Command mapping (main, continue, extend, export-chain)

4. **Week 4: Testing & Polish**
   - [ ] Integration tests
   - [ ] Performance testing (cache effectiveness)
   - [ ] E2E workflow tests
   - [ ] Checkpoint migration validation
   - [ ] Documentation updates
   - [ ] Backward compatibility tests

## Testing Strategy

### Server Tests
- Unit test each MCP tool (18 tools total)
- Mock LLM backends
- Test checkpoint recovery and migration
- Test error scenarios and error codes
- Test file path resolution edge cases
- Test queue management (single-threaded + queue)
- Test context caching effectiveness

### Client Tests
- Test MCP communication (stdio transport)
- Test display formatting (Rich UI)
- Test error handling and display
- Test progress polling logic
- Integration tests with real server
- Test all CLI commands

### E2E Tests
- Full story generation workflow
- Story extension workflow
- Chain export workflow
- Session resumption workflow
- Multi-session queue workflow
- Backend switching workflow
- Error recovery scenarios

### Performance Tests
- Context summarization caching
- Large context file handling (>10MB)
- Queue performance under load
- Progress polling overhead
- Checkpoint write performance

## Key Advantages

✅ **Multiple Frontends**: Easy to add web UI, mobile app, etc.
✅ **Testability**: Server logic fully isolated and testable
✅ **Scalability**: Server can handle queued requests from multiple clients
✅ **Integration**: Other tools can use MCP interface (e.g., IDE plugins)
✅ **Maintenance**: Clear separation of concerns (UI vs logic)
✅ **Remote Execution**: Run server anywhere, connect from anywhere
✅ **Caching**: Server-side caching improves performance
✅ **Monitoring**: Centralized logging and error tracking

## Key Implementation Details

### Transport
- **stdio**: Standard MCP transport (stdin/stdout)
- **Client spawns server**: On-demand process spawning
- **Daemon option**: Future enhancement for persistent server

### Error Handling
- **12 error codes**: Structured, recoverable errors
- **Recovery hints**: Guide users to resolution
- **Rich display**: Beautiful error formatting in CLI

### File Paths
- **XDG compliance**: `~/.config/`, `~/.local/share/`, `~/.cache/`
- **Absolute returns**: Server always returns full paths
- **Smart resolution**: Handles relative, home, and absolute paths

### Concurrency
- **Single-threaded**: 1 generation at a time
- **FIFO queue**: Up to 10 pending requests
- **Queue visibility**: `get_queue_status` tool

### Progress
- **Polling model**: 2-5 second intervals
- **Progress percent**: 0-100 based on phase completion
- **Future streaming**: Possible enhancement, not initial priority

## Migration Safety

- ✓ Keep existing monolith untouched
- ✓ Build MCP version in parallel
- ✓ Run both side-by-side during testing
- ✓ Gradual user migration
- ✓ Easy rollback if issues arise
