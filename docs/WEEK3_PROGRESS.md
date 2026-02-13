# Week 3 Progress: Thin CLI Client Implementation

**Date:** November 10, 2025  
**Status:** 🔄 **In Progress - MCP Client Complete, CLI Refactor Remaining**

---

## ✅ Completed: MCP Client Infrastructure

### What's Done

**1. MCP Client Wrapper** (`storyforge/client/mcp_client.py`)
- ✅ Full async MCP client implementation
- ✅ Stdio transport for server communication
- ✅ Wrapper methods for all 18 MCP tools:
  - Session management (5 tools)
  - Story generation (1 tool)
  - Extension (4 tools)
  - Content management (3 tools)
  - Image generation (1 tool)
  - Configuration (3 tools)
  - Refinement (1 tool)
- ✅ Connection lifecycle management
- ✅ Error handling with context
- ✅ Global client instance pattern
- ✅ `run_sync()` helper for async/sync bridge

**2. Server Entry Point** (`storyforge/server/__main__.py`)
- ✅ Module entry point for server (`python -m storyforge.server.mcp_server`)
- ✅ Imports existing server infrastructure

**3. Module Structure**
- ✅ `storyforge/client/__init__.py` with proper exports
- ✅ All type hints correct
- ✅ Lint checks passing (ruff, mypy)

### Code Stats
- **New Files:** 3
- **Lines Added:** 695
- **Type Safety:** 100%
- **Tests:** Existing 295 tests still passing

---

## 🚧 Remaining Work: CLI Refactor

### Tasks to Complete

#### 1. Refactor Main CLI Entry Point
**File:** `storyforge/StoryForge.py` (792 lines) → Needs significant refactoring

**Current State:**
- Monolithic CLI with embedded business logic
- Direct calls to PhaseExecutor, CheckpointManager, LLMBackend
- Complex prompt validation and confirmation flows
- Multiple Typer commands (main, continue, extend, config init)

**Required Changes:**
- [ ] Replace PhaseExecutor calls with `client.generate_story()`
- [ ] Replace checkpoint management with MCP session tools
- [ ] Add progress polling for async story generation
- [ ] Maintain Rich formatting and user interactions
- [ ] Keep Typer CLI structure and arguments
- [ ] Update extend command to use MCP extension tools
- [ ] Add new commands (list, status, delete, refine, images, backends, etc.)

#### 2. Implement Progress Polling
- [ ] Create polling loop (2-5 second intervals)
- [ ] Call `client.get_session_status()` for progress
- [ ] Display phase transitions with Rich progress bars
- [ ] Show queue position when waiting
- [ ] Handle phase errors and recovery

#### 3. Add New CLI Commands
Commands that don't exist yet but should:

- [ ] `sf list` - List sessions (filterable)
- [ ] `sf status <session_id>` - Check session status
- [ ] `sf delete <session_id>` - Delete session
- [ ] `sf refine <session_id>` - Refine story
- [ ] `sf images <session_id>` - Generate additional images
- [ ] `sf backends` - List available backends
- [ ] `sf chain <context_file>` - View story chain
- [ ] `sf export <context_file>` - Export chain

#### 4. Update Existing Commands
- [ ] `sf <prompt>` - Use `generate_story()` with progress polling
- [ ] `sf continue` - Use `continue_session()` with session selection
- [ ] `sf extend` - Use `extend_story()` with context file selection
- [ ] `sf config init` - Keep as local (no server call needed)

#### 5. Rich Display Formatters
**New File:** `storyforge/client/formatters.py` (to create)

- [ ] Format session lists as Rich tables
- [ ] Format progress bars for phases
- [ ] Format backend lists with capabilities
- [ ] Format story chains with previews
- [ ] Format error messages with recovery hints
- [ ] Format queue status

#### 6. Testing
- [ ] Update existing CLI tests to mock MCP client
- [ ] Add integration tests with server
- [ ] Test progress polling behavior
- [ ] Test error handling and recovery
- [ ] Test all new commands

---

## 📋 Implementation Strategy

### Phase 1: Minimal Viable Refactor (High Priority)
Focus on core generation workflow first:

1. **Update `main()` command**
   - Replace PhaseExecutor with `client.generate_story()`
   - Add basic progress polling
   - Maintain existing argument parsing
   - Keep confirmation prompts

2. **Update `continue` command**
   - Use `client.list_sessions()` to show recent sessions
   - Use `client.continue_session()` to resume
   - Add progress polling

3. **Update `extend` command**
   - Use `client.list_extendable_stories()`
   - Use `client.extend_story()`
   - Add progress polling

### Phase 2: Enhanced Commands (Medium Priority)
Add new commands for full MCP functionality:

4. **Session Management Commands**
   - `sf list` - Browse sessions
   - `sf status` - Check progress
   - `sf delete` - Cleanup

5. **Content & Refinement**
   - `sf refine` - Refine stories
   - `sf images` - Generate images
   - `sf chain` / `sf export` - Chain management

### Phase 3: Polish (Lower Priority)
6. **Rich Formatters** - Beautiful output
7. **Comprehensive Testing** - Full coverage
8. **Documentation** - User guide updates

---

## 🔧 Technical Considerations

### Async/Sync Bridge
- CLI is sync (Typer doesn't support async natively)
- Use `run_sync()` helper to call async MCP client
- Example pattern:
  ```python
  from .client import get_client, run_sync
  
  def generate_command(prompt: str, ...):
      client = get_client()
      result = run_sync(client.generate_story(prompt, ...))
      # Handle result
  ```

### Progress Polling Pattern
```python
import time
from rich.progress import Progress, SpinnerColumn, TextColumn

def poll_session_status(session_id: str, client: MCPClient):
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as progress:
        task = progress.add_task("Generating story...", total=None)
        
        while True:
            status = run_sync(client.get_session_status(session_id))
            
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                raise Exception(status.get("error_message"))
            
            # Update progress description
            phase = status.get("current_phase", "unknown")
            progress_pct = status.get("progress_percent", 0)
            progress.update(task, description=f"{phase}: {progress_pct}%")
            
            time.sleep(3)  # Poll every 3 seconds
    
    return status
```

### Error Handling
- MCP errors come back as structured responses
- Extract error code, message, and recovery hints
- Display with Rich formatting
- Example:
  ```python
  try:
      result = run_sync(client.generate_story(...))
  except Exception as e:
      if "backend_unavailable" in str(e):
          console.print("[red]Backend unavailable. Set API key.[/red]")
          console.print("[dim]Recovery: export GEMINI_API_KEY=your_key[/dim]")
      raise
  ```

---

## 📊 Estimated Effort

### Remaining Work
- **Phase 1 (MVP):** ~4-6 hours
  - Refactor main/continue/extend commands
  - Add progress polling
  - Update tests
  
- **Phase 2 (Enhanced):** ~3-4 hours
  - Add new commands
  - Session management
  - Content tools
  
- **Phase 3 (Polish):** ~2-3 hours
  - Rich formatters
  - Comprehensive testing
  - Documentation

**Total:** ~9-13 hours of focused work

---

## 🎯 Success Criteria

### Must Have (MVP)
- ✅ `sf <prompt>` generates story via MCP
- ✅ Progress polling shows phase updates
- ✅ `sf continue` resumes via MCP
- ✅ `sf extend` extends via MCP
- ✅ All existing tests passing
- ✅ No business logic in CLI (all in server)

### Should Have (Enhanced)
- ✅ All 18 MCP tools accessible via CLI
- ✅ Rich formatting preserved
- ✅ New commands documented
- ✅ Integration tests added

### Nice to Have (Polish)
- ✅ Beautiful Rich tables/progress bars
- ✅ Helpful error messages
- ✅ User guide updated
- ✅ Performance optimized

---

## 📝 Next Steps

### Immediate Actions
1. Create `storyforge/client/formatters.py` for Rich display helpers
2. Start refactoring `main()` command in `StoryForge.py`
3. Add progress polling implementation
4. Update tests to mock MCP client

### Questions to Resolve
- Should we keep `StoryForge.py` or create new `client/cli.py`?
  - **Recommendation:** Keep `StoryForge.py` for backward compatibility
- How to handle debug mode (test_story.txt)?
  - **Recommendation:** Keep as local fallback, bypass server
- Should config init stay local or go through MCP?
  - **Recommendation:** Stay local (no server state needed)

---

## 🐛 Known Issues

### Current
- None blocking - MCP client complete and tested

### Potential
- Server startup latency on first CLI call
- Windows compatibility (stdio transport)
- Progress polling overhead (mitigated by caching)

---

**Last Updated:** November 10, 2025  
**Next:** Begin CLI refactoring with Phase 1 (MVP)
