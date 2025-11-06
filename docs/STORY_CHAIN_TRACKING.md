# Story Chain Tracking

## TL;DR

Story chains automatically track extended stories across multiple generations:
- `sf extend` shows the full chain history before extending
- `sf export-chain` combines all parts of a chain into a single readable file
- Chains are tracked automatically - no extra user action needed
- Only works for stories extended at least once (2+ parts)

**[Skip to Usage Examples](#usage-examples)** | **[Skip to Implementation](#implementation-details)**

---

## Why Use Story Chains?

✓ **Never lose context** - See exactly which stories led to the current one  
✓ **Easy sharing** - Export the complete saga in one file  
✓ **Better workflow** - Know what you're extending before you commit  
✓ **Automatic tracking** - Works behind the scenes, nothing to configure  

## Quick Reference

| Feature | Command | What it Does |
|---------|---------|-------------|
| View chain while extending | `sf extend` | Shows lineage before continuing story |
| Export complete chain | `sf export-chain` | Combines all parts into one file |
| Export specific chain | `sf export-chain -c name` | Export by filename match |
| Custom output location | `sf export-chain -o file.txt` | Specify output filename |

---

## How It Works

```
Original Story (A)
    ↓ extend
Extended Story (B) ← references A
    ↓ extend  
Extended Story (C) ← references B

When viewing C, we trace back: C → B → A
```

Story chains work by:
1. **Tracking the parent** - When you extend a story, the parent filename is stored
2. **Following the links** - The chain is reconstructed by following parent references backward
3. **Showing the lineage** - All parts are displayed in chronological order
4. **Preventing cycles** - Circular references are detected and prevented

### What Gets Tracked

Each extended story stores:
- Parent story filename (in `**Extended From:**` metadata field)
- Original prompt
- Generation timestamp
- Characters, settings, and other metadata

### Current Limitation

Currently, when you extend a story multiple times, there's no automatic way to trace back through the entire chain. This feature adds that capability by storing parent references in context metadata.

---

## Usage Examples

### Example 1: Creating a Simple Chain (2 parts)

```bash
# 1. Generate the original story
sf "A wizard discovers a magical artifact" --characters Merlin --age-range young-adult
# Save as context (press 'c' after generation)
# Creates: wizard_artifact_20251105_120000.md

# 2. Extend the story
sf extend
# Select story #1
# The wizard's journey continues...
# Save as context (press 'c')
# Creates: wizard_artifact_20251105_120000_20251105_130000.md
```

### Example 2: Multi-Level Chain (3+ parts)

```bash
# 3. Extend again to continue the saga
sf extend
# Select the extended story
# Now displays the chain:
# 📚 Story Chain:
#   1. wizard_artifact_20251105_120000 (2025-11-05 12:00:00)
#   2. wizard_artifact_20251105_120000_20251105_130000 (2025-11-05 13:00:00)
#
# Continue the story further...
# Save as context (press 'c')
# Creates: wizard_artifact_20251105_120000_20251105_130000_20251105_140000.md
```

### Example 3: Exporting a Complete Chain

```bash
# Export interactively (shows only chains, not one-shots)
sf export-chain

# Output shows available chains:
# Available story chains to export:
#
# 1. wizard_artifact_20251105_120000_20251105_140000.md
#    📚 Chain: 3 parts
#       └─ Part 1: wizard_artifact_20251105_120000.md (original)
#       └─ Part 2: wizard_artifact_20251105_120000_20251105_130000.md
#       └─ Part 3: wizard_artifact_20251105_120000_20251105_140000.md (latest)
#    Generated: 2025-11-05 14:00:00
#    Prompt: A wizard discovers a magical artifact...
#
# Select a story chain to export: 1
#
# ✓ Complete story chain exported to: complete_story_wizard_artifact_20251105_150000.txt

# Export specific story
sf export-chain -c wizard_artifact -o my_wizard_saga.txt
```

### Context File Structure

**Original Story:**
```markdown
# Story Context: A wizard discovers a magical artifact

**Generated on:** 2025-11-05 12:00:00
**Original Prompt:** A wizard discovers a magical artifact
**Characters:** Merlin
**Setting:** Ancient forest

---
## Story Preview
Merlin walked through the ancient forest...
```

**Extended Story:**
```markdown
# Story Context: A wizard discovers a magical artifact

**Generated on:** 2025-11-05 13:00:00
**Original Prompt:** A wizard discovers a magical artifact
**Extended From:** wizard_artifact_20251105_120000    ← Parent reference
**Characters:** Merlin
**Setting:** Ancient forest, Dark cavern

---
## Story Preview
The artifact began to glow as Merlin entered the cavern...
```

### Exported Chain Output Format

```
================================================================================
COMPLETE STORY CHAIN
================================================================================

Generated: 2025-11-05 15:00:00
Total stories in chain: 3

Original prompt: A wizard discovers a magical artifact
Characters: Merlin
Setting: Ancient forest

================================================================================

================================================================================
PART 1 of 3
================================================================================
Generated: 2025-11-05 12:00:00
Source: wizard_artifact_20251105_120000.md
================================================================================

[Original story content...]

================================================================================
PART 2 of 3
================================================================================
Generated: 2025-11-05 13:00:00
Source: wizard_artifact_20251105_120000_20251105_130000.md
Extended from: wizard_artifact_20251105_120000
================================================================================

[First extension content...]

================================================================================
PART 3 of 3
================================================================================
Generated: 2025-11-05 14:00:00
Source: wizard_artifact_20251105_120000_20251105_140000.md
Extended from: wizard_artifact_20251105_120000_20251105_130000
================================================================================

[Second extension content...]

================================================================================
END OF STORY CHAIN
================================================================================
```

---

## Troubleshooting

**Q: Chain doesn't show all parts when I extend**  
A: Check that parent context files haven't been deleted. The chain breaks at missing files.

**Q: `export-chain` shows "No chains found"**  
A: The export command only lists extended stories (2+ parts). Use `sf extend` first to create a chain.

**Q: I see duplicate stories in the chain**  
A: This shouldn't happen due to cycle detection. If it does, the context metadata may be corrupted - try deleting and regenerating.

**Q: Can I export a single story (non-chain)?**  
A: Not via `export-chain` - it filters to chains only. For single stories, the original output file already contains the complete content.

---

## Implementation Details

> **Note for Developers:** This section contains technical implementation details. Users can skip this section.

### Overview

Five key components enable chain tracking:

1. **Store parent reference** when extending (in checkpoint data)
2. **Save parent reference** to context metadata (in saved .md files)
3. **Parse parent reference** when loading contexts
4. **Reconstruct chain** by following parent links backward
5. **Export chain** by combining all parts into one file

### 1. Add Parent Tracking to Context Metadata

**File:** `storyforge/phase_executor.py`  
**Method:** `_phase_context_save`  
**What to add:**

```python
context_content = f"# Story Context: {prompt_summary}\n\n"
context_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
context_content += f"**Original Prompt:** {prompt_summary}\n\n"

# Add parent story tracking for extensions
if self.checkpoint_data.context_data and self.checkpoint_data.context_data.get("source_context_file"):
    parent_file = Path(self.checkpoint_data.context_data["source_context_file"]).stem
    context_content += f"**Extended From:** {parent_file}\n\n"

# ... continue with existing metadata (characters, setting, etc.)
```

### 2. Store Parent Reference in Checkpoint Data

**File:** `storyforge/StoryForge.py`  
**Function:** `extend_story`  
**What to add:**

```python
# Prepare resolved configuration
resolved_config = {
    "backend": backend or config.get_field_value("system", "backend"),
    "output_directory": output_dir,
    # ... other existing config fields ...
    "continuation_mode": True,
    # Track source context file for chain tracking
    "source_context_file": str(selected_context["filepath"]),
}
```

### 3. Parse Parent Reference from Context Files

**File:** `storyforge/context.py`  
**Method:** `parse_context_metadata`  
**What to add:**

```python
# ... existing metadata parsing ...

# Extract parent story reference for chain tracking
extended_from_match = re.search(r"\*\*Extended From:\*\*\s*(.+)", content)
if extended_from_match:
    metadata["extended_from"] = extended_from_match.group(1).strip()
```

### 4. Add Chain Reconstruction Method

**File:** `storyforge/context.py`  
**Method:** `get_story_chain` (new method after `load_context_for_extension`)  

```python
def get_story_chain(self, filepath: Path) -> list[dict[str, Any]]:
    """
    Reconstruct the full story chain by tracing back parent references.
    
    Returns list of story metadata dicts in chronological order (oldest to newest).
    Prevents infinite loops via seen_files set.
    """
    chain: list[dict[str, Any]] = []
    current_file = filepath
    seen_files: set[str] = set()
    
    while current_file and str(current_file) not in seen_files:
        seen_files.add(str(current_file))
        
        try:
            metadata = self.parse_context_metadata(current_file)
            chain.insert(0, metadata)  # Add to beginning for chronological order
            
            # Check for parent reference
            if "extended_from" in metadata:
                parent_name = metadata["extended_from"]
                context_dir = self.get_context_directory()
                matching_files = list(context_dir.glob(f"{parent_name}*.md"))
                if matching_files:
                    current_file = matching_files[0]
                else:
                    break  # Parent not found
            else:
                break  # No parent reference, reached original story
        except Exception:
            break  # Error reading file
    
    return chain
```

### 5. Display Chain in Extend Command

**File:** `storyforge/StoryForge.py`  
**Function:** `extend_story`  
**What to add (after user selects a context):**

```python
selected_context = available_contexts[selection - 1]

# Show the story chain
story_chain = context_mgr.get_story_chain(selected_context["filepath"])
if len(story_chain) > 1:
    console.print("\n[bold cyan]📚 Story Chain:[/bold cyan]")
    for idx, story in enumerate(story_chain, 1):
        timestamp = story.get('timestamp', 'Unknown')
        prompt = story.get('prompt', 'No prompt')[:50]
        console.print(f"  {idx}. [dim]{story['filename']}[/dim]")
        console.print(f"     {timestamp} - {prompt}...")
    console.print()

# ... continue with existing preview display ...
```

### 6. Add Chain Export Method

**File:** `storyforge/context.py`  
**Method:** `write_chain_to_file` (new method after `get_story_chain`)

```python
def write_chain_to_file(self, filepath: Path, output_path: Path) -> Path:
    """
    Write the complete story chain to a single file.
    
    Implementation:
    1. Get the chain via get_story_chain()
    2. Add header with metadata (prompt, characters, total parts)
    3. For each story in chain:
       - Add separator with part number
       - Extract story content (skip metadata header)
       - Append to combined output
    4. Write to output_path
    
    Returns: Path to the created file
    """
    # See full implementation details in the complete code below
```

<details>
<summary>Full write_chain_to_file implementation (click to expand)</summary>
<details>
<summary>Full write_chain_to_file implementation (click to expand)</summary>

```python
def write_chain_to_file(self, filepath: Path, output_path: Path) -> Path:
    from datetime import datetime
    
    chain = self.get_story_chain(filepath)
    if not chain:
        raise ValueError("No stories found in chain")
    
    # Create the combined story content
    combined_content = []
    combined_content.append("=" * 80)
    combined_content.append("COMPLETE STORY CHAIN")
    combined_content.append("=" * 80)
    combined_content.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    combined_content.append(f"Total stories in chain: {len(chain)}\n")
    
    # Add metadata about the chain
    if chain:
        first_story = chain[0]
        combined_content.append(f"Original prompt: {first_story.get('prompt', 'Unknown')}")
        if 'characters' in first_story:
            combined_content.append(f"Characters: {first_story['characters']}")
        if 'setting' in first_story:
            combined_content.append(f"Setting: {first_story['setting']}")
    
    combined_content.append("=" * 80 + "\n")
    
    # Add each story in the chain
    for idx, story_meta in enumerate(chain, 1):
        combined_content.append(f"\n{'=' * 80}")
        combined_content.append(f"PART {idx} of {len(chain)}")
        combined_content.append(f"{'=' * 80}")
        combined_content.append(f"Generated: {story_meta.get('timestamp', 'Unknown')}")
        combined_content.append(f"Source: {story_meta.get('filename', 'Unknown')}")
        if idx > 1 and 'extended_from' in story_meta:
            combined_content.append(f"Extended from: {story_meta['extended_from']}")
        combined_content.append(f"{'=' * 80}\n")
        
        # Read and add the actual story content
        story_path = story_meta.get('filepath')
        if story_path and Path(story_path).exists():
            with open(story_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract just the story content (skip metadata header)
                story_start = content.find("---\n## Story Preview")
                if story_start != -1:
                    story_content = content[story_start + len("---\n## Story Preview\n\n"):]
                else:
                    parts = content.split("---", 2)
                    story_content = parts[2] if len(parts) > 2 else content
                combined_content.append(story_content.strip())
        else:
            combined_content.append(f"[Story content not found: {story_path}]")
        combined_content.append("\n")
    
    # Add footer
    combined_content.append("\n" + "=" * 80)
    combined_content.append("END OF STORY CHAIN")
    combined_content.append("=" * 80)
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_content))
    
    return output_path
```

</details>

### 7. Add CLI Command for Chain Export

**File:** `storyforge/StoryForge.py`  
**Command:** `export_chain` (new command after `extend_story`)

<details>
<summary>Full export_chain command implementation (click to expand)</summary>

```python
@app.command()
def export_chain(
    context: Annotated[Optional[str], typer.Option("--context", "-c")]=None,
    output: Annotated[Optional[str], typer.Option("--output", "-o")]=None,
    config_file: Annotated[Optional[str], typer.Option("--config")]=None,
) -> None:
    """Export a complete story chain to a single file."""
    console.print("[bold cyan]📚 Export Story Chain[/bold cyan]\n")
    
    config = Config.load_config(config_file)
    context_mgr = ContextManager(config)
    available_contexts = context_mgr.list_available_contexts()
    
    if not available_contexts:
        console.print("[yellow]No context files found.[/yellow]")
        raise typer.Exit(1)
    
    if context:
        matches = [ctx for ctx in available_contexts if context.lower() in ctx["filename"].lower()]
        selected_context = matches[0]
    else:
        # Interactive selection - filter to only extended stories
        contexts_with_chains = []
        for ctx in available_contexts:
            chain = context_mgr.get_story_chain(ctx["filepath"])
            if len(chain) > 1:  # Only include chains (2+ parts)
                contexts_with_chains.append({"context": ctx, "chain": chain})
        
        if not contexts_with_chains:
            console.print("[yellow]No extended story chains found.[/yellow]")
            raise typer.Exit(1)
        
        # Display chains with lineage
        console.print("[bold]Available story chains to export:[/bold]\n")
        for idx, ctx_chain in enumerate(contexts_with_chains, 1):
            ctx, chain = ctx_chain["context"], ctx_chain["chain"]
            console.print(f"{idx}. [cyan]{ctx['filename']}[/cyan]")
            console.print(f"   📚 Chain: {len(chain)} parts")
            for chain_idx, story in enumerate(chain, 1):
                label = " (original)" if chain_idx == 1 else (" (latest)" if chain_idx == len(chain) else "")
                console.print(f"      └─ Part {chain_idx}: {story['filename']}{label}")
            console.print(f"   Prompt: {ctx.get('prompt', 'N/A')[:60]}...\n")
        
        selection = typer.prompt("\nSelect a story chain to export", type=int)
        selected_context = contexts_with_chains[selection - 1]["context"]
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = selected_context["filename"].replace("_context", "").replace(".md", "")
        output_path = Path(f"complete_story_{base_name}_{timestamp}.txt")
    
    # Export the chain
    try:
        result_path = context_mgr.write_chain_to_file(selected_context["filepath"], output_path)
        story_chain = context_mgr.get_story_chain(selected_context["filepath"])
        console.print(f"\n[bold green]✓ Exported to:[/bold green] {result_path}")
        console.print(f"[dim]Total parts: {len(story_chain)}[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
```

</details>

---

## Implementation Checklist

### Core Chain Tracking
- [ ] Add parent tracking to `_phase_context_save` in `phase_executor.py`
- [ ] Store source context file in `extend_story` resolved config in `StoryForge.py`
- [ ] Update `parse_context_metadata` to extract `extended_from` field in `context.py`
- [ ] Add `get_story_chain` method to `ContextManager` in `context.py`
- [ ] Display story chain in `extend_story` command in `StoryForge.py`

### Chain Export Feature
- [ ] Add `write_chain_to_file` method to `ContextManager` in `context.py`
- [ ] Add `export_chain` command to `StoryForge.py`

### Testing & Documentation
- [ ] Add unit tests for chain tracking (see test plan below)
- [ ] Add integration tests for export command
- [ ] Update `README.md` with chain tracking info
- [ ] Update `docs/CONFIGURATION.md` if needed

<details>
<summary>Testing Strategy (click to expand)</summary>

### Unit Tests (`tests/test_context.py`)

```python
def test_get_story_chain_single_story(tmp_path):
    """Chain with 1 story returns list with 1 item"""

def test_get_story_chain_multiple_extensions(tmp_path):
    """Chain with 3 stories returns all in correct order"""

def test_get_story_chain_missing_parent(tmp_path):
    """Chain breaks gracefully when parent is deleted"""

def test_get_story_chain_circular_reference(tmp_path):
    """Circular references don't cause infinite loops"""

def test_write_chain_to_file_multiple_stories(tmp_path):
    """Export combines all parts in chronological order"""

def test_write_chain_to_file_creates_directory(tmp_path):
    """Export creates output directory if needed"""
```

### Integration Tests (`tests/test_cli_integration.py`)

```python
def test_extend_displays_story_chain(tmp_path):
    """'sf extend' shows chain before extending"""

def test_export_chain_command(tmp_path):
    """'sf export-chain' creates combined file"""

def test_export_chain_with_output_path(tmp_path):
    """'sf export-chain -o file.txt' respects custom path"""

def test_export_chain_with_context_arg(tmp_path):
    """'sf export-chain -c name' finds correct chain"""
```

</details>

---

## Future Enhancements

1. **Visual Chain Display** - Tree view showing branches and alternate endings
2. **Chain Statistics** - Total word count, character development tracking across chain
3. **Branch Support** - Allow forking stories into multiple alternate timelines
4. **Chain Search** - Full-text search across all stories in a chain
5. **Export Formats** - PDF, EPUB, or HTML export options
6. **Chain Merging** - Combine multiple branches into single narrative

---

## Related Files

- **`storyforge/context.py`** - Context management and metadata parsing
- **`storyforge/phase_executor.py`** - Story generation and context saving
- **`storyforge/StoryForge.py`** - CLI commands including `extend` and `export-chain`
- **`storyforge/checkpoint.py`** - Checkpoint data structure
- **`tests/test_context.py`** - Context manager unit tests
- **`tests/test_cli_integration.py`** - End-to-end CLI tests

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options and settings
- [README.md](../README.md) - General usage, features, and getting started
        
        selection = typer.prompt("\nSelect a story chain to export", type=int)
        if selection < 1 or selection > len(contexts_with_chains):
            console.print("[red]Invalid selection.[/red]")
            raise typer.Exit(1)
        
        selected_context = contexts_with_chains[selection - 1]["context"]
    
    # Show the story chain
    story_chain = context_mgr.get_story_chain(selected_context["filepath"])
    console.print(f"\n[bold cyan]📚 Story Chain ({len(story_chain)} part{'s' if len(story_chain) != 1 else ''}):[/bold cyan]")
    for idx, story in enumerate(story_chain, 1):
        timestamp = story.get('timestamp', 'Unknown')
        console.print(f"  {idx}. [dim]{story['filename']}[/dim] ({timestamp})")
    console.print()
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = selected_context["filename"].replace("_context", "").replace(".md", "")
        output_path = Path(f"complete_story_{base_name}_{timestamp}.txt")
    
    # Export the chain
    console.print(f"[bold]Exporting chain to:[/bold] {output_path}")
    
    try:
        result_path = context_mgr.write_chain_to_file(selected_context["filepath"], output_path)
        console.print(f"\n[bold green]✓ Complete story chain exported to:[/bold green] {result_path}")
        console.print(f"[dim]Total parts combined: {len(story_chain)}[/dim]")
    except Exception as e:
        console.print(f"[red]Error exporting chain: {e}[/red]")
        raise typer.Exit(1)
```

### Usage Example

```bash
# Export a story chain interactively
sf export-chain

# Output:
# 📚 Export Story Chain
#
# Available story chains to export:
#
# 1. wizard_artifact_20251105_120000_20251105_140000.md
#    📚 Chain: 3 parts
#       └─ Part 1: wizard_artifact_20251105_120000.md (original)
#       └─ Part 2: wizard_artifact_20251105_120000_20251105_130000.md
#       └─ Part 3: wizard_artifact_20251105_120000_20251105_140000.md (latest)
#    Generated: 2025-11-05 14:00:00
#    Prompt: A wizard discovers a magical artifact...
#
# 2. knight_quest_20251105_080000_20251105_090000.md
#    📚 Chain: 2 parts
#       └─ Part 1: knight_quest_20251105_080000.md (original)
#       └─ Part 2: knight_quest_20251105_080000_20251105_090000.md (latest)
#    Generated: 2025-11-05 09:00:00
#    Prompt: A knight on a quest...
#
# Select a story chain to export: 1
#
# 📚 Story Chain (3 parts):
#   1. wizard_artifact_20251105_120000.md (2025-11-05 12:00:00)
#   2. wizard_artifact_20251105_120000_20251105_130000.md (2025-11-05 13:00:00)
#   3. wizard_artifact_20251105_120000_20251105_140000.md (2025-11-05 14:00:00)
#
# Exporting chain to: complete_story_wizard_artifact_20251105_120000_20251105_140000_20251105_150000.txt
#
# ✓ Complete story chain exported to: complete_story_wizard_artifact_20251105_120000_20251105_140000_20251105_150000.txt
# Total parts combined: 3

# If no extended stories exist:
sf export-chain

# Output:
# 📚 Export Story Chain
#
# No extended story chains found. Only single stories available.
# Extend a story first with 'sf extend' to create a chain.

# Export specific story to specific file
sf export-chain -c wizard_artifact -o my_wizard_saga.txt

# Export using full context filename
sf export-chain -c wizard_artifact_20251105_120000_20251105_140000 -o wizard_complete.txt
```

### Output File Format

The exported file will look like this:

```
================================================================================
COMPLETE STORY CHAIN
================================================================================

Generated: 2025-11-05 15:00:00
Total stories in chain: 3

Original prompt: A wizard discovers a magical artifact
Characters: Merlin
Setting: Ancient forest

================================================================================

================================================================================
PART 1 of 3
================================================================================
Generated: 2025-11-05 12:00:00
Source: wizard_artifact_20251105_120000.md
================================================================================

Merlin walked through the ancient forest, his staff glowing softly in the
twilight. The trees whispered secrets of an artifact hidden deep within...

[Story content continues...]

================================================================================
PART 2 of 3
================================================================================
Generated: 2025-11-05 13:00:00
Source: wizard_artifact_20251105_120000_20251105_130000.md
Extended from: wizard_artifact_20251105_120000
================================================================================

The artifact's glow intensified as Merlin approached the ancient cavern...

[Story content continues...]

================================================================================
PART 3 of 3
================================================================================
Generated: 2025-11-05 14:00:00
Source: wizard_artifact_20251105_120000_20251105_140000.md
Extended from: wizard_artifact_20251105_120000_20251105_130000
================================================================================

With the artifact secured, Merlin knew his quest was only beginning...

[Story content continues...]

================================================================================
END OF STORY CHAIN
================================================================================
```

## Future Enhancements

1. **Visual Chain Display**: Add a tree view showing the story chain with branches
2. **Chain Statistics**: Show total length, character development across chain
3. **Branch Support**: Allow forking stories into multiple branches (create alternate endings)
4. **Chain Search**: Search across all stories in a chain
5. **Auto-naming**: Automatically suggest names for extended stories based on chain depth
6. **Export Formats**: Support exporting to PDF, EPUB, or HTML formats
7. **Chain Merging**: Combine multiple branches into a single narrative

## Related Files

- `storyforge/context.py` - Context management and metadata parsing
- `storyforge/phase_executor.py` - Story generation and context saving
- `storyforge/StoryForge.py` - CLI commands including `extend`
- `storyforge/checkpoint.py` - Checkpoint data structure
- `tests/test_context.py` - Context manager tests
- `tests/test_cli_integration.py` - End-to-end CLI tests

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [README.md](../README.md) - General usage and features
