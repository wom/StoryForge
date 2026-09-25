# Story Chain Tracking

## TL;DR

Story chains track saved story contexts across multiple generations:
- `sf extend` offers saved stories and a preview; the TUI also shows chain length
- `sf export-chain` combines all parts of a chain into a single readable file
- Check **Save as future story context** in the TUI, or answer yes to the classic context-save prompt, when finishing
  the original and each continuation you want to extend
- Only works for stories extended at least once (2+ parts)

---

## Quick Reference

| Feature | Command | What it Does |
|---------|---------|-------------|
| Choose a saved story to extend | `sf extend` | Shows a preview (and chain length in the TUI) |
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

Each extended story stores a `**Extended From:**` metadata field referencing its parent. Chains are reconstructed by following these references backward, with cycle detection to prevent infinite loops.

---

## Usage Examples

### Creating a Chain

```bash
# 1. Generate the original story
sf "A wizard discovers a magical artifact" --character Merlin
# In the TUI media step, check "Save as future story context" before finishing.
# In the classic interface, answer yes to "Save this story as future context?"

# 2. Extend it
sf extend
# Select the saved story, generate a continuation, then select
# the context-save option again before finishing.

# 3. Extend again
sf extend
# Select the saved continuation. The TUI shows its chain length and preview;
# use export-chain to read every part together.
```

### Exporting a Chain

```bash
# Interactive — lists only multi-part chains
sf export-chain

# By name match
sf export-chain -c wizard_artifact

# Custom output file
sf export-chain -c wizard_artifact -o my_saga.txt
```

The exported file contains all story parts in chronological order with section dividers and metadata.

---

## Troubleshooting

**Chain doesn't show all parts?**
Check that you saved future context for both the original and each continuation, and that parent
context files have not been deleted. A missing parent breaks the chain.

**`export-chain` shows "No chains found"?**
Only saved chains with 2+ parts are listed. Save the original as context, extend it, and save that continuation as
context before exporting.

**Can I export a single (non-chain) story?**
Not via `export-chain`. Single stories are already complete in their output directory.

---

## For Developers

Chain tracking is implemented across three files:

- **`storyforge/context.py`** — `get_story_chain()`, `write_chain_to_file()`, `parse_context_metadata()` (extracts `extended_from`)
- **`storyforge/phase_executor.py`** — `_phase_context_save()` writes the `**Extended From:**` parent reference
- **`storyforge/workflow.py`** — creates extension drafts and exports chains for both MCP-backed clients

See also: [CONFIGURATION.md](CONFIGURATION.md) · [README](../README.md)
