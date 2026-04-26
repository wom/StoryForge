# Story Chain Tracking

## TL;DR

Story chains automatically track extended stories across multiple generations:
- `sf extend` shows the full chain history before extending
- `sf export-chain` combines all parts of a chain into a single readable file
- Chains are tracked automatically — no extra user action needed
- Only works for stories extended at least once (2+ parts)

---

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

Each extended story stores a `**Extended From:**` metadata field referencing its parent. Chains are reconstructed by following these references backward, with cycle detection to prevent infinite loops.

---

## Usage Examples

### Creating a Chain

```bash
# 1. Generate the original story
sf "A wizard discovers a magical artifact" --character Merlin

# 2. Extend it
sf extend
# Select the story → generates a continuation → save as context

# 3. Extend again
sf extend
# Shows the chain:
#   📚 Story Chain:
#     1. wizard_artifact_20251105_120000
#     2. wizard_artifact_20251105_120000_20251105_130000
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
Check that parent context files haven't been deleted — the chain breaks at missing files.

**`export-chain` shows "No chains found"?**
Only extended stories (2+ parts) are listed. Use `sf extend` first to create a chain.

**Can I export a single (non-chain) story?**
Not via `export-chain`. Single stories are already complete in their output directory.

---

## For Developers

Chain tracking is implemented across three files:

- **`storyforge/context.py`** — `get_story_chain()`, `write_chain_to_file()`, `parse_context_metadata()` (extracts `extended_from`)
- **`storyforge/phase_executor.py`** — `_phase_context_save()` writes the `**Extended From:**` parent reference
- **`storyforge/StoryForge.py`** — `extend_story()` displays chain, `export_chain()` CLI command

See also: [CONFIGURATION.md](CONFIGURATION.md) · [README](../README.md)
