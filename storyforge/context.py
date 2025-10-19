"""
Context Management for StoryForge.

This module provides utilities for loading and managing story context from files
like character descriptions, background information, and story examples.

Implements extractive summarize-and-merge strategy to compact context while
preserving semantic content and respecting token budgets.
"""

from collections.abc import Callable
from pathlib import Path

from platformdirs import user_data_dir

# Use "StoryForge" as appauthor for user_data_dir to ensure user-agnostic,
# organization-consistent data storage


class ContextManager:
    """
    Manages story context loading and processing with extractive summarization.

    Supports:
    - Loading and concatenating multiple markdown context files
    - Extractive summarization to reduce token usage
    - Chunk-based relevance scoring against prompts
    - Token budget management with pinned high-value chunks
    - In-memory caching of context and summaries
    """

    def __init__(
        self,
        context_file_path: str | None = None,
        *,
        max_tokens: int | None = None,
        pinned_token_fraction: float = 0.2,
        tokenizer: Callable[[str], int] | None = None,
        summary_cache_dir: str | None = None,
    ):
        """
        Initialize the ContextManager.

        Summarization is always enabled. Context is intelligently compressed using
        extractive summarization while preserving relevant content.

        Args:
            context_file_path: Path to a specific context file. If None, searches standard locations.
            max_tokens: Token budget for returned context (best-effort). If None, returns all summaries.
            pinned_token_fraction: Fraction of budget reserved for high-relevance verbatim chunks (0-1).
            tokenizer: Optional callable to estimate tokens from text. Uses heuristic if None.
            summary_cache_dir: Optional directory for persisting summaries (not yet implemented).
        """
        self.context_file_path = context_file_path
        self._cached_context: str | None = None

        # Summarization configuration (always enabled)
        self.max_tokens = max_tokens
        self.pinned_token_fraction = pinned_token_fraction
        self.tokenizer = tokenizer
        self.summary_cache_dir = summary_cache_dir

        # Structured cache metadata: store summaries keyed by context signature
        self._summary_cache: dict[str, str] = {}

    def load_context(self) -> str | None:
        """
        Load context from all markdown files in the context directory.

        Returns:
            str: The concatenated context content from all .md files,
            or None if none found.

        New behavior:
        - Finds all .md files in the context directory (default or specified)
        - Sorts them by last modified date (oldest to newest)
        - Concatenates their contents

        """
        if self._cached_context is not None:
            return self._cached_context

        # If a specific file is set, use only that file
        if self.context_file_path:
            context_files = [Path(self.context_file_path)]
        else:
            import os

            # Allow tests to override the context directory via env var
            test_context_dir = os.environ.get("STORYTIME_TEST_CONTEXT_DIR")
            if test_context_dir:
                context_dir = Path(test_context_dir)
                if context_dir.exists() and context_dir.is_dir():
                    context_files = sorted(context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    context_files = []
            else:
                # Prefer ./context/ in the current working directory if it exists
                local_context_dir = Path("context")
                if local_context_dir.exists() and local_context_dir.is_dir():
                    context_files = sorted(local_context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    # Use lowercase 'storyforge' for normalized cross-platform paths
                    user_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
                    if user_dir.exists() and user_dir.is_dir():
                        context_files = sorted(user_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                    else:
                        context_files = []

        if not context_files:
            return None

        contents = []
        for file_path in context_files:
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        contents.append(f.read().strip())
                except OSError:
                    continue

        if not contents:
            return None

        self._cached_context = "\n\n".join(contents)
        return self._cached_context

    # --- Summarization helper methods ---

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses injected tokenizer if available, otherwise uses heuristic (chars/4).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count (minimum 1).
        """
        if self.tokenizer:
            try:
                return int(self.tokenizer(text))
            except Exception:
                pass
        return max(1, int(len(text) / 4))

    def _read_and_normalize(self, file_path: Path) -> list[str]:
        """
        Read file and return normalized paragraphs.

        Args:
            file_path: Path to file to read.

        Returns:
            List of paragraph strings (whitespace normalized).
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return []

        # Normalize newlines and split into paragraphs
        norm = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [p.strip() for p in norm.split("\n\n") if p.strip()]
        return paragraphs

    def _chunk_paragraphs(self, paragraphs: list[str], max_chunk_chars: int = 2000) -> list[str]:
        """
        Split large paragraphs into smaller chunks on sentence boundaries.

        Args:
            paragraphs: List of paragraph strings.
            max_chunk_chars: Maximum characters per chunk.

        Returns:
            List of chunk strings (may be larger than input if paragraphs are split).
        """
        chunks: list[str] = []
        for p in paragraphs:
            if len(p) <= max_chunk_chars:
                chunks.append(p)
            else:
                # Split on sentence boundaries (period + space)
                parts = [s.strip() for s in p.split(". ") if s.strip()]
                cur = []
                cur_len = 0
                for s in parts:
                    # Restore period if missing
                    piece = s + ("." if not s.endswith(".") else "")
                    if cur_len + len(piece) + 1 <= max_chunk_chars:
                        cur.append(piece)
                        cur_len += len(piece) + 1
                    else:
                        if cur:
                            chunks.append(" ".join(cur).strip())
                        cur = [piece]
                        cur_len = len(piece)
                if cur:
                    chunks.append(" ".join(cur).strip())
        return chunks

    def _extractive_summary(self, chunk: str, target_tokens: int) -> str:
        """
        Create extractive summary by selecting top sentences up to token budget.

        Args:
            chunk: Text to summarize.
            target_tokens: Target token count for summary.

        Returns:
            Extractive summary (subset of sentences from chunk).
        """
        # Split into sentences conservatively
        sentences = [s.strip() for s in chunk.replace("\n", " ").split(". ") if s.strip()]
        if not sentences:
            return chunk[: int(target_tokens * 4)] if target_tokens > 0 else chunk

        out: list[str] = []
        out_tokens = 0
        for s in sentences:
            # Restore period if missing
            s_text = s if s.endswith(".") else s + "."
            s_tokens = self._estimate_tokens(s_text)
            # Always include at least one sentence
            if out_tokens + s_tokens <= target_tokens or not out:
                out.append(s_text)
                out_tokens += s_tokens
            else:
                break

        return " ".join(out).strip()

    def _score_chunk(self, chunk: str, prompt: str) -> int:
        """
        Score chunk relevance to prompt using keyword overlap.

        Args:
            chunk: Text chunk to score.
            prompt: User prompt to compare against.

        Returns:
            Score (higher = more relevant). Returns 0 if either input is empty.
        """
        if not prompt or not chunk:
            return 0

        # Normalize to alphanumeric tokens for deterministic scoring
        prompt_words = {"".join(ch for ch in w.lower() if ch.isalnum()) for w in prompt.split()}
        chunk_words = ["".join(ch for ch in w.lower() if ch.isalnum()) for w in chunk.split()]
        score = sum(1 for w in chunk_words if w and w in prompt_words)
        return score

    def _merge_summaries_and_pins(self, items: list[dict], max_tokens: int | None) -> str:
        """
        Select pinned chunks and summaries to fit token budget.

        Uses greedy selection: pinned chunks first (up to pinned_token_fraction),
        then highest-scoring summaries by score/token ratio.

        Args:
            items: List of dicts with keys: 'summary', 'score', 'summary_tokens', 'is_pinned'.
            max_tokens: Token budget. If None, returns all summaries concatenated.

        Returns:
            Merged context string respecting budget.
        """
        if not items:
            return ""

        # No budget limit: return all summaries
        if max_tokens is None:
            return "\n\n".join(i["summary"] for i in items)

        budget = max_tokens
        pinned_budget = int(budget * self.pinned_token_fraction)
        remaining_budget = budget

        # Select pinned items first (sorted by score)
        pinned = [it for it in items if it.get("is_pinned")]
        pinned = sorted(pinned, key=lambda x: -x.get("score", 0))
        selected: list[str] = []
        used = 0
        for p in pinned:
            t = p.get("summary_tokens", self._estimate_tokens(p["summary"]))
            if used + t <= pinned_budget:
                selected.append(p["summary"])
                used += t

        remaining_budget -= used

        # Select others by score/token ratio
        others = [it for it in items if not it.get("is_pinned")]
        others = sorted(others, key=lambda x: -(x.get("score", 0) / max(1, x.get("summary_tokens", 1))))
        for o in others:
            t = o.get("summary_tokens", self._estimate_tokens(o["summary"]))
            if remaining_budget - t >= 0:
                selected.append(o["summary"])
                remaining_budget -= t
            else:
                # Try to truncate last item by characters if it helps
                if remaining_budget > 0:
                    approx_chars = int(remaining_budget * 4)
                    truncated = o["summary"][:approx_chars].rsplit(".", 1)[0]
                    if truncated:
                        selected.append(truncated.strip() + ".")
                        remaining_budget = 0
                break

        return "\n\n".join(s for s in selected if s)

    def _summarize_context(self, prompt: str) -> str | None:
        """
        Apply extractive summarize-and-merge to loaded context files.

        This method implements the full summarization pipeline:
        1. Discover and load context files
        2. Chunk paragraphs
        3. Create extractive summaries for each chunk
        4. Score chunks against prompt
        5. Select and merge summaries within token budget

        Args:
            prompt: User prompt for relevance scoring.

        Returns:
            Summarized context string, or None if no context files found.
        """
        # Discover files using existing logic
        if self.context_file_path:
            context_files = [Path(self.context_file_path)]
        else:
            import os

            test_context_dir = os.environ.get("STORYTIME_TEST_CONTEXT_DIR")
            if test_context_dir:
                context_dir = Path(test_context_dir)
                if context_dir.exists() and context_dir.is_dir():
                    context_files = sorted(context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    context_files = []
            else:
                local_context_dir = Path("context")
                if local_context_dir.exists() and local_context_dir.is_dir():
                    context_files = sorted(local_context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                else:
                    user_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
                    if user_dir.exists() and user_dir.is_dir():
                        context_files = sorted(user_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                    else:
                        context_files = []

        if not context_files:
            return None

        # Process each file: read, chunk, summarize, score
        all_items: list[dict] = []
        for file_path in context_files:
            if not file_path.exists():
                continue

            paragraphs = self._read_and_normalize(file_path)
            if not paragraphs:
                continue

            chunks = self._chunk_paragraphs(paragraphs)
            for chunk in chunks:
                chunk_tokens = self._estimate_tokens(chunk)
                # Target summary size: 20% of original, min 8 tokens
                target_tokens = max(8, int(chunk_tokens * 0.2))
                summary = self._extractive_summary(chunk, target_tokens)
                summary_tokens = self._estimate_tokens(summary)
                score = self._score_chunk(chunk, prompt)

                # Simple heuristic: pin chunks with very high scores (top 10% or score > threshold)
                # For now, we'll skip pinning and let all compete by score
                is_pinned = False

                all_items.append(
                    {
                        "chunk": chunk,
                        "summary": summary,
                        "score": score,
                        "chunk_tokens": chunk_tokens,
                        "summary_tokens": summary_tokens,
                        "is_pinned": is_pinned,
                    }
                )

        if not all_items:
            return None

        # Merge summaries within budget
        result = self._merge_summaries_and_pins(all_items, self.max_tokens)
        return result if result else None

    def _resolve_context_path(self) -> Path | None:
        """
        Resolve the context file path.

        Returns:
            Path: The resolved path to the context file.

        Future enhancements:
        - Support environment variable overrides
        - Search multiple default locations
        - Support URL-based context loading
        """
        if self.context_file_path:
            return Path(self.context_file_path)

        # Use cross-platform user data directory for context files
        # Normalized to lowercase 'storyforge' for consistent paths
        context_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        default_path = context_dir / "family.md"
        if default_path.exists():
            return default_path

        return None

    def extract_relevant_context(self, prompt: str) -> str | None:
        """
        Extract context relevant to the given prompt.

        Always applies extractive summarize-and-merge to intelligently compress
        context while preserving the most relevant content for the prompt.

        Args:
            prompt: The user's story prompt (used for relevance scoring).

        Returns:
            Summarized context string, or None if no context available.
        """
        return self._summarize_context(prompt)

    def clear_cache(self):
        """Clear all cached context data (raw and summarized)."""
        self._cached_context = None
        self._summary_cache.clear()


def get_default_context_manager() -> ContextManager:
    """
    Get a default context manager instance.

    Returns:
        ContextManager: Configured with default settings
    """
    return ContextManager()
