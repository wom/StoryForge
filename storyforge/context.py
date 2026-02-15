"""
Context Management for StoryForge.

This module provides utilities for loading and managing story context from files
like character descriptions, background information, and story examples.

Implements extractive summarize-and-merge strategy to compact context while
preserving semantic content and respecting token budgets.
"""

import json
import logging
import random
import re
from collections.abc import Callable
from math import ceil
from pathlib import Path
from typing import Any

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
    - Temporal stratified sampling for context file selection
    - Character registry for persistent character memory
    - Sentence deduplication to avoid repeated content
    - In-memory caching of context and summaries
    """

    # --- File selection constants ---
    RECENT_FILE_COUNT: int = 5
    STRATIFIED_THRESHOLD: int = 15
    SAMPLES_PER_ERA: int = 2
    NUM_ERAS: int = 5

    # --- Character registry constants ---
    MAX_REGISTRY_TOKENS: int = 1000
    MAX_TRAITS_PER_CHARACTER: int = 5
    CHARACTER_SCORE_BONUS: int = 3
    COMMON_STOPWORDS: set[str] = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "his",
        "her",
        "our",
        "your",
        "their",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "nor",
        "so",
        "up",
        "out",
        "if",
        "about",
        "into",
        "over",
        "after",
        "then",
        "once",
        "story",
        "context",
        "chapter",
        "part",
        "generated",
        "original",
        "prompt",
        "setting",
        "characters",
        "theme",
        "tone",
        "extended",
        "refinements",
        "applied",
    }

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

        # Context intelligence state
        self._has_old_context: bool = False
        self._known_characters: set[str] = set()

    def _discover_context_files(self) -> list[Path]:
        """Discover context files from configured path, env override, or default locations.

        Returns:
            List of Path objects for discovered .md context files, sorted by mtime.
        """
        if self.context_file_path:
            return [Path(self.context_file_path)]

        import os

        test_context_dir = os.environ.get("STORYFORGE_TEST_CONTEXT_DIR")
        if test_context_dir:
            context_dir = Path(test_context_dir)
            if context_dir.exists() and context_dir.is_dir():
                return sorted(context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
            return []

        # Prefer ./context/ in the current working directory if it exists
        local_context_dir = Path("context")
        if local_context_dir.exists() and local_context_dir.is_dir():
            return sorted(local_context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)

        # Use lowercase 'storyforge' for normalized cross-platform paths
        user_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
        if user_dir.exists() and user_dir.is_dir():
            return sorted(user_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)

        return []

    def _select_context_files(self, files: list[Path]) -> list[Path]:
        """Select a representative subset of context files using temporal stratified sampling.

        For small file pools (<= STRATIFIED_THRESHOLD), returns all files.
        For larger pools, always includes recent files and samples from temporal
        eras across the full history, ensuring both continuity and long-ago callbacks.

        Sets self._has_old_context = True when old-era files are included.

        Args:
            files: All discovered context files, sorted by mtime (oldest first).

        Returns:
            Selected subset, sorted by mtime (oldest first).
        """
        if len(files) <= self.STRATIFIED_THRESHOLD:
            return files

        # Always include the most recent files for continuity
        recent = files[-self.RECENT_FILE_COUNT :]
        old_pool = files[: -self.RECENT_FILE_COUNT]

        # Divide old files into temporal eras and sample from each
        eras = self._divide_into_eras(old_pool, self.NUM_ERAS)
        sampled: list[Path] = []
        for era in eras:
            k = min(self.SAMPLES_PER_ERA, len(era))
            sampled.extend(random.sample(era, k))

        self._has_old_context = bool(sampled)

        # Combine and restore chronological order
        combined = sampled + recent
        return sorted(combined, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _divide_into_eras(files: list[Path], num_eras: int) -> list[list[Path]]:
        """Divide a list of files into roughly equal temporal eras.

        Args:
            files: Files sorted by mtime (oldest first).
            num_eras: Number of eras to create.

        Returns:
            List of file lists, one per era.
        """
        if not files or num_eras <= 0:
            return []
        chunk_size = ceil(len(files) / num_eras)
        return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

    def _deduplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from text, keeping first occurrence.

        Comparison is case-insensitive and ignores leading/trailing whitespace.
        Handles sentences split across paragraph boundaries.

        Args:
            text: Text potentially containing duplicate sentences.

        Returns:
            Text with duplicate sentences removed.
        """
        if not text:
            return text

        # Split into sentences per paragraph, dedup across all paragraphs
        # First, split into paragraphs, then sentences within each paragraph
        paragraphs = text.split("\n\n")
        seen: set[str] = set()
        deduped_paragraphs: list[str] = []

        for para in paragraphs:
            parts = para.split(". ")
            unique: list[str] = []
            for part in parts:
                normalized = part.strip().lower()
                if normalized.endswith("."):
                    normalized = normalized[:-1]
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique.append(part)
            if unique:
                result_para = ". ".join(unique)
                deduped_paragraphs.append(result_para)

        result = "\n\n".join(deduped_paragraphs)
        # Ensure trailing period if original had one
        if text.rstrip().endswith(".") and not result.rstrip().endswith("."):
            result = result.rstrip() + "."
        return result

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

        context_files = self._discover_context_files()

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
                logging.getLogger(__name__).debug("Tokenizer failed, falling back to heuristic", exc_info=True)
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
        Score chunk relevance to prompt using keyword overlap and character bonus.

        Base score is keyword overlap count.  An additional CHARACTER_SCORE_BONUS
        is added for each known character name that appears in both the prompt
        and the chunk.

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
        base_score = sum(1 for w in chunk_words if w and w in prompt_words)

        # Character bonus: boost when known characters appear in both prompt and chunk
        character_bonus = 0
        if self._known_characters:
            prompt_lower = prompt.lower()
            chunk_lower = chunk.lower()
            for name in self._known_characters:
                name_lower = name.lower()
                if name_lower in prompt_lower and name_lower in chunk_lower:
                    character_bonus += self.CHARACTER_SCORE_BONUS

        return base_score + character_bonus

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

        # No budget limit: return all summaries (still deduplicated)
        if max_tokens is None:
            return self._deduplicate_sentences("\n\n".join(i["summary"] for i in items))

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

        return self._deduplicate_sentences("\n\n".join(s for s in selected if s))

    def _summarize_context(self, prompt: str) -> str | None:
        """
        Apply extractive summarize-and-merge to loaded context files.

        This method implements the full summarization pipeline:
        1. Discover and select context files (temporal stratified sampling)
        2. Populate known characters for scoring bonuses
        3. Chunk paragraphs
        4. Create extractive summaries for each chunk
        5. Score chunks against prompt (keyword + character bonus)
        6. Select and merge summaries within token budget
        7. Deduplicate sentences
        8. Prepend character registry

        Args:
            prompt: User prompt for relevance scoring.

        Returns:
            Summarized context string, or None if no context files found.
        """
        # Discover all files, then select a representative subset
        all_files = self._discover_context_files()

        if not all_files:
            return None

        # Populate known characters for scoring (from registry or metadata)
        # Always attempt population if set is empty (registry may have been
        # updated since last call, so a previous zero-result shouldn't block).
        if not self._known_characters:
            self._populate_known_characters()

        # Select files using temporal stratified sampling
        context_files = self._select_context_files(all_files)

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

        # Reserve tokens for character registry if available
        registry_text = self.format_registry_for_prompt()
        registry_tokens = self._estimate_tokens(registry_text) if registry_text else 0

        merge_budget = self.max_tokens
        if merge_budget is not None and registry_tokens > 0:
            merge_budget = max(1, merge_budget - registry_tokens)

        # Merge summaries within budget
        result = self._merge_summaries_and_pins(all_items, merge_budget)

        # Prepend character registry
        if registry_text and result:
            result = registry_text + "\n\n" + result
        elif registry_text:
            result = registry_text

        return result if result else None

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

    # --- Character Registry ---

    def _get_registry_path(self) -> Path:
        """Get the path to the character registry JSON file."""
        return self.get_context_directory() / "character_registry.json"

    def _load_registry(self) -> dict[str, Any]:
        """Load the character registry from disk.

        Returns:
            Registry dict with 'characters' and 'last_updated' keys.
            Returns empty structure if file is missing or corrupt.
        """
        registry_path = self._get_registry_path()
        if not registry_path.exists():
            return {"characters": {}, "last_updated": None}
        try:
            with open(registry_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "characters" not in data:
                return {"characters": {}, "last_updated": None}
            return data
        except (json.JSONDecodeError, OSError):
            return {"characters": {}, "last_updated": None}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """Save the character registry to disk.

        Args:
            registry: Registry dict to persist.
        """
        from datetime import datetime

        registry_path = self._get_registry_path()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry["last_updated"] = datetime.now().isoformat()
        try:
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
        except OSError:
            logging.getLogger(__name__).warning("Failed to save character registry", exc_info=True)

    def _extract_characters_from_story(self, content: str, metadata: dict[str, Any]) -> dict[str, list[str]]:
        """Extract character names and trait sentences from a story.

        Uses two sources:
        1. Explicit **Characters:** metadata field
        2. Heuristic NER: capitalized words appearing 3+ times that aren't stopwords

        Args:
            content: Full story text content.
            metadata: Parsed metadata dict from parse_context_metadata().

        Returns:
            Dict mapping character names to lists of trait sentences.
        """
        characters: dict[str, list[str]] = {}

        # Source 1: Explicit metadata
        char_field = metadata.get("characters", "")
        if char_field:
            for name in str(char_field).split(","):
                name = name.strip()
                if name and len(name) > 1:
                    characters[name] = []

        # Source 2: Heuristic NER — capitalized words appearing 3+ times
        # Extract story text (after ## Story header if present)
        story_match = re.search(r"##\s*Story\s*\n+(.+)", content, re.DOTALL)
        story_text = story_match.group(1) if story_match else content

        word_counts: dict[str, int] = {}
        for word in re.findall(r"\b([A-Z][a-z]{2,})\b", story_text):
            if word.lower() not in self.COMMON_STOPWORDS:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count >= 3 and word not in characters:
                characters[word] = []

        # Extract trait sentences: sentences containing character names
        sentences = [s.strip() for s in story_text.replace("\n", " ").split(". ") if s.strip()]
        for name in characters:
            traits = []
            for sentence in sentences:
                if name in sentence and len(sentence) > 15:
                    # Clean up and cap sentence length
                    trait = sentence if sentence.endswith(".") else sentence + "."
                    if len(trait) > 200:
                        trait = trait[:200].rsplit(" ", 1)[0] + "..."
                    traits.append(trait)
                    if len(traits) >= 3:
                        break
            characters[name] = traits

        return characters

    def build_character_registry(self) -> str:
        """Build the character registry by scanning all context files.

        Performs a full scan of every context file, extracts characters and traits,
        and saves the registry to disk.

        Returns:
            Formatted registry string for prompt injection.
        """
        context_files = self._discover_context_files()
        if not context_files:
            return ""

        registry: dict[str, Any] = {"characters": {}, "last_updated": None}

        for file_path in context_files:
            if not file_path.exists():
                continue
            try:
                metadata = self.parse_context_metadata(file_path)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                extracted = self._extract_characters_from_story(content, metadata)
                filename = file_path.stem

                for name, traits in extracted.items():
                    if name not in registry["characters"]:
                        registry["characters"][name] = {
                            "first_appeared": filename,
                            "appearances": [filename],
                            "traits": traits[: self.MAX_TRAITS_PER_CHARACTER],
                        }
                    else:
                        char = registry["characters"][name]
                        if filename not in char["appearances"]:
                            char["appearances"].append(filename)
                        # Add new unique traits up to cap
                        existing_lower = {t.lower() for t in char["traits"]}
                        for trait in traits:
                            if (
                                trait.lower() not in existing_lower
                                and len(char["traits"]) < self.MAX_TRAITS_PER_CHARACTER
                            ):
                                char["traits"].append(trait)
                                existing_lower.add(trait.lower())
            except Exception:
                logging.getLogger(__name__).debug("Skipping file during registry build: %s", file_path, exc_info=True)
                continue

        self._save_registry(registry)
        return self.format_registry_for_prompt()

    def update_character_registry(self, story_content: str, metadata: dict[str, Any], filename: str) -> None:
        """Incrementally update the character registry with a newly saved story.

        Extracts characters from the new story and merges them into the existing
        persistent registry without requiring a full rebuild.

        Args:
            story_content: Full text content of the new story.
            metadata: Metadata dict for the new story.
            filename: Stem filename of the saved context file.
        """
        registry = self._load_registry()
        extracted = self._extract_characters_from_story(story_content, metadata)

        for name, traits in extracted.items():
            if name not in registry["characters"]:
                registry["characters"][name] = {
                    "first_appeared": filename,
                    "appearances": [filename],
                    "traits": traits[: self.MAX_TRAITS_PER_CHARACTER],
                }
            else:
                char = registry["characters"][name]
                if filename not in char["appearances"]:
                    char["appearances"].append(filename)
                existing_lower = {t.lower() for t in char["traits"]}
                for trait in traits:
                    if trait.lower() not in existing_lower and len(char["traits"]) < self.MAX_TRAITS_PER_CHARACTER:
                        char["traits"].append(trait)
                        existing_lower.add(trait.lower())

        self._save_registry(registry)

    def format_registry_for_prompt(self) -> str:
        """Format the character registry as compact markdown for prompt injection.

        Loads the registry from disk and formats it as a brief character summary.
        Characters are sorted by number of appearances (most frequent first).
        If the registry exceeds MAX_REGISTRY_TOKENS, least-frequent characters
        are dropped.

        Returns:
            Formatted markdown string, or empty string if no characters.
        """
        registry = self._load_registry()
        characters = registry.get("characters", {})
        if not characters:
            return ""

        # Sort by appearance count (most frequent first)
        sorted_chars = sorted(
            characters.items(),
            key=lambda x: len(x[1].get("appearances", [])),
            reverse=True,
        )

        lines = ["## Known Characters"]
        for name, data in sorted_chars:
            appearances = len(data.get("appearances", []))
            traits = data.get("traits", [])
            # Compact format: name (N stories) - first trait summary
            trait_summary = "; ".join(t.rstrip(".") for t in traits[:2]) if traits else "no details"
            story_word = "story" if appearances == 1 else "stories"
            lines.append(f"**{name}** ({appearances} {story_word}) - {trait_summary}")

        result = "\n".join(lines)

        # Trim to budget by dropping least-frequent characters
        while self._estimate_tokens(result) > self.MAX_REGISTRY_TOKENS and len(lines) > 2:
            lines.pop()  # Remove last (least frequent) character
            result = "\n".join(lines)

        return result

    def _populate_known_characters(self) -> None:
        """Populate the set of known character names for scoring bonuses.

        Reads from the persistent character registry first. Falls back to
        scanning **Characters:** metadata from context files.
        """
        registry = self._load_registry()
        characters = registry.get("characters", {})

        if characters:
            self._known_characters = set(characters.keys())
            return

        # Fallback: scan metadata fields
        context_files = self._discover_context_files()
        for file_path in context_files:
            try:
                metadata = self.parse_context_metadata(file_path)
                char_field = metadata.get("characters", "")
                if char_field:
                    for name in str(char_field).split(","):
                        name = name.strip()
                        if name and len(name) > 1:
                            self._known_characters.add(name)
            except Exception:
                continue

    @property
    def has_old_context(self) -> bool:
        """Whether old-era context files were included via temporal sampling."""
        return self._has_old_context

    def clear_cache(self):
        """Clear all cached context data (raw and summarized)."""
        self._cached_context = None
        self._summary_cache.clear()

    def get_context_directory(self) -> Path:
        """
        Get the context directory path.

        Returns:
            Path: The context directory (user data directory).
        """
        return Path(user_data_dir("storyforge", "storyforge")) / "context"

    def list_available_contexts(self) -> list[dict[str, Any]]:
        """
        List all saved context files with metadata.

        Returns:
            List of dicts containing:
            - filepath: Path to context file
            - filename: Base name
            - timestamp: Generation timestamp
            - characters: List of character names
            - theme: Story theme
            - age_group: Target age
            - preview: First 200 chars of story
        """
        context_files: list[dict[str, Any]] = []
        context_dir = self.get_context_directory()

        if not context_dir.exists():
            return context_files

        for md_file in sorted(context_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                metadata = self.parse_context_metadata(md_file)
                context_files.append(metadata)
            except Exception:
                logging.getLogger(__name__).debug("Skipping unparseable context file %s", md_file)
                continue

        return context_files

    def parse_context_metadata(self, filepath: Path) -> dict[str, Any]:
        """
        Extract metadata from context file frontmatter.

        Context files are markdown with bold headers:
        # Story Context: [title]
        **Generated:** 2025-10-22 19:55:22
        **Characters:** Moe, Curly
        ...

        Args:
            filepath: Path to .md context file

        Returns:
            Dict with extracted metadata
        """
        metadata: dict[str, Any] = {
            "filepath": filepath,
            "filename": filepath.stem,
            "timestamp": "Unknown",
        }

        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            # Extract timestamp from **Generated on:** or **Generated:** field
            generated_match = re.search(r"\*\*Generated(?: on)?:\*\*\s*(.+)", content)
            if generated_match:
                metadata["timestamp"] = generated_match.group(1).strip()

            # Extract characters from **Characters:** field
            characters_match = re.search(r"\*\*Characters:\*\*\s*(.+)", content)
            if characters_match:
                metadata["characters"] = characters_match.group(1).strip()

            # Extract theme from **Theme:** field
            theme_match = re.search(r"\*\*Theme:\*\*\s*(.+)", content)
            if theme_match:
                metadata["theme"] = theme_match.group(1).strip()

            # Extract age group from **Age Group:** field
            age_match = re.search(r"\*\*Age Group:\*\*\s*(.+)", content)
            if age_match:
                metadata["age_group"] = age_match.group(1).strip()

            # Extract tone from **Tone:** field
            tone_match = re.search(r"\*\*Tone:\*\*\s*(.+)", content)
            if tone_match:
                metadata["tone"] = tone_match.group(1).strip()

            # Extract art style from **Art Style:** field
            art_style_match = re.search(r"\*\*Art Style:\*\*\s*(.+)", content)
            if art_style_match:
                metadata["art_style"] = art_style_match.group(1).strip()

            # Extract original prompt from **Original Prompt:** field
            prompt_match = re.search(r"\*\*Original Prompt:\*\*\s*(.+)", content)
            if prompt_match:
                metadata["prompt"] = prompt_match.group(1).strip()

            # Extract parent story reference for chain tracking
            extended_from_match = re.search(r"\*\*Extended From:\*\*\s*(.+)", content)
            if extended_from_match:
                metadata["extended_from"] = extended_from_match.group(1).strip()

            # Extract story preview - find the story section after metadata
            # Look for "## Story" or just get text after multiple newlines
            story_match = re.search(r"##\s*Story\s*\n+(.+)", content, re.DOTALL)
            if story_match:
                story_text = story_match.group(1).strip()
                # Get first 200 characters
                metadata["preview"] = story_text[:200].replace("\n", " ")
            else:
                # Fallback: get first 200 chars after metadata section
                lines = content.split("\n")
                story_lines = []
                in_story = False
                for line in lines:
                    if in_story:
                        story_lines.append(line)
                    elif line and not line.startswith("#") and not line.startswith("**"):
                        in_story = True
                        story_lines.append(line)

                if story_lines:
                    preview_text = " ".join(story_lines)
                    metadata["preview"] = preview_text[:200]

        except OSError as e:
            # Return basic metadata if file can't be read
            metadata["error"] = str(e)

        return metadata

    def load_context_for_extension(self, filepath: Path) -> tuple[str, dict[str, Any]]:
        """
        Load context file for story extension.

        Args:
            filepath: Path to context file

        Returns:
            Tuple of (story_content, original_parameters)
        """
        metadata = self.parse_context_metadata(filepath)

        # Load full story content
        try:
            with open(filepath, encoding="utf-8") as f:
                full_content = f.read()
        except OSError:
            full_content = ""

        return full_content, metadata

    def get_story_chain(self, filepath: Path) -> list[dict[str, Any]]:
        """
        Reconstruct the full story chain by tracing back parent references.

        This method follows the chain of extended stories backward from the given
        file to the original story, returning metadata for each story in the chain
        in chronological order (oldest to newest).

        Args:
            filepath: Path to the final story in the chain

        Returns:
            List of story metadata dicts in chronological order (oldest to newest).
            Each dict contains: filename, timestamp, prompt, characters, setting, etc.

        Example:
            >>> chain = context_mgr.get_story_chain(Path("story_extended_extended.md"))
            >>> for story in chain:
            ...     print(f"{story['filename']}: {story.get('prompt', 'No prompt')}")
            story_original.md: A knight on a quest
            story_extended.md: A knight on a quest (continued)
            story_extended_extended.md: A knight on a quest (final chapter)
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
                    # Look for exact filename match
                    parent_file = context_dir / f"{parent_name}.md"
                    if parent_file.exists():
                        current_file = parent_file
                    else:
                        # Parent not found, stop here
                        break
                else:
                    # No parent reference, reached the original story
                    break
            except Exception:
                logging.getLogger(__name__).debug("Error reading story chain file, stopping", exc_info=True)
                break

        return chain

    def write_chain_to_file(self, filepath: Path, output_path: Path) -> Path:
        """
        Write the complete story chain to a single file.

        Reconstructs the full story chain and combines all stories into a single
        output file, preserving the chronological order and adding separators
        between each story segment.

        Args:
            filepath: Path to the final story in the chain
            output_path: Path where the combined story should be written

        Returns:
            Path to the created file

        Example:
            >>> output = context_mgr.write_chain_to_file(
            ...     Path("story_extended.md"),
            ...     Path("complete_story.txt")
            ... )
            >>> print(f"Complete story written to: {output}")
            Complete story written to: complete_story.txt
        """
        from datetime import datetime

        chain = self.get_story_chain(filepath)

        if not chain or ("error" in chain[0] if chain else False):
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
            if "characters" in first_story:
                combined_content.append(f"Characters: {first_story['characters']}")
            if "setting" in first_story:
                combined_content.append(f"Setting: {first_story['setting']}")

        combined_content.append("=" * 80 + "\n")

        # Add each story in the chain
        for idx, story_meta in enumerate(chain, 1):
            combined_content.append(f"\n{'=' * 80}")
            combined_content.append(f"PART {idx} of {len(chain)}")
            combined_content.append(f"{'=' * 80}")
            combined_content.append(f"Generated: {story_meta.get('timestamp', 'Unknown')}")
            combined_content.append(f"Source: {story_meta.get('filename', 'Unknown')}")
            if idx > 1 and "extended_from" in story_meta:
                combined_content.append(f"Extended from: {story_meta['extended_from']}")
            combined_content.append(f"{'=' * 80}\n")

            # Read and add the actual story content
            story_path = story_meta.get("filepath")
            if story_path and Path(story_path).exists():
                with open(story_path, encoding="utf-8") as f:
                    content = f.read()
                    # Extract just the story content (skip metadata header)
                    story_start = content.find("---\n## Story Preview")
                    if story_start != -1:
                        story_content = content[story_start + len("---\n## Story Preview\n\n") :]
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
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_content))

        return output_path


def get_default_context_manager() -> ContextManager:
    """
    Get a default context manager instance.

    Returns:
        ContextManager: Configured with default settings
    """
    return ContextManager()
