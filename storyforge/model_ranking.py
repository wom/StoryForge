"""Version-aware model ranking for LLM providers.

Parses model names to extract version numbers and quality tiers, then produces
a composite score so the highest-quality available model is selected automatically.
This approach is self-maintaining: when providers ship new model versions, the
ranker will prefer them without code changes.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Tier scores: higher = better quality
GEMINI_TIERS: dict[str, int] = {
    "ultra": 4,
    "pro": 3,
    "flash": 2,
    "flash-lite": 1,
    "nano": 0,
}

OPENAI_TIERS: dict[str, int] = {
    "pro": 3,
    "": 3,  # base GPT models (no tier suffix) are top-tier
    "mini": 1,
    "nano": 0,
}

ANTHROPIC_TIERS: dict[str, int] = {
    "opus": 3,
    "sonnet": 2,
    "haiku": 1,
}

# Patterns to skip unless explicitly requested
SKIP_PATTERNS = re.compile(r"(preview|experimental|exp|thinking|search)", re.IGNORECASE)

# Image-capable model patterns
IMAGE_PATTERNS = re.compile(r"(image|imagen|dall-e|flash-image)", re.IGNORECASE)


def extract_gemini_info(name: str) -> tuple[float, str] | None:
    """Extract version and tier from a Gemini model name.

    Examples:
        gemini-3.1-pro-preview -> (3.1, "pro")
        gemini-2.5-flash -> (2.5, "flash")
        gemini-pro-latest -> (0.0, "pro")  # alias, no version
    """
    # Strip "models/" prefix
    if name.startswith("models/"):
        name = name[7:]

    # Match versioned: gemini-{major}.{minor}-{tier}
    m = re.match(r"gemini-(\d+)\.(\d+)-(\w+(?:-\w+)?)", name)
    if m:
        version = float(f"{m.group(1)}.{m.group(2)}")
        tier_raw = m.group(3).split("-")[0]  # "flash-image" -> "flash"
        # Check for compound tiers like "flash-lite"
        if "flash-lite" in name:
            tier_raw = "flash-lite"
        return (version, tier_raw)

    # Match legacy/alias: gemini-{tier}-latest or gemini-{tier}
    m = re.match(r"gemini-(\w+)(?:-latest)?$", name)
    if m:
        tier_raw = m.group(1)
        if tier_raw in GEMINI_TIERS:
            # Aliases get a high synthetic version since they auto-update
            return (99.0, tier_raw)

    return None


def extract_openai_info(name: str) -> tuple[float, str] | None:
    """Extract version and tier from an OpenAI model name.

    Examples:
        gpt-5.4 -> (5.4, "")
        gpt-5.4-mini -> (5.4, "mini")
        gpt-4.1-nano -> (4.1, "nano")
        gpt-image-1.5 -> (1.5, "")  # image model
        o3 -> (3.0, "")
        o4-mini -> (4.0, "mini")
    """
    # GPT models: gpt-{major}.{minor}[-tier]
    m = re.match(r"gpt-(\d+)\.(\d+)(?:-(\w+))?", name)
    if m:
        version = float(f"{m.group(1)}.{m.group(2)}")
        tier = m.group(3) or ""
        # Filter out non-tier suffixes like "chat", "turbo"
        if tier in ("mini", "nano", "pro"):
            return (version, tier)
        return (version, "")

    # GPT image models: gpt-image-{major}.{minor}
    # These are newer-gen than DALL-E, so add generation offset (10.0)
    m = re.match(r"gpt-image-(\d+)\.(\d+)", name)
    if m:
        version = float(f"{m.group(1)}.{m.group(2)}") + 10.0
        return (version, "")

    # GPT image models: gpt-image-{major}
    m = re.match(r"gpt-image-(\d+)$", name)
    if m:
        version = float(m.group(1)) + 10.0
        return (version, "")

    # Legacy GPT: gpt-{major}[-suffix]
    m = re.match(r"gpt-(\d+)(?:-(.+))?$", name)
    if m:
        version = float(m.group(1))
        suffix = m.group(2) or ""
        if "mini" in suffix:
            return (version, "mini")
        if "nano" in suffix:
            return (version, "nano")
        return (version, "")

    # o-series: o{number}[-tier]
    m = re.match(r"o(\d+)(?:-(\w+))?$", name)
    if m:
        version = float(m.group(1))
        tier = m.group(2) or ""
        if tier in ("mini", "nano", "pro"):
            return (version, tier)
        return (version, "")

    # DALL-E: dall-e-{version}
    m = re.match(r"dall-e-(\d+)", name)
    if m:
        version = float(m.group(1))
        return (version, "")

    return None


def extract_anthropic_info(name: str) -> tuple[float, str] | None:
    """Extract version and tier from an Anthropic model name.

    Examples:
        claude-opus-4-7 -> (4.7, "opus")
        claude-sonnet-4-6 -> (4.6, "sonnet")
        claude-haiku-4-5 -> (4.5, "haiku")
        claude-3-5-sonnet-20241022 -> (3.5, "sonnet")  # legacy format
    """
    # New format: claude-{tier}-{major}-{minor}
    m = re.match(r"claude-(\w+)-(\d+)-(\d+)", name)
    if m:
        tier = m.group(1)
        if tier in ANTHROPIC_TIERS:
            version = float(f"{m.group(2)}.{m.group(3)}")
            return (version, tier)

    # Legacy format: claude-{major}-{minor}-{tier}[-date]
    m = re.match(r"claude-(\d+)-(\d+)-(\w+)", name)
    if m:
        tier = m.group(3).split("-")[0]  # strip date suffix
        if tier in ANTHROPIC_TIERS:
            version = float(f"{m.group(1)}.{m.group(2)}")
            return (version, tier)

    # Versionless: claude-{tier} (aliases)
    m = re.match(r"claude-(\w+)$", name)
    if m:
        tier = m.group(1)
        if tier in ANTHROPIC_TIERS:
            return (99.0, tier)

    return None


def _get_tier_score(tier: str, provider: str) -> int:
    """Get numeric tier score for a provider's tier name."""
    if provider == "gemini":
        return GEMINI_TIERS.get(tier, 1)
    elif provider == "openai":
        return OPENAI_TIERS.get(tier, 2)
    elif provider == "anthropic":
        return ANTHROPIC_TIERS.get(tier, 1)
    return 1


def _should_skip(name: str, purpose: str, provider: str) -> bool:
    """Determine if a model should be skipped based on purpose and filters."""
    name_lower = name.lower()

    # Skip preview/experimental for stability
    if SKIP_PATTERNS.search(name):
        return True

    if purpose == "text":
        # Skip image-specific models for text generation
        if IMAGE_PATTERNS.search(name) and "flash" not in name_lower:
            return True
        # Skip small models for text (prefer quality)
        if provider == "gemini" and ("nano" in name_lower or "flash-lite" in name_lower):
            return True
        if provider == "openai" and ("nano" in name_lower or "mini" in name_lower):
            return True
        if provider == "anthropic" and "haiku" in name_lower:
            return True

    elif purpose == "image":
        # For image purpose, only keep image-capable models
        if provider == "openai":
            if not IMAGE_PATTERNS.search(name):
                return True
        if provider == "gemini":
            # Gemini image models typically have "image" or "flash" in name
            if not ("image" in name_lower or "flash" in name_lower):
                return True

    return False


def score_model(name: str, provider: str) -> float:
    """Compute a composite score for a model. Higher = better.

    Score = version * 10 + tier_score

    This ensures version dominates (gpt-5.4 > gpt-4.1 regardless of tier)
    while tier breaks ties within the same version (opus > sonnet).
    """
    if provider == "gemini":
        info = extract_gemini_info(name)
    elif provider == "openai":
        info = extract_openai_info(name)
    elif provider == "anthropic":
        info = extract_anthropic_info(name)
    else:
        return 0.0

    if info is None:
        return 0.0

    version, tier = info
    tier_score = _get_tier_score(tier, provider)
    return version * 10 + tier_score


def rank_models(
    models: list[dict[str, Any]],
    provider: str,
    purpose: str = "text",
    blocklist: list[str] | None = None,
) -> str | None:
    """Rank available models and return the best one for the given purpose.

    Args:
        models: List of model dicts (must have "name" key).
        provider: One of "gemini", "openai", "anthropic".
        purpose: "text" or "image" — applies appropriate filters.
        blocklist: Optional list of model names to skip.

    Returns:
        The best model name, or None if no suitable model found.
    """
    if not models:
        return None

    blocked = set(blocklist or [])
    scored: list[tuple[float, str]] = []

    for model in models:
        name: str = model.get("name", "")
        if not name:
            continue

        # Strip models/ prefix for comparison
        clean_name = name[7:] if name.startswith("models/") else name

        if clean_name in blocked:
            continue

        if _should_skip(clean_name, purpose, provider):
            logger.debug("Skipping model %s (filtered for purpose=%s)", clean_name, purpose)
            continue

        s = score_model(clean_name, provider)
        if s > 0:
            scored.append((s, clean_name))

    if not scored:
        return None

    # Sort by score descending, then by name for determinism
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0][1]
    logger.debug(
        "Ranked %d models for %s/%s, best: %s (score=%.1f)",
        len(scored),
        provider,
        purpose,
        best,
        scored[0][0],
    )
    return best
