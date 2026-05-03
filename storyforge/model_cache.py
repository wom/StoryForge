"""Disk-based model cache for LLM provider model discovery results.

Caches discovered model lists to minimize API calls. Models are refreshed
when the cache expires (default: 7 days) or when a model-not-found error
triggers invalidation.
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 604800  # 7 days
DEFAULT_CACHE_DIR = Path.home() / ".local" / "share" / "storyforge" / "model_cache"


class ModelCache:
    """Disk-based cache for provider model lists.

    Stores JSON files per backend with timestamps and TTL metadata.
    Uses atomic writes (temp file + os.replace) for safety.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._cache_dir = cache_dir
        self._ttl_seconds = ttl_seconds

    def cache_path(self, backend_name: str) -> Path:
        """Return the path for a backend's cache file."""
        return self._cache_dir / f"{backend_name}.json"

    def get(self, backend_name: str) -> list[dict[str, Any]] | None:
        """Return cached models if cache is valid, None if expired or missing."""
        if not self.is_valid(backend_name):
            return None

        path = self.cache_path(backend_name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            models: list[dict[str, Any]] = data["models"]
            return models
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to read model cache for %s: %s", backend_name, e)
            return None

    def set(self, backend_name: str, models: list[dict[str, Any]]) -> None:
        """Write models to cache with current timestamp and TTL metadata."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ttl_seconds": self._ttl_seconds,
            "models": models,
        }

        path = self.cache_path(backend_name)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Failed to write model cache for %s: %s", backend_name, e)

    def is_valid(self, backend_name: str) -> bool:
        """Check if cache exists and hasn't expired."""
        path = self.cache_path(backend_name)
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            timestamp_str = data["timestamp"]
            cached_time = datetime.fromisoformat(timestamp_str)
            ttl = data.get("ttl_seconds", self._ttl_seconds)
            age = (datetime.now(UTC) - cached_time).total_seconds()
            return bool(age < ttl)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to validate model cache for %s: %s", backend_name, e)
            return False

    def invalidate(self, backend_name: str) -> None:
        """Delete the cache file to force refresh on next access."""
        path = self.cache_path(backend_name)
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to invalidate model cache for %s: %s", backend_name, e)

    def clear_all(self) -> None:
        """Remove all cache files from the cache directory."""
        if not self._cache_dir.exists():
            return

        try:
            for entry in self._cache_dir.iterdir():
                if entry.is_file() and entry.suffix == ".json":
                    entry.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clear model cache directory: %s", e)
