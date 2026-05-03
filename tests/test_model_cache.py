"""Tests for ModelCache disk-based caching."""

import json
from datetime import UTC, datetime

import pytest

from storyforge.model_cache import DEFAULT_TTL_SECONDS, ModelCache


@pytest.fixture
def cache(tmp_path):
    """Create a ModelCache using a temporary directory."""
    return ModelCache(cache_dir=tmp_path / "model_cache")


@pytest.fixture
def sample_models():
    """Sample model data for testing."""
    return [
        {"name": "gpt-4o", "owned_by": "openai", "created": 1700000000},
        {"name": "gpt-5.2", "owned_by": "openai", "created": 1710000000},
    ]


class TestModelCache:
    """Tests for ModelCache disk-based caching."""

    def test_cache_set_and_get(self, cache, sample_models):
        """Set models, get them back."""
        cache.set("openai", sample_models)
        result = cache.get("openai")
        assert result == sample_models

    def test_cache_miss_returns_none(self, cache):
        """Get on nonexistent backend returns None."""
        assert cache.get("nonexistent") is None

    def test_cache_expired_returns_none(self, cache, tmp_path):
        """Set with old timestamp, verify get returns None."""
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write a cache file with a very old timestamp
        old_data = {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "models": [{"name": "old-model"}],
        }
        path = cache_dir / "openai.json"
        path.write_text(json.dumps(old_data), encoding="utf-8")

        assert cache.get("openai") is None

    def test_cache_is_valid_true(self, cache, sample_models):
        """Fresh cache is valid."""
        cache.set("openai", sample_models)
        assert cache.is_valid("openai") is True

    def test_cache_is_valid_false_expired(self, cache, tmp_path):
        """Expired cache is not valid."""
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        old_data = {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "models": [],
        }
        path = cache_dir / "openai.json"
        path.write_text(json.dumps(old_data), encoding="utf-8")

        assert cache.is_valid("openai") is False

    def test_cache_is_valid_false_missing(self, cache):
        """Missing cache is not valid."""
        assert cache.is_valid("openai") is False

    def test_cache_invalidate(self, cache, sample_models):
        """Invalidate removes the file."""
        cache.set("openai", sample_models)
        assert cache.cache_path("openai").exists()

        cache.invalidate("openai")
        assert not cache.cache_path("openai").exists()

    def test_cache_invalidate_missing_ok(self, cache):
        """Invalidate on nonexistent file doesn't error."""
        # Should not raise
        cache.invalidate("nonexistent")

    def test_cache_clear_all(self, cache, sample_models):
        """Clears all cached backends."""
        cache.set("openai", sample_models)
        cache.set("anthropic", [{"name": "claude-3"}])
        cache.set("gemini", [{"name": "gemini-2.5-pro"}])

        cache.clear_all()

        assert cache.get("openai") is None
        assert cache.get("anthropic") is None
        assert cache.get("gemini") is None

    def test_cache_path(self, cache, tmp_path):
        """Verify path construction."""
        expected = tmp_path / "model_cache" / "openai.json"
        assert cache.cache_path("openai") == expected

    def test_cache_custom_ttl(self, tmp_path):
        """Custom TTL is respected."""
        # Create cache with 1-second TTL
        short_cache = ModelCache(cache_dir=tmp_path / "model_cache", ttl_seconds=1)
        short_cache.set("openai", [{"name": "gpt-4o"}])

        # Write directly with a timestamp 2 seconds in the past
        cache_dir = tmp_path / "model_cache"
        old_data = {
            "timestamp": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            "ttl_seconds": 1,
            "models": [{"name": "gpt-4o"}],
        }
        path = cache_dir / "openai.json"
        path.write_text(json.dumps(old_data), encoding="utf-8")

        assert short_cache.is_valid("openai") is False

    def test_cache_corrupt_json(self, cache, tmp_path):
        """Corrupt JSON file returns None gracefully."""
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        path = cache_dir / "openai.json"
        path.write_text("this is not valid json {{{", encoding="utf-8")

        assert cache.get("openai") is None

    def test_cache_atomic_write(self, cache, sample_models):
        """Verify file exists after set (basic atomicity check)."""
        cache.set("openai", sample_models)
        path = cache.cache_path("openai")
        assert path.exists()

        # Verify content is valid JSON
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["models"] == sample_models
        assert "timestamp" in data
        assert "ttl_seconds" in data

    def test_cache_creates_directory(self, tmp_path):
        """Cache dir is created on first write."""
        cache_dir = tmp_path / "deep" / "nested" / "model_cache"
        assert not cache_dir.exists()

        cache = ModelCache(cache_dir=cache_dir)
        cache.set("openai", [{"name": "gpt-4o"}])

        assert cache_dir.exists()
        assert cache.get("openai") == [{"name": "gpt-4o"}]
