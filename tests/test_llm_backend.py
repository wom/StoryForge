import pytest
from storytime.llm_backend import LLMBackend

class DummyBackend(LLMBackend):
    pass

def test_generate_story_not_implemented():
    backend = DummyBackend()
    with pytest.raises(NotImplementedError):
        backend.generate_story("prompt")

def test_generate_image_not_implemented():
    backend = DummyBackend()
    with pytest.raises(NotImplementedError):
        backend.generate_image("prompt")

def test_generate_image_name_not_implemented():
    backend = DummyBackend()
    with pytest.raises(NotImplementedError):
        backend.generate_image_name("prompt", "story")
