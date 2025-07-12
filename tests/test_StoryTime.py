import pytest
from StoryTime import StoryApp

class DummyResponse:
    def __init__(self, text):
        self.candidates = [
            type('C', (), {"content": type('D', (), {"parts": [type('P', (), {"text": text, "inline_data": None})]})()})
        ]

class DummyImageResponse:
    def __init__(self, image_bytes):
        self.candidates = [
            type('C', (), {"content": type('D', (), {"parts": [type('P', (), {"text": '', "inline_data": type('I', (), {"data": image_bytes})()})]})()})
        ]

def test_generate_story(monkeypatch):
    app = StoryApp()
    dummy_text = "Once upon a time..."
    monkeypatch.setattr(app, 'generate_story', lambda prompt: dummy_text)
    assert app.generate_story("prompt") == dummy_text

def test_generate_image_name(monkeypatch):
    app = StoryApp()
    dummy_name = "my_story_image"
    monkeypatch.setattr(app, 'generate_image_name', lambda prompt, story: dummy_name)
    assert app.generate_image_name("prompt", "story") == dummy_name

def test_generate_story_error(monkeypatch):
    app = StoryApp()
    def raise_error(prompt):
        raise Exception("fail")
    monkeypatch.setattr(app, 'generate_story', raise_error)
    with pytest.raises(Exception):
        app.generate_story("prompt")

def test_generate_image_name_error(monkeypatch):
    app = StoryApp()
    def raise_error(prompt, story):
        raise Exception("fail")
    monkeypatch.setattr(app, 'generate_image_name', raise_error)
    with pytest.raises(Exception):
        app.generate_image_name("prompt", "story")
