import pytest

from storyforge.StoryTime import StoryApp


class DummyResponse:
    def __init__(self, text):
        self.candidates = [
            type(
                "C",
                (),
                {
                    "content": type(
                        "D",
                        (),
                        {"parts": [type("P", (), {"text": text, "inline_data": None})]},
                    )()
                },
            )
        ]


class DummyImageResponse:
    def __init__(self, image_bytes):
        self.candidates = [
            type(
                "C",
                (),
                {
                    "content": type(
                        "D",
                        (),
                        {
                            "parts": [
                                type(
                                    "P",
                                    (),
                                    {
                                        "text": "",
                                        "inline_data": type(
                                            "I", (), {"data": image_bytes}
                                        )(),
                                    },
                                )
                            ]
                        },
                    )()
                },
            )
        ]


def test_generate_story(monkeypatch):
    app = StoryApp()
    dummy_text = "Once upon a time..."
    monkeypatch.setattr(
        app.backend, "generate_story", lambda prompt, context=None: dummy_text
    )
    assert app.backend.generate_story("prompt") == dummy_text


def test_generate_image_name(monkeypatch):
    app = StoryApp()
    dummy_name = "my_story_image"
    monkeypatch.setattr(
        app.backend, "generate_image_name", lambda prompt, story: dummy_name
    )
    assert app.backend.generate_image_name("prompt", "story") == dummy_name


def test_generate_story_error(monkeypatch):
    app = StoryApp()

    def raise_error(prompt, context=None):
        raise ValueError("fail")

    monkeypatch.setattr(app.backend, "generate_story", raise_error)
    with pytest.raises(ValueError):
        app.backend.generate_story("prompt")


def test_generate_image_name_error(monkeypatch):
    app = StoryApp()

    def raise_error(prompt, story):
        raise ValueError("fail")

    monkeypatch.setattr(app.backend, "generate_image_name", raise_error)
    with pytest.raises(ValueError):
        app.backend.generate_image_name("prompt", "story")
