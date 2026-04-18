import io
from unittest.mock import MagicMock

from PIL import Image

import main


def _make_image_array():
    """Return a small numpy array image."""
    import numpy as np
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_max_tokens_is_passed_to_openai_payload(mocker):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="a caption"))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    mocker.patch("main.OpenAI", return_value=fake_client)
    mocker.patch("main.rate_limiter.reserve")

    main.generate_description_openai(
        api_key="sk-test",
        image=_make_image_array(),
        prompt="hi",
        detail="auto",
        max_tokens=777,
    )

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 777


def test_default_max_tokens_matches_constant(mocker):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="a caption"))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    mocker.patch("main.OpenAI", return_value=fake_client)
    mocker.patch("main.rate_limiter.reserve")

    main.generate_description_openai(
        api_key="sk-test",
        image=_make_image_array(),
        prompt="hi",
        detail="auto",
        max_tokens=main.DEFAULT_MAX_TOKENS,
    )

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 300


def test_caption_is_returned_unchanged(mocker):
    long_text = "A very detailed caption. " * 30
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=long_text))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    mocker.patch("main.OpenAI", return_value=fake_client)
    mocker.patch("main.rate_limiter.reserve")

    result = main.generate_description_openai(
        api_key="sk-test",
        image=_make_image_array(),
        prompt="hi",
        detail="auto",
        max_tokens=300,
    )
    assert result == long_text
