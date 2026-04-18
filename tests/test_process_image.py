import os

import pytest
from PIL import Image

import main


@pytest.fixture
def tmp_image(tmp_path):
    img_path = tmp_path / "pic.jpg"
    Image.new("RGB", (64, 64), "white").save(img_path)
    return img_path


@pytest.fixture
def control():
    return main.ProcessingControl()


def test_skipped_when_caption_exists(tmp_image, control, mocker):
    txt = tmp_image.with_suffix(".txt")
    txt.write_text("existing")

    mocked = mocker.patch("main.generate_description")
    success, status, desc = main.process_image(
        control, "OpenAI", "sk-x", "", str(tmp_image),
        "prompt", "auto", 300, "gpt-4o-mini", "", ""
    )
    assert (success, status) == (True, "SKIPPED")
    mocked.assert_not_called()


def test_error_string_does_not_write_caption_file(tmp_image, control, mocker):
    txt = tmp_image.with_suffix(".txt")
    assert not txt.exists()

    mocker.patch("main.generate_description", return_value="Error: rate limit exceeded")
    success, status, _ = main.process_image(
        control, "OpenAI", "sk-x", "", str(tmp_image),
        "prompt", "auto", 300, "gpt-4o-mini", "", ""
    )
    assert not txt.exists()
    assert success is False
    assert status == "ERROR"


def test_none_description_does_not_crash(tmp_image, control, mocker):
    mocker.patch("main.generate_description", return_value=None)
    success, status, _ = main.process_image(
        control, "OpenAI", "sk-x", "", str(tmp_image),
        "prompt", "auto", 300, "gpt-4o-mini", "", ""
    )
    assert success is False
    assert status == "ERROR"
    assert not tmp_image.with_suffix(".txt").exists()


def test_successful_caption_is_written(tmp_image, control, mocker):
    mocker.patch("main.generate_description", return_value="a nice picture")
    success, status, desc = main.process_image(
        control, "OpenAI", "sk-x", "", str(tmp_image),
        "prompt", "auto", 300, "gpt-4o-mini", "pre", "post"
    )
    assert (success, status) == (True, "SUCCESS")
    txt = tmp_image.with_suffix(".txt")
    assert txt.exists()
    assert txt.read_text(encoding="utf-8") == "pre, a nice picture, post"


def test_stopped_control_short_circuits(tmp_image, control, mocker):
    control.start()
    control.stop()
    mocked = mocker.patch("main.generate_description")
    success, status, _ = main.process_image(
        control, "OpenAI", "sk-x", "", str(tmp_image),
        "prompt", "auto", 300, "gpt-4o-mini", "", ""
    )
    assert (success, status) == (False, "STOPPED")
    mocked.assert_not_called()
