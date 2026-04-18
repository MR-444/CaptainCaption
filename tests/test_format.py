from PIL import Image

import main


def test_format_caption_description_only():
    assert main.format_caption("", "desc", "") == "desc"


def test_format_caption_all_parts():
    assert main.format_caption("pre", "desc", "post") == "pre, desc, post"


def test_format_caption_pre_only():
    assert main.format_caption("pre", "desc", "") == "pre, desc"


def test_format_caption_post_only():
    assert main.format_caption("", "desc", "post") == "desc, post"


def test_format_caption_strips_whitespace():
    assert main.format_caption("  pre  ", "  desc  ", "  post  ") == "pre, desc, post"


def test_format_caption_none_description_is_handled():
    assert main.format_caption("", None, "") == ""
    assert main.format_caption("pre", None, "post") == "pre, post"


def test_sanitize_api_key_removes_nbsp_and_whitespace():
    assert main.sanitize_api_key("sk-abc\xa0def ") == "sk-abcdef"


def test_sanitize_api_key_empty():
    assert main.sanitize_api_key("") == ""
    assert main.sanitize_api_key(None) is None


def test_sanitize_api_key_strips_non_ascii():
    assert main.sanitize_api_key("sk-abc\u00ffdef") == "sk-abcdef"


def test_sanitize_text_replaces_nbsp():
    assert main.sanitize_text("hello\xa0world") == "hello world"


def test_sanitize_text_strips():
    assert main.sanitize_text("  hello  ") == "hello"


def test_sanitize_text_empty():
    assert main.sanitize_text("") == ""
    assert main.sanitize_text(None) is None


def test_scale_image_downscales_wide():
    img = Image.new("RGB", (4096, 2048), "white")
    scaled = main.scale_image(img)
    assert scaled.width == main.MAX_IMAGE_WIDTH
    assert scaled.height == main.MAX_IMAGE_WIDTH // 2


def test_scale_image_leaves_small_unchanged():
    img = Image.new("RGB", (512, 512), "white")
    scaled = main.scale_image(img)
    assert scaled.size == (512, 512)


def test_scale_image_at_max_width_unchanged():
    img = Image.new("RGB", (main.MAX_IMAGE_WIDTH, 800), "white")
    scaled = main.scale_image(img)
    assert scaled.size == (main.MAX_IMAGE_WIDTH, 800)
