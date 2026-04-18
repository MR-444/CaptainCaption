import main


def test_default_prompt_exists():
    assert main.DEFAULT_PROMPT
    assert isinstance(main.DEFAULT_PROMPT, str)


def test_default_prompt_is_an_instruction():
    assert "describe" in main.DEFAULT_PROMPT.lower()


def test_default_prompt_mentions_expected_aspects():
    lower = main.DEFAULT_PROMPT.lower()
    expected = ["subjects", "actions", "setting", "mood"]
    matches = [word for word in expected if word in lower]
    assert matches, (
        f"DEFAULT_PROMPT should mention at least one of {expected}; "
        f"current value: {main.DEFAULT_PROMPT!r}"
    )


def test_default_prompt_reasonable_length():
    length = len(main.DEFAULT_PROMPT)
    assert 40 <= length <= 400, (
        f"DEFAULT_PROMPT length {length} outside expected 40-400 range"
    )


def test_default_max_tokens_is_reasonable():
    assert 50 <= main.DEFAULT_MAX_TOKENS <= 1000
