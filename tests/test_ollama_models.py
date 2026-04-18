import io
import json
from unittest.mock import MagicMock

import main


def _fake_response(payload):
    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return resp


def test_list_ollama_models_parses_model_field(mocker):
    payload = {
        "models": [
            {"model": "llama3.2-vision:latest", "size": 1},
            {"model": "qwen2.5-vl:7b", "size": 2},
        ]
    }
    mocker.patch(
        "urllib.request.urlopen",
        return_value=_fake_response(payload),
    )
    result = main.list_ollama_models("http://localhost:11434")
    assert result == ["llama3.2-vision:latest", "qwen2.5-vl:7b"]


def test_list_ollama_models_parses_name_field(mocker):
    payload = {"models": [{"name": "llava:13b"}]}
    mocker.patch(
        "urllib.request.urlopen",
        return_value=_fake_response(payload),
    )
    assert main.list_ollama_models("http://localhost:11434") == ["llava:13b"]


def test_list_ollama_models_returns_empty_on_failure(mocker):
    mocker.patch("urllib.request.urlopen", side_effect=ConnectionRefusedError())
    assert main.list_ollama_models("http://localhost:11434") == []


def test_list_ollama_models_returns_empty_on_timeout(mocker):
    import socket
    mocker.patch("urllib.request.urlopen", side_effect=socket.timeout())
    assert main.list_ollama_models("http://localhost:11434") == []


def test_list_ollama_models_deduplicates(mocker):
    payload = {
        "models": [
            {"model": "llama3.2-vision"},
            {"name": "llama3.2-vision"},
            {"model": "llava"},
        ]
    }
    mocker.patch(
        "urllib.request.urlopen",
        return_value=_fake_response(payload),
    )
    assert main.list_ollama_models("http://localhost:11434") == ["llama3.2-vision", "llava"]


def test_vision_filter_uses_capabilities(mocker):
    mocker.patch("main.list_ollama_models", return_value=["llava", "mistral", "llama3.2-vision"])

    def fake_caps(url, name, timeout=3.0):
        return {
            "llava": ["completion", "vision"],
            "mistral": ["completion"],
            "llama3.2-vision": ["completion", "vision"],
        }[name]

    mocker.patch("main.get_ollama_model_capabilities", side_effect=fake_caps)
    vision, other = main.list_ollama_vision_models("http://localhost:11434")
    assert vision == ["llava", "llama3.2-vision"]
    assert other == ["mistral"]


def test_vision_filter_falls_back_to_name_heuristic(mocker):
    """Older Ollama returns no 'capabilities' field — use name hints."""
    mocker.patch("main.list_ollama_models", return_value=["llava", "mistral", "moondream"])
    mocker.patch("main.get_ollama_model_capabilities", return_value=None)
    vision, other = main.list_ollama_vision_models("http://localhost:11434")
    assert set(vision) == {"llava", "moondream"}
    assert other == ["mistral"]


def test_vision_filter_empty_when_unreachable(mocker):
    mocker.patch("main.list_ollama_models", return_value=[])
    assert main.list_ollama_vision_models("http://localhost:11434") == ([], [])


def test_list_running_models_parses_response(mocker):
    payload = {
        "models": [
            {"model": "llama3.2-vision:latest"},
            {"name": "qwen2.5-vl:7b"},
        ]
    }
    mocker.patch("urllib.request.urlopen", return_value=_fake_response(payload))
    assert main.list_running_ollama_models("http://localhost:11434") == {
        "llama3.2-vision:latest",
        "qwen2.5-vl:7b",
    }


def test_list_running_models_returns_empty_on_failure(mocker):
    mocker.patch("urllib.request.urlopen", side_effect=ConnectionRefusedError())
    assert main.list_running_ollama_models("http://localhost:11434") == set()
