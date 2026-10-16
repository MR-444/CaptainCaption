# CaptainCaption 🖼️

Gradio UI for image captioning with **OpenAI** (cloud) or **Ollama** (local). Single images or batch folders.

## ✨ Features
- **Dual providers** — swap between OpenAI and a local Ollama instance from one UI.
- **Live model discovery** — the Ollama dropdown auto-populates from your `/api/tags`, filtered to **vision-capable** models, with `●` = already loaded / `○` = needs load.
- **Batch processing** with live progress bar and concurrent workers.
- **Smart skipping** — existing `.txt` caption files aren't re-processed, and failed calls never overwrite good captions.
- **Configurable timeouts** — set an Ollama timeout (covers model load + inference) so slow cold starts fail cleanly instead of hanging.
- **Rate-limited** OpenAI requests with atomic slot reservation to stay inside your quota.

## 🚀 Quick Start
```bash
git clone https://github.com/MR-444/CaptainCaption
cd CaptainCaption
pip install -r requirements.txt
python main.py      # open http://127.0.0.1:7860
```

### Provider setup
- **Ollama**: install [Ollama](https://ollama.com/), then `ollama pull llama3.2-vision` (or any vision model). Make sure `ollama serve` is running.
- **OpenAI**: grab a key at [platform.openai.com](https://platform.openai.com/api-keys); billing must be enabled.

## 🧠 Suggested vision models
| Provider | Model | Notes |
|---|---|---|
| Ollama | `qwen3-vl` (32B) | Best for OCR / fine detail. |
| Ollama | `moondream` (1.8B) | Tiny + fast; low-VRAM option. |
| OpenAI | `gpt-4o-mini` | Best $/quality for bulk. |
| OpenAI | `gpt-4o` | Highest quality. |

## 🛠️ Tips
- **Ollama not reachable** → start it (`ollama serve`), then click 🔄 in the UI.
- **OpenAI 429** → drop the worker slider to 1–2; the batch auto-stops on rate limits.
- **Caption length** — adjust via your prompt. Max-tokens just hard-caps OpenAI output. Ollama has no output cap; ask for "one sentence" / "detailed paragraph" in the prompt.
- **Prefix / Postfix** (Advanced accordion) wrap every caption, e.g. `masterpiece` / `highly detailed`.

## 🧪 Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## 📄 Credits
Originally by Lux42, now maintained by MR-444 / ONZU. MIT licensed.
