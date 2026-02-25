# CaptainCaption 🖼️

A Gradio-based image captioning tool that supports both **OpenAI's GPT-4 Vision API** (cloud) and **Ollama** (local) for generating detailed descriptions of images. Capable of processing single images or batch processing entire folders.

## ✨ Features
- **Dual Providers**: Choose between cloud (OpenAI) or completely free, offline local processing (Ollama).
- **Batch Processing**: Caption entire folders automatically with concurrent workers.
- **Smart Rate Limiting**: Built-in delays to prevent OpenAI quota exhaustion.
- **Intelligent Skipping**: Automatically skips images that already have a `.txt` caption file.
- **Customizable**: Adjust detail levels, token limits, system prompts, and worker threads on the fly.

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/MR-444/CaptainCaption
cd CaptainCaption
pip install -r requirements.txt
```
*(Note: This is a continuation of the original CaptainCaption project by Lux42, maintained by MR-444 / ONZU).*

### 2. Setup Provider

**Option A: Ollama (Local & Free)**
1. Install [Ollama](https://ollama.com/).
2. Pull a recommended vision model (e.g., `ollama pull llama3.2-vision`).
3. Ensure Ollama is running (`ollama serve`).

**Option B: OpenAI (Cloud)**
1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys) and ensure your account has billing credits.

### 3. Run the App
```bash
python main.py
```
Then navigate to `http://127.0.0.1:7860` in your web browser.

## 🧠 Model Recommendations

### Local (Ollama)
Local vision models have improved rapidly. Here are the best current options:
- **`llama3.2-vision`** (11B / 7.9GB): Meta's flagship open-source vision model. Excellent overall reasoning and detail. *(Recommended)*
- **`qwen2.5-vl`** (7B / 4.4GB): Exceptional capability for reading text (OCR) and identifying fine visual details.
- **`minicpm-v`** (8B / 5.5GB): Highly efficient, punches above its weight class for captioning quality.
- **`moondream`** (1.8B / 1.7GB): Extremely small and fast. Perfect for systems with limited RAM/VRAM.

To install a model, run: `ollama pull <model_name>`

### Cloud (OpenAI)
- **`gpt-4o-mini`**: Best value and highly capable for bulk processing large visual datasets.
- **`gpt-4o`**: Highest possible quality and reasoning.

## 💻 Hardware Requirements (For Ollama)
- **Minimum**: 8GB RAM (runs on CPU, but generation will be slow).
- **Recommended**: 16GB RAM + NVIDIA GPU with 6GB+ VRAM.

## 🛠️ Troubleshooting & Tips
- **"Cannot connect to Ollama":** Ensure the Ollama app is open in your system tray, or run `ollama serve` in a separate terminal.
- **OpenAI Rate Limits (Error 429):** If batch processing halts, lower your "Concurrent Workers" slider to 1 or 2.
- **Prompting:** Use the Prefix/Postfix fields in the Batch UI to automatically wrap your captions (e.g. prefix: "masterpiece, " postfix: ", highly detailed").

## 📄 License & Credits
- **Original Creator**: Lux42
- **Maintainer**: MR-444
- **License**: MIT
