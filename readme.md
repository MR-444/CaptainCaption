# CaptainCaption: GPT-4 Vision + Ollama Image Captioner

A Gradio-based image captioning tool that supports both **OpenAI's GPT-4 Vision API** (cloud) and **Ollama** (local) for generating detailed descriptions of images.

## About This Repository

This repository is a fork and continuation of the CaptainCaption project originally created by **Lux42**. I have cloned the repo from Lux42 and worked together with him. Some of my pull requests are included, but not in the ONZU fork. When he made it private or deleted it, my fork was switched to https://github.com/ONZU/CaptainCaption.

This version includes significant improvements, refactoring, and new features including **local Ollama support** for completely free, offline image captioning.

### Repository History

```
Lux42/CaptainCaption (original - now private/deleted)
    ↓ (collaborated with Lux42)
MR-444/CaptainCaption (this repository)
    ↓ (when original became unavailable)
ONZU/CaptainCaption (forked from this repo)
```

### What's New in This Version

- 🆕 **Ollama Support**: Run vision models locally - completely free and offline!
- ✨ **Multiple Model Support**: Choose between OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo) or Ollama (llava, llama3.2-vision, etc.)
- 🔧 **Improved Rate Limiter**: Sliding window implementation (OpenAI only)
- 🛡️ **Enhanced Error Handling**: Critical error detection that stops batch processing
- 🧹 **API Key Sanitization**: Automatic cleaning of API keys
- 🛑 **Cancellable Processing**: Stop batch processing at any time
- 🔒 **Thread-Safe Operations**: Proper locking mechanisms
- 📊 **Detailed Feedback**: Comprehensive batch processing summaries
- 🎨 **Better UI**: Provider selection with dynamic interface

## Features

- 🖼️ **Dual Provider Support**: Choose between cloud (OpenAI) or local (Ollama)
- 🆓 **Free Local Processing**: Use Ollama for unlimited free captions
- 📁 **Batch Processing**: Process entire folders with concurrent workers
- ⚡ **Smart Rate Limiting**: Prevents API quota exhaustion (OpenAI only)
- 🔄 **Intelligent Skipping**: Automatically skips images with existing captions
- 🎯 **Flexible Configuration**: Adjust detail level, token limits, and workers
- 🌐 **No Internet Required**: Ollama works completely offline

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/MR-444/CaptainCaption
cd CaptainCaption
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Provider Setup

#### Option A: OpenAI (Cloud - Paid)

Get your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)

#### Option B: Ollama (Local - Free)

1. **Install Ollama:**
   - Visit [ollama.com](https://ollama.com) and download for your OS
   - Or use: `curl -fsSL https://ollama.com/install.sh | sh` (Linux/macOS)

2. **Pull a Vision Model:**
   ```bash
   # Recommended for most users (4.7GB)
   ollama pull llava
   
   # Or use Meta's latest model (7.8GB)
   ollama pull llama3.2-vision
   ```

3. **Start Ollama Server:**
   ```bash
   ollama serve
   ```

## Usage

```bash
python main.py
```

Navigate to `http://127.0.0.1:7860`

## Provider Comparison

| Feature | OpenAI | Ollama |
|---------|--------|--------|
| **Cost** | ~$0.0015-$0.0085 per image | Free |
| **Quality** | Excellent | Good to Very Good |
| **Speed** | Fast (cloud) | Depends on hardware |
| **Internet** | Required | Not required |
| **Privacy** | Images sent to cloud | 100% local |

## Model Recommendations

### OpenAI
- **gpt-4o-mini**: Best value for large batches
- **gpt-4o**: Highest quality

### Ollama
- **llava**: Best balance (4.7GB)
- **llama3.2-vision**: Latest from Meta (7.8GB)
- **llava:13b**: Better quality (8GB)

## Quick Start Guide

### For Free Local Captions (Ollama)

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull llava`
3. Start server: `ollama serve`
4. Run app: `python main.py`
5. Select "Ollama" provider
6. Start captioning!

### For Cloud Captions (OpenAI)

1. Get API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Run app: `python main.py`
3. Select "OpenAI" provider
4. Enter API key
5. Start captioning!

## Troubleshooting

### Ollama Issues

**"Cannot connect to Ollama":**
```bash
# Start Ollama server
ollama serve
```

**"Model not found":**
```bash
# Pull the model
ollama pull llava
```

**Slow processing:**
- Use smaller model: `ollama pull llava:7b`
- Reduce concurrent workers

### OpenAI Issues

**Rate limit errors:**
- Reduce workers to 2-3
- Application automatically stops to prevent quota waste

**API key errors:**
- Re-copy key directly from OpenAI
- Verify account has credits

## Cost Estimation

### OpenAI (per 1000 images)
- **gpt-4o-mini**: $2-4
- **gpt-4o**: $3-8

### Ollama
- **Cost: $0** (completely free!)
- **Requirements**: 8GB RAM minimum, GPU recommended

## Hardware Requirements for Ollama

**Minimum:**
- 8GB RAM
- 5GB disk space

**Recommended:**
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 10GB disk space

**Optimal:**
- 32GB RAM
- NVIDIA GPU with 24GB VRAM
- 50GB disk space

## Best Practices

1. **Start with Ollama**: Test for free before paying
2. **Use llava**: Good balance of quality/speed
3. **Start with 2 Workers**: Increase based on hardware
4. **Backup Data**: Keep original images safe
5. **Test Prompts**: Use single image mode first

## Prompting Tips

### General Purpose
```
Describe this image in detail. Focus on the main subjects, actions, setting, and mood.
```

### For Training Data
```
Describe this image including: subject, lighting, composition, style, colors, and mood.
```

### Using Prefix/Postfix
- **Prefix**: "masterpiece, high quality, detailed"
- **Postfix**: "digital art, trending on artstation"

## Credits

- **Original Creator**: Lux42
- **This Repository**: [MR-444](https://github.com/MR-444)
- **Ollama**: [Ollama Team](https://ollama.com)
- **OpenAI**: GPT-4 Vision API

## License

MIT License - See LICENSE file for details.

## Support

- **OpenAI API**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Ollama**: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- **Application**: Open a GitHub issue

## FAQ

**Q: Which provider should I use?**
A: Start with Ollama (free) for testing. Use OpenAI for production quality.

**Q: Is Ollama really free?**
A: Yes! You just need hardware to run it locally.

**Q: Can I use both providers?**
A: Yes! Switch between them anytime in the UI.

**Q: Do I need GPU for Ollama?**
A: No, but it's much faster with GPU. CPU-only works.

---

**Happy Captioning! 🎨 🆓**
