# CaptainCaption (Improved & Maintained Fork)

A Gradio-based image captioning tool that uses the GPT-4 Vision API to generate detailed descriptions of images.

## About This Fork

This is an improved and actively maintained fork of CaptainCaption. The original repository by **42lux** is no longer available. This fork was initially based on [ONZU/CaptainCaption](https://github.com/ONZU/CaptainCaption) (which itself was forked from the original 42lux repository in December 2023).

This version includes significant refactoring, bug fixes, and new features while maintaining the core functionality and spirit of the original project.

### What's New in This Fork

- ✨ **Updated to GPT-4o**: Uses the latest OpenAI vision model (original used deprecated `gpt-4-vision-preview`)
- 🔧 **Improved Rate Limiter**: Sliding window implementation that won't hang indefinitely
- 🛡️ **Thread-Safe**: Proper locking mechanisms for concurrent operations
- 📊 **Better Feedback**: Detailed batch processing summaries with counts
- 🎨 **Enhanced UI**: Better organized interface with improved tooltips and controls
- 🐛 **Bug Fixes**: Fixed caption formatting, error handling, and edge cases
- 📝 **Better Documentation**: Comprehensive guides and changelog

### Fork History

```
42lux/CaptainCaption (original, no longer available)
    ↓
ONZU/CaptainCaption (fork from 42lux, Dec 2023)
    ↓
MR-444/CaptainCaption (this repository - improved & maintained)
```

## Comparison with Earlier Versions

| Feature | Earlier Versions | This Fork |
|---------|-----------------|-----------|
| OpenAI Model | gpt-4-vision-preview (deprecated) | gpt-4o (current) |
| Rate Limiting | Basic counter, could hang | Sliding window, smart waiting |
| Thread Safety | Race conditions possible | Fully thread-safe |
| Error Handling | Basic logging | Detailed logs with timestamps |
| Batch Summary | Generic message | Detailed counts (processed/skipped/errors) |
| Caption Formatting | Could add extra commas/spaces | Clean formatting function |
| UI | Basic interface | Enhanced with better organization |
| Max Workers | 4 | 8 (configurable) |
| Progress Feedback | Basic | Detailed with descriptions |

## Features

- 🖼️ **Prompt Engineering**: Customize the prompt for image description to get the most accurate and relevant captions
- 📁 **Batch Processing**: Process entire folders of images with customizable pre and post prompts
- ⚡ **Smart Rate Limiting**: Built-in rate limiter to prevent API quota exhaustion
- 🔄 **Intelligent Skipping**: Automatically skips images that already have captions
- 📊 **History Tracking**: Keep track of all captions generated in a session
- 🎯 **Flexible Configuration**: Adjust detail level, token limits, and concurrent workers

## Installation

1. Clone this repository:

```bash
git clone https://github.com/MR-444/CaptainCaption
cd CaptainCaption
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)

## Usage

### Running the Application

```bash
python main.py
```

Navigate to the provided URL (typically `http://127.0.0.1:7860`) to access the interface.

### Quick Start

1. **Setting Up API Key**: Enter your OpenAI API key in the provided textbox
2. **Single Image Mode** (Prompt Engineering tab):
   - Upload an image
   - Customize the prompt, detail level, and max tokens
   - Click "Generate Caption" to receive the image description
3. **Batch Processing** (Batch Processing tab):
   - Set the folder path containing your images
   - Customize prompt details and optional prefix/postfix
   - Set the number of concurrent workers
   - Click "Start Batch Processing"

**Note:** The application creates `.txt` files next to each image with the same filename. If a `.txt` file already exists, that image is skipped.

## Configuration

### Rate Limiting

By default, the rate limiter is set to **10 requests per 60 seconds**. You can adjust this in `main.py`:

```python
rate_limiter = RateLimiter(max_calls=10, period=60)
```

### Image Processing

- **Max Image Width:** 2048px (larger images are scaled down)
- **Image Format:** JPEG (for API upload)
- **Supported Input Formats:** PNG, JPG, JPEG

### API Model

The application uses `gpt-4o` model. You can change this in the `generate_description()` function if needed.

## File Structure

```
.
├── main.py              # Main application with Gradio UI
├── rate_limiter.py      # Rate limiting implementation
├── requirements.txt     # Python dependencies
├── CHANGELOG.md         # Detailed changes and improvements
└── error_log.txt        # Generated error log (if errors occur)
```

## Prompting Tips

### Good Prompts for Training Data:
```
Describe this image in detail. Focus on the main subjects, actions, setting, and mood.
```

### For Specific Use Cases:
```
# Photography
Describe this photo including: subject, lighting, composition, and mood.

# Art/Illustration
Describe this artwork including: style, medium, subject, colors, and composition.

# Product Images
Describe this product image including: item type, features, condition, and context.
```

### Using Prefix/Postfix:
- **Prefix**: Add tags like "masterpiece, high quality, detailed"
- **Postfix**: Add style tags like "digital art, trending on artstation"

## Troubleshooting

### Rate Limit Errors
- Reduce the number of concurrent workers
- The rate limiter will automatically pause when limits are reached

### API Key Issues
- Ensure your API key starts with `sk-`
- Check that your OpenAI account has credits
- Verify the key has API access enabled

### Folder Selection Not Working
- Manually copy and paste the folder path
- Ensure you have read/write permissions for the folder

### Images Not Processing
- Check `error_log.txt` for detailed error messages
- Verify images are in supported formats (PNG, JPG, JPEG)
- Ensure images aren't corrupted

## Cost Estimation

GPT-4 Vision API pricing (as of 2024):
- **Low detail:** ~$0.00255 per image
- **High detail:** ~$0.0085 per image (depending on size)
- **Auto detail:** API decides based on image

For 1000 images with auto detail, expect approximately **$3-8** in costs.

## Best Practices

1. **Test First**: Try a few images in single mode before batch processing
2. **Use Auto Detail**: Let the API optimize cost vs quality
3. **Start with 2 Workers**: Increase only if no rate limit issues
4. **Backup Your Data**: Keep original images safe
5. **Monitor Costs**: Check your OpenAI usage dashboard regularly

## Limitations and Considerations

- The accuracy of captions depends on the quality of the uploaded images and the clarity of the provided prompts
- The OpenAI API is rate-limited; the built-in rate limiter helps manage this, but large batches may still take time
- Internet connectivity is required for API communication
- API costs apply per image processed (see Cost Estimation section)

## Advanced Usage

### Custom Error Handling

Errors are automatically logged to `error_log.txt` with timestamps. Check this file if processing fails.

### Canceling Batch Processing

Click the "Cancel" button to stop batch processing at any time. Already processed images will keep their captions.

### Re-running Batch Processing

The application automatically skips images that already have `.txt` files. To reprocess:
1. Delete the existing `.txt` files
2. Run batch processing again

## Credits

- **Original Project**: CaptainCaption by **42lux** (repository no longer available)
- **Intermediate Fork**: [ONZU/CaptainCaption](https://github.com/ONZU/CaptainCaption) (Jung von Matt CREATORS)
- **This Fork**: Maintained and improved by [MR-444](https://github.com/MR-444)

Special thanks to the original creator 42lux for building the foundation of this tool, and to ONZU for maintaining a fork when the original became unavailable.

## License

This project is provided as-is for personal and commercial use.

## Contributing

Feel free to submit issues or pull requests for improvements.

## Support

For OpenAI API issues, visit [platform.openai.com/docs](https://platform.openai.com/docs)

For application issues, check the error log or open an issue.
