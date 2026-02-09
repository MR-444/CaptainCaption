# CaptainCaption: GPT-4 Vision Image Captioner

A Gradio-based image captioning tool that uses OpenAI's GPT-4 Vision API to generate detailed descriptions of images.

## About This Repository

This repository is a fork and continuation of the CaptainCaption project originally created by **Lux42**. I have cloned the repo from Lux42 and worked together with him. Some of my pull requests are included, but not in the ONZU fork. When he made it private or deleted it, my fork was switched to https://github.com/ONZU/CaptainCaption.

This version includes significant improvements, refactoring, and new features while maintaining the core functionality of the original project.

### Repository History

```
Lux42/CaptainCaption (original - now private/deleted)
    ↓ (collaborated with Lux42)
MR-444/CaptainCaption (this repository)
    ↓ (when original became unavailable)
ONZU/CaptainCaption (forked from this repo)
```

### What's New in This Version

- ✨ **Multiple Model Support**: Choose between gpt-4o-mini (cost-effective), gpt-4o (high quality), or gpt-4-turbo
- 🔧 **Improved Rate Limiter**: Sliding window implementation with smart waiting that won't hang indefinitely
- 🛡️ **Enhanced Error Handling**: Critical error detection that stops batch processing to prevent API quota waste
- 🧹 **API Key Sanitization**: Automatic cleaning of API keys to prevent Unicode encoding errors
- 🛑 **Cancellable Processing**: Stop batch processing at any time with the cancel button
- 🔒 **Thread-Safe Operations**: Proper locking mechanisms for concurrent operations
- 📊 **Detailed Feedback**: Comprehensive batch processing summaries with success/skip/error counts
- 🎨 **Better UI**: Improved interface with model selector and enhanced tooltips
- 📝 **Smart Caption Formatting**: Clean formatting with proper comma handling
- 🚨 **Smart Error Recovery**: Distinguishes between recoverable errors and critical failures

## Features

- 🖼️ **Single Image Mode**: Test and refine prompts on individual images
- 📁 **Batch Processing**: Process entire folders of images with concurrent workers
- ⚡ **Smart Rate Limiting**: Prevents API quota exhaustion with sliding window algorithm
- 🔄 **Intelligent Skipping**: Automatically skips images that already have captions
- 📊 **History Tracking**: Keep track of all captions generated in a session
- 🎯 **Flexible Configuration**: Adjust detail level, token limits, and concurrent workers
- 🛑 **Cancel Anytime**: Stop batch processing without losing already processed captions
- 🔧 **Model Selection**: Choose the right model for your needs and budget

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

#### Single Image Mode

1. Enter your OpenAI API key
2. Select your preferred model (gpt-4o-mini recommended for most users)
3. Upload an image
4. Customize the prompt, detail level, and max tokens
5. Click "Generate Caption"

#### Batch Processing Mode

1. Enter your OpenAI API key
2. Select your preferred model
3. Set the folder path containing your images (or use the 📂 button)
4. Customize prompt details and optional prefix/postfix
5. Set the number of concurrent workers (2-3 recommended)
6. Click "Start Batch Processing"
7. Use "Cancel" button to stop at any time if needed

**Note:** The application creates `.txt` files next to each image with the same filename. If a `.txt` file already exists, that image is automatically skipped.

## Model Comparison

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| **gpt-4o-mini** | Fast | Good | Low | Large batches, training data |
| **gpt-4o** | Medium | Excellent | Medium | High-quality captions |
| **gpt-4-turbo** | Medium | Very Good | Medium | Balance of quality and cost |

**Recommendation:** Start with `gpt-4o-mini` for testing, upgrade to `gpt-4o` for production quality.

## Configuration

### Rate Limiting

The rate limiter uses a **sliding window algorithm** that tracks individual API calls and allows new calls as old ones expire, preventing indefinite hangs.

Default: **10 requests per 60 seconds**

You can adjust this in `main.py`:
```python
rate_limiter = RateLimiter(max_calls=10, period=60)
```

### Image Processing

- **Max Image Width:** 2048px (larger images are automatically scaled down)
- **Image Format:** JPEG (for API upload)
- **Supported Input Formats:** PNG, JPG, JPEG

### Concurrent Workers

- **Minimum:** 1 worker (slowest, safest)
- **Maximum:** 8 workers (fastest, may hit rate limits)
- **Recommended:** 2-3 workers (good balance)

Higher worker counts process faster but increase the risk of hitting rate limits. The rate limiter will automatically throttle requests regardless of worker count.

## File Structure

```
.
├── main.py              # Main application with Gradio UI
├── rate_limiter.py      # Sliding window rate limiting implementation
├── requirements.txt     # Python dependencies
├── readme.md           # This file
└── error_log.txt       # Generated error log (created on first error)
```

## Advanced Features

### Error Handling

The application uses a sophisticated error handling system:

- **Critical Errors** (stop processing):
  - Rate limit errors (429)
  - Connection failures
  - Authentication errors
  
- **Recoverable Errors** (log and continue):
  - Individual file errors
  - Image format issues
  - Timeout errors

All errors are logged to `error_log.txt` with timestamps for debugging.

### API Key Sanitization

The application automatically:
- Removes non-ASCII characters from API keys
- Strips all whitespace
- Detects common copy-paste issues (from PDFs, formatted documents)

### Thread Safety

The `ProcessingControl` class ensures thread-safe operations:
- Lock-protected state management
- Safe cancellation mechanism
- Prevents race conditions in concurrent processing

### Rate Limiter Details

The `RateLimiter` class uses a sliding window approach:
```python
# Features:
- Thread-safe with lock protection
- Automatic cleanup of expired calls
- Smart waiting with timeout support
- Status reporting (current calls, remaining slots)
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

# Character/Person Description
Describe the person in this image: appearance, clothing, pose, expression, and setting.
```

### Using Prefix/Postfix:
- **Prefix**: Add quality tags like "masterpiece, high quality, detailed"
- **Postfix**: Add style tags like "digital art, trending on artstation"

Example with prefix/postfix:
```
Prefix: "masterpiece, best quality"
Description: "a young woman in a red dress standing in a garden"
Postfix: "highly detailed, trending on artstation"
Result: "masterpiece, best quality, a young woman in a red dress standing in a garden, highly detailed, trending on artstation"
```

## Troubleshooting

### Rate Limit Errors
- **Solution 1**: Reduce concurrent workers to 1-2
- **Solution 2**: Wait for the rate limit period to expire (60 seconds)
- **Note**: The application will automatically stop batch processing to prevent quota waste

### API Key Issues
- Ensure your API key starts with `sk-`
- Re-copy the key directly from OpenAI (avoid PDFs/formatted documents)
- Check that your OpenAI account has credits
- Verify the key has API access enabled

### Folder Selection Not Working
- Try manually copying and pasting the folder path
- Ensure you have read/write permissions for the folder
- Use absolute paths (e.g., `/home/user/images` not `~/images`)

### Images Not Processing
- Check `error_log.txt` for detailed error messages
- Verify images are in supported formats (PNG, JPG, JPEG)
- Ensure images aren't corrupted
- Check file permissions

### Batch Processing Stops Unexpectedly
- Check for critical error message in the results
- Review `error_log.txt` for details
- Common causes: rate limits, connection issues, invalid API key
- Solution: Fix the issue and restart batch processing (already processed images will be skipped)

### Unicode/Encoding Errors
- The application automatically sanitizes inputs
- If issues persist, check that your API key doesn't contain special characters
- Prompts support Unicode, but be cautious with exotic characters

## Cost Estimation

OpenAI Vision API pricing (approximate, check current pricing):
- **gpt-4o-mini**: ~$0.0015 per image (low detail) to ~$0.0045 (high detail)
- **gpt-4o**: ~$0.0025 per image (low detail) to ~$0.0085 (high detail)
- **gpt-4-turbo**: ~$0.01 per image

**Example costs for 1000 images:**
- gpt-4o-mini (auto): $2-4
- gpt-4o (auto): $3-8
- gpt-4-turbo (auto): $8-12

*Costs vary based on image size and detail level. Always monitor your OpenAI usage dashboard.*

## Best Practices

1. **Test First**: Try a few images in single mode before batch processing
2. **Use gpt-4o-mini First**: Perfect for testing and large batches
3. **Start with 2 Workers**: Increase only if no rate limit issues occur
4. **Backup Your Data**: Keep original images safe before batch processing
5. **Monitor Costs**: Check your OpenAI usage dashboard regularly
6. **Use Auto Detail**: Let the API optimize cost vs quality automatically
7. **Review Error Logs**: Check `error_log.txt` if processing fails
8. **Cancel If Needed**: Don't hesitate to use the cancel button if something seems wrong

## Limitations and Considerations

- Caption accuracy depends on image quality and prompt clarity
- OpenAI API is rate-limited; the built-in rate limiter helps manage this
- Internet connectivity required for API communication
- API costs apply per image processed (see Cost Estimation)
- Very large batches may take significant time even with multiple workers
- The application respects API rate limits and will automatically stop if limits are exceeded

## Advanced Usage

### Custom Error Handling

Errors are automatically logged to `error_log.txt` with:
- Timestamp
- Error type (RATE_LIMIT, CONNECTION, API_ERROR, UNEXPECTED)
- Error message
- Stack trace (for unexpected errors)

### Canceling Batch Processing

Click the "Cancel" button at any time. The application will:
- Stop processing new images
- Finish the current image being processed
- Keep all captions already generated
- Display summary of processed/skipped/error counts

### Re-running Batch Processing

The application automatically skips images with existing `.txt` files. To reprocess:
1. Delete the existing `.txt` files you want to regenerate
2. Run batch processing again
3. Only deleted captions will be regenerated

### Custom Rate Limiting

Modify the rate limiter in `main.py` for different API tiers:

```python
# Conservative (Tier 1)
rate_limiter = RateLimiter(max_calls=10, period=60)

# Moderate (Tier 2+)
rate_limiter = RateLimiter(max_calls=20, period=60)

# Aggressive (High tier)
rate_limiter = RateLimiter(max_calls=50, period=60)
```

⚠️ **Warning:** Setting too high may result in 429 errors and processing stops.

## Technical Details

### Rate Limiter Implementation

The `RateLimiter` class uses a sliding window algorithm:
- Stores timestamps of all API calls
- Automatically cleans up expired calls
- Thread-safe with lock protection
- Smart waiting that calculates optimal sleep time
- Prevents indefinite hangs with timeout support

### Error Recovery System

The application distinguishes between:
1. **Critical Errors** (CriticalBatchError):
   - Stop all processing immediately
   - Prevent quota waste
   - Require user intervention
   
2. **Recoverable Errors**:
   - Log error
   - Continue with next image
   - Include in error count

### Thread Safety

All shared state is protected with locks:
- `processing_control` for batch state
- `rate_limiter.lock` for call tracking
- Prevents race conditions in concurrent operations

## Contributing

Contributions are welcome! Areas of interest:
- Additional model support
- Enhanced error recovery
- UI improvements
- Performance optimizations
- Documentation improvements

## Credits

- **Original Creator**: Lux42 (collaborated on early development)
- **This Repository**: Maintained and improved by [MR-444](https://github.com/MR-444)
- **ONZU Fork**: [ONZU/CaptainCaption](https://github.com/ONZU/CaptainCaption)

Special thanks to Lux42 for creating the original version and collaborating on its development.

## License

MIT License - See LICENSE file for details.

This project is provided as-is for personal and commercial use.

## Support

- **OpenAI API Issues**: Visit [platform.openai.com/docs](https://platform.openai.com/docs)
- **Application Issues**: Check `error_log.txt` or open an issue on GitHub
- **Feature Requests**: Open an issue with the enhancement tag

---

**Happy Captioning! 🎨**
