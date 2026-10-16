# Project History & Fork Lineage

## Fork Chain

```
42lux/CaptainCaption (Original)
    ↓ Forked December 19, 2023
ONZU/CaptainCaption (Jung von Matt CREATORS)
    ↓ Forked
MR-444/CaptainCaption (This Repository)
```

## Timeline

### 2023-2024: Original Project (42lux/CaptainCaption)
- Initial creation of GPT-4 Vision captioning tool
- Basic Gradio interface
- Simple rate limiting with reset thread
- Batch processing capabilities
- Repository later deleted or made private

### December 2023: ONZU Fork
- **ONZU** (Jung von Matt CREATORS organization) forked the project
- Last update: December 19, 2023

### 2024-Present: MR-444 Fork (This Repository)
This fork represents a significant refactoring and improvement over the earlier versions.
Because I contributed to the 42lux/CaptainCaption this fork is the only maintained one.

## Major Improvements in This Fork

### 1. Updated AI Model
**Now**:
- gpt-4o (current production model)
- Ollama support

### 2. Rate Limiter Redesign
- **Now**:
  - Sliding window with timestamp tracking
  - Smart waiting that calculates when slots will be available
  - Self-cleaning (auto-removes old calls)
  - No separate reset thread needed

### 3. Thread Safety
- **Now**: `ProcessingControl` class with proper locking

### 4. Better User Experience
- Detailed batch processing summaries (processed/skipped/errors)
- Progress bars with descriptions
- Better error messages using Gradio's error system
- Input validation before API calls
- Increased max workers from 4 to 8

### 5. Code Quality
- Comprehensive docstrings
- Better separation of concerns
- Improved error handling and logging
- Edge case handling (empty folders, missing files, etc.)

### 6. Fixed Caption Formatting
- **Before**: `pre_prompt + ", " + description + " " + post_prompt`
  - Always added commas and spaces even when fields empty
  
- **Now**: `format_caption()` function
  - Only joins non-empty parts
  - Consistent comma separation
  - No extraneous punctuation

## About Missing Pull Requests

If you previously integrated pull requests that are no longer visible, this could be because:

1. **Repository Chain Break**: When 42lux deleted their repository, the PR history was lost
2. **Fork Source Change**: Your fork now shows ONZU as the parent, not 42lux
3. **Separate Development**: Work done on your fork is independent of ONZU's version


## Contributing to This Fork

This repository is the actively maintained version. Contributions welcome!

### Areas for Future Enhancement

- [ ] Batch retry for failed images
- [ ] Cost tracking and estimation
- [ ] Custom prompt templates library

## License

This project maintains the **MIT License** from the original repository.

## Acknowledgments

We stand on the shoulders of giants:
- **42lux**: Original creator who built the foundation
- **OpenAI**: For the amazing GPT-4 Vision API
- **Gradio**: For the excellent UI framework
