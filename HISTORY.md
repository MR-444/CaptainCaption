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
- Preserved the original codebase
- Last update: December 19, 2023

### 2024-Present: MR-444 Fork (This Repository)
This fork represents a significant refactoring and improvement over the earlier versions.

## Major Improvements in This Fork

### 1. Updated AI Model
- **Before**: `gpt-4-vision-preview` (deprecated)
- **Now**: `gpt-4o` (current production model)

### 2. Rate Limiter Redesign
- **Before**: 
  - Simple counter that reset every 60 seconds
  - Could hang indefinitely when limit reached
  - Required separate thread for periodic reset
  
- **Now**:
  - Sliding window with timestamp tracking
  - Smart waiting that calculates when slots will be available
  - Self-cleaning (auto-removes old calls)
  - No separate reset thread needed

### 3. Thread Safety
- **Before**: Global `is_processing` variable (race conditions)
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

### How to Preserve Your Changes

If you had integrated features not present in ONZU's fork:

1. **Document them**: Create a FEATURES.md listing your additions
2. **Tag releases**: Use Git tags to mark significant versions
3. **Keep changelogs**: Document all changes in CHANGELOG.md
4. **Make it clear**: Update README to highlight your unique improvements

## Contributing to This Fork

This repository is the actively maintained version with modern improvements. Contributions welcome!

### Areas for Future Enhancement

- [ ] Support for other vision models (Anthropic Claude, Google Gemini)
- [ ] Integration with dataset management tools
- [ ] Caption quality scoring/filtering
- [ ] Batch retry for failed images
- [ ] Cost tracking and estimation
- [ ] Custom prompt templates library
- [ ] A/B testing between different prompts
- [ ] Export captions to CSV/JSON formats

## License

This project maintains the **MIT License** from the original repository.

## Acknowledgments

We stand on the shoulders of giants:
- **42lux**: Original creator who built the foundation
- **ONZU**: Preserved the project when the original was removed
- **OpenAI**: For the amazing GPT-4 Vision API
- **Gradio**: For the excellent UI framework
- **Contributors**: Anyone who has improved this tool along the way
