"""
CaptainCaption - Image captioning tool with OpenAI and Ollama support.

This module provides a Gradio web interface for generating image captions
using either OpenAI's GPT-4 Vision API (cloud) or Ollama (local).
"""
import base64
import datetime
import io
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, Tk
from threading import Lock

import gradio as gr
import numpy as np
from PIL import Image
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from rate_limiter import RateLimiter

FOLDER_SYMBOL = '\U0001f4c2'  # 📂
MAX_IMAGE_WIDTH = 2048
IMAGE_FORMAT = "JPEG"

# Rate limiter: 10 requests per minute (conservative for API tier limits)
# Note: Only used for OpenAI, not for Ollama
rate_limiter = RateLimiter(max_calls=10, period=60)


# Custom exception for critical errors that should stop batch processing
class CriticalBatchError(Exception):
    """Exception raised when batch processing should stop immediately."""


def log_error(error, error_type="GENERAL", include_traceback=True):
    """
    Log errors with appropriate detail level.

    Args:
        error: The exception object
        error_type: Type of error for categorization
        include_traceback: Whether to include full traceback
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("error_log.txt", 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"[{timestamp}] {error_type}: {str(error)}\n")

        if include_traceback:
            log_file.write(traceback.format_exc())


def sanitize_text(text):
    """
    Remove non-ASCII characters and normalize whitespace.

    Args:
        text: Input text string

    Returns:
        Cleaned ASCII-safe string
    """
    if not text:
        return text

    # Replace non-breaking spaces and other unicode spaces with regular spaces
    text = text.replace('\xa0', ' ').replace('\u2009', ' ').replace('\u200b', '')

    # Normalize to ASCII (keep only ASCII characters)
    # For prompts, we want to keep unicode, so we'll encode/decode carefully
    return text.strip()


def sanitize_api_key(api_key):
    """
    Clean API key by removing non-ASCII characters and extra whitespace.

    Args:
        api_key: The API key string

    Returns:
        Cleaned ASCII-only API key
    """
    if not api_key:
        return api_key

    # Remove any non-ASCII characters (API keys should be ASCII only)
    cleaned = ''.join(char for char in api_key if ord(char) < 128)

    # Remove all whitespace
    cleaned = ''.join(cleaned.split())

    return cleaned


def generate_description_openai(api_key, image, prompt, detail, max_tokens, model="gpt-4o-mini", batch_mode=False):
    """
    Generate image description using OpenAI's vision API.

    Args:
        batch_mode: If True, raises CriticalBatchError for rate limit/connection errors
                   If False, returns error strings (for single image mode)
    """

    # Sanitize inputs to prevent Unicode encoding errors
    api_key = sanitize_api_key(api_key)
    prompt = sanitize_text(prompt)

    # Reserve rate limit slot BEFORE making the call
    rate_limiter.wait()
    rate_limiter.add_call()

    try:
        # Load and process image
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
        img = scale_image(img)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format=IMAGE_FORMAT)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Make API call
        client = OpenAI(api_key=api_key)
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": detail}}
                ]
            }],
            "max_tokens": max_tokens
        }

        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content

    except RateLimitError as e:
        # Common error - log concisely without traceback
        log_error(e, "RATE_LIMIT", include_traceback=False)

        # In batch mode, this is critical - stop processing
        if batch_mode:
            raise CriticalBatchError(f"Rate limit exceeded (429). Stopping batch processing to prevent further failures.")
        return f"Error: Rate limit exceeded. Please wait and try again."

    except APIConnectionError as e:
        # Network error - log concisely
        log_error(e, "CONNECTION", include_traceback=False)

        # In batch mode, connection errors are also critical
        if batch_mode:
            raise CriticalBatchError(f"Connection failed. Check your internet connection and try again.")
        return f"Error: Connection failed. Check your internet connection."

    except APIError as e:
        # API error - log with some detail but no full traceback
        log_error(e, "API_ERROR", include_traceback=False)

        # Check if this is an authentication error (also critical in batch mode)
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            if batch_mode:
                raise CriticalBatchError(f"API authentication failed. Check your API key.")

        return f"Error: API error - {str(e)}"

    except Exception as e:
        # Unexpected error - log everything for debugging
        log_error(e, "UNEXPECTED", include_traceback=True)
        return f"Error: {str(e)}"


def generate_description_ollama(ollama_url, image, prompt, model="llava", batch_mode=False):
    """
    Generate image description using Ollama's vision API (local).

    Args:
        ollama_url: Base URL for Ollama (e.g., http://localhost:11434)
        image: Image array or path
        prompt: Text prompt
        model: Ollama model name (llava, llama3.2-vision, etc.)
        batch_mode: If True, raises CriticalBatchError for critical errors
    """
    try:
        # Import ollama library
        try:
            from ollama import Client
        except ImportError:
            error_msg = "Ollama library not installed. Install with: pip install ollama"
            if batch_mode:
                raise CriticalBatchError(error_msg)
            return f"Error: {error_msg}"

        # Load and process image
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
        img = scale_image(img)

        # Convert to base64 for Ollama
        buffered = io.BytesIO()
        img.save(buffered, format=IMAGE_FORMAT)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Make API call to Ollama
        client = Client(host=ollama_url) if ollama_url else Client()
        response = client.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]  # Send base64 encoded image
            }],
            options={'num_ctx': 2048}  # Context window
        )

        return response['message']['content']

    except ConnectionError as e:
        log_error(e, "OLLAMA_CONNECTION", include_traceback=False)
        error_msg = f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?"
        if batch_mode:
            raise CriticalBatchError(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        log_error(e, "OLLAMA_ERROR", include_traceback=True)
        error_msg = f"Ollama error: {str(e)}"
        if batch_mode:
            # Check if it's a critical error
            if "model" in str(e).lower() and "not found" in str(e).lower():
                raise CriticalBatchError(f"Model '{model}' not found. Pull it with: ollama pull {model}")
        return f"Error: {error_msg}"


def generate_description(provider, api_key, ollama_url, image, prompt, detail, max_tokens, model, batch_mode=False):
    """
    Generate image description using selected provider (OpenAI or Ollama).
    """
    if provider == "OpenAI":
        return generate_description_openai(api_key, image, prompt, detail, max_tokens, model, batch_mode)
    elif provider == "Ollama":
        return generate_description_ollama(ollama_url, image, prompt, model, batch_mode)
    else:
        return "Error: Unknown provider"


# History tracking
columns = ["Time", "Prompt", "Caption"]


def clear_fields():
    """Clear all input fields and history."""
    return "", [], []


def update_history(history_list, prompt, response):
    """Add entry to history and return formatted table data."""
    if history_list is None:
        history_list = []
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    history_list.append({"Time": timestamp, "Prompt": prompt, "Caption": response})
    return history_list, [[entry[column] for column in columns] for entry in history_list]


def scale_image(img):
    """Scale image down if it exceeds maximum width."""
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        new_height = int(img.height * ratio)
        return img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    return img


def get_dir(file_path):
    """Split file path into directory and filename."""
    dir_path, file_name = os.path.split(file_path)
    return dir_path, file_name


def get_folder_path(folder_path=''):
    """Open folder selection dialog."""
    current_folder_path = folder_path
    initial_dir, _ = get_dir(folder_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    if sys.platform == 'darwin':
        root.call('wm', 'attributes', '.', '-topmost', True)

    folder_path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()

    if folder_path == '':
        folder_path = current_folder_path

    return folder_path


# Thread-safe processing control
class ProcessingControl:
    """Thread-safe control for batch processing state management."""

    def __init__(self):
        self.is_processing = False
        self.should_stop = False  # Flag for critical errors
        self.lock = Lock()

    def start(self):
        """Mark processing as started."""
        with self.lock:
            self.is_processing = True
            self.should_stop = False

    def stop(self):
        """Request processing to stop."""
        with self.lock:
            self.should_stop = True

    def finish(self):
        """Mark processing as finished."""
        with self.lock:
            self.is_processing = False
            self.should_stop = False

    def is_stopped(self):
        """Check if stop was requested."""
        with self.lock:
            return self.should_stop

    def get_status(self):
        """Get current processing status."""
        with self.lock:
            return self.is_processing, self.should_stop

# Dictionary to hold per-user processing controls
user_processing_controls = {}

def get_processing_control(request: gr.Request):
    session_id = request.session_hash if request else "default"
    if session_id not in user_processing_controls:
        user_processing_controls[session_id] = ProcessingControl()
    return user_processing_controls[session_id]

def process_image(processing_control, provider, api_key, ollama_url, image_path, prompt, detail, max_tokens, model, pre_prompt, post_prompt):
    """
    Process a single image and save its caption to a text file.

    Returns:
        tuple: (success: bool, status: str, description: str or None)
    """
    # Check if processing should stop
    if processing_control.is_stopped():
        return False, "STOPPED", None

    txt_filename = os.path.splitext(image_path)[0] + '.txt'

    # Skip if caption already exists
    if os.path.exists(txt_filename):
        return True, "SKIPPED", None

    try:
        # Generate description
        description = generate_description(
            provider, api_key, ollama_url, image_path, prompt, detail, max_tokens, model, batch_mode=True
        )

        # Format with pre/post prompts
        caption = format_caption(pre_prompt, description, post_prompt)

        # Save to file
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(caption)

        return True, "SUCCESS", description

    except CriticalBatchError as e:
        # Critical error - signal to stop processing
        processing_control.stop()
        return False, "CRITICAL", str(e)

    except Exception as e:
        log_error(e, f"ERROR_PROCESSING_{os.path.basename(image_path)}")
        return False, "ERROR", str(e)


def process_folder(processing_control, provider, api_key, ollama_url, folder_path, prompt, detail, max_tokens, model, pre_prompt, post_prompt, num_workers=2):
    """
    Process all images in a folder using concurrent workers.
    """
    # Start processing
    processing_control.start()

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg'}
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not file_list:
        processing_control.finish()
        return "No image files found in the specified folder."

    # Counters
    processed_count = 0
    skipped_count = 0
    error_count = 0
    critical_error_msg = None

    # Process images with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image_path in file_list:
            future = executor.submit(
                process_image,
                processing_control, provider, api_key, ollama_url, image_path, prompt, detail, max_tokens, model,
                pre_prompt, post_prompt
            )
            futures.append((future, os.path.basename(image_path)))

        # Collect results
        for future, filename in futures:
            try:
                success, status, description = future.result()

                if status == "SUCCESS":
                    processed_count += 1
                    print(f"✓ Processed: {filename}")
                elif status == "SKIPPED":
                    skipped_count += 1
                    print(f"⊘ Skipped: {filename} (already has caption)")
                elif status == "CRITICAL":
                    critical_error_msg = description
                    print(f"✗ CRITICAL ERROR: {description}")
                    break  # Break here ONLY after saving the actual description
                elif status == "ERROR":
                    error_count += 1
                    print(f"✗ Error: {filename} - {description}")
                elif status == "STOPPED":
                    break

            except Exception as e:
                error_count += 1
                log_error(e, f"FUTURE_ERROR_{filename}")
                print(f"✗ Future error: {filename}")

    # Finish processing
    processing_control.finish()

    # Generate results summary
    if critical_error_msg:
        summary = f"❌ BATCH PROCESSING STOPPED - CRITICAL ERROR\n\n"
        summary += f"{critical_error_msg}\n\n"
        summary += f"Results before stopping:\n"
        summary += f"- Total files: {len(file_list)}\n"
        summary += f"- Processed successfully: {processed_count}\n"
        summary += f"- Skipped (already exist): {skipped_count}\n"
        summary += f"- Errors: {error_count}\n"
        summary += f"- Remaining: {len(file_list) - processed_count - skipped_count - error_count}\n\n"
        summary += f"Please resolve the issue before retrying batch processing."
    else:
        summary = f"✓ Batch processing complete:\n"
        summary += f"- Total files: {len(file_list)}\n"
        summary += f"- Processed: {processed_count}\n"
        summary += f"- Skipped (already exist): {skipped_count}\n"
        summary += f"- Errors: {error_count}"

    if error_count > 0:
        summary += f"\n\nCheck error_log.txt for details on failed files."

    return summary


def format_caption(pre_prompt, description, post_prompt):
    """Format caption with optional pre/post prompts."""
    parts = []

    if pre_prompt.strip():
        parts.append(pre_prompt.strip())

    parts.append(description.strip())

    if post_prompt.strip():
        parts.append(post_prompt.strip())

    return ", ".join(parts)


# Gradio UI
with gr.Blocks(title="GPT-4 Vision Image Captioner") as app:
    gr.Markdown("# 🖼️ Image Captioner (OpenAI + Ollama)")
    gr.Markdown("Generate captions for single images or batch process entire folders using OpenAI or local Ollama models.")

    history_state = gr.State(lambda: [])

    with gr.Row():
        provider_selector = gr.Radio(
            ["OpenAI", "Ollama"],
            value="OpenAI",
            label="Provider",
            info="Choose between cloud (OpenAI) or local (Ollama)"
        )

    with gr.Row():
        api_key_input = gr.Textbox(
            scale=2,
            label="OpenAI API Key",
            placeholder="sk-...",
            type="password",
            info="Required for OpenAI only",
            visible=True
        )
        ollama_url_input = gr.Textbox(
            scale=2,
            label="Ollama URL",
            value="http://localhost:11434",
            info="URL where Ollama is running",
            visible=False
        )

    with gr.Row():
        openai_model_selector = gr.Dropdown(
            scale=1,
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            value="gpt-4o-mini",
            label="OpenAI Model",
            info="gpt-4o-mini: best value, gpt-4o: highest quality",
            visible=True
        )
        ollama_model_selector = gr.Dropdown(
            scale=1,
            choices=["llama3.2-vision", "qwen2.5-vl", "minicpm-v", "moondream", "llava", "llava:13b", "bakllava"],
            value="llama3.2-vision",
            label="Ollama Model",
            info="llama3.2-vision: best overall, qwen2.5-vl: best for OCR",
            visible=False
        )

    # Update visibility based on provider selection
    def update_provider_visibility(provider):
        """Update UI element visibility based on selected provider."""
        is_openai = provider == "OpenAI"
        return (
            gr.update(visible=is_openai),  # api_key_input
            gr.update(visible=not is_openai),  # ollama_url_input
            gr.update(visible=is_openai),  # openai_model_selector
            gr.update(visible=not is_openai),  # ollama_model_selector
            gr.update(visible=is_openai),  # detail_level (only for OpenAI)
            gr.update(visible=is_openai),  # max_tokens (only for OpenAI)
            gr.update(visible=is_openai),  # detail_level_dataset
            gr.update(visible=is_openai),  # max_tokens_dataset
        )

    with gr.Tab("Single Image"):
        image_input = gr.Image(label="Upload Image")

        with gr.Row():
            prompt_input = gr.Textbox(
                scale=6,
                label="Prompt",
                value="Describe this image in detail. Focus on the main subjects, actions, setting, and mood.",
                interactive=True
            )
            detail_level = gr.Radio(
                ["high", "low", "auto"],
                scale=2,
                label="Detail Level (OpenAI only)",
                value="auto",
                info="High = more tokens, better detail",
                visible=True
            )
            max_tokens_input = gr.Number(
                scale=1,
                value=300,
                label="Max Tokens (OpenAI only)",
                minimum=50,
                maximum=1000,
                visible=True
            )

        submit_button = gr.Button("Generate Caption", variant="primary")
        output = gr.Textbox(label="Generated Caption", lines=5)

        with gr.Accordion("History", open=False):
            history_table = gr.Dataframe(headers=columns, interactive=False)
            clear_button = gr.Button("Clear History")

    with gr.Tab("Batch Processing"):
        gr.Markdown("Process all images in a folder. Creates .txt files with captions next to each image.")
        gr.Markdown("⚠️ **OpenAI:** Processing will stop if rate limits occur. **Ollama:** No rate limits (local).")

        with gr.Row():
            folder_path_dataset = gr.Textbox(
                scale=8,
                label="Dataset Folder Path",
                placeholder="/path/to/your/images",
                interactive=True,
                info="Folder containing images to caption"
            )
            folder_button = gr.Button('📂', scale=1, size="sm")

        with gr.Row():
            prompt_input_dataset = gr.Textbox(
                scale=6,
                label="Prompt",
                value="Describe this image in detail. Focus on the main subjects, actions, setting, and mood.",
                interactive=True
            )
            detail_level_dataset = gr.Radio(
                ["high", "low", "auto"],
                scale=2,
                label="Detail Level (OpenAI only)",
                value="auto",
                visible=True
            )
            max_tokens_input_dataset = gr.Number(
                scale=1,
                value=300,
                label="Max Tokens (OpenAI only)",
                minimum=50,
                maximum=1000,
                visible=True
            )

        with gr.Row():
            pre_prompt_input = gr.Textbox(
                scale=1,
                label="Prefix",
                placeholder="e.g., 'masterpiece, high quality'",
                info="Added at the start of each caption",
                interactive=True
            )
            post_prompt_input = gr.Textbox(
                scale=1,
                label="Postfix",
                placeholder="e.g., 'trending on artstation'",
                info="Added at the end of each caption",
                interactive=True
            )

        with gr.Row():
            worker_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="Concurrent Workers",
                info="More workers = faster. OpenAI: may hit rate limits. Ollama: limited by hardware"
            )

        with gr.Row():
            submit_button_dataset = gr.Button("Start Batch Processing", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel", variant="stop", scale=1)

        processing_results_output = gr.Textbox(label="Processing Results", lines=5)

    # Event handlers
    def cancel_processing(request: gr.Request):
        """Cancel ongoing batch processing."""
        pc = get_processing_control(request)
        pc.stop()
        return "⚠️ Processing canceled by user"

    def on_single_image_submit(history_list, provider, api_key, ollama_url, openai_model, ollama_model, image, prompt, detail, max_tokens):
        """Handle single image caption generation."""
        # Select the right model based on provider
        model = openai_model if provider == "OpenAI" else ollama_model

        # Validate inputs based on provider
        if provider == "OpenAI":
            if not api_key.strip():
                raise gr.Error("Please enter your OpenAI API key")
            if '\xa0' in api_key or any(ord(c) >= 128 for c in api_key):
                raise gr.Error("API key contains invalid characters. Please re-copy your API key.")
        else:  # Ollama
            if not ollama_url.strip():
                raise gr.Error("Please enter Ollama URL")

        if image is None:
            raise gr.Error("Please upload an image")

        # Generate description
        description = generate_description(
            provider, api_key, ollama_url, image, prompt, detail, max_tokens, model, batch_mode=False
        )
        new_history_list, display_data = update_history(history_list, prompt, description)
        return description, display_data, new_history_list

    def on_batch_submit(request: gr.Request, provider, api_key, ollama_url, openai_model, ollama_model, folder_path, prompt, detail, max_tokens, pre_prompt, post_prompt, num_workers):
        """Handle batch folder processing."""
        pc = get_processing_control(request)

        # Select the right model based on provider
        model = openai_model if provider == "OpenAI" else ollama_model

        # Validate inputs based on provider
        if provider == "OpenAI":
            if not api_key.strip():
                raise gr.Error("Please enter your OpenAI API key")
            if '\xa0' in api_key or any(ord(c) >= 128 for c in api_key):
                raise gr.Error("API key contains invalid characters. Please re-copy your API key.")
        else:  # Ollama
            if not ollama_url.strip():
                raise gr.Error("Please enter Ollama URL")

        if not folder_path.strip():
            raise gr.Error("Please enter a folder path")

        result = process_folder(
            pc, provider, api_key, ollama_url, folder_path, prompt, detail, max_tokens, model,
            pre_prompt, post_prompt, num_workers=int(num_workers)
        )
        return result

    # Wire up events
    provider_selector.change(
        update_provider_visibility,
        inputs=[provider_selector],
        outputs=[
            api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector,
            detail_level, max_tokens_input,
            detail_level_dataset, max_tokens_input_dataset
        ]
    )

    clear_button.click(clear_fields, inputs=[], outputs=[output, history_table, history_state])

    folder_button.click(get_folder_path, outputs=folder_path_dataset, show_progress="hidden")

    submit_button.click(
        on_single_image_submit,
        inputs=[
            history_state, provider_selector, api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector,
            image_input, prompt_input, detail_level, max_tokens_input
        ],
        outputs=[output, history_table, history_state]
    )

    cancel_button.click(cancel_processing, inputs=[], outputs=[processing_results_output])

    submit_button_dataset.click(
        on_batch_submit,
        inputs=[
            provider_selector, api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector,
            folder_path_dataset,
            prompt_input_dataset,
            detail_level_dataset,
            max_tokens_input_dataset,
            pre_prompt_input,
            post_prompt_input,
            worker_slider
        ],
        outputs=[processing_results_output]
    )


if __name__ == "__main__":
    app.launch()
