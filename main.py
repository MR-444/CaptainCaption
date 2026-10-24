"""
CaptainCaption - Image captioning tool with OpenAI and Ollama support.

This module provides a Gradio web interface for generating image captions
using either OpenAI's GPT-4 Vision API (cloud) or Ollama (local).
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import io
import json
import urllib.request
import logging.handlers
import os
import pandas as pd
import re
import signal
import sys
import traceback
from tkinter import filedialog, Tk
from threading import Lock

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError

from rate_limiter import RateLimiter

# Graceful top-level import for Ollama client to satisfy static analysis
try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

APP_TITLE = "Image Captioner"
FOLDER_SYMBOL = '\U0001f4c2'  # 📂
DEFAULT_MAX_IMAGE_DIM = 1024
DEFAULT_PROMPT = "Describe this image in detail. Focus on the main subjects, actions, setting, and mood."
DEFAULT_MAX_TOKENS = 300
OLLAMA_DEFAULT_TIMEOUT = 180.0
OLLAMA_DEFAULT_NUM_CTX = 8192

# Allow large camera photos up to 150 megapixels safely
Image.MAX_IMAGE_PIXELS = 150_000_000

# Rate limiter: 10 requests per minute (conservative for API tier limits)
rate_limiter = RateLimiter(max_calls=10, period=60)


class CriticalBatchError(Exception):
    """Exception raised when batch processing should stop immediately."""


_error_logger = logging.getLogger("captaincaption.errors")
_error_logger.setLevel(logging.INFO)
_error_logger.propagate = False
if not _error_logger.handlers:
    _handler = logging.handlers.RotatingFileHandler(
        "error_log.txt", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    _error_logger.addHandler(_handler)


def log_error(error, error_type="GENERAL", include_traceback=True):
    """Log errors to a rotating file (5 MB × 3 backups, thread-safe)."""
    message = f"{error_type}: {error}"
    if include_traceback:
        message += "\n" + traceback.format_exc()
    _error_logger.info(message)


def sanitize_text(text):
    """Remove non-ASCII characters and normalize whitespace."""
    if not text:
        return text
    text = text.replace('\xa0', ' ').replace('\u2009', ' ').replace('\u200b', '')
    return text.strip()


def sanitize_api_key(api_key):
    """Clean API key by removing non-ASCII characters and extra whitespace."""
    if not api_key:
        return api_key
    cleaned = ''.join(char for char in api_key if ord(char) < 128)
    return ''.join(cleaned.split())


def prepare_image_base64(image_input, max_dim=DEFAULT_MAX_IMAGE_DIM, quality=85):
    """
    Optimized image preparation pipeline:
    1. Safely loads from numpy array or file path.
    2. Applies EXIF rotation (fixes sideways/upside-down camera photos).
    3. Converts RGBA, CMYK, P, and LA to standard RGB.
    4. Resizes along the longest edge to max_dim (aspect-ratio preserving).
    5. Encodes to an optimized, compressed JPEG Base64 payload.
    """
    if isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        img = Image.open(image_input)

    try:
        # Correct camera orientation from EXIF metadata
        img = ImageOps.exif_transpose(img)

        # Convert all color modes (CMYK, RGBA, Grayscale, etc.) to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Scale down if longest side exceeds max_dim
        max_edge = max(img.width, img.height)
        if max_edge > max_dim:
            ratio = max_dim / max_edge
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save to optimized JPEG in-memory buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    finally:
        img.close()


def generate_description_openai(api_key, image, prompt, detail, max_tokens, model="gpt-4o-mini",
                                batch_mode=False, max_image_dim=DEFAULT_MAX_IMAGE_DIM):
    """Generate image description using OpenAI's vision API."""
    api_key = sanitize_api_key(api_key)
    prompt = sanitize_text(prompt)

    if not rate_limiter.reserve(timeout=120.0):
        if batch_mode:
            raise CriticalBatchError("Rate limiter aborted or timed out.")
        return "Error: Request cancelled or timed out."

    try:
        img_base64 = prepare_image_base64(image, max_dim=max_image_dim)

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
        log_error(e, "RATE_LIMIT", include_traceback=False)
        if batch_mode:
            raise CriticalBatchError("Rate limit exceeded (429). Stopping batch processing to prevent further failures.")
        return "Error: Rate limit exceeded. Please wait and try again."

    except APIConnectionError as e:
        log_error(e, "CONNECTION", include_traceback=False)
        if batch_mode:
            raise CriticalBatchError("Connection failed. Check your internet connection and try again.")
        return "Error: Connection failed. Check your internet connection."

    except AuthenticationError as e:
        log_error(e, "AUTH", include_traceback=False)
        if batch_mode:
            raise CriticalBatchError("API authentication failed. Check your API key.")
        return "Error: API authentication failed. Check your API key."

    except APIError as e:
        log_error(e, "API_ERROR", include_traceback=False)
        if batch_mode and ("authentication" in str(e).lower() or "api key" in str(e).lower()):
            raise CriticalBatchError("API authentication failed. Check your API key.")
        return f"Error: API error - {str(e)}"

    except Exception as e:
        log_error(e, "UNEXPECTED", include_traceback=True)
        return f"Error: {str(e)}"


def generate_description_ollama(ollama_url, image, prompt, model="llava", batch_mode=False,
                                timeout=OLLAMA_DEFAULT_TIMEOUT, num_ctx=OLLAMA_DEFAULT_NUM_CTX,
                                max_image_dim=DEFAULT_MAX_IMAGE_DIM):
    """Generate image description using Ollama's vision API (local)."""
    if OllamaClient is None:
        error_msg = "Ollama library not installed. Install with: pip install ollama"
        if batch_mode:
            raise CriticalBatchError(error_msg)
        return f"Error: {error_msg}"

    try:
        img_base64 = prepare_image_base64(image, max_dim=max_image_dim)

        client = OllamaClient(host=ollama_url, timeout=timeout) if ollama_url else OllamaClient(timeout=timeout)

        chat_kwargs = {
            'model': model,
            'messages': [{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }],
            'options': {'num_ctx': int(num_ctx)},
            'think': False,  # Disables thinking/reasoning mode for speed
        }
        try:
            response = client.chat(**chat_kwargs)
        except TypeError:
            # Fallback if installed ollama-python package does not yet support the 'think' argument
            chat_kwargs.pop('think', None)
            response = client.chat(**chat_kwargs)

        content = response.get('message', {}).get('content') if isinstance(response, dict) else response.message.content
        if not content or not str(content).strip():
            resp_dump = dict(response) if isinstance(response, dict) else getattr(response, '__dict__', str(response))
            log_error(f"Ollama returned empty response. Metadata: {resp_dump}", "OLLAMA_EMPTY_RESPONSE",
                      include_traceback=False)
            return "Error: Ollama returned an empty response."

        # Safety cleanup: strip any residual <think>...</think> blocks from output
        cleaned_content = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL).strip()
        if not cleaned_content:
            return str(content).strip()

        return cleaned_content

    except ConnectionError as e:
        log_error(e, "OLLAMA_CONNECTION", include_traceback=False)
        error_msg = f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?"
        if batch_mode:
            raise CriticalBatchError(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        log_error(e, "OLLAMA_ERROR", include_traceback=True)
        error_str = str(e)

        if "timed out" in error_str.lower():
            error_msg = f"Ollama request timed out (>{timeout}s). Try increasing 'Timeout' or decreasing 'Image Dimension'."
        else:
            error_msg = f"Ollama error: {error_str}"

        if batch_mode:
            lower_err = error_str.lower()
            if "model" in lower_err and "not found" in lower_err:
                raise CriticalBatchError(f"Model '{model}' not found. Pull it with: ollama pull {model}")
            if "exceed_context_size_error" in lower_err or "exceeds the available context size" in lower_err:
                raise CriticalBatchError(
                    f"Context window exceeded ({error_str}). "
                    f"Lower the 'Max Image Dimension' or increase 'Ollama Context'."
                )

        return f"Error: {error_msg}"


def generate_description(provider, api_key, ollama_url, image, prompt, detail, max_tokens, model,
                         batch_mode=False, ollama_timeout=OLLAMA_DEFAULT_TIMEOUT,
                         ollama_num_ctx=OLLAMA_DEFAULT_NUM_CTX, max_image_dim=DEFAULT_MAX_IMAGE_DIM):
    """Generate image description using selected provider (OpenAI or Ollama)."""
    if provider == "OpenAI":
        return generate_description_openai(api_key, image, prompt, detail, max_tokens, model,
                                           batch_mode, max_image_dim=max_image_dim)
    elif provider == "Ollama":
        return generate_description_ollama(ollama_url, image, prompt, model, batch_mode,
                                           timeout=ollama_timeout, num_ctx=ollama_num_ctx,
                                           max_image_dim=max_image_dim)
    else:
        return "Error: Unknown provider"


# History tracking
columns = ["Time", "Prompt", "Caption"]


def clear_fields():
    return "", pd.DataFrame(columns=columns), []


def update_history(history_list, prompt, response):
    if history_list is None:
        history_list = []
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    history_list.append({"Time": timestamp, "Prompt": prompt, "Caption": response})
    return history_list, [[entry[column] for column in columns] for entry in history_list]

def get_folder_path(folder_path=''):
    current_folder_path = folder_path
    initial_dir = folder_path if os.path.isdir(folder_path) else os.path.dirname(folder_path)

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


class ProcessingControl:
    """Thread-safe control for batch processing state management."""

    def __init__(self):
        self.should_stop = False
        self.lock = Lock()

    def start(self):
        with self.lock:
            self.should_stop = False

    def stop(self):
        with self.lock:
            self.should_stop = True

    def finish(self):
        with self.lock:
            self.should_stop = False

    def is_stopped(self):
        with self.lock:
            return self.should_stop


user_processing_controls = {}
_controls_lock = Lock()


def get_processing_control(request: gr.Request):
    session_id = request.session_hash if request else "default"
    with _controls_lock:
        return user_processing_controls.setdefault(session_id, ProcessingControl())


def stop_all_processing():
    """Immediately stop all batch tasks and wake any rate-limited workers."""
    with _controls_lock:
        for pc in user_processing_controls.values():
            pc.stop()
    rate_limiter.abort()


def process_image(processing_control, provider, api_key, ollama_url, image_path, prompt, detail,
                  max_tokens, model, pre_prompt, post_prompt, ollama_timeout=OLLAMA_DEFAULT_TIMEOUT,
                  ollama_num_ctx=OLLAMA_DEFAULT_NUM_CTX, max_image_dim=DEFAULT_MAX_IMAGE_DIM):
    """Process a single image and save its caption to a text file."""
    if processing_control.is_stopped():
        return False, "STOPPED", None

    txt_filename = os.path.splitext(image_path)[0] + '.txt'

    # Skip ONLY if caption already exists AND is not empty
    if os.path.exists(txt_filename) and os.path.getsize(txt_filename) > 0:
        return True, "SKIPPED", None

    try:
        description = generate_description(
            provider, api_key, ollama_url, image_path, prompt, detail, max_tokens, model,
            batch_mode=True, ollama_timeout=ollama_timeout, ollama_num_ctx=ollama_num_ctx,
            max_image_dim=max_image_dim
        )

        if processing_control.is_stopped():
            return False, "STOPPED", None

        if not description or (isinstance(description, str) and description.startswith("Error:")):
            log_error(f"File '{os.path.basename(image_path)}' failed: {description}", "PROCESS_IMAGE_FAILED",
                      include_traceback=False)
            return False, "ERROR", description

        caption = format_caption(pre_prompt, description, post_prompt)

        if not caption.strip():
            return False, "ERROR", "Generated caption was empty"

        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(caption)

        return True, "SUCCESS", description

    except CriticalBatchError as e:
        processing_control.stop()
        rate_limiter.abort()
        return False, "CRITICAL", str(e)

    except Exception as e:
        log_error(e, f"ERROR_PROCESSING_{os.path.basename(image_path)}")
        return False, "ERROR", str(e)


def process_folder(processing_control, provider, api_key, ollama_url, folder_path, prompt, detail,
                   max_tokens, model, pre_prompt, post_prompt, num_workers=2,
                   ollama_timeout=OLLAMA_DEFAULT_TIMEOUT, ollama_num_ctx=OLLAMA_DEFAULT_NUM_CTX,
                   max_image_dim=DEFAULT_MAX_IMAGE_DIM, progress=gr.Progress()):
    """Process all images in a folder using concurrent workers."""
    # Reset any previous abort states on the rate limiter
    rate_limiter.reset()
    processing_control.start()

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
    file_list = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not file_list:
        processing_control.finish()
        return "No image files found in the specified folder."

    total = len(file_list)
    processed_count = 0
    skipped_count = 0
    error_count = 0
    critical_error_msg = None

    try:
        progress(0, desc=f"Starting batch ({total} files)", total=total)
    except Exception:
        pass

    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = []
    try:
        for image_path in file_list:
            future = executor.submit(
                process_image,
                processing_control, provider, api_key, ollama_url, image_path, prompt, detail, max_tokens, model,
                pre_prompt, post_prompt, ollama_timeout, ollama_num_ctx, max_image_dim
            )
            futures.append((future, os.path.basename(image_path)))

        for idx, (future, filename) in enumerate(futures, start=1):
            if processing_control.is_stopped():
                break

            try:
                _, status, description = future.result()

                if status == "SUCCESS":
                    processed_count += 1
                    print(f"✓ Processed: {filename}")
                elif status == "SKIPPED":
                    skipped_count += 1
                    print(f"⊘ Skipped: {filename} (already has caption)")
                elif status == "CRITICAL":
                    critical_error_msg = description
                    print(f"✗ CRITICAL ERROR: {description}")
                    break
                elif status == "ERROR":
                    error_count += 1
                    print(f"✗ Error: {filename} - {description}")
                elif status == "STOPPED":
                    break

                try:
                    progress(
                        idx / total,
                        desc=f"{idx}/{total} — {filename}",
                        total=total,
                    )
                except Exception:
                    pass

            except Exception as e:
                error_count += 1
                log_error(e, f"FUTURE_ERROR_{filename}")
                print(f"✗ Future error: {filename}")

    finally:
        for pending, _ in futures:
            pending.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        processing_control.finish()

    if critical_error_msg:
        summary = "❌ BATCH PROCESSING STOPPED - CRITICAL ERROR\n\n"
        summary += f"{critical_error_msg}\n\n"
        summary += "Results before stopping:\n"
        summary += f"- Total files: {len(file_list)}\n"
        summary += f"- Processed successfully: {processed_count}\n"
        summary += f"- Skipped (already had captions): {skipped_count}\n"
        summary += f"- Errors: {error_count}\n"
        summary += f"- Remaining: {len(file_list) - processed_count - skipped_count - error_count}\n\n"
        summary += "Please resolve the issue before retrying batch processing."
    elif processing_control.is_stopped():
        summary = "⚠️ Batch processing canceled by user.\n"
        summary += f"- Processed before stopping: {processed_count}\n"
        summary += f"- Skipped: {skipped_count}\n"
        summary += f"- Errors: {error_count}"
    else:
        summary = "✓ Batch processing complete:\n"
        summary += f"- Total files: {len(file_list)}\n"
        summary += f"- Processed: {processed_count}\n"
        summary += f"- Skipped (already had captions): {skipped_count}\n"
        summary += f"- Errors: {error_count}"

    if error_count > 0:
        summary += "\n\nCheck error_log.txt for details on failed files."

    return summary


_VISION_NAME_HINTS = (
    "llava", "bakllava", "moondream", "vision", "-vl", "minicpm-v",
    "gemma3", "pixtral", "internvl", "qwen2-vl", "qwen2.5-vl",
)


def list_ollama_models(url, timeout=3.0):
    base = (url or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    names = set()
    for entry in data.get("models", []) or []:
        name = entry.get("model") or entry.get("name")
        if name:
            names.add(name)
    return sorted(names)


def list_running_ollama_models(url, timeout=3.0):
    base = (url or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return set()

    loaded = set()
    for entry in data.get("models", []) or []:
        name = entry.get("model") or entry.get("name")
        if name:
            loaded.add(name)
    return loaded


def get_ollama_model_capabilities(url, model_name, timeout=3.0):
    base = (url or "http://localhost:11434").rstrip("/")
    req = urllib.request.Request(
        f"{base}/api/show",
        data=json.dumps({"name": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return data.get("capabilities")


def list_ollama_vision_models(url, timeout=3.0):
    names = list_ollama_models(url, timeout=timeout)
    if not names:
        return [], []

    workers = min(8, len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        caps_list = list(pool.map(
            lambda n: get_ollama_model_capabilities(url, n, timeout),
            names,
        ))

    vision, other, unknown = [], [], []
    for name, caps in zip(names, caps_list):
        if caps is None:
            unknown.append(name)
        elif isinstance(caps, (list, tuple, set)) and "vision" in caps:
            vision.append(name)
        else:
            other.append(name)

    if unknown and not vision and not other:
        for name in unknown:
            lower = name.lower()
            if any(hint in lower for hint in _VISION_NAME_HINTS):
                vision.append(name)
            else:
                other.append(name)
        return vision, other

    return vision, other + unknown


def format_caption(pre_prompt, description, post_prompt):
    parts = []
    if pre_prompt and pre_prompt.strip():
        parts.append(pre_prompt.strip())
    if description and description.strip():
        parts.append(description.strip())
    if post_prompt and post_prompt.strip():
        parts.append(post_prompt.strip())
    return ", ".join(parts)


# Gradio UI
with gr.Blocks(title=APP_TITLE) as app:
    gr.Markdown(
        f"""
        # 🖼️ {APP_TITLE}
        Generate captions for single images or batch-process entire folders using OpenAI or local Ollama models.
        """
    )

    history_state = gr.State(lambda: [])

    with gr.Group():
        provider_selector = gr.Radio(
            ["OpenAI", "Ollama"],
            value="OpenAI",
            label="Provider",
            info="Choose between cloud (OpenAI) or local (Ollama)"
        )

        with gr.Row(equal_height=True):
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
            openai_model_selector = gr.Dropdown(
                scale=2,
                choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                value="gpt-4o-mini",
                label="OpenAI Model",
                info="gpt-4o-mini: best value, gpt-4o: highest quality",
                visible=True
            )
            ollama_model_selector = gr.Dropdown(
                scale=2,
                choices=[],
                value="",
                label="Ollama Model",
                info="Detected from your Ollama instance",
                visible=False,
                allow_custom_value=True,
            )
            ollama_refresh_button = gr.Button(
                "🔄", scale=0, size="sm", visible=False,
                variant="secondary",
            )
            ollama_num_ctx_input = gr.Number(
                scale=1,
                value=OLLAMA_DEFAULT_NUM_CTX,
                label="Context (num_ctx)",
                info="Tokens for image + prompt",
                minimum=2048,
                maximum=32768,
                step=1024,
                visible=False,
            )
            ollama_timeout_input = gr.Number(
                scale=1,
                value=OLLAMA_DEFAULT_TIMEOUT,
                label="Timeout (s)",
                info="Model load + inference",
                minimum=10,
                maximum=900,
                visible=False,
            )

    with gr.Row():
        image_size_slider = gr.Slider(
            minimum=512,
            maximum=2048,
            value=DEFAULT_MAX_IMAGE_DIM,
            step=128,
            label="Max Image Dimension (px)",
            info="1024px is optimal. 2048px doubles processing time and VRAM with negligible caption difference."
        )

    def _decorate_model_choices(models, running):
        return [
            ((f"● {m}  (loaded)" if m in running else f"○ {m}"), m)
            for m in models
        ]

    def _ollama_model_dropdown_update(ollama_url, visible):
        if not visible:
            return gr.update(visible=False)
        vision, other = list_ollama_vision_models(ollama_url)
        running = list_running_ollama_models(ollama_url)
        if not vision and not other:
            return gr.update(
                choices=[], value="", visible=True,
                info=f"⚠️ Could not reach Ollama at {ollama_url}. Start it, then click 🔄",
            )
        models = vision or other
        note_kind = "vision-capable" if vision else "⚠️ no vision models, showing all"
        loaded_count = sum(1 for m in models if m in running)
        info = (
            f"{len(models)} {note_kind} · {loaded_count} already loaded "
            "(● = loaded, ○ = needs load)"
        )
        return gr.update(
            choices=_decorate_model_choices(models, running),
            value=next((m for m in models if m in running), models[0]),
            visible=True,
            info=info,
        )

    def update_provider_visibility(provider):
        is_openai = provider == "OpenAI"
        ollama_placeholder = gr.update(
            visible=not is_openai,
            info="Click 🔄 to load models from Ollama",
        ) if not is_openai else gr.update(visible=False)
        return (
            gr.update(visible=is_openai),      # api_key_input
            gr.update(visible=not is_openai),  # ollama_url_input
            gr.update(visible=is_openai),      # openai_model_selector
            ollama_placeholder,                # ollama_model_selector
            gr.update(visible=not is_openai),  # ollama_refresh_button
            gr.update(visible=not is_openai),  # ollama_num_ctx_input
            gr.update(visible=not is_openai),  # ollama_timeout_input
            gr.update(visible=is_openai),      # detail_level
            gr.update(visible=is_openai),      # max_tokens_input
            gr.update(visible=is_openai),      # detail_level_dataset
            gr.update(visible=is_openai),      # max_tokens_input_dataset
        )

    def refresh_ollama_models(ollama_url):
        return _ollama_model_dropdown_update(ollama_url, visible=True)

    def auto_refresh_on_ollama(provider, ollama_url):
        if provider != "Ollama":
            return gr.update()
        return _ollama_model_dropdown_update(ollama_url, visible=True)

    with gr.Tab("Single Image"):
        image_input = gr.Image(label="Upload Image", height=360)

        with gr.Row():
            prompt_input = gr.Textbox(
                scale=6,
                label="Prompt",
                value=DEFAULT_PROMPT,
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
                value=DEFAULT_MAX_TOKENS,
                label="Max Tokens (OpenAI only)",
                minimum=50,
                maximum=1000,
                visible=True
            )

        submit_button = gr.Button("Generate Caption", variant="primary")
        output = gr.Textbox(label="Generated Caption", lines=5)

        with gr.Accordion("History", open=False):
            history_table = gr.Dataframe(
                headers=columns,
                interactive=False,
                wrap=True,
            )
            clear_button = gr.Button("Clear History")

    with gr.Tab("Batch Processing"):
        gr.Markdown("Process all images in a folder. Creates `.txt` files with captions next to each image.")
        gr.Markdown("⚠️ **OpenAI:** Processing will stop if rate limits occur. **Ollama:** No rate limits (local).")

        with gr.Row():
            folder_path_dataset = gr.Textbox(
                scale=8,
                label="Dataset Folder Path",
                placeholder="/path/to/your/images",
                interactive=True,
                info="Folder containing images to caption"
            )
            folder_button = gr.Button(FOLDER_SYMBOL, scale=1, size="sm")

        with gr.Row():
            prompt_input_dataset = gr.Textbox(
                scale=6,
                label="Prompt",
                value=DEFAULT_PROMPT,
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
                value=DEFAULT_MAX_TOKENS,
                label="Max Tokens (OpenAI only)",
                minimum=50,
                maximum=1000,
                visible=True
            )

        with gr.Accordion("Advanced", open=False):
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
            worker_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="Concurrent Workers",
                info="For Ollama, 1-2 workers are fastest on single GPUs to avoid VRAM thrashing."
            )

        with gr.Row():
            submit_button_dataset = gr.Button("Start Batch Processing", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel", variant="stop", scale=1)

        processing_results_output = gr.Textbox(label="Processing Results", lines=5)

    def cancel_processing(request: gr.Request):
        pc = get_processing_control(request)
        pc.stop()
        rate_limiter.abort()
        gr.Info("Cancel requested — workers will stop after the current file.")
        return "⚠️ Processing canceled by user"

    def _validate_provider_inputs(provider, api_key, ollama_url):
        if provider == "OpenAI":
            if not api_key.strip():
                raise gr.Error("Please enter your OpenAI API key")
            if '\xa0' in api_key or any(ord(c) >= 128 for c in api_key):
                raise gr.Error("API key contains invalid characters. Please re-copy your API key.")
        else:
            if not ollama_url.strip():
                raise gr.Error("Please enter Ollama URL")

    def on_single_image_submit(history_list, provider, api_key, ollama_url, openai_model,
                               ollama_model, ollama_num_ctx, ollama_timeout, max_image_dim,
                               image, prompt, detail, max_tokens):
        model = openai_model if provider == "OpenAI" else ollama_model
        _validate_provider_inputs(provider, api_key, ollama_url)

        if image is None:
            raise gr.Error("Please upload an image")

        description = generate_description(
            provider, api_key, ollama_url, image, prompt, detail, max_tokens, model,
            batch_mode=False,
            ollama_timeout=float(ollama_timeout or OLLAMA_DEFAULT_TIMEOUT),
            ollama_num_ctx=int(ollama_num_ctx or OLLAMA_DEFAULT_NUM_CTX),
            max_image_dim=int(max_image_dim or DEFAULT_MAX_IMAGE_DIM),
        )
        new_history_list, display_data = update_history(history_list, prompt, description)
        return description, display_data, new_history_list

    def on_batch_submit(request: gr.Request, provider, api_key, ollama_url, openai_model,
                        ollama_model, ollama_num_ctx, ollama_timeout, max_image_dim, folder_path,
                        prompt, detail, max_tokens, pre_prompt, post_prompt, num_workers,
                        progress=gr.Progress()):
        pc = get_processing_control(request)
        model = openai_model if provider == "OpenAI" else ollama_model
        _validate_provider_inputs(provider, api_key, ollama_url)

        if not folder_path.strip():
            raise gr.Error("Please enter a folder path")
        if not os.path.isdir(folder_path):
            raise gr.Error(f"Folder not found: {folder_path}")

        return process_folder(
            pc, provider, api_key, ollama_url, folder_path, prompt, detail, max_tokens, model,
            pre_prompt, post_prompt, num_workers=int(num_workers),
            ollama_timeout=float(ollama_timeout or OLLAMA_DEFAULT_TIMEOUT),
            ollama_num_ctx=int(ollama_num_ctx or OLLAMA_DEFAULT_NUM_CTX),
            max_image_dim=int(max_image_dim or DEFAULT_MAX_IMAGE_DIM),
            progress=progress,
        )

    provider_selector.change(
        update_provider_visibility,
        inputs=[provider_selector],
        outputs=[
            api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector,
            ollama_refresh_button, ollama_num_ctx_input, ollama_timeout_input,
            detail_level, max_tokens_input,
            detail_level_dataset, max_tokens_input_dataset
        ],
        queue=False,
        show_progress="hidden",
    ).then(
        auto_refresh_on_ollama,
        inputs=[provider_selector, ollama_url_input],
        outputs=[ollama_model_selector],
        show_progress="minimal",
    )

    ollama_refresh_button.click(
        refresh_ollama_models,
        inputs=[ollama_url_input],
        outputs=[ollama_model_selector],
        show_progress="minimal",
    )

    clear_button.click(clear_fields, inputs=[], outputs=[output, history_table, history_state])
    folder_button.click(get_folder_path, outputs=folder_path_dataset, show_progress="hidden", inputs=[folder_path_dataset])

    submit_button.click(
        on_single_image_submit,
        inputs=[
            history_state, provider_selector, api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector, ollama_num_ctx_input,
            ollama_timeout_input, image_size_slider, image_input, prompt_input,
            detail_level, max_tokens_input
        ],
        outputs=[output, history_table, history_state]
    )

    cancel_button.click(cancel_processing, inputs=[], outputs=[processing_results_output])

    submit_button_dataset.click(
        on_batch_submit,
        inputs=[
            provider_selector, api_key_input, ollama_url_input,
            openai_model_selector, ollama_model_selector, ollama_num_ctx_input,
            ollama_timeout_input, image_size_slider, folder_path_dataset,
            prompt_input_dataset, detail_level_dataset, max_tokens_input_dataset,
            pre_prompt_input, post_prompt_input, worker_slider
        ],
        outputs=[processing_results_output]
    )


if __name__ == "__main__":
    def _sig_handler(signum, frame):
        print("\n🛑 Ctrl+C received. Terminating all processes immediately...")
        stop_all_processing()
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    try:
        app.launch(theme=gr.themes.Soft())
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt caught. Exiting...")
    finally:
        stop_all_processing()
        os._exit(0)
