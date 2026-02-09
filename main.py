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
rate_limiter = RateLimiter(max_calls=10, period=60)


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


def generate_description(api_key, image, prompt, detail, max_tokens, model="gpt-4o-mini"):
    """Generate image description using OpenAI's vision API."""
    
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
        return f"Error: Rate limit exceeded. Please wait and try again."
    
    except APIConnectionError as e:
        # Network error - log concisely
        log_error(e, "CONNECTION", include_traceback=False)
        return f"Error: Connection failed. Check your internet connection."
    
    except APIError as e:
        # API error - log with some detail but no full traceback
        log_error(e, "API_ERROR", include_traceback=False)
        return f"Error: API error - {str(e)}"
    
    except Exception as e:
        # Unexpected error - log everything for debugging
        log_error(e, "UNEXPECTED", include_traceback=True)
        return f"Error: {str(e)}"


# History tracking
history = []
columns = ["Time", "Prompt", "GPT4-Vision Caption"]


def clear_fields():
    """Clear all input fields and history."""
    global history
    history = []
    return "", []


def update_history(prompt, response):
    """Add entry to history and return formatted table data."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    history.append({"Time": timestamp, "Prompt": prompt, "GPT4-Vision Caption": response})
    return [[entry[column] for column in columns] for entry in history]


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
    def __init__(self):
        self.is_processing = False
        self.lock = Lock()
    
    def start(self):
        with self.lock:
            self.is_processing = True
    
    def stop(self):
        with self.lock:
            self.is_processing = False
    
    def is_active(self):
        with self.lock:
            return self.is_processing


processing_control = ProcessingControl()


def process_folder(api_key, folder_path, prompt, detail, max_tokens, model, pre_prompt="", post_prompt="",
                   progress=gr.Progress(), num_workers=2):
    """Process all images in a folder with batch captioning."""
    processing_control.start()

    if not os.path.isdir(folder_path):
        processing_control.stop()
        return f"Error: No such directory: {folder_path}"

    # Get list of image files
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not file_list:
        processing_control.stop()
        return f"No image files found in {folder_path}"
    
    progress(0, desc="Starting batch processing...")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    def process_file(file):
        """Process a single image file."""
        nonlocal processed_count, skipped_count, error_count
        
        if not processing_control.is_active():
            return "canceled"

        image_path = os.path.join(folder_path, file)
        txt_path = os.path.join(folder_path, os.path.splitext(file)[0] + ".txt")

        # Skip if caption file already exists
        if os.path.exists(txt_path):
            skipped_count += 1
            return "skipped"

        try:
            description = generate_description(api_key, image_path, prompt, detail, max_tokens, model)
            
            # Check if description is an error
            if description.startswith("Error:"):
                error_count += 1
                # Log which file failed
                with open("error_log.txt", 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Failed to process: {file} - {description}\n")
                return "error"
            
            # Format final caption with pre/post prompts
            final_caption = format_caption(pre_prompt, description, post_prompt)
            
            # Write caption to file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(final_caption)
            
            processed_count += 1
            return "success"
            
        except Exception as e:
            error_count += 1
            log_error(e, f"BATCH_PROCESSING ({file})", include_traceback=True)
            return "error"

    # Process files with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(process_file, file_list), 1):
            progress((i, len(file_list)), desc=f"Processing: {i}/{len(file_list)}")
            if not processing_control.is_active():
                break

    processing_control.stop()
    
    # Generate results summary
    summary = f"Batch processing complete:\n"
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
    gr.Markdown("# GPT-4 Vision Image Captioner")
    gr.Markdown("Generate captions for single images or batch process entire folders.")
    
    with gr.Row():
        api_key_input = gr.Textbox(
            scale=3,
            label="OpenAI API Key", 
            placeholder="sk-...", 
            type="password",
            info="Your OpenAI API key. Rate limited to prevent quota exhaustion."
        )
        model_selector = gr.Dropdown(
            scale=1,
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            value="gpt-4o-mini",
            label="Model",
            info="gpt-4o-mini: best value, gpt-4o: highest quality"
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
                label="Detail Level", 
                value="auto",
                info="High = more tokens, better detail"
            )
            max_tokens_input = gr.Number(
                scale=1, 
                value=300, 
                label="Max Tokens",
                minimum=50,
                maximum=1000
            )
        
        submit_button = gr.Button("Generate Caption", variant="primary")
        output = gr.Textbox(label="Generated Caption", lines=5)
        
        with gr.Accordion("History", open=False):
            history_table = gr.Dataframe(headers=columns, interactive=False)
            clear_button = gr.Button("Clear History")

    with gr.Tab("Batch Processing"):
        gr.Markdown("Process all images in a folder. Creates .txt files with captions next to each image.")
        
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
                label="Detail Level", 
                value="auto"
            )
            max_tokens_input_dataset = gr.Number(
                scale=1, 
                value=300, 
                label="Max Tokens",
                minimum=50,
                maximum=1000
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
                info="More workers = faster, but may hit rate limits. Recommended: 2-3"
            )
        
        with gr.Row():
            submit_button_dataset = gr.Button("Start Batch Processing", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel", variant="stop", scale=1)
        
        processing_results_output = gr.Textbox(label="Processing Results", lines=5)

    # Event handlers
    def cancel_processing():
        """Cancel ongoing batch processing."""
        processing_control.stop()
        return "⚠️ Processing canceled by user"

    def on_single_image_submit(api_key, model, image, prompt, detail, max_tokens):
        """Handle single image caption generation."""
        if not api_key.strip():
            raise gr.Error("Please enter your OpenAI API key")
        
        # Check for common API key issues
        if '\xa0' in api_key or any(ord(c) >= 128 for c in api_key):
            raise gr.Error("API key contains invalid characters. Please re-copy your API key (avoid copying from PDFs or formatted documents).")
        
        if image is None:
            raise gr.Error("Please upload an image")
        
        description = generate_description(api_key, image, prompt, detail, max_tokens, model)
        new_history = update_history(prompt, description)
        return description, new_history

    def on_batch_submit(api_key, model, folder_path, prompt, detail, max_tokens, pre_prompt, post_prompt, num_workers):
        """Handle batch folder processing."""
        if not api_key.strip():
            raise gr.Error("Please enter your OpenAI API key")
        
        # Check for common API key issues
        if '\xa0' in api_key or any(ord(c) >= 128 for c in api_key):
            raise gr.Error("API key contains invalid characters. Please re-copy your API key (avoid copying from PDFs or formatted documents).")
        
        if not folder_path.strip():
            raise gr.Error("Please enter a folder path")
        
        result = process_folder(
            api_key, folder_path, prompt, detail, max_tokens, model,
            pre_prompt, post_prompt, num_workers=int(num_workers)
        )
        return result

    # Wire up events
    clear_button.click(clear_fields, inputs=[], outputs=[output, history_table])
    
    folder_button.click(get_folder_path, outputs=folder_path_dataset, show_progress="hidden")
    
    submit_button.click(
        on_single_image_submit,
        inputs=[api_key_input, model_selector, image_input, prompt_input, detail_level, max_tokens_input],
        outputs=[output, history_table]
    )
    
    cancel_button.click(cancel_processing, inputs=[], outputs=[processing_results_output])
    
    submit_button_dataset.click(
        on_batch_submit,
        inputs=[
            api_key_input,
            model_selector,
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
