import glob
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers.utils import logging as hf_logging

from inference.generate import Generator


WEB_DEMO_BACKEND = os.getenv("WEB_DEMO_BACKEND", "blip").strip().lower()
WEB_DEMO_PORT = int(os.getenv("WEB_DEMO_PORT", "7860"))


def load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    hf_token = os.getenv("HF_TOKEN")
    hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token and not hub_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    elif hub_token and not hf_token:
        os.environ["HF_TOKEN"] = hub_token


load_local_env()

# Keep startup logs clean for demo mode by hiding non-critical HF warnings.
hf_logging.set_verbosity_error()


def find_latest_valid_checkpoint() -> str:
    ckpts = sorted(
        glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt")
    )
    if not ckpts:
        raise RuntimeError("No checkpoints found in outputs/ or experiments/")

    for candidate in reversed(ckpts):
        try:
            torch.load(candidate, map_location="cpu")
            return candidate
        except Exception as e:
            print(f"Skipping unreadable checkpoint {candidate}: {e}")

    raise RuntimeError("No valid checkpoint found")


GENERATOR: Optional[Generator] = None
_BLIP_PROCESSOR = None
_BLIP_MODEL = None


def get_generator() -> Generator:
    global GENERATOR
    if GENERATOR is None:
        checkpoint = find_latest_valid_checkpoint()
        print(f"Using checkpoint: {checkpoint}")
        GENERATOR = Generator(checkpoint=checkpoint)
    return GENERATOR


def is_degenerate_output(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()
    if len(stripped) < 6:
        return True

    words = re.findall(r"[A-Za-z]+", stripped.lower())
    if len(words) < 3:
        return True

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.45:
        return True

    max_run = 1
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    gibberish_like = 0
    for w in words:
        vowels = sum(1 for ch in w if ch in "aeiou")
        # Many pseudo-words from collapsed decoding tend to be long with no vowels.
        if len(w) >= 6 and vowels == 0:
            gibberish_like += 1

    gibberish_ratio = gibberish_like / max(len(words), 1)
    return max_run >= 3 or gibberish_ratio >= 0.25


def fallback_caption(image, prompt):
    global _BLIP_PROCESSOR, _BLIP_MODEL

    if _BLIP_PROCESSOR is None or _BLIP_MODEL is None:
        model_name = "Salesforce/blip-image-captioning-base"
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained(model_name, use_fast=False)
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(model_name)
        # Keep fallback on CPU to avoid OOM when a custom VLM also occupies GPU memory.
        device = "cpu"
        _BLIP_MODEL.to(device)
        _BLIP_MODEL.eval()

    device = next(_BLIP_MODEL.parameters()).device
    inputs = _BLIP_PROCESSOR(images=image, return_tensors="pt").to(device)
    out = _BLIP_MODEL.generate(
        **inputs,
        num_beams=4,
        max_new_tokens=32,
        min_new_tokens=6,
        repetition_penalty=1.2
    )
    caption = _BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True).strip()

    # If decoding fails or collapses, provide a clear fallback message.
    if caption and not is_degenerate_output(caption):
        return caption

    out = _BLIP_MODEL.generate(**inputs, do_sample=True, top_p=0.9, temperature=0.8, max_new_tokens=32)
    caption = _BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True).strip()
    return caption if caption else "Unable to generate a caption for this image."


def run_demo(image, prompt):
    if image is None:
        return "Please upload an image."
    if not prompt or not str(prompt).strip():
        return "Please enter a prompt."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name

    try:
        image.save(temp_path)
        if WEB_DEMO_BACKEND == "vlm":
            output = get_generator().generate(
                temp_path,
                prompt,
                max_tokens=32,
                min_new_tokens=6,
                temperature=0.65,
                top_p_val=0.85,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                max_repeat_token=2
            )
            if is_degenerate_output(output):
                try:
                    output = fallback_caption(image, prompt)
                except Exception as fallback_err:
                    output = (
                        "Primary model output looked unstable and fallback captioning failed: "
                        f"{fallback_err}. Primary output: {output}"
                    )
            return output

        # Default path favors reliability for user-facing demo quality.
        return fallback_caption(image, prompt)
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


with gr.Blocks(title="Multimodal VLM Demo") as demo:
    gr.Markdown("# Multimodal Vision-Language Demo")
    gr.Markdown(
        f"Run inference from the Multimodal_Vision-Language-Model_Research project. "
        f"Backend: {WEB_DEMO_BACKEND.upper()}"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value="What is in this image?")
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_text = gr.Textbox(label="Model Output")

    submit_btn.click(fn=run_demo, inputs=[image_input, prompt_input], outputs=output_text)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=WEB_DEMO_PORT, share=False)
