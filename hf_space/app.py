import os
import random
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILENAME = os.getenv("CHECKPOINT_FILENAME", "t3_mtl_nepali_final.safetensors")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "officialuser/chatterbox-nepali")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_checkpoint() -> Path:
    local_path = Path(CHECKPOINT_FILENAME)
    if local_path.exists():
        return local_path

    downloaded = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=CHECKPOINT_FILENAME,
        local_dir="model_cache",
    )
    return Path(downloaded)


def load_model() -> ChatterboxMultilingualTTS:
    model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    checkpoint_path = resolve_checkpoint()
    state = load_file(str(checkpoint_path), device="cpu")
    cleaned_state = {
        key.replace("patched_model.", "").replace("model.", ""): value
        for key, value in state.items()
    }
    model.t3.load_state_dict(cleaned_state, strict=False)
    model.t3.to(DEVICE).eval()
    model.s3gen.tokenizer.to(DEVICE)
    model.ve.to(DEVICE)
    return model


MODEL = None


def get_model() -> ChatterboxMultilingualTTS:
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


def generate(
    text: str,
    ref_audio: str,
    exaggeration: float,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
):
    if not text or not text.strip():
        raise gr.Error("Please enter Nepali text.")
    if not ref_audio:
        raise gr.Error("Please upload a clean 5-10 second reference voice.")

    if seed:
        set_seed(int(seed))

    model = get_model()
    with torch.inference_mode():
        wav = model.generate(
            text.strip(),
            language_id="ne",
            audio_prompt_path=ref_audio,
            exaggeration=float(exaggeration),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        )

    audio = wav.detach().cpu()
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    return (model.sr, audio.numpy())


with gr.Blocks(title="Chatterbox Nepali TTS") as demo:
    gr.Markdown("# Chatterbox Nepali TTS")
    gr.Markdown("Nepali text-to-speech with reference-audio voice cloning.")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Nepali text",
                lines=5,
                value="नमस्ते, म नेपाली आवाजको गुणस्तर परीक्षण गर्दैछु।",
            )
            ref_audio = gr.Audio(label="Reference voice", type="filepath")

            with gr.Accordion("Advanced settings", open=False):
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration")
                temperature = gr.Slider(0.1, 1.2, value=0.75, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
                repetition_penalty = gr.Slider(1.0, 1.8, value=1.25, label="Repetition penalty")
                seed = gr.Number(value=0, precision=0, label="Seed")

            button = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Audio(label="Generated audio")

    button.click(
        generate,
        inputs=[text, ref_audio, exaggeration, temperature, top_p, repetition_penalty, seed],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
