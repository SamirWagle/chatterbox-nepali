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
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "waglesameer5/chatterbox-nepali")
BASE_DIR = Path(__file__).resolve().parent
BUILTIN_VOICES = {
    "Benisha sample": BASE_DIR / "assets" / "benisha_ref_0012_16k.wav",
    "Achyut sample": BASE_DIR / "assets" / "achyut_ref_10s.wav",
}
REFERENCE_CHOICES = ["Benisha sample", "Achyut sample", "Upload audio", "Record microphone"]


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
    reference_source: str,
    uploaded_audio: str,
    recorded_audio: str,
    exaggeration: float,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
):
    if not text or not text.strip():
        raise gr.Error("Please enter Nepali text.")

    ref_audio = None
    if reference_source in BUILTIN_VOICES:
        ref_audio = str(BUILTIN_VOICES[reference_source])
    elif reference_source == "Upload audio":
        ref_audio = uploaded_audio
    elif reference_source == "Record microphone":
        ref_audio = recorded_audio

    if not ref_audio:
        raise gr.Error("Please choose, upload, or record a clean 5-10 second reference voice.")
    if not Path(ref_audio).exists():
        raise gr.Error("Reference audio file was not found. Please try uploading or recording again.")

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


def update_reference_inputs(reference_source: str):
    builtin_path = BUILTIN_VOICES.get(reference_source)
    return (
        gr.update(visible=reference_source == "Upload audio"),
        gr.update(visible=reference_source == "Record microphone"),
        gr.update(value=str(builtin_path) if builtin_path else None, visible=builtin_path is not None),
    )


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
            reference_source = gr.Dropdown(
                choices=REFERENCE_CHOICES,
                value="Benisha sample",
                label="Reference voice source",
            )
            builtin_preview = gr.Audio(
                label="Selected sample voice",
                value=str(BUILTIN_VOICES["Benisha sample"]),
                type="filepath",
                interactive=False,
            )
            uploaded_audio = gr.Audio(
                label="Upload reference voice",
                sources=["upload"],
                type="filepath",
                format="wav",
                visible=False,
            )
            recorded_audio = gr.Audio(
                label="Record reference voice",
                sources=["microphone"],
                type="filepath",
                format="wav",
                visible=False,
            )

            with gr.Accordion("Advanced settings", open=False):
                exaggeration = gr.Slider(0.0, 1.0, value=0.5, label="Exaggeration")
                temperature = gr.Slider(0.1, 1.2, value=0.75, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
                repetition_penalty = gr.Slider(1.0, 1.8, value=1.25, label="Repetition penalty")
                seed = gr.Number(value=0, precision=0, label="Seed")

            button = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Audio(label="Generated audio")

    gr.Examples(
        examples=[
            ["नमस्ते, म नेपाली आवाजको गुणस्तर परीक्षण गर्दैछु।", "Benisha sample"],
            ["आज काठमाडौँमा मौसम सफा छ, तर बेलुका हल्का चिसो बढ्न सक्छ।", "Benisha sample"],
            ["रेडियो समाचार पढ्दा गति धेरै छिटो पनि हुनु हुँदैन, धेरै ढिलो पनि हुनु हुँदैन।", "Achyut sample"],
        ],
        inputs=[text, reference_source],
    )

    reference_source.change(
        update_reference_inputs,
        inputs=[reference_source],
        outputs=[uploaded_audio, recorded_audio, builtin_preview],
    )

    button.click(
        generate,
        inputs=[
            text,
            reference_source,
            uploaded_audio,
            recorded_audio,
            exaggeration,
            temperature,
            top_p,
            repetition_penalty,
            seed,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
