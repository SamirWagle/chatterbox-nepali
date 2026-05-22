#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import soundfile as sf

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


DEFAULT_TEXTS: list[str] = [
    "नमस्ते। यो नेपाली च्याटरबक्स टीटीएसको परीक्षण आवाज हो।",
    "आजको मौसम सफा छ र हल्का बतास चलिरहेको छ।",
    "कृपया आफ्नो काम समयमै गर्नुहोस् र आराम गर्न पनि नबिर्सनुहोस्।",
    "नेपाल हिमाल, पहाड र तराईको सुन्दर देश हो, जहाँ विविध संस्कृति र भाषा पाइन्छ।",
    "मलाई पढ्न मन पर्छ, विशेष गरी इतिहास र विज्ञानसम्बन्धी किताबहरू।",
    "यदि तपाईंलाई यात्रा मन पर्छ भने पोखरा, लुम्बिनी र इलाम अवश्य घुम्नुहोस्।",
    "कम्प्युटरले हाम्रो दैनिक जीवनलाई धेरै सजिलो बनाइदिएको छ, तर सही तरिकाले प्रयोग गर्नुपर्छ।",
    "कफी भन्दा चिया पिउने बानी धेरैलाई हुन्छ, तर धेरै तातो चिया स्वास्थ्यका लागि राम्रो हुँदैन।",
    "भोलि बिहानै उठेर व्यायाम गर्छु भनेर सोच्नु राम्रो हो, तर त्यसलाई निरन्तरता दिनु अझ राम्रो हो।",
    "इन्द्रेणी वा इन्द्रधनुष प्रकाश र रंगबाट उत्पन्न भएको यस्तो घटना हो जसमा रंगीन प्रकाशको एउटा अर्धवृत आकाशमा देखिन्छ।",
]


def load_checkpoint_into(model_wrapper: ChatterboxMultilingualTTS, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        resume_state = load_file(str(checkpoint_path), device="cpu")
    else:
        resume_state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
    model_wrapper.t3.load_state_dict(cleaned_state, strict=False)


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    audio = wav.detach().cpu()
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    sf.write(str(path), audio.numpy(), sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Nepali TTS inference (loads model once, generates multiple WAVs)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to t3_nepali_epoch_20.pt (or .safetensors)")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference WAV (5-10s recommended)")
    parser.add_argument("--out_dir", type=str, default="batch_outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda:0 (use cpu if GPU is busy)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--texts_file", type=str, default="", help="Optional: newline-separated texts file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ref_audio_path = Path(args.ref_audio).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not ref_audio_path.exists():
        raise SystemExit(f"ref_audio not found: {ref_audio_path}")

    texts = DEFAULT_TEXTS
    if args.texts_file:
        tf = Path(args.texts_file).expanduser().resolve()
        if not tf.exists():
            raise SystemExit(f"texts_file not found: {tf}")
        texts = [line.strip() for line in tf.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not texts:
            raise SystemExit("texts_file is empty")

    print(f"Loading Chatterbox on {args.device}...")
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(args.device)

    print(f"Loading weights from: {checkpoint_path}")
    load_checkpoint_into(model_wrapper, checkpoint_path)

    model_wrapper.t3.to(args.device).eval()
    model_wrapper.s3gen.tokenizer.to(args.device)
    model_wrapper.ve.to(args.device)

    index_path = out_dir / "index.txt"
    with index_path.open("w", encoding="utf-8") as idx:
        idx.write(f"checkpoint={checkpoint_path}\n")
        idx.write(f"ref_audio={ref_audio_path}\n")
        idx.write(f"device={args.device}\n")
        idx.write("\n")

        for i, text in enumerate(texts, start=1):
            out_wav = out_dir / f"sample_{i:02d}.wav"
            idx.write(f"sample_{i:02d}.wav\t{text}\n")

            if out_wav.exists() and out_wav.stat().st_size > 0:
                print(f"[{i:02d}/{len(texts):02d}] exists -> {out_wav.name} (skip)")
                continue

            print(f"[{i:02d}/{len(texts):02d}] generating -> {out_wav.name}")
            try:
                with torch.inference_mode():
                    wav = model_wrapper.generate(
                        text,
                        language_id="ne",
                        audio_prompt_path=str(ref_audio_path),
                        exaggeration=float(args.exaggeration),
                        temperature=float(args.temperature),
                        repetition_penalty=float(args.repetition_penalty),
                        top_p=float(args.top_p),
                    )

                save_wav(out_wav, wav, model_wrapper.sr)
            except KeyboardInterrupt:
                print("\nInterrupted by user (Ctrl+C). Re-run to resume.")
                break
            except Exception as exc:
                print(f"⚠️ Failed on sample {i:02d}: {exc}")
                continue

    print(f"Done. Outputs in: {out_dir}")
    print(f"Index file: {index_path}")


if __name__ == "__main__":
    main()
