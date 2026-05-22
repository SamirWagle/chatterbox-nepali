import argparse
import torch
import soundfile as sf
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def main():
    parser = argparse.ArgumentParser(description="Standalone Nepali TTS Tester (Will NOT interfere with training)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to t3_nepali_epoch_X.pt")
    parser.add_argument("--text", type=str, required=True, help="Nepali text to synthesize")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--output", type=str, default="test_sample.wav", help="Output WAV path")
    parser.add_argument("--device", type=str, default="cpu", help="Forces CPU to keep your GPU free for the training process")
    parser.add_argument("--temperature", type=float, default=0.8, help="Randomness (0.1-1.0)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Higher = less repetition (1.0 = off)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    args = parser.parse_args()
    
    print(f"Loading Chatterbox firmly on {args.device} to avoid crashing your active training...")
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(args.device)
    
    print(f"Loading weights from: {args.checkpoint}")
    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        resume_state = load_file(args.checkpoint, device="cpu")
    else:
        resume_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    
    cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
    model_wrapper.t3.load_state_dict(cleaned_state, strict=False)
    
    model_wrapper.t3.to(args.device).eval()
    model_wrapper.s3gen.tokenizer.to(args.device)
    model_wrapper.ve.to(args.device)
    
    print(f"Synthesizing: '{args.text}'")
    with torch.inference_mode():
        try:
            val_wav = model_wrapper.generate(
                args.text, 
                language_id="ne", 
                audio_prompt_path=args.ref_audio,
                exaggeration=0.5,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p
            )
            
            audio = val_wav.detach().cpu()
            if audio.ndim == 2:
                audio = audio.squeeze(0)
            sf.write(args.output, audio.numpy(), model_wrapper.sr)
            print(f"✅ Success! Audio saved to {args.output}")
            
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
