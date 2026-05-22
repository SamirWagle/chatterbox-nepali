#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Chatterbox Nepali training tensors.")
    parser.add_argument("--manifest", required=True, help="Input jsonl with audio_path and text.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-sample .pt cache files.")
    parser.add_argument("--output-manifest", required=True, help="Output jsonl with cache_path entries.")
    parser.add_argument("--device", default="cpu", help="Model load device for feature extraction.")
    parser.add_argument("--max-items", type=int, default=0, help="Limit items, 0 means all.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before limiting.")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    in_manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_manifest = Path(args.output_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with in_manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)
    if args.max_items > 0:
        rows = rows[: args.max_items]

    model = ChatterboxMultilingualTTS.from_pretrained(args.device)
    tokenizer = model.tokenizer
    s3_tokenizer = model.s3gen.tokenizer.cpu()
    voice_encoder = model.ve.cpu()

    cached_rows = []
    tmp_manifest = out_manifest.with_suffix(out_manifest.suffix + ".tmp")

    for idx, item in enumerate(tqdm(rows, desc="cache")):
        cache_path = out_dir / f"{idx:08d}.pt"
        if cache_path.exists() and not args.overwrite:
            cached_rows.append({"cache_path": str(cache_path)})
            continue

        audio_path = item["audio_path"]
        text = item["text"]

        text_tokens = tokenizer.text_to_tokens(text, language_id="ne", lowercase=False).squeeze(0)
        text_tokens = F.pad(text_tokens, (1, 0), value=255)
        text_tokens = F.pad(text_tokens, (0, 1), value=0).long()

        wav, _ = librosa.load(audio_path, sr=S3_SR)
        with torch.no_grad():
            speech_tokens, _ = s3_tokenizer.forward([wav])
            speech_tokens = speech_tokens.squeeze(0).long()
            speaker_emb = torch.from_numpy(voice_encoder.embeds_from_wavs([wav], sample_rate=S3_SR))
            speaker_emb = speaker_emb.mean(axis=0, keepdim=True).float()

        tmp_cache = cache_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "text_tokens": text_tokens.cpu(),
                "speech_tokens": speech_tokens.cpu(),
                "speaker_emb": speaker_emb.cpu(),
            },
            tmp_cache,
        )
        os.replace(tmp_cache, cache_path)
        cached_rows.append({"cache_path": str(cache_path)})

        if idx % 250 == 0:
            with tmp_manifest.open("w", encoding="utf-8") as out:
                for row in cached_rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

    with tmp_manifest.open("w", encoding="utf-8") as out:
        for row in cached_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_manifest, out_manifest)
    print(f"wrote {len(cached_rows)} cached rows -> {out_manifest}")


if __name__ == "__main__":
    main()
