#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Chatterbox Nepali training manifest.jsonl from transcriptions.csv")
    p.add_argument(
        "--input-csv",
        required=True,
        help="Path to transcriptions.csv (must include header with audio_filepath,text,duration,language)",
    )
    p.add_argument("--output-jsonl", required=True, help="Where to write manifest.jsonl")
    p.add_argument("--min-dur", type=float, default=3.0, help="Minimum clip duration (seconds)")
    p.add_argument("--max-dur", type=float, default=10.0, help="Maximum clip duration (seconds)")
    p.add_argument("--language", default="ne", help="Keep rows where language matches (default: ne)")
    p.add_argument("--max-items", type=int, default=0, help="Limit number of rows (0 = no limit)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle rows before limiting")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed when shuffling")
    p.add_argument("--require-exists", action="store_true", help="Skip rows whose audio file is missing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"missing input csv: {input_csv}")

    rows: list[dict] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_filepath", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            audio_path = (row.get("audio_filepath") or "").strip()
            text = (row.get("text") or "").strip()
            if not audio_path or not text:
                continue

            if args.language:
                if (row.get("language") or "").strip() != args.language:
                    continue

            try:
                dur = float(row.get("duration") or 0.0)
            except Exception:
                dur = 0.0

            if dur and (dur < args.min_dur or dur > args.max_dur):
                continue

            if args.require_exists and not Path(audio_path).exists():
                continue

            # Training loader accepts absolute paths too.
            rows.append({"audio_path": audio_path, "text": text})

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)

    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as out:
        for item in rows:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.replace(tmp_path, out_path)

    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
