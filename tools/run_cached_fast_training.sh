#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
MAX_ITEMS="${MAX_ITEMS:-12000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-31}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
GPU_LIMIT_MIB="${GPU_LIMIT_MIB:-95000}"
AVAIL_LIMIT_GIB="${AVAIL_LIMIT_GIB:-24}"
RESUME_CKPT="${RESUME_CKPT:-${PROJECT_DIR}/t3_nepali_epoch_20.pt}"
EVAL_AUDIO_PATH="${EVAL_AUDIO_PATH:-}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/nepali_cleaned/manifest_22050_3to10.jsonl}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-${PROJECT_DIR}/cuda_deps}"

export PYTHONPATH="${PROJECT_DIR}/src:${EXTRA_PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs data/nepali_cleaned/cache_12k

CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/data/nepali_cleaned/cache_12k}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${PROJECT_DIR}/data/nepali_cleaned/cache_12k_manifest.jsonl}"
CACHE_LOG="${CACHE_LOG:-${PROJECT_DIR}/logs/cache_12k_${RUN_ID}.log}"
TRAIN_LOG="${TRAIN_LOG:-${PROJECT_DIR}/logs/train_cached_12k_${RUN_ID}.log}"
WATCH_LOG="${WATCH_LOG:-${PROJECT_DIR}/logs/train_cached_12k_${RUN_ID}_watchdog.log}"

existing_rows=0
if [ -f "${CACHE_MANIFEST}" ]; then
  existing_rows="$(wc -l < "${CACHE_MANIFEST}")"
fi

if [ "${existing_rows}" -lt "${MAX_ITEMS}" ]; then
  echo "cache build start $(date -Is), existing_rows=${existing_rows}, target=${MAX_ITEMS}" | tee -a "${CACHE_LOG}"
  python3 tools/build_nepali_feature_cache.py \
    --manifest "${MANIFEST}" \
    --output-dir "${CACHE_DIR}" \
    --output-manifest "${CACHE_MANIFEST}" \
    --device cpu \
    --max-items "${MAX_ITEMS}" \
    --shuffle \
    2>&1 | tee -a "${CACHE_LOG}"
else
  echo "cache already ready: ${CACHE_MANIFEST} (${existing_rows} rows)" | tee -a "${CACHE_LOG}"
fi

echo "training start $(date -Is)" | tee -a "${TRAIN_LOG}"
train_args=(
  src/chatterbox/train_nepali.py
  --manifest "${CACHE_MANIFEST}"
  --device cuda
  --resume_t3_weights "${RESUME_CKPT}"
  --batch_size "${BATCH_SIZE}"
  --accum_steps "${ACCUM_STEPS}"
  --num_workers 1
  --epochs "${TARGET_EPOCHS}"
  --lr "${LR}"
  --save_every 1
)

if [ -n "${EVAL_AUDIO_PATH}" ]; then
  train_args+=(--eval_audio_path "${EVAL_AUDIO_PATH}")
fi

setsid env \
  PYTHONPATH="${PYTHONPATH}" \
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  python3 "${train_args[@]}" \
  >> "${TRAIN_LOG}" 2>&1 < /dev/null &

train_pid="$!"
setsid tools/memory_watchdog.sh "${train_pid}" "${WATCH_LOG}" "${GPU_LIMIT_MIB}" "${AVAIL_LIMIT_GIB}" >/dev/null 2>&1 < /dev/null &
watch_pid="$!"
echo "TRAIN_PID=${train_pid} WATCH_PID=${watch_pid} TRAIN_LOG=${TRAIN_LOG} WATCH_LOG=${WATCH_LOG}" | tee -a "${TRAIN_LOG}"

set +e
wait "${train_pid}"
status="$?"
kill -TERM "${watch_pid}" 2>/dev/null || true
echo "training stop $(date -Is), status=${status}" | tee -a "${TRAIN_LOG}"
exit "${status}"
