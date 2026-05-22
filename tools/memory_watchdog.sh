#!/usr/bin/env bash
set -u

train_pid="$1"
watch_log="$2"
gpu_limit_mib="${3:-85000}"
avail_limit_gib="${4:-22}"
avail_limit_kib=$((avail_limit_gib * 1024 * 1024))

echo "watchdog start $(date -Is) train_pid=${train_pid} gpu_limit_mib=${gpu_limit_mib} avail_limit_gib=${avail_limit_gib}" >> "${watch_log}"

while kill -0 "${train_pid}" 2>/dev/null; do
  gpu_mib="$(
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v pid="${train_pid}" '
          {
            gsub(/ /, "", $1);
            gsub(/ MiB/, "", $2);
            gsub(/ /, "", $2);
            if ($1 == pid) print $2;
          }' \
      | tail -1
  )"
  gpu_mib="${gpu_mib:-0}"
  avail_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  avail_gib="$(awk -v k="${avail_kib}" 'BEGIN{printf "%.1f", k/1024/1024}')"

  echo "$(date -Is) pid=${train_pid} gpu_mib=${gpu_mib} avail_gib=${avail_gib}" >> "${watch_log}"

  if [ "${gpu_mib}" -gt "${gpu_limit_mib}" ] || [ "${avail_kib}" -lt "${avail_limit_kib}" ]; then
    echo "$(date -Is) LIMIT_EXCEEDED killing process group -${train_pid}" >> "${watch_log}"
    kill -TERM -- "-${train_pid}" 2>/dev/null || kill -TERM "${train_pid}" 2>/dev/null
    sleep 10
    kill -KILL -- "-${train_pid}" 2>/dev/null || kill -KILL "${train_pid}" 2>/dev/null
    exit 2
  fi

  sleep 30
done

echo "watchdog stop $(date -Is) train_pid=${train_pid}" >> "${watch_log}"
