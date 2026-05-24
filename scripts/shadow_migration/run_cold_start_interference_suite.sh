#!/usr/bin/env bash
# Run cold_start_interference benchmarks sequentially; one log per run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/scripts/shadow_migration/cold_start_interference.py"

run_one() {
  local log_name="$1"
  shift
  local log_path="logs/${log_name}"
  echo "==> $(date -Is) starting -> ${log_path}"
  "$PYTHON" "$SCRIPT" "$@" >"$log_path" 2>&1
  echo "==> $(date -Is) finished -> ${log_path}"
}

run_one cold_start_baseline_qwen3-0.6b.log \
  --mode baseline --model Qwen/Qwen3-0.6B

run_one cold_start_baseline_qwen3-8b.log \
  --mode baseline --model Qwen/Qwen3-8B

run_one cold_start_concurrent_qwen3-0.6b_cpus16.log \
  --mode concurrent --shadow-cpus 16 --model Qwen/Qwen3-0.6B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-0.6b_cpus32.log \
  --mode concurrent --shadow-cpus 32 --model Qwen/Qwen3-0.6B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-0.6b_cpus64.log \
  --mode concurrent --shadow-cpus 64 --model Qwen/Qwen3-0.6B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-0.6b_cpus128.log \
  --mode concurrent --shadow-cpus 128 --model Qwen/Qwen3-0.6B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-8b_cpus16.log \
  --mode concurrent --shadow-cpus 16 --model Qwen/Qwen3-8B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-8b_cpus32.log \
  --mode concurrent --shadow-cpus 32 --model Qwen/Qwen3-8B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-8b_cpus64.log \
  --mode concurrent --shadow-cpus 64 --model Qwen/Qwen3-8B --num-requests 4 --input-len 1024

run_one cold_start_concurrent_qwen3-8b_cpus128.log \
  --mode concurrent --shadow-cpus 128 --model Qwen/Qwen3-8B --num-requests 4 --input-len 1024

echo "All runs complete. Logs in ${ROOT}/logs/"
