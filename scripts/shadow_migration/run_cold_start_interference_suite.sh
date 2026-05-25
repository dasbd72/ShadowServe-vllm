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

for cpus in 8 16 32 64 128; do
  run_one "cold_start_concurrent_qwen3-0.6b_cpus${cpus}.log" \
    --mode concurrent --workload-cpus "$cpus" --model Qwen/Qwen3-0.6B --num-requests 4 --input-len 1024
done

for cpus in 8 16 32 64 128; do
  run_one "cold_start_concurrent_qwen3-8b_cpus${cpus}.log" \
    --mode concurrent --workload-cpus "$cpus" --model Qwen/Qwen3-8B --num-requests 4 --input-len 1024
done

for cpus in 1 2 4 8 16 32; do
  run_one "cold_start_mem_bw_qwen3-0.6b_cpus${cpus}.log" \
    --mode mem_bw --workload-cpus "$cpus" --model Qwen/Qwen3-0.6B
done

for cpus in 1 2 4 8 16 32; do
  run_one "cold_start_mem_bw_qwen3-8b_cpus${cpus}.log" \
    --mode mem_bw --workload-cpus "$cpus" --model Qwen/Qwen3-8B
done

echo "All runs complete. Logs in ${ROOT}/logs/"
