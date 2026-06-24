#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)-draft5-deciles}"
RUN_DIR="$PWD/lossy-spec-dec/runs/threshold-eval/humaneval/$RUN_ID"
mkdir -p "$RUN_DIR"

export LD_LIBRARY_PATH="$PWD/.venv-lossy-calibrate/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

THRESHOLDS="273332.96875,456.93878173828125,53.510555267333984,3.224769115447998,1.4434094429016113,1.167620301246643,1.0599793195724487,1.0148314237594604,1.0029233694076538,1.0003074407577515,1.0000004768371582"
EVAL_TEMPLATE=".venv-lossy-calibrate/bin/python lossy-spec-dec/run_sglang_humaneval_eval.py --task {task} --base-url {base_url} --model {model_id} --output-dir {output_dir} --num-threads 8 --max-tokens 512 --api chat"

.venv-lossy-calibrate/bin/python lossy-spec-dec/run_threshold_eval_sweep.py \
  --thresholds "$THRESHOLDS" \
  --output-dir "$RUN_DIR" \
  --tasks humaneval \
  --model-path cyankiwi/Qwen3.5-4B-AWQ-4bit \
  --model-id cyankiwi/Qwen3.5-4B-AWQ-4bit \
  --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 5 \
  --server-timeout 1200 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --server-args='--disable-cuda-graph' \
  --nemo-command-template "$EVAL_TEMPLATE" \
  2>&1 | tee "$RUN_DIR/sweep.log"

.venv-lossy-calibrate/bin/python lossy-spec-dec/plot_threshold_eval_sweep.py \
  --input "$RUN_DIR/runs.jsonl" \
  --output-dir "$RUN_DIR/plots" \
  2>&1 | tee "$RUN_DIR/plot.log"

echo "Run directory: $RUN_DIR"
