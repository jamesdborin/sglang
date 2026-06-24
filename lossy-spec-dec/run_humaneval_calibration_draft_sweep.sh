#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-.venv-lossy-calibrate/bin/python}"
export LD_LIBRARY_PATH="$PWD/.venv-lossy-calibrate/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

for DRAFT_SIZE in 8 12 16; do
  RUN_DIR="$PWD/lossy-spec-dec/runs/calibration/humaneval/${STAMP}-draft${DRAFT_SIZE}"
  mkdir -p "$RUN_DIR"

  {
    echo "#!/usr/bin/env bash"
    echo "cd '$PWD'"
    echo "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
    printf "%q " "$PYTHON" "lossy-spec-dec/calibrate_dataset.py" \
      "--model-path" "cyankiwi/Qwen3.5-4B-AWQ-4bit" \
      "--speculative-draft-model-path" "z-lab/Qwen3.5-4B-DFlash" \
      "--dataset" "humaneval" \
      "--run-dir" "$RUN_DIR" \
      "--server-timeout" "1200" \
      "--bins" "30" \
      "--speculative-num-draft-tokens" "$DRAFT_SIZE" \
      "--server-args=--disable-cuda-graph"
    echo
  } > "$RUN_DIR/command.sh"
  chmod +x "$RUN_DIR/command.sh"

  echo
  echo "=== Running HumanEval calibration with draft size ${DRAFT_SIZE} ==="
  echo "Run dir: $RUN_DIR"
  "$RUN_DIR/command.sh" 2>&1 | tee "$RUN_DIR/sglang.log"

  mkdir -p "$RUN_DIR/analysis"
  echo
  echo "=== Analyzing draft size ${DRAFT_SIZE} ==="
  "$PYTHON" lossy-spec-dec/analyze_calibration.py \
    "$RUN_DIR/calibration.jsonl" \
    --output-dir "$RUN_DIR/analysis" \
    --bins 30 \
    2>&1 | tee "$RUN_DIR/analysis/analyze.log"
done
