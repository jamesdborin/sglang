#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-lossy-calibrate}"

uv venv "$VENV_DIR" --python 3.12
uv pip install --python "$VENV_DIR/bin/python" -e "$ROOT_DIR/python"

cat <<EOF
Calibration environment is ready:
  source "$VENV_DIR/bin/activate"
  export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH:-}"

Example:
  python "$ROOT_DIR/lossy-spec-dec/calibrate_dataset.py" \\
    --model-path cyankiwi/Qwen3.5-4B-AWQ-4bit \\
    --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash \\
    --dataset gsm8k
EOF
