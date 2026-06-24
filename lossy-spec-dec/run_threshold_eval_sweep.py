#!/usr/bin/env python3
"""Run lossy-spec evaluation once per perplexity threshold and time each run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SWEEP = REPO_ROOT / "lossy-spec-dec" / "eval_sweep.py"


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--thresholds must contain at least one value")
    for value in values:
        if value < 1.0:
            raise ValueError(f"Perplexity thresholds must be >= 1.0: {value}")
    return values


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_label(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def build_eval_command(args: argparse.Namespace, threshold: float, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SWEEP),
        "--mode",
        "sweep",
        "--thresholds",
        f"{threshold:g}",
        "--output-dir",
        str(output_dir),
        "--tasks",
        args.tasks,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--server-timeout",
        str(args.server_timeout),
        "--nemo-command-template",
        args.nemo_command_template,
    ]
    if args.base_url:
        command += ["--base-url", args.base_url]
    else:
        command += [
            "--model-path",
            args.model_path,
            "--speculative-draft-model-path",
            args.speculative_draft_model_path,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    if args.model_id:
        command += ["--model-id", args.model_id]
    if args.speculative_num_steps is not None:
        command += ["--speculative-num-steps", str(args.speculative_num_steps)]
    if args.speculative_num_draft_tokens is not None:
        command += [
            "--speculative-num-draft-tokens",
            str(args.speculative_num_draft_tokens),
        ]
    if args.server_args:
        command += [f"--server-args={args.server_args}"]
    return command


def run_threshold(args: argparse.Namespace, threshold: float) -> list[dict[str, Any]]:
    threshold_dir = args.output_dir / f"threshold_{threshold_label(threshold)}"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    command = build_eval_command(args, threshold, threshold_dir)
    command_path = threshold_dir / "command.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        + " ".join(shlex.quote(part) for part in command)
        + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)

    started_at = utc_now()
    start = time.perf_counter()
    env = os.environ.copy()
    if args.ld_library_path_prefix:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            args.ld_library_path_prefix
            if not existing
            else f"{args.ld_library_path_prefix}:{existing}"
        )
    with (threshold_dir / "eval_sweep.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    finished_at = utc_now()

    rows = list(iter_jsonl(threshold_dir / "runs.jsonl"))
    if not rows:
        rows = [{"threshold": threshold, "task": None}]
    for row in rows:
        row.update(
            {
                "threshold": threshold,
                "elapsed_seconds": elapsed,
                "started_at": started_at,
                "finished_at": finished_at,
                "returncode": proc.returncode,
                "eval_output_dir": str(threshold_dir),
                "eval_log": str(threshold_dir / "eval_sweep.log"),
                "command": " ".join(shlex.quote(part) for part in command),
            }
        )
    if proc.returncode != 0 and not args.keep_going:
        write_jsonl(args.output_dir / "runs.jsonl", rows)
        write_csv(args.output_dir / "summary.csv", rows)
        raise RuntimeError(
            f"Threshold {threshold:g} failed with exit code {proc.returncode}. "
            f"See {threshold_dir / 'eval_sweep.log'}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", required=True, help="Comma-separated perplexity thresholds, e.g. 1,1.01,1.1,2,10")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "lossy-spec-dec" / "runs" / "threshold-eval" / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--tasks", default="gsm8k,humaneval")
    parser.add_argument("--base-url", help="Use an already-running SGLang endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", default="cyankiwi/Qwen3.5-4B-AWQ-4bit")
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--speculative-draft-model-path", default="z-lab/Qwen3.5-4B-DFlash")
    parser.add_argument("--speculative-num-steps", type=int)
    parser.add_argument("--speculative-num-draft-tokens", type=int)
    parser.add_argument("--server-args")
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--nemo-command-template", required=True)
    parser.add_argument("--ld-library-path-prefix", default=str(REPO_ROOT / ".venv-lossy-calibrate/lib/python3.12/site-packages/nvidia/cu13/lib"))
    parser.add_argument("--keep-going", action="store_true", help="Continue after a threshold fails.")
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for threshold in parse_thresholds(args.thresholds):
        print(f"Running threshold {threshold:g}...")
        rows = run_threshold(args, threshold)
        all_rows.extend(rows)
        write_jsonl(args.output_dir / "runs.jsonl", all_rows)
        write_csv(args.output_dir / "summary.csv", all_rows)
        print(f"  wrote {len(rows)} row(s)")
    print(f"Wrote combined results: {args.output_dir / 'runs.jsonl'}")
    print(f"Wrote combined summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
