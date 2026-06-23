#!/usr/bin/env python3
"""Run DFlash lossy-spec calibration on benchmark prompts and print histograms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SWEEP_PATH = REPO_ROOT / "lossy-spec-dec" / "eval_sweep.py"


def load_eval_sweep():
    spec = importlib.util.spec_from_file_location("lossy_eval_sweep", EVAL_SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EVAL_SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_prompts(dataset: str, split: str, limit: int | None) -> list[str]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        rows = load_dataset("gsm8k", "main", split=split)
        prompts = [
            "Solve the following grade school math problem. Show the reasoning, then give the final answer.\n\n"
            + row["question"]
            for row in tqdm(rows, desc="Preparing GSM8K prompts")
        ]
    elif dataset == "humaneval":
        rows = None
        errors = []
        for name in ("openai/openai_humaneval", "openai_humaneval"):
            try:
                rows = load_dataset(name, split=split)
                break
            except Exception as exc:  # noqa: BLE001 - try the common aliases
                errors.append(f"{name}: {exc}")
        if rows is None:
            raise RuntimeError("Could not load HumanEval dataset:\n" + "\n".join(errors))
        prompts = [
            "Complete the following Python function.\n\n" + row["prompt"]
            for row in tqdm(rows, desc="Preparing HumanEval prompts")
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts loaded for {dataset} split {split}")
    return prompts


def request_generate(eval_sweep: Any, base_url: str, prompt: str, args: argparse.Namespace):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    return eval_sweep.request_json("POST", f"{base_url}/generate", json=payload)


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def finite_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in records:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def make_histogram(values: list[float], bins: int) -> list[tuple[float, float, int]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, count) for i, count in enumerate(counts)]


def print_histogram(label: str, values: list[float], bins: int):
    print(f"\n{label}")
    if not values:
        print("  no finite values")
        return
    hist = make_histogram(values, bins)
    max_count = max(count for _, _, count in hist) or 1
    width = 42
    for start, end, count in hist:
        bar = "#" * max(1, round((count / max_count) * width)) if count else ""
        print(f"  [{start:9.4f}, {end:9.4f}) {count:6d} {bar}")


def print_summary(values: list[float], label: str):
    if not values:
        print(f"{label}: no finite values")
        return
    sorted_values = sorted(values)

    def percentile(p: float) -> float:
        idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * p)))
        return sorted_values[idx]

    print(
        f"{label}: count={len(values)} mean={statistics.fmean(values):.6g} "
        f"p50={percentile(0.50):.6g} p90={percentile(0.90):.6g} "
        f"p95={percentile(0.95):.6g} p99={percentile(0.99):.6g}"
    )


def build_server_args(args: argparse.Namespace, calibration_output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=args.base_url,
        model_path=args.model_path,
        model_id=args.model_path,
        speculative_draft_model_path=args.speculative_draft_model_path,
        host=args.host,
        port=args.port,
        mode="calibrate",
        threshold=args.threshold,
        thresholds=str(args.threshold),
        speculative_num_steps=args.speculative_num_steps,
        speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        server_args=args.server_args,
        output_dir=args.output_dir,
        calibration_output=str(calibration_output),
    )


def run_calibration(args: argparse.Namespace) -> Path:
    eval_sweep = load_eval_sweep()
    run_dir = args.output_dir / args.dataset / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration_output = run_dir / "calibration.jsonl"
    prompts_path = run_dir / "prompts.jsonl"

    prompts = load_dataset_prompts(args.dataset, args.split, args.limit)
    with prompts_path.open("w", encoding="utf-8") as fout:
        for prompt in prompts:
            fout.write(json.dumps({"text": prompt}, ensure_ascii=False) + "\n")

    server_args = build_server_args(args, calibration_output)
    process = None
    try:
        process, base_url = eval_sweep.launch_server(server_args, calibration_output)
        server_info = eval_sweep.wait_for_server(base_url, args.server_timeout)
        eval_sweep.validate_dflash_linear(server_info)
        if calibration_output.exists():
            calibration_output.unlink()
        eval_sweep.set_dflash_state(base_url, "calibrate", args.threshold, str(calibration_output))
        for prompt in tqdm(prompts, desc=f"Calibrating {args.dataset}", unit="req"):
            request_generate(eval_sweep, base_url, prompt, args)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    records = list(iter_records(calibration_output))
    perplexities = finite_values(records, "perplexity")
    mean_logprobs = finite_values(records, "mean_logprob")

    print(f"\nWrote calibration records: {calibration_output}")
    print(f"Wrote prompts: {prompts_path}")
    print_summary(perplexities, "perplexity")
    print_summary(mean_logprobs, "mean_logprob")
    print_histogram("Perplexity histogram", perplexities, args.bins)
    print_histogram("Mean logprob histogram", mean_logprobs, args.bins)
    return calibration_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--speculative-draft-model-path", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="Limit benchmark prompts for a smoke run.")
    parser.add_argument("--base-url", help="Attach to an existing SGLang endpoint instead of launching one.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "lossy-spec-dec" / "runs" / "calibration")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--speculative-num-steps", type=int, default=3)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--server-args", help="Extra arguments passed through to sglang.launch_server.")
    parser.add_argument("--bins", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    if args.threshold < 1.0:
        raise ValueError("--threshold must be at least 1.0 for perplexity scores")
    run_calibration(args)


if __name__ == "__main__":
    main()
