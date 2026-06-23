#!/usr/bin/env python3
"""Calibration and evaluation sweep harness for linear DFlash lossy spec decode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for local use
    yaml = None


DEFAULT_TASKS = ("gsm8k", "humaneval")
DEFAULT_PROMPTS = (
    "Solve carefully: a train travels 120 miles in 3 hours. What is its average speed?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain why the sum of two even numbers is even.",
    "Given a list of integers, describe an algorithm to find duplicates.",
)


def request_json(method: str, url: str, **kwargs):
    response = requests.request(method, url, timeout=kwargs.pop("timeout", 600), **kwargs)
    response.raise_for_status()
    return response.json()


def wait_for_server(base_url: str, timeout_s: int):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            return request_json("GET", f"{base_url}/server_info", timeout=5)
        except Exception as exc:  # noqa: BLE001 - surface the last connection error
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {base_url}: {last_error}")


def get_server_info(base_url: str) -> Dict[str, Any]:
    return request_json("GET", f"{base_url}/server_info")


def get_primary_internal_state(server_info: Dict[str, Any]) -> Dict[str, Any]:
    states = server_info.get("internal_states") or []
    return states[0] if states else {}


def merged_server_state(server_info: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(server_info)
    merged.update(get_primary_internal_state(server_info))
    return merged


def validate_dflash_linear(server_info: Dict[str, Any]):
    state = merged_server_state(server_info)
    errors = []
    if state.get("speculative_algorithm") != "STANDALONE":
        errors.append("speculative_algorithm is not STANDALONE")
    if state.get("speculative_eagle_topk") != 1:
        errors.append("speculative_eagle_topk is not 1")
    if state.get("disable_overlap_schedule") is True:
        errors.append("spec v2 overlap scheduling is not enabled")
    if state.get("enable_multi_layer_eagle"):
        errors.append("multi-layer EAGLE is enabled")
    if errors:
        raise RuntimeError("Endpoint is not linear DFlash/spec-v2: " + "; ".join(errors))


def set_dflash_state(
    base_url: str,
    mode: str,
    threshold: float,
    calibration_output: Optional[str],
):
    payload = {
        "server_args": {
            "dflash_lossy_spec_mode": mode,
            "dflash_lossy_spec_threshold": threshold,
        }
    }
    if calibration_output is not None:
        payload["server_args"]["dflash_lossy_spec_calibration_output"] = calibration_output
    result = request_json("POST", f"{base_url}/set_internal_state", json=payload)
    if isinstance(result, dict) and result.get("updated") is False:
        raise RuntimeError(f"Server rejected /set_internal_state: {result}")
    if isinstance(result, list) and any(
        item is False
        or (isinstance(item, dict) and item.get("updated") is False)
        for item in result
    ):
        raise RuntimeError(f"Server rejected /set_internal_state: {result}")
    if result is False:
        raise RuntimeError(f"Server rejected /set_internal_state: {result}")
    return result


def parse_thresholds(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {value}")
    return values


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompts = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def run_calibration_traffic(args, base_url: str):
    prompts = load_prompts(args.calibration_prompts_file)
    for i in range(args.calibration_requests):
        prompt = prompts[i % len(prompts)]
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }
        request_json("POST", f"{base_url}/generate", json=payload)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_summary(path: Path) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in iter_jsonl(path)]
    if not scores:
        return {"score_count": 0}
    scores_sorted = sorted(scores)

    def pct(p: float) -> float:
        idx = min(len(scores_sorted) - 1, max(0, round((len(scores_sorted) - 1) * p)))
        return scores_sorted[idx]

    return {
        "score_count": len(scores),
        "score_mean": statistics.fmean(scores),
        "score_p50": pct(0.50),
        "score_p90": pct(0.90),
        "score_p95": pct(0.95),
        "score_p99": pct(0.99),
    }


def flatten_metrics(prefix: str, value: Any, out: Dict[str, Any]):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_metrics(next_prefix, nested, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def load_metrics_from_dir(output_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for path in output_dir.rglob("eval_factory_metrics.json"):
        with open(path, "r", encoding="utf-8") as fin:
            flatten_metrics("eval_factory", json.load(fin), metrics)
    for path in output_dir.rglob("results.json"):
        with open(path, "r", encoding="utf-8") as fin:
            flatten_metrics("results", json.load(fin), metrics)
    if yaml is not None:
        for path in list(output_dir.rglob("results.yml")) + list(
            output_dir.rglob("results.yaml")
        ):
            with open(path, "r", encoding="utf-8") as fin:
                flatten_metrics("results", yaml.safe_load(fin), metrics)
    return metrics


def pick_metric(metrics: Dict[str, Any], needles: Iterable[str]):
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        key_lower = key.lower()
        if any(needle in key_lower for needle in needles):
            return value
    return None


def run_nemo_task(args, task: str, threshold: float, base_url: str, task_dir: Path):
    task_dir.mkdir(parents=True, exist_ok=True)
    template = args.nemo_command_template
    if not template:
        raise ValueError("--nemo-command-template is required for sweep mode")
    command = shlex.split(
        template.format(
            task=task,
            threshold=threshold,
            base_url=base_url.rstrip("/"),
            openai_base_url=f"{base_url.rstrip('/')}/v1",
            model=args.model_path or args.model_id,
            model_id=args.model_id or args.model_path,
            output_dir=str(task_dir),
        )
    )
    subprocess.run(command, check=True)
    return load_metrics_from_dir(task_dir)


def launch_server(args, calibration_output: Path):
    if args.base_url:
        return None, args.base_url.rstrip("/")
    if not args.model_path:
        raise ValueError("--model-path is required when --base-url is not provided")
    if not args.speculative_draft_model_path:
        raise ValueError(
            "--speculative-draft-model-path is required when launching SGLang"
        )

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-draft-model-path",
        args.speculative_draft_model_path,
        "--speculative-eagle-topk",
        "1",
        "--dflash-lossy-spec-mode",
        "calibrate" if args.mode == "calibrate" else "accept",
        "--dflash-lossy-spec-threshold",
        str(
            args.threshold
            if args.mode == "calibrate"
            else parse_thresholds(args.thresholds)[0]
        ),
        "--dflash-lossy-spec-calibration-output",
        str(calibration_output),
    ]
    if args.speculative_num_steps is not None:
        command += ["--speculative-num-steps", str(args.speculative_num_steps)]
    if args.speculative_num_draft_tokens is not None:
        command += [
            "--speculative-num-draft-tokens",
            str(args.speculative_num_draft_tokens),
        ]
    if args.server_args:
        command += shlex.split(args.server_args)

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "True"
    process = subprocess.Popen(command, env=env)
    return process, f"http://{args.host}:{args.port}"


def write_rows(output_dir: Path, rows: List[Dict[str, Any]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "runs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as fout:
        fout.write("| " + " | ".join(fieldnames) + " |\n")
        fout.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            fout.write(
                "| "
                + " | ".join(str(row.get(name, "")) for name in fieldnames)
                + " |\n"
            )


def run_calibrate(args, base_url: str, calibration_output: Path):
    if calibration_output.exists():
        calibration_output.unlink()
    set_dflash_state(
        base_url,
        "calibrate",
        args.threshold,
        str(calibration_output),
    )
    run_calibration_traffic(args, base_url)
    summary = score_summary(calibration_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_sweep(args, base_url: str, calibration_output: Path):
    rows = []
    runs_path = args.output_dir / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    thresholds = parse_thresholds(args.thresholds)
    for threshold in thresholds:
        for task in tasks:
            task_dir = args.output_dir / f"threshold_{threshold:g}" / task
            task_score_log = task_dir / "dflash_scores.jsonl"
            task_dir.mkdir(parents=True, exist_ok=True)
            if task_score_log.exists():
                task_score_log.unlink()
            set_dflash_state(base_url, "accept", threshold, str(task_score_log))
            task_metrics = run_nemo_task(args, task, threshold, base_url, task_dir)
            server_info = get_server_info(base_url)
            state = merged_server_state(server_info)
            row = {
                "threshold": threshold,
                "task": task,
                "score": state.get("dflash_lossy_spec_avg_score"),
                "quality_metric": pick_metric(
                    task_metrics,
                    ("pass@1", "accuracy", "exact_match", "f1", "score"),
                ),
                "throughput": pick_metric(
                    task_metrics,
                    ("throughput", "tokens_per_second", "requests_per_second"),
                ),
                "latency": pick_metric(task_metrics, ("latency", "time_per")),
                "avg_spec_accept_length": state.get("avg_spec_accept_length"),
                "dflash_avg_score": state.get("dflash_lossy_spec_avg_score"),
                "dflash_forced_accept_rate": state.get(
                    "dflash_lossy_spec_forced_accept_rate"
                ),
                "dflash_score_count": state.get("dflash_lossy_spec_score_count"),
                "dflash_forced_accept_count": state.get(
                    "dflash_lossy_spec_forced_accept_count"
                ),
            }
            row.update(score_summary(task_score_log))
            row.update(task_metrics)
            rows.append(row)
            with open(runs_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(row, sort_keys=True) + "\n")
    write_rows(args.output_dir, rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "sweep"], required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--speculative-draft-model-path")
    parser.add_argument("--speculative-num-steps", type=int)
    parser.add_argument("--speculative-num-draft-tokens", type=int)
    parser.add_argument("--server-args", help="Extra arguments for launch_server.")
    parser.add_argument("--output-dir", type=Path, default=Path("lossy-spec-dec/runs"))
    parser.add_argument("--calibration-output")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--thresholds", default="1.0,0.95,0.9,0.85,0.8")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--calibration-prompts-file")
    parser.add_argument("--calibration-requests", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument(
        "--nemo-command-template",
        help=(
            "Command template for NeMo Evaluator. Available placeholders: "
            "{task}, {threshold}, {base_url}, {openai_base_url}, {model}, "
            "{model_id}, {output_dir}."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_output = Path(
        args.calibration_output or args.output_dir / "calibration.jsonl"
    )
    process = None
    try:
        process, base_url = launch_server(args, calibration_output)
        server_info = wait_for_server(base_url, args.server_timeout)
        validate_dflash_linear(server_info)
        if args.mode == "calibrate":
            run_calibrate(args, base_url, calibration_output)
        else:
            run_sweep(args, base_url, calibration_output)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
