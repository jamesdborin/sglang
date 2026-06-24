#!/usr/bin/env python3
"""Run SGLang's local HumanEval evaluator and write parseable results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="humaneval")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--api", choices=["chat", "completion"], default="chat")
    args = parser.parse_args()

    if args.task != "humaneval":
        raise ValueError(f"Only humaneval is supported by this adapter, got {args.task}")
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    from sglang.test.run_eval import run_eval

    eval_args = SimpleNamespace(
        api=args.api,
        base_url=args.base_url.rstrip("/"),
        categories=None,
        chat_template_kwargs=None,
        dataset_path="THUDM/LongBench-v2",
        eval_name="humaneval",
        gsm8k_data_path=None,
        host="127.0.0.1",
        max_context_length=None,
        max_tokens=args.max_tokens,
        min_context_length=None,
        min_p=None,
        mixed_prefix_gsm8k_secondary_pool_size=15,
        mixed_prefix_gsm8k_seed=42,
        model=args.model,
        num_examples=args.num_examples,
        num_shots=5,
        num_threads=args.num_threads,
        port=None,
        reasoning_effort=None,
        repeat=1,
        return_latency=False,
        stop=["Question", "Assistant:", "<|separator|>"],
        temperature=args.temperature,
        thinking_mode=None,
        top_k=None,
        top_p=args.top_p,
    )
    metrics = run_eval(eval_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    report_stem = f"humaneval_{args.model.replace('/', '_')}"
    for suffix in (".json", ".html"):
        source = Path("/tmp") / f"{report_stem}{suffix}"
        if source.exists():
            shutil.copy2(source, args.output_dir / source.name)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
