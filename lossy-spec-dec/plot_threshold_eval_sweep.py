#!/usr/bin/env python3
"""Plot threshold-sweep speed and quality charts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


QUALITY_KEYS = (
    "quality_metric",
    "eval_factory.pass@1",
    "results.pass@1",
    "eval_factory.accuracy",
    "results.accuracy",
    "eval_factory.exact_match",
    "results.exact_match",
    "eval_factory.f1",
    "results.f1",
    "eval_factory.score",
    "results.score",
)
THROUGHPUT_KEYS = (
    "throughput",
    "eval_factory.throughput",
    "results.throughput",
    "eval_factory.tokens_per_second",
    "results.tokens_per_second",
    "eval_factory.output_tokens_per_second",
    "results.output_tokens_per_second",
    "eval_factory.requests_per_second",
    "results.requests_per_second",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fin:
            return [dict(row) for row in csv.DictReader(fin)]
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def pick_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def prepared_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        threshold = pick_float(row, ("threshold",))
        throughput = pick_float(row, THROUGHPUT_KEYS)
        quality = pick_float(row, QUALITY_KEYS)
        if threshold is None:
            continue
        task = str(row.get("task") or row.get("benchmark") or "eval")
        points.append(
            {
                "threshold": threshold,
                "task": task,
                "throughput": throughput,
                "quality": quality,
                "elapsed_seconds": pick_float(row, ("elapsed_seconds",)),
                "forced_accept_rate": pick_float(row, ("dflash_forced_accept_rate",)),
                "avg_accept_length": pick_float(row, ("avg_spec_accept_length",)),
            }
        )
    return points


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#20242a} .small{font-size:12px} .muted{fill:#656d76} .axis{stroke:#24292f;stroke-width:1.3} .grid{stroke:#d8dee4;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width / 2:.1f}" y="30" font-size="20" font-weight="700" text-anchor="middle">{escape(title)}</text>',
    ]


def write_svg(path: Path, parts: list[str]):
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def nice_range(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def ticks(lo: float, hi: float, count: int = 6) -> list[float]:
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def fmt(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.2g}"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.3g}"


def color_for(index: int) -> str:
    colors = ("#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2")
    return colors[index % len(colors)]


def line_chart(
    path: Path,
    title: str,
    points: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
):
    usable = [point for point in points if point.get(x_key) is not None and point.get(y_key) is not None]
    if not usable:
        placeholder_chart(path, title, f"No rows with both {x_key} and {y_key}")
        return

    width, height = 960, 620
    left, right, top, bottom = 82, 170, 70, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    x_lo, x_hi = nice_range([float(point[x_key]) for point in usable])
    y_lo, y_hi = nice_range([float(point[y_key]) for point in usable])

    def sx(value: float) -> float:
        return left + ((value - x_lo) / (x_hi - x_lo)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - y_lo) / (y_hi - y_lo)) * plot_h

    parts = svg_start(width, height, title)
    for tick in ticks(x_lo, x_hi):
        x = sx(tick)
        parts.append(f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="small muted" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{escape(fmt(tick))}</text>')
    for tick in ticks(y_lo, y_hi):
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small muted" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{escape(fmt(tick))}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in usable:
        by_task[str(point["task"])].append(point)
    for idx, task in enumerate(sorted(by_task)):
        color = color_for(idx)
        task_points = sorted(by_task[task], key=lambda point: float(point[x_key]))
        polyline = " ".join(
            f"{sx(float(point[x_key])):.2f},{sy(float(point[y_key])):.2f}"
            for point in task_points
        )
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for point in task_points:
            x = sx(float(point[x_key]))
            y = sy(float(point[y_key]))
            threshold = fmt(float(point["threshold"]))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>')
            parts.append(f'<text class="small muted" x="{x + 7:.2f}" y="{y - 7:.2f}">{escape(threshold)}</text>')
        legend_y = top + idx * 24
        parts.append(f'<rect x="{left + plot_w + 36}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{left + plot_w + 56}" y="{legend_y}">{escape(task)}</text>')

    parts.append(f'<text class="small muted" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(f'<text class="small muted" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    write_svg(path, parts)


def scatter_chart(path: Path, title: str, points: list[dict[str, Any]]):
    usable = [
        point
        for point in points
        if point.get("throughput") is not None and point.get("quality") is not None
    ]
    if not usable:
        placeholder_chart(path, title, "No rows with both throughput and quality")
        return
    line_chart(
        path,
        title,
        usable,
        "throughput",
        "quality",
        "Throughput",
        "Benchmark score",
    )


def placeholder_chart(path: Path, title: str, message: str):
    parts = svg_start(900, 420, title)
    parts.append(f'<text x="450" y="210" font-size="16" text-anchor="middle">{escape(message)}</text>')
    write_svg(path, parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Combined runs.jsonl or summary.csv from run_threshold_eval_sweep.py")
    parser.add_argument("--output-dir", type=Path, default=Path("lossy-spec-dec/images"))
    parser.add_argument("--summary-name", default="threshold_plot_data.json")
    args = parser.parse_args()

    points = prepared_points(load_rows(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / args.summary_name, points)
    line_chart(
        args.output_dir / "tok-vs-threshold.svg",
        "Throughput vs Perplexity Threshold",
        points,
        "threshold",
        "throughput",
        "Perplexity threshold",
        "Throughput",
    )
    line_chart(
        args.output_dir / "benchmark-vs-threshold.svg",
        "Benchmark Score vs Perplexity Threshold",
        points,
        "threshold",
        "quality",
        "Perplexity threshold",
        "Benchmark score",
    )
    scatter_chart(
        args.output_dir / "benchmark-vs-tok.svg",
        "Benchmark Score vs Throughput",
        points,
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
