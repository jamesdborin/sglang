#!/usr/bin/env python3
"""Create calibration analysis plots from a DFlash calibration JSONL file."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PERCENTILES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def score_from_row(row: dict[str, Any]) -> float | None:
    return finite_float(row.get("score")) or finite_float(row.get("perplexity"))


def accept_len_from_row(row: dict[str, Any]) -> int | None:
    value = row.get("normal_accept_len", row.get("accept_len", row.get("accepted_length")))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


def reverse_percentiles(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        f"P{p}": percentile(sorted_values, 1.0 - p / 100.0)
        for p in PERCENTILES
    }


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#20242a} .small{font-size:12px} .axis{stroke:#333;stroke-width:1.2} .grid{stroke:#e1e5ea;stroke-width:1} .muted{fill:#626a73}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="700">{escape(title)}</text>',
    ]


def write_svg(path: Path, parts: list[str]):
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def histogram(values: list[float], bins: int) -> list[tuple[float, float, int]]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]


def nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    if lo == hi:
        return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def plot_perplexity_histogram(path: Path, values: list[float], bins: int, pct: dict[str, float]):
    width, height = 1200, 720
    left, right, top, bottom = 78, 330, 70, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    positive = [value for value in values if value > 0]
    log_values = [math.log10(value) for value in positive]
    hist = histogram(log_values, bins)
    max_count = max(count for _, _, count in hist)
    lo, hi = min(log_values), max(log_values)

    parts = svg_start(width, height, "Calibration Perplexity Histogram")
    for tick in nice_ticks(0, max_count, 6):
        y = top + plot_h - (tick / max_count) * plot_h if max_count else top + plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small muted" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    for tick in nice_ticks(lo, hi, 7):
        x = left + ((tick - lo) / (hi - lo)) * plot_w if hi != lo else left
        label = f"{10 ** tick:.3g}"
        parts.append(f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="small muted" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{escape(label)}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')

    bar_gap = 1
    for start, end, count in hist:
        x = left + ((start - lo) / (hi - lo)) * plot_w if hi != lo else left
        x2 = left + ((end - lo) / (hi - lo)) * plot_w if hi != lo else left + plot_w
        bar_h = (count / max_count) * plot_h if max_count else 0
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x + bar_gap:.2f}" y="{y:.2f}" width="{max(1, x2 - x - 2 * bar_gap):.2f}" height="{bar_h:.2f}" fill="#5c7cfa" opacity="0.78"/>'
        )

    marker_colors = {
        "P0": "#c92a2a",
        "P50": "#1864ab",
        "P90": "#2b8a3e",
        "P99": "#9c36b5",
        "P100": "#e67700",
    }
    for label, color in marker_colors.items():
        value = pct[label]
        if value <= 0:
            continue
        lx = math.log10(value)
        x = left + ((lx - lo) / (hi - lo)) * plot_w if hi != lo else left
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text class="small" x="{x + 4:.2f}" y="{top + 16}" fill="{color}">{label}</text>')

    parts.append(f'<text class="small muted" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">Perplexity, log scale</text>')
    parts.append(f'<text class="small muted" transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Calibration chunks</text>')

    table_x = left + plot_w + 42
    parts.append(f'<text x="{table_x}" y="{top + 4}" font-size="15" font-weight="700">Reverse percentiles</text>')
    parts.append(f'<text class="small muted" x="{table_x}" y="{top + 24}">P100 = minimum / least surprise</text>')
    for i, label in enumerate([f"P{p}" for p in PERCENTILES]):
        y = top + 54 + i * 24
        parts.append(f'<text class="small" x="{table_x}" y="{y}">{label}</text>')
        parts.append(f'<text class="small" x="{table_x + 74}" y="{y}" text-anchor="end">{pct[label]:.6g}</text>')
    parts.append(f'<text class="small muted" x="{table_x}" y="{height - 38}">n={len(values)} min={min(values):.6g} max={max(values):.6g}</text>')
    write_svg(path, parts)


def plot_accept_length_histogram(path: Path, lengths: list[int]):
    width, height = 900, 560
    left, right, top, bottom = 78, 42, 70, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    counts = Counter(lengths)
    keys = list(range(min(counts), max(counts) + 1))
    max_count = max(counts.values())
    parts = svg_start(width, height, "Accepted Speculation Length Histogram")
    for tick in nice_ticks(0, max_count, 6):
        y = top + plot_h - (tick / max_count) * plot_h if max_count else top + plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small muted" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    step = plot_w / len(keys)
    for i, key in enumerate(keys):
        count = counts[key]
        bar_h = (count / max_count) * plot_h
        x = left + i * step + step * 0.15
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{step * 0.7:.2f}" height="{bar_h:.2f}" fill="#12b886" opacity="0.8"/>')
        parts.append(f'<text class="small muted" x="{left + i * step + step / 2:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{key}</text>')
        parts.append(f'<text class="small" x="{left + i * step + step / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle">{count}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="small muted" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">Accepted length</text>')
    parts.append(f'<text class="small muted" transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Calibration chunks</text>')
    parts.append(f'<text class="small muted" x="{left}" y="{height - 12}">n={len(lengths)} mean={statistics.fmean(lengths):.4g}</text>')
    write_svg(path, parts)


def plot_correlation(path: Path, pairs: list[tuple[int, float]]):
    width, height = 940, 620
    left, right, top, bottom = 78, 44, 70, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [length for length, _ in pairs]
    ys = [1.0 / ppl for _, ppl in pairs if ppl > 0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min, y_max = y_min - 0.05, y_max + 0.05
    parts = svg_start(width, height, "Accepted Length vs 1/Perplexity")
    for tick in nice_ticks(y_min, y_max, 6):
        y = top + plot_h - ((tick - y_min) / (y_max - y_min)) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small muted" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.3g}</text>')
    for tick in range(x_min, x_max + 1):
        x = left + ((tick - x_min) / max(1, x_max - x_min)) * plot_w
        parts.append(f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="small muted" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{tick}</text>')

    buckets: dict[int, list[float]] = defaultdict(list)
    for length, ppl in pairs:
        if ppl <= 0:
            continue
        yv = 1.0 / ppl
        buckets[length].append(yv)
        x = left + ((length - x_min) / max(1, x_max - x_min)) * plot_w
        y = top + plot_h - ((yv - y_min) / (y_max - y_min)) * plot_h
        jitter = ((len(buckets[length]) % 11) - 5) * 1.6
        parts.append(f'<circle cx="{x + jitter:.2f}" cy="{y:.2f}" r="2.1" fill="#4263eb" opacity="0.22"/>')

    mean_points = []
    for length in sorted(buckets):
        mean_y = statistics.fmean(buckets[length])
        x = left + ((length - x_min) / max(1, x_max - x_min)) * plot_w
        y = top + plot_h - ((mean_y - y_min) / (y_max - y_min)) * plot_h
        mean_points.append((x, y))
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#e03131"/>')
    if len(mean_points) > 1:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in mean_points)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#e03131" stroke-width="2.5"/>')

    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="small muted" x="{left + plot_w / 2}" y="{height - 30}" text-anchor="middle">Accepted length</text>')
    parts.append(f'<text class="small muted" transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">1 / perplexity</text>')
    parts.append(f'<text class="small muted" x="{left}" y="{height - 12}">Blue: chunks, red: mean by accepted length</text>')
    write_svg(path, parts)


def analyze(calibration_path: Path, output_dir: Path, bins: int):
    records = list(iter_jsonl(calibration_path))
    perplexities: list[float] = []
    lengths: list[int] = []
    pairs: list[tuple[int, float]] = []
    for row in records:
        perplexity = score_from_row(row)
        accept_len = accept_len_from_row(row)
        if perplexity is not None:
            perplexities.append(perplexity)
        if accept_len is not None:
            lengths.append(accept_len)
        if perplexity is not None and accept_len is not None and perplexity > 0:
            pairs.append((accept_len, perplexity))

    if not perplexities:
        raise ValueError(f"No finite perplexity scores found in {calibration_path}")
    if not lengths:
        raise ValueError(f"No accepted lengths found in {calibration_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pct = reverse_percentiles(perplexities)
    plot_perplexity_histogram(output_dir / "perplexity_histogram.svg", perplexities, bins, pct)
    plot_accept_length_histogram(output_dir / "accepted_length_histogram.svg", lengths)
    plot_correlation(output_dir / "acceptance_vs_inverse_perplexity.svg", pairs)

    summary = {
        "calibration_path": str(calibration_path),
        "row_count": len(records),
        "paired_count": len(pairs),
        "perplexity_min_p100": min(perplexities),
        "perplexity_max_p0": max(perplexities),
        "perplexity_mean": statistics.fmean(perplexities),
        "accept_len_min": min(lengths),
        "accept_len_max": max(lengths),
        "accept_len_mean": statistics.fmean(lengths),
        "reverse_percentiles": pct,
    }
    (output_dir / "calibration_histogram_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir or args.calibration_jsonl.parent / "analysis"
    analyze(args.calibration_jsonl, output_dir, args.bins)


if __name__ == "__main__":
    main()
