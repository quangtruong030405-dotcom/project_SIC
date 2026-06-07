from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def simple_svg_line_chart(path: str | Path, series: dict[str, list[float]], width: int = 920, height: int = 460) -> None:
    if not series:
        return
    margin = 54
    max_len = max(len(v) for v in series.values())
    all_values = [x for values in series.values() for x in values]
    y_min, y_max = min(all_values), max(all_values)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    colors = ["#1167b1", "#c43d32", "#207245", "#7f3c8d"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Best-so-far validation RMSE</text>',
    ]
    for idx, (name, values) in enumerate(series.items()):
        points = []
        for i, value in enumerate(values):
            x = margin + (width - 2 * margin) * (i / max(max_len - 1, 1))
            y = height - margin - (height - 2 * margin) * ((value - y_min) / (y_max - y_min))
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[idx % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{width-margin-140}" y="{margin + idx*22}" fill="{color}" font-family="Arial" font-size="14">{name}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def simple_svg_actual_predicted(path: str | Path, actual: list[float], predicted: list[float], width: int = 920, height: int = 460) -> None:
    limit = min(len(actual), len(predicted), 240)
    simple_svg_line_chart(
        path,
        {
            "Actual": list(actual[:limit]),
            "Predicted": list(predicted[:limit]),
        },
        width=width,
        height=height,
    )


def simple_svg_histogram(path: str | Path, errors: list[float], bins: int = 20, width: int = 920, height: int = 460) -> None:
    if not errors:
        return
    values = [float(x) for x in errors]
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
        counts[idx] += 1
    margin = 54
    max_count = max(counts) or 1
    bar_w = (width - 2 * margin) / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Test error distribution</text>',
    ]
    for i, count in enumerate(counts):
        x = margin + i * bar_w
        h = (height - 2 * margin) * count / max_count
        y = height - margin - h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-2,1):.2f}" height="{h:.2f}" fill="#1167b1"/>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
