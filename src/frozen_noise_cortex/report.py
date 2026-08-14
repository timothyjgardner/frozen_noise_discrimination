"""Dependency-free CSV and SVG reporting."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable


Result = dict[str, float]


def write_csv(results: Iterable[Result], path: str | Path) -> None:
    rows = list(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "duration_ms",
        "readout_noise_sd",
        "accuracy_mean",
        "accuracy_sem",
        "clean_distance_mean",
        "clean_distance_sem",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_accuracy_svg(results: Iterable[Result], path: str | Path) -> None:
    rows = list(results)
    if not rows:
        raise ValueError("results cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 820, 500
    left, right, top, bottom = 82, 28, 54, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    durations = sorted({row["duration_ms"] for row in rows})
    noises = sorted({row["readout_noise_sd"] for row in rows})
    log_min, log_max = math.log10(min(durations)), math.log10(max(durations))

    def x_position(duration_ms: float) -> float:
        return left + (math.log10(duration_ms) - log_min) / (log_max - log_min) * plot_width

    def y_position(accuracy: float) -> float:
        return top + (1.0 - accuracy) / 0.5 * plot_height

    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Frozen-noise classification accuracy by duration</title>',
        '<desc id="desc">Accuracy curves approach chance sooner as post-pooling readout noise increases.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="82" y="28" font-family="system-ui" font-size="20" font-weight="600">Frozen-noise classification</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        svg.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="system-ui" font-size="12" fill="#374151">{tick:.1f}</text>'
        )
    for duration in durations:
        x = x_position(duration)
        svg.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        label = f"{duration/1000:g} s" if duration >= 1_000 else f"{duration:g} ms"
        svg.append(
            f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="system-ui" font-size="12" fill="#374151">{label}</text>'
        )
    svg.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#6b7280"/>'
    )

    for index, noise in enumerate(noises):
        color = colors[index % len(colors)]
        series = sorted(
            (row for row in rows if row["readout_noise_sd"] == noise),
            key=lambda row: row["duration_ms"],
        )
        points = " ".join(
            f'{x_position(row["duration_ms"]):.1f},{y_position(row["accuracy_mean"]):.1f}'
            for row in series
        )
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for row in series:
            x = x_position(row["duration_ms"])
            y = y_position(row["accuracy_mean"])
            low = y_position(max(0.5, row["accuracy_mean"] - row["accuracy_sem"]))
            high = y_position(min(1.0, row["accuracy_mean"] + row["accuracy_sem"]))
            svg.extend(
                [
                    f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{high:.1f}" y2="{low:.1f}" stroke="{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>',
                ]
            )
        legend_x = left + 12 + index * 150
        svg.append(
            f'<line x1="{legend_x}" x2="{legend_x+22}" y1="{top+16}" y2="{top+16}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{legend_x+28}" y="{top+20}" font-family="system-ui" font-size="12" fill="#111827">noise {noise:g} sp/s</text>'
        )

    svg.extend(
        [
            f'<text x="{left + plot_width/2:.1f}" y="{height-16}" text-anchor="middle" font-family="system-ui" font-size="13" fill="#111827">Frozen sample duration (log scale)</text>',
            f'<text x="20" y="{top + plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height/2:.1f})" font-family="system-ui" font-size="13" fill="#111827">Test accuracy</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8")

