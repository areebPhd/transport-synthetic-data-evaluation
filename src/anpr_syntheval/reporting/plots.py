"""Small dependency-free SVG plotting helpers for aggregate results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence


def write_bar_svg(
    labels: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    title: str,
    y_label: str,
    width: int = 1200,
    height: int = 700,
) -> Path:
    """Write a publication-sized SVG without embedding row-level labels."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    maximum = max([abs(float(item)) for item in values] + [1e-12])
    left, top, bottom = 110, 80, 150
    plot_width, plot_height = width - left - 40, height - top - bottom
    bar_width = plot_width / max(len(values), 1) * 0.7
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="sans-serif" font-size="24">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#1d2730"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#1d2730"/>',
        f'<text x="25" y="{top+plot_height/2}" transform="rotate(-90 25 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(y_label)}</text>',
    ]
    for index, (label, raw_value) in enumerate(zip(labels, values)):
        value = float(raw_value)
        bar_height = abs(value) / maximum * plot_height
        x = left + (index + 0.15) * plot_width / max(len(values), 1)
        y = top + plot_height - bar_height
        pieces.extend([
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2878a8"/>',
            f'<text x="{x+bar_width/2:.2f}" y="{top+plot_height+24}" text-anchor="end" transform="rotate(-35 {x+bar_width/2:.2f} {top+plot_height+24})" font-family="sans-serif" font-size="13">{html.escape(str(label))}</text>',
            f'<text x="{x+bar_width/2:.2f}" y="{max(y-6, top+14):.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.3g}</text>',
        ])
    pieces.append("</svg>\n")
    target.write_text("".join(pieces), encoding="utf-8")
    return target
