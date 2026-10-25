"""Self-contained, row-free HTML summary."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api import EvaluationReport


def _cell(value: object) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value))


def write_html(report: "EvaluationReport", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in report.metrics:
        rows.append(
            "<tr>"
            f"<td>{_cell(metric.family)}</td><td>{_cell(metric.metric_id)}</td>"
            f"<td>{_cell(metric.comparison)}</td><td>{_cell(metric.feature)}</td>"
            f"<td>{_cell(metric.value)}</td><td>{_cell(metric.direction)}</td>"
            f"<td>{_cell(metric.ci_low)}</td><td>{_cell(metric.ci_high)}</td>"
            f"<td>{_cell(metric.reason)}</td></tr>"
        )
    validation = "".join(
        "<li>" + html.escape(f"{item.role}: {item.code} ({item.severity}) — {item.message}") + "</li>"
        for item in report.validation
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SynthEval report</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#18212b}} table{{border-collapse:collapse;width:100%;font-size:.82rem}}
th,td{{border:1px solid #ccd4dd;padding:.4rem;vertical-align:top}} th{{background:#e8f0f5;position:sticky;top:0}}
.notice{{border-left:5px solid #b45f06;background:#fff6e8;padding:1rem}} code{{background:#eef2f4;padding:.1rem .25rem}}
</style></head><body><h1>SynthEval aggregate report</h1>
<p class="notice">Empirical privacy-risk diagnostics are not anonymity or formal privacy guarantees. A propensity AUC near 0.5 is not proof of equivalence. No row-level data are included.</p>
<p>Framework version: <code>{_cell(report.metadata.get('framework_version'))}</code>; scientific digest: <code>{_cell(report.canonical_digest())}</code>.</p>
<h2>Validation</h2><ul>{validation}</ul><h2>Metric-level results</h2>
<table><thead><tr><th>Family</th><th>Metric</th><th>Comparison</th><th>Feature</th><th>Value</th><th>Direction</th><th>CI low</th><th>CI high</th><th>Reason/limitation</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    path.write_text(content, encoding="utf-8")
