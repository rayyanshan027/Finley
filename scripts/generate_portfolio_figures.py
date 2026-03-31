from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lightweight SVG figures for the portfolio README."
    )
    parser.add_argument(
        "--benchmark-input",
        default="results/benchmark_summary.csv",
        help="Benchmark summary CSV path.",
    )
    parser.add_argument(
        "--adaptation-input",
        default="results/adaptation_summary.csv",
        help="Adaptation summary CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/figures",
        help="Directory for generated SVG figures.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def svg_document(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="chart">',
            '<style>',
            ".title { font: 700 20px Georgia, serif; fill: #16202b; }",
            ".subtitle { font: 13px Georgia, serif; fill: #4f6475; }",
            ".axis { stroke: #6f8292; stroke-width: 1; }",
            ".grid { stroke: #d7e0e6; stroke-width: 1; stroke-dasharray: 4 4; }",
            ".label { font: 12px monospace; fill: #22313d; }",
            ".small { font: 11px monospace; fill: #5b6d7b; }",
            ".legend { font: 12px Georgia, serif; fill: #22313d; }",
            '</style>',
            *body,
            '</svg>',
        ]
    )


def render_benchmark_chart(rows: list[dict[str, str]]) -> str:
    width = 920
    height = 470
    left = 90
    right = 40
    top = 90
    bottom = 70
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_value = max(float(row["xgboost_mae"]) for row in rows)
    slot = chart_width / len(rows)
    bar_width = 26
    gap = 8

    body = [
        '<rect width="100%" height="100%" fill="#f7f4ec"/>',
        '<text x="34" y="38" class="title">9-Animal Strict LOSO MAE</text>',
        '<text x="34" y="60" class="subtitle">Custom nonlinear baseline vs XGBoost. Lower is better.</text>',
    ]

    for step in range(6):
        value = max_value * step / 5
        y = top + chart_height - (value / max_value) * chart_height
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')

    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top+chart_height}" x2="{width-right}" y2="{top+chart_height}" class="axis"/>')

    for index, row in enumerate(rows):
        x_center = left + slot * index + slot / 2
        nonlinear = float(row["nonlinear_mae"])
        xgboost = float(row["xgboost_mae"])
        nonlinear_h = (nonlinear / max_value) * chart_height
        xgboost_h = (xgboost / max_value) * chart_height
        nonlinear_x = x_center - bar_width - gap / 2
        xgboost_x = x_center + gap / 2
        nonlinear_y = top + chart_height - nonlinear_h
        xgboost_y = top + chart_height - xgboost_h
        winner_fill = "#c46a2e" if row["winner"] == "nonlinear" else "#2b6f8a"
        body.append(
            f'<rect x="{nonlinear_x:.1f}" y="{nonlinear_y:.1f}" width="{bar_width}" height="{nonlinear_h:.1f}" fill="#c46a2e" rx="3"/>'
        )
        body.append(
            f'<rect x="{xgboost_x:.1f}" y="{xgboost_y:.1f}" width="{bar_width}" height="{xgboost_h:.1f}" fill="#2b6f8a" rx="3"/>'
        )
        body.append(
            f'<text x="{x_center:.1f}" y="{top+chart_height+18}" text-anchor="middle" class="label">{escape(row["animal"].upper())}</text>'
        )
        body.append(
            f'<circle cx="{x_center:.1f}" cy="{top-18}" r="5" fill="{winner_fill}"/>'
        )

    legend_x = width - 250
    body.extend(
        [
            f'<rect x="{legend_x}" y="26" width="14" height="14" fill="#c46a2e" rx="2"/>',
            f'<text x="{legend_x + 22}" y="38" class="legend">Nonlinear MAE</text>',
            f'<rect x="{legend_x}" y="50" width="14" height="14" fill="#2b6f8a" rx="2"/>',
            f'<text x="{legend_x + 22}" y="62" class="legend">XGBoost MAE</text>',
            f'<text x="{34}" y="{height-18}" class="small">Winner markers above each animal: orange = nonlinear, blue = XGBoost.</text>',
        ]
    )

    return svg_document(width, height, body)


def render_adaptation_chart(rows: list[dict[str, str]]) -> str:
    width = 920
    height = 470
    left = 90
    right = 40
    top = 90
    bottom = 70
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_value = max(float(row["baseline_mae"]) for row in rows)
    slot = chart_width / len(rows)
    bar_width = 28
    gap = 10

    body = [
        '<rect width="100%" height="100%" fill="#eef6f2"/>',
        '<text x="34" y="38" class="title">7-Animal Adaptation Improvement</text>',
        '<text x="34" y="60" class="subtitle">Hard-session MAE before and after one adapted epoch plus per-unit residual correction.</text>',
    ]

    for step in range(6):
        value = max_value * step / 5
        y = top + chart_height - (value / max_value) * chart_height
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')

    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top+chart_height}" x2="{width-right}" y2="{top+chart_height}" class="axis"/>')

    for index, row in enumerate(rows):
        x_center = left + slot * index + slot / 2
        baseline = float(row["baseline_mae"])
        adaptive = float(row["adaptive_1epoch_mae"])
        baseline_h = (baseline / max_value) * chart_height
        adaptive_h = (adaptive / max_value) * chart_height
        baseline_x = x_center - bar_width - gap / 2
        adaptive_x = x_center + gap / 2
        baseline_y = top + chart_height - baseline_h
        adaptive_y = top + chart_height - adaptive_h
        improvement = (baseline - adaptive) / baseline * 100.0
        body.append(
            f'<rect x="{baseline_x:.1f}" y="{baseline_y:.1f}" width="{bar_width}" height="{baseline_h:.1f}" fill="#6f5f90" rx="3"/>'
        )
        body.append(
            f'<rect x="{adaptive_x:.1f}" y="{adaptive_y:.1f}" width="{bar_width}" height="{adaptive_h:.1f}" fill="#2f8f6b" rx="3"/>'
        )
        body.append(
            f'<text x="{x_center:.1f}" y="{top+chart_height+18}" text-anchor="middle" class="label">{escape(row["animal"].upper())}</text>'
        )
        body.append(
            f'<text x="{x_center:.1f}" y="{top-10}" text-anchor="middle" class="small">-{improvement:.0f}%</text>'
        )

    legend_x = width - 300
    body.extend(
        [
            f'<rect x="{legend_x}" y="26" width="14" height="14" fill="#6f5f90" rx="2"/>',
            f'<text x="{legend_x + 22}" y="38" class="legend">Baseline hard-session MAE</text>',
            f'<rect x="{legend_x}" y="50" width="14" height="14" fill="#2f8f6b" rx="2"/>',
            f'<text x="{legend_x + 22}" y="62" class="legend">1-epoch adaptive MAE</text>',
        ]
    )

    return svg_document(width, height, body)


def main() -> None:
    args = parse_args()
    benchmark_rows = load_csv(Path(args.benchmark_input))
    adaptation_rows = load_csv(Path(args.adaptation_input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "benchmark_9_animals.svg").write_text(
        render_benchmark_chart(benchmark_rows),
        encoding="utf-8",
    )
    (output_dir / "adaptation_7_animals.svg").write_text(
        render_adaptation_chart(adaptation_rows),
        encoding="utf-8",
    )

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
