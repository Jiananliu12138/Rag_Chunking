"""
Recompute RAGAS summary metrics from an existing output JSON without rerunning RAGAS.

Typical usage:

    python server/test_script/recompute_ragas_summary.py --input /path/to/ragas_report.json
    python server/test_script/recompute_ragas_summary.py --input /path/to/ragas_report.json --output /path/to/recomputed_summary.json

Supported input formats:
1. Full RAGAS report JSON containing a top-level ``samples`` field.
2. A JSON array consisting directly of sample objects.

This script includes real ``0`` scores in the averages and excludes only invalid values
such as ``null``, ``NaN``, and ``inf``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


POSITIVE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "context_entity_recall",
)

NOISE_METRICS = (
    "noise_sensitivity_relevant",
    "noise_sensitivity_irrelevant",
)

SUMMARY_METRICS = POSITIVE_METRICS + NOISE_METRICS + ("ragas_score",)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute RAGAS summary metrics from an existing JSON report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a RAGAS output JSON containing samples.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for the recomputed summary JSON.",
    )
    return parser


def _derive_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.recomputed_summary.json")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [sample for sample in samples if isinstance(sample, dict)]
    elif isinstance(payload, list):
        return [sample for sample in payload if isinstance(sample, dict)]

    raise ValueError("Input JSON must be a report object with `samples` or a sample array.")


def _extract_metric(sample: dict[str, Any], metric_name: str) -> float | None:
    if metric_name == "ragas_score":
        return _coerce_number(sample.get("ragas_score"))

    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _coerce_number(metrics.get(metric_name))


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "valid_count": 0,
        }

    return {
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "valid_count": len(values),
    }


def _recompute_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric_name in SUMMARY_METRICS:
        values: list[float] = []
        for sample in samples:
            value = _extract_metric(sample, metric_name)
            if value is not None:
                values.append(value)
        summary[metric_name] = _summarize(values)
    return summary


def _print_summary(summary: dict[str, dict[str, Any]], sample_count: int) -> None:
    print("=" * 80)
    print("Recomputed RAGAS summary from existing samples")
    print(f"Sample count : {sample_count}")
    print("-" * 80)
    for metric_name in SUMMARY_METRICS:
        item = summary[metric_name]
        print(
            f"{metric_name:<28} "
            f"mean={item['mean']:.4f} "
            f"min={item['min']:.4f} "
            f"max={item['max']:.4f} "
            f"valid_count={item['valid_count']}"
        )
    print("=" * 80)


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    payload = _load_json(input_path)
    samples = _extract_samples(payload)
    if not samples:
        raise ValueError("No valid samples found in input JSON.")

    summary = _recompute_summary(samples)
    _print_summary(summary, len(samples))

    output_path = Path(args.output).expanduser() if args.output.strip() else _derive_output_path(input_path)
    output_payload = {
        "report_type": "ragas_recomputed_summary",
        "input_path": str(input_path),
        "sample_count": len(samples),
        "summary": summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
