"""
Run file-input RAGAS evaluation locally without starting the FastAPI server.

Typical usage:

    python server/test_script/run_ragas_file_eval.py
    python server/test_script/run_ragas_file_eval.py --input /path/to/results.json --output /path/to/ragas_report.json
    python server/test_script/run_ragas_file_eval.py --jobs /path/to/ragas_jobs.json

You can either:
1. Edit the DEFAULT_* values below for your usual weekend workflow.
2. Override them with command-line flags when needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SERVER_DIR.parent

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.config import get_settings as _get_settings

_settings = _get_settings()

# Edit these two most often.
DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_PATH = ""

# Defaults pulled from app/config.py (which reads .env).
# Command-line flags still override these.
DEFAULT_VLLM_API_BASE = _settings.DEFAULT_LLM_API_BASE
DEFAULT_VLLM_API_KEY = _settings.DEFAULT_LLM_API_KEY
DEFAULT_VLLM_MODEL_NAME = _settings.DEFAULT_LLM_MODEL
DEFAULT_EMBEDDING_MODEL_PATH = _settings.DEFAULT_EMBEDDING_MODEL
DEFAULT_EMBEDDING_MAX_TOKENS: int | None = _settings.DEFAULT_EMBEDDING_MAX_TOKENS
DEFAULT_DEVICE = _settings.DEFAULT_RAGAS_DEVICE
DEFAULT_ENABLE_CACHE: bool | None = _settings.DEFAULT_RAGAS_ENABLE_CACHE
DEFAULT_CACHE_DIR = _settings.DEFAULT_RAGAS_CACHE_DIR
DEFAULT_MAX_WORKERS: int | None = _settings.RAGAS_MAX_WORKERS


def _derive_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.ragas_report.json")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run file-input RAGAS evaluation directly through EvalService."
    )
    parser.add_argument(
        "--jobs",
        dest="jobs_path",
        default="",
        help="Optional JSON file containing a list of sequential RAGAS file-eval jobs.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help="Input JSON path. Supports sample_results.json format or standard RAGAS format.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=DEFAULT_OUTPUT_PATH,
        help="Optional output report JSON path. Defaults to <input>.ragas_report.json.",
    )
    parser.add_argument("--vllm-api-base", default=DEFAULT_VLLM_API_BASE)
    parser.add_argument("--vllm-api-key", default=DEFAULT_VLLM_API_KEY)
    parser.add_argument("--vllm-model-name", default=DEFAULT_VLLM_MODEL_NAME)
    parser.add_argument("--embedding-model-path", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument(
        "--embedding-max-tokens",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_TOKENS,
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--enable-cache", type=_parse_bool, default=DEFAULT_ENABLE_CACHE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Run evaluation without writing the detailed JSON report file.",
    )
    return parser


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_request_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_path).expanduser()
    if not str(input_path).strip():
        raise ValueError("Input path is required. Edit DEFAULT_INPUT_PATH or pass --input.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path: str | None
    if args.no_output:
        output_path = None
    else:
        configured_output = _clean_optional_string(args.output_path)
        output_path = str(
            Path(configured_output).expanduser()
            if configured_output
            else _derive_output_path(input_path)
        )

    kwargs: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": output_path,
    }

    optional_string_fields = {
        "vllm_api_base": args.vllm_api_base,
        "vllm_api_key": args.vllm_api_key,
        "vllm_model_name": args.vllm_model_name,
        "embedding_model_path": args.embedding_model_path,
        "device": args.device,
        "cache_dir": args.cache_dir,
    }
    for key, value in optional_string_fields.items():
        cleaned = _clean_optional_string(value)
        if cleaned is not None:
            kwargs[key] = cleaned

    if args.embedding_max_tokens is not None:
        kwargs["embedding_max_tokens"] = args.embedding_max_tokens
    if args.enable_cache is not None:
        kwargs["enable_cache"] = args.enable_cache
    if args.max_workers is not None:
        kwargs["max_workers"] = args.max_workers

    return kwargs


def _load_jobs_file(jobs_path: str) -> list[dict[str, Any]]:
    path = Path(jobs_path).expanduser()
    if not str(path).strip():
        raise ValueError("Jobs path is empty.")
    if not path.exists():
        raise FileNotFoundError(f"Jobs file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Jobs file must be a non-empty JSON array.")

    jobs: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Job #{index} must be a JSON object.")
        jobs.append(item)
    return jobs


def _request_kwargs_from_job(job: dict[str, Any], index: int) -> dict[str, Any]:
    raw_input = job.get("input_path")
    if raw_input is None or not str(raw_input).strip():
        raise ValueError(f"Job #{index} is missing input_path.")

    input_path = Path(str(raw_input)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Job #{index} input file does not exist: {input_path}")

    no_output = bool(job.get("no_output", False))
    raw_output = job.get("output_path")
    if no_output:
        output_path: str | None = None
    elif raw_output is not None and str(raw_output).strip():
        output_path = str(Path(str(raw_output)).expanduser())
    else:
        output_path = str(_derive_output_path(input_path))

    kwargs: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": output_path,
    }

    optional_string_keys = (
        "vllm_api_base",
        "vllm_api_key",
        "vllm_model_name",
        "embedding_model_path",
        "device",
        "cache_dir",
    )
    for key in optional_string_keys:
        value = job.get(key)
        cleaned = _clean_optional_string(str(value)) if value is not None else None
        if cleaned is not None:
            kwargs[key] = cleaned

    optional_passthrough_keys = (
        "embedding_max_tokens",
        "enable_cache",
        "max_workers",
    )
    for key in optional_passthrough_keys:
        value = job.get(key)
        if value is not None:
            kwargs[key] = value

    return kwargs


def _print_summary(result: Any, elapsed_seconds: float, output_path: str | None) -> None:
    summary = result.summary
    print("=" * 80)
    print("RAGAS file evaluation finished")
    print(f"Sample count : {result.sample_count}")
    print(f"Elapsed      : {elapsed_seconds:.2f}s")
    print(f"Output report: {output_path or '<disabled>'}")
    print("-" * 80)
    print(
        "ragas_score                "
        f"mean={summary.ragas_score.mean:.4f} "
        f"min={summary.ragas_score.min:.4f} "
        f"max={summary.ragas_score.max:.4f} "
        f"valid={summary.ragas_score.valid_count} "
        f"error={summary.ragas_score.error_count}"
    )
    print(
        "faithfulness               "
        f"mean={summary.faithfulness.mean:.4f} "
        f"min={summary.faithfulness.min:.4f} "
        f"max={summary.faithfulness.max:.4f} "
        f"valid={summary.faithfulness.valid_count} "
        f"error={summary.faithfulness.error_count}"
    )
    print(
        "answer_relevancy           "
        f"mean={summary.answer_relevancy.mean:.4f} "
        f"min={summary.answer_relevancy.min:.4f} "
        f"max={summary.answer_relevancy.max:.4f} "
        f"valid={summary.answer_relevancy.valid_count} "
        f"error={summary.answer_relevancy.error_count}"
    )
    print(
        "context_recall             "
        f"mean={summary.context_recall.mean:.4f} "
        f"min={summary.context_recall.min:.4f} "
        f"max={summary.context_recall.max:.4f} "
        f"valid={summary.context_recall.valid_count} "
        f"error={summary.context_recall.error_count}"
    )
    print(
        "context_precision          "
        f"mean={summary.context_precision.mean:.4f} "
        f"min={summary.context_precision.min:.4f} "
        f"max={summary.context_precision.max:.4f} "
        f"valid={summary.context_precision.valid_count} "
        f"error={summary.context_precision.error_count}"
    )
    print(
        "context_entity_recall      "
        f"mean={summary.context_entity_recall.mean:.4f} "
        f"min={summary.context_entity_recall.min:.4f} "
        f"max={summary.context_entity_recall.max:.4f} "
        f"valid={summary.context_entity_recall.valid_count} "
        f"error={summary.context_entity_recall.error_count}"
    )
    print(
        "noise_sensitivity_relevant "
        f"mean={summary.noise_sensitivity_relevant.mean:.4f} "
        f"min={summary.noise_sensitivity_relevant.min:.4f} "
        f"max={summary.noise_sensitivity_relevant.max:.4f} "
        f"valid={summary.noise_sensitivity_relevant.valid_count} "
        f"error={summary.noise_sensitivity_relevant.error_count}"
    )
    print(
        "noise_sensitivity_irrelevant "
        f"mean={summary.noise_sensitivity_irrelevant.mean:.4f} "
        f"min={summary.noise_sensitivity_irrelevant.min:.4f} "
        f"max={summary.noise_sensitivity_irrelevant.max:.4f} "
        f"valid={summary.noise_sensitivity_irrelevant.valid_count} "
        f"error={summary.noise_sensitivity_irrelevant.error_count}"
    )
    print("=" * 80)


def _run_request(
    request_kwargs: dict[str, Any],
    *,
    label: str | None = None,
) -> tuple[int, str]:
    """返回 (exit_code, error_message)，成功时 error_message 为空。"""
    if label:
        print(label)
    print(f"Input file   : {request_kwargs['input_path']}")
    print(f"Output report: {request_kwargs.get('output_path') or '<disabled>'}")
    print("Starting RAGAS file evaluation...")

    started_at = time.perf_counter()
    try:
        from app.schemas.eval_schema import RAGASEvalFileRequest
        from app.services.eval_service import EvalService

        service = EvalService()
        request = RAGASEvalFileRequest(**request_kwargs)
        result = service.evaluate_ragas_file(request)
    except ModuleNotFoundError as exc:
        elapsed = time.perf_counter() - started_at
        msg = f"Missing dependency after {elapsed:.2f}s: {exc}"
        print(
            f"{msg}. Install the Python requirements in this environment first.",
            file=sys.stderr,
        )
        return 1, msg
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        msg = f"RAGAS file evaluation failed after {elapsed:.2f}s: {exc}"
        print(msg, file=sys.stderr)
        return 1, msg

    elapsed = time.perf_counter() - started_at
    _print_summary(result, elapsed, request_kwargs.get("output_path"))
    return 0, ""


def _run_jobs(jobs: list[dict[str, Any]], jobs_path: str) -> int:
    total = len(jobs)
    succeeded = 0
    failed: list[str] = []
    run_log: list[dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        job_name = _clean_optional_string(str(job.get("name"))) if job.get("name") is not None else None
        label = f"[{index}/{total}] {job_name}" if job_name else f"[{index}/{total}]"
        entry: dict[str, Any] = {"job": label, "index": index}

        try:
            request_kwargs = _request_kwargs_from_job(job, index)
        except Exception as exc:
            msg = str(exc)
            print(f"{label} configuration error: {msg}", file=sys.stderr)
            failed.append(label)
            entry.update(status="error", error=msg)
            run_log.append(entry)
            continue

        entry["input_path"] = request_kwargs.get("input_path")
        entry["output_path"] = request_kwargs.get("output_path")

        exit_code, error_msg = _run_request(request_kwargs, label=label)
        if exit_code != 0:
            print(f"{label} failed, skipping to next job.", file=sys.stderr)
            failed.append(label)
            entry.update(status="failed", error=error_msg)
            run_log.append(entry)
            continue

        succeeded += 1
        entry["status"] = "succeeded"
        run_log.append(entry)

    # 写 run log JSON
    log_path = Path(jobs_path).with_suffix(".run_log.json")
    log_data = {
        "total": total,
        "succeeded": succeeded,
        "failed": len(failed),
        "jobs": run_log,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"Queue finished: {succeeded}/{total} succeeded, {len(failed)}/{total} failed.")
    print(f"Run log: {log_path}")
    if failed:
        print("Failed jobs: " + ", ".join(failed), file=sys.stderr)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    jobs_path = _clean_optional_string(args.jobs_path)
    if jobs_path:
        try:
            jobs = _load_jobs_file(jobs_path)
        except Exception as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        return _run_jobs(jobs, jobs_path)

    try:
        request_kwargs = _build_request_kwargs(args)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    exit_code, _ = _run_request(request_kwargs)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
