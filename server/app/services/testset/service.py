"""Testset generation service — orchestrates RAGAS testset generation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragas.run_config import RunConfig
from ragas.testset.synthesizers.generate import TestsetGenerator

from app.config import get_settings

from .builders import (
    build_embedding_model,
    build_llm,
    build_query_distribution,
    build_transforms,
)
from .chunks_loader import extract_reference_contexts_meta, load_chunks
from .patches import apply_all_patches

_logger = logging.getLogger("testset_service")

# Type alias for progress callback: (stage, pct, message) -> None
ProgressCallback = Callable[[str, int, str], None]


def _noop_progress(stage: str, pct: int, message: str = "") -> None:
    pass


class TestsetService:

    @staticmethod
    def _resolve(explicit: Any, settings_attr: str, fallback: Any = None) -> Any:
        """Return explicit value if truthy, else settings value, else fallback."""
        if explicit not in (None, ""):
            return explicit
        val = getattr(get_settings(), settings_attr, None)
        if val not in (None, ""):
            return val
        return fallback

    @staticmethod
    def generate(
        request,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        progress = on_progress or _noop_progress
        settings = get_settings()

        # tiktoken cache
        cache_dir = settings.TIKTOKEN_CACHE_DIR
        if cache_dir:
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache_dir)

        # Resolve config: request > settings > hardcoded defaults
        r = TestsetService._resolve
        llm_base_url = r(request.llm_api_base, "DEFAULT_LLM_API_BASE", "http://127.0.0.1:8001/v1")
        llm_api_key = r(request.llm_api_key, "DEFAULT_LLM_API_KEY", "EMPTY")
        llm_model = r(request.llm_model, "DEFAULT_LLM_MODEL")
        llm_max_tokens = r(request.llm_max_tokens, "TESTSET_LLM_MAX_TOKENS", 2048)
        llm_timeout = r(request.llm_timeout, "TESTSET_LLM_TIMEOUT", 3600)
        embedding_model_path = r(request.embedding_model_path, "DEFAULT_EMBEDDING_MODEL")
        embedding_device = r(request.embedding_device, "DEFAULT_EMBEDDING_DEVICE", "cuda")
        max_workers = request.max_workers or settings.TESTSET_MAX_WORKERS or 18
        max_retries = request.max_retries or settings.TESTSET_MAX_RETRIES or 3
        max_wait = request.max_wait_seconds or settings.TESTSET_MAX_WAIT_SECONDS or 3600
        run_timeout = request.run_timeout_seconds or settings.TESTSET_RUN_TIMEOUT_SECONDS or 3600

        # Quiet noisy ragas filters
        logging.getLogger("ragas.testset.transforms.filters").setLevel(logging.ERROR)

        # Stage 1: preflight
        progress("preflight", 5, "Checking LLM endpoint...")
        _preflight_check(llm_base_url, llm_api_key, llm_model, llm_timeout)

        # Apply idempotent patches
        apply_all_patches()

        # Stage 2: load chunks
        progress("loading_chunks", 10, "Loading chunks...")
        chunks_path = Path(request.chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        chunks = load_chunks(chunks_path)
        _logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
        progress("loading_chunks", 15, f"Loaded {len(chunks)} chunks")

        # Stage 3: build models
        progress("building_models", 20, "Building LLM wrapper...")
        llm = build_llm(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            max_tokens=int(llm_max_tokens),
            timeout=int(llm_timeout),
        )
        progress("building_models", 25, "Building embedding model...")
        embedding_model = build_embedding_model(
            model_path=embedding_model_path,
            device=embedding_device,
        )

        # Stage 4: build transforms & distribution
        progress("building_transforms", 30, "Building transforms & query distribution...")
        skip_custom_node_filter = request.skip_custom_node_filter
        transforms = build_transforms(llm, embedding_model) if skip_custom_node_filter else None
        query_distribution = build_query_distribution(llm)
        progress("building_transforms", 35, "Transforms ready")

        # Create generator
        generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
        generator._mc_last_failed_rows = []

        run_config = RunConfig(
            timeout=int(run_timeout),
            max_retries=int(max_retries),
            max_wait=int(max_wait),
            max_workers=int(max_workers),
        )

        _logger.info(
            "Starting testset generation: size=%d, workers=%d, timeout=%ds",
            request.testset_size,
            max_workers,
            run_timeout,
        )

        # Stage 5: generate (the long stage, 35% -> 85%)
        progress("generating", 35, f"Generating {request.testset_size} QA pairs...")

        result = generator.generate_with_chunks(
            chunks=chunks,
            testset_size=request.testset_size,
            transforms=transforms,
            transforms_llm=None,
            transforms_embedding_model=None,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=None,
            token_usage_parser=None,
            with_debugging_logs=request.with_debugging_logs,
            raise_exceptions=request.fail_fast,
            return_executor=False,
        )

        # Stage 6: process results
        progress("processing", 85, "Processing results...")
        rows, failed_rows = _split_success_and_failed_rows(result)
        failed_rows.extend(getattr(generator, "_mc_last_failed_rows", []))

        extract_reference_contexts_meta(rows)
        progress("processing", 90, f"{len(rows)} succeeded, {len(failed_rows)} failed")

        # Stage 7: write output
        progress("writing", 92, "Writing output files...")
        output_file = request.output_file
        if not output_file:
            output_file = str(
                chunks_path.parent / f"{chunks_path.stem}_testset.jsonl"
            )
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _write_jsonl(output_path, rows)
        _logger.info("Generated %d samples -> %s", len(rows), output_path)

        failed_output = None
        if failed_rows:
            failed_path = output_path.with_name(f"{output_path.stem}.failed.jsonl")
            _write_jsonl(failed_path, failed_rows)
            failed_output = str(failed_path)
            _logger.warning(
                "%d failed generations saved to %s", len(failed_rows), failed_path
            )

        progress("writing", 98, "Done")

        return {
            "output_file": str(output_path),
            "failed_file": failed_output,
            "total_generated": len(rows),
            "total_failed": len(failed_rows),
            "testset_size_requested": request.testset_size,
            "samples_preview": rows[:5] if rows else [],
        }


def _preflight_check(base_url: str, api_key: str, model: str, timeout: int) -> None:
    from urllib import request as urllib_request

    url = f"{base_url.rstrip('/')}/models"
    req = urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(req, timeout=min(timeout, 30)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"LLM endpoint preflight failed: {url} unreachable. Details: {exc}"
        ) from exc

    available = {
        item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
    }
    if available and model not in available:
        _logger.warning(
            "Model '%s' not in /models list. Available (sample): %s",
            model,
            sorted(list(available))[:5],
        )


def _split_success_and_failed_rows(result: Any) -> tuple:
    if not hasattr(result, "to_list") or not callable(getattr(result, "to_list")):
        raise TypeError(
            f"Expected result with .to_list(), got {type(result).__name__}"
        )
    raw_rows = result.to_list()
    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if isinstance(row, dict):
            rows.append(row)
        else:
            failed_rows.append({
                "index": index,
                "type": type(row).__name__,
                "repr": repr(row)[:1000],
            })
    return rows, failed_rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            row_with_id = {"question_id": idx, **row}
            f.write(json.dumps(row_with_id, ensure_ascii=False) + "\n")
