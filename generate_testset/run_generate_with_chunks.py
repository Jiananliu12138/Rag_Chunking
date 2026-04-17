"""
Run chunk-input RAGAS testset generation locally with optional sequential jobs.

Typical usage:

    python generate_testset/run_generate_with_chunks.py
    python generate_testset/run_generate_with_chunks.py --input /path/to/chunks.json --output /path/to/testset.jsonl
    python generate_testset/run_generate_with_chunks.py --jobs /path/to/generate_jobs.json

You can either:
1. Edit the DEFAULT_* values below for your usual workflow.
2. Override them with command-line flags when needed.
"""

import argparse
import gc
import json
import logging
import os
import math
import random
import sys
import time
import traceback

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from langchain_core.documents import Document

from ragas.run_config import RunConfig
from ragas.testset.synthesizers.generate import TestsetGenerator

from testset_config import (
    build_embedding_model,
    build_llm,
    build_query_distribution,
    build_transforms,
    patch_multihop_prompt,
)

os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"
Chunk = Union[str, Document]

# ===== Static config (edit directly) =====
CHUNKS_FILE = Path("/data/h50056789/Rag_Chunking/Chunk_Result/Llamaindex_Chunk/qasper_llamaindex_chunk.json")
OUTPUT_FILE = Path("/data/h50056789/Rag_Chunking/gov_report_llamaindex_QA_pairs.json")
TESTSET_SIZE = 10
WITH_DEBUGGING_LOGS = True

MAX_WORKERS = 18
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 3600
RUN_TIMEOUT_SECONDS = 3600
FAIL_FAST = False

SKIP_CUSTOM_NODE_FILTER = True
QUIET_FILTER_WARNINGS = True

LLM_PREFLIGHT_TIMEOUT_SECONDS = 3600

CHUNK_META_PREFIX = "<<<MC_META>>>"
CHUNK_META_SUFFIX = "<<<END_MC_META>>>"

# These are the values you are most likely to edit for ad-hoc runs.
DEFAULT_INPUT_PATH = str(CHUNKS_FILE)
DEFAULT_OUTPUT_PATH = str(OUTPUT_FILE)


def _derive_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.testset.jsonl")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chunk-input RAGAS testset generation directly."
    )
    parser.add_argument(
        "--jobs",
        dest="jobs_path",
        default="",
        help="Optional JSON file containing a list of sequential generation jobs.",
    )
    parser.add_argument(
        "--input",
        "--chunks",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help="Input chunks path (.txt/.json/.jsonl).",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="",
        help="Optional output JSONL path. Defaults to DEFAULT_OUTPUT_PATH for the default input, otherwise <input>.testset.jsonl.",
    )
    parser.add_argument("--testset-size", type=int, default=TESTSET_SIZE)
    parser.add_argument(
        "--with-debugging-logs",
        type=_parse_bool,
        default=WITH_DEBUGGING_LOGS,
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--max-wait-seconds", type=int, default=MAX_WAIT_SECONDS)
    parser.add_argument("--run-timeout-seconds", type=int, default=RUN_TIMEOUT_SECONDS)
    parser.add_argument("--fail-fast", type=_parse_bool, default=FAIL_FAST)
    parser.add_argument(
        "--skip-custom-node-filter",
        type=_parse_bool,
        default=SKIP_CUSTOM_NODE_FILTER,
        help="Preserves the existing script behavior. When true, build custom transforms.",
    )
    parser.add_argument(
        "--quiet-filter-warnings",
        type=_parse_bool,
        default=QUIET_FILTER_WARNINGS,
    )
    parser.add_argument(
        "--llm-preflight-timeout-seconds",
        type=int,
        default=LLM_PREFLIGHT_TIMEOUT_SECONDS,
    )
    return parser


def _build_request_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    raw_input = _clean_optional_string(args.input_path)
    if raw_input is None:
        raise ValueError("Input path is required. Edit DEFAULT_INPUT_PATH or pass --input.")

    input_path = Path(raw_input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    configured_output = _clean_optional_string(args.output_path)
    if configured_output is not None:
        output_path = Path(configured_output).expanduser()
    elif input_path == Path(DEFAULT_INPUT_PATH).expanduser() and DEFAULT_OUTPUT_PATH.strip():
        output_path = Path(DEFAULT_OUTPUT_PATH).expanduser()
    else:
        output_path = _derive_output_path(input_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "testset_size": args.testset_size,
        "with_debugging_logs": args.with_debugging_logs,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "max_wait_seconds": args.max_wait_seconds,
        "run_timeout_seconds": args.run_timeout_seconds,
        "fail_fast": args.fail_fast,
        "skip_custom_node_filter": args.skip_custom_node_filter,
        "quiet_filter_warnings": args.quiet_filter_warnings,
        "llm_preflight_timeout_seconds": args.llm_preflight_timeout_seconds,
        "llm_base_url": None,
        "llm_api_key": None,
        "llm_model": None,
    }


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
    raw_input = job.get("input_path", job.get("chunks_path"))
    if raw_input is None or not str(raw_input).strip():
        raise ValueError(f"Job #{index} is missing input_path.")

    input_path = Path(str(raw_input)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Job #{index} input file does not exist: {input_path}")

    raw_output = job.get("output_path")
    if raw_output is not None and str(raw_output).strip():
        output_path = Path(str(raw_output)).expanduser()
    else:
        output_path = _derive_output_path(input_path)

    kwargs: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "testset_size": job.get("testset_size", TESTSET_SIZE),
        "with_debugging_logs": job.get("with_debugging_logs", WITH_DEBUGGING_LOGS),
        "max_workers": job.get("max_workers", MAX_WORKERS),
        "max_retries": job.get("max_retries", MAX_RETRIES),
        "max_wait_seconds": job.get("max_wait_seconds", MAX_WAIT_SECONDS),
        "run_timeout_seconds": job.get("run_timeout_seconds", RUN_TIMEOUT_SECONDS),
        "fail_fast": job.get("fail_fast", FAIL_FAST),
        "skip_custom_node_filter": job.get(
            "skip_custom_node_filter", SKIP_CUSTOM_NODE_FILTER
        ),
        "quiet_filter_warnings": job.get(
            "quiet_filter_warnings", QUIET_FILTER_WARNINGS
        ),
        "llm_preflight_timeout_seconds": job.get(
            "llm_preflight_timeout_seconds", LLM_PREFLIGHT_TIMEOUT_SECONDS
        ),
        # Optional per-job LLM overrides. When absent, falls back to module-level
        # defaults in testset_config.
        "llm_base_url": _clean_optional_string(job.get("llm_base_url")),
        "llm_api_key": _clean_optional_string(job.get("llm_api_key")),
        "llm_model": _clean_optional_string(job.get("llm_model")),
    }
    return kwargs


# ===== Chunk loading =====


def _load_chunks(path: Path) -> List[Chunk]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "splits" in data:
            source_filepath = str(data.get("filepath") or "")
            splits = data.get("splits")
            if not isinstance(splits, list):
                raise ValueError("`splits` in JSON must be a list")
            return _normalize_split_tuples(splits, source_filepath)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list, or a dict containing `splits`")
        return _normalize_chunks(data)

    if suffix == ".jsonl":
        rows: List[Any] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
        return _normalize_chunks(rows)

    raise ValueError(f"Unsupported chunks file format: {suffix}")


def _normalize_split_tuples(
    splits: Sequence[Any],
    source_filepath: str = "",
) -> List[Chunk]:
    normalized: List[Chunk] = []
    for item in splits:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise TypeError("Each split item must be [chunk_text, article_id, chunk_id]")

        chunk_text = item[0]
        article_id = item[1]
        chunk_id = item[2]
        if not isinstance(chunk_text, str):
            raise TypeError("split[0] must be a string")
        text = chunk_text.strip()
        if not text:
            continue

        metadata = {
            "source_article_id": article_id,
            "source_chunk_id": chunk_id,
        }
        if source_filepath:
            metadata["source_filepath"] = source_filepath

        text_with_meta = _inject_chunk_meta_into_text(
            text=text,
            source_article_id=article_id,
            source_chunk_id=chunk_id,
            source_filepath=source_filepath or None,
        )
        normalized.append(Document(page_content=text_with_meta, metadata=metadata))

    return normalized


def _normalize_chunks(items: Sequence[Any]) -> List[Chunk]:
    normalized: List[Chunk] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue

        if isinstance(item, dict):
            page_content = item.get("page_content") or item.get("text") or ""
            metadata = item.get("metadata", {})
            if isinstance(page_content, str) and page_content.strip():
                text = page_content.strip()
                source_article_id = metadata.get("source_article_id")
                source_chunk_id = metadata.get("source_chunk_id")
                source_filepath = metadata.get("source_filepath")
                if source_article_id is not None or source_chunk_id is not None or source_filepath:
                    text = _inject_chunk_meta_into_text(
                        text=text,
                        source_article_id=source_article_id,
                        source_chunk_id=source_chunk_id,
                        source_filepath=source_filepath,
                    )
                normalized.append(Document(page_content=text, metadata=metadata))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            normalized.extend(_normalize_split_tuples([item]))
            continue

        raise TypeError(
            "Chunk items must be str, dict, or [text, article_id, chunk_id]"
        )

    return normalized


# ===== Chunk meta injection / extraction =====


def _inject_chunk_meta_into_text(
    text: str,
    source_article_id: Any = None,
    source_chunk_id: Any = None,
    source_filepath: Any = None,
) -> str:
    payload = {
        "doc_id": source_article_id,
        "chunk_id": source_chunk_id,
        "source_filepath": source_filepath,
    }
    meta_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{CHUNK_META_PREFIX}{meta_json}{CHUNK_META_SUFFIX}\n{text}"


def extract_all_chunk_meta_from_text(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"metas": [], "content": text}

    metas: List[Dict[str, Any]] = []
    pieces: List[str] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        prefix_pos = text.find(CHUNK_META_PREFIX, cursor)
        if prefix_pos == -1:
            pieces.append(text[cursor:])
            break

        pieces.append(text[cursor:prefix_pos])
        suffix_pos = text.find(CHUNK_META_SUFFIX, prefix_pos + len(CHUNK_META_PREFIX))
        if suffix_pos == -1:
            pieces.append(text[prefix_pos:])
            break

        meta_json = text[prefix_pos + len(CHUNK_META_PREFIX):suffix_pos]
        try:
            meta = json.loads(meta_json)
            if isinstance(meta, dict):
                metas.append(meta)
        except json.JSONDecodeError:
            pass

        cursor = suffix_pos + len(CHUNK_META_SUFFIX)
        if cursor < text_len and text[cursor] == "\n":
            cursor += 1

    content = "".join(pieces).lstrip("\n")
    return {"metas": metas, "content": content}


def _extract_reference_contexts_meta(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        contexts = row.get("reference_contexts")
        if not isinstance(contexts, list):
            continue

        cleaned_contexts: List[Any] = []
        context_metas: List[Any] = []
        for ctx in contexts:
            if not isinstance(ctx, str):
                cleaned_contexts.append(ctx)
                context_metas.append(None)
                continue

            parsed = extract_all_chunk_meta_from_text(ctx)
            cleaned_contexts.append(parsed["content"])
            context_metas.append(parsed["metas"])

        row["reference_contexts"] = cleaned_contexts
        existing_meta = row.get("meta")
        if not isinstance(existing_meta, dict):
            existing_meta = {}
        existing_meta["reference_contexts"] = context_metas
        row["meta"] = existing_meta


# ===== Preflight =====


_PREFLIGHT_CACHE: set[tuple[str, str]] = set()


def _preflight_check_model_endpoint(
    timeout_seconds: int = LLM_PREFLIGHT_TIMEOUT_SECONDS,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    force: bool = False,
):
    """Probe the LLM /models endpoint. Results are cached per (base_url, model)
    so queueing many jobs against the same endpoint doesn't re-check each time.
    """
    from testset_config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    from urllib import request

    effective_base = (base_url or LLM_BASE_URL).rstrip("/")
    effective_key = api_key or LLM_API_KEY
    effective_model = model or LLM_MODEL
    cache_key = (effective_base, effective_model)
    if cache_key in _PREFLIGHT_CACHE and not force:
        return

    models_url = f"{effective_base}/models"
    req = request.Request(
        models_url,
        headers={"Authorization": f"Bearer {effective_key}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"LLM endpoint preflight failed: {models_url} is unreachable or invalid. "
            f"Details: {exc}"
        ) from exc

    available = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if available and effective_model not in available:
        print(
            f"[warn] Model '{effective_model}' not in /models list. "
            f"Available (sample): {sorted(list(available))[:5]}"
        )

    _PREFLIGHT_CACHE.add(cache_key)


def _configure_logging(quiet_filter_warnings: bool = QUIET_FILTER_WARNINGS):
    logger = logging.getLogger("ragas.testset.transforms.filters")
    logger.setLevel(logging.ERROR if quiet_filter_warnings else logging.NOTSET)


# ===== Result processing helpers =====


def _split_success_and_failed_rows(result: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not hasattr(result, "to_list") or not callable(getattr(result, "to_list")):
        raise TypeError(
            "Expected generation result with callable .to_list(), "
            f"got {type(result).__name__}"
        )

    raw_rows = result.to_list()
    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        if isinstance(row, dict):
            rows.append(row)
            continue

        failed_rows.append(
            {
                "index": index,
                "type": type(row).__name__,
                "repr": repr(row),
            }
        )

    return rows, failed_rows


def _safe_repr(value: Any, max_len: int = 1000) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _is_nan_like(value: Any) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _build_failed_generation_record(
    index: int,
    sample: Any,
    additional_info: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "index": index,
        "type": type(sample).__name__,
        "repr": _safe_repr(sample),
        "reason": reason,
    }
    for key, value in additional_info.items():
        if key not in record:
            record[key] = value
    return record


# ===== Ragas monkey-patches (error tolerance) =====


def _patch_ragas_transforms_error_handling() -> None:
    """Patch Extractor and RelationshipBuilder to skip failures gracefully."""
    from ragas.testset.transforms.base import Extractor, RelationshipBuilder
    import logging as _logging

    _ext_logger = _logging.getLogger("ragas.testset.transforms.base")

    # ── 1. Extractor: skip individual node extraction failures ──────────
    if not getattr(Extractor.generate_execution_plan, "_mc_fault_tolerant", False):

        def _fault_tolerant_generate_execution_plan(self, kg):
            async def safe_apply_extract(node):
                try:
                    property_name, property_value = await self.extract(node)
                    if node.get_property(property_name) is None:
                        node.add_property(property_name, property_value)
                    else:
                        _ext_logger.warning(
                            "Property '%s' already exists in node '%.6s'. Skipping!",
                            property_name,
                            node.id,
                        )
                except Exception as exc:
                    _ext_logger.warning(
                        "[%s] Extraction failed for node '%.6s', skipping. "
                        "%s: %s",
                        self.__class__.__name__,
                        node.id,
                        type(exc).__name__,
                        str(exc)[:300],
                    )

            filtered = self.filter(kg)
            plan = [safe_apply_extract(node) for node in filtered.nodes]
            _ext_logger.debug(
                "Created %d coroutines for %s (fault-tolerant)",
                len(plan),
                self.__class__.__name__,
            )
            return plan

        _fault_tolerant_generate_execution_plan._mc_fault_tolerant = True
        Extractor.generate_execution_plan = _fault_tolerant_generate_execution_plan
        _ext_logger.info("Patched Extractor.generate_execution_plan for fault tolerance")

    # ── 2. RelationshipBuilder: skip when nodes are missing properties ──
    if not getattr(RelationshipBuilder.generate_execution_plan, "_mc_fault_tolerant", False):

        def _fault_tolerant_rb_execution_plan(self, kg):
            async def safe_apply_build_relationships(filtered_kg, original_kg):
                try:
                    relationships = await self.transform(filtered_kg)
                    original_kg.relationships.extend(relationships)
                except Exception as exc:
                    _ext_logger.warning(
                        "[%s] Relationship building failed, skipping. %s: %s",
                        self.__class__.__name__,
                        type(exc).__name__,
                        str(exc)[:300],
                    )

            filtered_kg = self.filter(kg)
            plan = [safe_apply_build_relationships(filtered_kg=filtered_kg, original_kg=kg)]
            _ext_logger.debug(
                "Created %d coroutines for %s (fault-tolerant)",
                len(plan),
                self.__class__.__name__,
            )
            return plan

        _fault_tolerant_rb_execution_plan._mc_fault_tolerant = True
        RelationshipBuilder.generate_execution_plan = _fault_tolerant_rb_execution_plan
        _ext_logger.info("Patched RelationshipBuilder.generate_execution_plan for fault tolerance")

    # ── 3. OverlapScoreBuilder / JaccardSimilarityBuilder: skip missing-property pairs ──
    try:
        from ragas.testset.transforms.relationship_builders.traditional import (
            OverlapScoreBuilder,
            JaccardSimilarityBuilder,
        )
    except ImportError:
        return

    if not getattr(OverlapScoreBuilder, "_mc_patched_transform", False):

        async def _safe_overlap_transform(self, kg):
            from ragas.testset.graph import Relationship as _Rel

            distance_measure = self.distance_measure_map[self.distance_measure]
            noisy_items = self._get_noisy_items(kg.nodes, self.property_name)
            relationships = []
            skipped = 0
            for i, node_x in enumerate(kg.nodes):
                for j, node_y in enumerate(kg.nodes):
                    if i >= j:
                        continue
                    node_x_items = node_x.get_property(self.property_name)
                    node_y_items = node_y.get_property(self.property_name)
                    if node_x_items is None or node_y_items is None:
                        skipped += 1
                        continue
                    if self.key_name is not None:
                        node_x_items = node_x_items.get(self.key_name, [])
                        node_y_items = node_y_items.get(self.key_name, [])

                    overlaps = []
                    overlapped_items = []
                    for x in node_x_items:
                        if x not in noisy_items:
                            for y in node_y_items:
                                if y not in noisy_items:
                                    similarity = 1 - distance_measure.distance(
                                        x.lower(), y.lower()
                                    )
                                    verdict = similarity >= self.distance_threshold
                                    overlaps.append(verdict)
                                    if verdict:
                                        overlapped_items.append((x, y))

                    similarity = self._overlap_score(overlaps)
                    if similarity >= self.threshold:
                        relationships.append(
                            _Rel(
                                source=node_x,
                                target=node_y,
                                type=f"{self.property_name}_overlap",
                                properties={
                                    f"{self.property_name}_{self.new_property_name}": similarity,
                                    "overlapped_items": overlapped_items,
                                },
                                bidirectional=True,
                            )
                        )

            if skipped:
                _ext_logger.info(
                    "OverlapScoreBuilder: skipped %d node pairs with missing '%s'",
                    skipped,
                    self.property_name,
                )
            return relationships

        OverlapScoreBuilder.transform = _safe_overlap_transform
        OverlapScoreBuilder._mc_patched_transform = True
        _ext_logger.info("Patched OverlapScoreBuilder.transform to skip missing properties")

    if not getattr(JaccardSimilarityBuilder, "_mc_patched_find", False):

        def _safe_jaccard_find(self, kg):
            import itertools

            similar_pairs = set()
            skipped = 0
            for (i, node1), (j, node2) in itertools.combinations(enumerate(kg.nodes), 2):
                items1 = node1.get_property(self.property_name)
                items2 = node2.get_property(self.property_name)
                if items1 is None or items2 is None:
                    skipped += 1
                    continue
                if self.key_name is not None:
                    items1 = items1.get(self.key_name, [])
                    items2 = items2.get(self.key_name, [])
                similarity = self._jaccard_similarity(set(items1), set(items2))
                if similarity >= self.threshold:
                    similar_pairs.add((i, j, similarity))

            if skipped:
                _ext_logger.info(
                    "JaccardSimilarityBuilder: skipped %d node pairs with missing '%s'",
                    skipped,
                    self.property_name,
                )
            return list(similar_pairs)

        JaccardSimilarityBuilder._find_similar_embedding_pairs = _safe_jaccard_find
        JaccardSimilarityBuilder._mc_patched_find = True
        _ext_logger.info("Patched JaccardSimilarityBuilder to skip missing properties")


def _patch_ragas_safe_generate() -> None:
    """Patch TestsetGenerator.generate to tolerate NaN / malformed samples."""
    import ragas.testset.synthesizers.generate as ragas_generate

    if getattr(ragas_generate.TestsetGenerator.generate, "_mc_safe_patched", False):
        return

    def _safe_generate(
        self,
        testset_size: int,
        query_distribution: Any = None,
        num_personas: int = 3,
        run_config: Any = None,
        batch_size: Any = None,
        callbacks: Any = None,
        token_usage_parser: Any = None,
        with_debugging_logs: bool = False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> Any:
        if run_config is not None and isinstance(self.llm, ragas_generate.BaseRagasLLM):
            self.llm.set_run_config(run_config)

        query_distribution = query_distribution or ragas_generate.default_query_distribution(
            self.llm,
            self.knowledge_graph,
            self.llm_context,
        )
        callbacks = callbacks or []
        ragas_callbacks = {}

        if token_usage_parser is not None:
            from ragas.cost import CostCallbackHandler

            cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
            ragas_callbacks["cost_cb"] = cost_cb
        else:
            cost_cb = None

        for cb in ragas_callbacks.values():
            if isinstance(callbacks, ragas_generate.BaseCallbackManager):
                callbacks.add_handler(cb)
            else:
                callbacks.append(cb)

        testset_generation_rm, testset_generation_grp = ragas_generate.new_group(
            name=ragas_generate.RAGAS_TESTSET_GENERATION_GROUP_NAME,
            inputs={"testset_size": testset_size},
            callbacks=callbacks,
        )

        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.experimental.testset.synthesizers", logging.DEBUG)
            patch_logger("ragas.experimental.testset.graph", logging.DEBUG)
            patch_logger("ragas.experimental.testset.transforms", logging.DEBUG)

        if self.persona_list is None:
            self.persona_list = ragas_generate.generate_personas_from_kg(
                llm=self.llm,
                kg=self.knowledge_graph,
                num_personas=num_personas,
                callbacks=callbacks,
            )
        else:
            ragas_generate.random.shuffle(self.persona_list)

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        scenario_generation_rm, scenario_generation_grp = ragas_generate.new_group(
            name="Scenario Generation",
            inputs={"splits": splits},
            callbacks=testset_generation_grp,
        )

        exec = ragas_generate.Executor(
            desc="Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
        )

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        # Shuffle KG nodes so each synthesizer's generate_scenarios picks from
        # a randomized order, preventing same-document scenarios from clustering.
        random.shuffle(self.knowledge_graph.nodes)
        for i, (scenario, _) in enumerate(query_distribution):
            exec.submit(
                scenario.generate_scenarios,
                n=splits[i],
                knowledge_graph=self.knowledge_graph,
                persona_list=self.persona_list[:num_personas],
                callbacks=scenario_generation_grp,
            )

        try:
            scenario_sample_list = exec.results()
        except Exception as e:
            scenario_generation_rm.on_chain_error(e)
            raise e
        else:
            scenario_generation_rm.on_chain_end(
                outputs={"scenario_sample_list": scenario_sample_list}
            )

        sample_generation_rm, sample_generation_grp = ragas_generate.new_group(
            name="Sample Generation",
            inputs={"scenario_sample_list": scenario_sample_list},
            callbacks=testset_generation_grp,
        )
        exec = ragas_generate.Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size,
        )
        additional_testset_info: List[Dict[str, Any]] = []
        for i, (synthesizer, _) in enumerate(query_distribution):
            for sample in scenario_sample_list[i]:
                exec.submit(
                    synthesizer.generate_sample,
                    scenario=sample,
                    callbacks=sample_generation_grp,
                )
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                    }
                )

        if return_executor:
            self._mc_last_failed_rows = []
            return exec

        try:
            eval_samples = exec.results()
        except Exception as e:
            sample_generation_rm.on_chain_error(e)
            raise e
        else:
            sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})

        failed_rows: List[Dict[str, Any]] = []
        if len(eval_samples) != len(additional_testset_info):
            failed_rows.append(
                {
                    "index": -1,
                    "type": "length_mismatch",
                    "repr": "",
                    "reason": (
                        "eval_samples and additional_testset_info lengths differ: "
                        f"{len(eval_samples)} vs {len(additional_testset_info)}"
                    ),
                }
            )

        testsets = []
        for index, (sample, additional_info) in enumerate(
            zip(eval_samples, additional_testset_info)
        ):
            if _is_nan_like(sample):
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason="eval_sample is NaN",
                    )
                )
                continue

            try:
                testsets.append(
                    ragas_generate.TestsetSample(
                        eval_sample=sample,
                        **additional_info,
                    )
                )
            except Exception as exc:
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )

        self._mc_last_failed_rows = failed_rows

        testset = ragas_generate.Testset(samples=testsets, cost_cb=cost_cb)
        testset_generation_rm.on_chain_end({"testset": testset})
        ragas_generate.track(
            ragas_generate.TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[
                    e.__class__.__name__.lower() for e, _ in query_distribution
                ],
                evolution_percentages=[p for _, p in query_distribution],
                num_rows=len(testsets),
                language="english",
            )
        )
        return testset

    _safe_generate._mc_safe_patched = True
    ragas_generate.TestsetGenerator.generate = _safe_generate


# ===== Output =====


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            row_with_id = {"question_id": idx, **row}
            f.write(json.dumps(row_with_id, ensure_ascii=False) + "\n")


# ===== Main =====


def _print_summary(
    *,
    elapsed_seconds: float,
    input_path: str,
    output_path: str,
    requested_size: int,
    generated_rows: int,
    failed_rows: int,
    failed_output_path: str | None,
) -> None:
    print("=" * 80)
    print("RAGAS testset generation finished")
    print(f"Input chunks  : {input_path}")
    print(f"Output file   : {output_path}")
    print(f"Requested size: {requested_size}")
    print(f"Generated rows: {generated_rows}")
    print(f"Failed rows   : {failed_rows}")
    print(f"Elapsed       : {elapsed_seconds:.2f}s")
    if failed_output_path:
        print(f"Failed details: {failed_output_path}")
    print("=" * 80)


def _run_request(
    request_kwargs: dict[str, Any],
    *,
    label: str | None = None,
    embedding_model: Any = None,
) -> tuple[int, str, dict[str, Any]]:
    if label:
        print(label)

    input_path = request_kwargs["input_path"]
    output_path = request_kwargs["output_path"]
    testset_size = request_kwargs["testset_size"]

    print(f"Input chunks  : {input_path}")
    print(f"Output file   : {output_path}")
    print(f"Requested size: {testset_size}")
    print("Starting RAGAS testset generation...")

    started_at = time.perf_counter()
    try:
        _configure_logging(request_kwargs["quiet_filter_warnings"])
        _preflight_check_model_endpoint(
            request_kwargs["llm_preflight_timeout_seconds"],
            base_url=request_kwargs.get("llm_base_url"),
            api_key=request_kwargs.get("llm_api_key"),
            model=request_kwargs.get("llm_model"),
        )
        _patch_ragas_transforms_error_handling()
        patch_multihop_prompt()
        _patch_ragas_safe_generate()

        chunks = _load_chunks(Path(input_path))

        llm = build_llm(
            base_url=request_kwargs.get("llm_base_url"),
            api_key=request_kwargs.get("llm_api_key"),
            model=request_kwargs.get("llm_model"),
        )
        if embedding_model is None:
            embedding_model = build_embedding_model()
        transforms = (
            build_transforms(llm, embedding_model)
            if request_kwargs["skip_custom_node_filter"]
            else None
        )
        query_distribution = build_query_distribution(llm)

        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embedding_model,
        )
        generator._mc_last_failed_rows = []

        result = generator.generate_with_chunks(
            chunks=chunks,
            testset_size=testset_size,
            transforms=transforms,
            transforms_llm=None,
            transforms_embedding_model=None,
            query_distribution=query_distribution,
            run_config=RunConfig(
                timeout=request_kwargs["run_timeout_seconds"],
                max_retries=request_kwargs["max_retries"],
                max_wait=request_kwargs["max_wait_seconds"],
                max_workers=request_kwargs["max_workers"],
            ),
            callbacks=None,
            token_usage_parser=None,
            with_debugging_logs=request_kwargs["with_debugging_logs"],
            raise_exceptions=request_kwargs["fail_fast"],
            return_executor=False,
        )
        rows, failed_rows = _split_success_and_failed_rows(result)
        failed_rows.extend(getattr(generator, "_mc_last_failed_rows", []))

        _extract_reference_contexts_meta(rows)
        random.shuffle(rows)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_file, rows)

        failed_output_file: Path | None = None
        if failed_rows:
            failed_output_file = output_file.with_name(
                f"{output_file.stem}.failed.jsonl"
            )
            _write_jsonl(failed_output_file, failed_rows)
            print(
                f"[warn] Encountered {len(failed_rows)} failed generations. "
                f"Details saved to {failed_output_file.resolve()}"
            )

        if len(rows) < testset_size:
            print(
                f"[warn] Requested {testset_size} samples, generated {len(rows)}. "
                "This usually means some generations failed or scenario candidates were insufficient after graph transforms."
            )
    except ModuleNotFoundError as exc:
        elapsed = time.perf_counter() - started_at
        msg = f"Missing dependency after {elapsed:.2f}s: {exc}"
        tb_str = traceback.format_exc()
        print(
            f"{msg}. Install the Python requirements in this environment first.",
            file=sys.stderr,
        )
        print(tb_str, file=sys.stderr)
        return 1, msg, {"traceback": tb_str, "elapsed_seconds": elapsed}
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        msg = f"Testset generation failed after {elapsed:.2f}s: {exc}"
        tb_str = traceback.format_exc()
        print(msg, file=sys.stderr)
        print(tb_str, file=sys.stderr)
        return 1, msg, {"traceback": tb_str, "elapsed_seconds": elapsed}

    elapsed = time.perf_counter() - started_at
    summary = {
        "input_path": input_path,
        "output_path": output_path,
        "requested_size": testset_size,
        "generated_rows": len(rows),
        "failed_rows": len(failed_rows),
        "failed_output_path": str(failed_output_file) if failed_output_file else None,
        "elapsed_seconds": elapsed,
    }
    _print_summary(**summary)
    return 0, "", summary


def _try_empty_cuda_cache() -> None:
    """Best-effort release of CUDA cache between jobs. Silently no-ops if torch
    isn't importable or CUDA isn't available."""
    try:
        import torch  # type: ignore
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_run_log(
    jobs_path: str, total: int, succeeded: int, failed: list[str], run_log: list[dict[str, Any]]
) -> Path:
    log_path = Path(jobs_path).with_suffix(".run_log.json")
    log_data = {
        "total": total,
        "succeeded": succeeded,
        "failed": len(failed),
        "jobs": run_log,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    return log_path


def _run_jobs(jobs: list[dict[str, Any]], jobs_path: str) -> int:
    total = len(jobs)
    succeeded = 0
    failed: list[str] = []
    run_log: list[dict[str, Any]] = []

    # ── Phase 1: parse every job's kwargs up-front ──────────────────────
    # Config errors still only kill the offending job, but this lets us do
    # cross-job validation (e.g. output_path conflicts) before any real work.
    prepared: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for index, job in enumerate(jobs, start=1):
        job_name = (
            _clean_optional_string(str(job.get("name")))
            if job.get("name") is not None
            else None
        )
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
        entry["testset_size"] = request_kwargs.get("testset_size")
        prepared.append((index, label, request_kwargs, entry))

    # ── Phase 2: cross-job validation — fail fast on output_path conflicts ──
    # Two jobs writing to the same path would silently overwrite each other, so
    # treat this as a configuration error for the whole queue.
    seen_outputs: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for _, label, kwargs, _entry in prepared:
        out = kwargs["output_path"]
        prev = seen_outputs.get(out)
        if prev is not None:
            conflicts.append((label, prev, out))
        else:
            seen_outputs[out] = label

    if conflicts:
        for label, prev_label, out in conflicts:
            print(
                f"[error] {label} output_path '{out}' conflicts with {prev_label}",
                file=sys.stderr,
            )
        print(
            "Aborting queue before any job runs (fix duplicate output_path values).",
            file=sys.stderr,
        )
        # Record conflict state in every affected entry so the run_log reflects it.
        conflict_labels = {c[0] for c in conflicts} | {c[1] for c in conflicts}
        for _, label, _kwargs, entry in prepared:
            if label in conflict_labels:
                entry.update(status="aborted", error="output_path conflict")
                run_log.append(entry)
                failed.append(label)
            else:
                entry.update(status="skipped", error="aborted due to output_path conflict in queue")
                run_log.append(entry)
        log_path = _write_run_log(jobs_path, total, succeeded, failed, run_log)
        print(f"Run log: {log_path}")
        return 2

    # ── Phase 3: build the embedding model once and share across jobs ──
    # bge-m3 is ~2GB on GPU; rebuilding per job wastes load time and can
    # fragment CUDA memory. The model is effectively stateless so sharing is safe.
    shared_embedding_model: Any = None
    if prepared:
        try:
            shared_embedding_model = build_embedding_model()
        except Exception as exc:
            tb_str = traceback.format_exc()
            print(f"[error] Failed to build shared embedding model: {exc}", file=sys.stderr)
            print(tb_str, file=sys.stderr)
            for _, label, _kwargs, entry in prepared:
                entry.update(status="failed", error=str(exc), traceback=tb_str)
                run_log.append(entry)
                failed.append(label)
            log_path = _write_run_log(jobs_path, total, succeeded, failed, run_log)
            print(f"Run log: {log_path}")
            return 1

    # ── Phase 4: run each job, cleaning up GPU state between them ──
    for _index, label, request_kwargs, entry in prepared:
        exit_code, error_msg, summary = _run_request(
            request_kwargs,
            label=label,
            embedding_model=shared_embedding_model,
        )

        # Release transient objects (KnowledgeGraph, intermediate embeddings,
        # closures held by monkey-patched methods). Shared embedding model is
        # still referenced by `shared_embedding_model` so it won't be freed.
        gc.collect()
        _try_empty_cuda_cache()

        if exit_code != 0:
            print(f"{label} failed, skipping to next job.", file=sys.stderr)
            failed.append(label)
            entry.update(status="failed", error=error_msg, **summary)
            run_log.append(entry)
            continue

        succeeded += 1
        entry.update(status="succeeded", **summary)
        run_log.append(entry)

    log_path = _write_run_log(jobs_path, total, succeeded, failed, run_log)

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
    exit_code, _, _ = _run_request(request_kwargs)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
