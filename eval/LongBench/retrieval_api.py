import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    API_BASE: str = "http://127.0.0.1:8081"

    # Input can be jsonl (one query per line) or json list.
    # Each item supports:
    # - user_input / input / query
    # - _id / id
    # - answers (list[str]) or reference (str)
    INPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_data.jsonl"

    # Output: sample_results-like + retrieval_meta
    OUTPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_results_api.json"

    COLLECTION_NAME: str = "lumber_chunk"
    EMBED_MODEL_PATH: str = r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
    EMBED_DIM: int = 1024
    TOP_K: int = 5

    ENABLE_RAG: bool = True
    USE_HYBRID_SEARCH: bool = True

    # Optional metadata filters for retrieval.
    FILEPATH_FILTER: str | list[str] | None = None
    DOC_ID_FILTER: str | list[str] | None = None

    LLM_API_BASE: str = "http://localhost:8005/v1"
    LLM_MODEL_NAME: str = r"/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct"
    TEMPERATURE: float = 0.1
    MAX_NEW_TOKENS: int = 1280

    TIMEOUT_SECONDS: int = 600


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _normalize_item(raw: dict, idx: int) -> dict:
    query = raw.get("user_input") or raw.get("input") or raw.get("query") or ""
    qid = raw.get("_id") or raw.get("id") or f"q_{idx}"

    if isinstance(raw.get("answers"), list):
        answers = raw.get("answers")
    elif raw.get("reference") is not None:
        answers = [str(raw.get("reference"))]
    else:
        answers = []

    reference_contexts = raw.get("reference_contexts")
    if isinstance(reference_contexts, list):
        gold_reference_contexts = [str(x) for x in reference_contexts]
    elif reference_contexts is None:
        gold_reference_contexts = []
    else:
        gold_reference_contexts = [str(reference_contexts)]

    return {
        "_id": qid,
        "input": query,
        "answers": answers,
        "gold_reference_contexts": gold_reference_contexts,
        "gold_meta": raw.get("meta"),
    }


def _flatten_reference_meta(meta_obj: Any) -> list[dict]:
    """Flatten row.meta.reference_contexts into a list of {doc_id, chunk_id, source_filepath}."""
    if not isinstance(meta_obj, dict):
        return []
    ref_ctx = meta_obj.get("reference_contexts")
    if ref_ctx is None:
        return []

    out: list[dict] = []

    def walk(x: Any) -> None:
        if isinstance(x, list):
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            doc_id = x.get("doc_id")
            chunk_id = x.get("chunk_id")
            source_filepath = x.get("source_filepath")
            if doc_id is not None or chunk_id is not None or source_filepath is not None:
                out.append(
                    {
                        "doc_id": str(doc_id) if doc_id is not None else None,
                        "chunk_id": str(chunk_id) if chunk_id is not None else None,
                        "source_filepath": str(source_filepath) if source_filepath is not None else None,
                    }
                )
            return

    walk(ref_ctx)
    return out


def _build_gold_reference_items(contexts: list[str], metas: list[dict]) -> list[dict]:
    """
    Unified item format:
    {"text": str, "filepath": str|None, "doc_id": str|None, "chunk_id": str|None}
    """
    max_len = max(len(contexts), len(metas))
    items: list[dict] = []
    for i in range(max_len):
        text = contexts[i] if i < len(contexts) else None
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        items.append(
            {
                "text": text,
                "filepath": meta.get("source_filepath"),
                "doc_id": meta.get("doc_id"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    return items


def _load_inputs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(_normalize_item(json.loads(line), idx))
        return rows

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_normalize_item(item, idx + 1) for idx, item in enumerate(data) if isinstance(item, dict)]
    if isinstance(data, dict):
        return [_normalize_item(data, 1)]
    raise ValueError(f"Unsupported input format in {path}")


def main() -> None:
    endpoint = f"{Config.API_BASE}/api/v1/retrieval/generate"
    rows = _load_inputs(Config.INPUT_PATH)
    logger.info("Loaded %d queries from %s", len(rows), Config.INPUT_PATH)

    output_rows: list[dict[str, Any]] = []
    total_failed = 0

    for row in tqdm(rows, desc="retrieval+generation", unit="q"):
        query = row["input"]
        if not query:
            total_failed += 1
            continue

        payload = {
            "query": query,
            "collection_name": Config.COLLECTION_NAME,
            "embed_model_path": Config.EMBED_MODEL_PATH,
            "embed_dim": Config.EMBED_DIM,
            "top_k": Config.TOP_K,
            "enable_rag": Config.ENABLE_RAG,
            "use_hybrid_search": Config.USE_HYBRID_SEARCH,
            "filepath": Config.FILEPATH_FILTER,
            "doc_id": Config.DOC_ID_FILTER,
            "llm_api_base": Config.LLM_API_BASE,
            "llm_model_name": Config.LLM_MODEL_NAME,
            "temperature": Config.TEMPERATURE,
            "max_new_tokens": Config.MAX_NEW_TOKENS,
        }

        try:
            result = _post_json(endpoint, payload, timeout=Config.TIMEOUT_SECONDS)
            if not result.get("success", False):
                raise RuntimeError(result.get("message", "unknown API error"))

            data = result.get("data") or {}
            context_items = data.get("context_items") or []
            rag_retrieval = [
                {
                    "text": item.get("text"),
                    "filepath": item.get("filepath"),
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                }
                for item in context_items
            ]
            gold_reference_meta = _flatten_reference_meta(row.get("gold_meta"))
            gold_reference = _build_gold_reference_items(
                row.get("gold_reference_contexts", []),
                gold_reference_meta,
            )

            output_rows.append(
                {
                    "_id": row["_id"],
                    "input": row["input"],
                    "llm_ans": data.get("answer", ""),
                    "answers": row.get("answers", []),
                    "rag_retrieval": rag_retrieval,
                    "gold_reference": gold_reference,
                }
            )
        except Exception as exc:
            total_failed += 1
            logger.exception("Failed on _id=%s: %s", row.get("_id"), exc)

    os.makedirs(str(Path(Config.OUTPUT_PATH).parent), exist_ok=True)
    with open(Config.OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)

    logger.info("Saved output: %s", Config.OUTPUT_PATH)
    logger.info("Processed=%d failed=%d", len(output_rows), total_failed)


if __name__ == "__main__":
    main()
