import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from urllib import request

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from openai import AsyncOpenAI

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    NERExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
    ThemesExtractor,
)

Chunk = Union[str, Document]

# ===== Static config (edit directly) =====
CHUNKS_FILE = Path("/data/h50056789/Rag_Chunking/test_database/2wikimqa_llamaindex_chunk.json")
OUTPUT_FILE = Path("/data/h50056789/Rag_Chunking/test_database/rag_testset.jsonl")
TESTSET_SIZE = 10
WITH_DEBUGGING_LOGS = True

LLM_BASE_URL = "http://127.0.0.1:8001/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
LLM_PROVIDER = "openai"
LLM_ADAPTER = "auto"
LLM_TIMEOUT_SECONDS = 120
LLM_PREFLIGHT_TIMEOUT_SECONDS = 8

MAX_WORKERS = 4
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 20
RUN_TIMEOUT_SECONDS = 120

# In many local-vLLM runs, CustomNodeFilter produces excessive "no summary" logs
# and does not filter effectively for pre-chunked inputs. Keep this True by default.
SKIP_CUSTOM_NODE_FILTER = True
QUIET_FILTER_WARNINGS = True

EMBEDDING_MODEL_PATH = Path(r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5")
EMBEDDING_DEVICE = "cuda"


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

        normalized.append(Document(page_content=text, metadata=metadata))

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
                normalized.append(Document(page_content=page_content, metadata=metadata))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            normalized.extend(_normalize_split_tuples([item]))
            continue

        raise TypeError(
            "Chunk items must be str, dict, or [text, article_id, chunk_id]"
        )

    return normalized


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_passage_prefix(text: str) -> str:
    # Handle strings like: "Passage 1:\n<chunk_text>"
    return re.sub(r"^\s*Passage\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def _build_chunk_id_index(chunks: Sequence[Chunk]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        if isinstance(chunk, Document):
            chunk_text = chunk.page_content
            meta = dict(chunk.metadata or {})
            ref = {
                "doc_id": meta.get("source_article_id"),
                "chunk_id": meta.get("source_chunk_id"),
                "source_filepath": meta.get("source_filepath"),
            }
        else:
            chunk_text = chunk
            ref = {"doc_id": None, "chunk_id": None, "source_filepath": None}

        key = _normalize_text_for_match(chunk_text)
        index.setdefault(key, []).append(ref)
    return index


def _attach_reference_ids(rows: List[Dict[str, Any]], chunks: Sequence[Chunk]) -> None:
    chunk_id_index = _build_chunk_id_index(chunks)
    usage_cursor: Dict[str, int] = {}

    for row in rows:
        contexts = row.get("reference_contexts")
        if not isinstance(contexts, list):
            row["reference_context_ids"] = []
            continue

        refs: List[Dict[str, Any]] = []
        for ctx in contexts:
            if not isinstance(ctx, str):
                refs.append({"doc_id": None, "chunk_id": None, "matched": False})
                continue

            content = _strip_passage_prefix(ctx)
            key = _normalize_text_for_match(content)
            candidates = chunk_id_index.get(key, [])
            if not candidates:
                refs.append({"doc_id": None, "chunk_id": None, "matched": False})
                continue

            pos = usage_cursor.get(key, 0)
            chosen = candidates[pos % len(candidates)]
            usage_cursor[key] = pos + 1

            refs.append(
                {
                    "doc_id": chosen.get("doc_id"),
                    "chunk_id": chosen.get("chunk_id"),
                    "source_filepath": chosen.get("source_filepath"),
                    "matched": True,
                }
            )

        row["reference_context_ids"] = refs


def _build_llm():
    # Use async client so ragas async transforms do not block the event loop.
    client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_SECONDS
    )
    return llm_factory(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        client=client,
        adapter=LLM_ADAPTER,
    )


def _build_transforms(llm, embedding_model):
    def filter_chunks(node):
        return node.type.name == "CHUNK"

    summary_extractor = SummaryExtractor(llm=llm, filter_nodes=filter_chunks)
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=filter_chunks,
    )
    theme_extractor = ThemesExtractor(llm=llm, filter_nodes=filter_chunks)
    ner_extractor = NERExtractor(llm=llm, filter_nodes=filter_chunks)
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=filter_chunks,
    )
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)

    transforms = [
        summary_extractor,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]
    return transforms


def _build_embedding_model():
    model_path = EMBEDDING_MODEL_PATH.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local embedding model path not found: {model_path.resolve()}"
        )
    langchain_embeddings = LCHuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(langchain_embeddings)


def _preflight_check_model_endpoint():
    base = LLM_BASE_URL.rstrip("/")
    models_url = f"{base}/models"
    req = request.Request(
        models_url,
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=LLM_PREFLIGHT_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"LLM endpoint preflight failed: {models_url} is unreachable or invalid. "
            f"Details: {exc}"
        ) from exc

    available = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if available and LLM_MODEL not in available:
        print(
            f"[warn] Model '{LLM_MODEL}' not in /models list. "
            f"Available (sample): {sorted(list(available))[:5]}"
        )


def _configure_logging():
    if QUIET_FILTER_WARNINGS:
        logging.getLogger("ragas.testset.transforms.filters").setLevel(logging.ERROR)


def main():
    _configure_logging()
    _preflight_check_model_endpoint()

    chunks = _load_chunks(CHUNKS_FILE)
    llm = _build_llm()
    embedding_model = _build_embedding_model()
    transforms = _build_transforms(llm, embedding_model) if SKIP_CUSTOM_NODE_FILTER else None

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
    )

    testset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=TESTSET_SIZE,
        transforms=transforms,
        transforms_llm=None,
        transforms_embedding_model=None,
        query_distribution=None,
        run_config=RunConfig(
            timeout=RUN_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            max_wait=MAX_WAIT_SECONDS,
            max_workers=MAX_WORKERS,
        ),
        callbacks=None,
        token_usage_parser=None,
        with_debugging_logs=WITH_DEBUGGING_LOGS,
        raise_exceptions=True,
        return_executor=False,
    )

    rows = testset.to_list()
    _attach_reference_ids(rows, chunks)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(rows) < TESTSET_SIZE:
        print(
            f"[warn] Requested {TESTSET_SIZE} samples, generated {len(rows)}. "
            "This usually means scenario candidates were insufficient after graph transforms."
        )
    print(f"Done: {len(rows)} samples -> {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
