import json
from pathlib import Path
from typing import Any, List, Sequence, Union

from langchain_core.documents import Document
from openai import OpenAI

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.testset.synthesizers.generate import TestsetGenerator

Chunk = Union[str, Document]

# ===== Static config (edit directly) =====
CHUNKS_FILE = Path("chunked_input.json")
OUTPUT_FILE = Path("rag_testset.jsonl")
TESTSET_SIZE = 20
WITH_DEBUGGING_LOGS = False

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_PROVIDER = "openai"
LLM_ADAPTER = "auto"

EMBEDDING_MODEL_PATH = Path(r"F:\models\bge-m3")
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


def _build_llm():
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return llm_factory(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        client=client,
        adapter=LLM_ADAPTER,
    )


def _build_embedding_model():
    model_path = EMBEDDING_MODEL_PATH.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local embedding model path not found: {model_path.resolve()}"
        )
    return HuggingFaceEmbeddings(model=str(model_path), device=EMBEDDING_DEVICE)


def main():
    chunks = _load_chunks(CHUNKS_FILE)
    llm = _build_llm()
    embedding_model = _build_embedding_model()

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
    )

    testset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=TESTSET_SIZE,
        transforms=None,
        transforms_llm=None,
        transforms_embedding_model=None,
        query_distribution=None,
        run_config=None,
        callbacks=None,
        token_usage_parser=None,
        with_debugging_logs=WITH_DEBUGGING_LOGS,
        raise_exceptions=True,
        return_executor=False,
    )

    rows = testset.to_list()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done: {len(rows)} samples -> {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
