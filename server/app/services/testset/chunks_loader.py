"""Chunk loading, normalization, and metadata injection/extraction."""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from langchain_core.documents import Document

Chunk = Union[str, Document]

CHUNK_META_PREFIX = "<<<MC_META>>>"
CHUNK_META_SUFFIX = "<<<END_MC_META>>>"


def load_chunks(path: Path) -> List[Chunk]:
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

        text_with_meta = inject_chunk_meta(
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
                    text = inject_chunk_meta(
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


def inject_chunk_meta(
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


def extract_all_chunk_meta(text: str) -> Dict[str, Any]:
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


def extract_reference_contexts_meta(rows: List[Dict[str, Any]]) -> None:
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

            parsed = extract_all_chunk_meta(ctx)
            cleaned_contexts.append(parsed["content"])
            context_metas.append(parsed["metas"])

        row["reference_contexts"] = cleaned_contexts
        existing_meta = row.get("meta")
        if not isinstance(existing_meta, dict):
            existing_meta = {}
        existing_meta["reference_contexts"] = context_metas
        row["meta"] = existing_meta
