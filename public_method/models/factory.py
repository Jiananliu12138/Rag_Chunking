from __future__ import annotations

import json
from threading import RLock
from typing import Any

from langchain_core.embeddings import Embeddings

try:
    from app.core.exceptions import ModelLoadException
except Exception:
    class ModelLoadException(RuntimeError):
        pass

_MODEL_CACHE: dict[str, Any] = {}
_ADAPTER_CACHE: dict[str, Any] = {}
_CACHE_LOCK = RLock()


def _normalize_device(device: str | None) -> str:
    if device:
        return str(device).strip()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _freeze(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        return repr(value)


def _cache_key(kind: str, **parts: Any) -> str:
    normalized = {k: parts[k] for k in sorted(parts)}
    return f"{kind}:{_freeze(normalized)}"


def get_sentence_transformer(
    model_path: str,
    *,
    device: str | None = None,
    cache_folder: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    max_seq_length: int | None = None,
):
    normalized_device = _normalize_device(device)
    normalized_kwargs = dict(model_kwargs or {})
    normalized_kwargs.setdefault("device", normalized_device)
    normalized_path = str(model_path).strip()
    normalized_max_seq_length = (
        int(max_seq_length) if max_seq_length is not None else None
    )
    key = _cache_key(
        "sentence_transformer",
        model_path=normalized_path,
        device=normalized_device,
        cache_folder=cache_folder,
        model_kwargs=normalized_kwargs,
        max_seq_length=normalized_max_seq_length,
    )
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                normalized_path,
                cache_folder=cache_folder,
                **normalized_kwargs,
            )
            if normalized_max_seq_length is not None:
                model.max_seq_length = normalized_max_seq_length
        except Exception as exc:
            raise ModelLoadException(
                f"嵌入模型加载失败 ({normalized_path}): {exc}"
            ) from exc
        _MODEL_CACHE[key] = model
        return model


def get_cross_encoder(
    model_path: str,
    *,
    device: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    max_length: int | None = None,
):
    normalized_device = _normalize_device(device)
    normalized_kwargs = dict(model_kwargs or {})
    normalized_kwargs.setdefault("device", normalized_device)
    normalized_path = str(model_path).strip()
    normalized_max_length = int(max_length) if max_length is not None else None
    key = _cache_key(
        "cross_encoder",
        model_path=normalized_path,
        device=normalized_device,
        model_kwargs=normalized_kwargs,
        max_length=normalized_max_length,
    )
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(
                normalized_path,
                max_length=normalized_max_length,
                **normalized_kwargs,
            )
        except Exception as exc:
            raise ModelLoadException(
                f"重排模型加载失败 ({normalized_path}): {exc}"
            ) from exc
        _MODEL_CACHE[key] = model
        return model


class SharedSentenceTransformerEmbeddings(Embeddings):
    """LangChain embeddings adapter backed by a shared SentenceTransformer."""

    def __init__(
        self,
        *,
        model_path: str,
        client: Any,
        encode_kwargs: dict[str, Any] | None = None,
        query_encode_kwargs: dict[str, Any] | None = None,
        show_progress: bool = False,
    ) -> None:
        self.model_name = str(model_path).strip()
        self._client = client
        self.encode_kwargs = dict(encode_kwargs or {})
        self.query_encode_kwargs = dict(query_encode_kwargs or {})
        self.show_progress = show_progress

        if not hasattr(client, "encode"):
            raise TypeError("SharedSentenceTransformerEmbeddings requires a client with an encode() method")

    def _embed(
        self,
        texts: list[str],
        encode_kwargs: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        normalized_texts = [text.replace("\n", " ") for text in texts]
        embeddings = self._client.encode(
            normalized_texts,
            show_progress_bar=self.show_progress,
            **(encode_kwargs or {}),
        )
        if isinstance(embeddings, list):
            return embeddings
        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        query_kwargs = (
            self.query_encode_kwargs
            if self.query_encode_kwargs
            else self.encode_kwargs
        )
        return self._embed([text], query_kwargs)[0]


def get_langchain_embeddings(
    model_path: str,
    *,
    device: str | None = None,
    cache_folder: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    encode_kwargs: dict[str, Any] | None = None,
    query_encode_kwargs: dict[str, Any] | None = None,
    show_progress: bool = False,
    max_seq_length: int | None = None,
):
    normalized_model_kwargs = dict(model_kwargs or {})
    normalized_device = _normalize_device(device)
    normalized_model_kwargs.setdefault("device", normalized_device)
    normalized_encode_kwargs = dict(encode_kwargs or {})
    normalized_query_kwargs = dict(query_encode_kwargs or {})
    normalized_path = str(model_path).strip()
    normalized_max_seq_length = (
        int(max_seq_length) if max_seq_length is not None else None
    )
    key = _cache_key(
        "langchain_embeddings",
        model_path=normalized_path,
        device=normalized_device,
        cache_folder=cache_folder,
        model_kwargs=normalized_model_kwargs,
        encode_kwargs=normalized_encode_kwargs,
        query_encode_kwargs=normalized_query_kwargs,
        show_progress=show_progress,
        max_seq_length=normalized_max_seq_length,
    )

    with _CACHE_LOCK:
        cached = _ADAPTER_CACHE.get(key)
        if cached is not None:
            return cached

        client = get_sentence_transformer(
            model_path=normalized_path,
            device=normalized_device,
            cache_folder=cache_folder,
            model_kwargs=normalized_model_kwargs,
            max_seq_length=normalized_max_seq_length,
        )
        adapter = SharedSentenceTransformerEmbeddings(
            model_path=normalized_path,
            client=client,
            encode_kwargs=normalized_encode_kwargs,
            query_encode_kwargs=normalized_query_kwargs,
            show_progress=show_progress,
        )
        _ADAPTER_CACHE[key] = adapter
        return adapter


def get_ragas_embeddings(
    model_path: str,
    *,
    device: str | None = None,
    cache_folder: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    encode_kwargs: dict[str, Any] | None = None,
    query_encode_kwargs: dict[str, Any] | None = None,
    show_progress: bool = False,
    max_seq_length: int | None = None,
):
    normalized_path = str(model_path).strip()
    normalized_device = _normalize_device(device)
    normalized_model_kwargs = dict(model_kwargs or {})
    normalized_model_kwargs.setdefault("device", normalized_device)
    normalized_encode_kwargs = dict(encode_kwargs or {})
    normalized_query_kwargs = dict(query_encode_kwargs or {})
    normalized_max_seq_length = (
        int(max_seq_length) if max_seq_length is not None else None
    )
    key = _cache_key(
        "ragas_embeddings",
        model_path=normalized_path,
        device=normalized_device,
        cache_folder=cache_folder,
        model_kwargs=normalized_model_kwargs,
        encode_kwargs=normalized_encode_kwargs,
        query_encode_kwargs=normalized_query_kwargs,
        show_progress=show_progress,
        max_seq_length=normalized_max_seq_length,
    )

    with _CACHE_LOCK:
        cached = _ADAPTER_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except Exception as exc:
            raise ModelLoadException(
                f"RAGAS embeddings wrapper 加载失败: {exc}"
            ) from exc

        langchain_embeddings = get_langchain_embeddings(
            model_path=normalized_path,
            device=normalized_device,
            cache_folder=cache_folder,
            model_kwargs=normalized_model_kwargs,
            encode_kwargs=normalized_encode_kwargs,
            query_encode_kwargs=normalized_query_kwargs,
            show_progress=show_progress,
            max_seq_length=normalized_max_seq_length,
        )
        adapter = LangchainEmbeddingsWrapper(langchain_embeddings)
        _ADAPTER_CACHE[key] = adapter
        return adapter


def get_llamaindex_embedding(
    model_path: str,
    *,
    device: str | None = None,
    cache_folder: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    encode_kwargs: dict[str, Any] | None = None,
    query_encode_kwargs: dict[str, Any] | None = None,
    show_progress: bool = False,
    max_seq_length: int | None = None,
):
    normalized_path = str(model_path).strip()
    normalized_device = _normalize_device(device)
    normalized_model_kwargs = dict(model_kwargs or {})
    normalized_model_kwargs.setdefault("device", normalized_device)
    normalized_encode_kwargs = dict(encode_kwargs or {})
    normalized_query_kwargs = dict(query_encode_kwargs or {})
    normalized_max_seq_length = (
        int(max_seq_length) if max_seq_length is not None else None
    )
    key = _cache_key(
        "llamaindex_embeddings",
        model_path=normalized_path,
        device=normalized_device,
        cache_folder=cache_folder,
        model_kwargs=normalized_model_kwargs,
        encode_kwargs=normalized_encode_kwargs,
        query_encode_kwargs=normalized_query_kwargs,
        show_progress=show_progress,
        max_seq_length=normalized_max_seq_length,
    )

    with _CACHE_LOCK:
        cached = _ADAPTER_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from llama_index.embeddings.langchain import LangchainEmbedding
        except Exception as exc:
            raise ModelLoadException(
                f"LlamaIndex LangChain adapter 加载失败: {exc}"
            ) from exc

        langchain_embeddings = get_langchain_embeddings(
            model_path=normalized_path,
            device=normalized_device,
            cache_folder=cache_folder,
            model_kwargs=normalized_model_kwargs,
            encode_kwargs=normalized_encode_kwargs,
            query_encode_kwargs=normalized_query_kwargs,
            show_progress=show_progress,
            max_seq_length=normalized_max_seq_length,
        )
        adapter = LangchainEmbedding(langchain_embeddings)
        _ADAPTER_CACHE[key] = adapter
        return adapter


def clear_embedding_runtime_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _ADAPTER_CACHE.clear()
