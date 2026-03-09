"""
Vector index service.
Handles embedding model loading, index build/add, and collection management.
"""
from threading import Lock
from typing import Any, Tuple

from app.config import get_settings
from app.core.exceptions import IndexBuildException, ModelLoadException
from app.core.logging_config import logger
from app.repositories.file_repository import FileRepository
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.index_schema import (
    CollectionInfo,
    CollectionInspectItem,
    CollectionInspectResult,
    CollectionListResult,
    IndexAddRequest,
    IndexAddResult,
    IndexBuildRequest,
    IndexBuildResult,
    IndexDeleteByMetadataRequest,
)

_EMBED_MODEL_CACHE: dict[str, Any] = {}
_EMBED_MODEL_CACHE_LOCK = Lock()


class IndexService:
    def __init__(self, milvus_repo: MilvusRepository):
        self._repo = milvus_repo

    @staticmethod
    def _load_langchain_embed(embed_model_path: str):
        """Load and cache the embedding model used by retrieval and indexing."""
        cache_key = str(embed_model_path).strip()
        with _EMBED_MODEL_CACHE_LOCK:
            cached = _EMBED_MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached

            try:
                from app.core.path_setup import ensure_paths

                ensure_paths()
                from embeddings.base import HuggingfaceEmbeddings  # noqa: PLC0415

                model = HuggingfaceEmbeddings(
                    model_name=embed_model_path,
                    model_kwargs={"device": "cpu"},
                )
            except ImportError:
                model = None
            except Exception as exc:
                raise ModelLoadException(
                    f"嵌入模型加载失败 ({embed_model_path}): {exc}"
                ) from exc

            if model is None:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    model = HuggingFaceEmbeddings(
                        model_name=embed_model_path,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                except Exception as exc:
                    raise ModelLoadException(
                        f"嵌入模型加载失败 ({embed_model_path}): {exc}"
                    ) from exc

            _EMBED_MODEL_CACHE[cache_key] = model
            return model

    @classmethod
    def _load_chunks_and_metadata_from_file(
        cls, docs_path: str
    ) -> Tuple[list[str], list[dict[str, Any]]]:
        """
        Load chunks and metadata from a chunk-result JSON file.

        Preferred format:
        {
          "filepath": "...",
          "splits": [[text, doc_id, chunk_id], ...],
          "time_cost": ...
        }
        """
        try:
            raw = FileRepository.read_json(docs_path)
        except Exception as exc:
            raise IndexBuildException(
                f"读取分块结果文件失败 ({docs_path}): {exc}"
            ) from exc

        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []

        if isinstance(raw, dict) and "splits" in raw:
            filepath = raw.get("filepath")
            splits = raw.get("splits")
            if isinstance(splits, list):
                for item in splits:
                    if isinstance(item, list) and item:
                        text = item[0]
                        doc_id = item[1] if len(item) > 1 else None
                        chunk_id = item[2] if len(item) > 2 else None
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
                            md: dict[str, Any] = {}
                            if isinstance(filepath, str) and filepath:
                                md["file_path"] = filepath
                            if doc_id is not None:
                                md["doc_id"] = str(doc_id)
                            if chunk_id is not None:
                                md["chunk_id"] = str(chunk_id)
                            metadatas.append(md)

        if not chunks:
            try:
                parsed_chunks = FileRepository.parse_chunks_from_json(raw)
            except ValueError as exc:
                raise IndexBuildException(str(exc)) from exc
            chunks = parsed_chunks
            metadatas = [{} for _ in chunks]

        if not chunks:
            raise IndexBuildException(f"分块结果文件中未解析到任何文本块: {docs_path}")

        logger.info("从分块文件解析到 %d 个文本块", len(chunks))
        return chunks, metadatas

    @classmethod
    def _load_chunks_from_file(cls, docs_path: str) -> list[str]:
        """Compatibility wrapper that returns only chunk texts."""
        chunks, _ = cls._load_chunks_and_metadata_from_file(docs_path)
        return chunks

    def build_index(self, request: IndexBuildRequest) -> IndexBuildResult:
        logger.info(
            "构建索引请求: collection=%s, docs_path=%s",
            request.collection_name,
            request.docs_path,
        )
        if getattr(request, "docs_paths", None):
            all_chunks: list[str] = []
            all_metadatas: list[dict[str, Any]] = []
            for path in request.docs_paths or []:
                chunks, metadatas = self._load_chunks_and_metadata_from_file(path)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
            chunks, metadatas = all_chunks, all_metadatas
            logger.info(
                "本次构建合并 %d 个分块文件，总文本块数 %d",
                len(request.docs_paths or []),
                len(chunks),
            )
        else:
            chunks, metadatas = self._load_chunks_and_metadata_from_file(
                request.docs_path
            )

        settings = get_settings()
        model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM

        embed_model = self._load_langchain_embed(model_path)
        info = self._repo.build_index(
            collection_name=request.collection_name,
            chunks=chunks,
            metadatas=metadatas,
            langchain_embed=embed_model,
            embed_dim=embed_dim,
            overwrite=True,
            batch_size=request.batch_size,
            enable_sparse=request.enable_sparse,
        )
        return IndexBuildResult(**info)

    def add_index(self, request: IndexAddRequest) -> IndexAddResult:
        logger.info(
            "追加索引请求: collection=%s, docs_path=%s",
            request.collection_name,
            request.docs_path,
        )
        if getattr(request, "docs_paths", None):
            all_chunks: list[str] = []
            all_metadatas: list[dict[str, Any]] = []
            for path in request.docs_paths or []:
                chunks, metadatas = self._load_chunks_and_metadata_from_file(path)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
            chunks, metadatas = all_chunks, all_metadatas
            logger.info(
                "本次追加合并 %d 个分块文件，总文本块数 %d",
                len(request.docs_paths or []),
                len(chunks),
            )
        else:
            chunks, metadatas = self._load_chunks_and_metadata_from_file(
                request.docs_path
            )

        settings = get_settings()
        model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL

        embed_model = self._load_langchain_embed(model_path)
        info = self._repo.add_index(
            collection_name=request.collection_name,
            chunks=chunks,
            metadatas=metadatas,
            langchain_embed=embed_model,
            batch_size=request.batch_size,
        )
        return IndexAddResult(
            collection_name=info["collection_name"],
            added_chunks=info["added_chunks"],
            time_cost=info["time_cost"],
            milvus_uri=info["milvus_uri"],
            filepaths=info.get("filepaths", []),
            doc_ids=info.get("doc_ids", []),
            doc_chunk_map=info.get("doc_chunk_map", {}),
        )

    def list_collections(self) -> CollectionListResult:
        raw = self._repo.list_collections()
        collections = [CollectionInfo(**item) for item in raw]
        return CollectionListResult(collections=collections, total=len(collections))

    def delete_collection(self, collection_name: str) -> None:
        self._repo.delete_collection(collection_name)

    def inspect_collections(self) -> CollectionInspectResult:
        raw = self._repo.inspect_all_collections()
        items = [CollectionInspectItem(**item) for item in raw]
        return CollectionInspectResult(collections=items, total=len(items))

    def delete_by_metadata(
        self,
        collection_name: str,
        request: IndexDeleteByMetadataRequest,
    ) -> None:
        self._repo.delete_by_metadata(
            collection_name=collection_name,
            filepath=request.filepath,
            doc_ids=request.doc_ids,
        )
