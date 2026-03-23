"""
Vector index service.
Handles embedding model loading, index build/add, and collection management.
"""
from typing import Any, Tuple

from app.config import get_settings
from app.core.exceptions import IndexBuildException, ModelLoadException
from app.core.logging_config import logger
from app.core.model_factory import get_langchain_embeddings
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


class IndexService:
    def __init__(self, milvus_repo: MilvusRepository):
        self._repo = milvus_repo

    @staticmethod
    def _load_langchain_embed(
        embed_model_path: str,
        *,
        embedding_device: str | None = None,
        embedding_max_tokens: int | None = None,
    ):
        """Load and cache the embedding model used by retrieval and indexing."""
        try:
            embed_model = get_langchain_embeddings(
                model_path=embed_model_path,
                device=embedding_device,
                encode_kwargs={"normalize_embeddings": True},
                max_seq_length=embedding_max_tokens,
            )
            resolved_max_tokens = IndexService._configure_embedding_max_tokens(
                embed_model,
                embedding_max_tokens=embedding_max_tokens,
            )
            return embed_model, resolved_max_tokens
        except Exception as exc:
            raise ModelLoadException(
                f"嵌入模型加载失败 ({embed_model_path}): {exc}"
            ) from exc

    @staticmethod
    def _configure_embedding_max_tokens(
        embed_model,
        *,
        embedding_max_tokens: int | None = None,
    ) -> int | None:
        client = getattr(embed_model, "_client", None)
        if client is None:
            return None

        configured_max_tokens = (
            int(embedding_max_tokens) if embedding_max_tokens is not None else None
        )
        model_max_tokens = getattr(client, "max_seq_length", None)
        resolved_max_tokens = configured_max_tokens or model_max_tokens

        if configured_max_tokens and model_max_tokens != configured_max_tokens:
            try:
                client.max_seq_length = configured_max_tokens
                logger.info(
                    "嵌入模型 max_seq_length 已显式设置为 %d（原值=%s）",
                    configured_max_tokens,
                    model_max_tokens,
                )
            except Exception as exc:
                logger.warning(
                    "无法将嵌入模型 max_seq_length 设置为 %d: %s",
                    configured_max_tokens,
                    exc,
                )
            resolved_max_tokens = configured_max_tokens

        if resolved_max_tokens is not None:
            try:
                resolved_max_tokens = int(resolved_max_tokens)
            except (TypeError, ValueError):
                return None
            if resolved_max_tokens <= 0:
                return None

        return resolved_max_tokens

    @staticmethod
    def _count_embedding_tokens(tokenizer, text: str) -> int | None:
        try:
            normalized_text = text.replace("\n", " ")
            encoded = tokenizer(
                normalized_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return len(input_ids[0])
                return len(input_ids)
        except Exception:
            return None
        return None

    @classmethod
    def _warn_on_overlong_chunks(
        cls,
        *,
        chunks: list[str],
        metadatas: list[dict[str, Any]],
        embed_model,
        embedding_max_tokens: int | None,
        operation: str,
        collection_name: str,
    ) -> None:
        if not embedding_max_tokens:
            logger.info(
                "未配置嵌入模型 max token 长度，跳过索引前超长 chunk 检测: collection=%s",
                collection_name,
            )
            return

        client = getattr(embed_model, "_client", None)
        tokenizer = getattr(client, "tokenizer", None) if client is not None else None
        if tokenizer is None:
            logger.warning(
                "嵌入模型未暴露 tokenizer，无法执行索引前超长检测: collection=%s",
                collection_name,
            )
            return

        overlong_examples: list[str] = []
        overlong_count = 0
        max_seen_tokens = 0

        for idx, chunk in enumerate(chunks):
            token_count = cls._count_embedding_tokens(tokenizer, chunk)
            if token_count is None:
                continue
            max_seen_tokens = max(max_seen_tokens, token_count)
            if token_count <= embedding_max_tokens:
                continue

            overlong_count += 1
            if len(overlong_examples) < 5:
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                file_path = metadata.get("file_path") if isinstance(metadata, dict) else None
                doc_id = metadata.get("doc_id") if isinstance(metadata, dict) else None
                chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None
                overlong_examples.append(
                    "idx=%d tokens=%d file_path=%s doc_id=%s chunk_id=%s"
                    % (
                        idx,
                        token_count,
                        file_path or "-",
                        doc_id or "-",
                        chunk_id or "-",
                    )
                )

        if overlong_count == 0:
            logger.info(
                "索引前超长检测完成: collection=%s, operation=%s, total_chunks=%d, max_seen_tokens=%d, limit=%d, overlong=0",
                collection_name,
                operation,
                len(chunks),
                max_seen_tokens,
                embedding_max_tokens,
            )
            return

        logger.warning(
            "检测到超长 chunk: collection=%s, operation=%s, overlong=%d/%d, embedding_max_tokens=%d, max_seen_tokens=%d。"
            "这些 chunk 在 SentenceTransformers 编码时会被截断。示例: %s",
            collection_name,
            operation,
            overlong_count,
            len(chunks),
            embedding_max_tokens,
            max_seen_tokens,
            " | ".join(overlong_examples),
        )

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
        embedding_device = settings.DEFAULT_EMBEDDING_DEVICE
        embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM
        embedding_max_tokens = (
            request.embed_max_tokens
            if getattr(request, "embed_max_tokens", None) is not None
            else settings.DEFAULT_EMBEDDING_MAX_TOKENS
        )

        embed_model, resolved_max_tokens = self._load_langchain_embed(
            model_path,
            embedding_device=embedding_device,
            embedding_max_tokens=embedding_max_tokens,
        )
        self._warn_on_overlong_chunks(
            chunks=chunks,
            metadatas=metadatas,
            embed_model=embed_model,
            embedding_max_tokens=resolved_max_tokens,
            operation="build_index",
            collection_name=request.collection_name,
        )
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
        embedding_device = settings.DEFAULT_EMBEDDING_DEVICE
        embedding_max_tokens = (
            request.embed_max_tokens
            if getattr(request, "embed_max_tokens", None) is not None
            else settings.DEFAULT_EMBEDDING_MAX_TOKENS
        )

        embed_model, resolved_max_tokens = self._load_langchain_embed(
            model_path,
            embedding_device=embedding_device,
            embedding_max_tokens=embedding_max_tokens,
        )
        self._warn_on_overlong_chunks(
            chunks=chunks,
            metadatas=metadatas,
            embed_model=embed_model,
            embedding_max_tokens=resolved_max_tokens,
            operation="add_index",
            collection_name=request.collection_name,
        )
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
