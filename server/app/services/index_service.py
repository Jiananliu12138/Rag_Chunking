"""
向量索引服务层。
负责嵌入模型加载、向量索引构建与 collection 管理。
"""
from typing import Any, Tuple

from app.config import get_settings
from app.core.exceptions import IndexBuildException, ModelLoadException
from app.core.logging_config import logger
from app.repositories.file_repository import FileRepository
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.index_schema import (
    CollectionInfo,
    CollectionListResult,
    IndexBuildRequest,
    IndexBuildResult,
    IndexAddRequest,
    IndexAddResult,
)


class IndexService:

    def __init__(self, milvus_repo: MilvusRepository):
        self._repo = milvus_repo

    # ── 内部：加载嵌入模型 ────────────────────────────────────────────────────

    @staticmethod
    def _load_langchain_embed(embed_model_path: str):
        """加载 LangChain 格式的 HuggingFace 嵌入模型。"""
        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()
            from embeddings.base import HuggingfaceEmbeddings  # noqa: PLC0415
            return HuggingfaceEmbeddings(model_name=embed_model_path)
        except ImportError:
            pass
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=embed_model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise ModelLoadException(f"嵌入模型加载失败 ({embed_model_path}): {exc}") from exc

    # ── 内部：从分块结果 JSON 中解析文本块及元数据 ─────────────────────────────

    @classmethod
    def _load_chunks_and_metadata_from_file(cls, docs_path: str) -> Tuple[list[str], list[dict[str, Any]]]:
        """
        从分块结果 JSON 文件中加载文本块列表及元数据。

        目前特别支持 MoC/our_metrics/test_data/test.json 这种结构：
        {
          "filepath": "...",
          "splits": [[text, doc_id], ...],
          "time_cost": ...
        }

        其中：
        - metadata["filepath"]      来自顶层 filepath
        - metadata["source_doc_id"] 来自每个 split 的第二个元素（避免与 LlamaIndex 内置 doc_id 冲突）

        其他兼容格式则退化为只有文本、metadata 为空字典。
        """
        try:
            raw = FileRepository.read_json(docs_path)
        except Exception as exc:
            raise IndexBuildException(f"读取分块结果文件失败 ({docs_path}): {exc}") from exc

        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []

        # 特例：带 filepath + splits 的结构（来自 MoC 评估流水线）
        if isinstance(raw, dict) and "splits" in raw:
            filepath = raw.get("filepath")
            splits = raw.get("splits")
            if isinstance(splits, list):
                for item in splits:
                    if isinstance(item, list) and item:
                        text = item[0]
                        doc_id = item[1] if len(item) > 1 else None
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
                            md: dict[str, Any] = {}
                            if isinstance(filepath, str) and filepath:
                                md["filepath"] = filepath
                            if doc_id is not None:
                                md["source_doc_id"] = str(doc_id)  # 避免与 LlamaIndex doc_id 冲突
                            metadatas.append(md)

        # 其他通用格式：复用 FileRepository.parse_chunks_from_json，只返回文本
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
        """兼容旧调用，只返回文本块列表。"""
        chunks, _ = cls._load_chunks_and_metadata_from_file(docs_path)
        return chunks

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def build_index(self, request: IndexBuildRequest) -> IndexBuildResult:
        logger.info(
            "构建索引请求: collection=%s，docs_path=%s",
            request.collection_name,
            request.docs_path,
        )
        chunks, metadatas = self._load_chunks_and_metadata_from_file(request.docs_path)
        settings = get_settings()
        model_path = settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = settings.DEFAULT_EMBEDDING_DIM

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
        """
        向已有索引中追加数据。

        输入语义与 build_index 相同：使用 docs_path 指向分块结果 JSON，
        但底层调用 MilvusRepository.add_index（overwrite 固定为 False）。
        """
        logger.info(
            "追加索引请求: collection=%s，docs_path=%s",
            request.collection_name,
            request.docs_path,
        )
        chunks, metadatas = self._load_chunks_and_metadata_from_file(request.docs_path)
        settings = get_settings()
        model_path = settings.DEFAULT_EMBEDDING_MODEL

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
        )

    def list_collections(self) -> CollectionListResult:
        raw = self._repo.list_collections()
        collections = [CollectionInfo(**item) for item in raw]
        return CollectionListResult(collections=collections, total=len(collections))

    def delete_collection(self, collection_name: str) -> None:
        self._repo.delete_collection(collection_name)
