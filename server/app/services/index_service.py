"""
向量索引服务层。
负责嵌入模型加载、向量索引构建与 collection 管理。
"""
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

    # ── 内部：从分块结果 JSON 中解析文本块 ─────────────────────────────────────

    @classmethod
    def _load_chunks_from_file(cls, docs_path: str) -> list[str]:
        """从分块结果 JSON 文件中加载文本块列表。"""
        try:
            raw = FileRepository.read_json(docs_path)
        except Exception as exc:
            raise IndexBuildException(f"读取分块结果文件失败 ({docs_path}): {exc}") from exc

        try:
            chunks = FileRepository.parse_chunks_from_json(raw)
        except ValueError as exc:
            raise IndexBuildException(str(exc)) from exc
        if not chunks:
            raise IndexBuildException(f"分块结果文件中未解析到任何文本块: {docs_path}")
        logger.info("从分块文件解析到 %d 个文本块", len(chunks))
        return chunks

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def build_index(self, request: IndexBuildRequest) -> IndexBuildResult:
        logger.info(
            "构建索引请求: collection=%s，docs_path=%s",
            request.collection_name,
            request.docs_path,
        )
        chunks = self._load_chunks_from_file(request.docs_path)
        settings = get_settings()
        model_path = settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = settings.DEFAULT_EMBEDDING_DIM

        embed_model = self._load_langchain_embed(model_path)
        info = self._repo.build_index(
            collection_name=request.collection_name,
            chunks=chunks,
            langchain_embed=embed_model,
            embed_dim=embed_dim,
            overwrite=True,
            batch_size=request.batch_size,
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
        chunks = self._load_chunks_from_file(request.docs_path)
        settings = get_settings()
        model_path = settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = settings.DEFAULT_EMBEDDING_DIM

        embed_model = self._load_langchain_embed(model_path)
        info = self._repo.add_index(
            collection_name=request.collection_name,
            chunks=chunks,
            langchain_embed=embed_model,
            embed_dim=embed_dim,
            batch_size=request.batch_size,
        )
        return IndexAddResult(
            collection_name=info["collection_name"],
            added_chunks=info["added_chunks"],
            time_cost=info["time_cost"],
            milvus_uri=info["milvus_uri"],
        )

    def list_collections(self) -> CollectionListResult:
        raw = self._repo.list_collections()
        collections = [CollectionInfo(**item) for item in raw]
        return CollectionListResult(collections=collections, total=len(collections))

    def delete_collection(self, collection_name: str) -> None:
        self._repo.delete_collection(collection_name)
