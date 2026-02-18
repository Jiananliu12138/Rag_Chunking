"""
向量索引服务层。
负责嵌入模型加载、向量索引构建与 collection 管理。
"""
from app.core.exceptions import IndexBuildException, ModelLoadException
from app.core.logging_config import logger
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.index_schema import (
    CollectionInfo,
    CollectionListResult,
    IndexBuildRequest,
    IndexBuildResult,
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

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def build_index(self, request: IndexBuildRequest) -> IndexBuildResult:
        logger.info(
            "构建索引请求: collection=%s，块数=%d",
            request.collection_name, len(request.chunks),
        )
        embed_model = self._load_langchain_embed(request.embed_model_path)
        info = self._repo.build_index(
            collection_name=request.collection_name,
            chunks=request.chunks,
            langchain_embed=embed_model,
            embed_dim=request.embed_dim,
            overwrite=request.overwrite,
            batch_size=request.batch_size,
        )
        return IndexBuildResult(**info)

    def list_collections(self) -> CollectionListResult:
        raw = self._repo.list_collections()
        collections = [CollectionInfo(**item) for item in raw]
        return CollectionListResult(collections=collections, total=len(collections))

    def delete_collection(self, collection_name: str) -> None:
        self._repo.delete_collection(collection_name)
