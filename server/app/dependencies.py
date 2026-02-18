"""
依赖注入容器。
通过 FastAPI Depends 机制提供服务实例，方便后续扩展为单例或带连接池的版本。
"""
from functools import lru_cache

from app.config import get_settings
from app.repositories.milvus_repository import MilvusRepository
from app.services.chunk_service import ChunkService
from app.services.component_eval_service import ComponentEvalService
from app.services.eval_service import EvalService
from app.services.index_service import IndexService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_milvus_repository() -> MilvusRepository:
    settings = get_settings()
    return MilvusRepository(milvus_data_dir=settings.MILVUS_DATA_DIR)


def get_chunk_service() -> ChunkService:
    return ChunkService()


def get_index_service() -> IndexService:
    return IndexService(get_milvus_repository())


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_milvus_repository())


def get_eval_service() -> EvalService:
    return EvalService()


def get_component_eval_service() -> ComponentEvalService:
    return ComponentEvalService()
