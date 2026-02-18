from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import (
    CollectionNotFoundException,
    ModelLoadException,
    RetrievalException,
    to_http_exception,
)
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.common import BaseResponse
from app.schemas.retrieval_schema import (
    RAGRequest,
    RAGResult,
    SearchRequest,
    SearchResult,
)
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["检索与生成 / Retrieval & Generation"])

_ERR_404 = {
    "description": "指定的 collection 不存在，请先调用 /index/build 构建索引",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "Collection 'wiki_chunks' 不存在，请先构建索引", "data": None}
        }
    },
}
_ERR_500 = {
    "description": "检索或 LLM 调用失败",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "LLM 生成失败: Connection refused", "data": None}
        }
    },
}


def _get_retrieval_service() -> RetrievalService:
    return RetrievalService(MilvusRepository())


@router.post(
    "/search",
    response_model=BaseResponse[SearchResult],
    summary="向量相似度检索",
    description=(
        "在指定 collection 中执行向量相似度检索，返回与查询最相关的 top-k 文档片段。\n\n"
        "**前置条件**：对应 collection 已通过 `/index/build` 构建完毕，"
        "且 `embed_model_path` 与构建索引时使用的模型一致（维度需匹配）。\n\n"
        "返回结果中每条 `score` 为余弦相似度（0–1，越高越相关）。"
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def search(
    request: SearchRequest,
    service: Annotated[RetrievalService, Depends(_get_retrieval_service)],
) -> BaseResponse[SearchResult]:
    try:
        result = service.search(request)
        return BaseResponse.ok(result)
    except (CollectionNotFoundException, ModelLoadException, RetrievalException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/generate",
    response_model=BaseResponse[RAGResult],
    summary="RAG 检索增强生成",
    description=(
        "完整的 RAG 流程：**向量检索** → **拼接上下文** → **vLLM 生成答案**。\n\n"
        "**流程详情**：\n"
        "1. 使用 `embed_model_path` 将 `query` 嵌入为向量\n"
        "2. 在 Milvus 中检索 `top_k` 个最相关文档片段\n"
        "3. 将文档拼接为上下文，构造 Qwen 格式 Prompt\n"
        "4. 调用 `llm_api_base` 的 `/completions` 接口生成回答\n\n"
        "**前置条件**：\n"
        "- 目标 collection 已构建索引\n"
        "- `llm_api_base` 对应的 vLLM 服务已启动且模型已加载\n\n"
        "响应中 `contexts` 字段返回实际使用的检索文档，可用于后续 RAGAS 评估。"
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def rag_generate(
    request: RAGRequest,
    service: Annotated[RetrievalService, Depends(_get_retrieval_service)],
) -> BaseResponse[RAGResult]:
    try:
        result = service.rag_generate(request)
        return BaseResponse.ok(result)
    except (CollectionNotFoundException, ModelLoadException, RetrievalException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
