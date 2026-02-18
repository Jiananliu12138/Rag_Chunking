from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import ChunkingException, to_http_exception
from app.schemas.chunk_schema import (
    ChunkMethod,
    ChunkMethodInfo,
    ChunkRequest,
    ChunkResult,
    LlamaIndexChunkParams,
    LumberChunkParams,
    SemanticChunkParams,
    TokenChunkParams,
)
from app.schemas.common import BaseResponse
from app.services.chunk_service import ChunkService

router = APIRouter(prefix="/chunks", tags=["分块 / Chunking"])

_ERR_422 = {
    "description": "请求参数校验失败或分块过程出错",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "语义分块需要提供 semantic_params", "data": None}
        }
    },
}
_ERR_500 = {
    "description": "服务器内部错误",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "tiktoken 初始化失败", "data": None}
        }
    },
}


def _get_chunk_service() -> ChunkService:
    return ChunkService()


@router.post(
    "/chunk",
    response_model=BaseResponse[ChunkResult],
    summary="对文本执行分块",
    description=(
        "对输入文本执行指定的分块策略，返回分块结果。\n\n"
        "支持四种方法：\n"
        "- **token**：基于 tiktoken 的固定 Token 大小分块（默认），"
        "支持 `split_by_character` 优先按字符切割、`chunk_overlap_token_size` 滑动窗口重叠\n"
        "- **semantic**：基于 HuggingFace 嵌入模型计算句间语义相似度，在相似度骤降处切割\n"
        "- **llamaindex**：LlamaIndex `SimpleNodeParser` 固定窗口分块\n"
        "- **lumber**：调用 vLLM API，由 LLM 识别文档主题转换边界进行内容感知分块\n\n"
        "**注意**：semantic 方法须提供 `semantic_params.embed_model_path`；"
        "lumber 方法须提供 `lumber_params` 且对应 vLLM 服务已启动。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def chunk_text(
    request: ChunkRequest,
    service: Annotated[ChunkService, Depends(_get_chunk_service)],
) -> BaseResponse[ChunkResult]:
    try:
        result = service.chunk(request)
        return BaseResponse.ok(result)
    except ChunkingException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/methods",
    response_model=BaseResponse[list[ChunkMethodInfo]],
    summary="获取所有可用分块方法",
    description=(
        "返回当前支持的所有分块方法列表及其完整参数 JSON Schema。\n\n"
        "可用于前端动态渲染表单，或在调用 `/chunk` 前确认各方法所需参数。"
    ),
)
def list_methods() -> BaseResponse[list[ChunkMethodInfo]]:
    methods = [
        ChunkMethodInfo(
            name=ChunkMethod.TOKEN,
            description="基于 tiktoken 的固定 Token 大小分块，支持字符优先切割和滑动窗口重叠。",
            params_schema=TokenChunkParams.model_json_schema(),
        ),
        ChunkMethodInfo(
            name=ChunkMethod.SEMANTIC,
            description="基于 HuggingFace 嵌入模型计算语义相似度，在相似度骤降处切割。",
            params_schema=SemanticChunkParams.model_json_schema(),
        ),
        ChunkMethodInfo(
            name=ChunkMethod.LLAMAINDEX,
            description="LlamaIndex SimpleNodeParser，按固定 chunk_size 和 chunk_overlap 切割。",
            params_schema=LlamaIndexChunkParams.model_json_schema(),
        ),
        ChunkMethodInfo(
            name=ChunkMethod.LUMBER,
            description="调用 LLM（通过 vLLM API）识别主题转换边界，实现内容感知分块。",
            params_schema=LumberChunkParams.model_json_schema(),
        ),
    ]
    return BaseResponse.ok(methods)
