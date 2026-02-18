from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import ChunkingException, to_http_exception
from app.schemas.chunk_schema import (
    ChunkMethod,
    ChunkFileRequest,
    ChunkFileResult,
    ChunkMethodInfo,
    ChunkTextRequest,
    ChunkTextResult,
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
    "/chunk-file",
    response_model=BaseResponse[ChunkFileResult],
    summary="对文件执行分块",
    description=(
        "对本地 jsonl 文件执行指定方法的分块，结果写入输出目录。\n\n"
        "支持四种方法：\n"
        "- **token**：基于 tiktoken 的固定 Token 大小分块（默认）\n"
        "- **semantic**：基于 HuggingFace 嵌入模型语义相似度切割\n"
        "- **llamaindex**：LlamaIndex SimpleNodeParser 固定窗口分块\n"
        "- **lumber**：调用 vLLM API 识别主题转换边界分块\n\n"
        "返回格式：success, output_file, message。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def chunk_file_endpoint(
    request: ChunkFileRequest,
    service: Annotated[ChunkService, Depends(_get_chunk_service)],
) -> BaseResponse[ChunkFileResult]:
    try:
        result = service.chunk_file(request)
        return BaseResponse.ok(result)
    except ChunkingException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/chunk-text",
    response_model=BaseResponse[ChunkTextResult],
    summary="对文本执行分块",
    description=(
        "对传入文本执行指定方法的分块。\n\n"
        "支持四种方法：\n"
        "- **token**：基于 tiktoken 的固定 Token 大小分块（默认）\n"
        "- **semantic**：基于 HuggingFace 嵌入模型语义相似度切割\n"
        "- **llamaindex**：LlamaIndex SimpleNodeParser 固定窗口分块\n"
        "- **lumber**：调用 vLLM API 识别主题转换边界分块\n\n"
        "返回格式与 Chunking_Methods 各模块一致：success, splits（每项 [text]）, time_cost, message。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def chunk_text_endpoint(
    request: ChunkTextRequest,
    service: Annotated[ChunkService, Depends(_get_chunk_service)],
) -> BaseResponse[ChunkTextResult]:
    try:
        result = service.chunk_text(request)
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
        "可用于前端动态渲染表单，或在调用分块接口前确认各方法所需参数。"
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