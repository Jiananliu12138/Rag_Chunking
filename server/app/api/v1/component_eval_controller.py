from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import EvaluationException, to_http_exception
from app.schemas.common import BaseResponse
from app.schemas.eval_schema import (
    ChunkQualityRequest,
    ChunkQualityResult,
    ChunkStickinessRequest,
    ChunkStickinessResult,
)
from app.services.component_eval_service import ComponentEvalService

router = APIRouter(prefix="/component-eval", tags=["组件评估 / Component Evaluation"])

_ERR_422 = {
    "description": "请求参数校验失败（如 chunks 数量不足 2 个）",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "至少需要 2 个文本块才能评估", "data": None}
        }
    },
}
_ERR_500 = {
    "description": "模型加载失败或评估过程出现异常",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "Chunk 质量评估失败: CUDA out of memory", "data": None}
        }
    },
}


def _get_component_eval_service() -> ComponentEvalService:
    return ComponentEvalService()


@router.post(
    "/chunk-quality",
    response_model=BaseResponse[ChunkQualityResult],
    summary="分块质量评估（BC + 语义不相似度）",
    description=(
        "对文本块列表进行**组件级**质量评估，无需下游检索/生成结果即可运行。\n\n"
        "计算所有相邻块对之间的两类指标：\n\n"
        "| 指标 | 公式 | 含义 | 方向 |\n"
        "|------|------|------|------|\n"
        "| **Semantic Dissimilarity** | `1 - cosine(emb_i, emb_j)` | "
        "相邻块的语义不相似度，越高说明块间语义边界越分明 | ↑ |\n"
        "| **Boundary Clarity (BC)** | `ppl(chunk_j \\| chunk_i) / ppl(chunk_j)` | "
        "边界清晰度，越接近 1 说明前块对后块的预测帮助越小，即块间独立性越强 | → 1 |\n\n"
        "**模型要求**：\n"
        "- `ppl_model_name`：本地因果语言模型（如 InternLM3-8B、Qwen3-4B），用于困惑度计算\n"
        "- `sim_model_name`：sentence-transformers 兼容模型，用于语义相似度\n\n"
        "响应 `details` 字段包含每对相邻块的逐对评分。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_chunk_quality(
    request: ChunkQualityRequest,
    service: Annotated[ComponentEvalService, Depends(_get_component_eval_service)],
) -> BaseResponse[ChunkQualityResult]:
    try:
        result = service.evaluate_chunk_quality(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/chunk-stickiness",
    response_model=BaseResponse[ChunkStickinessResult],
    summary="分块黏连度评估（结构熵）",
    description=(
        "通过构建**块间依赖图**并计算**结构熵**来量化文本块的黏连程度。\n\n"
        "**算法步骤**：\n"
        "1. 对每对块 (i, j) 计算困惑度 `ppl(chunk_j | chunk_i)`，构建带权完全图\n"
        "2. 按 token 长度和位置距离对边权归一化\n"
        "3. 按 `threshold` 筛选强关联边\n"
        "4. 分别对**完全图**和**上三角图（不完全图）**计算结构熵\n\n"
        "**结果解读**：结构熵越低，块间独立性越强，黏连度越低，分块质量越好。\n\n"
        "**参数说明**：\n"
        "- `threshold`：边权阈值（0-1），高于此值的边表示两块强相关，默认 0.8\n"
        "- `delta`：位置距离惩罚系数，> 0 时对非相邻块降权，默认 0（不惩罚）\n\n"
        "响应返回完全图和不完全图各自的结构熵值。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_chunk_stickiness(
    request: ChunkStickinessRequest,
    service: Annotated[ComponentEvalService, Depends(_get_component_eval_service)],
) -> BaseResponse[ChunkStickinessResult]:
    try:
        result = service.evaluate_chunk_stickiness(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
