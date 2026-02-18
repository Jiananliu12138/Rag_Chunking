from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import EvaluationException, to_http_exception
from app.schemas.common import BaseResponse
from app.schemas.eval_schema import (
    RAGASEvalRequest,
    RAGASEvalResult,
    TraditionalEvalRequest,
    TraditionalEvalResult,
)
from app.services.eval_service import EvalService

router = APIRouter(prefix="/eval", tags=["端到端评估 / End-to-End Evaluation"])

_ERR_422 = {
    "description": "请求参数校验失败（如 predictions 与 answers 长度不一致）",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "predictions 与 answers 长度不一致", "data": None}
        }
    },
}
_ERR_500 = {
    "description": "评估过程中出现异常（模型加载失败、依赖库缺失等）",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "metrics_lite 模块导入失败", "data": None}
        }
    },
}


def _get_eval_service() -> EvalService:
    return EvalService()


@router.post(
    "/traditional",
    response_model=BaseResponse[TraditionalEvalResult],
    summary="传统指标评估（F1 / ROUGE / BLEU / BERTScore）",
    description=(
        "计算预测答案与参考答案之间的经典 NLP 评估指标：\n\n"
        "| 指标 | 说明 | 越高越好 |\n"
        "|------|------|---------|\n"
        "| **F1** | 词级别 Precision × Recall 调和均值（LongBench 标准） | ✅ |\n"
        "| **ROUGE-L** | 最长公共子序列 F1 | ✅ |\n"
        "| **BLEU-1/2/3/4** | N-gram 精确率，n=1/2/3/4 | ✅ |\n"
        "| **BERTScore** | 基于 roberta-large 的语义相似度 F1（可选，需 GPU） | ✅ |\n\n"
        "每条样本的 `answers` 字段支持多个参考答案，各指标取最高分后再平均。\n\n"
        "若 `enable_bert_score=false`，响应中 `bert_score_f1` 字段为 `null`，计算速度更快。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_traditional(
    request: TraditionalEvalRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[TraditionalEvalResult]:
    try:
        result = service.evaluate_traditional(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/ragas",
    response_model=BaseResponse[RAGASEvalResult],
    summary="RAGAS 端到端评估",
    description=(
        "使用 RAGAS 框架对 RAG 系统进行多维度质量评估。\n\n"
        "| 指标 | 说明 | 方向 |\n"
        "|------|------|------|\n"
        "| **Faithfulness** | 答案与上下文的事实一致性 | ↑ 越高越好 |\n"
        "| **Answer Relevancy** | 答案与问题的相关性 | ↑ 越高越好 |\n"
        "| **Context Recall** | 参考答案被上下文覆盖的比例 | ↑ 越高越好 |\n"
        "| **Context Precision** | 上下文中与答案相关的比例 | ↑ 越高越好 |\n"
        "| **Context Entity Recall** | 实体级别的上下文覆盖率 | ↑ 越高越好 |\n"
        "| **Noise Sensitivity (Relevant)** | 相关噪声对答案的干扰程度 | ↓ 越低越好 |\n"
        "| **Noise Sensitivity (Irrelevant)** | 无关噪声对答案的干扰程度 | ↓ 越低越好 |\n\n"
        "**前置条件**：\n"
        "- `vllm_api_base` 对应的 vLLM 服务已启动\n"
        "- `embedding_model_path` 指向本地嵌入模型目录\n\n"
        "响应中 `summary` 包含每个指标的均值/最小值/最大值；`samples` 包含逐样本明细。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_ragas(
    request: RAGASEvalRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[RAGASEvalResult]:
    try:
        result = service.evaluate_ragas(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
