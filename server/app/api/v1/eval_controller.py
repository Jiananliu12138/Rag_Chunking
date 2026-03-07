from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import EvaluationException, to_http_exception
from app.schemas.common import BaseResponse
from app.schemas.eval_schema import (
    RAGASEvalFileRequest,
    RAGASEvalRequest,
    RAGASEvalResult,
    RetrievalEvalFileRequest,
    RetrievalEvalRequest,
    RetrievalEvalResult,
    TraditionalEvalFileRequest,
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
        "**输入方式**（二选一）：\n"
        "1. 直接传 `predictions` 和 `answers` 字段\n"
        "2. 传 `test` 字段（评估结果 JSON 格式，如 sample_results.json）\n\n"
        "每条样本的 `answers` 字段支持多个参考答案，各指标取最高分后再平均。\n\n"
        "**BERTScore 配置**：`enable_bert_score`、`bert_score_model`、`bert_score_device` 为可选字段，"
        "未提供时从环境变量/配置读取（`DEFAULT_ENABLE_BERT_SCORE`、`DEFAULT_BERT_SCORE_MODEL`、`DEFAULT_BERT_SCORE_DEVICE`）。\n\n"
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
    "/traditional-file",
    response_model=BaseResponse[TraditionalEvalResult],
    summary="传统指标评估（从文件读取）",
    description=(
        "从 JSON 文件读取评估结果并计算传统指标。\n\n"
        "**输入文件格式**：与 `sample_results.json` 一致，列表格式，每项包含：\n"
        "- `llm_ans`: 模型预测答案\n"
        "- `answers`: 参考答案列表\n"
        "- `_id`, `input`, `retrieval_list` 等（可选）\n\n"
        "文件读取与解析统一由 `file_repository` 处理。\n\n"
        "**BERTScore 配置**：`enable_bert_score`、`bert_score_model`、`bert_score_device` 为可选字段，"
        "未提供时从环境变量/配置读取。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_traditional_file(
    request: TraditionalEvalFileRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[TraditionalEvalResult]:
    try:
        result = service.evaluate_traditional_file(request)
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
        "- vLLM 服务已启动（地址从配置或请求参数读取）\n"
        "- 嵌入模型路径已配置（从配置或请求参数读取）\n\n"
        "**配置参数**：`vllm_api_base`、`vllm_api_key`、`vllm_model_name`、`embedding_model_path`、"
        "`device`、`enable_cache`、`cache_dir` 均为可选字段，未提供时从环境变量/配置读取。\n\n"
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


@router.post(
    "/ragas-file",
    response_model=BaseResponse[RAGASEvalResult],
    summary="RAGAS 端到端评估（从文件读取）",
    description=(
        "从 JSON 文件读取 RAGAS 数据集并执行多维度质量评估。\n\n"
        "**输入文件格式**（支持两种）：\n"
        "1. **标准 RAGAS 格式**：字典，包含 `question`、`answer`、`contexts`、`ground_truth` 四个列表字段\n"
        "2. **sample_results.json 格式**：列表，每项包含 `input`（问题）、`llm_ans`（答案）、"
        "`answers`（参考答案列表）、`retrieval_list`（上下文列表）\n\n"
        "文件读取与解析统一由 `file_repository` 处理。\n\n"
        "**配置参数**：所有 vLLM/嵌入模型/设备配置均为可选，未提供时从环境变量/配置读取。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_ragas_file(
    request: RAGASEvalFileRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[RAGASEvalResult]:
    try:
        result = service.evaluate_ragas_file(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/retrieval",
    response_model=BaseResponse[RetrievalEvalResult],
    summary="检索质量评估（Retrieval）",
    description=(
        "评估检索结果与标注参考之间的一致性，输出 MAP / MRR / nDCG / R-Precision 以及 P@k、Recall@k、nDCG@k。\n\n"
        "输入支持对象或对象列表，每项至少包含：`rag_retrieval` 与 `gold_reference`。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_retrieval(
    request: RetrievalEvalRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[RetrievalEvalResult]:
    try:
        result = service.evaluate_retrieval(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/retrieval-file",
    response_model=BaseResponse[RetrievalEvalResult],
    summary="检索质量评估（从文件读取）",
    description=(
        "从 JSON 文件读取检索结果并评估。\n\n"
        "文件可为单个对象或对象列表，字段格式与 `/eval/retrieval` 一致。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def evaluate_retrieval_file(
    request: RetrievalEvalFileRequest,
    service: Annotated[EvalService, Depends(_get_eval_service)],
) -> BaseResponse[RetrievalEvalResult]:
    try:
        result = service.evaluate_retrieval_file(request)
        return BaseResponse.ok(result)
    except EvaluationException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
