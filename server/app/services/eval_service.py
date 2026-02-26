"""
端到端评估服务层。
封装传统指标（F1/ROUGE/BLEU/BERTScore）和 RAGAS 评估两条路径。
"""
import numpy as np

from app.config import get_settings
from app.core.exceptions import EvaluationException
from app.core.logging_config import logger
from app.core.path_setup import ensure_paths
from app.repositories.file_repository import FileRepository
from app.schemas.eval_schema import (
    RAGASEvalFileRequest,
    RAGASEvalRequest,
    RAGASEvalResult,
    RAGASMetricSummary,
    RAGASSummary,
    TraditionalEvalFileRequest,
    TraditionalEvalRequest,
    TraditionalEvalResult,
)


class EvalService:

    # ── 传统指标 ──────────────────────────────────────────────────────────────

    def evaluate_traditional(self, request: TraditionalEvalRequest) -> TraditionalEvalResult:
        ensure_paths()

        # 从 request 提取 predictions 和 answers
        if request.test is not None:
            # 方式1：从 test 字段提取（评估结果 JSON 格式）
            predictions, answers = FileRepository.parse_eval_results_from_json(request.test)
        elif request.predictions is not None and request.answers is not None:
            # 方式2：直接使用 predictions 和 answers
            predictions = request.predictions
            answers = request.answers
        else:
            raise EvaluationException("必须提供 test 字段或 predictions+answers 字段")

        if len(predictions) != len(answers):
            raise EvaluationException("predictions 与 answers 长度不一致")

        # 从配置读取 BERTScore 参数（request 中未提供时使用配置默认值）
        settings = get_settings()
        enable_bert_score = (
            request.enable_bert_score
            if request.enable_bert_score is not None
            else settings.DEFAULT_ENABLE_BERT_SCORE
        )
        bert_score_model = (
            request.bert_score_model if request.bert_score_model else settings.DEFAULT_BERT_SCORE_MODEL
        )
        bert_score_device = (
            request.bert_score_device if request.bert_score_device else settings.DEFAULT_BERT_SCORE_DEVICE
        )

        try:
            # 调用 eval_lite.py 中的带参数版本函数
            # ensure_paths() 已将 EVAL_LONGBENCH_DIR 加入 sys.path
            from eval_lite import calculate_traditional_metrics_with_params  # noqa: PLC0415

            scores = calculate_traditional_metrics_with_params(
                predictions=predictions,
                answers=answers,
                enable_bert_score=enable_bert_score,
                bert_score_model=bert_score_model,
                bert_score_device=bert_score_device,
                hf_home=None,  # 服务层不设置 HF_HOME，使用系统默认
            )

            return TraditionalEvalResult(
                f1=float(scores["f1"]),
                rouge_l=float(scores["rouge_l"]),
                bleu_1=float(scores["bleu_1"]),
                bleu_2=float(scores["bleu_2"]),
                bleu_3=float(scores["bleu_3"]),
                bleu_4=float(scores["bleu_4"]),
                bert_score_f1=float(scores["bert_score_f1"]) if scores.get("bert_score_f1") is not None else None,
                sample_count=len(predictions),
            )
        except ImportError as exc:
            raise EvaluationException(f"无法导入 eval_lite 模块: {exc}") from exc
        except Exception as exc:
            logger.exception("传统指标评估失败: %s", exc)
            raise EvaluationException(f"传统指标评估失败: {exc}") from exc

    def evaluate_traditional_file(self, request: TraditionalEvalFileRequest) -> TraditionalEvalResult:
        """从 JSON 文件读取评估结果并计算传统指标。"""
        ensure_paths()

        try:
            # 读取文件并提取 predictions 和 answers
            data = FileRepository.read_json(request.input_path)
            predictions, answers = FileRepository.parse_eval_results_from_json(data)

            if len(predictions) != len(answers):
                raise EvaluationException("从文件解析出的 predictions 与 answers 长度不一致")

            # 从配置读取 BERTScore 参数（request 中未提供时使用配置默认值）
            settings = get_settings()
            enable_bert_score = (
                request.enable_bert_score
                if request.enable_bert_score is not None
                else settings.DEFAULT_ENABLE_BERT_SCORE
            )
            bert_score_model = (
                request.bert_score_model if request.bert_score_model else settings.DEFAULT_BERT_SCORE_MODEL
            )
            bert_score_device = (
                request.bert_score_device if request.bert_score_device else settings.DEFAULT_BERT_SCORE_DEVICE
            )

            # 调用 eval_lite.py 中的带参数版本函数
            from eval_lite import calculate_traditional_metrics_with_params  # noqa: PLC0415

            scores = calculate_traditional_metrics_with_params(
                predictions=predictions,
                answers=answers,
                enable_bert_score=enable_bert_score,
                bert_score_model=bert_score_model,
                bert_score_device=bert_score_device,
                hf_home=None,
            )

            return TraditionalEvalResult(
                f1=float(scores["f1"]),
                rouge_l=float(scores["rouge_l"]),
                bleu_1=float(scores["bleu_1"]),
                bleu_2=float(scores["bleu_2"]),
                bleu_3=float(scores["bleu_3"]),
                bleu_4=float(scores["bleu_4"]),
                bert_score_f1=float(scores["bert_score_f1"]) if scores.get("bert_score_f1") is not None else None,
                sample_count=len(predictions),
            )
        except FileNotFoundError as exc:
            raise EvaluationException(f"输入文件不存在: {exc}") from exc
        except ImportError as exc:
            raise EvaluationException(f"无法导入 eval_lite 模块: {exc}") from exc
        except Exception as exc:
            logger.exception("传统指标文件评估失败: %s", exc)
            raise EvaluationException(f"传统指标文件评估失败: {exc}") from exc

    # ── RAGAS 评估 ────────────────────────────────────────────────────────────

    def evaluate_ragas(self, request: RAGASEvalRequest) -> RAGASEvalResult:
        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()

            # 从配置读取 RAGAS 参数（request 中未提供时使用配置默认值）
            settings = get_settings()
            vllm_api_base = request.vllm_api_base or settings.DEFAULT_RAGAS_VLLM_API_BASE
            vllm_api_key = request.vllm_api_key or settings.DEFAULT_RAGAS_VLLM_API_KEY
            vllm_model_name = request.vllm_model_name or settings.DEFAULT_RAGAS_VLLM_MODEL_NAME
            embedding_model_path = (
                request.embedding_model_path or settings.DEFAULT_RAGAS_EMBEDDING_MODEL_PATH
            )
            device = request.device or settings.DEFAULT_RAGAS_DEVICE
            enable_cache = (
                request.enable_cache if request.enable_cache is not None else settings.DEFAULT_RAGAS_ENABLE_CACHE
            )
            cache_dir = request.cache_dir or settings.DEFAULT_RAGAS_CACHE_DIR

            if not vllm_model_name:
                raise EvaluationException("未配置 RAGAS LLM 模型（DEFAULT_RAGAS_VLLM_MODEL_NAME）")
            if not embedding_model_path:
                raise EvaluationException("未配置 RAGAS 嵌入模型（DEFAULT_RAGAS_EMBEDDING_MODEL_PATH）")

            from eval_ragas import RAGASEvaluator  # noqa: PLC0415

            evaluator = RAGASEvaluator(
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                embedding_model_path=embedding_model_path,
                device=device,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
            )

            # 仅支持一种输入方式：request.test
            # test 为原始 JSON（标准 RAGAS 或 sample_results.json 格式），
            # 统一复用 FileRepository.parse_ragas_dataset_from_json 进行拆解。
            if request.test is None:
                raise EvaluationException("RAGAS 评估必须提供 test 字段（原始评估 JSON）")

            dataset = FileRepository.parse_ragas_dataset_from_json(request.test)
            raw = evaluator.evaluate(dataset)

            def _make_summary(name: str) -> RAGASMetricSummary:
                return RAGASMetricSummary(
                    mean=raw["summary"].get(f"{name}_mean", 0.0),
                    min=raw["summary"].get(f"{name}_min", 0.0),
                    max=raw["summary"].get(f"{name}_max", 0.0),
                )

            summary = RAGASSummary(
                ragas_score=_make_summary("ragas_score"),
                faithfulness=_make_summary("faithfulness"),
                answer_relevancy=_make_summary("answer_relevancy"),
                context_recall=_make_summary("context_recall"),
                context_precision=_make_summary("context_precision"),
                context_entity_recall=_make_summary("context_entity_recall"),
                noise_sensitivity_relevant=_make_summary("noise_sensitivity_relevant"),
                noise_sensitivity_irrelevant=_make_summary("noise_sensitivity_irrelevant"),
            )

            # 如果来自 test，question 是 dataset["question"]；否则是 request.dataset.question，长度等价
            sample_count = len(dataset["question"])

            return RAGASEvalResult(
                summary=summary,
                sample_count=sample_count,
                samples=raw.get("samples", []),
            )
        except Exception as exc:
            logger.exception("RAGAS 评估失败: %s", exc)
            raise EvaluationException(f"RAGAS 评估失败: {exc}") from exc

    def evaluate_ragas_file(self, request: RAGASEvalFileRequest) -> RAGASEvalResult:
        """从 JSON 文件读取 RAGAS 数据集并评估。"""
        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()

            # 读取文件并解析 RAGAS 数据集
            data = FileRepository.read_json(request.input_path)
            dataset = FileRepository.parse_ragas_dataset_from_json(data)

            # 从配置读取 RAGAS 参数（request 中未提供时使用配置默认值）
            settings = get_settings()
            vllm_api_base = request.vllm_api_base or settings.DEFAULT_RAGAS_VLLM_API_BASE
            vllm_api_key = request.vllm_api_key or settings.DEFAULT_RAGAS_VLLM_API_KEY
            vllm_model_name = request.vllm_model_name or settings.DEFAULT_RAGAS_VLLM_MODEL_NAME
            embedding_model_path = (
                request.embedding_model_path or settings.DEFAULT_RAGAS_EMBEDDING_MODEL_PATH
            )
            device = request.device or settings.DEFAULT_RAGAS_DEVICE
            enable_cache = (
                request.enable_cache if request.enable_cache is not None else settings.DEFAULT_RAGAS_ENABLE_CACHE
            )
            cache_dir = request.cache_dir or settings.DEFAULT_RAGAS_CACHE_DIR

            if not vllm_model_name:
                raise EvaluationException("未配置 RAGAS LLM 模型（DEFAULT_RAGAS_VLLM_MODEL_NAME）")
            if not embedding_model_path:
                raise EvaluationException("未配置 RAGAS 嵌入模型（DEFAULT_RAGAS_EMBEDDING_MODEL_PATH）")

            from eval_ragas import RAGASEvaluator  # noqa: PLC0415

            evaluator = RAGASEvaluator(
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                embedding_model_path=embedding_model_path,
                device=device,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
            )

            raw = evaluator.evaluate(dataset)

            def _make_summary(name: str) -> RAGASMetricSummary:
                return RAGASMetricSummary(
                    mean=raw["summary"].get(f"{name}_mean", 0.0),
                    min=raw["summary"].get(f"{name}_min", 0.0),
                    max=raw["summary"].get(f"{name}_max", 0.0),
                )

            summary = RAGASSummary(
                ragas_score=_make_summary("ragas_score"),
                faithfulness=_make_summary("faithfulness"),
                answer_relevancy=_make_summary("answer_relevancy"),
                context_recall=_make_summary("context_recall"),
                context_precision=_make_summary("context_precision"),
                context_entity_recall=_make_summary("context_entity_recall"),
                noise_sensitivity_relevant=_make_summary("noise_sensitivity_relevant"),
                noise_sensitivity_irrelevant=_make_summary("noise_sensitivity_irrelevant"),
            )

            return RAGASEvalResult(
                summary=summary,
                sample_count=len(dataset["question"]),
                samples=raw.get("samples", []),
            )
        except FileNotFoundError as exc:
            raise EvaluationException(f"输入文件不存在: {exc}") from exc
        except Exception as exc:
            logger.exception("RAGAS 文件评估失败: %s", exc)
            raise EvaluationException(f"RAGAS 文件评估失败: {exc}") from exc
