"""
端到端评估服务层。
封装传统指标（F1/ROUGE/BLEU/BERTScore）和 RAGAS 评估两条路径。
"""
import numpy as np
from threading import Lock

from app.config import get_settings
from app.core.exceptions import EvaluationException
from app.core.logging_config import logger
from app.core.model_factory import get_ragas_embeddings
from app.core.path_setup import ensure_paths
from app.repositories.file_repository import FileRepository
from app.schemas.eval_schema import (
    RAGASEvalFileRequest,
    RAGASEvalRequest,
    RAGASEvalResult,
    RAGASMetricSummary,
    RAGASSummary,
    RetrievalEvalFileRequest,
    RetrievalEvalRequest,
    RetrievalEvalResult,
    TraditionalEvalFileRequest,
    TraditionalEvalRequest,
    TraditionalEvalResult,
)

_RAGAS_EVALUATOR_CACHE: dict[tuple[str, str, str, str, str, bool, str], object] = {}
_RAGAS_EVALUATOR_CACHE_LOCK = Lock()


class EvalService:

    @staticmethod
    def _get_ragas_evaluator(
        *,
        vllm_api_base: str,
        vllm_api_key: str,
        vllm_model_name: str,
        embedding_model_path: str,
        embedding_device: str | None,
        embedding_max_tokens: int | None,
        device: str,
        enable_cache: bool,
        cache_dir: str,
    ):
        cache_key = (
            str(vllm_api_base).strip(),
            str(vllm_api_key).strip(),
            str(vllm_model_name).strip(),
            str(embedding_model_path).strip(),
            str(embedding_device).strip(),
            str(embedding_max_tokens).strip(),
            str(device).strip(),
            bool(enable_cache),
            str(cache_dir).strip(),
        )
        with _RAGAS_EVALUATOR_CACHE_LOCK:
            cached = _RAGAS_EVALUATOR_CACHE.get(cache_key)
            if cached is not None:
                return cached

            from public_method.evaluation.longbench.eval_ragas import RAGASEvaluator

            eval_embeddings = get_ragas_embeddings(
                model_path=embedding_model_path,
                device=embedding_device,
                encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
                max_seq_length=embedding_max_tokens,
            )
            evaluator = RAGASEvaluator(
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                embedding_model_path=embedding_model_path,
                eval_embeddings=eval_embeddings,
                device=device,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
            )
            _RAGAS_EVALUATOR_CACHE[cache_key] = evaluator
            return evaluator

    @staticmethod
    def _normalize_retrieval_rows(rows: list[dict]) -> list[dict]:
        """
        对齐 retrieval_api.py / retrieval_service.py 生成格式：
        - rag_retrieval 里若没有 score，则回退使用 rerank_score 或 retrieval_score
        - 字段缺失时补默认空列表，保证评估器稳定
        """
        normalized: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rag_items = row.get("rag_retrieval") if isinstance(row.get("rag_retrieval"), list) else []
            norm_rag: list[dict] = []
            for item in rag_items:
                if not isinstance(item, dict):
                    continue
                score = item.get("score")
                if score is None:
                    score = item.get("rerank_score")
                if score is None:
                    score = item.get("retrieval_score")
                merged = dict(item)
                if score is not None:
                    merged["score"] = score
                norm_rag.append(merged)

            gold = row.get("gold_reference") if isinstance(row.get("gold_reference"), list) else []
            merged_row = dict(row)
            merged_row["rag_retrieval"] = norm_rag
            merged_row["gold_reference"] = gold
            normalized.append(merged_row)
        return normalized

    @staticmethod
    def _build_traditional_rows_from_lists(
        predictions: list[str],
        answers: list[list[str]],
    ) -> list[dict]:
        if len(predictions) != len(answers):
            raise EvaluationException("predictions 与 answers 长度不一致")

        rows: list[dict] = []
        for idx, (prediction, refs) in enumerate(zip(predictions, answers), start=1):
            normalized_refs = [str(ref) for ref in refs if ref is not None and str(ref).strip()]
            if not normalized_refs:
                raise EvaluationException(f"answers[{idx - 1}] 不能为空")
            rows.append(
                {
                    "row_index": idx,
                    "question_id": None,
                    "record_id": None,
                    "question": None,
                    "prediction": str(prediction),
                    "references": normalized_refs,
                }
            )
        return rows

    def _resolve_traditional_rows(
        self,
        request: TraditionalEvalRequest | TraditionalEvalFileRequest,
    ) -> list[dict]:
        if getattr(request, "test", None) is not None:
            rows = FileRepository.parse_traditional_eval_rows_from_json(request.test)
        elif getattr(request, "predictions", None) is not None and getattr(request, "answers", None) is not None:
            rows = self._build_traditional_rows_from_lists(request.predictions, request.answers)
        else:
            raise EvaluationException("必须提供 test 字段或 predictions+answers 字段")

        if not rows:
            raise EvaluationException("没有可评估的样本，请检查 llm_ans/prediction 与 answer/reference/answers 字段")
        return rows

    @staticmethod
    def _pick_single_runtime_value(
        rows: list[dict],
        field_name: str,
    ) -> str | None:
        values = {
            str(row.get(field_name)).strip()
            for row in rows
            if row.get(field_name) is not None and str(row.get(field_name)).strip()
        }
        if len(values) == 1:
            return next(iter(values))
        if len(values) > 1:
            logger.warning(
                "Multiple generation runtime values found for %s; falling back to service defaults unless overridden.",
                field_name,
            )
        return None

    def _run_llm_judge(
        self,
        *,
        rows: list[dict],
        request: TraditionalEvalRequest | TraditionalEvalFileRequest,
        settings,
    ) -> dict:
        enable_llm_judge = (
            request.enable_llm_judge
            if request.enable_llm_judge is not None
            else settings.DEFAULT_ENABLE_LLM_JUDGE
        )
        if not enable_llm_judge:
            return {}

        generation_api_base = self._pick_single_runtime_value(rows, "generation_api_base")
        generation_model_name = self._pick_single_runtime_value(rows, "generation_model_name")

        vllm_api_base = request.vllm_api_base or generation_api_base or settings.DEFAULT_LLM_API_BASE
        vllm_api_key = request.vllm_api_key or settings.DEFAULT_LLM_API_KEY
        vllm_model_name = request.vllm_model_name or generation_model_name or settings.DEFAULT_LLM_MODEL

        if not vllm_api_base:
            raise EvaluationException("未配置 judge 模型 API 地址（DEFAULT_LLM_API_BASE）")
        if not vllm_model_name:
            raise EvaluationException("未配置 judge 模型名称（DEFAULT_LLM_MODEL）")

        try:
            from public_method.evaluation.end_to_end import JudgeSample, evaluate_answer_equivalence

            judge_samples = [
                JudgeSample(
                    row_index=int(row["row_index"]),
                    question_id=row.get("question_id"),
                    record_id=row.get("record_id"),
                    question=row.get("question"),
                    prediction=row["prediction"],
                    references=list(row["references"]),
                )
                for row in rows
            ]
            return evaluate_answer_equivalence(
                samples=judge_samples,
                api_base=vllm_api_base,
                api_key=vllm_api_key,
                model_name=vllm_model_name,
            )
        except Exception as exc:
            logger.exception("LLM judge evaluation failed: %s", exc)
            raise EvaluationException(f"LLM judge 评估失败: {exc}") from exc

    def _evaluate_traditional_core(
        self,
        *,
        rows: list[dict],
        request: TraditionalEvalRequest | TraditionalEvalFileRequest,
    ) -> TraditionalEvalResult:
        predictions = [row["prediction"] for row in rows]
        answers = [row["references"] for row in rows]

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

        from public_method.evaluation.longbench.eval_lite import (
            calculate_traditional_metrics_with_params,
        )

        scores = calculate_traditional_metrics_with_params(
            predictions=predictions,
            answers=answers,
            enable_bert_score=enable_bert_score,
            bert_score_model=bert_score_model,
            bert_score_device=bert_score_device,
            hf_home=None,
        )
        judge_scores = self._run_llm_judge(rows=rows, request=request, settings=settings)

        return TraditionalEvalResult(
            f1=float(scores["f1"]),
            rouge_l=float(scores["rouge_l"]),
            bleu_1=float(scores["bleu_1"]),
            bleu_2=float(scores["bleu_2"]),
            bleu_3=float(scores["bleu_3"]),
            bleu_4=float(scores["bleu_4"]),
            bert_score_f1=float(scores["bert_score_f1"]) if scores.get("bert_score_f1") is not None else None,
            sample_count=len(rows),
            llm_judge_success_rate=(
                float(judge_scores["llm_judge_success_rate"])
                if judge_scores.get("llm_judge_success_rate") is not None
                else None
            ),
            llm_judge_correct_count=judge_scores.get("llm_judge_correct_count"),
            llm_judge_incorrect_count=judge_scores.get("llm_judge_incorrect_count"),
            llm_judge_model=judge_scores.get("llm_judge_model"),
            llm_judge_prompt_version=judge_scores.get("llm_judge_prompt_version"),
            judge_details=judge_scores.get("judge_details", []),
        )

    # ── 传统指标 ──────────────────────────────────────────────────────────────

    def evaluate_traditional(self, request: TraditionalEvalRequest) -> TraditionalEvalResult:
        ensure_paths()

        try:
            rows = self._resolve_traditional_rows(request)
            return self._evaluate_traditional_core(rows=rows, request=request)
        except ImportError as exc:
            raise EvaluationException(f"无法导入 eval_lite 模块: {exc}") from exc
        except Exception as exc:
            logger.exception("传统指标评估失败: %s", exc)
            raise EvaluationException(f"传统指标评估失败: {exc}") from exc

    def evaluate_traditional_file(self, request: TraditionalEvalFileRequest) -> TraditionalEvalResult:
        """从 JSON 文件读取评估结果并计算传统指标。"""
        ensure_paths()

        try:
            data = FileRepository.read_json(request.input_path)
            rows = FileRepository.parse_traditional_eval_rows_from_json(data)
            result = self._evaluate_traditional_core(rows=rows, request=request)
            if request.output_path:
                payload = result.model_dump()
                FileRepository.write_json(
                    request.output_path,
                    {
                        "summary": {k: v for k, v in payload.items() if k != "judge_details"},
                        "judge_details": payload.get("judge_details", []),
                    },
                )
            return result
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
            vllm_api_base = request.vllm_api_base or settings.DEFAULT_LLM_API_BASE
            vllm_api_key = request.vllm_api_key or settings.DEFAULT_LLM_API_KEY
            vllm_model_name = request.vllm_model_name or settings.DEFAULT_LLM_MODEL
            embedding_model_path = (
                request.embedding_model_path or settings.DEFAULT_EMBEDDING_MODEL
            )
            embedding_device = settings.DEFAULT_EMBEDDING_DEVICE
            embedding_max_tokens = (
                request.embedding_max_tokens
                if request.embedding_max_tokens is not None
                else settings.DEFAULT_EMBEDDING_MAX_TOKENS
            )
            device = request.device or settings.DEFAULT_RAGAS_DEVICE
            enable_cache = (
                request.enable_cache if request.enable_cache is not None else settings.DEFAULT_RAGAS_ENABLE_CACHE
            )
            cache_dir = request.cache_dir or settings.DEFAULT_RAGAS_CACHE_DIR

            if not vllm_model_name:
                raise EvaluationException("未配置默认 LLM 模型（DEFAULT_LLM_MODEL）")
            if not embedding_model_path:
                raise EvaluationException("未配置默认嵌入模型（DEFAULT_EMBEDDING_MODEL）")

            evaluator = self._get_ragas_evaluator(
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                embedding_model_path=embedding_model_path,
                embedding_device=embedding_device,
                embedding_max_tokens=embedding_max_tokens,
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
            vllm_api_base = request.vllm_api_base or settings.DEFAULT_LLM_API_BASE
            vllm_api_key = request.vllm_api_key or settings.DEFAULT_LLM_API_KEY
            vllm_model_name = request.vllm_model_name or settings.DEFAULT_LLM_MODEL
            embedding_model_path = (
                request.embedding_model_path or settings.DEFAULT_EMBEDDING_MODEL
            )
            embedding_device = settings.DEFAULT_EMBEDDING_DEVICE
            embedding_max_tokens = (
                request.embedding_max_tokens
                if request.embedding_max_tokens is not None
                else settings.DEFAULT_EMBEDDING_MAX_TOKENS
            )
            device = request.device or settings.DEFAULT_RAGAS_DEVICE
            enable_cache = (
                request.enable_cache if request.enable_cache is not None else settings.DEFAULT_RAGAS_ENABLE_CACHE
            )
            cache_dir = request.cache_dir or settings.DEFAULT_RAGAS_CACHE_DIR

            if not vllm_model_name:
                raise EvaluationException("未配置默认 LLM 模型（DEFAULT_LLM_MODEL）")
            if not embedding_model_path:
                raise EvaluationException("未配置默认嵌入模型（DEFAULT_EMBEDDING_MODEL）")

            evaluator = self._get_ragas_evaluator(
                vllm_api_base=vllm_api_base,
                vllm_api_key=vllm_api_key,
                vllm_model_name=vllm_model_name,
                embedding_model_path=embedding_model_path,
                embedding_device=embedding_device,
                embedding_max_tokens=embedding_max_tokens,
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

            result = RAGASEvalResult(
                summary=summary,
                sample_count=len(dataset["question"]),
                samples=raw.get("samples", []),
            )
            if request.output_path:
                FileRepository.write_json(
                    request.output_path,
                    {
                        "summary": result.summary.model_dump(),
                        "sample_count": result.sample_count,
                    },
                )
            return result
        except FileNotFoundError as exc:
            raise EvaluationException(f"输入文件不存在: {exc}") from exc
        except Exception as exc:
            logger.exception("RAGAS 文件评估失败: %s", exc)
            raise EvaluationException(f"RAGAS 文件评估失败: {exc}") from exc

    # ── Retrieval 评估 ───────────────────────────────────────────────────────

    def evaluate_retrieval(self, request: RetrievalEvalRequest) -> RetrievalEvalResult:
        ensure_paths()
        cuts = tuple(request.cuts) if request.cuts else (1, 3, 5, 10)
        skip_empty_gold = (
            request.skip_empty_gold
            if request.skip_empty_gold is not None
            else True
        )

        try:
            from public_method.evaluation.longbench.eval_retrieval import RetrievalEvaluator

            evaluator = RetrievalEvaluator(cuts=cuts, skip_empty_gold=skip_empty_gold)
            rows = request.test if isinstance(request.test, list) else [request.test]
            rows = [x for x in rows if isinstance(x, dict)]
            rows = self._normalize_retrieval_rows(rows)
            if not rows:
                raise EvaluationException("Retrieval 评估必须提供对象或对象列表格式的 test")

            raw = evaluator.evaluate_rows(rows)
            return RetrievalEvalResult(
                meta=raw.get("meta", {}),
                aggregated=raw.get("aggregated", {}),
                per_query=raw.get("per_query", {}),
                diagnostics=raw.get("diagnostics", []),
            )
        except Exception as exc:
            logger.exception("Retrieval 评估失败: %s", exc)
            raise EvaluationException(f"Retrieval 评估失败: {exc}") from exc

    def evaluate_retrieval_file(self, request: RetrievalEvalFileRequest) -> RetrievalEvalResult:
        ensure_paths()
        cuts = tuple(request.cuts) if request.cuts else (1, 3, 5, 10)
        skip_empty_gold = (
            request.skip_empty_gold
            if request.skip_empty_gold is not None
            else True
        )

        try:
            from public_method.evaluation.longbench.eval_retrieval import RetrievalEvaluator

            evaluator = RetrievalEvaluator(cuts=cuts, skip_empty_gold=skip_empty_gold)
            rows = evaluator._load_rows(request.input_path)
            rows = self._normalize_retrieval_rows(rows)
            raw = evaluator.evaluate_rows(rows)
            if request.output_path:
                with open(request.output_path, "w", encoding="utf-8") as f:
                    import json
                    json.dump(raw, f, ensure_ascii=False, indent=2)

            return RetrievalEvalResult(
                meta=raw.get("meta", {}),
                aggregated=raw.get("aggregated", {}),
                per_query=raw.get("per_query", {}),
                diagnostics=raw.get("diagnostics", []),
            )
        except FileNotFoundError as exc:
            raise EvaluationException(f"输入文件不存在: {exc}") from exc
        except Exception as exc:
            logger.exception("Retrieval 文件评估失败: %s", exc)
            raise EvaluationException(f"Retrieval 文件评估失败: {exc}") from exc
