"""
端到端评估服务层。
封装传统指标（F1/ROUGE/BLEU/BERTScore）和 RAGAS 评估两条路径。
"""
import math
from datetime import datetime, timezone
from threading import Lock
from typing import Any

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

_RAGAS_EVALUATOR_CACHE: dict[tuple, object] = {}
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
        max_workers: int,
        request_timeout: int,
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
            int(max_workers),
            int(request_timeout),
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
                max_workers=max_workers,
                request_timeout=request_timeout,
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

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    @staticmethod
    def _mean(values: list[float], digits: int = 4) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), digits)

    @staticmethod
    def _preview_text(value: Any, limit: int = 200) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split())
        if not text:
            return None
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    @staticmethod
    def _resolve_query_id(row: dict[str, Any], index: int) -> str:
        raw = row.get("_id")
        if raw is None:
            return str(index)
        if isinstance(raw, str) and not raw.strip():
            return str(index)
        return str(raw)

    @classmethod
    def _extract_nested_value(cls, payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return cls._safe_number(current)

    @classmethod
    def _rank_items(
        cls,
        items: list[dict[str, Any]],
        *,
        value_path: tuple[str, ...],
        reverse: bool,
        limit: int = 3,
        index_key: str = "index",
        question_key: str = "question",
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for item in items:
            value = cls._extract_nested_value(item, value_path)
            if value is None:
                continue
            raw_index = item.get(index_key)
            order = raw_index if isinstance(raw_index, int) else 0
            ranked.append((value, order, item))

        ranked.sort(key=lambda entry: (-entry[0], entry[1]) if reverse else (entry[0], entry[1]))
        highlights: list[dict[str, Any]] = []
        for value, _, item in ranked[:limit]:
            highlight = {
                index_key: item.get(index_key),
                question_key: item.get(question_key),
                "value": round(value, 4),
            }
            if "ragas_score" in item:
                highlight["ragas_score"] = item.get("ragas_score")
            if "metrics" in item:
                highlight["metrics"] = item.get("metrics")
            if "num_overlap" in item:
                highlight["num_overlap"] = item.get("num_overlap")
            if "num_retrieved" in item:
                highlight["num_retrieved"] = item.get("num_retrieved")
            if "num_gold" in item:
                highlight["num_gold"] = item.get("num_gold")
            highlights.append(highlight)
        return highlights

    def _resolve_traditional_runtime_metadata(
        self,
        *,
        rows: list[dict[str, Any]],
        request: TraditionalEvalRequest | TraditionalEvalFileRequest,
    ) -> dict[str, Any]:
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

        enable_llm_judge = (
            request.enable_llm_judge
            if request.enable_llm_judge is not None
            else settings.DEFAULT_ENABLE_LLM_JUDGE
        )
        generation_api_base = self._pick_single_runtime_value(rows, "generation_api_base")
        generation_model_name = self._pick_single_runtime_value(rows, "generation_model_name")
        judge_api_base = request.vllm_api_base or generation_api_base or settings.DEFAULT_LLM_API_BASE
        judge_model_name = request.vllm_model_name or generation_model_name or settings.DEFAULT_LLM_MODEL

        return {
            "bert_score_enabled": bool(enable_bert_score),
            "bert_score_model": bert_score_model if enable_bert_score else None,
            "bert_score_device": bert_score_device if enable_bert_score else None,
            "llm_judge_enabled": bool(enable_llm_judge),
            "llm_judge_api_base": judge_api_base if enable_llm_judge else None,
            "llm_judge_model": judge_model_name if enable_llm_judge else None,
        }

    def _build_traditional_file_report(
        self,
        *,
        rows: list[dict[str, Any]],
        result: TraditionalEvalResult,
        request: TraditionalEvalFileRequest,
    ) -> dict[str, Any]:
        payload = result.model_dump(mode="json")
        summary = {key: value for key, value in payload.items() if key != "judge_details"}
        judge_details = payload.get("judge_details", [])
        judge_by_row = {
            int(detail["row_index"]): detail
            for detail in judge_details
            if isinstance(detail, dict) and self._safe_number(detail.get("row_index")) is not None
        }

        prediction_lengths = [len(row["prediction"]) for row in rows]
        reference_counts = [len(row["references"]) for row in rows]
        reference_lengths = [len(reference) for row in rows for reference in row["references"]]
        samples: list[dict[str, Any]] = []
        for row in rows:
            row_index = int(row["row_index"])
            samples.append(
                {
                    "row_index": row_index,
                    "question_id": row.get("question_id"),
                    "record_id": row.get("record_id"),
                    "question": row.get("question"),
                    "prediction": row["prediction"],
                    "references": list(row["references"]),
                    "prediction_length_chars": len(row["prediction"]),
                    "reference_count": len(row["references"]),
                    "reference_lengths_chars": [len(reference) for reference in row["references"]],
                    "llm_judge": judge_by_row.get(row_index),
                }
            )

        failed_judgements = [
            {
                "row_index": detail.get("row_index"),
                "question_id": detail.get("question_id"),
                "record_id": detail.get("record_id"),
                "question": detail.get("question"),
                "reason": detail.get("reason"),
            }
            for detail in judge_details
            if isinstance(detail, dict) and int(detail.get("score", 0)) == 0
        ]

        runtime = self._resolve_traditional_runtime_metadata(rows=rows, request=request)
        return {
            "report_meta": {
                "report_type": "traditional_eval_report",
                "generated_at": self._timestamp_utc(),
                "input_path": request.input_path,
                "output_path": request.output_path,
            },
            "runtime": {
                **runtime,
                "llm_judge_prompt_version": result.llm_judge_prompt_version,
            },
            "dataset_overview": {
                "sample_count": len(rows),
                "questions_with_text": sum(1 for row in rows if row.get("question")),
                "questions_with_question_id": sum(1 for row in rows if row.get("question_id") is not None),
                "questions_with_record_id": sum(1 for row in rows if row.get("record_id") is not None),
                "avg_prediction_length_chars": self._mean(prediction_lengths, digits=2),
                "avg_reference_count": self._mean(reference_counts, digits=2),
                "avg_reference_length_chars": self._mean(reference_lengths, digits=2),
            },
            "summary": summary,
            "metric_breakdown": {
                "lexical_overlap": {
                    "f1": result.f1,
                    "rouge_l": result.rouge_l,
                    "bleu_family": {
                        "bleu_1": result.bleu_1,
                        "bleu_2": result.bleu_2,
                        "bleu_3": result.bleu_3,
                        "bleu_4": result.bleu_4,
                    },
                },
                "semantic_similarity": {
                    "bert_score_f1": result.bert_score_f1,
                },
                "llm_judge": {
                    "success_rate": result.llm_judge_success_rate,
                    "correct_count": result.llm_judge_correct_count,
                    "incorrect_count": result.llm_judge_incorrect_count,
                    "model": result.llm_judge_model,
                    "prompt_version": result.llm_judge_prompt_version,
                },
            },
            "judge_overview": {
                "enabled": bool(runtime["llm_judge_enabled"]),
                "evaluated_count": len(judge_details),
                "correct_count": result.llm_judge_correct_count,
                "incorrect_count": result.llm_judge_incorrect_count,
                "failed_cases": failed_judgements,
            },
            "samples": samples,
            "judge_details": judge_details,
        }

    def _build_ragas_file_report(
        self,
        *,
        dataset: dict[str, list[Any]],
        result: RAGASEvalResult,
        request: RAGASEvalFileRequest,
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        context_counts = [len(contexts) for contexts in dataset["contexts"]]
        question_lengths = [len(str(question)) for question in dataset["question"]]
        answer_lengths = [len(str(answer)) for answer in dataset["answer"]]
        ground_truth_lengths = [len(str(ground_truth)) for ground_truth in dataset["ground_truth"]]
        samples = result.model_dump(mode="json").get("samples", [])

        return {
            "report_meta": {
                "report_type": "ragas_eval_report",
                "generated_at": self._timestamp_utc(),
                "input_path": request.input_path,
                "output_path": request.output_path,
            },
            "runtime": runtime,
            "dataset_overview": {
                "sample_count": len(dataset["question"]),
                "total_contexts": sum(context_counts),
                "avg_contexts_per_sample": self._mean(context_counts, digits=2),
                "max_contexts_per_sample": max(context_counts) if context_counts else 0,
                "min_contexts_per_sample": min(context_counts) if context_counts else 0,
                "avg_question_length_chars": self._mean(question_lengths, digits=2),
                "avg_answer_length_chars": self._mean(answer_lengths, digits=2),
                "avg_ground_truth_length_chars": self._mean(ground_truth_lengths, digits=2),
            },
            "summary": result.summary.model_dump(mode="json"),
            "sample_count": result.sample_count,
            "samples": samples,
            "sample_highlights": {
                "best_ragas_score": self._rank_items(samples, value_path=("ragas_score",), reverse=True),
                "lowest_ragas_score": self._rank_items(samples, value_path=("ragas_score",), reverse=False),
                "lowest_faithfulness": self._rank_items(
                    samples,
                    value_path=("metrics", "faithfulness"),
                    reverse=False,
                ),
                "highest_noise_sensitivity_relevant": self._rank_items(
                    samples,
                    value_path=("metrics", "noise_sensitivity_relevant"),
                    reverse=True,
                ),
            },
        }

    def _build_retrieval_file_report(
        self,
        *,
        rows: list[dict[str, Any]],
        raw: dict[str, Any],
        request: RetrievalEvalFileRequest,
        cuts: tuple[int, ...],
        skip_empty_gold: bool,
    ) -> dict[str, Any]:
        diagnostics = raw.get("diagnostics", [])
        per_query = raw.get("per_query", {})
        diagnostics_by_query = {
            str(item.get("query_id")): item
            for item in diagnostics
            if isinstance(item, dict) and item.get("query_id") is not None
        }

        query_details: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            query_id = self._resolve_query_id(row, idx)
            metrics = per_query.get(query_id)
            if not isinstance(metrics, dict):
                continue

            rag_retrieval = row.get("rag_retrieval") if isinstance(row.get("rag_retrieval"), list) else []
            gold_reference = row.get("gold_reference") if isinstance(row.get("gold_reference"), list) else []
            diagnostic = diagnostics_by_query.get(query_id, {})
            query_details.append(
                {
                    "query_id": query_id,
                    "question_id": row.get("question_id"),
                    "question": row.get("input") or row.get("question") or row.get("query"),
                    "num_gold": diagnostic.get("num_gold", len(gold_reference)),
                    "num_retrieved": diagnostic.get("num_retrieved", len(rag_retrieval)),
                    "num_overlap": diagnostic.get("num_overlap"),
                    "metrics": metrics,
                    "retrieved_preview": [
                        {
                            "doc_id": item.get("doc_id"),
                            "chunk_id": item.get("chunk_id"),
                            "score": self._safe_number(item.get("score")),
                            "text_preview": self._preview_text(item.get("text")),
                        }
                        for item in rag_retrieval[:5]
                        if isinstance(item, dict)
                    ],
                    "gold_preview": [
                        {
                            "doc_id": item.get("doc_id"),
                            "chunk_id": item.get("chunk_id"),
                            "text_preview": self._preview_text(item.get("text")),
                        }
                        for item in gold_reference[:5]
                        if isinstance(item, dict)
                    ],
                }
            )

        max_cut = max(cuts) if cuts else None
        report = {
            "report_meta": {
                "report_type": "retrieval_eval_report",
                "generated_at": self._timestamp_utc(),
                "input_path": request.input_path,
                "output_path": request.output_path,
            },
            "runtime": {
                "cuts": list(cuts),
                "skip_empty_gold": skip_empty_gold,
            },
            "meta": raw.get("meta", {}),
            "aggregated": raw.get("aggregated", {}),
            "per_query": per_query,
            "diagnostics": diagnostics,
            "query_details": query_details,
            "query_highlights": {
                "best_map": self._rank_items(query_details, value_path=("metrics", "map"), reverse=True, index_key="query_id"),
                "lowest_map": self._rank_items(query_details, value_path=("metrics", "map"), reverse=False, index_key="query_id"),
                "best_ndcg": self._rank_items(query_details, value_path=("metrics", "ndcg"), reverse=True, index_key="query_id"),
            },
        }
        if max_cut is not None:
            report["query_highlights"][f"lowest_recall_{max_cut}"] = self._rank_items(
                query_details,
                value_path=("metrics", f"recall_{max_cut}"),
                reverse=False,
                index_key="query_id",
            )
        return report

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
                timeout=settings.EVAL_HTTP_TIMEOUT,
                max_workers=settings.EVAL_PARALLEL_WORKERS,
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
                FileRepository.write_json(
                    request.output_path,
                    self._build_traditional_file_report(rows=rows, result=result, request=request),
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
                max_workers=settings.RAGAS_MAX_WORKERS,
                request_timeout=settings.EVAL_HTTP_TIMEOUT,
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
                max_workers=settings.RAGAS_MAX_WORKERS,
                request_timeout=settings.EVAL_HTTP_TIMEOUT,
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
                runtime = {
                    "vllm_api_base": vllm_api_base,
                    "vllm_model_name": vllm_model_name,
                    "embedding_model_path": embedding_model_path,
                    "embedding_device": embedding_device,
                    "embedding_max_tokens": embedding_max_tokens,
                    "device": device,
                    "enable_cache": enable_cache,
                    "cache_dir": cache_dir,
                }
                FileRepository.write_json(
                    request.output_path,
                    self._build_ragas_file_report(
                        dataset=dataset,
                        result=result,
                        request=request,
                        runtime=runtime,
                    ),
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
                FileRepository.write_json(
                    request.output_path,
                    self._build_retrieval_file_report(
                        rows=rows,
                        raw=raw,
                        request=request,
                        cuts=cuts,
                        skip_empty_gold=skip_empty_gold,
                    ),
                )

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
