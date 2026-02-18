"""
端到端评估服务层。
封装传统指标（F1/ROUGE/BLEU/BERTScore）和 RAGAS 评估两条路径。
"""
import numpy as np

from app.core.exceptions import EvaluationException
from app.core.logging_config import logger
from app.core.path_setup import ensure_paths
from app.schemas.eval_schema import (
    RAGASDataset,
    RAGASEvalRequest,
    RAGASEvalResult,
    RAGASMetricSummary,
    RAGASSummary,
    TraditionalEvalRequest,
    TraditionalEvalResult,
)


class EvalService:

    # ── 传统指标 ──────────────────────────────────────────────────────────────

    def evaluate_traditional(self, request: TraditionalEvalRequest) -> TraditionalEvalResult:
        ensure_paths()
        if len(request.predictions) != len(request.answers):
            raise EvaluationException("predictions 与 answers 长度不一致")

        try:
            from metrics_lite import qa_f1_score  # noqa: PLC0415
        except ImportError as exc:
            raise EvaluationException(f"metrics_lite 模块导入失败: {exc}") from exc

        try:
            from rouge import Rouge
            from nltk.translate.bleu_score import SmoothingFunction

            rouge_inst = Rouge()
            smooth = SmoothingFunction().method1

            pairs = list(zip(request.predictions, request.answers))
            f1_scores = [max(qa_f1_score(p, gt) for gt in gts) for p, gts in pairs]
            rouge_l_scores = [self._rouge_l_score(p, gts, rouge_inst) for p, gts in pairs]
            bleu_scores = [self._bleu_scores(p, gts, smooth) for p, gts in pairs]

            b1 = [s[0] for s in bleu_scores]
            b2 = [s[1] for s in bleu_scores]
            b3 = [s[2] for s in bleu_scores]
            b4 = [s[3] for s in bleu_scores]

            bert_f1 = None
            if request.enable_bert_score:
                bert_f1 = self._calc_bert_score(
                    request.predictions,
                    request.answers,
                    request.bert_score_model,
                    request.bert_score_device,
                )

            return TraditionalEvalResult(
                f1=float(np.mean(f1_scores)),
                rouge_l=float(np.mean(rouge_l_scores)),
                bleu_1=float(np.mean(b1)),
                bleu_2=float(np.mean(b2)),
                bleu_3=float(np.mean(b3)),
                bleu_4=float(np.mean(b4)),
                bert_score_f1=bert_f1,
                sample_count=len(request.predictions),
            )
        except EvaluationException:
            raise
        except Exception as exc:
            logger.exception("传统指标评估失败: %s", exc)
            raise EvaluationException(f"传统指标评估失败: {exc}") from exc

    @staticmethod
    def _rouge_l_score(pred: str, ground_truths: list[str], rouge_inst) -> float:
        r_l = 0.0
        for gt in ground_truths:
            try:
                if pred.strip() and gt.strip():
                    r_l = max(r_l, rouge_inst.get_scores(pred, gt)[0]["rouge-l"]["f"])
            except Exception:
                pass
        return r_l

    @staticmethod
    def _bleu_scores(pred: str, ground_truths: list[str], smooth) -> tuple[float, float, float, float]:
        from nltk.translate.bleu_score import sentence_bleu  # noqa: PLC0415

        try:
            pred_tokens = pred.split()
            refs_tokens = [gt.split() for gt in ground_truths]
            b1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
            b2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            b3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth)
            b4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            return b1, b2, b3, b4
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def _calc_bert_score(
        predictions: list[str],
        answers: list[list[str]],
        model_type: str,
        device: str,
    ) -> float:
        try:
            from bert_score import score as bert_score_fn
            import torch

            refs = [" ".join(gts) for gts in answers]
            _, _, F1 = bert_score_fn(
                predictions, refs,
                model_type=model_type,
                verbose=False,
                device=device,
                batch_size=16,
            )
            result = F1.mean().item()
            del F1
            torch.cuda.empty_cache()
            return result
        except Exception as exc:
            logger.warning("BERTScore 计算失败: %s", exc)
            return 0.0

    # ── RAGAS 评估 ────────────────────────────────────────────────────────────

    def evaluate_ragas(self, request: RAGASEvalRequest) -> RAGASEvalResult:
        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()

            from eval_ragas import RAGASEvaluator  # noqa: PLC0415

            evaluator = RAGASEvaluator(
                vllm_api_base=request.vllm_api_base,
                vllm_api_key=request.vllm_api_key,
                vllm_model_name=request.vllm_model_name,
                embedding_model_path=request.embedding_model_path,
                device=request.device,
                enable_cache=request.enable_cache,
                cache_dir=request.cache_dir,
            )

            dataset = {
                "question": request.dataset.question,
                "answer": request.dataset.answer,
                "contexts": request.dataset.contexts,
                "ground_truth": request.dataset.ground_truth,
            }
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
                sample_count=len(request.dataset.question),
                samples=raw.get("samples", []),
            )
        except Exception as exc:
            logger.exception("RAGAS 评估失败: %s", exc)
            raise EvaluationException(f"RAGAS 评估失败: {exc}") from exc
