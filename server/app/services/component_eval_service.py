"""
组件级评估服务层。
提供分块质量（边界清晰度 + 语义不相似度）和分块黏连度两类评估能力。
"""
from app.core.exceptions import EvaluationException, ModelLoadException
from app.core.logging_config import logger
from app.schemas.eval_schema import (
    ChunkPairResult,
    ChunkQualityRequest,
    ChunkQualityResult,
    ChunkStickinessRequest,
    ChunkStickinessResult,
)


class ComponentEvalService:

    # ── Chunk 质量评估（BC + 语义不相似度） ───────────────────────────────────

    def evaluate_chunk_quality(self, request: ChunkQualityRequest) -> ChunkQualityResult:
        if len(request.chunks) < 2:
            raise EvaluationException("至少需要 2 个文本块才能评估")

        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()
            from chunk_eval_refactored import ChunkEvaluator, EvaluatorConfig  # noqa: PLC0415

            config = EvaluatorConfig(
                ppl_model_name=request.ppl_model_name,
                sim_model_name=request.sim_model_name,
                enable_semantic_similarity=request.enable_semantic_similarity,
                enable_boundary_clarity=request.enable_boundary_clarity,
                device_map=request.device_map,
            )
            evaluator = ChunkEvaluator(config)
            evaluator.load_models()
            agg = evaluator.evaluate_chunks(request.chunks, show_progress=False)

            details = [
                ChunkPairResult(
                    semantic_dissimilarity=r.semantic_dissimilarity if request.enable_semantic_similarity else None,
                    boundary_clarity=r.boundary_clarity if request.enable_boundary_clarity else None,
                )
                for r in agg.individual_results
            ]

            return ChunkQualityResult(
                avg_semantic_dissimilarity=agg.semantic_dissimilarity_avg,
                avg_boundary_clarity=agg.boundary_clarity_avg,
                num_pairs=agg.num_pairs,
                details=details,
            )
        except EvaluationException:
            raise
        except Exception as exc:
            logger.exception("Chunk 质量评估失败: %s", exc)
            raise EvaluationException(f"Chunk 质量评估失败: {exc}") from exc

    # ── Chunk 黏连度评估（结构熵） ────────────────────────────────────────────

    def evaluate_chunk_stickiness(self, request: ChunkStickinessRequest) -> ChunkStickinessResult:
        if len(request.chunks) < 2:
            raise EvaluationException("至少需要 2 个文本块才能评估")

        try:
            from app.core.path_setup import ensure_paths
            ensure_paths()
            from relation_eval_refactored import StickinessEvaluator, StickinessConfig  # noqa: PLC0415

            config = StickinessConfig(
                model_path=request.model_path,
                threshold=request.threshold,
                delta=request.delta,
                device_map=request.device_map,
            )
            evaluator = StickinessEvaluator(config)
            evaluator.load_model()
            result = evaluator.evaluate_chunks(request.chunks)

            return ChunkStickinessResult(
                structural_entropy_complete=result.structural_entropy_complete,
                structural_entropy_incomplete=result.structural_entropy_incomplete,
            )
        except EvaluationException:
            raise
        except Exception as exc:
            logger.exception("Chunk 黏连度评估失败: %s", exc)
            raise EvaluationException(f"Chunk 黏连度评估失败: {exc}") from exc
