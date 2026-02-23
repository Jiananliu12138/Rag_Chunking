"""
组件级评估服务层。
提供分块质量（边界清晰度 + 语义不相似度）和分块黏连度两类评估能力。
"""
from app.config import get_settings
from app.core.exceptions import EvaluationException, ModelLoadException
from app.core.logging_config import logger
from app.repositories.file_repository import FileRepository
from app.schemas.eval_schema import (
    ChunkPairResult,
    ChunkQualityFileRequest,
    ChunkQualityRequest,
    ChunkQualityResult,
    ChunkStickinessFileRequest,
    ChunkStickinessRequest,
    ChunkStickinessResult,
)
from typing import Optional

class ComponentEvalService:

    # ── Chunk 质量评估（BC + 语义不相似度） ───────────────────────────────────

    def _evaluate_chunk_quality_core(
        self,
        chunks: list[str],
        enable_semantic_similarity: Optional[bool],
        enable_boundary_clarity: Optional[bool],
    ) -> ChunkQualityResult:
        if len(chunks) < 2:
            raise EvaluationException(f"至少需要 2 个文本块才能评估，当前只有 {len(chunks)} 个")

        try:
            from app.core.path_setup import ensure_paths

            ensure_paths()
            from chunk_eval_refactored import ChunkEvaluator, EvaluatorConfig  # noqa: PLC0415

            settings = get_settings()
            if not settings.COMPONENT_PPL_MODEL_PATH:
                raise ModelLoadException("未配置组件评估困惑度模型路径（COMPONENT_PPL_MODEL_PATH）")
            # 解析最终的开关配置（请求未提供时走全局配置）
            use_semantic = (
                enable_semantic_similarity
                if enable_semantic_similarity is not None
                else settings.COMPONENT_ENABLE_SEMANTIC_SIMILARITY
            )
            use_boundary = (
                enable_boundary_clarity
                if enable_boundary_clarity is not None
                else settings.COMPONENT_ENABLE_BOUNDARY_CLARITY
            )

            if use_semantic and not settings.COMPONENT_SIM_MODEL_PATH:
                raise ModelLoadException("未配置组件评估语义相似度模型路径（COMPONENT_SIM_MODEL_PATH）")

            config = EvaluatorConfig(
                ppl_model_name=settings.COMPONENT_PPL_MODEL_PATH,
                sim_model_name=settings.COMPONENT_SIM_MODEL_PATH,
                enable_semantic_similarity=use_semantic,
                enable_boundary_clarity=use_boundary,
            )
            evaluator = ChunkEvaluator(config)
            evaluator.load_models()
            agg = evaluator.evaluate_chunks(chunks, show_progress=False)

            details = [
                ChunkPairResult(
                    semantic_dissimilarity=(
                        r.semantic_dissimilarity if use_semantic else None
                    ),
                    boundary_clarity=(r.boundary_clarity if use_boundary else None),
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

    def evaluate_chunk_quality(self, request: ChunkQualityRequest) -> ChunkQualityResult:
        try:
            chunks = FileRepository.parse_chunks_from_json(request.chunks)
        except ValueError as exc:
            raise EvaluationException(f"不支持的 chunks 格式: {exc}") from exc

        return self._evaluate_chunk_quality_core(
            chunks=chunks,
            enable_semantic_similarity=request.enable_semantic_similarity,
            enable_boundary_clarity=request.enable_boundary_clarity,
        )

    def evaluate_chunk_quality_file(self, request: ChunkQualityFileRequest) -> ChunkQualityResult:
        try:
            raw = FileRepository.read_json(request.input_path)
            chunks = FileRepository.parse_chunks_from_json(raw)
        except FileNotFoundError as exc:
            raise EvaluationException(f"分块结果文件不存在: {exc}") from exc
        except ValueError as exc:
            raise EvaluationException(f"不支持的分块结果 JSON 格式: {exc}") from exc

        return self._evaluate_chunk_quality_core(
            chunks=chunks,
            enable_semantic_similarity=request.enable_semantic_similarity,
            enable_boundary_clarity=request.enable_boundary_clarity,
        )

    # ── Chunk 黏连度评估（结构熵） ────────────────────────────────────────────

    def evaluate_chunk_stickiness(self, request: ChunkStickinessRequest) -> ChunkStickinessResult:
        try:
            # 解析并规范化分块格式为纯文本列表
            chunks = FileRepository.parse_chunks_from_json(request.chunks)
        except ValueError as exc:
            raise EvaluationException(f"不支持的 chunks 格式: {exc}") from exc

        if len(chunks) < 2:
            raise EvaluationException(f"至少需要 2 个文本块才能评估，当前只有 {len(chunks)} 个")

        return self._evaluate_chunk_stickiness_core(
            chunks=chunks,
            threshold=request.threshold,
            delta=request.delta,
        )

    def evaluate_chunk_stickiness_file(self, request: ChunkStickinessFileRequest) -> ChunkStickinessResult:
        try:
            raw = FileRepository.read_json(request.input_path)
            chunks = FileRepository.parse_chunks_from_json(raw)
        except FileNotFoundError as exc:
            raise EvaluationException(f"分块结果文件不存在: {exc}") from exc
        except ValueError as exc:
            raise EvaluationException(f"不支持的分块结果 JSON 格式: {exc}") from exc

        if len(chunks) < 2:
            raise EvaluationException(f"至少需要 2 个文本块才能评估，当前只有 {len(chunks)} 个")

        return self._evaluate_chunk_stickiness_core(
            chunks=chunks,
            threshold=request.threshold,
            delta=request.delta,
        )

    def _evaluate_chunk_stickiness_core(
        self,
        chunks: list[str],
        threshold: Optional[float],
        delta: Optional[float],
    ) -> ChunkStickinessResult:
        try:
            from app.core.path_setup import ensure_paths

            ensure_paths()
            from relation_eval_refactored import StickinessEvaluator, StickinessConfig  # noqa: PLC0415

            settings = get_settings()
            if not settings.STICKINESS_MODEL_PATH:
                raise ModelLoadException("未配置黏连度评估模型路径（STICKINESS_MODEL_PATH）")

            use_threshold = threshold if threshold is not None else settings.STICKINESS_THRESHOLD
            use_delta = delta if delta is not None else settings.STICKINESS_DELTA

            config = StickinessConfig(
                model_path=settings.STICKINESS_MODEL_PATH,
                threshold=use_threshold,
                delta=use_delta,
            )
            evaluator = StickinessEvaluator(config)
            evaluator.load_model()
            result = evaluator.evaluate_chunks(chunks)

            return ChunkStickinessResult(
                structural_entropy_complete=result.structural_entropy_complete,
                structural_entropy_incomplete=result.structural_entropy_incomplete,
                graph_complete=result.graph_complete,
                graph_incomplete=result.graph_incomplete,
            )
        except EvaluationException:
            raise
        except Exception as exc:
            logger.exception("Chunk 黏连度评估失败: %s", exc)
            raise EvaluationException(f"Chunk 黏连度评估失败: {exc}") from exc
