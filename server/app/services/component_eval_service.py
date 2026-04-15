import math

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

    @staticmethod
    def _limit_chunks_for_file_eval(chunks: list[str], max_eval_chunks: Optional[int]) -> list[str]:
        if max_eval_chunks is None or max_eval_chunks <= 0:
            return chunks
        return chunks[:max_eval_chunks]

    # ── Chunk 质量评估（BC + 语义不相似度） ───────────────────────────────────

    def _evaluate_chunk_quality_core(
        self,
        chunks: list[str],
        enable_semantic_similarity: Optional[bool],
        enable_boundary_clarity: Optional[bool],
        score_temperature: Optional[float],
        sim_model_path: Optional[str],
        vllm_api_base: Optional[str],
        vllm_model_name: Optional[str],
        max_workers: Optional[int] = None,
    ) -> ChunkQualityResult:
        if len(chunks) < 2:
            raise EvaluationException(f"至少需要 2 个文本块才能评估，当前只有 {len(chunks)} 个")

        try:
            from app.core.path_setup import ensure_paths

            ensure_paths()
            from public_method.evaluation.component.chunk_eval_refactored import (
                ChunkEvaluator,
                EvaluatorConfig,
            )

            settings = get_settings()
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
            use_score_temperature = (
                score_temperature
                if score_temperature is not None
                else settings.COMPONENT_SCORE_TEMPERATURE
            )

            sim_path = sim_model_path or settings.DEFAULT_EMBEDDING_MODEL
            if use_semantic and not sim_path:
                raise ModelLoadException("未配置默认嵌入模型（DEFAULT_EMBEDDING_MODEL），且请求未指定 sim_model_path")

            vllm_base = vllm_api_base or settings.DEFAULT_LLM_API_BASE
            vllm_model = vllm_model_name or settings.DEFAULT_LLM_MODEL
            if not vllm_base or not vllm_model:
                raise ModelLoadException("Chunk 评估现在仅支持 vLLM，请配置 DEFAULT_LLM_API_BASE / DEFAULT_LLM_MODEL 或在请求中提供 vllm_api_base / vllm_model_name")

            config = EvaluatorConfig(
                sim_model_name=sim_path,
                enable_semantic_similarity=use_semantic,
                enable_boundary_clarity=use_boundary,
                score_temperature=use_score_temperature,
                use_vllm=True,
                vllm_api_base=vllm_base,
                vllm_model_name=vllm_model,
                max_workers=max_workers or settings.EVAL_PARALLEL_WORKERS,
                request_timeout=settings.EVAL_HTTP_TIMEOUT,
            )
            evaluator = ChunkEvaluator(config)
            evaluator.load_models()

            # 显式提示下一步耗时来源：开启 BC/语义相似度时，下面的 encode 和
            # vLLM 并发请求都会跑几十秒到几分钟（尤其是端点冷启动），不打 log
            # 的话日志只有一行 "开始评估" 就静默等待。
            num_pairs = max(len(chunks) - 1, 0)
            if use_semantic:
                logger.info(
                    "准备对 %d 条 chunks 调用 embedding 模型 (%s) 批量编码...",
                    len(chunks),
                    sim_path,
                )
            if use_boundary:
                logger.info(
                    "准备对 %d 对相邻 chunks 并发调用 vLLM (%s) 计算困惑度 (max_workers=%d)...",
                    num_pairs,
                    vllm_model,
                    config.max_workers,
                )

            agg = evaluator.evaluate_chunks(chunks, show_progress=True)

            details = [
                ChunkPairResult(
                    semantic_dissimilarity=(
                        r.semantic_dissimilarity
                        if use_semantic and math.isfinite(r.semantic_dissimilarity)
                        else None
                    ),
                    boundary_clarity=(
                        r.boundary_clarity
                        if use_boundary and math.isfinite(r.boundary_clarity)
                        else None
                    ),
                )
                for r in agg.individual_results
            ]

            return ChunkQualityResult(
                avg_semantic_dissimilarity=agg.semantic_dissimilarity_avg,
                avg_boundary_clarity=agg.boundary_clarity_avg,
                semantic_dissimilarity_valid_count=agg.semantic_dissimilarity_valid_count,
                semantic_dissimilarity_error_count=agg.semantic_dissimilarity_error_count,
                boundary_clarity_valid_count=agg.boundary_clarity_valid_count,
                boundary_clarity_error_count=agg.boundary_clarity_error_count,
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
            score_temperature=request.score_temperature,
            sim_model_path=request.sim_model_path,
            vllm_api_base=request.vllm_api_base,
            vllm_model_name=request.vllm_model_name,
            max_workers=request.max_workers,
        )

    def evaluate_chunk_quality_file(self, request: ChunkQualityFileRequest) -> ChunkQualityResult:
        try:
            raw = FileRepository.read_json(request.input_path)
            chunks = FileRepository.parse_chunks_from_json(raw)
            chunks = self._limit_chunks_for_file_eval(chunks, request.max_eval_chunks)
        except FileNotFoundError as exc:
            raise EvaluationException(f"分块结果文件不存在: {exc}") from exc
        except ValueError as exc:
            raise EvaluationException(f"不支持的分块结果 JSON 格式: {exc}") from exc

        result = self._evaluate_chunk_quality_core(
            chunks=chunks,
            enable_semantic_similarity=request.enable_semantic_similarity,
            enable_boundary_clarity=request.enable_boundary_clarity,
            score_temperature=request.score_temperature,
            sim_model_path=request.sim_model_path,
            vllm_api_base=request.vllm_api_base,
            vllm_model_name=request.vllm_model_name,
            max_workers=request.max_workers,
        )
        if request.output_path:
            FileRepository.write_json(request.output_path, result.model_dump(mode="json"))
        return result

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
            score_temperature=request.score_temperature,
            vllm_api_base=request.vllm_api_base,
            vllm_model_name=request.vllm_model_name,
            max_workers=request.max_workers,
        )

    def evaluate_chunk_stickiness_file(self, request: ChunkStickinessFileRequest) -> ChunkStickinessResult:
        try:
            raw = FileRepository.read_json(request.input_path)
            chunks = FileRepository.parse_chunks_from_json(raw)
            chunks = self._limit_chunks_for_file_eval(chunks, request.max_eval_chunks)
        except FileNotFoundError as exc:
            raise EvaluationException(f"分块结果文件不存在: {exc}") from exc
        except ValueError as exc:
            raise EvaluationException(f"不支持的分块结果 JSON 格式: {exc}") from exc

        if len(chunks) < 2:
            raise EvaluationException(f"至少需要 2 个文本块才能评估，当前只有 {len(chunks)} 个")

        result = self._evaluate_chunk_stickiness_core(
            chunks=chunks,
            threshold=request.threshold,
            delta=request.delta,
            score_temperature=request.score_temperature,
            vllm_api_base=request.vllm_api_base,
            vllm_model_name=request.vllm_model_name,
            max_workers=request.max_workers,
        )
        if request.output_path:
            FileRepository.write_json(request.output_path, result.model_dump(mode="json"))
        return result

    def _evaluate_chunk_stickiness_core(
        self,
        chunks: list[str],
        threshold: Optional[float],
        delta: Optional[float],
        score_temperature: Optional[float],
        vllm_api_base: Optional[str],
        vllm_model_name: Optional[str],
        max_workers: Optional[int] = None,
    ) -> ChunkStickinessResult:
        try:
            from app.core.path_setup import ensure_paths

            ensure_paths()
            from public_method.evaluation.component.relation_eval_refactored import (
                StickinessConfig,
                StickinessEvaluator,
            )

            settings = get_settings()
            use_threshold = threshold if threshold is not None else settings.STICKINESS_THRESHOLD
            use_delta = delta if delta is not None else settings.STICKINESS_DELTA
            use_score_temperature = (
                score_temperature
                if score_temperature is not None
                else settings.STICKINESS_SCORE_TEMPERATURE
            )

            vllm_base = vllm_api_base or settings.DEFAULT_LLM_API_BASE
            vllm_model = vllm_model_name or settings.DEFAULT_LLM_MODEL
            if not vllm_base or not vllm_model:
                raise ModelLoadException("Chunk 黏连度评估现在仅支持 vLLM，请配置 DEFAULT_LLM_API_BASE / DEFAULT_LLM_MODEL 或在请求中提供 vllm_api_base / vllm_model_name")

            config = StickinessConfig(
                threshold=use_threshold,
                delta=use_delta,
                score_temperature=use_score_temperature,
                use_vllm=True,
                vllm_api_base=vllm_base,
                vllm_model_name=vllm_model,
                max_workers=max_workers or settings.EVAL_PARALLEL_WORKERS,
                request_timeout=settings.EVAL_HTTP_TIMEOUT,
            )
            evaluator = StickinessEvaluator(config)
            evaluator.load_model()
            result = evaluator.evaluate_chunks(chunks)

            return ChunkStickinessResult(
                structural_entropy_complete=result.structural_entropy_complete,
                structural_entropy_incomplete=result.structural_entropy_incomplete,
                normalized_structural_entropy_complete=result.normalized_structural_entropy_complete,
                normalized_structural_entropy_incomplete=result.normalized_structural_entropy_incomplete,
                graph_complete=result.graph_complete,
                graph_incomplete=result.graph_incomplete,
            )
        except EvaluationException:
            raise
        except Exception as exc:
            logger.exception("Chunk 黏连度评估失败: %s", exc)
            raise EvaluationException(f"Chunk 黏连度评估失败: {exc}") from exc
