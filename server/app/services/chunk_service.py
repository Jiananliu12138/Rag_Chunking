import os

from app.config import get_settings
from app.core.path_setup import ensure_paths
from app.core.exceptions import ChunkingException
from app.core.logging_config import logger
from app.schemas.chunk_schema import (
    ChunkMethod,
    ChunkFileRequest,
    ChunkFileResult,
    ChunkTextRequest,
    ChunkTextResult,
    TokenChunkParams,
    SemanticChunkParams,
    LlamaIndexChunkParams,
    LumberChunkParams,
)


class ChunkService:

    # ── 文本分块（按 method 分发到四模块的 chunk_text）───────────────────────

    def chunk_text(self, request: ChunkTextRequest) -> ChunkTextResult:
        """调用 public_method 中的 chunk_text 实现。"""
        ensure_paths()
        logger.info("开始文本分块，方法=%s，文本长度=%d", request.method, len(request.text))

        settings = get_settings()
        default_workers = settings.CHUNK_NUM_WORKERS
        cache_dir = settings.TIKTOKEN_CACHE_DIR or None

        if request.method == ChunkMethod.TOKEN:
            from public_method.chunking.lightrag_token_chunk import chunk_text as _chunk_text

            kwargs = {
                "text_input": request.text,
                "num_workers": default_workers,
                "cache_dir": cache_dir,
            }
            if request.token_params:
                p = request.token_params
                kwargs.update(
                    chunk_token_size=p.chunk_token_size,
                    chunk_overlap_token_size=p.chunk_overlap_token_size,
                    min_chunk_tokens=p.min_chunk_tokens,
                    split_by_character=p.split_by_character,
                    split_use_regex=p.split_use_regex,
                    split_by_character_only=p.split_by_character_only,
                )
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.SEMANTIC:
            from public_method.chunking.semantic_chunk import chunk_text as _chunk_text

            p = request.semantic_params
            model_path = (
                (p.embed_model_path if p and p.embed_model_path else None)
                or settings.DEFAULT_EMBEDDING_MODEL
            )
            if not model_path:
                raise ChunkingException("DEFAULT_EMBEDDING_MODEL 未配置，无法进行语义分块")

            buffer_size = p.buffer_size if p else 1
            breakpoint_threshold = (
                p.breakpoint_percentile_threshold if p else 74
            )
            kwargs = {
                "text_input": request.text,
                "embed_model_path": model_path,
                "buffer_size": buffer_size,
                "breakpoint_threshold": breakpoint_threshold,
                "num_workers": default_workers,
            }
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.LLAMAINDEX:
            from public_method.chunking.chunk_llamaindex import chunk_text as _chunk_text  # noqa: PLC0415
            kwargs = {
                "text_input": request.text,
                "num_workers": default_workers,
                "cache_dir": cache_dir,
            }
            if request.llamaindex_params:
                p = request.llamaindex_params
                kwargs.update(
                    chunk_size=p.chunk_size,
                    chunk_overlap=p.chunk_overlap,
                )
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.LUMBER:
            from public_method.chunking.lumber_chunk import chunk_text as _chunk_text  # noqa: PLC0415

            p = request.lumber_params
            model_type = (p.model_type if p and p.model_type else None) or settings.DEFAULT_LLM_MODEL
            if not model_type:
                raise ChunkingException("DEFAULT_LLM_MODEL 未配置，无法进行 Lumber 分块")
            base = p.llm_api_base if p and p.llm_api_base else settings.DEFAULT_LLM_API_BASE
            ds_base_url = base.rsplit("/v1", 1)[0] if base.endswith("/v1") else base

            temperature = (
                p.temperature if p and p.temperature is not None else 0.2
            )
            max_tokens = (
                p.max_tokens if p and p.max_tokens is not None else 64
            )

            kwargs = {
                "text_input": request.text,
                "model_type": model_type,
                "ds_base_url": ds_base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_workers": default_workers,
            }
            raw = _chunk_text(**kwargs)

        else:
            raise ChunkingException(f"未知分块方法: {request.method}")

        logger.info("文本分块完成，块数=%d，耗时=%.2fs", len(raw["splits"]), raw["time_cost"])
        return ChunkTextResult(
            success=raw["success"],
            splits=raw["splits"],
            time_cost=raw["time_cost"],
            message=raw["message"],
        )

    # ── 文件分块（按 method 分发到四模块的 chunk_file）───────────────────────

    def chunk_file(self, request: ChunkFileRequest) -> ChunkFileResult:
        """调用 public_method 中的 chunk_file 实现。"""
        ensure_paths()
        logger.info("开始文件分块，方法=%s，输入=%s", request.method, request.input_file)

        settings = get_settings()
        default_workers = settings.CHUNK_NUM_WORKERS
        cache_dir = settings.TIKTOKEN_CACHE_DIR or None

        if request.method == ChunkMethod.TOKEN:
            from public_method.chunking.lightrag_token_chunk import chunk_file as _chunk_file

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
                "num_workers": default_workers,
                "cache_dir": cache_dir,
            }
            if request.token_params:
                p = request.token_params
                kwargs.update(
                    chunk_token_size=p.chunk_token_size,
                    chunk_overlap_token_size=p.chunk_overlap_token_size,
                    min_chunk_tokens=p.min_chunk_tokens,
                    split_by_character=p.split_by_character,
                    split_use_regex=p.split_use_regex,
                    split_by_character_only=p.split_by_character_only,
                )
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.SEMANTIC:
            from public_method.chunking.semantic_chunk import chunk_file as _chunk_file

            p = request.semantic_params
            model_path = (
                (p.embed_model_path if p and p.embed_model_path else None)
                or settings.DEFAULT_EMBEDDING_MODEL
            )
            if not model_path:
                raise ChunkingException("DEFAULT_EMBEDDING_MODEL 未配置，无法进行语义文件分块")

            buffer_size = p.buffer_size if p else 1
            breakpoint_threshold = (
                p.breakpoint_percentile_threshold if p else 74
            )
            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
                "embed_model_path": model_path,
                "buffer_size": buffer_size,
                "breakpoint_threshold": breakpoint_threshold,
                "num_workers": default_workers,
            }
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.LLAMAINDEX:
            from public_method.chunking.chunk_llamaindex import chunk_file as _chunk_file  # noqa: PLC0415

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
                "num_workers": default_workers,
                "cache_dir": cache_dir,
            }
            if request.llamaindex_params:
                p = request.llamaindex_params
                kwargs.update(
                    chunk_size=p.chunk_size,
                    chunk_overlap=p.chunk_overlap,
                )
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.LUMBER:
            from public_method.chunking.lumber_chunk import chunk_file as _chunk_file  # noqa: PLC0415

            p = request.lumber_params
            model_type = (p.model_type if p and p.model_type else None) or settings.DEFAULT_LLM_MODEL
            if not model_type:
                raise ChunkingException("DEFAULT_LLM_MODEL 未配置，无法进行 Lumber 文件分块")

            base = p.llm_api_base if p and p.llm_api_base else settings.DEFAULT_LLM_API_BASE
            ds_base_url = base.rsplit("/v1", 1)[0] if base.endswith("/v1") else base

            temperature = (
                p.temperature if p and p.temperature is not None else 0.2
            )
            max_tokens = (
                p.max_tokens if p and p.max_tokens is not None else 64
            )

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
                "model_type": model_type,
                "ds_base_url": ds_base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_workers": default_workers,
            }
            raw = _chunk_file(**kwargs)

        else:
            raise ChunkingException(f"未知分块方法: {request.method}")

        logger.info("文件分块完成，输出=%s", raw["output_file"])
        return ChunkFileResult(
            success=raw["success"],
            output_file=raw["output_file"],
            message=raw["message"],
        )
