"""
分块服务层。
将各种分块算法统一封装，对外暴露 chunk_text / chunk_file 接口。
"""
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
        """调用 Chunking_Methods 各模块的 chunk_text。"""
        ensure_paths()
        logger.info("开始文本分块，方法=%s，文本长度=%d", request.method, len(request.text))

        if request.method == ChunkMethod.TOKEN:
            from lightrag_token_chunk import chunk_text as _chunk_text

            kwargs = {"text_input": request.text}
            if request.token_params:
                p = request.token_params
                kwargs.update(
                    chunk_token_size=p.chunk_token_size,
                    chunk_overlap_token_size=p.chunk_overlap_token_size,
                    split_by_character=p.split_by_character,
                    split_by_character_only=p.split_by_character_only,
                    cache_dir=p.cache_dir,
                )
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.SEMANTIC:
            if not request.semantic_params:
                raise ChunkingException("语义分块需要提供 semantic_params（含 embed_model_path）")
            from semantic_chunk import chunk_text as _chunk_text  # noqa: PLC0415

            p = request.semantic_params
            kwargs = {
                "text_input": request.text,
                "embed_model_path": p.embed_model_path,
                "buffer_size": p.buffer_size,
                "breakpoint_threshold": p.breakpoint_percentile_threshold,
            }
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.LLAMAINDEX:
            from chunk_llamaindex import chunk_text as _chunk_text  # noqa: PLC0415

            kwargs = {"text_input": request.text}
            if request.llamaindex_params:
                p = request.llamaindex_params
                kwargs.update(
                    chunk_size=p.chunk_size,
                    chunk_overlap=p.chunk_overlap,
                    cache_dir=p.cache_dir,
                )
            raw = _chunk_text(**kwargs)

        elif request.method == ChunkMethod.LUMBER:
            if not request.lumber_params:
                raise ChunkingException("Lumber 分块需要提供 lumber_params")
            from lumber_chunk import chunk_text as _chunk_text  

            p = request.lumber_params
            kwargs = {
                "text_input": request.text,
                "model_type": p.model_type,
                "ds_base_url": p.llm_api_base,
                "temperature": p.temperature,
                "max_tokens": p.max_tokens,
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
        """调用 Chunking_Methods 各模块的 chunk_file。"""
        ensure_paths()
        logger.info("开始文件分块，方法=%s，输入=%s", request.method, request.input_file)

        if request.method == ChunkMethod.TOKEN:
            from lightrag_token_chunk import chunk_file as _chunk_file  # noqa: PLC0415

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
            }
            if request.token_params:
                p = request.token_params
                kwargs.update(
                    chunk_token_size=p.chunk_token_size,
                    chunk_overlap_token_size=p.chunk_overlap_token_size,
                    split_by_character=p.split_by_character,
                    split_by_character_only=p.split_by_character_only,
                    num_workers=p.num_workers,
                    cache_dir=p.cache_dir,
                )
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.SEMANTIC:
            from semantic_chunk import chunk_file as _chunk_file  # noqa: PLC0415

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
            }
            if request.semantic_params:
                p = request.semantic_params
                kwargs.update(
                    embed_model_path=p.embed_model_path,
                    buffer_size=p.buffer_size,
                    breakpoint_threshold=p.breakpoint_percentile_threshold,
                    num_workers=p.num_workers,
                )
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.LLAMAINDEX:
            from chunk_llamaindex import chunk_file as _chunk_file  # noqa: PLC0415

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
            }
            if request.llamaindex_params:
                p = request.llamaindex_params
                kwargs.update(
                    chunk_size=p.chunk_size,
                    chunk_overlap=p.chunk_overlap,
                    num_workers=p.num_workers,
                    cache_dir=p.cache_dir,
                )
            raw = _chunk_file(**kwargs)

        elif request.method == ChunkMethod.LUMBER:
            from lumber_chunk import chunk_file as _chunk_file  # noqa: PLC0415

            kwargs = {
                "input_file": request.input_file,
                "output_dir": request.output_dir,
            }
            if request.lumber_params:
                p = request.lumber_params
                kwargs.update(
                    model_type=p.model_type,
                    ds_base_url=p.llm_api_base,
                    temperature=p.temperature,
                    max_tokens=p.max_tokens,
                    num_workers=p.num_workers,
                )
            raw = _chunk_file(**kwargs)

        else:
            raise ChunkingException(f"未知分块方法: {request.method}")

        logger.info("文件分块完成，输出=%s", raw["output_file"])
        return ChunkFileResult(
            success=raw["success"],
            output_file=raw["output_file"],
            message=raw["message"],
        )
