"""
分块服务层。
将各种分块算法统一封装，对外暴露统一的 chunk(text, method, params) 接口。
"""
import time
from typing import Any

from app.core.path_setup import ensure_paths
from app.core.exceptions import ChunkingException
from app.core.logging_config import logger
from app.schemas.chunk_schema import (
    ChunkMethod,
    ChunkRequest,
    ChunkResult,
    TokenChunkParams,
    SemanticChunkParams,
    LlamaIndexChunkParams,
    LumberChunkParams,
)


class ChunkService:

    # ── Token 分块 ────────────────────────────────────────────────────────────

    def _chunk_token(self, text: str, params: TokenChunkParams) -> list[str]:
        ensure_paths()
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("o200k_base")
        except Exception as exc:
            raise ChunkingException(f"tiktoken 初始化失败: {exc}") from exc

        from lightrag_token_chunk import chunking_by_token_size  # noqa: PLC0415

        results = chunking_by_token_size(
            tokenizer=tokenizer,
            content=text,
            split_by_character=params.split_by_character,
            split_by_character_only=params.split_by_character_only,
            chunk_overlap_token_size=params.chunk_overlap_token_size,
            chunk_token_size=params.chunk_token_size,
        )
        return [r["content"] for r in results if r.get("content", "").strip()]

    # ── Semantic 分块 ─────────────────────────────────────────────────────────

    def _chunk_semantic(self, text: str, params: SemanticChunkParams) -> list[str]:
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core import Document

            embed_model = HuggingFaceEmbedding(model_name=params.embed_model_path)
            splitter = SemanticSplitterNodeParser(
                buffer_size=params.buffer_size,
                breakpoint_percentile_threshold=params.breakpoint_percentile_threshold,
                embed_model=embed_model,
            )
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
            return [
                node.text if hasattr(node, "text") else node.get_content()
                for node in nodes
            ]
        except Exception as exc:
            raise ChunkingException(f"语义分块失败: {exc}") from exc

    # ── LlamaIndex 固定大小分块 ───────────────────────────────────────────────

    def _chunk_llamaindex(self, text: str, params: LlamaIndexChunkParams) -> list[str]:
        try:
            from llama_index.core.node_parser import SimpleNodeParser
            from llama_index.core import Document

            parser = SimpleNodeParser.from_defaults(
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap,
            )
            doc = Document(text=text)
            nodes = parser.get_nodes_from_documents([doc], show_progress=False)
            return [
                node.text if hasattr(node, "text") else node.get_content()
                for node in nodes
            ]
        except Exception as exc:
            raise ChunkingException(f"LlamaIndex 分块失败: {exc}") from exc

    # ── Lumber 分块（LLM 驱动） ───────────────────────────────────────────────

    def _chunk_lumber(self, text: str, params: LumberChunkParams) -> list[str]:
        """
        使用 LLM API 识别语义边界并分块。
        模仿 lumber_chunk.py 的核心逻辑，将文本切分为若干段落后调用 LLM 确定切割点。
        """
        ensure_paths()
        try:
            import re
            import requests

            # 将文本按段落预分割
            raw_paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
            if not raw_paragraphs:
                return [text]

            system_prompt = (
                "You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.\n\n"
                "Task: Find the first paragraph(not the first one) where the content clearly changes "
                "compared to the previous paragraphs.\n\n"
                "Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.\n\n"
                "Additional Considerations: Avoid very long groups of paragraphs. "
                "Aim for a good balance between identifying content shifts and keeping groups manageable."
            )

            id_paragraphs = [f"ID {i}: {p}" for i, p in enumerate(raw_paragraphs)]
            chunks: list[str] = []
            current_start = 0

            while current_start < len(raw_paragraphs) - 1:
                word_count = 0
                window_end = current_start
                while word_count < 550 and window_end + 1 < len(raw_paragraphs):
                    window_end += 1
                    word_count += len(raw_paragraphs[window_end].split())

                window_text = "\n".join(id_paragraphs[current_start: window_end + 1])
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{window_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                try:
                    resp = requests.post(
                        f"{params.llm_api_base}/v1/completions",
                        json={
                            "model": params.model_type,
                            "temperature": params.temperature,
                            "prompt": prompt,
                            "max_tokens": params.max_tokens,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    answer_text: str = resp.json()["choices"][0]["text"]
                except Exception as api_exc:
                    logger.warning("Lumber LLM 调用失败，回退到整窗口分块: %s", api_exc)
                    chunks.append("\n".join(raw_paragraphs[current_start: window_end + 1]))
                    current_start = window_end + 1
                    continue

                # 解析 LLM 返回的切割点 ID
                import re as _re
                match = _re.search(r"ID\s+(\d+)", answer_text)
                if match:
                    split_id = int(match.group(1))
                    split_id = max(current_start + 1, min(split_id, window_end))
                else:
                    split_id = window_end

                chunks.append("\n".join(raw_paragraphs[current_start:split_id]))
                current_start = split_id

            # 最后一段
            if current_start < len(raw_paragraphs):
                chunks.append("\n".join(raw_paragraphs[current_start:]))

            return [c for c in chunks if c.strip()]
        except Exception as exc:
            raise ChunkingException(f"Lumber 分块失败: {exc}") from exc

    # ── 统一入口 ──────────────────────────────────────────────────────────────

    def chunk(self, request: ChunkRequest) -> ChunkResult:
        logger.info("开始分块，方法=%s，文本长度=%d", request.method, len(request.text))
        start = time.time()

        if request.method == ChunkMethod.TOKEN:
            params = request.token_params or TokenChunkParams()
            chunks = self._chunk_token(request.text, params)

        elif request.method == ChunkMethod.SEMANTIC:
            if not request.semantic_params:
                raise ChunkingException("语义分块需要提供 semantic_params（含 embed_model_path）")
            chunks = self._chunk_semantic(request.text, request.semantic_params)

        elif request.method == ChunkMethod.LLAMAINDEX:
            params = request.llamaindex_params or LlamaIndexChunkParams()
            chunks = self._chunk_llamaindex(request.text, params)

        elif request.method == ChunkMethod.LUMBER:
            if not request.lumber_params:
                raise ChunkingException("Lumber 分块需要提供 lumber_params")
            chunks = self._chunk_lumber(request.text, request.lumber_params)

        else:
            raise ChunkingException(f"未知分块方法: {request.method}")

        elapsed = time.time() - start
        logger.info("分块完成，块数=%d，耗时=%.2fs", len(chunks), elapsed)
        return ChunkResult(
            chunks=chunks,
            chunk_count=len(chunks),
            method=request.method,
            time_cost=elapsed,
        )
