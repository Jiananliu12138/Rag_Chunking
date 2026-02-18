"""
检索与生成服务层。
封装向量检索、RAG 生成两个核心能力。
"""
from app.core.exceptions import ModelLoadException, RetrievalException
from app.core.logging_config import logger
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.retrieval_schema import (
    RAGRequest,
    RAGResult,
    SearchRequest,
    SearchResult,
    SearchResultItem,
)
from app.services.index_service import IndexService


class RetrievalService:

    def __init__(self, milvus_repo: MilvusRepository):
        self._repo = milvus_repo

    # ── 内部：初始化 LLM ─────────────────────────────────────────────────────

    @staticmethod
    def _build_llm_prompt(context_str: str, query: str) -> str:
        return (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            f"Query: {query}\n"
            "Answer:"
        )

    @staticmethod
    def _call_vllm(
        prompt: str,
        api_base: str,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        import requests

        payload = {
            "model": model_name,
            "prompt": (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        resp = requests.post(f"{api_base}/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def search(self, request: SearchRequest) -> SearchResult:
        logger.info(
            "执行检索: collection=%s, query=%s..., top_k=%d",
            request.collection_name, request.query[:30], request.top_k,
        )
        embed_model = IndexService._load_langchain_embed(request.embed_model_path)
        raw_results = self._repo.search(
            collection_name=request.collection_name,
            query=request.query,
            langchain_embed=embed_model,
            embed_dim=request.embed_dim,
            top_k=request.top_k,
        )
        items = [SearchResultItem(**r) for r in raw_results]
        return SearchResult(
            query=request.query,
            results=items,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )

    def rag_generate(self, request: RAGRequest) -> RAGResult:
        logger.info(
            "RAG 生成: collection=%s, query=%s...",
            request.collection_name, request.query[:30],
        )
        # Step 1: 向量检索
        embed_model = IndexService._load_langchain_embed(request.embed_model_path)
        raw_results = self._repo.search(
            collection_name=request.collection_name,
            query=request.query,
            langchain_embed=embed_model,
            embed_dim=request.embed_dim,
            top_k=request.top_k,
        )
        contexts = [r["text"] for r in raw_results]
        context_str = "\n\n".join(contexts)

        # Step 2: 构造 Prompt 并调用 LLM
        prompt = self._build_llm_prompt(context_str, request.query)
        try:
            answer = self._call_vllm(
                prompt=prompt,
                api_base=request.llm_api_base,
                model_name=request.llm_model_name,
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
            )
        except Exception as exc:
            logger.exception("LLM 调用失败: %s", exc)
            raise RetrievalException(f"LLM 生成失败: {exc}") from exc

        return RAGResult(
            query=request.query,
            answer=answer,
            contexts=contexts,
            collection_name=request.collection_name,
        )
