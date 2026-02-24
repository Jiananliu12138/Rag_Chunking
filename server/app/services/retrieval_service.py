"""
检索与生成服务层。
封装向量检索、RAG 生成两个核心能力。
"""
import json
import os

from tqdm import tqdm

from app.config import get_settings
from app.core.exceptions import ModelLoadException, RetrievalException
from app.core.logging_config import logger
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.retrieval_schema import (
    RAGGenerateFileRequest,
    RAGGenerateFileResult,
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

    # ── 内部：RAG Prompt 与 LLM 调用（与 retrieval_lite / Qwen_7B_Chat 保持一致）────

    @staticmethod
    def _build_llm_prompt(context_str: str, query: str) -> str:
        """与 eval/LongBench/retrieval_lite.py 中 prompt 完全一致。"""
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
        """
        通过 vLLM completions 接口调用，prompt 包装与 Qwen_7B_Chat.request() 一致：
        system + user( RAG prompt ) + assistant，模型只生成 Answer 部分。
        """
        import requests

        # 与 retrieval_lite 中 Qwen_7B_Chat.request 的包装格式一致
        full_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ).format(prompt=prompt)
        payload = {
            "model": model_name,
            "prompt": full_prompt,
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
        # 没有过滤条件 → 普通全库检索；有条件 → 调用带 metadata 过滤的方法
        if request.filepath is None and request.doc_id is None:
            raw_results = self._repo.search(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=request.embed_dim,
                top_k=request.top_k,
                use_hybrid_search=request.use_hybrid_search,
            )
        else:
            raw_results = self._repo.search_with_metadata_filter(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=request.embed_dim,
                top_k=request.top_k,
                filepath=request.filepath,
                doc_id=request.doc_id,
                use_hybrid_search=request.use_hybrid_search,
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
        if request.filepath is None and request.doc_id is None:
            raw_results = self._repo.search(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=request.embed_dim,
                top_k=request.top_k,
                use_hybrid_search=request.use_hybrid_search,
            )
        else:
            raw_results = self._repo.search_with_metadata_filter(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=request.embed_dim,
                top_k=request.top_k,
                filepath=request.filepath,
                doc_id=request.doc_id,
                use_hybrid_search=request.use_hybrid_search,
            )
        # 同时保留纯文本上下文与带 metadata 的完整结果，方便前端展示与调试
        from app.schemas.retrieval_schema import SearchResultItem  # 避免循环导入

        context_items = [SearchResultItem(**r) for r in raw_results]
        contexts = [item.text for item in context_items]
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
            context_items=context_items,
            collection_name=request.collection_name,
        )

    def rag_generate_file(self, request: RAGGenerateFileRequest) -> RAGGenerateFileResult:
        """
        仿照 eval/LongBench/retrieval_lite.py：从 jsonl 读入问题，逐条检索+生成，结果写入 JSON。
        每行 JSON 需含 input（查询）、_id、answers 等；输出列表元素为 {_id, input, llm_ans, answers, retrieval_list}。
        """
        settings = get_settings()
        embed_model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM
        llm_api_base = request.llm_api_base or settings.DEFAULT_VLLM_API_BASE
        llm_model_name = request.llm_model_name or settings.DEFAULT_VLLM_MODEL_NAME
        if not embed_model_path:
            raise RetrievalException("未配置嵌入模型（DEFAULT_EMBEDDING_MODEL）")
        if not llm_model_name:
            raise RetrievalException("未配置 LLM 模型（DEFAULT_VLLM_MODEL_NAME）")

        embed_model = IndexService._load_langchain_embed(embed_model_path)
        retrieval_save_list: list[dict] = []
        total_failed = 0

        try:
            with open(request.input_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise RetrievalException(f"找不到输入文件: {request.input_path}")
        except Exception as exc:
            raise RetrievalException(f"读取输入文件失败: {exc}") from exc

        with tqdm(total=len(lines), desc="检索+生成", unit="问题") as pbar:
            for line in lines:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("跳过非法 JSON 行: %s", e)
                    total_failed += 1
                    pbar.update(1)
                    continue

                query = data.get("input") or data.get("query") or ""
                if not query:
                    logger.warning("跳过无 input/query 的行 (id=%s)", data.get("_id"))
                    total_failed += 1
                    pbar.update(1)
                    continue

                try:
                    pbar.set_postfix(
                        {"ID": data.get("_id", ""), "Query": query[:10] + "..." if len(query) > 30 else query}
                    )
                    raw_results = self._repo.search(
                        collection_name=request.collection_name,
                        query=query,
                        langchain_embed=embed_model,
                        embed_dim=embed_dim,
                        top_k=request.top_k,
                    )
                    # 与在线 RAG 接口保持一致：既保留纯文本，也保留带 metadata 的完整结果
                    from app.schemas.retrieval_schema import SearchResultItem  # 延迟导入避免循环依赖

                    context_items = [SearchResultItem(**r) for r in raw_results]
                    context_texts = [item.text for item in context_items]
                    context_str = "\n\n".join(context_texts)
                    prompt = self._build_llm_prompt(context_str, query)
                    llm_ans = self._call_vllm(
                        prompt=prompt,
                        api_base=llm_api_base,
                        model_name=llm_model_name,
                        temperature=request.temperature,
                        max_new_tokens=request.max_new_tokens,
                    )
                    save = {
                        "_id": data.get("_id"),
                        "input": query,
                        "llm_ans": llm_ans,
                        "answers": data.get("answers", []),
                        # 兼容原有字段：只包含纯文本
                        "retrieval_list": context_texts,
                        # 新增：完整的检索结果（含 score、filepath、doc_id）
                        "retrieval_items": [item.model_dump() for item in context_items],
                    }
                    retrieval_save_list.append(save)
                except Exception as e:
                    logger.exception("处理单条失败 (ID=%s): %s", data.get("_id", "unknown"), e)
                    total_failed += 1
                pbar.update(1)

        out_dir = os.path.dirname(request.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(request.output_path, "w", encoding="utf-8") as f:
            json.dump(retrieval_save_list, f, indent=4, ensure_ascii=False)

        logger.info("RAG 文件生成完成: 输出=%s, 成功=%d, 失败=%d", request.output_path, len(retrieval_save_list), total_failed)
        return RAGGenerateFileResult(
            output_file=request.output_path,
            total_processed=len(retrieval_save_list),
            total_failed=total_failed,
            message=f"成功 {len(retrieval_save_list)} 条，失败 {total_failed} 条",
        )
