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
        from app.schemas.retrieval_schema import SearchResultItem  # 避免循环导入

        # Step 1: （可选）向量检索 + 上下文构造
        if request.enable_rag:
            logger.info(
                "RAG 生成: collection=%s, query=%s...",
                request.collection_name,
                request.query[:30],
            )
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

            context_items = [SearchResultItem(**r) for r in raw_results]
            contexts = [item.text for item in context_items]
            context_str = "\n\n".join(contexts)
        else:
            # 纯 LLM 调用：不做任何检索，仅用用户 query 作为上下文
            logger.info(
                "纯 LLM 生成（未启用 RAG）: query=%s...",
                request.query[:30],
            )
            context_items: list[SearchResultItem] = []
            contexts: list[str] = []
            context_str = ""

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
        每行 JSON 需含 user_input/input/query、_id、answers 等；
        输出列表元素为 {_id, input, llm_ans, answers, rag_retrieval, gold_reference}。
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

        def _flatten_reference_meta(meta_obj: object) -> list[dict]:
            if not isinstance(meta_obj, dict):
                return []
            ref_ctx = meta_obj.get("reference_contexts")
            if ref_ctx is None:
                return []

            out: list[dict] = []

            def walk(x: object) -> None:
                if isinstance(x, list):
                    for it in x:
                        walk(it)
                    return
                if isinstance(x, dict):
                    doc_id = x.get("doc_id")
                    chunk_id = x.get("chunk_id")
                    source_filepath = x.get("source_filepath")
                    if doc_id is not None or chunk_id is not None or source_filepath is not None:
                        out.append(
                            {
                                "doc_id": str(doc_id) if doc_id is not None else None,
                                "chunk_id": str(chunk_id) if chunk_id is not None else None,
                                "filepath": str(source_filepath) if source_filepath is not None else None,
                            }
                        )

            walk(ref_ctx)
            return out

        def _build_gold_reference_items(contexts: list[str], metas: list[dict]) -> list[dict]:
            max_len = max(len(contexts), len(metas))
            items: list[dict] = []
            for i in range(max_len):
                text = contexts[i] if i < len(contexts) else None
                meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                items.append(
                    {
                        "text": text,
                        "filepath": meta.get("filepath"),
                        "doc_id": meta.get("doc_id"),
                        "chunk_id": meta.get("chunk_id"),
                    }
                )
            return items

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

                query = data.get("user_input") or data.get("input") or data.get("query") or ""
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
                    rag_retrieval = [
                        {
                            "text": item.text,
                            "filepath": item.filepath,
                            "doc_id": item.doc_id,
                            "chunk_id": item.chunk_id,
                        }
                        for item in context_items
                    ]
                    reference_contexts = data.get("reference_contexts")
                    if isinstance(reference_contexts, list):
                        gold_reference_contexts = [str(x) for x in reference_contexts]
                    elif reference_contexts is None:
                        gold_reference_contexts = []
                    else:
                        gold_reference_contexts = [str(reference_contexts)]
                    gold_reference_meta = _flatten_reference_meta(data.get("meta"))
                    gold_reference = _build_gold_reference_items(
                        gold_reference_contexts,
                        gold_reference_meta,
                    )
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
                        "rag_retrieval": rag_retrieval,
                        "gold_reference": gold_reference,
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
