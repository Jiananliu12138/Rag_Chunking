"""
检索与生成服务层。
封装向量检索、RAG 生成两个核心能力。
"""
import json
import os
from typing import Any, Optional

from tqdm import tqdm


from app.config import get_settings
from app.core.model_factory import get_cross_encoder
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
        通过 vLLM/OpenAI-compatible chat.completions 接口调用。
        """
        import requests

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        resp = requests.post(f"{api_base}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise RetrievalException(f"LLM 返回中缺少 choices: {data}")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            if text_parts:
                return "".join(text_parts).strip()
        raise RetrievalException(f"LLM 返回了无法解析的 chat content: {data}")

    @staticmethod
    def _normalize_retrieve_and_final_top_k(
        request_top_k: int,
        rerank_candidate_k: Optional[int],
        rerank_top_k: Optional[int],
    ) -> tuple[int, int]:
        final_top_k = int(rerank_top_k) if rerank_top_k is not None else int(request_top_k)
        retrieve_top_k = int(rerank_candidate_k) if rerank_candidate_k is not None else final_top_k
        retrieve_top_k = max(retrieve_top_k, final_top_k)
        return retrieve_top_k, final_top_k

    @staticmethod
    def _to_search_result_items(raw_results: list[dict[str, Any]]) -> list[SearchResultItem]:
        items: list[SearchResultItem] = []
        for raw in raw_results:
            item = SearchResultItem(**raw)
            if item.retrieval_score is None and item.score is not None:
                item = item.model_copy(update={"retrieval_score": float(item.score)})
            items.append(item)
        return items

    @staticmethod
    def _validate_rerank_type(rerank_type: str) -> str:
        normalized = rerank_type.strip().lower()
        if normalized != "cross_encoder":
            raise RetrievalException(f"不支持的 rerank_type: {rerank_type}（当前仅支持 cross_encoder）")
        return normalized

    @staticmethod
    def _count_tokens(
        tokenizer: Any,
        text: str,
        *,
        text_pair: str | None = None,
    ) -> int | None:
        try:
            normalized_text = text.replace("\n", " ")
            normalized_pair = text_pair.replace("\n", " ") if text_pair is not None else None
            encoded = tokenizer(
                normalized_text,
                text_pair=normalized_pair,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return len(input_ids[0])
                return len(input_ids)
        except Exception:
            return None
        return None

    @classmethod
    def _warn_on_overlong_embedding_query(
        cls,
        *,
        query: str,
        embed_model: Any,
        embedding_max_tokens: int | None,
        operation: str,
        collection_name: str,
    ) -> None:
        if not embedding_max_tokens:
            return
        client = getattr(embed_model, "_client", None)
        tokenizer = getattr(client, "tokenizer", None) if client is not None else None
        if tokenizer is None:
            return
        token_count = cls._count_tokens(tokenizer, query)
        if token_count is None or token_count <= embedding_max_tokens:
            return
        logger.warning(
            "检测到超长 query embedding 输入: collection=%s, operation=%s, query_tokens=%d, embedding_max_tokens=%d。"
            "该 query 在 SentenceTransformers 编码时会被截断。",
            collection_name,
            operation,
            token_count,
            embedding_max_tokens,
        )

    @classmethod
    def _warn_on_overlong_rerank_pairs(
        cls,
        *,
        query: str,
        context_items: list[SearchResultItem],
        rerank_model: Any,
        rerank_max_length: int | None,
        operation: str,
        collection_name: str,
    ) -> None:
        if not rerank_max_length:
            return
        tokenizer = getattr(rerank_model, "tokenizer", None)
        if tokenizer is None:
            return

        overlong_count = 0
        max_seen_tokens = 0
        examples: list[str] = []
        for idx, item in enumerate(context_items):
            token_count = cls._count_tokens(tokenizer, query, text_pair=item.text)
            if token_count is None:
                continue
            max_seen_tokens = max(max_seen_tokens, token_count)
            if token_count <= rerank_max_length:
                continue
            overlong_count += 1
            if len(examples) < 5:
                examples.append(
                    "idx=%d tokens=%d file_path=%s doc_id=%s chunk_id=%s"
                    % (
                        idx,
                        token_count,
                        item.filepath or "-",
                        item.doc_id or "-",
                        item.chunk_id or "-",
                    )
                )

        if overlong_count == 0:
            return
        logger.warning(
            "检测到超长 rerank pair: collection=%s, operation=%s, overlong=%d/%d, rerank_max_length=%d, max_seen_tokens=%d。"
            "这些 query+chunk pair 在 CrossEncoder 编码时会被截断。示例: %s",
            collection_name,
            operation,
            overlong_count,
            len(context_items),
            rerank_max_length,
            max_seen_tokens,
            " | ".join(examples),
        )

    @staticmethod
    def _load_cross_encoder(
        model_path: str,
        device: str,
        *,
        max_length: int | None = None,
    ) -> Any:
        try:
            return get_cross_encoder(
                model_path=model_path,
                device=device,
                max_length=max_length,
            )
        except Exception as exc:
            raise RetrievalException(f"加载 rerank 模型失败: {exc}") from exc

    def _rerank_items_cross_encoder(
        self,
        *,
        query: str,
        context_items: list[SearchResultItem],
        model_path: str,
        device: str,
        max_length: int | None,
        final_top_k: int,
        operation: str,
        collection_name: str,
    ) -> list[SearchResultItem]:
        if not context_items:
            return []
        model = self._load_cross_encoder(
            model_path=model_path,
            device=device,
            max_length=max_length,
        )
        self._warn_on_overlong_rerank_pairs(
            query=query,
            context_items=context_items,
            rerank_model=model,
            rerank_max_length=max_length,
            operation=operation,
            collection_name=collection_name,
        )
        pairs = [(query, item.text) for item in context_items]
        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:
            raise RetrievalException(f"执行 rerank 失败: {exc}") from exc

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        scored: list[tuple[float, float, int, SearchResultItem]] = []
        for idx, (item, raw_score) in enumerate(zip(context_items, raw_scores)):
            rerank_score = float(raw_score)
            retrieval_score = (
                float(item.retrieval_score)
                if item.retrieval_score is not None
                else (float(item.score) if item.score is not None else float("-inf"))
            )
            scored.append((rerank_score, retrieval_score, idx, item))

        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        reranked_items: list[SearchResultItem] = []
        for rerank_score, retrieval_score, _, item in scored[:final_top_k]:
            reranked_items.append(
                item.model_copy(
                    update={
                        "score": rerank_score,
                        "rerank_score": rerank_score,
                        "retrieval_score": retrieval_score if retrieval_score != float("-inf") else None,
                    }
                )
            )
        return reranked_items

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def search(self, request: SearchRequest) -> SearchResult:
        logger.info(
            "执行检索: collection=%s, query=%s..., top_k=%d",
            request.collection_name, request.query[:30], request.top_k,
        )
        settings = get_settings()
        embed_model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM
        retrieve_top_k, final_top_k = self._normalize_retrieve_and_final_top_k(
            request_top_k=request.top_k,
            rerank_candidate_k=request.rerank_candidate_k if request.rerank_enabled else None,
            rerank_top_k=request.rerank_top_k if request.rerank_enabled else None,
        )
        embedding_max_tokens = (
            request.embed_max_tokens
            if request.embed_max_tokens is not None
            else settings.DEFAULT_EMBEDDING_MAX_TOKENS
        )
        embed_model, _ = IndexService._load_langchain_embed(
            embed_model_path,
            embedding_device=settings.DEFAULT_EMBEDDING_DEVICE,
            embedding_max_tokens=embedding_max_tokens,
        )
        self._warn_on_overlong_embedding_query(
            query=request.query,
            embed_model=embed_model,
            embedding_max_tokens=embedding_max_tokens,
            operation="search",
            collection_name=request.collection_name,
        )
        # 没有过滤条件 → 普通全库检索；有条件 → 调用带 metadata 过滤的方法
        if request.filepath is None and request.doc_id is None:
            raw_results = self._repo.search(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=embed_dim,
                top_k=retrieve_top_k,
                use_hybrid_search=request.use_hybrid_search,
            )
        else:
            raw_results = self._repo.search_with_metadata_filter(
                collection_name=request.collection_name,
                query=request.query,
                langchain_embed=embed_model,
                embed_dim=embed_dim,
                top_k=retrieve_top_k,
                filepath=request.filepath,
                doc_id=request.doc_id,
                use_hybrid_search=request.use_hybrid_search,
            )
        items = self._to_search_result_items(raw_results)
        if request.rerank_enabled:
            self._validate_rerank_type(request.rerank_type)
            rerank_model_path = request.rerank_model_path or settings.DEFAULT_RERANK_MODEL
            rerank_device = request.rerank_device or settings.DEFAULT_RERANK_DEVICE
            rerank_max_length = (
                request.rerank_max_length
                if request.rerank_max_length is not None
                else settings.DEFAULT_RERANK_MAX_LENGTH
            )
            if not rerank_model_path:
                raise RetrievalException("已启用 rerank，但未提供 rerank_model_path 且服务端未配置默认模型")
            items = self._rerank_items_cross_encoder(
                query=request.query,
                context_items=items,
                model_path=rerank_model_path,
                device=rerank_device,
                max_length=rerank_max_length,
                final_top_k=final_top_k,
                operation="search",
                collection_name=request.collection_name,
            )
        else:
            items = items[:final_top_k]
        return SearchResult(
            query=request.query,
            results=items,
            collection_name=request.collection_name,
            top_k=final_top_k,
        )

    def rag_generate(self, request: RAGRequest) -> RAGResult:
        settings = get_settings()
        llm_api_base = request.llm_api_base or settings.DEFAULT_LLM_API_BASE
        llm_model_name = request.llm_model_name or settings.DEFAULT_LLM_MODEL
        temperature = (
            request.temperature
            if request.temperature is not None
            else settings.DEFAULT_LLM_TEMPERATURE
        )
        max_new_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else settings.DEFAULT_LLM_MAX_TOKENS
        )
        if not llm_model_name:
            raise RetrievalException("未配置 LLM 模型（DEFAULT_LLM_MODEL）")

        # Step 1: （可选）向量检索 + 上下文构造
        if request.enable_rag:
            logger.info(
                "RAG 生成: collection=%s, query=%s...",
                request.collection_name,
                request.query[:30],
            )
            embed_model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL
            embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM
            retrieve_top_k, final_top_k = self._normalize_retrieve_and_final_top_k(
                request_top_k=request.top_k,
                rerank_candidate_k=request.rerank_candidate_k if request.rerank_enabled else None,
                rerank_top_k=request.rerank_top_k if request.rerank_enabled else None,
            )
            embedding_max_tokens = (
                request.embed_max_tokens
                if request.embed_max_tokens is not None
                else settings.DEFAULT_EMBEDDING_MAX_TOKENS
            )
            embed_model, _ = IndexService._load_langchain_embed(
                embed_model_path,
                embedding_device=settings.DEFAULT_EMBEDDING_DEVICE,
                embedding_max_tokens=embedding_max_tokens,
            )
            self._warn_on_overlong_embedding_query(
                query=request.query,
                embed_model=embed_model,
                embedding_max_tokens=embedding_max_tokens,
                operation="rag_generate",
                collection_name=request.collection_name,
            )

            if request.filepath is None and request.doc_id is None:
                raw_results = self._repo.search(
                    collection_name=request.collection_name,
                    query=request.query,
                    langchain_embed=embed_model,
                    embed_dim=embed_dim,
                    top_k=retrieve_top_k,
                    use_hybrid_search=request.use_hybrid_search,
                )
            else:
                raw_results = self._repo.search_with_metadata_filter(
                    collection_name=request.collection_name,
                    query=request.query,
                    langchain_embed=embed_model,
                    embed_dim=embed_dim,
                    top_k=retrieve_top_k,
                    filepath=request.filepath,
                    doc_id=request.doc_id,
                    use_hybrid_search=request.use_hybrid_search,
                )

            context_items = self._to_search_result_items(raw_results)
            if request.rerank_enabled:
                self._validate_rerank_type(request.rerank_type)
                rerank_model_path = request.rerank_model_path or settings.DEFAULT_RERANK_MODEL
                rerank_device = request.rerank_device or settings.DEFAULT_RERANK_DEVICE
                rerank_max_length = (
                    request.rerank_max_length
                    if request.rerank_max_length is not None
                    else settings.DEFAULT_RERANK_MAX_LENGTH
                )
                if not rerank_model_path:
                    raise RetrievalException("已启用 rerank，但未提供 rerank_model_path 且服务端未配置默认模型")
                context_items = self._rerank_items_cross_encoder(
                    query=request.query,
                    context_items=context_items,
                    model_path=rerank_model_path,
                    device=rerank_device,
                    max_length=rerank_max_length,
                    final_top_k=final_top_k,
                    operation="rag_generate",
                    collection_name=request.collection_name,
                )
            else:
                context_items = context_items[:final_top_k]
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
                api_base=llm_api_base,
                model_name=llm_model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
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
        每行 JSON 需含 user_input/input/query，gold 参考可来自 reference_contexts + meta.reference_contexts。
        输出列表元素为 {_id, question_id, input, llm_ans, reference, rag_retrieval, gold_reference, generation_api_base, generation_model_name}。
        """
        settings = get_settings()
        embed_model_path = request.embed_model_path or settings.DEFAULT_EMBEDDING_MODEL
        embed_dim = request.embed_dim or settings.DEFAULT_EMBEDDING_DIM
        embedding_max_tokens = (
            request.embed_max_tokens
            if request.embed_max_tokens is not None
            else settings.DEFAULT_EMBEDDING_MAX_TOKENS
        )
        llm_api_base = request.llm_api_base or settings.DEFAULT_LLM_API_BASE
        llm_model_name = request.llm_model_name or settings.DEFAULT_LLM_MODEL
        temperature = (
            request.temperature
            if request.temperature is not None
            else settings.DEFAULT_LLM_TEMPERATURE
        )
        max_new_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else settings.DEFAULT_LLM_MAX_TOKENS
        )
        if not embed_model_path:
            raise RetrievalException("未配置嵌入模型（DEFAULT_EMBEDDING_MODEL）")
        if not llm_model_name:
            raise RetrievalException("未配置 LLM 模型（DEFAULT_LLM_MODEL）")
        retrieve_top_k, final_top_k = self._normalize_retrieve_and_final_top_k(
            request_top_k=request.top_k,
            rerank_candidate_k=request.rerank_candidate_k if request.rerank_enabled else None,
            rerank_top_k=request.rerank_top_k if request.rerank_enabled else None,
        )
        rerank_model_path = request.rerank_model_path or settings.DEFAULT_RERANK_MODEL
        rerank_device = request.rerank_device or settings.DEFAULT_RERANK_DEVICE
        rerank_max_length = (
            request.rerank_max_length
            if request.rerank_max_length is not None
            else settings.DEFAULT_RERANK_MAX_LENGTH
        )
        if request.rerank_enabled:
            self._validate_rerank_type(request.rerank_type)
            if not rerank_model_path:
                raise RetrievalException("已启用 rerank，但未提供 rerank_model_path 且服务端未配置默认模型")

        embed_model, _ = IndexService._load_langchain_embed(
            embed_model_path,
            embedding_device=settings.DEFAULT_EMBEDDING_DEVICE,
            embedding_max_tokens=embedding_max_tokens,
        )
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
            for row_idx, line in enumerate(lines, start=1):
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
                    logger.warning("跳过无 input/query 的行 (auto_id=%s)", row_idx)
                    total_failed += 1
                    pbar.update(1)
                    continue

                try:
                    auto_id = row_idx
                    pbar.set_postfix(
                        {"ID": auto_id, "Query": query[:10] + "..." if len(query) > 30 else query}
                    )
                    self._warn_on_overlong_embedding_query(
                        query=query,
                        embed_model=embed_model,
                        embedding_max_tokens=embedding_max_tokens,
                        operation=f"rag_generate_file:auto_id={auto_id}",
                        collection_name=request.collection_name,
                    )
                    raw_results = self._repo.search(
                        collection_name=request.collection_name,
                        query=query,
                        langchain_embed=embed_model,
                        embed_dim=embed_dim,
                        top_k=retrieve_top_k,
                        use_hybrid_search=request.use_hybrid_search,
                    )
                    # 与在线 RAG 接口保持一致：既保留纯文本，也保留带 metadata 的完整结果
                    context_items = self._to_search_result_items(raw_results)
                    if request.rerank_enabled:
                        context_items = self._rerank_items_cross_encoder(
                            query=query,
                            context_items=context_items,
                            model_path=rerank_model_path,
                            device=rerank_device,
                            max_length=rerank_max_length,
                            final_top_k=final_top_k,
                            operation=f"rag_generate_file:auto_id={auto_id}",
                            collection_name=request.collection_name,
                        )
                    else:
                        context_items = context_items[:final_top_k]
                    context_texts = [item.text for item in context_items]
                    context_str = "\n\n".join(context_texts)
                    rag_retrieval = [
                        {
                            "text": item.text,
                            "retrieval_score": item.retrieval_score,
                            "rerank_score": item.rerank_score,
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
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
                    source_question_id = data.get("question_id")
                    source_record_id = data.get("_id")
                    save = {
                        "_id": source_record_id if source_record_id is not None else (source_question_id if source_question_id is not None else auto_id),
                        "question_id": source_question_id if source_question_id is not None else auto_id,
                        "input": query,
                        "llm_ans": llm_ans,
                        "reference": data.get("reference"),
                        "rag_retrieval": rag_retrieval,
                        "gold_reference": gold_reference,
                        "generation_api_base": llm_api_base,
                        "generation_model_name": llm_model_name,
                    }
                    retrieval_save_list.append(save)
                except Exception as e:
                    logger.exception("处理单条失败 (auto_id=%s): %s", row_idx, e)
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
