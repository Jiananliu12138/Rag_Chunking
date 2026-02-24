"""
Milvus 向量数据库访问层。
封装所有与 Milvus Lite（本地 .db 文件）的交互，对上层屏蔽底层细节。
"""
import os
import time
from pathlib import Path
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.embeddings.langchain import LangchainEmbedding

from app.config import get_settings
from app.core.exceptions import (
    CollectionNotFoundException,
    IndexBuildException,
    RetrievalException,
)
from app.core.logging_config import logger


class MilvusRepository:
    def __init__(self):
        self._settings = get_settings()
        self._data_dir = Path(self._settings.MILVUS_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        uri = (self._settings.MILVUS_URI or "").strip()
        self._online_uri: Optional[str] = uri or None

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _db_path(self, collection_name: str) -> str:
        """根据模式返回 uri：
        - 在线模式：返回在线 URI（不拼接 collection 名）
        - Lite 模式：返回本地 Milvus Lite 数据目录（使用 MILVUS_DATA_DIR）

        注意：Milvus Lite 的本地 uri 应该是一个目录路径，而不是按 collection
        拆分的多个 .db 文件，因此这里直接返回配置中的 MILVUS_DATA_DIR。
        """
        if self._online_uri:
            return self._online_uri
        # 本地 Lite 模式：直接使用配置中的数据目录作为 uri
        return str(self._data_dir / f"{collection_name}.db")

    def _make_embed_model(self, langchain_embed):
        """将 LangChain embedding 包装成 LlamaIndex 格式并注入全局 Settings。"""
        wrapped = LangchainEmbedding(langchain_embed)
        Settings.embed_model = wrapped
        Settings.llm = None
        setting = get_settings()
        try:
            import tiktoken 
            cache_dir = setting.TIKTOKEN_CACHE_DIR
            if cache_dir:
                os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        except ImportError:
            pass
        Settings.tokenizer = lambda text: text.split()
        return wrapped

    def _build_vector_store(self, collection_name: str, overwrite: bool) -> MilvusVectorStore:
        enable_sparse = self._settings.MILVUS_ENABLE_SPARSE
        enable_hybrid = self._settings.MILVUS_ENABLE_HYBRID_SEARCH and enable_sparse
        hybrid_kwargs: dict = {}
        if enable_hybrid:
            hybrid_kwargs["hybrid_ranker"] = self._settings.MILVUS_HYBRID_RANKER
            if self._settings.MILVUS_HYBRID_RANKER == "RRFRanker":
                hybrid_kwargs["hybrid_ranker_params"] = {
                    "k": self._settings.MILVUS_HYBRID_RANKER_K
                }

        sparse_kwargs: dict = {}
        if enable_sparse:
            # 强制使用 Milvus 内置 BM25，而不是默认的 BGEM3 / FlagEmbedding
            sparse_kwargs["sparse_embedding_function"] = BM25BuiltInFunction(
                input_field_names="text",           # 与默认 text_key 一致
                output_field_names="sparse_embedding",  # 与默认 sparse_embedding_field 一致
            )

        return MilvusVectorStore(
            uri=self._db_path(collection_name),
            collection_name=collection_name,
            overwrite=overwrite,
            enable_dense=True,
            enable_sparse=enable_sparse,
            **sparse_kwargs,
            **hybrid_kwargs,
        )

    def _build_vector_store_with_dim(
        self, collection_name: str, dim: int, overwrite: bool
    ) -> MilvusVectorStore:
        enable_sparse = self._settings.MILVUS_ENABLE_SPARSE
        enable_hybrid = self._settings.MILVUS_ENABLE_HYBRID_SEARCH and enable_sparse
        hybrid_kwargs: dict = {}
        if enable_hybrid:
            hybrid_kwargs["hybrid_ranker"] = self._settings.MILVUS_HYBRID_RANKER
            if self._settings.MILVUS_HYBRID_RANKER == "RRFRanker":
                hybrid_kwargs["hybrid_ranker_params"] = {
                    "k": self._settings.MILVUS_HYBRID_RANKER_K
                }

        sparse_kwargs: dict = {}
        if enable_sparse:
            sparse_kwargs["sparse_embedding_function"] = BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse_embedding",
            )

        return MilvusVectorStore(
            uri=self._db_path(collection_name),
            collection_name=collection_name,
            dim=dim,
            overwrite=overwrite,
            enable_dense=True,
            enable_sparse=enable_sparse,
            **sparse_kwargs,
            **hybrid_kwargs,
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def collection_exists(self, collection_name: str) -> bool:
        return Path(self._db_path(collection_name)).exists()

    def list_collections(self) -> list[dict]:
        collections = []
        for db_file in self._data_dir.glob("*.db"):
            collections.append(
                {
                    "name": db_file.stem,
                    "db_file": str(db_file),
                    "size_bytes": db_file.stat().st_size,
                }
            )
        return collections

    def delete_collection(self, collection_name: str) -> None:
        db_file = Path(self._db_path(collection_name))
        if not db_file.exists():
            raise CollectionNotFoundException(
                f"Collection '{collection_name}' 不存在"
            )
        db_file.unlink()
        logger.info("已删除 collection: %s", collection_name)

    def build_index(
        self,
        collection_name: str,
        chunks: list[str],
        langchain_embed,
        embed_dim: int,
        overwrite: bool = True,
        batch_size: int = 100,
        metadatas: Optional[list[dict]] | None = None,
        enable_sparse: Optional[bool] = None,
    ) -> dict:
        try:
            start = time.time()

            raw_count = len(chunks)
            nodes: list[TextNode] = []
            for idx, text in enumerate(chunks):
                if isinstance(text, str) and text.strip():
                    metadata = None
                    if metadatas and idx < len(metadatas):
                        md = metadatas[idx]
                        if isinstance(md, dict) and md:
                            metadata = md
                    if metadata:
                        nodes.append(TextNode(text=text, metadata=metadata))
                    else:
                        nodes.append(TextNode(text=text))
            indexed = len(nodes)

            logger.info(
                "[Milvus] 开始构建索引: collection=%s，原始块数=%d，有效块数=%d，批大小=%d",
                collection_name,
                raw_count,
                indexed,
                batch_size,
            )

            if indexed == 0:
                raise IndexBuildException("没有可用的文本块用于构建索引")

            # 打印前若干个示例块，方便排查数据问题
            preview_count = min(3, indexed)
            logger.info("=" * 60)
            logger.info("[Milvus] 示例文本块（前 %d 个）:", preview_count)
            logger.info("=" * 60)
            for i, node in enumerate(nodes[:preview_count], 1):
                content = node.get_content()
                preview = content[:200] + ("..." if len(content) > 200 else "")
                logger.info(
                    "[示例 %d] (长度: %d 字符)\n%s\n%s",
                    i,
                    len(content),
                    preview,
                    "-" * 60,
                )

            # 包装嵌入模型并构建向量存储
            self._make_embed_model(langchain_embed)
            # 若请求显式指定是否启用稀疏，则在首次建索引时覆盖默认配置
            if enable_sparse is not None:
                original_sparse = self._settings.MILVUS_ENABLE_SPARSE
                self._settings.MILVUS_ENABLE_SPARSE = enable_sparse
            else:
                original_sparse = None
            try:
                vector_store = self._build_vector_store_with_dim(
                    collection_name, embed_dim, overwrite
                )
            finally:
                if original_sparse is not None:
                    self._settings.MILVUS_ENABLE_SPARSE = original_sparse
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 分批写入节点，仿照 base_lite 的批处理结构
            total_batches = (indexed + batch_size - 1) // batch_size
            logger.info("[Milvus] 开始分批写入，共 %d 批", total_batches)

            for batch_start in range(0, indexed, batch_size):
                batch_no = batch_start // batch_size + 1
                batch_nodes = nodes[batch_start : batch_start + batch_size]

                logger.info(
                    "[Milvus] 写入第 %d/%d 批（%d 个节点）...",
                    batch_no,
                    total_batches,
                    len(batch_nodes),
                )

                VectorStoreIndex(
                    batch_nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )

                # 后续批次使用追加模式
                vector_store = self._build_vector_store(collection_name, overwrite=False)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

            elapsed = time.time() - start
            logger.info("[Milvus] 索引构建完成，collection=%s，耗时 %.2fs", collection_name, elapsed)

            # 汇总 metadata，便于前端展示「本次索引涉及哪些文档 / doc_id」
            unique_filepaths: set[str] = set()
            unique_doc_ids: set[str] = set()
            if metadatas:
                for md in metadatas:
                    if not isinstance(md, dict):
                        continue
                    fp = md.get("filepath")
                    if isinstance(fp, str) and fp:
                        unique_filepaths.add(fp)
                    doc = md.get("source_doc_id")
                    if doc is not None:
                        # 统一转成字符串，兼容数字 / 字符串 doc_id
                        unique_doc_ids.add(str(doc))

            return {
                "collection_name": collection_name,
                "total_chunks": len(chunks),
                "indexed_chunks": indexed,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
                "filepaths": sorted(unique_filepaths),
                "doc_ids": sorted(unique_doc_ids),
            }
        except Exception as exc:
            logger.exception("索引构建失败: %s", exc)
            raise IndexBuildException(f"索引构建失败: {exc}") from exc

    def add_index(
        self,
        collection_name: str,
        chunks: list[str],
        langchain_embed,
        batch_size: int = 8000,
        metadatas: Optional[list[dict]] | None = None,
    ) -> dict:
        try:
            start = time.time()

            raw_count = len(chunks)
            nodes: list[TextNode] = []
            for idx, text in enumerate(chunks):
                if isinstance(text, str) and text.strip():
                    metadata = None
                    if metadatas and idx < len(metadatas):
                        md = metadatas[idx]
                        if isinstance(md, dict) and md:
                            metadata = md
                    if metadata:
                        nodes.append(TextNode(text=text, metadata=metadata))
                    else:
                        nodes.append(TextNode(text=text))
            added = len(nodes)

            logger.info(
                "[Milvus] 开始追加数据到索引: collection=%s，原始块数=%d，有效块数=%d，批大小=%d",
                collection_name,
                raw_count,
                added,
                batch_size,
            )

            if added == 0:
                raise IndexBuildException("没有可用的文本块用于追加到索引")

            self._make_embed_model(langchain_embed)

            vector_store = self._build_vector_store(collection_name, overwrite=False)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            total_batches = (added + batch_size - 1) // batch_size
            logger.info("[Milvus] 开始分批追加，共 %d 批", total_batches)

            for batch_start in range(0, added, batch_size):
                batch_no = batch_start // batch_size + 1
                batch_nodes = nodes[batch_start : batch_start + batch_size]

                logger.info(
                    "[Milvus] 追加第 %d/%d 批（%d 个节点）...",
                    batch_no,
                    total_batches,
                    len(batch_nodes),
                )

                VectorStoreIndex(
                    batch_nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )

            elapsed = time.time() - start
            logger.info(
                "[Milvus] 数据追加完成，collection=%s，耗时 %.2fs", collection_name, elapsed
            )

            unique_filepaths: set[str] = set()
            unique_doc_ids: set[str] = set()
            if metadatas:
                for md in metadatas:
                    if not isinstance(md, dict):
                        continue
                    fp = md.get("filepath")
                    if isinstance(fp, str) and fp:
                        unique_filepaths.add(fp)
                    doc = md.get("source_doc_id")
                    if doc is not None:
                        unique_doc_ids.add(str(doc))

            return {
                "collection_name": collection_name,
                "added_chunks": added,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
                "filepaths": sorted(unique_filepaths),
                "doc_ids": sorted(unique_doc_ids),
            }
        except Exception as exc:
            logger.exception("追加索引失败: %s", exc)
            raise IndexBuildException(f"追加索引失败: {exc}") from exc

    def load_query_engine(
        self,
        collection_name: str,
        langchain_embed,
        embed_dim: int,
        top_k: int = 5,
        use_hybrid_search: Optional[bool] = None,
    ) -> RetrieverQueryEngine:
        """
        加载已有 collection 并返回可执行检索的 QueryEngine。
        """
        if not self.collection_exists(collection_name):
            raise CollectionNotFoundException(
                f"Collection '{collection_name}' 不存在，请先构建索引"
            )
        try:
            self._make_embed_model(langchain_embed)
            vector_store = self._build_vector_store_with_dim(
                collection_name, embed_dim, overwrite=False
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex([], storage_context=storage_context)
            # 计算本次请求是否启用 Hybrid：请求参数优先，其次全局配置；同时要求全局允许稀疏
            effective_use_hybrid = (
                use_hybrid_search
                if use_hybrid_search is not None
                else self._settings.MILVUS_ENABLE_HYBRID_SEARCH
            )
            effective_use_hybrid = effective_use_hybrid and self._settings.MILVUS_ENABLE_SPARSE

            if effective_use_hybrid:
                retriever = VectorIndexRetriever(
                    index=vector_index,
                    similarity_top_k=top_k,
                    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                )
            else:
                retriever = VectorIndexRetriever(
                    index=vector_index,
                    similarity_top_k=top_k,
                )
            return RetrieverQueryEngine(retriever=retriever)
        except CollectionNotFoundException:
            raise
        except Exception as exc:
            logger.exception("加载 QueryEngine 失败: %s", exc)
            raise RetrievalException(f"加载检索引擎失败: {exc}") from exc

    def search(
        self,
        collection_name: str,
        query: str,
        langchain_embed,
        embed_dim: int,
        top_k: int = 5,
        use_hybrid_search: Optional[bool] = None,
    ) -> list[dict]:
        """
        在指定 collection 中检索与 query 最相关的文档（不带任何 metadata 过滤）。

        Returns:
            [{"text": ..., "score": ...}, ...]
        """
        try:
            engine = self.load_query_engine(
                collection_name,
                langchain_embed,
                embed_dim,
                top_k,
                use_hybrid_search=use_hybrid_search,
            )
            response = engine.query(query)
            results: list[dict] = []
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                meta = getattr(node, "metadata", {}) or {}
                results.append(
                    {
                        "text": node.get_content(),
                        "score": float(node_with_score.score)
                        if node_with_score.score is not None
                        else None,
                        "filepath": meta.get("filepath"),
                        # 统一以字符串形式返回 doc_id，兼容数字 / 字符串
                        "doc_id": str(meta["source_doc_id"]) if "source_doc_id" in meta else None,
                    }
                )
            return results
        except CollectionNotFoundException:
            raise
        except Exception as exc:
            logger.exception("检索失败: %s", exc)
            raise RetrievalException(f"检索失败: {exc}") from exc

    def search_with_metadata_filter(
        self,
        collection_name: str,
        query: str,
        langchain_embed,
        embed_dim: int,
        top_k: int = 5,
        filepath: Optional[str] = None,
        doc_id: Optional[str] = None,
        use_hybrid_search: Optional[bool] = None,
    ) -> list[dict]:
        """
        在指定 collection 中检索时，基于 metadata（如 filepath、doc_id）做条件过滤。
        使用 LlamaIndex 的 MetadataFilters，在 Milvus 端执行过滤，而非 Python 侧后过滤。
        """
        try:
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundException(
                    f"Collection '{collection_name}' 不存在，请先构建索引"
                )

            # 设置嵌入模型 & 加载向量存储
            self._make_embed_model(langchain_embed)
            vector_store = self._build_vector_store_with_dim(
                collection_name, embed_dim, overwrite=False
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex([], storage_context=storage_context)

            # 构造 metadata filters
            from llama_index.core.vector_stores import (  # 本地导入避免未使用警告
                MetadataFilter,
                MetadataFilters,
                FilterCondition,
                FilterOperator,
            )

            filter_list: list[MetadataFilter] = []
            if filepath is not None:
                filter_list.append(
                    MetadataFilter(
                        key="filepath",
                        value=str(filepath),
                        operator=FilterOperator.EQ,
                    )
                )
            if doc_id is not None:
                filter_list.append(
                    MetadataFilter(
                        key="source_doc_id",
                        value=str(doc_id),
                        operator=FilterOperator.EQ,
                    )
                )

            filters_obj = (
                MetadataFilters(filters=filter_list, condition=FilterCondition.AND)
                if filter_list
                else None
            )

            # 计算本次请求是否启用 Hybrid：请求参数优先，其次全局配置；同时要求全局允许稀疏
            effective_use_hybrid = (
                use_hybrid_search
                if use_hybrid_search is not None
                else self._settings.MILVUS_ENABLE_HYBRID_SEARCH
            )
            effective_use_hybrid = effective_use_hybrid and self._settings.MILVUS_ENABLE_SPARSE

            retriever_kwargs = {
                "similarity_top_k": top_k,
                "filters": filters_obj,
            }
            if effective_use_hybrid:
                retriever_kwargs["vector_store_query_mode"] = VectorStoreQueryMode.HYBRID

            retriever = vector_index.as_retriever(**retriever_kwargs)
            response_nodes = retriever.retrieve(query)

            results: list[dict] = []
            for node_with_score in response_nodes:
                node = node_with_score.node
                meta = getattr(node, "metadata", {}) or {}
                results.append(
                    {
                        "text": node.get_content(),
                        "score": float(node_with_score.score)
                        if node_with_score.score is not None
                        else None,
                        "filepath": meta.get("filepath"),
                        "doc_id": str(meta["source_doc_id"]) if "source_doc_id" in meta else None,
                    }
                )
            return results
        except CollectionNotFoundException:
            raise
        except Exception as exc:
            logger.exception("带过滤条件检索失败: %s", exc)
            raise RetrievalException(f"检索失败: {exc}") from exc
