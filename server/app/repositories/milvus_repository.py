"""
Milvus 向量数据库访问层。
封装所有与 Milvus Lite（本地 .db 文件）的交互，对上层屏蔽底层细节。
"""
import os
import threading
import time
from pathlib import Path
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import MilvusClient
from llama_index.embeddings.langchain import LangchainEmbedding

from app.config import get_settings
from app.core.exceptions import (
    ChunkingException,
    CollectionNotFoundException,
    IndexBuildException,
    RetrievalException,
)
from app.core.logging_config import logger


class MilvusRepository:
    # Protects LlamaIndex global Settings mutation (_make_embed_model)
    _settings_lock = threading.Lock()
    # Per-collection locks for meta JSON read-modify-write
    _meta_locks: dict[str, threading.Lock] = {}
    _meta_locks_guard = threading.Lock()

    def __init__(self):
        self._settings = get_settings()
        self._data_dir = Path(self._settings.MILVUS_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        uri = (self._settings.MILVUS_URI or "").strip()
        self._online_uri: Optional[str] = uri or None

    def _get_meta_lock(self, collection_name: str) -> threading.Lock:
        with self._meta_locks_guard:
            if collection_name not in self._meta_locks:
                self._meta_locks[collection_name] = threading.Lock()
            return self._meta_locks[collection_name]

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

    def _build_text_node(self, text: str, metadata: Optional[dict]) -> TextNode:
        node_kwargs: dict = {"text": text}
        if metadata:
            node_kwargs["metadata"] = metadata
        doc_id = metadata.get("doc_id") if isinstance(metadata, dict) else None
        if doc_id is not None:
            node_kwargs["relationships"] = {
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=str(doc_id))
            }
        return TextNode(**node_kwargs)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def collection_exists(self, collection_name: str) -> bool:
        return Path(self._db_path(collection_name)).exists()

    def list_collections(self) -> list[dict]:
        collections: list[dict] = []
        for db_file in self._data_dir.glob("*.db"):
            name = db_file.stem
            size = db_file.stat().st_size

            # 尝试读取同名 meta JSON，补充 filepaths / doc_ids / doc_chunk_map 信息
            meta_path = self._data_dir / f"{name}_meta.json"
            filepaths: list[str] = []
            doc_ids: list[str] = []
            doc_chunk_map: dict[str, list[str]] = {}
            if meta_path.exists():
                try:
                    import json

                    raw = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
                    fps = raw.get("filepaths") or []
                    dids = raw.get("doc_ids") or []
                    dc_map = raw.get("doc_chunk_map") or {}
                    if isinstance(fps, list):
                        filepaths = [str(x) for x in fps if isinstance(x, (str, int))]
                    if isinstance(dids, list):
                        doc_ids = [str(x) for x in dids if isinstance(x, (str, int))]
                    if isinstance(dc_map, dict):
                        for k, v in dc_map.items():
                            if not isinstance(v, list):
                                continue
                            key = str(k)
                            doc_chunk_map[key] = [str(x) for x in v if isinstance(x, (str, int))]
                except Exception as exc:  # pragma: no cover - 读取失败不影响主流程
                    logger.warning("读取 collection meta JSON 失败: %s", exc)

            collections.append(
                {
                    "name": name,
                    "db_file": str(db_file),
                    "size_bytes": size,
                    "filepaths": filepaths,
                    "doc_ids": doc_ids,
                    "doc_chunk_map": doc_chunk_map,
                }
            )
        return collections

    def inspect_collection(self, collection_name: str) -> dict:
        """
        查看单个 collection 的字段信息和示例数据，便于前端展示 schema / 动态元数据。
        """
        uri = self._db_path(collection_name)
        client = MilvusClient(uri=uri)

        # 1) schema 信息
        try:
            schema = client.describe_collection(collection_name)
        except Exception as exc:  # pragma: no cover - 容错
            logger.warning("describe_collection 失败: %s", exc)
            schema = None

        # 2) 抽一条样本行，看有哪些字段（含动态元数据）
        try:
            sample = client.query(
                collection_name=collection_name,
                limit=1,
                output_fields=["*"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("sample query 失败: %s", exc)
            sample = []

        all_fields: list[str] = []
        if sample:
            all_fields = list(sample[0].keys())

        # 根据 schema 动态推断「预定义字段」：所有 schema 中声明的字段名
        schema_field_names: list[str] = []
        if isinstance(schema, dict):
            fields = schema.get("fields") or []
            try:
                schema_field_names = [f.get("name") for f in fields if isinstance(f, dict) and "name" in f]
            except Exception:
                schema_field_names = []

        # 预定义字段 = schema 中的字段名；动态字段 = sample 行中多出来的字段
        predefined_fields = schema_field_names
        dynamic_fields = [f for f in all_fields if f not in schema_field_names]

        # 额外：从 meta JSON 中读取该 collection 级别的 filepaths / doc_ids / doc_chunk_map 信息
        meta_filepaths: list[str] = []
        meta_doc_ids: list[str] = []
        meta_doc_chunk_map: dict[str, list[str]] = {}
        try:
            meta_path = self._data_dir / f"{collection_name}_meta.json"
            if meta_path.exists():
                import json

                raw = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
                fps = raw.get("filepaths") or []
                dids = raw.get("doc_ids") or []
                dc_map = raw.get("doc_chunk_map") or {}
                if isinstance(fps, list):
                    meta_filepaths = [str(x) for x in fps if isinstance(x, (str, int))]
                if isinstance(dids, list):
                    meta_doc_ids = [str(x) for x in dids if isinstance(x, (str, int))]
                if isinstance(dc_map, dict):
                    for k, v in dc_map.items():
                        if not isinstance(v, list):
                            continue
                        key = str(k)
                        meta_doc_chunk_map[key] = [str(x) for x in v if isinstance(x, (str, int))]
        except Exception as exc:  # pragma: no cover - 容错，不影响主流程
            logger.warning("inspect_collection 读取 meta JSON 失败: %s", exc)

        return {
            "collection_name": collection_name,
            "uri": uri,
            "schema": schema,
            "predefined_fields": predefined_fields,
            "dynamic_fields": dynamic_fields,
            "filepaths": meta_filepaths,
            "doc_ids": meta_doc_ids,
            "doc_chunk_map": meta_doc_chunk_map,
        }

    def inspect_all_collections(self) -> list[dict]:
        """
        结合物理信息（.db 文件）和逻辑信息（schema + 动态字段）返回所有 collection 的详情。
        """
        details: list[dict] = []
        for item in self.list_collections():
            name = item["name"]
            try:
                info = self.inspect_collection(name)
                info["db_file"] = item["db_file"]
                info["size_bytes"] = item["size_bytes"]
                details.append(info)
            except Exception as exc:  # pragma: no cover
                logger.warning("inspect_all_collections: collection=%s 失败: %s", name, exc)
        return details

    def delete_collection(self, collection_name: str) -> None:
        db_file = Path(self._db_path(collection_name))
        if not db_file.exists():
            raise CollectionNotFoundException(
                f"Collection '{collection_name}' 不存在"
            )
        # 删除主 .db 文件
        try:
            db_file.unlink()
        except Exception as exc:
            logger.warning("删除 .db 文件失败 (%s): %s", db_file, exc)

        # 同时删除可能存在的锁文件（例如 test_chunks.db.lock）
        lock_file = db_file.with_suffix(db_file.suffix + ".lock")
        if lock_file.exists():
            try:
                lock_file.unlink()
            except Exception as exc:
                logger.warning("删除 .lock 文件失败 (%s): %s", lock_file, exc)

        # 删除对应的 meta JSON（如 <collection_name>_meta.json）
        meta_path = self._data_dir / f"{collection_name}_meta.json"
        if meta_path.exists():
            try:
                meta_path.unlink()
            except Exception as exc:
                logger.warning("删除 meta JSON 失败 (%s): %s", meta_path, exc)

        logger.info(
            "已删除 collection: %s（含 .db、.lock 及 meta JSON 文件，如存在）",
            collection_name,
        )

    def delete_by_metadata(
        self,
        collection_name: str,
        filepath: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> None:
        """
        按 metadata 条件删除部分向量（不可逆）。
        支持按 file_path / doc_id 批量删除。
        """
        if filepath is None and (not doc_ids):
            # 参数校验错误，应该返回 422，而不是 500
            raise ChunkingException("至少需要提供 filepath 或 doc_ids 之一用于删除条件")

        if not self.collection_exists(collection_name):
            raise CollectionNotFoundException(
                f"Collection '{collection_name}' 不存在"
            )

        # 构造向量存储（连接已有 collection）
        vector_store = self._build_vector_store(collection_name, overwrite=False)

        # 构造 metadata filters，与 search_with_metadata_filter 保持一致
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
                    key="file_path",
                    value=str(filepath),
                    operator=FilterOperator.EQ,
                )
            )
        if doc_ids:
            filter_list.append(
                MetadataFilter(
                    key="doc_id",
                    value=[str(d) for d in doc_ids],
                    operator=FilterOperator.IN,
                )
            )

        filters_obj = MetadataFilters(filters=filter_list, condition=FilterCondition.AND)

        logger.info(
            "按 metadata 删除向量: collection=%s, filepath=%s, doc_ids=%s",
            collection_name,
            filepath,
            doc_ids,
        )
        vector_store.delete_nodes(filters=filters_obj)

        # 同步更新 collection 对应的 meta JSON，将被删除的 filepath / doc_ids 从列表中移除
        with self._get_meta_lock(collection_name):
            try:
                meta_path = self._data_dir / f"{collection_name}_meta.json"
                if meta_path.exists():
                    import json

                    raw_meta = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
                    existing_filepaths: set[str] = set()
                    existing_doc_ids: set[str] = set()
                    existing_doc_chunk_map: dict[str, list[str]] = {}
                    existing_file_doc_map: dict[str, list[str]] = {}
                    for fp in raw_meta.get("filepaths", []):
                        if isinstance(fp, str) and fp:
                            existing_filepaths.add(fp)
                    for d in raw_meta.get("doc_ids", []):
                        if isinstance(d, str) and d:
                            existing_doc_ids.add(d)
                    dc_map = raw_meta.get("doc_chunk_map") or {}
                    if isinstance(dc_map, dict):
                        for k, v in dc_map.items():
                            if not isinstance(v, list):
                                continue
                            key = str(k)
                            existing_doc_chunk_map[key] = [str(x) for x in v if isinstance(x, (str, int))]
                    fd_map = raw_meta.get("file_doc_map") or {}
                    if isinstance(fd_map, dict):
                        for k, v in fd_map.items():
                            if not isinstance(v, list):
                                continue
                            key = str(k)
                            existing_file_doc_map[key] = [str(x) for x in v if isinstance(x, (str, int))]

                    # 删除本次操作对应的 filepath / doc_ids
                    if filepath is not None:
                        filepath_key = str(filepath)
                        existing_filepaths.discard(filepath_key)
                        related_doc_ids = existing_file_doc_map.pop(filepath_key, [])
                        for doc_key in related_doc_ids:
                            existing_doc_ids.discard(doc_key)
                            existing_doc_chunk_map.pop(doc_key, None)
                    if doc_ids:
                        for d in doc_ids:
                            doc_key = str(d)
                            existing_doc_ids.discard(doc_key)
                            existing_doc_chunk_map.pop(doc_key, None)
                            for fp_key in list(existing_file_doc_map.keys()):
                                docs = [x for x in existing_file_doc_map[fp_key] if x != doc_key]
                                if docs:
                                    existing_file_doc_map[fp_key] = docs
                                else:
                                    existing_file_doc_map.pop(fp_key, None)
                                    existing_filepaths.discard(fp_key)

                    new_meta = {
                        "collection_name": collection_name,
                        "filepaths": sorted(existing_filepaths),
                        "doc_ids": sorted(existing_doc_ids),
                        "doc_chunk_map": existing_doc_chunk_map,
                        "file_doc_map": existing_file_doc_map,
                    }
                    meta_path.write_text(
                        json.dumps(new_meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as exc:  # pragma: no cover - 元信息更新失败不影响主流程
                logger.warning("更新 collection meta JSON 失败（delete_by_metadata）: %s", exc)

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
                    nodes.append(self._build_text_node(text=text, metadata=metadata))
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

            # 包装嵌入模型并构建向量存储（锁保护全局 Settings 突变）
            with self._settings_lock:
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
            unique_filepaths: list[str] = []
            unique_doc_ids: list[str] = []
            doc_chunk_map: dict[str, list[str]] = {}
            file_doc_map: dict[str, list[str]] = {}
            _seen_filepaths: set[str] = set()
            _seen_doc_ids: set[str] = set()
            _seen_file_doc_pairs: set[tuple[str, str]] = set()
            _seen_doc_chunk_pairs: set[tuple[str, str]] = set()
            if metadatas:
                for md in metadatas:
                    if not isinstance(md, dict):
                        continue
                    fp = md.get("file_path")
                    if isinstance(fp, str) and fp:
                        if fp not in _seen_filepaths:
                            _seen_filepaths.add(fp)
                            unique_filepaths.append(fp)
                    doc = md.get("doc_id")
                    doc_str: Optional[str] = None
                    if doc is not None:
                        doc_str = str(doc)
                        if doc_str not in _seen_doc_ids:
                            _seen_doc_ids.add(doc_str)
                            unique_doc_ids.append(doc_str)
                    cid = md.get("chunk_id")
                    cid_str: Optional[str] = None
                    if cid is not None:
                        cid_str = str(cid)
                    if doc_str is not None and isinstance(fp, str) and fp:
                        pair = (fp, doc_str)
                        if pair not in _seen_file_doc_pairs:
                            _seen_file_doc_pairs.add(pair)
                            file_doc_map.setdefault(fp, []).append(doc_str)
                    if doc_str is not None and cid_str is not None:
                        pair = (doc_str, cid_str)
                        if pair not in _seen_doc_chunk_pairs:
                            _seen_doc_chunk_pairs.add(pair)
                            doc_chunk_map.setdefault(doc_str, []).append(cid_str)

            # 将本次构建涉及的 filepath / doc_ids 按 collection 维度写入一个 meta JSON，便于后续前端快速读取
            with self._get_meta_lock(collection_name):
                try:
                    meta_path = self._data_dir / f"{collection_name}_meta.json"
                    meta_data = {
                        "collection_name": collection_name,
                        "filepaths": unique_filepaths,
                        "doc_ids": unique_doc_ids,
                        "doc_chunk_map": doc_chunk_map,
                        "file_doc_map": file_doc_map,
                    }
                    import json
                    meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:  # pragma: no cover - 记录失败不影响主流程
                    logger.warning("写入 collection meta JSON 失败: %s", exc)

            return {
                "collection_name": collection_name,
                "total_chunks": len(chunks),
                "indexed_chunks": indexed,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
                "filepaths": unique_filepaths,
                "doc_ids": unique_doc_ids,
                "doc_chunk_map": doc_chunk_map,
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
                    nodes.append(self._build_text_node(text=text, metadata=metadata))
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

            with self._settings_lock:
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

            unique_filepaths: list[str] = []
            unique_doc_ids: list[str] = []
            doc_chunk_map: dict[str, list[str]] = {}
            file_doc_map: dict[str, list[str]] = {}
            _seen_filepaths: set[str] = set()
            _seen_doc_ids: set[str] = set()
            _seen_file_doc_pairs: set[tuple[str, str]] = set()
            _seen_doc_chunk_pairs: set[tuple[str, str]] = set()
            if metadatas:
                for md in metadatas:
                    if not isinstance(md, dict):
                        continue
                    fp = md.get("file_path")
                    if isinstance(fp, str) and fp:
                        if fp not in _seen_filepaths:
                            _seen_filepaths.add(fp)
                            unique_filepaths.append(fp)
                    doc = md.get("doc_id")
                    doc_str: Optional[str] = None
                    if doc is not None:
                        doc_str = str(doc)
                        if doc_str not in _seen_doc_ids:
                            _seen_doc_ids.add(doc_str)
                            unique_doc_ids.append(doc_str)
                    cid = md.get("chunk_id")
                    cid_str: Optional[str] = None
                    if cid is not None:
                        cid_str = str(cid)
                    if doc_str is not None and isinstance(fp, str) and fp:
                        pair = (fp, doc_str)
                        if pair not in _seen_file_doc_pairs:
                            _seen_file_doc_pairs.add(pair)
                            file_doc_map.setdefault(fp, []).append(doc_str)
                    if doc_str is not None and cid_str is not None:
                        pair = (doc_str, cid_str)
                        if pair not in _seen_doc_chunk_pairs:
                            _seen_doc_chunk_pairs.add(pair)
                            doc_chunk_map.setdefault(doc_str, []).append(cid_str)

            # 将本次追加涉及的 filepath / doc_ids 合并进 collection 的 meta JSON（若存在则增量合并）
            with self._get_meta_lock(collection_name):
                try:
                    meta_path = self._data_dir / f"{collection_name}_meta.json"
                    existing_filepaths: list[str] = []
                    existing_doc_ids: list[str] = []
                    existing_doc_chunk_map: dict[str, list[str]] = {}
                    existing_file_doc_map: dict[str, list[str]] = {}
                    _seen_existing_filepaths: set[str] = set()
                    _seen_existing_doc_ids: set[str] = set()
                    if meta_path.exists():
                        import json
                        raw_meta = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
                        for fp in raw_meta.get("filepaths", []):
                            if isinstance(fp, str) and fp:
                                if fp not in _seen_existing_filepaths:
                                    _seen_existing_filepaths.add(fp)
                                    existing_filepaths.append(fp)
                        for d in raw_meta.get("doc_ids", []):
                            if isinstance(d, str) and d:
                                if d not in _seen_existing_doc_ids:
                                    _seen_existing_doc_ids.add(d)
                                    existing_doc_ids.append(d)
                        raw_dc_map = raw_meta.get("doc_chunk_map") or {}
                        if isinstance(raw_dc_map, dict):
                            for k, v in raw_dc_map.items():
                                if not isinstance(v, list):
                                    continue
                                key = str(k)
                                vals = [str(x) for x in v if isinstance(x, (str, int))]
                                if vals:
                                    existing_doc_chunk_map[key] = vals
                        raw_fd_map = raw_meta.get("file_doc_map") or {}
                        if isinstance(raw_fd_map, dict):
                            for k, v in raw_fd_map.items():
                                if not isinstance(v, list):
                                    continue
                                key = str(k)
                                vals = [str(x) for x in v if isinstance(x, (str, int))]
                                if vals:
                                    existing_file_doc_map[key] = vals

                    merged_filepaths = existing_filepaths[:]
                    for fp in unique_filepaths:
                        if fp not in _seen_existing_filepaths:
                            _seen_existing_filepaths.add(fp)
                            merged_filepaths.append(fp)

                    merged_doc_ids = existing_doc_ids[:]
                    for d in unique_doc_ids:
                        if d not in _seen_existing_doc_ids:
                            _seen_existing_doc_ids.add(d)
                            merged_doc_ids.append(d)

                    merged_doc_chunk_map: dict[str, list[str]] = {
                        k: v[:] for k, v in existing_doc_chunk_map.items()
                    }
                    for d, cids in doc_chunk_map.items():
                        target = merged_doc_chunk_map.setdefault(d, [])
                        seen = set(target)
                        for c in cids:
                            if c not in seen:
                                seen.add(c)
                                target.append(c)

                    merged_file_doc_map: dict[str, list[str]] = {
                        k: v[:] for k, v in existing_file_doc_map.items()
                    }
                    for fp, dids in file_doc_map.items():
                        target = merged_file_doc_map.setdefault(fp, [])
                        seen = set(target)
                        for d in dids:
                            if d not in seen:
                                seen.add(d)
                                target.append(d)

                    new_meta = {
                        "collection_name": collection_name,
                        "filepaths": merged_filepaths,
                        "doc_ids": merged_doc_ids,
                        "doc_chunk_map": merged_doc_chunk_map,
                        "file_doc_map": merged_file_doc_map,
                    }
                    import json as _json
                    meta_path.write_text(_json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:  # pragma: no cover
                    logger.warning("更新 collection meta JSON 失败: %s", exc)

            return {
                "collection_name": collection_name,
                "added_chunks": added,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
                "filepaths": unique_filepaths,
                "doc_ids": unique_doc_ids,
                "doc_chunk_map": doc_chunk_map,
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
            # 仅做向量检索时，直接走 retriever，避免 query_engine.query 触发
            # response synthesizer（compact/refine）导致上下文窗口校验失败。
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundException(
                    f"Collection '{collection_name}' 不存在，请先构建索引"
                )

            with self._settings_lock:
                self._make_embed_model(langchain_embed)
                vector_store = self._build_vector_store_with_dim(
                    collection_name, embed_dim, overwrite=False
                )
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                vector_index = VectorStoreIndex([], storage_context=storage_context)

            effective_use_hybrid = (
                use_hybrid_search
                if use_hybrid_search is not None
                else self._settings.MILVUS_ENABLE_HYBRID_SEARCH
            )
            effective_use_hybrid = effective_use_hybrid and self._settings.MILVUS_ENABLE_SPARSE

            retriever_kwargs = {"similarity_top_k": top_k}
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
                        "filepath": meta.get("file_path"),
                        "doc_id": str(getattr(node, "ref_doc_id", None))
                        if getattr(node, "ref_doc_id", None) is not None
                        else None,
                        "chunk_id": str(meta.get("chunk_id"))
                        if meta.get("chunk_id") is not None
                        else None,
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

            # 设置嵌入模型 & 加载向量存储（锁保护全局 Settings 突变）
            with self._settings_lock:
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

            # filepath 支持：
            # - 单个字符串：EQ 过滤
            # - 字符串列表 / 元组 / 集合：IN 过滤
            if filepath is not None:
                if isinstance(filepath, (list, tuple, set)):
                    values = [str(p) for p in filepath]
                    op = FilterOperator.IN
                else:
                    values = str(filepath)
                    op = FilterOperator.EQ
                filter_list.append(
                    MetadataFilter(
                        key="file_path",
                        value=values,
                        operator=op,
                    )
                )

            # doc_id 同理，支持单个或多个
            if doc_id is not None:
                if isinstance(doc_id, (list, tuple, set)):
                    values = [str(d) for d in doc_id]
                    op = FilterOperator.IN
                else:
                    values = str(doc_id)
                    op = FilterOperator.EQ
                filter_list.append(
                    MetadataFilter(
                        key="doc_id",
                        value=values,
                        operator=op,
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
                        "filepath": meta.get("file_path"),
                        "doc_id": str(getattr(node, "ref_doc_id", None))
                        if getattr(node, "ref_doc_id", None) is not None
                        else None,
                        "chunk_id": str(meta.get("chunk_id"))
                        if meta.get("chunk_id") is not None
                        else None,
                    }
                )
            return results
        except CollectionNotFoundException:
            raise
        except Exception as exc:
            logger.exception("带过滤条件检索失败: %s", exc)
            raise RetrievalException(f"检索失败: {exc}") from exc
