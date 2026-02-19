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
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding

from app.config import get_settings
from app.core.exceptions import (
    CollectionNotFoundException,
    IndexBuildException,
    RetrievalException,
)
from app.core.logging_config import logger


class MilvusRepository:
    """
    Milvus Lite 本地存储仓库。
    每个 collection 对应一个独立的 .db 文件，存储于 MILVUS_DATA_DIR。
    """

    def __init__(self, milvus_data_dir: Optional[str] = None):
        """
        Milvus 连接模式：
        - 如果显式传入 milvus_data_dir 或配置中的 MILVUS_DATA_DIR 非空：
            使用 Milvus Lite，本地 .db 文件，路径为 <data_dir>/<collection>.db
        - 否则如果环境变量 MILVUS_URI 存在：
            使用在线 Milvus / Milvus Server，uri 取该值
        - 否则：
            退回到默认的 Lite 目录 settings.MILVUS_DATA_DIR
        """
        settings = get_settings()

        # Lite 模式：优先使用参数或配置的 MILVUS_DATA_DIR
        effective_data_dir = milvus_data_dir or settings.MILVUS_DATA_DIR
        effective_data_dir = (effective_data_dir or "").strip()

        self._online_uri: Optional[str] = None
        if effective_data_dir:
            # 本地 Lite 模式
            self._data_dir = Path(effective_data_dir)
            self._data_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 尝试读取在线 Milvus URI
            uri = os.getenv("MILVUS_URI", "").strip()
            if uri:
                # 在线模式：后续构建 vector_store 时直接使用该 uri
                self._online_uri = uri
                # data_dir 仍然指向默认目录，用于 list_collections 等本地操作时兜底
                self._data_dir = Path(settings.MILVUS_DATA_DIR)
                self._data_dir.mkdir(parents=True, exist_ok=True)
                logger.info("[Milvus] 使用在线 Milvus URI: %s", uri)
            else:
                # 都没配时，退回默认 Lite 目录
                self._data_dir = Path(settings.MILVUS_DATA_DIR)
                self._data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "[Milvus] 未配置 MILVUS_DATA_DIR/MILVUS_URI，使用默认 Lite 目录: %s",
                    self._data_dir,
                )

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _db_path(self, collection_name: str) -> str:
        """根据模式返回 uri：
        - 在线模式：返回在线 URI（不拼接 collection 名）
        - Lite 模式：返回本地 .db 文件路径
        """
        if self._online_uri:
            return self._online_uri
        return str(self._data_dir / f"{collection_name}.db")

    def _make_embed_model(self, langchain_embed):
        """将 LangChain embedding 包装成 LlamaIndex 格式并注入全局 Settings。"""
        wrapped = LangchainEmbedding(langchain_embed)
        Settings.embed_model = wrapped
        Settings.llm = None
        # 尝试设置 tiktoken 本地缓存路径（优先使用环境变量）
        # 在 Windows（spawn）或无网络环境下可避免 tiktoken 自动下载失败
        try:
            import tiktoken  # noqa: F401
            cache_dir = os.getenv("TIKTOKEN_CACHE_DIR", "").strip()
            if cache_dir:
                os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        except ImportError:
            pass
        Settings.tokenizer = lambda text: text.split()
        return wrapped

    def _build_vector_store(self, collection_name: str, overwrite: bool) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=self._db_path(collection_name),
            collection_name=collection_name,
            overwrite=overwrite,
        )

    def _build_vector_store_with_dim(
        self, collection_name: str, dim: int, overwrite: bool
    ) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=self._db_path(collection_name),
            collection_name=collection_name,
            dim=dim,
            overwrite=overwrite,
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
    ) -> dict:
        """
        从文本块列表构建向量索引并持久化到 Milvus Lite .db 文件。

        逻辑结构参考 eval/LongBench/base_lite.py::construct_index：
        - 接收已经分好块的纯文本列表 `chunks`
        - 过滤无效/空白文本，构造节点
        - 打印若干示例块（通过日志）便于检查数据分布
        - 包装嵌入模型，构建 Milvus 向量存储
        - 按批次写入节点，并在日志中输出批次进度信息

        Returns:
            包含索引统计信息的字典。
        """
        try:
            start = time.time()

            # 1. 过滤并构造节点，类似 base_lite.construct_index 中的 Node 列表
            raw_count = len(chunks)
            nodes: list[TextNode] = []
            for text in chunks:
                if isinstance(text, str) and text.strip():
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

            # 2. 打印前若干个示例块，方便排查数据问题
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

            # 3. 包装嵌入模型并构建向量存储
            self._make_embed_model(langchain_embed)
            vector_store = self._build_vector_store_with_dim(
                collection_name, embed_dim, overwrite
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 4. 分批写入节点，仿照 base_lite 的批处理结构
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
            return {
                "collection_name": collection_name,
                "total_chunks": len(chunks),
                "indexed_chunks": indexed,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
            }
        except Exception as exc:
            logger.exception("索引构建失败: %s", exc)
            raise IndexBuildException(f"索引构建失败: {exc}") from exc

    def add_index(
        self,
        collection_name: str,
        chunks: list[str],
        langchain_embed,
        embed_dim: int,  # 目前仅用于与 build_index 接口保持一致，实际追加时由已有 collection 决定 dim
        batch_size: int = 8000,
    ) -> dict:
        """
        向已有索引中追加数据。

        参考 eval/LongBench/base_lite.py::add_index 的结构：
        - 接收新的文本块列表 `chunks`
        - 构造节点并过滤无效文本
        - 使用追加模式（overwrite=False）写入已有 collection
        - 分批追加，并在日志中输出进度
        """
        try:
            start = time.time()

            # 1. 过滤并构造节点
            raw_count = len(chunks)
            nodes: list[TextNode] = []
            for text in chunks:
                if isinstance(text, str) and text.strip():
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

            # 2. 包装嵌入模型
            self._make_embed_model(langchain_embed)

            # 3. 追加模式打开向量存储（overwrite 永远为 False）
            vector_store = self._build_vector_store(collection_name, overwrite=False)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 4. 分批追加
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
            return {
                "collection_name": collection_name,
                "added_chunks": added,
                "time_cost": elapsed,
                "milvus_uri": self._db_path(collection_name),
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
            retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k)
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
    ) -> list[dict]:
        """
        在指定 collection 中检索与 query 最相关的文档。

        Returns:
            [{"text": ..., "score": ...}, ...]
        """
        try:
            engine = self.load_query_engine(
                collection_name, langchain_embed, embed_dim, top_k
            )
            response = engine.query(query)
            results = []
            for node in response.source_nodes:
                results.append(
                    {
                        "text": node.node.get_content(),
                        "score": float(node.score) if node.score is not None else None,
                    }
                )
            return results
        except CollectionNotFoundException:
            raise
        except Exception as exc:
            logger.exception("检索失败: %s", exc)
            raise RetrievalException(f"检索失败: {exc}") from exc
