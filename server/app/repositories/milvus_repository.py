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
        settings = get_settings()
        self._data_dir = Path(milvus_data_dir or settings.MILVUS_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _db_path(self, collection_name: str) -> str:
        return str(self._data_dir / f"{collection_name}.db")

    def _make_embed_model(self, langchain_embed):
        """将 LangChain embedding 包装成 LlamaIndex 格式并注入全局 Settings。"""
        wrapped = LangchainEmbedding(langchain_embed)
        Settings.embed_model = wrapped
        Settings.llm = None
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

        Returns:
            包含索引统计信息的字典。
        """
        try:
            start = time.time()
            self._make_embed_model(langchain_embed)

            nodes = [TextNode(text=chunk) for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
            indexed = len(nodes)
            logger.info(
                "开始构建索引 collection=%s，有效块数=%d，批大小=%d",
                collection_name, indexed, batch_size,
            )

            vector_store = self._build_vector_store_with_dim(
                collection_name, embed_dim, overwrite
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            total_batches = (indexed + batch_size - 1) // batch_size
            for batch_idx in range(0, indexed, batch_size):
                batch_no = batch_idx // batch_size + 1
                batch = nodes[batch_idx: batch_idx + batch_size]
                logger.info("写入第 %d/%d 批（%d 个节点）...", batch_no, total_batches, len(batch))
                VectorStoreIndex(batch, storage_context=storage_context, show_progress=False)
                # 后续批次追加写入
                vector_store = self._build_vector_store(collection_name, overwrite=False)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

            elapsed = time.time() - start
            logger.info("索引构建完成，耗时 %.2fs", elapsed)
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
