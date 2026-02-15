"""
Milvus 服务器版本的检索器
连接到 Docker 启动的 Milvus 服务器 (localhost:19530)
"""

# ============================================================
# 标准库导入
# ============================================================
from abc import ABC
import os
import json
import asyncio
from pathlib import Path

# ============================================================
# LlamaIndex 导入
# ============================================================
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode as Node
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding

# ============================================================
# LangChain 导入
# ============================================================
from langchain_core.embeddings import Embeddings


# ============================================================
# 主类定义
# ============================================================
class BaseRetrieverLite(ABC):
    """
    基于 Milvus 服务器的检索器
    
    连接到 Docker 启动的 Milvus 服务器
    默认连接: localhost:19530
    
    支持多种 JSON 格式的分块数据输入
    """
    
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int = 2,
            host: str = "localhost",
            port: int = 19530,
            milvus_data_dir: str = None,
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.host = host
        self.port = port
        
        if milvus_data_dir:
            if not os.path.exists(milvus_data_dir):
                os.makedirs(milvus_data_dir)
            self.milvus_uri = os.path.join(milvus_data_dir, f"{collection_name}.db")
            print(f"[Milvus] 使用 Milvus Lite, 数据文件: {self.milvus_uri}")
        else:
            self.milvus_uri = f"http://{self.host}:{self.port}"
            print(f"[Milvus] 连接到服务器: {self.milvus_uri}")
        
        print(f"[Milvus] 连接到服务器: {self.milvus_uri}")

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()
        
        if add_index:
            self.add_index()

        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

        # 组装查询引擎
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def _parse_chunks_from_json(self, data):
        """
        从不同格式的JSON中提取文本块
        
        支持格式：
        1. ["chunk1", "chunk2", ...]
        2. [["chunk1", label1], ["chunk2", label2], ...]
        3. {"filepath": "...", "splits": [["chunk1", label1], ...], "time_cost": ...}
        4. {"final_chunks": ["chunk1", "chunk2", ...]}
        5. [{"name": "...", "final_chunks": ["chunk1", ...]}, ...]
        """
        chunks = []
        
        # 格式1: 简单列表 ["chunk1", "chunk2", ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            chunks = data
            print(f"[Milvus] 检测到格式: 简单列表")
        
        # 格式2: 列表嵌套 [["chunk1", label1], ["chunk2", label2], ...]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            chunks = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in data]
            print(f"[Milvus] 检测到格式: 列表嵌套（提取第一个元素）")
        
        # 格式3: 字典格式 {"filepath": "...", "splits": [...], ...}
        elif isinstance(data, dict):
            if "splits" in data:
                splits = data["splits"]
                if isinstance(splits, list) and len(splits) > 0:
                    if isinstance(splits[0], list):
                        # splits 是 [["chunk", label], ...]
                        chunks = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in splits]
                        print(f"[Milvus] 检测到格式: 字典+splits（列表嵌套）")
                    elif isinstance(splits[0], str):
                        # splits 是 ["chunk1", "chunk2", ...]
                        chunks = splits
                        print(f"[Milvus] 检测到格式: 字典+splits（简单列表）")
            
            # 格式4: {"final_chunks": ["chunk1", ...]}
            elif "final_chunks" in data:
                chunks = data["final_chunks"]
                print(f"[Milvus] 检测到格式: 字典+final_chunks")
        
        # 格式5: [{"name": "...", "final_chunks": [...]}, ...]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            for item in data:
                if "final_chunks" in item:
                    chunks.extend(item["final_chunks"])
            print(f"[Milvus] 检测到格式: 列表+字典+final_chunks")
        
        else:
            raise ValueError(f"[Milvus] 不支持的JSON格式: {type(data)}")
        
        return chunks

    def construct_index(self):
        """
        从 JSON 文件构建向量索引
        支持多种 JSON 格式
        """
        print(f"[Milvus] 开始构建索引: {self.collection_name}")
        
        # 读取分块数据
        folder_path = self.docs_directory  
        nodes = []
        
        with open(folder_path, 'r', encoding='utf-8') as file:  
            raw_data = json.load(file)
        
        # 解析不同格式的 JSON
        chunks = self._parse_chunks_from_json(raw_data)
        
        print(f"[Milvus] 读取到 {len(chunks)} 个文本块")
        
        # 打印前3个示例块
        print(f"\n{'='*60}")
        print(f"示例文本块（前3个）:")
        print(f"{'='*60}")
        for idx, chunk_text in enumerate(chunks[:3], 1):
            preview = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
            print(f"\n[示例 {idx}] (长度: {len(chunk_text)} 字符)")
            print(f"{preview}...")
            print(f"{'-'*60}")
        
        # 创建节点并过滤短文本
        for chunk_text in chunks:
            # 跳过非字符串类型
            if not isinstance(chunk_text, str):
                continue
            
            # 英语：按空格分词，汉语：len(chunk_text)<10
            if len(chunk_text.split(' ')) < 10:
                continue
            
            node1 = Node(text=chunk_text)
            nodes.append(node1)
        
        print(f"\n[Milvus] 过滤后剩余 {len(nodes)} 个有效文本块")
        
        # 包装嵌入模型
        self.embed_model = LangchainEmbedding(self.embed_model)
        
        # 使用 Settings 配置
        Settings.embed_model = self.embed_model
        Settings.llm = None  # 不使用 LLM
        
        # 创建 Milvus 向量存储（连接到 Docker 服务器）
        # 使用 http:// 格式的 URI 连接到服务器
        vector_store = MilvusVectorStore(
            uri=self.milvus_uri,
            dim=self.embed_dim,
            overwrite=True,  # 首次创建时覆盖
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 分批处理节点
        batch_size = 100  # 英语：100，汉语：1000
        total_batches = (len(nodes) + batch_size - 1) // batch_size
        
        print(f"[Milvus] 开始分批写入，共 {total_batches} 批")
        
        for spilt_ids in range(0, len(nodes), batch_size):
            batch_num = spilt_ids // batch_size + 1
            batch_nodes = nodes[spilt_ids:spilt_ids+batch_size]
            
            print(f"[Milvus] 处理第 {batch_num}/{total_batches} 批 ({len(batch_nodes)} 个节点)...")
            
            self.vector_index = VectorStoreIndex(
                batch_nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            print(f"[Milvus] 第 {batch_num} 批索引完成！")

            # 后续批次使用追加模式
            vector_store = MilvusVectorStore(
                uri=self.milvus_uri,
                overwrite=False,  # 不覆盖，追加数据
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"[Milvus] 索引构建完成！集合名称: {self.collection_name}")

    def add_index(self):
        """
        向已有索引中追加数据（可选功能）
        """
        print(f"[Milvus] 开始追加数据到索引: {self.collection_name}")
        
        if hasattr(self, 'docs_type') and self.docs_type == 'json':
            from llama_index.core import download_loader
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        from llama_index.core.node_parser import SimpleNodeParser
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        print(f"[Milvus] 解析得到 {len(nodes)} 个新节点")
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        
        # 使用 Settings 配置
        Settings.embed_model = self.embed_model
        Settings.llm = None
        
        # 追加模式
        vector_store = MilvusVectorStore(
            uri=self.milvus_uri,
            overwrite=False,  # 不覆盖
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 分批追加
        batch_size = 8000
        for spilt_ids in range(0, len(nodes), batch_size):
            self.vector_index = VectorStoreIndex(
                nodes[spilt_ids:spilt_ids+batch_size],
                storage_context=storage_context,
                show_progress=True
            )
            print(f"[Milvus] 追加第 {spilt_ids} 批完成！")

        print(f"[Milvus] 数据追加完成！")

    def load_index_from_milvus(self):
        """
        从 Milvus 服务器加载已有索引
        """
        print(f"[Milvus] 加载已有索引: {self.collection_name}")
        
        # 连接到 Milvus 服务器
        vector_store = MilvusVectorStore(
            uri=self.milvus_uri,
            overwrite=False,
            dim=self.embed_dim, 
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 包装嵌入模型
        if not isinstance(self.embed_model, LangchainEmbedding):
            self.embed_model = LangchainEmbedding(self.embed_model)
        
        # 使用 Settings 配置
        Settings.embed_model = self.embed_model
        Settings.llm = None
        
        # 创建空索引（数据已在 Milvus 中）
        self.vector_index = VectorStoreIndex(
            [],  # 空列表
            storage_context=storage_context,
        )
        
        print(f"[Milvus] 索引加载成功！")

    def search_docs(self, query_text: str):
        """
        检索与查询文本最相似的文档
        
        Args:
            query_text: 查询文本
            
        Returns:
            检索结果（包含相关文档）
        """
        response_vector = self.query_engine.query(query_text)
        return response_vector
    
    def get_collection_info(self):
        """
        获取集合信息（用于调试）
        """
        return {
            'host': self.host,
            'port': self.port,
            'collection_name': self.collection_name,
            'embed_dim': self.embed_dim
        }

    def get_storage_info(self):
        """
        获取存储信息（兼容性接口）
        """
        return {
            'uri': self.milvus_uri,
            'size_mb': 0,  # Milvus Server 模式下无法直接获取文件大小
            'exists': True
        }


# ============================================================
# 兼容性别名
# ============================================================
class BaseRetriever(BaseRetrieverLite):
    """
    兼容原有代码的别名
    """
    pass
