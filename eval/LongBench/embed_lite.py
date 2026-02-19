import os
import json
import asyncio
import logging
from base_lite import BaseRetrieverLite
from embeddings.base import HuggingfaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"

class Config:
    EMBEDDING_NAME = '/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5'
    EMBEDDING_DIM = 1024
    DOCS_PATH = '/data/h50056789/Rag_Chunking/Chunk_Result/Lumber_Chunk/2wikimqa_lumber_chunk_Qwen2.5-7B-Instruct.json'
    COLLECTION_NAME = "lumber_chunk"
    MILVUS_DATA_DIR = '/data/h50056789/Rag_Chunking/milvus_data'

async def main():
    logger.info("开始构建向量索引...")
    logger.info(f"文档路径: {Config.DOCS_PATH}")
    logger.info(f"Collection: {Config.COLLECTION_NAME}")

    embed_model = HuggingfaceEmbeddings(model_name=Config.EMBEDDING_NAME)
    print('[Milvus] 嵌入模型加载完成...')

    # BaseRetrieverLite(...) 入参/输入说明（按本文件实际传参）：
    # - docs_directory（Config.DOCS_PATH）: 分块结果文件路径（JSON 文件）。内容需可解析出 chunk 文本列表
    #   （解析规则见 base_lite.py 的 _parse_chunks_from_json()，支持多种 JSON 结构，核心是能拿到 chunk 文本）。
    # - embed_model: 已加载的嵌入模型实例（此处为 HuggingfaceEmbeddings）。
    # - embed_dim（Config.EMBEDDING_DIM）: 向量维度，必须与 embed_model 输出维度一致（如 1024）。
    # - construct_index: 是否新建并构建索引（True 表示从 docs_directory 构建）。
    # - add_index: 是否向已有索引追加数据（True 表示追加；此处为 False）。
    # - collection_name（Config.COLLECTION_NAME）: Milvus collection 名称。
    # - similarity_top_k: 检索 top-k（构建阶段通常只是保存该配置，供后续 query 使用）。
    # - milvus_data_dir（Config.MILVUS_DATA_DIR）: Milvus Lite 本地数据目录（.db 文件所在目录）。

    retriever = BaseRetrieverLite(
        docs_directory=Config.DOCS_PATH,
        embed_model=embed_model,
        embed_dim=Config.EMBEDDING_DIM,
        construct_index=True,
        add_index=False,
        collection_name=Config.COLLECTION_NAME,
        similarity_top_k=5,
        milvus_data_dir=Config.MILVUS_DATA_DIR
    )

    storage_info = retriever.get_storage_info()
    logger.info(f"Milvus 存储信息: {storage_info}")
    
    logger.info("✅ 索引构建完成！")

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    asyncio.run(main())
