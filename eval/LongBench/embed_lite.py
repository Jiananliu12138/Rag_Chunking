import os
import json
import logging
from base_lite import BaseRetrieverLite
from embeddings.base import HuggingfaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056787/workspaces/lightrag/tiktoken_cache"

class Config:
    EMBEDDING_NAME = '/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5'
    EMBEDDING_DIM = 1024
    DOCS_PATH = '/data/h50056789/Rag_Chunking/MoC/our_metrics/test_data/Qwen3-4B_0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt.json'
    COLLECTION_NAME = "test_chunks"
    MILVUS_DATA_DIR = '/data/h50056789/Rag_Chunking/milvus_data'

def main():
    logger.info("开始构建向量索引...")
    logger.info(f"文档路径: {Config.DOCS_PATH}")
    logger.info(f"Collection: {Config.COLLECTION_NAME}")

    embed_model = HuggingfaceEmbeddings(model_name=Config.EMBEDDING_NAME)
    print('[Milvus] 嵌入模型加载完成...')

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

    main()
