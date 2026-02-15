"""
Milvus Lite 使用示例
展示如何使用 base_lite.py 进行离线向量检索
"""

import json
import asyncio
from base_lite import BaseRetrieverLite
from embeddings.base import HuggingfaceEmbeddings


def example_1_construct_index():
    """
    示例1：首次构建索引
    """
    print("=" * 60)
    print("示例1：构建 Milvus Lite 索引")
    print("=" * 60)
    
    # 1. 初始化嵌入模型
    embed_model = HuggingfaceEmbeddings(
        model_name='F:/thesis/models/bge-base-en-v1.5'
    )
    
    # 2. 创建检索器并构建索引
    retriever = BaseRetrieverLite(
        docs_directory=r'F:\thesis\Meta-Chunking\MoC\our_metrics\test_data\Qwen3-4B_0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt.json',  # 使用原始字符串
        embed_model=embed_model,
        embed_dim=768,  # bge-base-en-v1.5 是 768 维
        construct_index=True,  # 构建新索引
        collection_name='test_chunks',
        similarity_top_k=5,
        # milvus_data_dir='./milvus_data' 
    ) 
    # 3. 查看存储信息
    storage_info = retriever.get_storage_info()
    print(f"\n存储信息:")
    print(f"  - 文件路径: {storage_info['uri']}")
    print(f"  - 文件大小: {storage_info['size_mb']} MB")
    print(f"  - 是否存在: {storage_info['exists']}")
    
    # 4. 测试检索
    query = "What is quantum computing?"
    print(f"\n测试检索: {query}")
    results = retriever.search_docs(query)
    print(f"检索结果:\n{results}")
    print(f"\n检索到的文档数量: {len(results.source_nodes)}")
    for i, node_with_score in enumerate(results.source_nodes, 1):
        print(f"\n[文档 {i}]")
        print(f"  分数: {node_with_score.score}")
        print(f"  内容: {node_with_score.node.get_content()[:100]}...") # 只显示前100字符
        print(f"  元数据: {node_with_score.node.metadata}")

def example_2_load_existing_index():
    """
    示例2：加载已有索引（无需重新构建）
    """
    print("\n" + "=" * 60)
    print("示例2：加载已有 Milvus Lite 索引")
    print("=" * 60)
    
    # 1. 初始化嵌入模型（必须与构建时相同）
    embed_model = HuggingfaceEmbeddings(
        model_name='F:/thesis/models/bge-base-en-v1.5'
    )
    
    # 2. 创建检索器并构建索引
    retriever = BaseRetrieverLite(
        docs_directory=r'F:\thesis\Meta-Chunking\MoC\our_metrics\test_data\Qwen3-4B_0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt.json',  # 使用原始字符串
        embed_model=embed_model,
        embed_dim=768,  # bge-base-en-v1.5 是 768 维
        construct_index=False,  # 构建新索引
        collection_name='test_chunks',
        similarity_top_k=1,
        # milvus_data_dir='./milvus_data' 
    ) 
    
    print("✅ 索引加载成功！")
    
    # 3. 直接使用检索
    query = "Roman Smishko (Ukrainian: Роман Володимирович Смішко)"
    results = retriever.search_docs(query)
    print(f"\n检索结果:\n{results}")
    print(f"\n检索到的文档数量: {len(results.source_nodes)}")
    for i, node_with_score in enumerate(results.source_nodes, 1):
        print(f"\n[文档 {i}]")
        print(f"  分数: {node_with_score.score}")
        print(f"  内容: {node_with_score.node.get_content()[:100]}...") # 只显示前100字符
        print(f"  元数据: {node_with_score.node.metadata}")


def example_3_chinese_data():
    """
    示例3：处理中文数据
    """
    print("\n" + "=" * 60)
    print("示例3：处理中文数据")
    print("=" * 60)
    
    # 中文使用 bge-base-zh-v1.5
    embed_model = HuggingfaceEmbeddings(
        model_name='BAAI/bge-base-zh-v1.5'
    )
    
    retriever = BaseRetrieverLite(
        docs_directory='chinese_chunks.json',
        embed_model=embed_model,
        embed_dim=768,  # bge-base-zh-v1.5 的维度
        construct_index=True,
        collection_name='chinese_chunks',
        similarity_top_k=5,
        milvus_data_dir='./milvus_data'
    )
    
    # 注意：中文数据在 base_lite.py 的 construct_index() 中
    # 需要将过滤条件改为 len(i) < 10（按字符数）
    
    query = "什么是人工智能？"
    results = retriever.search_docs(query)
    print(f"\n检索结果:\n{results}")


def example_4_batch_queries():
    """
    示例4：批量查询
    """
    print("\n" + "=" * 60)
    print("示例4：批量查询")
    print("=" * 60)
    
    embed_model = HuggingfaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5'
    )
    
    retriever = BaseRetrieverLite(
        docs_directory='chunk_result.json',
        embed_model=embed_model,
        embed_dim=1024,
        construct_index=False,  # 使用已有索引
        collection_name='qasper_chunks',
        similarity_top_k=3,  # 每个查询返回3个结果
        milvus_data_dir='./milvus_data'
    )
    
    # 批量查询
    queries = [
        "What is artificial intelligence?",
        "How does neural network work?",
        "What are the applications of machine learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n查询 {i}: {query}")
        results = retriever.search_docs(query)
        print(f"结果: {results[:200]}...")  # 只显示前200个字符


def example_5_multiple_collections():
    """
    示例5：管理多个数据集
    """
    print("\n" + "=" * 60)
    print("示例5：管理多个数据集")
    print("=" * 60)
    
    embed_model = HuggingfaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5'
    )
    
    # 创建多个检索器，对应不同的数据集
    datasets = [
        ('qasper_chunks.json', 'qasper'),
        ('hotpotqa_chunks.json', 'hotpotqa'),
        ('2wikimqa_chunks.json', '2wikimqa')
    ]
    
    retrievers = {}
    for docs_path, name in datasets:
        print(f"\n处理数据集: {name}")
        retrievers[name] = BaseRetrieverLite(
            docs_directory=docs_path,
            embed_model=embed_model,
            embed_dim=1024,
            construct_index=True,  # 分别构建索引
            collection_name=name,
            similarity_top_k=5,
            milvus_data_dir='./milvus_data'  # 同一目录，不同文件
        )
        
        # 每个数据集会生成独立的 .db 文件
        # 例如: ./milvus_data/qasper.db
        #      ./milvus_data/hotpotqa.db
        #      ./milvus_data/2wikimqa.db
    
    # 使用不同的检索器
    query = "What is the capital of France?"
    
    print(f"\n查询: {query}")
    for name, retriever in retrievers.items():
        print(f"\n从 {name} 检索:")
        results = retriever.search_docs(query)
        print(f"  结果: {results[:150]}...")


def example_6_check_storage():
    """
    示例6：检查所有已创建的索引
    """
    print("\n" + "=" * 60)
    print("示例6：检查存储信息")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    milvus_data_dir = './milvus_data'
    
    if not os.path.exists(milvus_data_dir):
        print(f"目录不存在: {milvus_data_dir}")
        return
    
    # 列出所有 .db 文件
    db_files = list(Path(milvus_data_dir).glob('*.db'))
    
    if not db_files:
        print("未找到任何索引文件")
        return
    
    print(f"找到 {len(db_files)} 个索引文件:\n")
    
    total_size = 0
    for db_file in db_files:
        size_mb = db_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  📁 {db_file.name}")
        print(f"     - 大小: {size_mb:.2f} MB")
        print(f"     - 路径: {db_file}")
        print()
    
    print(f"总大小: {total_size:.2f} MB")


def example_7_inspect_data():
    """
    示例7：直接查看 Milvus 中的数据内容
    """
    print("\n" + "=" * 60)
    print("示例7：查看 Milvus 数据内容")
    print("=" * 60)
    
    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("请先安装 pymilvus: pip install pymilvus")
        return
    
    # 连接到本地 Docker Milvus
    client = MilvusClient(uri="http://localhost:19530")
    
    collection_name = 'test_chunks'
    
    if not client.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    # 获取集合统计信息
    # 注意：get_collection_stats 在新版 SDK 可能有变化，这里直接用 query count 或者 describe_collection
    desc = client.describe_collection(collection_name)
    print(f"集合名称: {desc['collection_name']}")
    
    # 查询前 3 条数据
    # filter expression is required for query in some versions, "" means all? No, usually "id >= 0" or similar if int64
    # For string IDs (LlamaIndex uses string IDs), we can use a limit with empty filter if supported, or iterator
    
    res = client.query(
        collection_name=collection_name,
        filter="", # 空字符串在 MilvusClient 中通常意味着无过滤
        output_fields=["*"],
        limit=3
    )
    
    print(f"\n前 3 条数据示例:")
    for i, item in enumerate(res, 1):
        print(f"\n[记录 {i}]")
        for k, v in item.items():
            if k == 'vector' or k == 'embedding': # LlamaIndex 默认字段可能是 embedding 或 vector
                # 向量太长，只显示维度
                dim = len(v)
                print(f"  {k}: <向量, 维度 {dim}>")
            elif isinstance(v, str) and len(v) > 100:
                # 文本太长，截断显示
                print(f"  {k}: {v[:100]}...")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    # 尝试应用 nest_asyncio 以允许嵌套事件循环
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("提示: 未安装 nest_asyncio，如果遇到事件循环错误，请尝试 pip install nest_asyncio")

    async def main():
        # 运行示例
        
        # 首次使用：构建索引
        # example_1_construct_index()
        
        # 后续使用：加载已有索引
        example_2_load_existing_index()
        
        # 中文数据处理
        # example_3_chinese_data()
        
        # 批量查询
        # example_4_batch_queries()
        
        # 多数据集管理
        # example_5_multiple_collections()
        
        # 检查存储
        # example_6_check_storage()

        # 查看数据内容
        # example_7_inspect_data()
        
        print("\n" + "=" * 60)
        print("✅ 示例完成！")
        print("=" * 60)

    # 在事件循环中运行
    asyncio.run(main())
