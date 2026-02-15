"""
RAG评估系统 - 重构版本
支持多种LLM、检索器和评估任务的模块化评估框架
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from loguru import logger

from src.datasets.xinhua import get_task_datasets
from evaluator import BaseEvaluator
from src.llms import GPT, Qwen_7B_Chat
from src.tasks.summary import Summary
from src.tasks.continue_writing import ContinueWriting
from src.tasks.hallucinated_modified import HalluModified
from src.tasks.quest_answer import QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs
from src.retrievers import BaseRetriever, CustomBM25Retriever, EnsembleRetriever, EnsembleRerankRetriever
from src.embeddings.base import HuggingfaceEmbeddings


@dataclass
class ModelConfig:
    """LLM模型配置"""
    model_name: str = 'qwen7b'
    temperature: float = 0.1
    max_new_tokens: int = 1280


@dataclass
class DatasetConfig:
    """数据集配置"""
    data_path: str = 'data/crud_split/split_merged.json'
    shuffle: bool = True


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    embedding_name: str = 'sentence-transformers/bge-base-zh-v1.5'
    embedding_dim: int = 768


@dataclass
class IndexConfig:
    """索引配置"""
    docs_path: str = 'data/tmp'
    docs_type: str = 'txt'
    chunk_size: int = 128
    chunk_overlap: int = 0
    construct_index: bool = False
    add_index: bool = False
    collection_name: str = 'docs_80k_chuncksize_128_0'


@dataclass
class RetrieverConfig:
    """检索器配置"""
    retriever_name: str = 'base'
    retrieve_top_k: int = 8


@dataclass
class MetricConfig:
    """评估指标配置"""
    quest_eval: bool = False
    bert_score_eval: bool = False


@dataclass
class EvaluationConfig:
    """评估配置"""
    task: str = 'event_summary'
    num_threads: int = 1
    show_progress_bar: bool = True
    contain_original_data: bool = False


@dataclass
class RAGEvaluatorConfig:
    """RAG评估器总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'RAGEvaluatorConfig':
        """从命令行参数创建配置"""
        return cls(
            model=ModelConfig(
                model_name=args.model_name,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens
            ),
            dataset=DatasetConfig(
                data_path=args.data_path,
                shuffle=args.shuffle
            ),
            embedding=EmbeddingConfig(
                embedding_name=args.embedding_name,
                embedding_dim=args.embedding_dim
            ),
            index=IndexConfig(
                docs_path=args.docs_path,
                docs_type=args.docs_type,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                construct_index=args.construct_index,
                add_index=args.add_index,
                collection_name=args.collection_name
            ),
            retriever=RetrieverConfig(
                retriever_name=args.retriever_name,
                retrieve_top_k=args.retrieve_top_k
            ),
            metric=MetricConfig(
                quest_eval=args.quest_eval,
                bert_score_eval=args.bert_score_eval
            ),
            evaluation=EvaluationConfig(
                task=args.task,
                num_threads=args.num_threads,
                show_progress_bar=args.show_progress_bar,
                contain_original_data=args.contain_original_data
            )
        )


class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_llm(config: ModelConfig):
        """根据配置创建LLM实例"""
        if config.model_name.startswith("gpt"):
            logger.info(f"创建GPT模型: {config.model_name}")
            return GPT(
                model_name=config.model_name,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens
            )
        elif config.model_name == "qwen7b":
            logger.info(f"创建Qwen模型: {config.model_name}")
            return Qwen_7B_Chat(
                model_name=config.model_name,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens
            )
        else:
            raise ValueError(f"不支持的模型类型: {config.model_name}")


class RetrieverFactory:
    """检索器工厂类"""
    
    @staticmethod
    def create_retriever(
        retriever_config: RetrieverConfig,
        index_config: IndexConfig,
        embed_model: Any
    ):
        """根据配置创建检索器实例"""
        retriever_name = retriever_config.retriever_name
        
        common_params = {
            'embed_model': embed_model,
            'chunk_size': index_config.chunk_size,
            'chunk_overlap': index_config.chunk_overlap,
            'similarity_top_k': retriever_config.retrieve_top_k,
            'construct_index': index_config.construct_index
        }
        
        if retriever_name == "base":
            logger.info("创建基础向量检索器")
            return BaseRetriever(
                index_config.docs_path,
                embed_dim=index_config.embedding_dim,
                add_index=index_config.add_index,
                collection_name=index_config.collection_name,
                **common_params
            )
        
        elif retriever_name == "bm25":
            logger.info("创建BM25检索器")
            return CustomBM25Retriever(
                index_config.docs_path,
                **common_params
            )
        
        elif retriever_name == "hybrid":
            logger.info("创建混合检索器")
            return EnsembleRetriever(
                index_config.docs_path,
                embed_dim=index_config.embedding_dim,
                add_index=index_config.add_index,
                collection_name=index_config.collection_name,
                **common_params
            )
        
        elif retriever_name == "hybrid-rerank":
            logger.info("创建混合重排序检索器")
            return EnsembleRerankRetriever(
                index_config.docs_path,
                embed_dim=index_config.embedding_dim,
                add_index=index_config.add_index,
                collection_name=index_config.collection_name,
                **common_params
            )
        
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_name}")


class TaskManager:
    """任务管理器"""
    
    TASK_MAPPING = {
        'event_summary': [Summary],
        'continuing_writing': [ContinueWriting],
        'hallu_modified': [HalluModified],
        'quest_answer': [QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs],
        'all': [Summary, ContinueWriting, HalluModified]
    }
    
    @classmethod
    def get_task_classes(cls, task_name: str) -> List:
        """获取任务类列表"""
        if task_name not in cls.TASK_MAPPING:
            raise ValueError(
                f"不支持的任务: {task_name}. "
                f"可用任务: {list(cls.TASK_MAPPING.keys())}"
            )
        return cls.TASK_MAPPING[task_name]
    
    @classmethod
    def create_tasks(cls, task_name: str, metric_config: MetricConfig) -> List:
        """创建任务实例列表"""
        task_classes = cls.get_task_classes(task_name)
        tasks = [
            task_cls(
                use_quest_eval=metric_config.quest_eval,
                use_bert_score=metric_config.bert_score_eval
            )
            for task_cls in task_classes
        ]
        logger.info(f"创建了 {len(tasks)} 个任务实例: {task_name}")
        return tasks


class RAGEvaluationRunner:
    """RAG评估执行器"""
    
    def __init__(self, config: RAGEvaluatorConfig):
        self.config = config
        self.llm = None
        self.embed_model = None
        self.retriever = None
        self.tasks = None
        self.datasets = None
        
    def setup(self):
        """初始化所有组件"""
        logger.info("=" * 60)
        logger.info("开始初始化RAG评估系统")
        logger.info("=" * 60)
        
        # 创建LLM
        self.llm = LLMFactory.create_llm(self.config.model)
        
        # 创建嵌入模型
        logger.info(f"加载嵌入模型: {self.config.embedding.embedding_name}")
        self.embed_model = HuggingfaceEmbeddings(
            model_name=self.config.embedding.embedding_name
        )
        
        # 创建检索器
        self.retriever = RetrieverFactory.create_retriever(
            self.config.retriever,
            self.config.index,
            self.embed_model
        )
        
        # 创建任务
        self.tasks = TaskManager.create_tasks(
            self.config.evaluation.task,
            self.config.metric
        )
        
        # 加载数据集
        logger.info(f"加载数据集: {self.config.dataset.data_path}")
        self.datasets = get_task_datasets(
            self.config.dataset.data_path,
            self.config.evaluation.task
        )
        
        logger.info("=" * 60)
        logger.info("初始化完成，准备开始评估")
        logger.info("=" * 60)
        
    def run(self):
        """执行评估"""
        if self.tasks is None or self.datasets is None:
            raise RuntimeError("请先调用 setup() 方法初始化组件")
        
        logger.info(f"开始评估任务: {self.config.evaluation.task}")
        logger.info(f"任务数量: {len(self.tasks)}")
        
        for idx, (task, dataset) in enumerate(zip(self.tasks, self.datasets), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"执行任务 {idx}/{len(self.tasks)}: {task.__class__.__name__}")
            logger.info(f"{'='*60}")
            
            evaluator = BaseEvaluator(
                task=task,
                llm=self.llm,
                retriever=self.retriever,
                dataset=dataset,
                num_threads=self.config.evaluation.num_threads
            )
            
            evaluator.run(
                show_progress_bar=self.config.evaluation.show_progress_bar,
                contain_original_data=self.config.evaluation.contain_original_data
            )
            
            logger.info(f"任务 {idx} 完成")
        
        logger.info(f"\n{'='*60}")
        logger.info("所有评估任务完成！")
        logger.info(f"{'='*60}")


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='RAG系统评估工具 - 重构版',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型相关参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument(
        '--model_name', 
        default='qwen7b',
        help='LLM模型名称 (支持: qwen7b, gpt-*)'
    )
    model_group.add_argument(
        '--temperature', 
        type=float, 
        default=0.1,
        help='生成温度，控制随机性'
    )
    model_group.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=1280,
        help='最大生成token数'
    )
    
    # 数据集相关参数
    dataset_group = parser.add_argument_group('数据集配置')
    dataset_group.add_argument(
        '--data_path', 
        default='data/crud_split/split_merged.json',
        help='数据集路径'
    )
    dataset_group.add_argument(
        '--shuffle', 
        type=bool, 
        default=True,
        help='是否打乱数据集'
    )
    
    # 嵌入模型相关参数
    embed_group = parser.add_argument_group('嵌入模型配置')
    embed_group.add_argument(
        '--embedding_name',
        default='sentence-transformers/bge-base-zh-v1.5',
        help='嵌入模型名称'
    )
    embed_group.add_argument(
        '--embedding_dim', 
        type=int, 
        default=768,
        help='嵌入向量维度'
    )
    
    # 索引相关参数
    index_group = parser.add_argument_group('索引配置')
    index_group.add_argument(
        '--docs_path', 
        default='data/tmp',
        help='文档路径'
    )
    index_group.add_argument(
        '--docs_type', 
        default='txt',
        help='文档类型'
    )
    index_group.add_argument(
        '--chunk_size', 
        type=int, 
        default=128,
        help='分块大小（token数）'
    )
    index_group.add_argument(
        '--chunk_overlap', 
        type=int, 
        default=0,
        help='分块重叠大小'
    )
    index_group.add_argument(
        '--construct_index', 
        action='store_true',
        help='是否构建索引'
    )
    index_group.add_argument(
        '--add_index', 
        action='store_true',
        help='是否添加索引'
    )
    index_group.add_argument(
        '--collection_name',
        default='docs_80k_chuncksize_128_0',
        help='向量库集合名称'
    )
    
    # 检索器相关参数
    retriever_group = parser.add_argument_group('检索器配置')
    retriever_group.add_argument(
        '--retrieve_top_k', 
        type=int, 
        default=8,
        help='检索top-k个文档'
    )
    retriever_group.add_argument(
        '--retriever_name',
        default='base',
        choices=['base', 'bm25', 'hybrid', 'hybrid-rerank'],
        help='检索器类型'
    )
    
    # 评估指标相关参数
    metric_group = parser.add_argument_group('评估指标配置')
    metric_group.add_argument(
        '--quest_eval', 
        action='store_true',
        help='是否使用RAGQuestEval指标'
    )
    metric_group.add_argument(
        '--bert_score_eval', 
        action='store_true',
        help='是否使用BERTScore指标'
    )
    
    # 评估相关参数
    eval_group = parser.add_argument_group('评估配置')
    eval_group.add_argument(
        '--task',
        default='event_summary',
        choices=['event_summary', 'continuing_writing', 'hallu_modified', 'quest_answer', 'all'],
        help='评估任务类型'
    )
    eval_group.add_argument(
        '--num_threads', 
        type=int, 
        default=1,
        help='并行线程数'
    )
    eval_group.add_argument(
        '--show_progress_bar',
        type=bool, 
        default=True,
        help='是否显示进度条'
    )
    eval_group.add_argument(
        '--contain_original_data', 
        action='store_true',
        help='是否包含原始数据'
    )
    
    return parser


def main():
    """主函数"""
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 创建配置
    config = RAGEvaluatorConfig.from_args(args)
    
    # 打印配置
    logger.info("配置信息:")
    logger.info(f"模型: {config.model.model_name}")
    logger.info(f"检索器: {config.retriever.retriever_name}")
    logger.info(f"任务: {config.evaluation.task}")
    logger.info(f"数据集: {config.dataset.data_path}")
    
    # 创建并运行评估器
    runner = RAGEvaluationRunner(config)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
