"""
评估指标:
1. Boundary Clarity (BC): 边界清晰度，基于困惑度比
2. Semantic Dissimilarity: 语义不相似度，基于embedding余弦相似度
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class EvaluatorConfig:
    """评估器配置类"""
    
    # 模型配置
    ppl_model_name: str = 'internlm3-8b-instruct'
    sim_model_name: str = 'BAAI/all-MiniLM-L6-v2'
    device_map: str = "auto"
    
    # 输入输出路径
    input_file: str = 'merge_data/db_qa_semantic_68.json'
    output_file: str = 'mergechunk_eval3/db_qa_semantic_68.json'
    
    # 评估选项
    enable_semantic_similarity: bool = True
    enable_boundary_clarity: bool = True
    
    # 日志配置
    log_level: str = 'INFO'
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """验证配置"""
        if not self.enable_semantic_similarity and not self.enable_boundary_clarity:
            raise ValueError("至少需要启用一个评估指标")


# ============================================================================
# 评估结果类
# ============================================================================

@dataclass
class EvaluationResult:
    """单对文本块的评估结果"""
    semantic_dissimilarity: float = 0.0
    boundary_clarity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'no_semantic_similarity': self.semantic_dissimilarity,
            'no_relative_perplexity': self.boundary_clarity
        }


@dataclass
class AggregatedResults:
    """聚合的评估结果"""
    semantic_dissimilarity_avg: float = 0.0
    boundary_clarity_avg: float = 0.0
    num_pairs: int = 0
    individual_results: List[EvaluationResult] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"评估结果摘要:\n"
            f"  文本块对数: {self.num_pairs}\n"
            f"  平均语义不相似度: {self.semantic_dissimilarity_avg:.4f}\n"
            f"  平均边界清晰度(BC): {self.boundary_clarity_avg:.4f}"
        )


# ============================================================================
# 核心评估器类
# ============================================================================

class ChunkEvaluator:
    """文本分块质量评估器"""
    
    def __init__(self, config: EvaluatorConfig):
        """
        初始化评估器
        
        Args:
            config: 评估器配置
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 延迟加载模型
        self.ppl_model = None
        self.ppl_tokenizer = None
        self.sim_model = None
        
        self.logger.info("评估器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('ChunkEvaluator')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s')
        )
        logger.addHandler(console_handler)
        
        # 文件处理器（可选）
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            )
            logger.addHandler(file_handler)
        
        return logger
    
    def load_models(self):
        """加载所需模型"""
        
        # 加载困惑度计算模型
        if self.config.enable_boundary_clarity:
            self.logger.info(f"加载困惑度模型: {self.config.ppl_model_name}")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(
                self.config.ppl_model_name,
                trust_remote_code=True
            )
            self.ppl_model = AutoModelForCausalLM.from_pretrained(
                self.config.ppl_model_name,
                trust_remote_code=True,
                device_map=self.config.device_map
            )
            self.ppl_model.eval()
            self.logger.info("✓ 困惑度模型加载完成")
        
        # 加载语义相似度模型
        if self.config.enable_semantic_similarity:
            self.logger.info(f"加载语义模型: {self.config.sim_model_name}")
            self.sim_model = SentenceTransformer(self.config.sim_model_name)
            self.logger.info("✓ 语义模型加载完成")
    
    def calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        计算两个文本块的语义相似度
        
        Args:
            text1: 第一个文本块
            text2: 第二个文本块
        
        Returns:
            余弦相似度 (0-1之间，越高越相似)
        """
        if self.sim_model is None:
            raise RuntimeError("语义模型未加载，请先调用 load_models()")
        
        # 编码文本
        embeddings = self.sim_model.encode(
            [text1, text2],
            normalize_embeddings=True
        )
        
        # 计算余弦相似度
        similarity = float(embeddings[0] @ embeddings[1].T)
        return similarity
    
    def calculate_perplexity(
        self,
        context: str,
        target: str
    ) -> float:
        """
        计算给定上下文时预测目标文本的困惑度
        
        Args:
            context: 上文（可以是空字符串表示无上文）
            target: 目标文本
        
        Returns:
            困惑度（交叉熵损失）
        """
        if self.ppl_model is None or self.ppl_tokenizer is None:
            raise RuntimeError("困惑度模型未加载，请先调用 load_models()")
        
        # Tokenize
        context_tokens = self.ppl_tokenizer(
            context, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        target_tokens = self.ppl_tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # 拼接输入
        input_ids = torch.cat([
            context_tokens["input_ids"].to(self.ppl_model.device),
            target_tokens["input_ids"].to(self.ppl_model.device)
        ], dim=-1)
        
        attention_mask = torch.cat([
            context_tokens["attention_mask"].to(self.ppl_model.device),
            target_tokens["attention_mask"].to(self.ppl_model.device)
        ], dim=-1)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.ppl_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
        
        # 计算困惑度
        context_length = context_tokens["input_ids"].shape[1]
        
        # 获取预测的 logits 和真实标签
        shift_logits = outputs.logits[..., context_length-1:-1, :].contiguous()
        shift_labels = input_ids[..., context_length:].contiguous()
        
        # 只计算有效位置的损失
        active = (attention_mask[:, context_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        
        # 计算交叉熵损失（即困惑度）
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        perplexity = loss_fn(active_logits, active_labels).item()
        
        return perplexity
    
    def calculate_boundary_clarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        计算边界清晰度 (Boundary Clarity)
        
        BC = ppl(text2 | text1) / ppl(text2)
        
        Args:
            text1: 前一个文本块
            text2: 后一个文本块
        
        Returns:
            BC值（越接近1表示边界越清晰）
        """
        # 计算无上文时的困惑度
        ppl_without_context = self.calculate_perplexity(' ', text2)
        
        # 计算有上文时的困惑度
        ppl_with_context = self.calculate_perplexity(text1, text2)
        
        # 计算BC
        bc = ppl_with_context / ppl_without_context if ppl_without_context > 0 else float('inf')
        
        return bc
    
    def evaluate_pair(
        self,
        text1: str,
        text2: str
    ) -> EvaluationResult:
        """
        评估一对相邻文本块
        
        Args:
            text1: 第一个文本块
            text2: 第二个文本块
        
        Returns:
            评估结果
        """
        result = EvaluationResult()
        
        # 计算语义不相似度
        if self.config.enable_semantic_similarity:
            similarity = self.calculate_semantic_similarity(text1, text2)
            result.semantic_dissimilarity = 1.0 - similarity
        
        # 计算边界清晰度
        if self.config.enable_boundary_clarity:
            result.boundary_clarity = self.calculate_boundary_clarity(text1, text2)
        
        return result
    
    def evaluate_chunks(
        self,
        chunks: List[str],
        show_progress: bool = True
    ) -> AggregatedResults:
        """
        评估文本块列表
        
        Args:
            chunks: 文本块列表
            show_progress: 是否显示进度条
        
        Returns:
            聚合的评估结果
        """
        if len(chunks) < 2:
            raise ValueError(f"至少需要2个文本块，当前只有 {len(chunks)} 个")
        
        self.logger.info(f"开始评估 {len(chunks)} 个文本块 ({len(chunks)-1} 对)")
        
        results = []
        sem_scores = []
        bc_scores = []
        
        # 遍历所有相邻对
        iterator = range(len(chunks) - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="评估进度")
        
        for i in iterator:
            try:
                text1 = chunks[i]
                text2 = chunks[i + 1]
                
                # 评估当前对
                result = self.evaluate_pair(text1, text2)
                results.append(result)
                
                if self.config.enable_semantic_similarity:
                    sem_scores.append(result.semantic_dissimilarity)
                if self.config.enable_boundary_clarity:
                    bc_scores.append(result.boundary_clarity)
                
            except Exception as e:
                self.logger.error(f"评估第 {i} 对时出错: {e}")
                continue
        
        # 计算平均值
        aggregated = AggregatedResults(
            semantic_dissimilarity_avg=sum(sem_scores) / len(sem_scores) if sem_scores else 0.0,
            boundary_clarity_avg=sum(bc_scores) / len(bc_scores) if bc_scores else 0.0,
            num_pairs=len(results),
            individual_results=results
        )
        
        self.logger.info("评估完成")
        self.logger.info(str(aggregated))
        
        return aggregated
    
    def evaluate_from_file(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> AggregatedResults:
        """
        从文件读取文本块并评估
        
        Args:
            input_file: 输入文件路径（JSON数组或包含splits的JSON对象）
            output_file: 输出文件路径
        
        Returns:
            聚合的评估结果
        """
        input_file = input_file or self.config.input_file
        output_file = output_file or self.config.output_file
        
        # 读取输入文件
        self.logger.info(f"读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析输入格式
        if isinstance(data, list):
            # 格式1: 直接的文本块列表 ["块1", "块2", ...]
            if len(data) > 0 and isinstance(data[0], str):
                chunks = data
            # 格式1b: 嵌套列表 [[chunk, label], ...]
            elif len(data) > 0 and isinstance(data[0], list):
                chunks = [item[0] for item in data]
            # 格式1c: 对象数组 [{"name": "...", "final_chunks": [...]}, ...]
            elif len(data) > 0 and isinstance(data[0], dict):
                if 'final_chunks' in data[0]:
                    # 只取第一个对象的chunks
                    chunks = data[0]['final_chunks']
                    if 'name' in data[0]:
                        self.logger.info(f"数据名称: {data[0]['name']}")
                else:
                    raise ValueError(f"对象缺少 'final_chunks' 字段")
            else:
                raise ValueError(f"不支持的列表格式")
        elif isinstance(data, dict):
            # 格式2: {"filepath": "...", "splits": [[text, label], ...], "time_cost": ...}
            if 'splits' in data:
                chunks = [split[0] for split in data['splits']]
                self.logger.info(f"文件路径: {data.get('filepath', 'N/A')}")
                if 'time_cost' in data:
                    self.logger.info(f"分块耗时: {data['time_cost']:.2f}秒")
            # 格式3: {"name": "...", "final_chunks": [...]}
            elif 'final_chunks' in data:
                chunks = data['final_chunks']
                if 'name' in data:
                    self.logger.info(f"数据名称: {data['name']}")
            else:
                raise ValueError(f"字典格式缺少 'splits' 或 'final_chunks' 字段")
        else:
            raise ValueError(f"不支持的输入格式，期望 list 或 dict")
        
        self.logger.info(f"成功读取 {len(chunks)} 个文本块")
        
        # 加载模型
        self.load_models()
        
        # 开始评估
        start_time = time.time()
        results = self.evaluate_chunks(chunks)
        elapsed = time.time() - start_time
        
        # 保存结果
        if output_file:
            self.logger.info(f"保存结果到: {output_file}")
            
            # 确保输出目录存在
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 构造完整的输出数据（包含平均值）
            output_data = {
                "summary": {
                    "num_pairs": results.num_pairs,
                    "avg_semantic_dissimilarity": results.semantic_dissimilarity_avg,
                    "avg_boundary_clarity": results.boundary_clarity_avg
                },
                "details": [r.to_dict() for r in results.individual_results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            self.logger.info("✓ 结果保存成功")
        
        self.logger.info(f"总执行时间: {elapsed:.2f} 秒")
        
        return results


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文本分块质量评估工具')
    parser.add_argument('--input', type=str, default='/data/h50056789/Rag_Chunking/MoC/our _metrics/test_data/Qwen3-4B_0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt.json',
                       help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='/data/h50056789/Rag_Chunking/MoC/our _metrics/test_data/eval.json',
                       help='输出JSON文件路径')
    parser.add_argument('--ppl-model', type=str, default='/data/h50056789/Rag_chunk_bench/model/Qwen3-4B', 
                       help='困惑度计算模型')
    parser.add_argument('--sim-model', type=str, default='/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5',
                       help='语义相似度模型')
    parser.add_argument('--log-file', type=str, default=None, help='/data/h50056789/Rag_Chunking/MoC/our _metrics/test_data.log')
    parser.add_argument('--disable-semantic', action='store_true', 
                       help='禁用语义相似度评估')
    parser.add_argument('--disable-bc', action='store_true',
                       help='禁用边界清晰度评估')
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvaluatorConfig(
        input_file=args.input,
        output_file=args.output,
        ppl_model_name=args.ppl_model,
        sim_model_name=args.sim_model,
        log_file=args.log_file,
        enable_semantic_similarity=not args.disable_semantic,
        enable_boundary_clarity=not args.disable_bc
    )
    
    # 创建评估器并运行
    evaluator = ChunkEvaluator(config)
    evaluator.evaluate_from_file()


if __name__ == '__main__':
    main()


# ============================================================================
# 使用示例
# ============================================================================

"""
命令行使用:
-----------
python chunk_eval_refactored.py \\
    --input data/chunks.json \\
    --output results/eval.json \\
    --log-file results/eval.log

代码使用:
---------
from chunk_eval_refactored import ChunkEvaluator, EvaluatorConfig

# 方式1: 使用配置类
config = EvaluatorConfig(
    input_file='data/chunks.json',
    output_file='results/eval.json'
)
evaluator = ChunkEvaluator(config)
results = evaluator.evaluate_from_file()

# 方式2: 直接评估列表
chunks = ["文本块1", "文本块2", "文本块3"]
evaluator.load_models()
results = evaluator.evaluate_chunks(chunks)
print(results)
"""
