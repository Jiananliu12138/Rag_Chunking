"""
Chunk Stickiness 评估 - 重构版
评估文本块之间的黏连度（越低越好）
"""

import json
import math
import copy
import heapq
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class StickinessConfig:
    """黏连度评估配置"""
    model_path: str = 'model/internlm3-8b-instruct'
    input_file: str = 'relation_data/chunk_original.json'
    output_dir: str = 'relation_data_eval1'
    threshold: float = 0.8
    delta: float = 0.0
    device_map: str = "auto"
    log_level: str = 'INFO'


@dataclass
class StickinessResult:
    """黏连度评估结果"""
    structural_entropy_complete: float = 0.0
    structural_entropy_incomplete: float = 0.0
    graph_complete: Dict = field(default_factory=dict)
    graph_incomplete: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'structural_entropy_complete_graph': self.structural_entropy_complete,
            'structural_entropy_incomplete_graph': self.structural_entropy_incomplete
        }


class PerplexityCalculator:
    """困惑度计算器"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_ppl_for_next(self, context: str, target: str) -> Tuple[float, int]:
        """
        计算给定context预测target的困惑度
        返回: (总损失, target的token数)
        """
        context_tokens = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
        target_tokens = self.tokenizer(target, return_tensors="pt", add_special_tokens=False)
        
        input_ids = torch.cat([
            context_tokens["input_ids"].to(self.model.device),
            target_tokens["input_ids"].to(self.model.device)
        ], dim=-1)
        
        attention_mask = torch.cat([
            context_tokens["attention_mask"].to(self.model.device),
            target_tokens["attention_mask"].to(self.model.device)
        ], dim=-1)
        
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
        
        past_length = context_tokens["input_ids"].shape[1]
        shift_logits = response.logits[..., past_length-1:-1, :].contiguous()
        shift_labels = input_ids[..., past_length:].contiguous()
        
        active = (attention_mask[:, past_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        
        return loss.sum().item(), shift_labels.shape[1]


class GraphBuilder:
    """图构建器"""
    
    def __init__(self, ppl_calculator: PerplexityCalculator):
        self.ppl_calc = ppl_calculator
    
    def create_complete_graph(self, chunks: List[str]) -> Dict[int, Dict[int, float]]:
        """
        创建完全图 (Graph_1)
        节点i到节点j的边权重 = 给定chunks[i]预测chunks[j]的困惑度
        """
        n = len(chunks)
        graph = {i: {} for i in range(n)}
        
        # 计算每个节点的自环权重（无context预测自己）
        for i in range(n):
            sum_ppl, ppl_len = self.ppl_calc.get_ppl_for_next(' ', chunks[i])
            graph[i][i] = sum_ppl
        
        # 计算所有节点对之间的边权重
        for i in range(n):
            for j in range(n):
                if i != j:
                    sum_ppl, ppl_len = self.ppl_calc.get_ppl_for_next(chunks[i], chunks[j])
                    graph[i][j] = sum_ppl
        
        return graph
    
    def create_incomplete_graph(self, complete_graph: Dict) -> Dict[int, Dict[int, float]]:
        """
        创建不完全图 (Graph_2)
        只保留上三角矩阵（i <= j的边）
        """
        n = len(complete_graph)
        graph = {i: {} for i in range(n)}
        
        for i in range(n):
            for j in range(i, n):
                graph[i][j] = complete_graph[i][j]
        
        return graph
    
    def create_normalized_graph(
        self,
        complete_graph: Dict,
        token_nums: List[int],
        delta: float = 0.0
    ) -> Dict[int, Dict[int, float]]:
        """
        创建归一化图 (Graph_3)
        边权重考虑token长度归一化和位置距离惩罚
        """
        n = len(complete_graph)
        graph = {i: {} for i in range(n)}
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    graph[i][i] = 1.0
                else:
                    # 归一化权重计算
                    ppl_self = complete_graph[j][j] / token_nums[j]
                    ppl_given = complete_graph[i][j] / token_nums[j]
                    #块之间越相关，ppl_given越小，weight_temp越大
                    weight_temp = (math.exp(ppl_self) - math.exp(ppl_given)) / math.exp(ppl_self)
                    #weight_temp接近一，块之间相关
                    # 添加位置距离惩罚
                    position_penalty = delta * abs(i - j) / (n - 1) if n > 1 else 0
                    #块之间越相关，weight越小，距离越近weight越大，保留weight>0.8 即 edge < 0.2的边
                    weight = -weight_temp + 1 + position_penalty
                    
                    graph[i][j] = weight
        
        return graph


class StructuralEntropyCalculator:
    """结构熵计算器"""
    
    @staticmethod
    def build_degree_distribution(edges: List[Dict]) -> Dict[int, int]:
        """构建节点度分布"""
        degree = defaultdict(int)
        for edge in edges:
            degree[edge['row']] += 1
            degree[edge['column']] += 1
        return degree
    
    @staticmethod
    def calculate_entropy(degree_dist: Dict[int, int]) -> float:
        """计算结构熵"""
        total_degree = sum(degree_dist.values())
        if total_degree == 0:
            return 0.0
        
        entropy = 0.0
        for degree in degree_dist.values():
            if degree > 0:
                p = degree / total_degree
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def find_edges_above_threshold(
        graph: Dict[int, Dict[int, float]],
        threshold: float
    ) -> List[Dict]:
        """找出权重大于阈值的边"""
        edges = []
        for row_key, row_value in graph.items():
            for col_key, value in row_value.items():
                if value > threshold and row_key != col_key:
                    edges.append({
                        'row': row_key,
                        'column': col_key,
                        'value': value
                    })
        return edges


class StickinessEvaluator:
    """文本块黏连度评估器"""
    
    def __init__(self, config: StickinessConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        self.model = None
        self.tokenizer = None
        self.ppl_calc = None
        self.graph_builder = None
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('StickinessEvaluator')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        
        return logger
    
    def load_model(self):
        """加载模型"""
        self.logger.info(f"加载模型: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            device_map=self.config.device_map
        )
        self.model.eval()
        
        self.ppl_calc = PerplexityCalculator(self.model, self.tokenizer)
        self.graph_builder = GraphBuilder(self.ppl_calc)
        
        self.logger.info("✓ 模型加载完成")
    
    def evaluate_chunks(self, chunks: List[str]) -> StickinessResult:
        """评估文本块列表的黏连度"""
        n = len(chunks)
        self.logger.info(f"评估 {n} 个文本块的黏连度")
        
        # Step 1: 计算token数量
        token_nums = [
            self.tokenizer.encode(chunk, return_tensors='pt').shape[1]
            for chunk in chunks
        ]
        
        # Step 2: 构建完全图
        self.logger.info("构建完全图...")
        graph_complete = self.graph_builder.create_complete_graph(chunks)
        
        # Step 3: 构建归一化图
        self.logger.info("构建归一化图...")
        graph_normalized = self.graph_builder.create_normalized_graph(
            graph_complete,
            token_nums,
            self.config.delta
        )
        
        # Step 4: 计算完全图的结构熵
        edges_complete = StructuralEntropyCalculator.find_edges_above_threshold(
            graph_normalized,
            self.config.threshold
        )
        degree_dist_complete = StructuralEntropyCalculator.build_degree_distribution(edges_complete)
        entropy_complete = StructuralEntropyCalculator.calculate_entropy(degree_dist_complete)
        
        # Step 5: 构建不完全图并计算结构熵
        graph_incomplete = self.graph_builder.create_incomplete_graph(graph_normalized)
        edges_incomplete = StructuralEntropyCalculator.find_edges_above_threshold(
            graph_incomplete,
            self.config.threshold
        )
        degree_dist_incomplete = StructuralEntropyCalculator.build_degree_distribution(edges_incomplete)
        entropy_incomplete = StructuralEntropyCalculator.calculate_entropy(degree_dist_incomplete)
        
        result = StickinessResult(
            structural_entropy_complete=entropy_complete,
            structural_entropy_incomplete=entropy_incomplete,
            graph_complete=graph_normalized,
            graph_incomplete=graph_incomplete
        )
        
        self.logger.info(f"完全图结构熵: {entropy_complete:.4f}")
        self.logger.info(f"不完全图结构熵: {entropy_incomplete:.4f}")
        
        return result
    
    def evaluate_from_file(
        self,
        input_file: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[float, float]:
        """从文件读取并评估多个文档"""
        input_file = input_file or self.config.input_file
        output_dir = output_dir or self.config.output_dir
        
        # 读取输入
        self.logger.info(f"读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析输入格式
        if isinstance(data, list):
            # 格式1: 直接的文本块列表 ["块1", "块2", ...]
            if len(data) > 0 and isinstance(data[0], str):
                documents = [{'final_chunks': data}]
            # 格式2: 嵌套列表 [[chunk, label], ...]
            elif len(data) > 0 and isinstance(data[0], list):
                chunks = [item[0] for item in data]
                documents = [{'final_chunks': chunks}]
            # 格式3: 对象数组 [{"name": "...", "final_chunks": [...]}, ...]
            elif len(data) > 0 and isinstance(data[0], dict) and 'final_chunks' in data[0]:
                documents = data
            else:
                raise ValueError(f"不支持的列表格式")
        elif isinstance(data, dict):
            # 格式4: {"filepath": "...", "splits": [[text, label], ...], "time_cost": ...}
            if 'splits' in data:
                chunks = [split[0] for split in data['splits']]
                documents = [{'final_chunks': chunks, 'filepath': data.get('filepath', '')}]
                if 'time_cost' in data:
                    self.logger.info(f"分块耗时: {data['time_cost']:.2f}秒")
            # 格式5: {"name": "...", "final_chunks": [...]}
            elif 'final_chunks' in data:
                documents = [data]
            else:
                raise ValueError(f"字典格式缺少 'splits' 或 'final_chunks' 字段")
        else:
            raise ValueError(f"不支持的输入格式，期望 list 或 dict")
        
        self.logger.info(f"共 {len(documents)} 个文档")
        
        # 加载模型
        self.load_model()
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 评估所有文档
        entropy_complete_list = []
        entropy_incomplete_list = []
        
        for idx, doc in enumerate(tqdm(documents, desc="评估文档")):
            chunks = doc['final_chunks']
            
            result = self.evaluate_chunks(chunks)
            
            entropy_complete_list.append(result.structural_entropy_complete)
            entropy_incomplete_list.append(result.structural_entropy_incomplete)
            
            # 保存每个文档的图
            with open(f"{output_dir}/GC_doc{idx}.jsonl", 'a', encoding='utf-8') as f:
                json.dump(result.graph_complete, f, ensure_ascii=False)
                f.write('\n')
            
            with open(f"{output_dir}/GIC_doc{idx}.jsonl", 'a', encoding='utf-8') as f:
                json.dump(result.graph_incomplete, f, ensure_ascii=False)
                f.write('\n')
        
        # 计算平均值
        avg_complete = sum(entropy_complete_list) / len(entropy_complete_list)
        avg_incomplete = sum(entropy_incomplete_list) / len(entropy_incomplete_list)
        
        # 保存汇总结果
        summary = {
            'num_documents': len(documents),
            'avg_structural_entropy_complete': avg_complete,
            'avg_structural_entropy_incomplete': avg_incomplete,
            'individual_results': [
                {
                    'doc_id': i,
                    'entropy_complete': ec,
                    'entropy_incomplete': ei
                }
                for i, (ec, ei) in enumerate(zip(entropy_complete_list, entropy_incomplete_list))
            ]
        }
        
        summary_file = f"{output_dir}/summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        self.logger.info("="*60)
        self.logger.info("最终结果:")
        self.logger.info(f"  完全图平均结构熵: {avg_complete:.4f}")
        self.logger.info(f"  不完全图平均结构熵: {avg_incomplete:.4f}")
        self.logger.info(f"  结果已保存到: {summary_file}")
        self.logger.info("="*60)
        
        return avg_complete, avg_incomplete


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='文本块黏连度评估工具')
    parser.add_argument('--input', type=str, default='/data/h50056789/Rag_Chunking/MoC/our _metrics/test_data/test_star_structure.json')
    parser.add_argument('--output-dir', type=str, default='/data/h50056789/Rag_Chunking/MoC/our _metrics/test_data/eval_graph_dep.json')
    parser.add_argument('--model', type=str, default='/data/h50056789/Rag_chunk_bench/model/Qwen3-4B')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=0.0)
    
    args = parser.parse_args()
    
    config = StickinessConfig(
        model_path=args.model,
        input_file=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        delta=args.delta
    )
    
    evaluator = StickinessEvaluator(config)
    evaluator.evaluate_from_file()


if __name__ == '__main__':
    main()
