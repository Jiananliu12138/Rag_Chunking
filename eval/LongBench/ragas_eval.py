#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 端到端评估器 (简化版)
仅执行评估，假设输入数据集已准备好
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.cache import DiskCacheBackend
from langchain_huggingface import HuggingFaceEmbeddings


def _is_nan(value):
    """检查值是否为 NaN"""
    try:
        import math
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


class RAGASEvaluator:
    """
    RAGAS 评估器
    使用现代 RAGAS API (0.2+) 直接调用 AsyncOpenAI 和 embedding_factory
    """
    
    def __init__(
        self,
        vllm_api_base: str = "http://localhost:8001/v1",
        vllm_api_key: str = "EMPTY",
        vllm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        embedding_model_path: str = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5",
        device: str = "cuda:0",
        enable_cache: bool = True,
        cache_dir: str = "./ragas_cache",
    ):
        """
        初始化 RAGAS 评估器
        
        Args:
            vllm_api_base: vLLM API 地址
            vllm_api_key: API Key（本地部署可以是任意值）
            vllm_model_name: 模型名称
            embedding_model_path: Embedding 模型路径
            device: GPU 设备
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
        """
        print(f"\n{'='*70}")
        print(f"初始化 RAGAS 评估器")
        print(f"{'='*70}")
        
        # 创建缓存（可选但强烈推荐）
        cache = DiskCacheBackend(cache_dir=cache_dir) if enable_cache else None
        
        # ========== LLM 配置 ==========
        # 创建 AsyncOpenAI 客户端（与 local_rag.py 第 419-421 行完全一致）
        self.openai_client = AsyncOpenAI(
            api_key=vllm_api_key,
            base_url=vllm_api_base,
        )
        # 使用 RAGAS 的 llm_factory 包装
        self.eval_llm = llm_factory(
            model=vllm_model_name,
            provider="openai",
            client=self.openai_client,
            cache=cache,
            temperature=0.0,
            max_tokens=2048,
        )
        print(f"\n✅ LLM 初始化完成:")
        print(f"  API Base: {vllm_api_base}")
        print(f"  模型: {vllm_model_name}")
        
        # ========== Embeddings 配置 ==========
        # 使用 LangChain HuggingFaceEmbeddings + RAGAS Wrapper
        print(f"\n初始化 Embeddings:")
        print(f"  模型: {embedding_model_path}")
        print(f"  设备: {device}")
        
        # 创建 LangChain HuggingFaceEmbeddings
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': device},
            encode_kwargs={'batch_size': 16, 'normalize_embeddings': True}
        )
        
        # 使用 RAGAS 的 LangchainEmbeddingsWrapper 包装
        self.eval_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        print("✅ Embeddings 初始化完成")
        
        if enable_cache:
            print(f"\n💾 缓存已启用: ./ragas_cache")
            print("   提示: 相同的文本/prompt 不会重复调用模型")
    
    def evaluate(
        self,
        dataset: Dict[str, List[Any]],
        metrics: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        执行 RAGAS 评估
        
        Args:
            dataset: 数据集字典，包含:
                - question: List[str] - 问题列表
                - answer: List[str] - 答案列表
                - contexts: List[List[str]] - 上下文列表（每个问题的上下文是一个字符串列表）
                - ground_truth: List[str] - 参考答案列表
            metrics: 评估指标列表，默认使用全部指标
        
        Returns:
            评估结果字典
        """
        # 验证数据集格式
        required_keys = ["question", "answer", "contexts", "ground_truth"]
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"数据集缺少必需字段: {key}")
            if not isinstance(dataset[key], list):
                raise ValueError(f"字段 {key} 必须是列表")
        
        # 验证数据集长度一致
        lengths = [len(dataset[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            raise ValueError(f"数据集各字段长度不一致: {dict(zip(required_keys, lengths))}")
        
        n_samples = lengths[0]
        print(f"\n{'='*70}")
        print(f"🚀 RAGAS 评估")
        print(f"{'='*70}")
        print(f"样本数: {n_samples}")
        print(f"评估指标: Faithfulness, Answer Relevancy, Context Recall, Context Precision, Context Entity Recall, Noise Sensitivity (Relevant), Noise Sensitivity (Irrelevant)")
        print(f"{'='*70}\n")
        
        # 创建 HuggingFace Dataset
        eval_dataset = Dataset.from_dict(dataset)
        
        # 默认使用全部指标
        if metrics is None:
            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextRecall(),
                ContextPrecision(),
                ContextEntityRecall(),
                NoiseSensitivity(mode="relevant"),    # 噪声敏感性（相关模式）
                NoiseSensitivity(mode="irrelevant"),  # 噪声敏感性（不相关模式）
            ]
        
        # 执行评估
        print("开始评估...")
        eval_results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
        )
        
        # 转换为 DataFrame
        df = eval_results.to_pandas()
        
        # 提取结果
        results = {
            "samples": [],
            "summary": {},
        }
        
        # 逐样本结果
        for idx, row in df.iterrows():
            sample_result = {
                "index": idx,
                "question": dataset["question"][idx],
                "answer": dataset["answer"][idx][:200] + "..." if len(dataset["answer"][idx]) > 200 else dataset["answer"][idx],
                "ground_truth": dataset["ground_truth"][idx][:200] + "..." if len(dataset["ground_truth"][idx]) > 200 else dataset["ground_truth"][idx],
                "contexts_count": len(dataset["contexts"][idx]),
                "metrics": {
                    "faithfulness": float(row.get("faithfulness", 0)) if not _is_nan(row.get("faithfulness", 0)) else 0.0,
                    "answer_relevancy": float(row.get("answer_relevancy", 0)) if not _is_nan(row.get("answer_relevancy", 0)) else 0.0,
                    "context_recall": float(row.get("context_recall", 0)) if not _is_nan(row.get("context_recall", 0)) else 0.0,
                    "context_precision": float(row.get("context_precision", 0)) if not _is_nan(row.get("context_precision", 0)) else 0.0,
                    "context_entity_recall": float(row.get("context_entity_recall", 0)) if not _is_nan(row.get("context_entity_recall", 0)) else 0.0,
                    "noise_sensitivity_relevant": float(row.get("noise_sensitivity(mode=relevant)", row.get("noise_sensitivity", 0))) if not _is_nan(row.get("noise_sensitivity(mode=relevant)", row.get("noise_sensitivity", 0))) else 0.0,
                    "noise_sensitivity_irrelevant": float(row.get("noise_sensitivity(mode=irrelevant)", 0)) if not _is_nan(row.get("noise_sensitivity(mode=irrelevant)", 0)) else 0.0,
                },
            }
            
            # 计算单个样本的平均分（不包含噪声敏感性，因为它是越低越好的指标）
            metrics_for_avg = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "context_entity_recall"]
            valid_metrics = [sample_result["metrics"][m] for m in metrics_for_avg if not _is_nan(sample_result["metrics"][m]) and sample_result["metrics"][m] > 0]
            sample_result["ragas_score"] = round(sum(valid_metrics) / len(valid_metrics), 4) if valid_metrics else 0.0
            
            results["samples"].append(sample_result)
        
        # 计算总体统计
        metrics_data = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_recall": [],
            "context_precision": [],
            "context_entity_recall": [],
            "noise_sensitivity_relevant": [],
            "noise_sensitivity_irrelevant": [],
            "ragas_score": [],
        }
        
        for sample in results["samples"]:
            # 正向指标（越高越好）
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "context_entity_recall"]:
                value = sample["metrics"][metric_name]
                if not _is_nan(value) and value > 0:
                    metrics_data[metric_name].append(value)
            
            # 反向指标（越低越好） - 包含 0 值
            for metric_name in ["noise_sensitivity_relevant", "noise_sensitivity_irrelevant"]:
                value = sample["metrics"][metric_name]
                if not _is_nan(value):
                    metrics_data[metric_name].append(value)
            
            ragas_score = sample["ragas_score"]
            if not _is_nan(ragas_score) and ragas_score > 0:
                metrics_data["ragas_score"].append(ragas_score)
        
        # 计算平均值
        for metric_name, values in metrics_data.items():
            if values:
                results["summary"][f"{metric_name}_mean"] = round(sum(values) / len(values), 4)
                results["summary"][f"{metric_name}_min"] = round(min(values), 4)
                results["summary"][f"{metric_name}_max"] = round(max(values), 4)
            else:
                results["summary"][f"{metric_name}_mean"] = 0.0
                results["summary"][f"{metric_name}_min"] = 0.0
                results["summary"][f"{metric_name}_max"] = 0.0
        
        # 打印总结
        print(f"\n{'='*70}")
        print(f"✅ 评估完成")
        print(f"{'='*70}")
        print(f"总体得分:")
        print(f"  RAGAS Score: {results['summary']['ragas_score_mean']:.4f}")
        print(f"  Faithfulness: {results['summary']['faithfulness_mean']:.4f}")
        print(f"  Answer Relevancy: {results['summary']['answer_relevancy_mean']:.4f}")
        print(f"  Context Recall: {results['summary']['context_recall_mean']:.4f}")
        print(f"  Context Precision: {results['summary']['context_precision_mean']:.4f}")
        print(f"  Context Entity Recall: {results['summary']['context_entity_recall_mean']:.4f}")
        print(f"  Noise Sensitivity (Relevant): {results['summary']['noise_sensitivity_relevant_mean']:.4f} (↓越低越好)")
        print(f"  Noise Sensitivity (Irrelevant): {results['summary']['noise_sensitivity_irrelevant_mean']:.4f} (↓越低越好)")
        print(f"{'='*70}\n")
        
        return results
    
    def evaluate_from_file(
        self,
        input_json_path: str,
        output_json_path: Optional[str] = None,
        metrics: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        从 JSON 文件读取数据集，执行评估，并保存结果到 JSON 文件
        
        Args:
            input_json_path: 输入 JSON 文件路径，支持两种格式:
                格式1 (标准 RAGAS 格式):
                {
                    "question": ["问题1", "问题2", ...],
                    "answer": ["答案1", "答案2", ...],
                    "contexts": [["上下文1-1", "上下文1-2"], ["上下文2-1"], ...],
                    "ground_truth": ["参考答案1", "参考答案2", ...]
                }
                
                格式2 (sample_results.json 格式):
                [
                    {
                        "_id": "q1",
                        "input": "问题1",
                        "llm_ans": "答案1",
                        "answers": ["参考答案1"],
                        "retrieval_list": ["上下文1-1", "上下文1-2"]
                    },
                    ...
                ]
                
            output_json_path: 输出 JSON 文件路径（可选，默认在输入文件同目录下生成）
            metrics: 评估指标列表，默认使用全部指标
        
        Returns:
            评估结果字典
        """
        # 读取输入数据集
        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_json_path}")
        
        print(f"\n{'='*70}")
        print(f"📂 从文件加载数据集")
        print(f"{'='*70}")
        print(f"输入文件: {input_json_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 检测并转换数据格式
        if isinstance(raw_data, list):
            # 格式2: sample_results.json 格式
            print("检测到格式: sample_results.json (列表格式)")
            dataset = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": []
            }
            
            for item in raw_data:
                dataset["question"].append(item["input"])
                dataset["answer"].append(item["llm_ans"])
                dataset["contexts"].append(item.get("retrieval_list", []))
                # 将多个参考答案合并为一个字符串
                dataset["ground_truth"].append(" ".join(item["answers"]))
            
            print(f"✅ 数据转换完成: {len(dataset['question'])} 个样本")
            
        elif isinstance(raw_data, dict):
            # 格式1: 标准 RAGAS 格式
            print("检测到格式: 标准 RAGAS 格式 (字典格式)")
            dataset = raw_data
        else:
            raise ValueError(f"不支持的数据格式: {type(raw_data)}")
        
        # 执行评估
        results = self.evaluate(dataset, metrics=metrics)
        
        # 确定输出路径
        if output_json_path is None:
            output_json_path = input_path.parent / f"{input_path.stem}_ragas_results.json"
        else:
            output_json_path = Path(output_json_path)
        
        # 创建输出目录
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        print(f"\n{'='*70}")
        print(f"💾 保存评估结果")
        print(f"{'='*70}")
        print(f"输出文件: {output_json_path}")
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评估结果已保存")
        print(f"{'='*70}\n")
        
        return results


if __name__ == "__main__":
    # 解决事件循环问题（RAGAS 使用异步调用）
    try:
        import nest_asyncio
        nest_asyncio.apply()
        print("✅ nest_asyncio 已应用（支持嵌套事件循环）")
    except ImportError:
        print("⚠️ nest_asyncio 未安装，如果遇到事件循环问题请安装: pip install nest_asyncio")
    
    # 配置参数
    INPUT_FILE = "/data/h50056789/Rag_Chunking/eval/LongBench/sample_results.json"
    OUTPUT_FILE = "/data/h50056789/Rag_Chunking/eval/LongBench/ragas_eval_results.json"
    VLLM_API = "http://localhost:8001/v1"
    EMBEDDING_MODEL = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
    DEVICE = "cuda:1"
    CACHE_DIR = "./ragas_cache"
    
    print("\n" + "="*70)
    print("RAGAS 评估器启动")
    print("="*70)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"vLLM API: {VLLM_API}")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print(f"设备: {DEVICE}")
    print("="*70 + "\n")
    
    # 初始化评估器
    evaluator = RAGASEvaluator(
        vllm_api_base=VLLM_API,
        embedding_model_path=EMBEDDING_MODEL,
        device=DEVICE,
        enable_cache=True,
        cache_dir=CACHE_DIR
    )
    
    # 从文件评估
    results = evaluator.evaluate_from_file(
        input_json_path=INPUT_FILE,
        output_json_path=OUTPUT_FILE
    )
    
    print("\n" + "="*70)
    print("✅ 评估完成!")
    print("="*70)
