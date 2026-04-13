#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAGAS 端到端评估器 (简化版)
仅执行评估，假设输入数据集已准备好
"""

import json
import logging
import os
import sys
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
from ragas.cache import DiskCacheBackend
from ragas.run_config import RunConfig

from public_method.models.factory import get_ragas_embeddings

_LOGGER = logging.getLogger("ragas_eval")


def _ensure_logger_visible(logger: logging.Logger) -> None:
    """确保 ``logger`` 的 INFO 至少能落到一个 handler。

    场景区分：
    - 在 server 进程里跑：root logger 已经被 ``setup_logging`` 配过 stdout
      handler，这里向上找祖先就会命中，什么都不做。
    - 直接 ``python eval_ragas.py`` 或在未初始化 logging 的脚本里跑：root
      默认 WARNING + 无 handler，``_LOGGER.info`` 会被吞掉。这种情况给
      ``logger`` 自己挂一个 stdout StreamHandler，并把 propagate 关掉避
      免双倍输出。
    """
    cur: Optional[logging.Logger] = logger
    has_handler = False
    while cur is not None:
        if cur.handlers:
            has_handler = True
            break
        if not cur.propagate:
            break
        cur = cur.parent

    if not has_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False

    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


def _is_nan(value):
    """检查值是否为 NaN"""
    try:
        import math
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _coerce_metric_value(value: Any) -> Optional[float]:
    """Normalize metric values while preserving real zeroes."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if _is_nan(number) or number in (float("inf"), float("-inf")):
        return None

    return number


class _LoggerProgressBar:
    """伪装成 tqdm 的进度对象，把 update() 转成 logger 输出。

    用途：注入到 ``ragas.evaluate(_pbar=...)``，让 RAGAS 内部 Executor
    在每个 (metric, sample) 任务完成时调用 ``self.update(1)``。在没有 tty
    的 server 进程里 tqdm 写到 stderr 是看不到的，这里改成走 logger 走
    stdout，每 ~5% 打一行 ``[RAGAS] 进度 X/total``，和 CS 的格式对齐。
    """

    def __init__(
        self,
        total: int,
        logger: logging.Logger,
        label: str = "RAGAS",
        log_every: Optional[int] = None,
        progress_file_path: Optional[str] = None,
    ) -> None:
        self.total = max(int(total), 0)
        self.n = 0
        self._logger = logger
        self._label = label
        self._log_every = int(log_every) if log_every else max(1, self.total // 20)
        self._file = None
        if progress_file_path:
            Path(progress_file_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(progress_file_path, "w", encoding="utf-8")
        _ensure_logger_visible(self._logger)
        self._log(f"{self._label} 启动: total={self.total}")

    def _log(self, msg: str) -> None:
        self._logger.info(msg)
        if self._file:
            self._file.write(msg + "\n")
            self._file.flush()

    def update(self, n: int = 1) -> None:
        if n <= 0:
            return
        self.n += int(n)
        if self.total and (self.n >= self.total or self.n % self._log_every == 0):
            done = min(self.n, self.total)
            pct = done * 100 // self.total
            self._log(f"{self._label} 进度 {done}/{self.total} ({pct}%)")

    def close(self) -> None:
        if self.total and self.n < self.total:
            self._log(f"{self._label} 收尾 {self.n}/{self.total}")
        if self._file:
            self._file.close()
            self._file = None

    def refresh(self) -> None:  # noqa: D401 - tqdm 兼容方法
        pass

    def reset(self, total: Optional[int] = None) -> None:
        if total is not None:
            self.total = int(total)
        self.n = 0

    def set_description(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def set_postfix(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def __enter__(self) -> "_LoggerProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RAGASEvaluator:
    """
    RAGAS 评估器
    使用现代 RAGAS API (0.2+) 直接调用 AsyncOpenAI 和 embedding_factory
    """
    
    def __init__(
        self,
        vllm_api_base: str = "http://localhost:8005/v1",
        vllm_api_key: str = "EMPTY",
        vllm_model_name: str = "/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct",
        embedding_model_path: str = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5",
        eval_embeddings: Optional[Any] = None,
        device: str = "cuda:0",
        enable_cache: bool = True,
        cache_dir: str = "./ragas_cache",
        max_workers: int = 16,
        request_timeout: int = 600,
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
        self.cache_dir = cache_dir

        print(f"\n{'='*70}")
        print(f"初始化 RAGAS 评估器")
        print(f"{'='*70}")

        # RunConfig：并发度 + 单次 LLM/embedding 调用超时（防止 vLLM 排队时被早早 timeout）
        self.run_config = RunConfig(
            max_workers=int(max_workers),
            timeout=int(request_timeout),
        )
        print(f"  并发: max_workers={max_workers}, timeout={request_timeout}s")

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
        
        self.eval_embeddings = eval_embeddings or get_ragas_embeddings(
            model_path=embedding_model_path,
            device=device,
            encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
        )
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

        print(f"\n{'='*70}")
        print(f"🚀 RAGAS 评估")
        print(f"{'='*70}")
        print(f"样本数: {n_samples}")
        print(f"评估指标: Faithfulness, Answer Relevancy, Context Recall, Context Precision, Context Entity Recall, Noise Sensitivity (Relevant), Noise Sensitivity (Irrelevant)")
        print(f"{'='*70}\n")

        # 创建 HuggingFace Dataset
        eval_dataset = Dataset.from_dict(dataset)

        # RAGAS 每个 (metric, sample) 是一个 executor.submit，因此 total = n_samples * len(metrics)
        total_units = n_samples * len(metrics)
        if sys.stderr.isatty():
            from tqdm import tqdm
            progress_bar = tqdm(total=total_units, desc="RAGAS evaluation")
        else:
            pf_path = str(Path(self.cache_dir) / "ragas_progress.log") if self.cache_dir else None
            progress_bar = _LoggerProgressBar(
                total=total_units,
                logger=_LOGGER,
                label="RAGAS",
                progress_file_path=pf_path,
            )

        # 执行评估
        print("开始评估...")
        eval_results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            run_config=self.run_config,
            _pbar=progress_bar,
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
                    "faithfulness": _coerce_metric_value(row.get("faithfulness")),
                    "answer_relevancy": _coerce_metric_value(row.get("answer_relevancy")),
                    "context_recall": _coerce_metric_value(row.get("context_recall")),
                    "context_precision": _coerce_metric_value(row.get("context_precision")),
                    "context_entity_recall": _coerce_metric_value(row.get("context_entity_recall")),
                    "noise_sensitivity_relevant": _coerce_metric_value(
                        row.get("noise_sensitivity(mode=relevant)", row.get("noise_sensitivity"))
                    ),
                    "noise_sensitivity_irrelevant": _coerce_metric_value(
                        row.get("noise_sensitivity(mode=irrelevant)")
                    ),
                },
            }
            
            # 计算单个样本的平均分（不包含噪声敏感性，因为它是越低越好的指标）
            metrics_for_avg = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "context_entity_recall"]
            valid_metrics = [
                sample_result["metrics"][m]
                for m in metrics_for_avg
                if sample_result["metrics"][m] is not None
            ]
            sample_result["ragas_score"] = round(sum(valid_metrics) / len(valid_metrics), 4) if valid_metrics else None
            
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
                if value is not None:
                    metrics_data[metric_name].append(value)
            
            # 反向指标（越低越好） - 包含 0 值
            for metric_name in ["noise_sensitivity_relevant", "noise_sensitivity_irrelevant"]:
                value = sample["metrics"][metric_name]
                if value is not None:
                    metrics_data[metric_name].append(value)
            
            ragas_score = sample["ragas_score"]
            if ragas_score is not None:
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
    
    INPUT_FILE = "/data/h50056789/Rag_Chunking/eval/LongBench/sample_results.json"
    OUTPUT_DIR = "/data/h50056789/Rag_Chunking/eval/LongBench"
    VLLM_API = "http://localhost:8005/v1"
    EMBEDDING_MODEL = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
    DEVICE = "cuda:1"
    CACHE_DIR = "./ragas_cache"
    
    input_basename = os.path.basename(INPUT_FILE).replace('.json', '')
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_ragas_eval.json")
    
    print("\n" + "="*70)
    print("RAGAS 评估器启动")
    print("="*70)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {output_file}")
    print(f"vLLM API: {VLLM_API}")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print(f"设备: {DEVICE}")
    print("="*70 + "\n")
    
    evaluator = RAGASEvaluator(
        vllm_api_base=VLLM_API,
        embedding_model_path=EMBEDDING_MODEL,
        device=DEVICE,
        enable_cache=True,
        cache_dir=CACHE_DIR
    )
    
    results = evaluator.evaluate_from_file(
        input_json_path=INPUT_FILE,
        output_json_path=output_file
    )
    
    print("\n" + "="*70)
    print("✅ 评估完成!")
    print("="*70)
