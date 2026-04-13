"""
评估指标:
1. Boundary Clarity (BC): 边界清晰度，基于困惑度比
2. Semantic Dissimilarity: 语义不相似度，基于embedding余弦相似度
"""

import json
import time
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import requests

from public_method.evaluation._parallel import parallel_map


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

    # 使用 vLLM API 计算困惑度时的配置（仅影响 PPL / BC，语义模型仍走本地）
    use_vllm: bool = False
    vllm_api_base: Optional[str] = None  # 例如 http://localhost:8005/v1
    vllm_model_name: Optional[str] = None
    
    # 输入输出路径（仅在命令行模式下使用）
    input_file: str = 'merge_data/db_qa_semantic_68.json'
    output_file: str = 'mergechunk_eval3/db_qa_semantic_68.json'
    
    # 评估选项
    enable_semantic_similarity: bool = True
    enable_boundary_clarity: bool = True
    score_temperature: float = 6.0

    # 最多评估多少个文本块（-1 表示使用全部）
    max_eval_chunks: int = -1

    # 并发与超时（仅在 use_vllm=True 时生效）
    max_workers: int = 16
    request_timeout: int = 600

    # 日志配置（仅控制台）
    log_level: str = 'INFO'
    
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
            'no_semantic_similarity': self.semantic_dissimilarity if math.isfinite(self.semantic_dissimilarity) else None,
            'no_relative_perplexity': self.boundary_clarity if math.isfinite(self.boundary_clarity) else None
        }


@dataclass
class AggregatedResults:
    """聚合的评估结果"""
    semantic_dissimilarity_avg: Optional[float] = None
    boundary_clarity_avg: Optional[float] = None
    semantic_dissimilarity_valid_count: int = 0
    semantic_dissimilarity_error_count: int = 0
    boundary_clarity_valid_count: int = 0
    boundary_clarity_error_count: int = 0
    num_pairs: int = 0
    individual_results: List[EvaluationResult] = field(default_factory=list)
    
    def __str__(self) -> str:
        sem_avg = f"{self.semantic_dissimilarity_avg:.4f}" if self.semantic_dissimilarity_avg is not None else "None"
        bc_avg = f"{self.boundary_clarity_avg:.4f}" if self.boundary_clarity_avg is not None else "None"
        return (
            f"评估结果摘要:\n"
            f"  文本块对数: {self.num_pairs}\n"
            f"  平均语义不相似度: {sem_avg} "
            f"(valid={self.semantic_dissimilarity_valid_count}, error={self.semantic_dissimilarity_error_count})\n"
            f"  平均边界清晰度(BC): {bc_avg} "
            f"(valid={self.boundary_clarity_valid_count}, error={self.boundary_clarity_error_count})"
        )


# ============================================================================
# 核心评估器类
# ============================================================================

class VLLMPerplexityClient:
    """
    通过 vLLM OpenAI-compatible /completions 接口近似计算困惑度。
    基本思路：
      - ppl(text) ~= -avg_logprob(text_tokens)
      - ppl(target | context) 通过分别计算 context+target 与 context 的 token 数来截取 target 段 logprob
    """

    def __init__(self, api_base: str, model_name: str, timeout: int = 600):
        self.api_base = api_base
        self.model_name = model_name
        self.timeout = timeout

    def _get_token_logprobs(self, text: str) -> List[float]:
        """
        调用 /completions 接口，使用 echo+logprobs 拿到 prompt token 的 logprob 列表。
        依赖 vLLM 的 OpenAI 兼容行为。
        """
        payload = {
            "model": self.model_name,
            "prompt": text,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        resp = requests.post(f"{self.api_base}/completions", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # choices[0].logprobs.token_logprobs 对应整个 prompt 的 token 对数概率
        logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
        # 过滤掉 None（例如特殊 token）
        return [lp for lp in logprobs if lp is not None]

    def _mean_neg_logprob(self, text: str) -> float:
        token_logprobs = self._get_token_logprobs(text)
        if not token_logprobs:
            return 0.0
        # 负对数概率的平均值，作为“困惑度”的 proxy
        return -sum(token_logprobs) / len(token_logprobs)

    def perplexity(self, context: str, target: str) -> float:
        """
        近似 ppl(target | context)：
          - 先拿 context+target 的 token_logprobs
          - 再拿 context 的 token_logprobs，做差得到 target 段
        """
        full = self._get_token_logprobs(context + target)
        ctx = self._get_token_logprobs(context) if context.strip() else []
        if not full:
            return 0.0
        if len(full) <= len(ctx):
            # 退化情况：长度异常时就整体当成 target
            target_logprobs = full
        else:
            target_logprobs = full[len(ctx):]
        return -sum(target_logprobs) / max(len(target_logprobs), 1)


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
        
        # 延迟加载模型 / 客户端
        self.ppl_model = None
        self.ppl_tokenizer = None
        self.vllm_client: Optional[VLLMPerplexityClient] = None
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
        
        return logger
    
    def load_models(self):
        """加载所需模型"""
        # 加载困惑度计算模型或 vLLM 客户端
        if self.config.enable_boundary_clarity:
            if self.config.use_vllm:
                if not (self.config.vllm_api_base and self.config.vllm_model_name):
                    raise ValueError("use_vllm=True 时必须提供 vllm_api_base 与 vllm_model_name")
                self.logger.info(
                    f"使用 vLLM API 计算困惑度: {self.config.vllm_model_name} @ {self.config.vllm_api_base}"
                )
                self.vllm_client = VLLMPerplexityClient(
                    api_base=self.config.vllm_api_base.rstrip("/"),
                    model_name=self.config.vllm_model_name,
                    timeout=self.config.request_timeout,
                )
                self.logger.info(
                    f"✓ vLLM 困惑度客户端初始化完成 (max_workers={self.config.max_workers}, timeout={self.config.request_timeout}s)"
                )
            else:
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
        # vLLM 模式：通过远程 API 近似计算困惑度
        if self.config.use_vllm:
            if self.vllm_client is None:
                raise RuntimeError("vLLM 客户端未初始化，请先调用 load_models()")
            return self.vllm_client.perplexity(context, target)

        # 本地 HuggingFace 模式
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
        loss = loss_fn(active_logits, active_labels).item()
        perplexity = math.exp(loss)

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
        if (
            not math.isfinite(ppl_without_context)
            or not math.isfinite(ppl_with_context)
            or ppl_without_context <= 0
        ):
            self.logger.warning(
                "Skipping invalid boundary clarity value: ppl_without_context=%s, ppl_with_context=%s",
                ppl_without_context,
                ppl_with_context,
            )
            return float('nan')

        bc = ppl_with_context/ppl_without_context
        score = (bc ** self.config.score_temperature) / (1 + bc ** self.config.score_temperature)

        return score
    
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

        优化点：
        - 语义相似度：一次性 batch encode 所有 chunks，比逐对 encode 快得多
        - 边界清晰度：所有 (text1, text2) 对的 vLLM 调用通过 ThreadPoolExecutor 并发

        Args:
            chunks: 文本块列表
            show_progress: 是否显示进度条

        Returns:
            聚合的评估结果
        """
        if len(chunks) < 2:
            raise ValueError(f"至少需要2个文本块，当前只有 {len(chunks)} 个")

        num_pairs = len(chunks) - 1
        self.logger.info(f"开始评估 {len(chunks)} 个文本块 ({num_pairs} 对)")

        # ── 语义不相似度：一次性 batch encode ───────────────────────────
        sem_dissimilarities: List[float] = [float('nan')] * num_pairs
        if self.config.enable_semantic_similarity:
            if self.sim_model is None:
                raise RuntimeError("语义模型未加载，请先调用 load_models()")
            embeddings = self.sim_model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            # 相邻余弦相似度 → 不相似度
            for i in range(num_pairs):
                similarity = float(embeddings[i] @ embeddings[i + 1].T)
                sem_dissimilarities[i] = 1.0 - similarity

        # ── 边界清晰度：并发计算所有相邻对 ──────────────────────────────
        bc_values: List[float] = [float('nan')] * num_pairs
        if self.config.enable_boundary_clarity:
            # 仅 vLLM 模式启用并发；本地 GPU 推理不适合多线程争抢同一模型
            workers = self.config.max_workers if self.config.use_vllm else 1

            def _bc(idx: int) -> tuple[int, float]:
                try:
                    return idx, self.calculate_boundary_clarity(chunks[idx], chunks[idx + 1])
                except Exception as exc:
                    self.logger.error(f"评估第 {idx} 对边界清晰度时出错: {exc}")
                    return idx, float('nan')

            bc_results = parallel_map(
                _bc,
                range(num_pairs),
                max_workers=workers,
                desc="边界清晰度",
                show_progress=show_progress,
                logger=self.logger,
            )
            for idx, value in bc_results:
                bc_values[idx] = value

        # ── 聚合 ───────────────────────────────────────────────────────
        results: List[EvaluationResult] = []
        sem_scores: List[float] = []
        bc_scores: List[float] = []
        for i in range(num_pairs):
            r = EvaluationResult(
                semantic_dissimilarity=sem_dissimilarities[i],
                boundary_clarity=bc_values[i],
            )
            results.append(r)
            if self.config.enable_semantic_similarity and math.isfinite(r.semantic_dissimilarity):
                sem_scores.append(r.semantic_dissimilarity)
            if self.config.enable_boundary_clarity and math.isfinite(r.boundary_clarity):
                bc_scores.append(r.boundary_clarity)

        aggregated = AggregatedResults(
            semantic_dissimilarity_avg=sum(sem_scores) / len(sem_scores) if sem_scores else None,
            boundary_clarity_avg=sum(bc_scores) / len(bc_scores) if bc_scores else None,
            semantic_dissimilarity_valid_count=len(sem_scores),
            semantic_dissimilarity_error_count=(num_pairs - len(sem_scores)) if self.config.enable_semantic_similarity else 0,
            boundary_clarity_valid_count=len(bc_scores),
            boundary_clarity_error_count=(num_pairs - len(bc_scores)) if self.config.enable_boundary_clarity else 0,
            num_pairs=len(results),
            individual_results=results
        )

        self.logger.info("评估完成")
        self.logger.info(
            "semantic_dissimilarity_avg=%s (valid=%s, error=%s), boundary_clarity_avg=%s (valid=%s, error=%s)",
            aggregated.semantic_dissimilarity_avg,
            aggregated.semantic_dissimilarity_valid_count,
            aggregated.semantic_dissimilarity_error_count,
            aggregated.boundary_clarity_avg,
            aggregated.boundary_clarity_valid_count,
            aggregated.boundary_clarity_error_count,
        )

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
        
        total_chunks = len(chunks)
        self.logger.info(f"成功读取 {total_chunks} 个文本块")

        # 如果配置了 max_eval_chunks，只评估前 N 个文本块
        if self.config.max_eval_chunks is not None and self.config.max_eval_chunks > 0:
            if total_chunks > self.config.max_eval_chunks:
                self.logger.info(
                    f"仅对前 {self.config.max_eval_chunks} 个文本块进行评估（总共 {total_chunks} 个）"
                )
                chunks = chunks[: self.config.max_eval_chunks]
        
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
                    "avg_boundary_clarity": results.boundary_clarity_avg,
                    "semantic_dissimilarity_valid_count": results.semantic_dissimilarity_valid_count,
                    "semantic_dissimilarity_error_count": results.semantic_dissimilarity_error_count,
                    "boundary_clarity_valid_count": results.boundary_clarity_valid_count,
                    "boundary_clarity_error_count": results.boundary_clarity_error_count,
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
    """主函数：用于本地实验，参数直接在代码里写死。"""
    # 根据你自己的实验路径/模型路径在这里改
    config = EvaluatorConfig(
        input_file='/data/h50056789/Rag_Chunking/component_eval/test_data/test.json',
        output_file='/data/h50056789/Rag_Chunking/component_eval/test_data/eval.json',
        ppl_model_name='/data/h50056789/Rag_chunk_bench/model/Qwen3-4B',
        sim_model_name='/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5',
        enable_semantic_similarity=True,
        enable_boundary_clarity=True,
        max_eval_chunks=200,  # 例如：最多评估前 200 个文本块；改成 -1 表示全部
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
from public_method.evaluation.component.chunk_eval_refactored import ChunkEvaluator, EvaluatorConfig

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
