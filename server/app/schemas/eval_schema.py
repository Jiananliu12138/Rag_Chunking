from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


# ── 传统指标评估 ──────────────────────────────────────────────────────────────

class TraditionalEvalRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    "The transformer architecture uses self-attention mechanisms.",
                    "BERT is a bidirectional encoder representation model.",
                ],
                "answers": [
                    ["The transformer uses attention mechanisms.", "Transformer relies on self-attention."],
                    ["BERT stands for Bidirectional Encoder Representations from Transformers."],
                ],
                "enable_bert_score": False,
                "bert_score_model": "roberta-large",
                "bert_score_device": "cuda:0",
            }
        }
    )

    predictions: list[str] = Field(..., min_length=1, description="模型预测答案列表")
    answers: list[list[str]] = Field(..., min_length=1, description="参考答案列表（每条可含多个参考）")
    enable_bert_score: bool = Field(False, description="是否计算 BERTScore（需要 GPU）")
    bert_score_model: str = Field("roberta-large", description="BERTScore 使用的模型")
    bert_score_device: str = Field("cuda:0", description="BERTScore 计算设备")


class TraditionalEvalResult(BaseModel):
    f1: float
    rouge_l: float
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    bert_score_f1: Optional[float] = None
    sample_count: int


# ── RAGAS 评估 ────────────────────────────────────────────────────────────────

class RAGASDataset(BaseModel):
    question: list[str]
    answer: list[str]
    contexts: list[list[str]]
    ground_truth: list[str]


class RAGASEvalRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset": {
                    "question": ["What is RAG?", "How does BERT work?"],
                    "answer": ["RAG combines retrieval with generation.", "BERT uses bidirectional transformers."],
                    "contexts": [
                        ["Retrieval-Augmented Generation (RAG) is a technique..."],
                        ["BERT is a pre-trained model based on transformers..."],
                    ],
                    "ground_truth": [
                        "RAG retrieves relevant documents and uses them to generate answers.",
                        "BERT is trained on masked language modeling and next sentence prediction.",
                    ],
                },
                "vllm_api_base": "http://localhost:8005/v1",
                "vllm_api_key": "EMPTY",
                "vllm_model_name": "/path/to/Qwen2.5-7B-Instruct",
                "embedding_model_path": "/path/to/bge-large-en-v1.5",
                "device": "cuda:0",
                "enable_cache": True,
                "cache_dir": "./ragas_cache",
            }
        }
    )

    dataset: RAGASDataset = Field(..., description="RAGAS 格式的评估数据集")
    vllm_api_base: str = Field("http://localhost:8005/v1", description="vLLM API 地址")
    vllm_api_key: str = Field("EMPTY", description="API Key")
    vllm_model_name: str = Field(..., description="评估用 LLM 模型名称")
    embedding_model_path: str = Field(..., description="评估用嵌入模型路径")
    device: str = Field("cuda:0", description="评估设备")
    enable_cache: bool = Field(True, description="是否启用 RAGAS 磁盘缓存")
    cache_dir: str = Field("./ragas_cache", description="缓存目录")


class RAGASMetricSummary(BaseModel):
    mean: float
    min: float
    max: float


class RAGASSummary(BaseModel):
    ragas_score: RAGASMetricSummary
    faithfulness: RAGASMetricSummary
    answer_relevancy: RAGASMetricSummary
    context_recall: RAGASMetricSummary
    context_precision: RAGASMetricSummary
    context_entity_recall: RAGASMetricSummary
    noise_sensitivity_relevant: RAGASMetricSummary
    noise_sensitivity_irrelevant: RAGASMetricSummary


class RAGASEvalResult(BaseModel):
    summary: RAGASSummary
    sample_count: int
    samples: list[dict[str, Any]]


# ── 组件级 Chunk 质量评估 ─────────────────────────────────────────────────────

class ChunkQualityRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": [
                    "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
                    "BERT is based on the transformer encoder and trained with masked language modeling.",
                    "GPT uses the transformer decoder for autoregressive language generation.",
                ],
                "ppl_model_name": "/path/to/internlm3-8b-instruct",
                "sim_model_name": "/path/to/bge-large-en-v1.5",
                "enable_semantic_similarity": True,
                "enable_boundary_clarity": True,
                "device_map": "auto",
            }
        }
    )

    chunks: list[str] = Field(..., min_length=2, description="待评估的文本块列表（至少2个）")
    ppl_model_name: str = Field(..., description="用于困惑度计算的语言模型路径")
    sim_model_name: str = Field("BAAI/all-MiniLM-L6-v2", description="语义相似度模型路径")
    enable_semantic_similarity: bool = Field(True, description="是否计算语义不相似度")
    enable_boundary_clarity: bool = Field(True, description="是否计算边界清晰度 BC")
    device_map: str = Field("auto", description="模型设备映射策略")


class ChunkPairResult(BaseModel):
    semantic_dissimilarity: Optional[float] = None
    boundary_clarity: Optional[float] = None


class ChunkQualityResult(BaseModel):
    avg_semantic_dissimilarity: float
    avg_boundary_clarity: float
    num_pairs: int
    details: list[ChunkPairResult]


# ── 组件级 Chunk 黏连度评估 ───────────────────────────────────────────────────

class ChunkStickinessRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": [
                    "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
                    "BERT is based on the transformer encoder and trained with masked language modeling.",
                    "GPT uses the transformer decoder for autoregressive language generation.",
                ],
                "model_path": "/path/to/internlm3-8b-instruct",
                "threshold": 0.8,
                "delta": 0.0,
                "device_map": "auto",
            }
        }
    )

    chunks: list[str] = Field(..., min_length=2, description="待评估的文本块列表（至少2个）")
    model_path: str = Field(..., description="用于困惑度计算的语言模型路径")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="边权重阈值")
    delta: float = Field(0.0, ge=0.0, description="位置距离惩罚系数")
    device_map: str = Field("auto", description="模型设备映射策略")


class ChunkStickinessResult(BaseModel):
    structural_entropy_complete: float = Field(..., description="完全图结构熵")
    structural_entropy_incomplete: float = Field(..., description="不完全图结构熵")
