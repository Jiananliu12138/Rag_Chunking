from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


# ── 传统指标评估 ──────────────────────────────────────────────────────────────

class TraditionalEvalRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "方式1：直接传 predictions 和 answers",
                    "value": {
                        "predictions": [
                            "The transformer architecture uses self-attention mechanisms.",
                            "BERT is a bidirectional encoder representation model.",
                        ],
                        "answers": [
                            ["The transformer uses attention mechanisms.", "Transformer relies on self-attention."],
                            ["BERT stands for Bidirectional Encoder Representations from Transformers."],
                        ],
                        "enable_bert_score": False,
                    },
                },
                {
                    "summary": "方式2：传 test 字段（评估结果 JSON 格式）",
                    "value": {
                        "test": [
                            {
                                "_id": "q1",
                                "input": "What is a transformer?",
                                "llm_ans": "A transformer is a neural architecture...",
                                "answers": ["A transformer uses attention mechanisms."],
                            },
                            {
                                "_id": "q2",
                                "input": "What is BERT?",
                                "llm_ans": "BERT is a bidirectional model...",
                                "answers": ["BERT stands for Bidirectional Encoder..."],
                            },
                        ],
                        "enable_bert_score": False,
                    },
                },
            ]
        }
    )

    predictions: Optional[list[str]] = Field(None, description="模型预测答案列表（与 test 二选一）")
    answers: Optional[list[list[str]]] = Field(None, description="参考答案列表（与 test 二选一）")
    test: Optional[list[dict[str, Any]]] = Field(
        None,
        description="评估结果 JSON 格式（列表，每项含 llm_ans、answers 等），与 predictions/answers 二选一",
    )
    enable_bert_score: Optional[bool] = Field(
        None, description="是否计算 BERTScore（可选，未提供时从配置读取 DEFAULT_ENABLE_BERT_SCORE）"
    )
    bert_score_model: Optional[str] = Field(
        None, description="BERTScore 使用的模型（可选，未提供时从配置读取 DEFAULT_BERT_SCORE_MODEL）"
    )
    bert_score_device: Optional[str] = Field(
        None, description="BERTScore 计算设备（可选，未提供时从配置读取 DEFAULT_BERT_SCORE_DEVICE）"
    )


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

class RAGASEvalRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test": [
                    {
                        "_id": "q1",
                        "input": "Who is Peter Rosegger?",
                        "llm_ans": "Peter Rosegger was an Austrian writer and poet...",
                        "answers": [
                            "He was an Austrian writer and poet from Krieglach in the province of Styria."
                        ],
                        "retrieval_list": [
                            "Passage 3: Peter Rosegger ...",
                            "Early life ...",
                        ],
                    }
                ],
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

    test: Any = Field(
        ...,
        description=(
            "必填：直接传入原始评估 JSON（如 sample_results.json 或标准 RAGAS 格式），"
            "服务端会自动拆解为 question/answer/contexts/ground_truth 供 RAGAS 使用。"
        ),
    )
    vllm_api_base: Optional[str] = Field(
        None, description="vLLM API 地址（可选，未提供时从配置读取 DEFAULT_RAGAS_VLLM_API_BASE）"
    )
    vllm_api_key: Optional[str] = Field(
        None, description="API Key（可选，未提供时从配置读取 DEFAULT_RAGAS_VLLM_API_KEY）"
    )
    vllm_model_name: Optional[str] = Field(
        None, description="评估用 LLM 模型名称（可选，未提供时从配置读取 DEFAULT_RAGAS_VLLM_MODEL_NAME）"
    )
    embedding_model_path: Optional[str] = Field(
        None, description="评估用嵌入模型路径（可选，未提供时从配置读取 DEFAULT_RAGAS_EMBEDDING_MODEL_PATH）"
    )
    device: Optional[str] = Field(
        None, description="评估设备（可选，未提供时从配置读取 DEFAULT_RAGAS_DEVICE）"
    )
    enable_cache: Optional[bool] = Field(
        None, description="是否启用 RAGAS 磁盘缓存（可选，未提供时从配置读取 DEFAULT_RAGAS_ENABLE_CACHE）"
    )
    cache_dir: Optional[str] = Field(
        None, description="缓存目录（可选，未提供时从配置读取 DEFAULT_RAGAS_CACHE_DIR）"
    )


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


class RAGASEvalFileRequest(BaseModel):
    """从 JSON 文件读取 RAGAS 数据集并评估。"""
    input_path: str = Field(
        ...,
        description=(
            "输入 JSON 文件路径，支持两种格式：\n"
            "1. 标准 RAGAS 格式：{\"question\": [...], \"answer\": [...], \"contexts\": [...], \"ground_truth\": [...]}\n"
            "2. sample_results.json 格式：列表，每项含 input/llm_ans/answers/retrieval_list"
        ),
    )
    vllm_api_base: Optional[str] = Field(
        None, description="vLLM API 地址（可选，未提供时从配置读取）"
    )
    vllm_api_key: Optional[str] = Field(None, description="API Key（可选，未提供时从配置读取）")
    vllm_model_name: Optional[str] = Field(None, description="评估用 LLM 模型名称（可选，未提供时从配置读取）")
    embedding_model_path: Optional[str] = Field(None, description="评估用嵌入模型路径（可选，未提供时从配置读取）")
    device: Optional[str] = Field(None, description="评估设备（可选，未提供时从配置读取）")
    enable_cache: Optional[bool] = Field(None, description="是否启用 RAGAS 磁盘缓存（可选，未提供时从配置读取）")
    cache_dir: Optional[str] = Field(None, description="缓存目录（可选，未提供时从配置读取）")


# ── 组件级 Chunk 质量评估 ─────────────────────────────────────────────────────

class ChunkQualityRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": {
                    "splits": [
                        [
                            "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
                            "label1",
                        ],
                        [
                            "BERT is based on the transformer encoder and trained with masked language modeling.",
                            "label2",
                        ],
                        [
                            "GPT uses the transformer decoder for autoregressive language generation.",
                            "label3",
                        ],
                    ]
                },
                "enable_semantic_similarity": True,
                "enable_boundary_clarity": True,
            }
        }
    )

    chunks: Any = Field(
        ...,
        description=(
            "待评估的分块结果，支持多种 JSON 格式："
            "['chunk1', 'chunk2', ...]、[['chunk1', label], ...]、"
            "{'splits': [...]}, {'final_chunks': [...]}, "
            "[{'name': 'doc', 'final_chunks': [...]}, ...] 等；"
            "服务端会解析为纯文本块列表。"
        ),
    )
    enable_semantic_similarity: Optional[bool] = Field(
        None,
        description="是否计算语义不相似度（可选，未提供时从配置读取 COMPONENT_ENABLE_SEMANTIC_SIMILARITY）",
    )
    enable_boundary_clarity: Optional[bool] = Field(
        None,
        description="是否计算边界清晰度 BC（可选，未提供时从配置读取 COMPONENT_ENABLE_BOUNDARY_CLARITY）",
    )
    ppl_model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的困惑度模型路径；未提供时使用 COMPONENT_PPL_MODEL_PATH。",
    )
    sim_model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的语义相似度模型路径；未提供时使用 COMPONENT_SIM_MODEL_PATH。",
    )
    use_vllm: Optional[bool] = Field(
        None,
        description="可选：是否使用 vLLM API 计算困惑度；未提供时默认为 False（本地模型）。",
    )
    vllm_api_base: Optional[str] = Field(
        None,
        description="可选：vLLM 服务地址（例如 http://localhost:8005/v1），use_vllm=True 时可覆盖默认配置。",
    )
    vllm_model_name: Optional[str] = Field(
        None,
        description="可选：vLLM 模型名称，use_vllm=True 时可覆盖默认配置。",
    )


class ChunkPairResult(BaseModel):
    semantic_dissimilarity: Optional[float] = None
    boundary_clarity: Optional[float] = None


class ChunkQualityResult(BaseModel):
    avg_semantic_dissimilarity: float
    avg_boundary_clarity: float
    num_pairs: int
    details: list[ChunkPairResult]


class ChunkQualityFileRequest(BaseModel):
    """从分块结果文件读取并进行 Chunk 质量评估（BC + 语义不相似度）。"""

    input_path: str = Field(
        ...,
        description=(
            "分块结果 JSON 文件路径，支持与 chunking 输出一致的多种格式："
            "['chunk1', 'chunk2', ...]、[['chunk1', label], ...]、"
            "{'splits': [...]}, {'final_chunks': [...]}, "
            "[{'name': 'doc', 'final_chunks': [...]}, ...] 等；"
            "服务端会解析为纯文本块列表。"
        ),
    )
    enable_semantic_similarity: Optional[bool] = Field(
        None,
        description="是否计算语义不相似度（可选，未提供时从配置读取 COMPONENT_ENABLE_SEMANTIC_SIMILARITY）",
    )
    enable_boundary_clarity: Optional[bool] = Field(
        None,
        description="是否计算边界清晰度 BC（可选，未提供时从配置读取 COMPONENT_ENABLE_BOUNDARY_CLARITY）",
    )
    ppl_model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的困惑度模型路径；未提供时使用 COMPONENT_PPL_MODEL_PATH。",
    )
    sim_model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的语义相似度模型路径；未提供时使用 COMPONENT_SIM_MODEL_PATH。",
    )
    use_vllm: Optional[bool] = Field(
        None,
        description="可选：是否使用 vLLM API 计算困惑度；未提供时默认为 False（本地模型）。",
    )
    vllm_api_base: Optional[str] = Field(
        None,
        description="可选：vLLM 服务地址（例如 http://localhost:8005/v1），use_vllm=True 时可覆盖默认配置。",
    )
    vllm_model_name: Optional[str] = Field(
        None,
        description="可选：vLLM 模型名称，use_vllm=True 时可覆盖默认配置。",
    )


# ── 组件级 Chunk 黏连度评估 ───────────────────────────────────────────────────

class ChunkStickinessRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": {
                    "final_chunks": [
                        "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
                        "BERT is based on the transformer encoder and trained with masked language modeling.",
                        "GPT uses the transformer decoder for autoregressive language generation.",
                    ]
                },
                "threshold": 0.8,
                "delta": 0.0,
            }
        }
    )

    chunks: Any = Field(
        ...,
        description=(
            "待评估的分块结果，支持多种 JSON 格式："
            "['chunk1', 'chunk2', ...]、[['chunk1', label], ...]、"
            "{'splits': [...]}, {'final_chunks': [...]}, "
            "[{'name': 'doc', 'final_chunks': [...]}, ...] 等；"
            "服务端会解析为纯文本块列表。"
        ),
    )
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="边权重阈值")
    delta: float = Field(0.0, ge=0.0, description="位置距离惩罚系数")
    model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的黏连度评估模型路径；未提供时使用 STICKINESS_MODEL_PATH。",
    )
    use_vllm: Optional[bool] = Field(
        None,
        description="可选：是否使用 vLLM API 计算困惑度；未提供时默认为 False（本地模型）。",
    )
    vllm_api_base: Optional[str] = Field(
        None,
        description="可选：vLLM 服务地址（例如 http://localhost:8005/v1），use_vllm=True 时可覆盖默认配置。",
    )
    vllm_model_name: Optional[str] = Field(
        None,
        description="可选：vLLM 模型名称，use_vllm=True 时可覆盖默认配置。",
    )


class ChunkStickinessResult(BaseModel):
    structural_entropy_complete: float = Field(..., description="完全图结构熵")
    structural_entropy_incomplete: float = Field(..., description="不完全图结构熵")
    graph_complete: dict[int, dict[int, float]] = Field(
        ...,
        description=(
            "归一化后的完全图邻接矩阵，graph_complete[i][j] 为从块 i 到块 j 的权重；"
            "权重越小表示两块越相关，权重越大表示越不相关。"
        ),
    )
    graph_incomplete: dict[int, dict[int, float]] = Field(
        ...,
        description=(
            "不完全图（上三角）邻接矩阵，只保留 i <= j 的边；"
            "用于计算不完全图结构熵。"
        ),
    )


class ChunkStickinessFileRequest(BaseModel):
    """从分块结果文件读取并进行 Chunk 黏连度评估（结构熵）。"""

    input_path: str = Field(
        ...,
        description=(
            "分块结果 JSON 文件路径，支持与 chunking 输出一致的多种格式："
            "['chunk1', 'chunk2', ...]、[['chunk1', label], ...]、"
            "{'splits': [...]}, {'final_chunks': [...]}, "
            "[{'name': 'doc', 'final_chunks': [...]}, ...] 等；"
            "服务端会解析为纯文本块列表。"
        ),
    )
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="边权重阈值（可选，未提供时从配置读取 STICKINESS_THRESHOLD）"
    )
    delta: Optional[float] = Field(
        None, ge=0.0, description="位置距离惩罚系数（可选，未提供时从配置读取 STICKINESS_DELTA）"
    )
    model_path: Optional[str] = Field(
        None,
        description="可选：覆盖默认的黏连度评估模型路径；未提供时使用 STICKINESS_MODEL_PATH。",
    )
    use_vllm: Optional[bool] = Field(
        None,
        description="可选：是否使用 vLLM API 计算困惑度；未提供时默认为 False（本地模型）。",
    )
    vllm_api_base: Optional[str] = Field(
        None,
        description="可选：vLLM 服务地址（例如 http://localhost:8005/v1），use_vllm=True 时可覆盖默认配置。",
    )
    vllm_model_name: Optional[str] = Field(
        None,
        description="可选：vLLM 模型名称，use_vllm=True 时可覆盖默认配置。",
    )


# ── 文件输入的传统指标评估 ─────────────────────────────────────────────────────

class TraditionalEvalFileRequest(BaseModel):
    """从 JSON 文件读取评估结果并计算传统指标。"""
    input_path: str = Field(..., description="评估结果 JSON 文件路径（格式同 sample_results.json）")
    enable_bert_score: Optional[bool] = Field(
        None, description="是否计算 BERTScore（可选，未提供时从配置读取 DEFAULT_ENABLE_BERT_SCORE）"
    )
    bert_score_model: Optional[str] = Field(
        None, description="BERTScore 使用的模型（可选，未提供时从配置读取 DEFAULT_BERT_SCORE_MODEL）"
    )
    bert_score_device: Optional[str] = Field(
        None, description="BERTScore 计算设备（可选，未提供时从配置读取 DEFAULT_BERT_SCORE_DEVICE）"
    )
