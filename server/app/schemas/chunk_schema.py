from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkMethod(str, Enum):
    TOKEN = "token"
    SEMANTIC = "semantic"
    LLAMAINDEX = "llamaindex"
    LUMBER = "lumber"


# ── Token 分块参数 ────────────────────────────────────────────────────────────

class TokenChunkParams(BaseModel):
    chunk_token_size: int = Field(1200, ge=64, description="每个块的最大 token 数")
    chunk_overlap_token_size: int = Field(100, ge=0, description="相邻块的重叠 token 数")
    split_by_character: Optional[str] = Field("\n\n", description="优先在该字符处切割")
    split_by_character_only: bool = Field(False, description="仅在指定字符处切割")


# ── Semantic 分块参数 ─────────────────────────────────────────────────────────

class SemanticChunkParams(BaseModel):
    embed_model_path: str = Field(..., description="HuggingFace 嵌入模型本地路径")
    buffer_size: int = Field(1, ge=1, description="SemanticSplitter 缓冲句子数")
    breakpoint_percentile_threshold: int = Field(74, ge=1, le=99, description="语义断点百分位阈值")


# ── LlamaIndex 分块参数 ───────────────────────────────────────────────────────

class LlamaIndexChunkParams(BaseModel):
    chunk_size: int = Field(512, ge=32, description="每个块的最大字符/token 数")
    chunk_overlap: int = Field(50, ge=0, description="相邻块重叠大小")


# ── Lumber 分块参数 ───────────────────────────────────────────────────────────

class LumberChunkParams(BaseModel):
    llm_api_base: str = Field("http://localhost:8005", description="vLLM 服务地址")
    model_type: str = Field("Qwen2.5-7B-Instruct", description="模型名称")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(3072, ge=128)


# ── 请求 / 响应 ───────────────────────────────────────────────────────────────

class ChunkRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Token 分块（默认）",
                    "value": {
                        "text": "Natural language processing (NLP) is a field of AI...\n\nDeep learning has revolutionized NLP...",
                        "method": "token",
                        "token_params": {
                            "chunk_token_size": 1200,
                            "chunk_overlap_token_size": 100,
                            "split_by_character": "\n\n",
                        },
                    },
                },
                {
                    "summary": "语义分块",
                    "value": {
                        "text": "Climate change is one of the greatest challenges...\n\nRenewable energy sources include solar and wind...",
                        "method": "semantic",
                        "semantic_params": {
                            "embed_model_path": "/path/to/bge-large-en-v1.5",
                            "buffer_size": 1,
                            "breakpoint_percentile_threshold": 74,
                        },
                    },
                },
                {
                    "summary": "LlamaIndex 固定窗口分块",
                    "value": {
                        "text": "The history of artificial intelligence dates back to the 1950s...",
                        "method": "llamaindex",
                        "llamaindex_params": {"chunk_size": 512, "chunk_overlap": 50},
                    },
                },
                {
                    "summary": "Lumber LLM 驱动分块",
                    "value": {
                        "text": "Section 1: Introduction to Machine Learning...\n\nSection 2: Deep Neural Networks...",
                        "method": "lumber",
                        "lumber_params": {
                            "llm_api_base": "http://localhost:8005",
                            "model_type": "Qwen2.5-7B-Instruct",
                            "temperature": 0.2,
                            "max_tokens": 3072,
                        },
                    },
                },
            ]
        }
    )

    text: str = Field(..., min_length=1, description="待分块的原始文本")
    method: ChunkMethod = Field(ChunkMethod.TOKEN, description="分块方法")
    token_params: Optional[TokenChunkParams] = None
    semantic_params: Optional[SemanticChunkParams] = None
    llamaindex_params: Optional[LlamaIndexChunkParams] = None
    lumber_params: Optional[LumberChunkParams] = None


class ChunkResult(BaseModel):
    chunks: list[str] = Field(..., description="分块后的文本列表")
    chunk_count: int = Field(..., description="分块数量")
    method: ChunkMethod
    time_cost: float = Field(..., description="耗时（秒）")


class ChunkMethodInfo(BaseModel):
    name: ChunkMethod
    description: str
    params_schema: dict[str, Any]
