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
    embed_model_path: Optional[str] = Field(
        None,
        description=(
            "可选：覆盖默认的嵌入模型路径。"
            "如不提供，将使用环境变量 DEFAULT_EMBEDDING_MODEL 配置的模型。"
        ),
    )
    buffer_size: int = Field(1, ge=1, description="SemanticSplitter 缓冲句子数")
    breakpoint_percentile_threshold: int = Field(74, ge=1, le=99, description="语义断点百分位阈值")


# ── LlamaIndex 分块参数 ───────────────────────────────────────────────────────

class LlamaIndexChunkParams(BaseModel):
    chunk_size: int = Field(512, ge=32, description="每个块的最大字符/token 数")
    chunk_overlap: int = Field(50, ge=0, description="相邻块重叠大小")
    # cache_dir / num_workers 不再从请求传入，统一从环境和模块默认读取


# ── Lumber 分块参数 ───────────────────────────────────────────────────────────

class LumberChunkParams(BaseModel):
    llm_api_base: Optional[str] = Field(
        None,
        description=(
            "可选：覆盖默认 vLLM 服务地址。"
            "如不提供，将使用 DEFAULT_LLM_API_BASE（通常形如 http://host:port/v1）。"
        ),
    )
    model_type: Optional[str] = Field(
        None,
        description=(
            "可选：覆盖默认模型名称。"
            "如不提供，将使用 DEFAULT_LLM_MODEL。"
        ),
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="可选：覆盖默认温度参数（默认取 DEFAULT_LLM_TEMPERATURE）。"
    )
    max_tokens: Optional[int] = Field(
        None, ge=128, description="可选：覆盖默认生成最大 Token 数（默认取 DEFAULT_LLM_MAX_TOKENS）。"
    )


# ── 文本分块请求 ─────────────────────────────────────────────────────────────

class ChunkTextRequest(BaseModel):
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
                            "llm_api_base": "http://localhost:8001/v1",
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


class ChunkTextResult(BaseModel):
    """与 public_method 各实现的 chunk_text 返回格式一致。"""
    success: bool = Field(..., description="是否成功")
    splits: list[list[str]] = Field(..., description="分块结果，每项为 [text] 或 [text, doc_id]")
    time_cost: float = Field(..., description="耗时（秒）")
    message: str = Field(..., description="结果说明")


# ── 文件分块请求 / 响应 ───────────────────────────────────────────────────────

class ChunkFileRequest(BaseModel):
    method: ChunkMethod = Field(ChunkMethod.TOKEN, description="分块方法")
    input_file: str = Field(..., description="输入文件路径（如 .jsonl）")
    output_dir: str = Field(..., description="输出目录路径")
    token_params: Optional[TokenChunkParams] = None
    semantic_params: Optional[SemanticChunkParams] = None
    llamaindex_params: Optional[LlamaIndexChunkParams] = None
    lumber_params: Optional[LumberChunkParams] = None


class ChunkFileResult(BaseModel):
    success: bool = Field(..., description="是否成功")
    output_file: str = Field(..., description="输出文件路径")
    message: str = Field(..., description="结果说明")


# ── 方法列表 ─────────────────────────────────────────────────────────────────

class ChunkMethodInfo(BaseModel):
    name: ChunkMethod
    description: str
    params_schema: dict[str, Any]
