from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkMethod(str, Enum):
    TOKEN = "token"
    SEMANTIC = "semantic"
    LLAMAINDEX = "llamaindex"
    LUMBER = "lumber"


class TokenChunkParams(BaseModel):
    chunk_token_size: int = Field(1200, ge=64, description="Max tokens per chunk")
    chunk_overlap_token_size: int = Field(100, ge=0, description="Overlap tokens between chunks")
    min_chunk_tokens: int = Field(0, ge=0, description="Merge chunks shorter than this token count into a neighboring chunk")
    split_by_character: Optional[str] = Field("\n\n", description="Preferred split delimiter or regex pattern")
    split_use_regex: bool = Field(False, description="Treat split_by_character as a regex pattern")
    split_by_character_only: bool = Field(False, description="Only split by the delimiter")


class SemanticChunkParams(BaseModel):
    embed_api_base: Optional[str] = Field(
        None,
        description=(
            "OpenAI-compatible embedding API base URL (e.g. vLLM endpoint). "
            "If omitted, the service falls back to DEFAULT_EMBEDDING_BASE."
        ),
    )
    embed_model_name: Optional[str] = Field(
        None,
        description=(
            "Embedding model name served at the API endpoint. "
            "If omitted, the service falls back to DEFAULT_EMBEDDING_NAME."
        ),
    )
    buffer_size: int = Field(1, ge=1, description="Sentence buffer size for SemanticSplitter")
    breakpoint_percentile_threshold: int = Field(
        74,
        ge=1,
        le=99,
        description="Semantic breakpoint percentile threshold",
    )


class LlamaIndexChunkParams(BaseModel):
    chunk_size: int = Field(512, ge=32, description="Max tokens per chunk")
    chunk_overlap: int = Field(50, ge=0, description="Overlap between chunks")


class LumberChunkParams(BaseModel):
    llm_api_base: Optional[str] = Field(
        None,
        description=(
            "Optional vLLM/OpenAI-compatible base URL. "
            "If omitted, the service falls back to DEFAULT_LLM_API_BASE."
        ),
    )
    model_type: Optional[str] = Field(
        None,
        description=(
            "Optional model name override. "
            "If omitted, the service falls back to DEFAULT_LLM_MODEL."
        ),
    )
    temperature: Optional[float] = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Lumber generation temperature",
    )
    max_tokens: Optional[int] = Field(
        64,
        ge=16,
        description="Lumber generation max tokens",
    )


class ChunkTextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Token chunking",
                    "value": {
                        "text": "Natural language processing (NLP) is a field of AI...\n\nDeep learning has revolutionized NLP...",
                        "method": "token",
                        "token_params": {
                            "chunk_token_size": 1200,
                            "chunk_overlap_token_size": 100,
                            "min_chunk_tokens": 0,
                            "split_by_character": "\n\n",
                            "split_use_regex": False,
                        },
                    },
                },
                {
                    "summary": "Semantic chunking",
                    "value": {
                        "text": "Climate change is one of the greatest challenges...\n\nRenewable energy sources include solar and wind...",
                        "method": "semantic",
                        "semantic_params": {
                            "embed_api_base": "http://localhost:8001/v1",
                            "embed_model_name": "BAAI/bge-m3",
                            "buffer_size": 1,
                            "breakpoint_percentile_threshold": 74,
                        },
                    },
                },
                {
                    "summary": "LlamaIndex fixed-window chunking",
                    "value": {
                        "text": "The history of artificial intelligence dates back to the 1950s...",
                        "method": "llamaindex",
                        "llamaindex_params": {"chunk_size": 512, "chunk_overlap": 50},
                    },
                },
                {
                    "summary": "Lumber LLM-driven chunking",
                    "value": {
                        "text": "Section 1: Introduction to Machine Learning...\n\nSection 2: Deep Neural Networks...",
                        "method": "lumber",
                        "lumber_params": {
                            "llm_api_base": "http://localhost:8001/v1",
                            "model_type": "Qwen2.5-7B-Instruct",
                            "temperature": 0.2,
                            "max_tokens": 64,
                        },
                    },
                },
            ]
        }
    )

    text: str = Field(..., min_length=1, description="Source text to chunk")
    method: ChunkMethod = Field(ChunkMethod.TOKEN, description="Chunking method")
    num_workers: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="Number of parallel workers (multiprocessing.Pool). Falls back to CHUNK_NUM_WORKERS if omitted.",
    )
    token_params: Optional[TokenChunkParams] = None
    semantic_params: Optional[SemanticChunkParams] = None
    llamaindex_params: Optional[LlamaIndexChunkParams] = None
    lumber_params: Optional[LumberChunkParams] = None


class ChunkTextResult(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    splits: list[list[str]] = Field(..., description="Chunk list, each item is [text] or [text, doc_id]")
    time_cost: float = Field(..., description="Time cost in seconds")
    message: str = Field(..., description="Result message")


class ChunkFileRequest(BaseModel):
    method: ChunkMethod = Field(ChunkMethod.TOKEN, description="Chunking method")
    input_file: str = Field(..., description="Input file path, e.g. .jsonl")
    output_dir: str = Field(..., description="Output directory path")
    num_workers: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="Number of parallel workers (multiprocessing.Pool). Falls back to CHUNK_NUM_WORKERS if omitted.",
    )
    token_params: Optional[TokenChunkParams] = None
    semantic_params: Optional[SemanticChunkParams] = None
    llamaindex_params: Optional[LlamaIndexChunkParams] = None
    lumber_params: Optional[LumberChunkParams] = None


class ChunkFileResult(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    output_file: str = Field(..., description="Output file path")
    message: str = Field(..., description="Result message")


class ChunkMethodInfo(BaseModel):
    name: ChunkMethod
    description: str
    params_schema: dict[str, Any]
