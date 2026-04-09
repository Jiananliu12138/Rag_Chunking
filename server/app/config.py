from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App metadata
    APP_TITLE: str = "Meta-Chunking RAG Server"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Backend service for Meta-Chunking. "
        "Provides chunking, indexing, retrieval, and evaluation APIs."
    )

    # Storage / runtime
    MILVUS_DATA_DIR: str = "/data/h50056789/Rag_Chunking/test_database"
    MILVUS_URI: str = ""
    TIKTOKEN_CACHE_DIR: str = "/data/h50056789/Rag_Chunking/tiktoken_cache"
    CHUNK_NUM_WORKERS: int = 4  # 分块任务的进程池 worker 数（multiprocessing）

    # Embedding defaults
    DEFAULT_EMBEDDING_MODEL: str = "/data/h50056789/Rag_Chunking/model/BAAI/bge-m3"
    DEFAULT_EMBEDDING_DEVICE: str = "cuda:0"
    DEFAULT_EMBEDDING_DIM: int = 1024
    DEFAULT_EMBEDDING_MAX_TOKENS: int | None = 8192

    # Shared model defaults
    DEFAULT_LLM_API_BASE: str = "http://localhost:8001/v1"
    DEFAULT_LLM_API_KEY: str = "EMPTY"
    DEFAULT_LLM_MODEL: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    DEFAULT_LLM_TEMPERATURE: float = 0.0
    DEFAULT_LLM_MAX_TOKENS: int = 8192

    DEFAULT_RETRIEVE_TOP_K: int = 5
    DEFAULT_RERANK_MODEL: str = "/data/h50056789/Rag_Chunking/model/BAAI/bge-reranker-v2-m3"
    DEFAULT_RERANK_DEVICE: str = "cuda:0"
    DEFAULT_RERANK_MAX_LENGTH: int | None = 8192

    # 检索生成（rag_generate_file）的线程池 worker 数（ThreadPoolExecutor, IO 密集型）
    RETRIEVAL_GENERATION_WORKERS: int = 4

    # Milvus / hybrid search
    MILVUS_ENABLE_SPARSE: bool = True
    MILVUS_ENABLE_HYBRID_SEARCH: bool = True
    MILVUS_HYBRID_RANKER: str = "RRFRanker"
    MILVUS_HYBRID_RANKER_K: int = 60

    # Traditional evaluation defaults
    DEFAULT_ENABLE_BERT_SCORE: bool = False
    DEFAULT_BERT_SCORE_MODEL: str = "roberta-large"
    DEFAULT_BERT_SCORE_DEVICE: str = "cuda:0"
    DEFAULT_ENABLE_LLM_JUDGE: bool = False

    # RAGAS evaluation runtime defaults
    DEFAULT_RAGAS_DEVICE: str = "cuda:0"
    DEFAULT_RAGAS_ENABLE_CACHE: bool = True
    DEFAULT_RAGAS_CACHE_DIR: str = "./ragas_cache"

    # Component evaluation defaults
    COMPONENT_ENABLE_SEMANTIC_SIMILARITY: bool = True
    COMPONENT_ENABLE_BOUNDARY_CLARITY: bool = True
    COMPONENT_SCORE_TEMPERATURE: float = 6.0

    # Stickiness evaluation defaults
    STICKINESS_THRESHOLD: float = 0.6
    STICKINESS_DELTA: float = 0.05
    STICKINESS_SCORE_TEMPERATURE: float = 6.0

    # 评估并行（ThreadPoolExecutor, IO 密集型; vLLM 默认 max-num-seqs=256，并发空间充足）
    EVAL_PARALLEL_WORKERS: int = 16  # 分块评估BC，CS，传统评估（LLM-Judge）线程池 worker 数
    EVAL_HTTP_TIMEOUT: int = 600
    RAGAS_MAX_WORKERS: int = 16  # RAGAS 评估的 asyncio 并发 worker 数 协程

    # Testset generation defaults
    TESTSET_LLM_BASE_URL: str = ""
    TESTSET_LLM_API_KEY: str = ""
    TESTSET_LLM_MODEL: str = ""
    TESTSET_LLM_MAX_TOKENS: int = 2048
    TESTSET_LLM_TIMEOUT: int = 3600
    TESTSET_EMBEDDING_MODEL_PATH: str = ""
    TESTSET_EMBEDDING_DEVICE: str = ""
    TESTSET_MAX_WORKERS: int = 18  # RAGAS testset 生成的 asyncio 并发 worker 数 协程
    TESTSET_MAX_RETRIES: int = 3
    TESTSET_MAX_WAIT_SECONDS: int = 3600
    TESTSET_RUN_TIMEOUT_SECONDS: int = 3600

    @property
    def python_paths(self) -> list[str]:
        return [str(BASE_DIR)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
