import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # F:\thesis\Meta-Chunking


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用基础 ──────────────────────────────────────────────────
    APP_TITLE: str = "Meta-Chunking RAG Server"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "基于 Meta-Chunking 流水线的 RAG 后端服务，"
        "提供分块、向量索引构建、检索生成及评估能力。"
    )
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── 路径 ─────────────────────────────────────────────────────
    MILVUS_DATA_DIR: str = "F:/thesis/Meta-Chunking/database"
    MILVUS_URI: str = ""
    CHUNKING_METHODS_DIR: str = str(BASE_DIR / "Chunking_Methods")
    EVAL_LONGBENCH_DIR: str = str(BASE_DIR / "eval" / "LongBench")
    MOC_METRICS_DIR: str = str(BASE_DIR / "component_eval" / "chunk")

    # ── 嵌入模型默认配置 ──────────────────────────────────────────
    DEFAULT_EMBEDDING_MODEL: str = ""
    DEFAULT_EMBEDDING_DIM: int = 1024

    # ── LLM 默认配置 ─────────────────────────────────────────────
    DEFAULT_VLLM_API_BASE: str = "http://localhost:8005/v1"
    DEFAULT_VLLM_MODEL_NAME: str = ""
    DEFAULT_LLM_TEMPERATURE: float = 0.1
    DEFAULT_LLM_MAX_TOKENS: int = 1280

    # ── 检索默认配置 ─────────────────────────────────────────────
    DEFAULT_RETRIEVE_TOP_K: int = 5

    # ── Milvus / Hybrid Search 配置 ─────────────────────────────────────────
    # 是否在 Milvus 中启用稀疏向量（BM25）能力
    MILVUS_ENABLE_SPARSE: bool = True
    # 是否在检索阶段启用 Hybrid Search（dense + sparse 一起用）
    MILVUS_ENABLE_HYBRID_SEARCH: bool = True
    # Hybrid Search 使用的 ranker 类型（RRFRanker 或 WeightedRanker）
    MILVUS_HYBRID_RANKER: str = "RRFRanker"
    # RRFRanker 的 k 参数（仅在使用 RRFRanker 时生效）
    MILVUS_HYBRID_RANKER_K: int = 60

    # ── 分块 / tiktoken 默认配置 ─────────────────────────────────
    # 从环境变量中读取对应值（.env 中写 TIKTOKEN_CACHE_DIR / CHUNK_NUM_WORKERS 即可）
    TIKTOKEN_CACHE_DIR: str = ""
    CHUNK_NUM_WORKERS: int = 4

    # ── 分块其他默认配置 ─────────────────────────────────────────
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_TOKEN_CHUNK_SIZE: int = 1200
    DEFAULT_TOKEN_OVERLAP: int = 100

    # ── 评估默认配置 ─────────────────────────────────────────────
    DEFAULT_ENABLE_BERT_SCORE: bool = False
    DEFAULT_BERT_SCORE_MODEL: str = "roberta-large"
    DEFAULT_BERT_SCORE_DEVICE: str = "cuda:0"
    
    # ── RAGAS 评估默认配置 ───────────────────────────────────────
    DEFAULT_RAGAS_VLLM_API_BASE: str = "http://localhost:8005/v1"
    DEFAULT_RAGAS_VLLM_API_KEY: str = "EMPTY"
    DEFAULT_RAGAS_VLLM_MODEL_NAME: str = ""
    DEFAULT_RAGAS_EMBEDDING_MODEL_PATH: str = ""
    DEFAULT_RAGAS_DEVICE: str = "cuda:0"
    DEFAULT_RAGAS_ENABLE_CACHE: bool = True
    DEFAULT_RAGAS_CACHE_DIR: str = "./ragas_cache"

    # ── 组件级评估默认配置 ───────────────────────────────────────
    # Chunk 质量评估（BC + 语义不相似度）
    COMPONENT_PPL_MODEL_PATH: str = ""
    COMPONENT_SIM_MODEL_PATH: str = ""
    COMPONENT_ENABLE_SEMANTIC_SIMILARITY: bool = True
    COMPONENT_ENABLE_BOUNDARY_CLARITY: bool = True

    # Chunk 黏连度评估（结构熵）
    STICKINESS_MODEL_PATH: str = ""
    STICKINESS_THRESHOLD: float = 0.8
    STICKINESS_DELTA: float = 0.0

    @property
    def python_paths(self) -> list[str]:
        return [
            self.EVAL_LONGBENCH_DIR,
            self.CHUNKING_METHODS_DIR,
            self.MOC_METRICS_DIR,
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
