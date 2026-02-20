import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # F:\thesis\Meta-Chunking


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/data/h50056789/Rag_Chunking/.env",
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
    MILVUS_DATA_DIR: str = str(BASE_DIR / "milvus_data")
    MILVUS_URI: str = ""
    CHUNKING_METHODS_DIR: str = str(BASE_DIR / "Chunking_Methods")
    EVAL_LONGBENCH_DIR: str = str(BASE_DIR / "eval" / "LongBench")
    MOC_METRICS_DIR: str = str(BASE_DIR / "MoC" / "our_metrics")

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

    # ── 分块 / tiktoken 默认配置 ─────────────────────────────────
    # 从环境变量中读取对应值（.env 中写 TIKTOKEN_CACHE_DIR / CHUNK_NUM_WORKERS 即可）
    TIKTOKEN_CACHE_DIR: str = ""
    CHUNK_NUM_WORKERS: int = 4

    # ── 分块其他默认配置 ─────────────────────────────────────────
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_TOKEN_CHUNK_SIZE: int = 1200
    DEFAULT_TOKEN_OVERLAP: int = 100

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
