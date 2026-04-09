"""Testset generation request / response schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class TestsetGenerateRequest(BaseModel):
    """POST /api/v1/testset/generate body."""

    chunks_file: str = Field(..., description="Server-side path to chunks file (json/jsonl/txt)")
    output_file: Optional[str] = Field(None, description="Output JSONL path; auto-derived if omitted")
    testset_size: int = Field(10, ge=1, description="Number of QA pairs to generate")

    # Run config
    max_workers: Optional[int] = Field(None, description="Parallel workers (default from settings)")
    max_retries: Optional[int] = Field(None)
    max_wait_seconds: Optional[int] = Field(None)
    run_timeout_seconds: Optional[int] = Field(None)
    fail_fast: bool = Field(False, description="Raise on first generation failure")
    with_debugging_logs: bool = False
    skip_custom_node_filter: bool = Field(True, description="Use custom transforms (recommended)")

    # LLM config (optional, falls back to settings -> defaults)
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_max_tokens: Optional[int] = None
    llm_timeout: Optional[int] = None

    # Embedding config (optional, falls back to settings -> defaults)
    embedding_model_path: Optional[str] = None
    embedding_device: Optional[str] = None


class TestsetGenerateFileRequest(BaseModel):
    """POST /api/v1/testset/generate-file — same as above but chunks_file is required."""

    chunks_file: str = Field(..., description="Server-side path to chunks file")
    output_file: Optional[str] = None
    testset_size: int = Field(10, ge=1)

    max_workers: Optional[int] = None
    max_retries: Optional[int] = None
    max_wait_seconds: Optional[int] = None
    run_timeout_seconds: Optional[int] = None
    fail_fast: bool = False
    with_debugging_logs: bool = False
    skip_custom_node_filter: bool = True

    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_max_tokens: Optional[int] = None
    llm_timeout: Optional[int] = None

    embedding_model_path: Optional[str] = None
    embedding_device: Optional[str] = None


class TestsetGenerateResult(BaseModel):
    """Response payload for testset generation."""

    output_file: str
    failed_file: Optional[str] = None
    total_generated: int
    total_failed: int
    testset_size_requested: int
    samples_preview: list[Any] = Field(default_factory=list, description="First 5 QA pairs")


class TestsetTaskSubmitted(BaseModel):
    """Response when a testset generation task is submitted."""

    task_id: str


class TestsetTaskStatus(BaseModel):
    """Response for polling task progress."""

    task_id: str
    status: str  # pending | running | completed | failed
    stage: str = ""
    progress_pct: int = 0
    message: str = ""
    result: Optional[TestsetGenerateResult] = None
    error: Optional[str] = None
