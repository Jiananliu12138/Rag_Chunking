from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the transformer architecture in deep learning?",
                "collection_name": "wiki_chunks",
                "embed_model_path": "/path/to/bge-large-en-v1.5",
                "embed_dim": 1024,
                "top_k": 5,
            }
        }
    )

    query: str = Field(..., min_length=1, description="检索查询文本")
    collection_name: str = Field(..., description="目标 collection 名称")
    embed_model_path: Optional[str] = Field(None, description="嵌入模型路径，空则从配置读取")
    embed_dim: Optional[int] = Field(None, ge=64, description="嵌入向量维度，空则从配置读取")
    embed_max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖嵌入模型 max_seq_length；未提供时使用 DEFAULT_EMBEDDING_MAX_TOKENS。",
    )
    top_k: int = Field(5, ge=1, le=100, description="返回最相关文档数量")
    use_hybrid_search: Optional[bool] = Field(
        None,
        description="是否使用 Hybrid Search（dense+sparse）；未显式传时使用服务端默认配置。",
    )
    rerank_enabled: bool = Field(
        False,
        description="是否启用 rerank（默认 false）。",
    )
    rerank_type: str = Field(
        "cross_encoder",
        description="rerank 类型，支持 cross_encoder（本地）或 vllm（远端 API）。",
    )
    rerank_model_path: Optional[str] = Field(
        None,
        description="本地 cross_encoder 模型路径；在 vllm 模式下也可作为 rerank_model_name 的回退值。",
    )
    rerank_device: Optional[str] = Field(
        None,
        description="rerank 设备（如 cpu / cuda:0），空则从配置读取。",
    )
    rerank_max_length: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖 CrossEncoder 的 max_length；未提供时使用 DEFAULT_RERANK_MAX_LENGTH。",
    )
    rerank_candidate_k: Optional[int] = Field(
        None,
        ge=1,
        le=200,
        description="rerank 召回候选数；为空时使用 max(top_k, rerank_top_k)。",
    )
    rerank_top_k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="rerank 后保留条数；为空时使用 top_k。",
    )
    rerank_api_base: Optional[str] = Field(
        None,
        description="vLLM rerank 服务地址（例如 http://localhost:8002/v1）；仅 rerank_type=vllm 时使用。",
    )
    rerank_api_key: Optional[str] = Field(
        None,
        description="vLLM rerank 服务 API Key；为空时回退到服务端默认配置。",
    )
    rerank_model_name: Optional[str] = Field(
        None,
        description="vLLM rerank 模型名；仅 rerank_type=vllm 时使用。",
    )
    filepath: str | list[str] | None = Field(
        None,
        description=(
            "可选：按来源文件过滤，仅检索来自这些 filepath 的文本块。"
            "既支持单个字符串，也支持字符串列表。"
        ),
    )
    doc_id: str | list[str] | None = Field(
        None,
        description=(
            "可选：按 doc_id 过滤，仅检索指定文档 ID 的文本块。"
            "既支持单个字符串，也支持字符串列表。"
        ),
    )


class SearchResultItem(BaseModel):
    text: str = Field(..., description="检索到的文本内容")
    score: Optional[float] = Field(None, description="相似度分数（0-1，越高越相关）")
    retrieval_score: Optional[float] = Field(None, description="原始检索分数（rerank 前）")
    rerank_score: Optional[float] = Field(None, description="重排分数（rerank 后）")
    filepath: Optional[str] = Field(None, description="该文本块的来源文件路径（如有）")
    doc_id: Optional[str] = Field(None, description="该文本块所属文档的 ID（如有）")
    chunk_id: Optional[str] = Field(None, description="该文本块的分块 ID（如有）")


class SearchResult(BaseModel):
    query: str
    results: list[SearchResultItem]
    collection_name: str
    top_k: int


class RAGRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the main advantages of transformer models?",
                "collection_name": "wiki_chunks",
                "embed_model_path": "/path/to/bge-large-en-v1.5",
                "embed_dim": 1024,
                "top_k": 5,
                "llm_api_base": "http://localhost:8001/v1",
                "llm_model_name": "/path/to/Qwen2.5-7B-Instruct",
                "temperature": 0.1,
                "max_new_tokens": 1280,
            }
        }
    )

    query: str = Field(..., min_length=1, description="用户问题")
    collection_name: str = Field(..., description="目标 collection 名称")
    embed_model_path: Optional[str] = Field(None, description="嵌入模型路径，空则从配置读取")
    embed_dim: Optional[int] = Field(None, ge=64, description="嵌入向量维度，空则从配置读取")
    embed_max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖嵌入模型 max_seq_length；未提供时使用 DEFAULT_EMBEDDING_MAX_TOKENS。",
    )
    top_k: int = Field(5, ge=1, le=100, description="检索 top-k 数量")
    enable_rag: bool = Field(
        True,
        description="是否启用 RAG（向量检索+上下文拼接）。为 false 时仅调用 LLM，不做检索。",
    )
    use_hybrid_search: Optional[bool] = Field(
        None,
        description="是否使用 Hybrid Search（dense+sparse）；未显式传时使用服务端默认配置。",
    )
    rerank_enabled: bool = Field(
        False,
        description="是否启用 rerank（默认 false）。",
    )
    rerank_type: str = Field(
        "cross_encoder",
        description="rerank 类型，支持 cross_encoder（本地）或 vllm（远端 API）。",
    )
    rerank_model_path: Optional[str] = Field(
        None,
        description="本地 cross_encoder 模型路径；在 vllm 模式下也可作为 rerank_model_name 的回退值。",
    )
    rerank_device: Optional[str] = Field(
        None,
        description="rerank 设备（如 cpu / cuda:0），空则从配置读取。",
    )
    rerank_max_length: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖 CrossEncoder 的 max_length；未提供时使用 DEFAULT_RERANK_MAX_LENGTH。",
    )
    rerank_candidate_k: Optional[int] = Field(
        None,
        ge=1,
        le=200,
        description="rerank 召回候选数；为空时使用 max(top_k, rerank_top_k)。",
    )
    rerank_top_k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="rerank 后保留条数；为空时使用 top_k。",
    )
    rerank_api_base: Optional[str] = Field(
        None,
        description="vLLM rerank 服务地址（例如 http://localhost:8002/v1）；仅 rerank_type=vllm 时使用。",
    )
    rerank_api_key: Optional[str] = Field(
        None,
        description="vLLM rerank 服务 API Key；为空时回退到服务端默认配置。",
    )
    rerank_model_name: Optional[str] = Field(
        None,
        description="vLLM rerank 模型名；仅 rerank_type=vllm 时使用。",
    )
    filepath: str | list[str] | None = Field(
        None,
        description=(
            "可选：按来源文件过滤，仅检索来自这些 filepath 的文本块。"
            "既支持单个字符串，也支持字符串列表。"
        ),
    )
    doc_id: str | list[str] | None = Field(
        None,
        description=(
            "可选：按 doc_id 过滤，仅检索指定文档 ID 的文本块。"
            "既支持单个字符串，也支持字符串列表。"
        ),
    )
    llm_api_base: Optional[str] = Field(None, description="vLLM API 地址，空则从配置读取")
    llm_model_name: Optional[str] = Field(None, description="LLM 模型名称/路径，空则从配置读取")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="生成温度，空则从配置读取")
    max_new_tokens: Optional[int] = Field(None, ge=64, description="最大生成 token 数，空则从配置读取")


class RAGResult(BaseModel):
    query: str
    answer: str
    contexts: list[str] = Field(..., description="检索到的上下文纯文本列表（用于拼接到 Prompt）")
    context_items: list[SearchResultItem] = Field(
        ...,
        description="包含分数与元数据的上下文列表（与 /retrieval/search 的结果项结构一致）",
    )
    collection_name: str


class RAGGenerateFileRequest(BaseModel):
    """文件输入输出的 RAG 生成：读 jsonl，逐条检索+生成，结果写 json。"""
    input_path: str = Field(..., description="输入 jsonl 路径，每行 JSON 含 input（查询）及 _id、answers 等")
    output_path: str = Field(
        ...,
        description=(
            "输出 JSON 文件路径，保存列表 "
            "[{_id, question_id, input, llm_ans, answer, rag_retrieval, gold_reference, ...}, ...]"
        ),
    )
    collection_name: str = Field(..., description="目标 collection 名称")
    embed_model_path: Optional[str] = Field(None, description="嵌入模型路径，空则从配置读取")
    embed_dim: Optional[int] = Field(None, ge=64, description="嵌入维度，空则从配置读取")
    embed_max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖嵌入模型 max_seq_length；未提供时使用 DEFAULT_EMBEDDING_MAX_TOKENS。",
    )
    top_k: int = Field(5, ge=1, le=100, description="检索 top-k")
    use_hybrid_search: Optional[bool] = Field(
        None,
        description="是否使用 Hybrid Search（dense+sparse）；未显式传时使用服务端默认配置。",
    )
    rerank_enabled: bool = Field(
        False,
        description="是否启用 rerank（默认 false）。",
    )
    rerank_type: str = Field(
        "cross_encoder",
        description="rerank 类型，支持 cross_encoder（本地）或 vllm（远端 API）。",
    )
    rerank_model_path: Optional[str] = Field(
        None,
        description="本地 cross_encoder 模型路径；在 vllm 模式下也可作为 rerank_model_name 的回退值。",
    )
    rerank_device: Optional[str] = Field(
        None,
        description="rerank 设备（如 cpu / cuda:0），空则从配置读取。",
    )
    rerank_max_length: Optional[int] = Field(
        None,
        ge=1,
        description="可选：覆盖 CrossEncoder 的 max_length；未提供时使用 DEFAULT_RERANK_MAX_LENGTH。",
    )
    rerank_candidate_k: Optional[int] = Field(
        None,
        ge=1,
        le=200,
        description="rerank 召回候选数；为空时使用 max(top_k, rerank_top_k)。",
    )
    rerank_top_k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="rerank 后保留条数；为空时使用 top_k。",
    )
    rerank_api_base: Optional[str] = Field(
        None,
        description="vLLM rerank 服务地址（例如 http://localhost:8002/v1）；仅 rerank_type=vllm 时使用。",
    )
    rerank_api_key: Optional[str] = Field(
        None,
        description="vLLM rerank 服务 API Key；为空时回退到服务端默认配置。",
    )
    rerank_model_name: Optional[str] = Field(
        None,
        description="vLLM rerank 模型名；仅 rerank_type=vllm 时使用。",
    )
    llm_api_base: Optional[str] = Field(None, description="vLLM API 地址，空则从配置读取")
    llm_model_name: Optional[str] = Field(None, description="LLM 模型名，空则从配置读取")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="生成温度，空则从配置读取")
    max_new_tokens: Optional[int] = Field(None, ge=64, description="最大生成 token 数，空则从配置读取")
    max_workers: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="并行工作线程数。未指定时使用 RAG_FILE_MAX_WORKERS 配置（默认 4）。",
    )


class RAGGenerateFileResult(BaseModel):
    output_file: str = Field(..., description="结果文件路径")
    total_processed: int = Field(..., description="成功处理条数")
    total_failed: int = Field(0, description="失败条数")
    message: str = Field(..., description="说明")
