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
    embed_model_path: str = Field(..., description="嵌入模型路径")
    embed_dim: int = Field(1024, ge=64, description="嵌入向量维度")
    top_k: int = Field(5, ge=1, le=100, description="返回最相关文档数量")
    use_hybrid_search: Optional[bool] = Field(
        None,
        description="是否使用 Hybrid Search（dense+sparse）；未显式传时使用服务端默认配置。",
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
    filepath: Optional[str] = Field(None, description="该文本块的来源文件路径（如有）")
    doc_id: Optional[str] = Field(None, description="该文本块所属文档的 ID（如有）")


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
                "llm_api_base": "http://localhost:8005/v1",
                "llm_model_name": "/path/to/Qwen2.5-7B-Instruct",
                "temperature": 0.1,
                "max_new_tokens": 1280,
            }
        }
    )

    query: str = Field(..., min_length=1, description="用户问题")
    collection_name: str = Field(..., description="目标 collection 名称")
    embed_model_path: str = Field(..., description="嵌入模型路径")
    embed_dim: int = Field(1024, ge=64, description="嵌入向量维度")
    top_k: int = Field(5, ge=1, le=100, description="检索 top-k 数量")
    enable_rag: bool = Field(
        True,
        description="是否启用 RAG（向量检索+上下文拼接）。为 false 时仅调用 LLM，不做检索。",
    )
    use_hybrid_search: Optional[bool] = Field(
        None,
        description="是否使用 Hybrid Search（dense+sparse）；未显式传时使用服务端默认配置。",
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
    llm_api_base: str = Field("http://localhost:8005/v1", description="vLLM API 地址")
    llm_model_name: str = Field(..., description="LLM 模型名称/路径")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="生成温度，越低越稳定")
    max_new_tokens: int = Field(1280, ge=64, description="最大生成 token 数")


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
    output_path: str = Field(..., description="输出 JSON 文件路径，保存列表 [{_id, input, llm_ans, answers, retrieval_list}, ...]")
    collection_name: str = Field(..., description="目标 collection 名称")
    embed_model_path: Optional[str] = Field(None, description="嵌入模型路径，空则从配置读取")
    embed_dim: Optional[int] = Field(None, ge=64, description="嵌入维度，空则从配置读取")
    top_k: int = Field(5, ge=1, le=100, description="检索 top-k")
    llm_api_base: Optional[str] = Field(None, description="vLLM API 地址，空则从配置读取")
    llm_model_name: Optional[str] = Field(None, description="LLM 模型名，空则从配置读取")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="生成温度")
    max_new_tokens: int = Field(1280, ge=64, description="最大生成 token 数")


class RAGGenerateFileResult(BaseModel):
    output_file: str = Field(..., description="结果文件路径")
    total_processed: int = Field(..., description="成功处理条数")
    total_failed: int = Field(0, description="失败条数")
    message: str = Field(..., description="说明")
