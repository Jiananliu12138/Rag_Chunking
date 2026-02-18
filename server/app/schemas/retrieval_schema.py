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


class SearchResultItem(BaseModel):
    text: str = Field(..., description="检索到的文本内容")
    score: Optional[float] = Field(None, description="相似度分数（0-1，越高越相关）")


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
    llm_api_base: str = Field("http://localhost:8005/v1", description="vLLM API 地址")
    llm_model_name: str = Field(..., description="LLM 模型名称/路径")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="生成温度，越低越稳定")
    max_new_tokens: int = Field(1280, ge=64, description="最大生成 token 数")


class RAGResult(BaseModel):
    query: str
    answer: str
    contexts: list[str] = Field(..., description="检索到的上下文文本列表")
    collection_name: str
