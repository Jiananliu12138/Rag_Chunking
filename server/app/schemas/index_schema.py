from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class IndexBuildRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_name": "wiki_chunks",
                "chunks": [
                    "Natural language processing is a subfield of AI...",
                    "Deep learning models have achieved state-of-the-art results...",
                    "The transformer architecture was introduced in 2017...",
                ],
                "embed_model_path": "/path/to/bge-large-en-v1.5",
                "embed_dim": 1024,
                "overwrite": True,
                "batch_size": 100,
            }
        }
    )

    collection_name: str = Field(..., min_length=1, description="Milvus collection 名称")
    chunks: list[str] = Field(..., min_length=1, description="待索引的文本块列表")
    embed_model_path: str = Field(..., description="嵌入模型路径")
    embed_dim: int = Field(1024, ge=64, description="嵌入向量维度")
    overwrite: bool = Field(True, description="是否覆盖已有 collection")
    batch_size: int = Field(100, ge=1, le=5000, description="批量写入大小")


class IndexBuildResult(BaseModel):
    collection_name: str
    total_chunks: int
    indexed_chunks: int
    time_cost: float
    milvus_uri: str


class CollectionInfo(BaseModel):
    name: str
    db_file: str
    size_bytes: int


class CollectionListResult(BaseModel):
    collections: list[CollectionInfo]
    total: int
