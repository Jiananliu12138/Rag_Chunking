from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class IndexBuildRequest(BaseModel):
    """构建/重建索引用的请求体（从分块结果 JSON 文件构建）。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_name": "lumber_chunk",
                "docs_path": "/path/to/chunk_result.json",
                "batch_size": 100,
            }
        }
    )

    collection_name: str = Field(..., min_length=1, description="Milvus collection 名称")
    docs_path: str = Field(
        ...,
        min_length=1,
        description=(
            "分块结果 JSON 文件路径。内容需可解析出 chunk 文本，"
            "格式参考 eval/LongBench/base_lite.py::_parse_chunks_from_json。"
        ),
    )
    batch_size: int = Field(100, ge=1, le=5000, description="批量写入大小（每批写入的文本块数）")


class IndexAddRequest(BaseModel):
    """向已有索引追加数据的请求体。字段与构建索引相同。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_name": "lumber_chunk",
                "docs_path": "/path/to/new_chunk_result.json",
                "batch_size": 8000,
            }
        }
    )

    collection_name: str = Field(..., min_length=1, description="Milvus collection 名称")
    docs_path: str = Field(
        ...,
        min_length=1,
        description=(
            "分块结果 JSON 文件路径（将追加到已有 collection 中）。"
        ),
    )
    batch_size: int = Field(8000, ge=1, le=50000, description="批量追加大小（每批追加的文本块数）")


class IndexBuildResult(BaseModel):
    collection_name: str
    total_chunks: int
    indexed_chunks: int
    time_cost: float
    milvus_uri: str
    filepaths: list[str] = Field(
        default_factory=list,
        description="本次构建索引涉及的源文件路径列表（来自分块结果中的 filepath 元数据）",
    )
    doc_ids: list[int] = Field(
        default_factory=list,
        description="本次构建索引涉及的文档 ID 列表（来自分块结果中的 doc_id 元数据）",
    )


class IndexAddResult(BaseModel):
    collection_name: str
    added_chunks: int
    time_cost: float
    milvus_uri: str
    filepaths: list[str] = Field(
        default_factory=list,
        description="本次追加索引涉及的源文件路径列表（来自分块结果中的 filepath 元数据）",
    )
    doc_ids: list[int] = Field(
        default_factory=list,
        description="本次追加索引涉及的文档 ID 列表（来自分块结果中的 doc_id 元数据）",
    )


class CollectionInfo(BaseModel):
    name: str
    db_file: str
    size_bytes: int


class CollectionListResult(BaseModel):
    collections: list[CollectionInfo]
    total: int
