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
    docs_paths: Optional[list[str]] = Field(
        None,
        description=(
            "可选：一次性构建多个分块结果文件到同一个 collection。"
            "若提供 docs_paths，则会依次读取并合并所有文件；docs_path 字段将被忽略。"
        ),
    )
    batch_size: int = Field(100, ge=1, le=5000, description="批量写入大小（每批写入的文本块数）")
    enable_sparse: Optional[bool] = Field(
        None,
        description="是否为该 collection 启用稀疏向量（BM25）；为空则使用服务端默认配置。",
    )
    embed_model_path: Optional[str] = Field(
        None,
        description=(
            "可选：覆盖默认的嵌入模型路径。未提供时使用配置 DEFAULT_EMBEDDING_MODEL。"
        ),
    )
    embed_dim: Optional[int] = Field(
        None,
        ge=64,
        description=(
            "可选：覆盖默认的嵌入维度。未提供时使用配置 DEFAULT_EMBEDDING_DIM。"
        ),
    )


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
    docs_paths: Optional[list[str]] = Field(
        None,
        description=(
            "可选：一次性向同一个 collection 追加多个分块结果文件。"
            "若提供 docs_paths，则会依次读取并合并所有文件；docs_path 字段将被忽略。"
        ),
    )
    batch_size: int = Field(8000, ge=1, le=50000, description="批量追加大小（每批追加的文本块数）")
    embed_model_path: Optional[str] = Field(
        None,
        description=(
            "可选：覆盖默认的嵌入模型路径。未提供时使用配置 DEFAULT_EMBEDDING_MODEL。"
            "请确保与构建该 collection 时使用的模型/维度保持一致。"
        ),
    )


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
    doc_ids: list[str] = Field(
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
    doc_ids: list[str] = Field(
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


class CollectionInspectItem(BaseModel):
    collection_name: str
    uri: str
    db_file: str
    size_bytes: int
    schema: Optional[dict] = Field(
        None,
        description="Milvus describe_collection 返回的 schema 信息（可能为 None）。",
    )
    predefined_fields: list[str] = Field(
        default_factory=list,
        description="MilvusVectorStore 预定义的字段名列表（id/doc_id/text/embedding/sparse_embedding 等）。",
    )
    dynamic_fields: list[str] = Field(
        default_factory=list,
        description="从示例数据中推断出的动态元数据字段名列表（如 filepath、source_doc_id 等）。",
    )


class CollectionInspectResult(BaseModel):
    collections: list[CollectionInspectItem]
    total: int


class IndexDeleteByMetadataRequest(BaseModel):
    """按 metadata 条件删除部分向量的请求体。"""

    filepath: Optional[str] = Field(
        None,
        description="可选：按来源文件删除，仅删除该 filepath 对应的文本块向量。",
    )
    doc_ids: Optional[list[str]] = Field(
        None,
        description="可选：按 doc_id 列表删除，仅删除这些文档 ID 对应的文本块向量。",
    )
