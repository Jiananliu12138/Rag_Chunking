from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from app.core.exceptions import (
    CollectionNotFoundException,
    IndexBuildException,
    ModelLoadException,
    to_http_exception,
)
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.common import BaseResponse
from app.schemas.index_schema import (
    CollectionListResult,
    IndexBuildRequest,
    IndexBuildResult,
)
from app.services.index_service import IndexService

router = APIRouter(prefix="/index", tags=["向量索引 / Index"])

_ERR_404 = {
    "description": "指定的 collection 不存在",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "Collection 'wiki_chunks' 不存在", "data": None}
        }
    },
}
_ERR_422 = {
    "description": "请求参数校验失败",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "chunks 列表不能为空", "data": None}
        }
    },
}
_ERR_500 = {
    "description": "索引构建或模型加载失败",
    "content": {
        "application/json": {
            "example": {"success": False, "message": "嵌入模型加载失败 (/path/to/model): ...", "data": None}
        }
    },
}


def _get_index_service() -> IndexService:
    return IndexService(MilvusRepository())


@router.post(
    "/build",
    response_model=BaseResponse[IndexBuildResult],
    summary="构建向量索引",
    description=(
        "将文本块列表嵌入为向量并写入 Milvus Lite 本地数据库。\n\n"
        "**存储规则**：每个 collection 对应 `milvus_data/<collection_name>.db` 文件。\n\n"
        "**参数说明**：\n"
        "- `overwrite=true`：先清空旧数据再写入（默认），适合重建索引\n"
        "- `overwrite=false`：追加写入，适合增量更新\n"
        "- `batch_size`：每批写入节点数，建议 100–500；过大可能导致内存问题\n\n"
        "**前置条件**：`embed_model_path` 指向本地 HuggingFace 嵌入模型目录（如 `bge-large-en-v1.5`）。"
    ),
    responses={422: _ERR_422, 500: _ERR_500},
)
def build_index(
    request: IndexBuildRequest,
    service: Annotated[IndexService, Depends(_get_index_service)],
) -> BaseResponse[IndexBuildResult]:
    try:
        result = service.build_index(request)
        return BaseResponse.ok(result, message="索引构建成功")
    except (IndexBuildException, ModelLoadException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/collections",
    response_model=BaseResponse[CollectionListResult],
    summary="列出所有 collection",
    description=(
        "扫描 `milvus_data/` 目录，返回所有已存在的 collection 及其文件大小。\n\n"
        "可用于确认索引是否已成功构建，或在检索前选择目标 collection。"
    ),
)
def list_collections(
    service: Annotated[IndexService, Depends(_get_index_service)],
) -> BaseResponse[CollectionListResult]:
    result = service.list_collections()
    return BaseResponse.ok(result)


@router.delete(
    "/collections/{collection_name}",
    response_model=BaseResponse[None],
    summary="删除指定 collection",
    description=(
        "删除对应的 Milvus Lite `.db` 文件，**操作不可逆**，删除后需重新构建索引。\n\n"
        "路径参数 `collection_name` 须与 `/index/build` 时传入的名称完全一致。"
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def delete_collection(
    collection_name: Annotated[str, Path(description="要删除的 collection 名称")],
    service: Annotated[IndexService, Depends(_get_index_service)],
) -> BaseResponse[None]:
    try:
        service.delete_collection(collection_name)
        return BaseResponse.ok(None, message=f"Collection '{collection_name}' 已删除")
    except CollectionNotFoundException as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
