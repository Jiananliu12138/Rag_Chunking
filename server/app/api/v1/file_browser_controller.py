from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.repositories.file_repository import FileRepository
from app.schemas.common import BaseResponse
from app.schemas.file_browser_schema import FileBrowserResult

router = APIRouter(prefix="/files", tags=["System / Files"])


@router.get(
    "/browse",
    response_model=BaseResponse[FileBrowserResult],
    summary="Browse server filesystem",
    description=(
        "Browse directories on the machine running the backend service and return "
        "absolute paths for files and folders."
    ),
)
def browse_files(
    path: Annotated[str | None, Query(description="Optional file or directory path")] = None,
) -> BaseResponse[FileBrowserResult]:
    try:
        result = FileRepository.list_directory(path)
        return BaseResponse.ok(FileBrowserResult(**result))
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
