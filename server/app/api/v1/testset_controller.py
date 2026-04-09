from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import EvaluationException, to_http_exception
from app.schemas.common import BaseResponse
from app.schemas.testset_schema import (
    TestsetGenerateFileRequest,
    TestsetGenerateRequest,
    TestsetGenerateResult,
    TestsetTaskStatus,
    TestsetTaskSubmitted,
)
from app.services.testset import TestsetService
from app.services.testset.task_manager import (
    create_task,
    get_task,
    submit_task,
    update_progress,
)

router = APIRouter(prefix="/testset", tags=["测试集生成 / Testset Generation"])


def _get_service() -> TestsetService:
    return TestsetService()


def _run_generate(service: TestsetService, request, task_id: str):
    """Wrapper that wires task_manager progress updates into service.generate."""

    def on_progress(stage: str, pct: int, message: str = ""):
        update_progress(task_id, stage, pct, message)

    return service.generate(request, on_progress=on_progress)


@router.post(
    "/generate",
    response_model=BaseResponse[TestsetTaskSubmitted],
    summary="提交 QA 测试集生成任务",
    description=(
        "异步提交生成任务，立即返回 task_id。\n"
        "通过 GET /testset/tasks/{task_id} 轮询进度和结果。"
    ),
)
def generate_testset(
    request: TestsetGenerateRequest,
    service: Annotated[TestsetService, Depends(_get_service)],
) -> BaseResponse[TestsetTaskSubmitted]:
    # Validate chunks file exists before submitting
    from pathlib import Path

    if not Path(request.chunks_file).exists():
        raise HTTPException(status_code=404, detail=f"Chunks file not found: {request.chunks_file}")

    task = create_task()
    submit_task(task.task_id, _run_generate, service, request, task.task_id)
    return BaseResponse.ok(TestsetTaskSubmitted(task_id=task.task_id))


@router.post(
    "/generate-file",
    response_model=BaseResponse[TestsetTaskSubmitted],
    summary="提交 QA 测试集生成任务（从文件）",
    description="功能与 /generate 相同，schema 独立以便前端区分调用场景。",
)
def generate_testset_file(
    request: TestsetGenerateFileRequest,
    service: Annotated[TestsetService, Depends(_get_service)],
) -> BaseResponse[TestsetTaskSubmitted]:
    from pathlib import Path

    if not Path(request.chunks_file).exists():
        raise HTTPException(status_code=404, detail=f"Chunks file not found: {request.chunks_file}")

    task = create_task()
    submit_task(task.task_id, _run_generate, service, request, task.task_id)
    return BaseResponse.ok(TestsetTaskSubmitted(task_id=task.task_id))


@router.get(
    "/tasks/{task_id}",
    response_model=BaseResponse[TestsetTaskStatus],
    summary="查询生成任务进度",
)
def get_task_status(task_id: str) -> BaseResponse[TestsetTaskStatus]:
    state = get_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    result_data = None
    if state.result is not None:
        result_data = TestsetGenerateResult(**state.result)

    return BaseResponse.ok(
        TestsetTaskStatus(
            task_id=state.task_id,
            status=state.status.value,
            stage=state.stage,
            progress_pct=state.progress_pct,
            message=state.message,
            result=result_data,
            error=state.error,
        )
    )
