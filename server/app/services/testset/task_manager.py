"""In-memory async task manager for long-running testset generation."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    stage: str = ""
    progress_pct: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_tasks: Dict[str, TaskState] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="testset")


def create_task() -> TaskState:
    task_id = uuid.uuid4().hex[:12]
    state = TaskState(task_id=task_id)
    with _lock:
        _tasks[task_id] = state
    return state


def get_task(task_id: str) -> Optional[TaskState]:
    with _lock:
        return _tasks.get(task_id)


def update_progress(task_id: str, stage: str, pct: int, message: str = "") -> None:
    with _lock:
        state = _tasks.get(task_id)
        if state:
            state.status = TaskStatus.RUNNING
            state.stage = stage
            state.progress_pct = min(pct, 100)
            state.message = message


def complete_task(task_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        state = _tasks.get(task_id)
        if state:
            state.status = TaskStatus.COMPLETED
            state.stage = "completed"
            state.progress_pct = 100
            state.result = result


def fail_task(task_id: str, error: str) -> None:
    with _lock:
        state = _tasks.get(task_id)
        if state:
            state.status = TaskStatus.FAILED
            state.stage = "failed"
            state.error = error


def submit_task(task_id: str, fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Submit a function to run in the background thread pool."""

    def _wrapper():
        try:
            result = fn(*args, **kwargs)
            complete_task(task_id, result)
        except Exception as exc:
            fail_task(task_id, str(exc))

    update_progress(task_id, "queued", 0, "Task queued")
    _executor.submit(_wrapper)
