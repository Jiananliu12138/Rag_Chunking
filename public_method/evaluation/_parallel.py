"""并发执行工具，用于评估流水线。

评估任务大多是 vLLM HTTP 调用（requests.post 期间释放 GIL）以及 PyTorch
推理调用（前向传播也释放 GIL），都适合用线程池并发。

`parallel_map` 保证返回顺序与输入顺序一致，且把 worker 中的异常抛到调用方。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> List[R]:
    """对 ``items`` 并发执行 ``fn``，按输入顺序返回结果列表。

    Args:
        fn: 单元素处理函数。
        items: 待处理元素序列。
        max_workers: 线程数上限；<=1 时退化为串行。
        desc: tqdm 进度条描述。
        show_progress: 是否显示进度条。
    """
    materialised = list(items)
    if not materialised:
        return []

    workers = max(1, int(max_workers))
    if workers == 1 or len(materialised) == 1:
        iterator: Iterable[T] = materialised
        if show_progress:
            iterator = tqdm(materialised, desc=desc)
        return [fn(item) for item in iterator]

    results: List[Optional[R]] = [None] * len(materialised)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(fn, item): idx
            for idx, item in enumerate(materialised)
        }
        progress = tqdm(total=len(materialised), desc=desc) if show_progress else None
        try:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    return results  # type: ignore[return-value]
