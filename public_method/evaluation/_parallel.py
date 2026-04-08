"""并发执行工具，用于评估流水线。

评估任务大多是 vLLM HTTP 调用（requests.post 期间释放 GIL）以及 PyTorch
推理调用（前向传播也释放 GIL），都适合用线程池并发。

`parallel_map` 保证返回顺序与输入顺序一致，且把 worker 中的异常抛到调用方。

进度反馈说明：
- 交互式终端用 tqdm 进度条（默认开启）
- server / docker / 非 tty 环境下 tqdm 的 \r 刷新会被 line-based 日志收集器吞掉，
  这种场景应该传入一个 ``logger`` 让 ``parallel_map`` 走 logging 走 stdout 输出周期性进度
"""

from __future__ import annotations

import logging
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
    logger: Optional[logging.Logger] = None,
    log_every: Optional[int] = None,
) -> List[R]:
    """对 ``items`` 并发执行 ``fn``，按输入顺序返回结果列表。

    Args:
        fn: 单元素处理函数。
        items: 待处理元素序列。
        max_workers: 线程数上限；<=1 时退化为串行。
        desc: 进度描述（同时用于 tqdm 和 logger 输出前缀）。
        show_progress: 是否显示 tqdm 进度条（仅在 tty 下能正常显示）。
        logger: 可选 logger；传入后会周期性 log.info 输出进度，适合非 tty 环境。
        log_every: 多少个完成后记一次日志；不传则按 ~5% 自动取。
    """
    materialised = list(items)
    total = len(materialised)
    if total == 0:
        return []

    workers = max(1, int(max_workers))

    if logger is not None and log_every is None:
        log_every = max(1, total // 20)  # ~5%

    label = desc or "parallel_map"

    def _log_progress(done: int) -> None:
        if logger is None or not log_every:
            return
        if done == total or done % log_every == 0:
            logger.info("%s 进度 %d/%d", label, done, total)

    if workers == 1 or total == 1:
        if logger is not None:
            logger.info("%s 启动: total=%d, workers=1", label, total)
        iterator: Iterable[T]
        if show_progress:
            iterator = tqdm(materialised, desc=desc)
        else:
            iterator = materialised
        results_serial: List[R] = []
        for idx, item in enumerate(iterator, start=1):
            results_serial.append(fn(item))
            _log_progress(idx)
        return results_serial

    if logger is not None:
        logger.info("%s 启动: total=%d, workers=%d", label, total, workers)

    results: List[Optional[R]] = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(fn, item): idx
            for idx, item in enumerate(materialised)
        }
        progress = tqdm(total=total, desc=desc) if show_progress else None
        completed = 0
        try:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
                completed += 1
                if progress is not None:
                    progress.update(1)
                _log_progress(completed)
        finally:
            if progress is not None:
                progress.close()
    return results  # type: ignore[return-value]
