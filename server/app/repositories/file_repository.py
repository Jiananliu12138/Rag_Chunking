"""
文件系统访问层。
负责分块结果、评估结果等中间文件的读写。
"""
import json
import os
from pathlib import Path
from typing import Any

from app.core.logging_config import logger


class FileRepository:

    @staticmethod
    def read_json(file_path: str) -> Any:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: str, data: Any, indent: int = 2) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info("结果已写入: %s", file_path)

    @staticmethod
    def read_jsonl(file_path: str) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def parse_chunks_from_json(data: Any) -> list[str]:
        """
        兼容现有流水线的多种 JSON 分块格式，统一返回字符串列表。
        支持格式：
          1. ["chunk1", "chunk2", ...]
          2. [["chunk1", label], ...]
          3. {"splits": [["chunk1", label], ...], ...}
          4. {"final_chunks": ["chunk1", ...]}
          5. [{"final_chunks": ["chunk1", ...]}, ...]
        """
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return data
            if data and isinstance(data[0], list):
                return [item[0] for item in data if isinstance(item, list) and item]
            if data and isinstance(data[0], dict):
                chunks: list[str] = []
                for item in data:
                    if "final_chunks" in item:
                        chunks.extend(item["final_chunks"])
                return chunks

        if isinstance(data, dict):
            if "splits" in data:
                splits = data["splits"]
                if splits and isinstance(splits[0], list):
                    return [s[0] for s in splits if s]
                if splits and isinstance(splits[0], str):
                    return splits
            if "final_chunks" in data:
                return data["final_chunks"]

        raise ValueError(f"不支持的分块 JSON 格式: {type(data)}")
