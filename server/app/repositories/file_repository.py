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
        支持格式与 index_service 原 _parse_chunks_from_json 一致。
        """
        chunks: list[str] = []

        # 1. ["chunk1", "chunk2", ...]
        if isinstance(data, list) and data and isinstance(data[0], str):
            chunks = [chunk for chunk in data if isinstance(chunk, str)]

        # 2. [["chunk1", label1], ["chunk2", label2], ...]
        elif isinstance(data, list) and data and isinstance(data[0], list):
            for item in data:
                if isinstance(item, list) and item:
                    text = item[0]
                    if isinstance(text, str):
                        chunks.append(text)

        # 3/4. {"splits": [...]} 或 {"final_chunks": [...]}
        elif isinstance(data, dict):
            if "splits" in data:
                splits = data["splits"]
                if isinstance(splits, list) and splits:
                    if isinstance(splits[0], list):
                        for item in splits:
                            if isinstance(item, list) and item:
                                text = item[0]
                                if isinstance(text, str):
                                    chunks.append(text)
                    elif isinstance(splits[0], str):
                        chunks = [chunk for chunk in splits if isinstance(chunk, str)]

            elif "final_chunks" in data and isinstance(data["final_chunks"], list):
                chunks = [
                    chunk for chunk in data["final_chunks"] if isinstance(chunk, str)
                ]

        # 5. [{"name": "...", "final_chunks": [...]}, ...]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            for item in data:
                if "final_chunks" in item and isinstance(item["final_chunks"], list):
                    for chunk in item["final_chunks"]:
                        if isinstance(chunk, str):
                            chunks.append(chunk)

        else:
            raise ValueError(f"不支持的分块 JSON 格式: {type(data)}")

        return chunks
