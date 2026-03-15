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
    def list_roots() -> list[str]:
        if os.name == "nt":
            roots: list[str] = []
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    roots.append(drive)
            return roots or [str(Path.cwd().anchor or Path.cwd())]

        return [os.path.sep]

    @staticmethod
    def resolve_directory(path: str | None) -> Path:
        if path and str(path).strip():
            candidate = Path(str(path).strip()).expanduser()
            if candidate.is_file():
                return candidate.parent.resolve()
            return candidate.resolve()
        return Path.cwd().resolve()

    @staticmethod
    def list_directory(path: str | None = None) -> dict[str, Any]:
        directory = FileRepository.resolve_directory(path)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"路径不是目录: {directory}")

        entries: list[dict[str, Any]] = []
        for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                stat = item.stat()
                size_bytes = None if item.is_dir() else int(stat.st_size)
            except OSError:
                size_bytes = None
            entries.append(
                {
                    "name": item.name,
                    "path": str(item.resolve()),
                    "is_dir": item.is_dir(),
                    "size_bytes": size_bytes,
                }
            )

        parent_path = None
        if directory.parent != directory:
            parent_path = str(directory.parent.resolve())

        return {
            "current_path": str(directory),
            "parent_path": parent_path,
            "roots": FileRepository.list_roots(),
            "entries": entries,
        }

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if isinstance(item, str)]
        return []

    @staticmethod
    def _extract_context_texts(value: Any) -> list[str]:
        if isinstance(value, list):
            texts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
            return texts
        return []

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
    def parse_eval_results_from_json(data: Any) -> tuple[list[str], list[list[str]]]:
        """
        从评估结果 JSON 中提取 predictions 和 answers。
        
        输入格式：列表，每项为 {"llm_ans": str, "answers": list[str], ...}
        
        Returns:
            (predictions, answers) 元组
        """
        if not isinstance(data, list):
            raise ValueError(f"评估结果 JSON 应为列表格式，当前为 {type(data)}")
        
        predictions: list[str] = []
        answers: list[list[str]] = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
            llm_ans = item.get("llm_ans") or item.get("prediction") or ""
            ans_list = (
                FileRepository._normalize_string_list(item.get("answers"))
                or FileRepository._normalize_string_list(item.get("ground_truth"))
                or FileRepository._normalize_string_list(item.get("answer"))
            )
            if not ans_list:
                continue
            
            predictions.append(str(llm_ans))
            answers.append(ans_list)
        
        return predictions, answers

    @staticmethod
    def parse_ragas_dataset_from_json(data: Any) -> dict[str, list]:
        """
        从 JSON 中解析 RAGAS 数据集格式。
        
        支持两种输入格式：
        1. 标准 RAGAS 格式：{"question": [...], "answer": [...], "contexts": [...], "ground_truth": [...]}
        2. sample_results.json 格式：列表，每项为 {"input": str, "llm_ans": str, "answers": list[str], "retrieval_list": list[str]}
        
        Returns:
            RAGAS 格式的数据集字典
        """
        if isinstance(data, dict):
            # 格式1: 标准 RAGAS 格式
            required_keys = ["question", "answer", "contexts", "ground_truth"]
            if all(key in data for key in required_keys):
                return {
                    "question": [str(q) for q in data["question"]],
                    "answer": [str(a) for a in data["answer"]],
                    "contexts": [
                        [str(ctx) for ctx in ctx_list if isinstance(ctx, str)]
                        for ctx_list in data["contexts"]
                    ],
                    "ground_truth": [str(gt) for gt in data["ground_truth"]],
                }
            else:
                raise ValueError(f"标准 RAGAS 格式缺少必需字段，需要: {required_keys}")
        
        elif isinstance(data, list):
            # 格式2: sample_results.json 格式
            dataset = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": [],
            }
            for item in data:
                if not isinstance(item, dict):
                    continue
                question = item.get("input") or item.get("question") or ""
                answer = item.get("llm_ans") or item.get("answer") or ""
                contexts = (
                    FileRepository._extract_context_texts(item.get("retrieval_list"))
                    or FileRepository._extract_context_texts(item.get("contexts"))
                    or FileRepository._extract_context_texts(item.get("rag_retrieval"))
                )
                ground_truth_list = (
                    FileRepository._normalize_string_list(item.get("answers"))
                    or FileRepository._normalize_string_list(item.get("ground_truth"))
                    or FileRepository._normalize_string_list(item.get("answer"))
                )
                ground_truth = " ".join(ground_truth_list)
                
                dataset["question"].append(str(question))
                dataset["answer"].append(str(answer))
                dataset["contexts"].append([str(ctx) for ctx in contexts if isinstance(ctx, str)])
                dataset["ground_truth"].append(ground_truth)
            
            return dataset
        else:
            raise ValueError(f"不支持的 RAGAS 数据集格式: {type(data)}")

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
