"""
组件级分块评估接口测试脚本。

覆盖接口：
- POST /component-eval/chunk-quality
- POST /component-eval/chunk-quality-file
- POST /component-eval/chunk-stickiness
- POST /component-eval/chunk-stickiness-file

使用方式（先启动服务）：

    cd F:\thesis\Meta-Chunking\server
    uvicorn app.main:app --reload --port 8080

在项目根目录执行：

    python server/test_script/test_component_eval.py

环境变量：
    META_CHUNKING_BASE_URL  默认 http://localhost:8080/api/v1
    COMPONENT_PPL_MODEL_PATH / COMPONENT_SIM_MODEL_PATH / STICKINESS_MODEL_PATH  请在 .env 中配置
"""

import json
import os
from typing import Any, Dict, List

import requests

BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 获取项目根目录（server/test_script/.. -> server/.. -> 项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 默认分块结果文件路径（可按需修改）
DEFAULT_CHUNK_JSON_PATH = os.path.join(PROJECT_ROOT, "test_script", "sample_chunks.json")


def _print_response(resp: requests.Response) -> None:
    """辅助打印 HTTP 响应。"""
    print("=" * 80)
    print(f"URL      : {resp.request.method} {resp.url}")
    print(f"Status   : {resp.status_code}")
    print("Request  :")
    try:
        if resp.request.body:
            body = (
                resp.request.body.decode("utf-8")
                if isinstance(resp.request.body, bytes)
                else resp.request.body
            )
            if len(body) > 800:
                print(body[:800] + "...")
            else:
                print(body)
    except Exception:
        print("<无法解析请求体>")

    print("Response :")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text)
    print("=" * 80)
    print()


def ensure_sample_chunks_json(path: str = DEFAULT_CHUNK_JSON_PATH) -> str:
    """如不存在，写一个简单的分块结果 JSON 文件，便于 file 接口测试."""
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 使用与 FileRepository.parse_chunks_from_json 兼容的几种格式之一
    sample_chunks: List[str] = [
        "Chunk 1: The transformer architecture was introduced in the paper 'Attention is All You Need'.",
        "Chunk 2: BERT is based on the transformer encoder and trained with masked language modeling.",
        "Chunk 3: GPT uses the transformer decoder for autoregressive language generation.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_chunks, f, ensure_ascii=False, indent=2)
    print(f"已创建示例分块文件: {path}\n")
    return path


def test_chunk_quality_direct() -> None:
    """测试 /component-eval/chunk-quality —— 直接传 chunks。"""
    url = f"{BASE_URL}/component-eval/chunk-quality"
    payload: Dict[str, Any] = {
        "chunks": [
            "Chunk 1: The transformer architecture was introduced in the paper 'Attention is All You Need'.",
            "Chunk 2: BERT is based on the transformer encoder and trained with masked language modeling.",
            "Chunk 3: GPT uses the transformer decoder for autoregressive language generation.",
        ],
        # 这两个开关可选，不传则走配置 COMPONENT_ENABLE_SEMANTIC_SIMILARITY / COMPONENT_ENABLE_BOUNDARY_CLARITY
        "enable_semantic_similarity": True,
        "enable_boundary_clarity": True,
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


def test_chunk_quality_file() -> None:
    """测试 /component-eval/chunk-quality-file —— 从分块文件读取。"""
    input_path = ensure_sample_chunks_json()
    url = f"{BASE_URL}/component-eval/chunk-quality-file"
    payload: Dict[str, Any] = {
        "input_path": input_path,
        # 开关可选，不传则走配置
        "enable_semantic_similarity": True,
        "enable_boundary_clarity": True,
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


def test_chunk_stickiness_direct() -> None:
    """测试 /component-eval/chunk-stickiness —— 直接传 chunks。"""
    url = f"{BASE_URL}/component-eval/chunk-stickiness"
    payload: Dict[str, Any] = {
        "chunks": [
            "Chunk 1: The transformer architecture was introduced in the paper 'Attention is All You Need'.",
            "Chunk 2: BERT is based on the transformer encoder and trained with masked language modeling.",
            "Chunk 3: GPT uses the transformer decoder for autoregressive language generation.",
        ],
        # 阈值 / 距离惩罚可选，不传则走 STICKINESS_THRESHOLD / STICKINESS_DELTA
        "threshold": 0.5,
        "delta": 0.1,
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


def test_chunk_stickiness_file() -> None:
    """测试 /component-eval/chunk-stickiness-file —— 从分块文件读取。"""
    input_path = ensure_sample_chunks_json()
    url = f"{BASE_URL}/component-eval/chunk-stickiness-file"
    payload: Dict[str, Any] = {
        "input_path": input_path,
        # 阈值 / 距离惩罚可选，不传则走 STICKINESS_THRESHOLD / STICKINESS_DELTA
        "threshold": 0.8,
        "delta": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


if __name__ == "__main__":
    print(f"BASE_URL = {BASE_URL}")
    print(f"DEFAULT_CHUNK_JSON_PATH = {DEFAULT_CHUNK_JSON_PATH}")
    print()
    print("注意：请先在 .env 中配置 COMPONENT_PPL_MODEL_PATH / COMPONENT_SIM_MODEL_PATH / STICKINESS_MODEL_PATH")
    print()

    print("=" * 80)
    print("测试 1: /component-eval/chunk-quality (直接传 chunks)")
    print("=" * 80)
    # test_chunk_quality_direct()

    print("=" * 80)
    print("测试 2: /component-eval/chunk-quality-file (从文件读取)")
    print("=" * 80)
    # test_chunk_quality_file()

    print("=" * 80)
    print("测试 3: /component-eval/chunk-stickiness (直接传 chunks)")
    print("=" * 80)
    # 黏连度评估较重，可按需注释
    # test_chunk_stickiness_direct()

    print("=" * 80)
    print("测试 4: /component-eval/chunk-stickiness-file (从文件读取)")
    print("=" * 80)
    test_chunk_stickiness_file()

    print("\n测试脚本结束。")
